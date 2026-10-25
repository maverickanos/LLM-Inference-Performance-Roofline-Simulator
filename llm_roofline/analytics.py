"""Analytical performance model: prefill, decode, KV capacity, TP scaling.

The model is a rigorous napkin, not a simulator:

  prefill  : compute-bound.  time = FLOPs / (peak * MFU)
  decode   : time/step = max(compute, HBM traffic) + all-reduce cost
  capacity : KV tokens = (HBM - weights - reserve) / kv_bytes_per_token
  TP       : weights & KV shard across ranks; 2 ring all-reduces per layer

Everything downstream (TTFT, TPOT, tokens/s, energy/token, $/Mtok, the
roofline plots, the critical batch size) is algebra over those four lines.
Assumed efficiencies (MFU, HBM efficiency) are INPUTS, honestly labeled — this
tool predicts the shape of the performance surface and its hard bounds; it
does not predict kernel efficiency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .gpus import GPUSpec
from .models import ModelSpec

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp8": 1}


@dataclass(frozen=True)
class Assumptions:
    weight_dtype: str = "bf16"
    kv_dtype: str = "bf16"
    mfu_prefill: float = 0.5    # fraction of dense peak achieved on prefill GEMMs
    mfu_decode: float = 0.5     # fraction of dense peak achieved on decode GEMMs
    hbm_efficiency: float = 0.9  # achievable fraction of peak HBM bandwidth
    allreduce_latency_s: float = 5e-6  # per-op launch/sync latency (NCCL-ish)
    kv_reserve_frac: float = 0.10      # HBM reserved for activations/fragmentation
    act_bytes: int = 2                 # activation dtype for TP all-reduces

    @property
    def w_bytes(self) -> int:
        return DTYPE_BYTES[self.weight_dtype]

    @property
    def kv_bytes(self) -> int:
        return DTYPE_BYTES[self.kv_dtype]


# --------------------------------------------------------------------- FLOPs

def decode_flops_per_token(m: ModelSpec, context: int) -> float:
    """FLOPs to generate one token at a given KV length.

    2 * (active matmul params) for the GEMMs, plus attention scores:
    per head, QK^T is 2*qk_dim*context and A·V is 2*v_dim*context.
    """
    linear = 2.0 * m.matmul_params_decode()
    attn = 2.0 * m.layers * m.n_heads * context * (m.qk_dim + m.v_dim)
    return linear + attn


def prefill_flops(m: ModelSpec, prompt_len: int) -> float:
    """FLOPs to prefill a prompt of L tokens.

    Linear term: 2 * P * L (logits computed for the last position only, so the
    LM head is charged once). Quadratic term: causal attention over L*L/2
    positions -> layers * heads * L^2 * (qk_dim + v_dim). The quadratic term is
    negligible at 2k context and dominant past ~100k — that is why it is here.
    """
    linear = 2.0 * m.matmul_params_prefill() * prompt_len
    lm_head = 2.0 * m.vocab * m.hidden
    attn = float(m.layers) * m.n_heads * prompt_len * prompt_len * (m.qk_dim + m.v_dim)
    return linear + lm_head + attn


# ------------------------------------------------------------------ TP comms

def allreduce_time(msg_bytes: float, tp: int, gpu: GPUSpec, a: Assumptions) -> float:
    """Idealized ring all-reduce of msg_bytes across tp ranks.

    Each rank moves 2*(tp-1)/tp * msg_bytes; links are full-duplex so the
    per-direction rate is interconnect_gbps / 2. A fixed per-op latency models
    launch/sync overhead — it dominates for the tiny messages of batch-1 decode.
    """
    if tp <= 1:
        return 0.0
    vol = 2.0 * (tp - 1) / tp * msg_bytes
    link = gpu.interconnect_gbps / 2.0 * 1e9
    return vol / link + a.allreduce_latency_s


def tp_comm_time(m: ModelSpec, gpu: GPUSpec, tp: int, tokens: int, a: Assumptions) -> float:
    """Per-forward-pass TP collective cost: 2 all-reduces per layer (after
    attention out-proj and after MLP down-proj), each of tokens*hidden acts."""
    if tp <= 1:
        return 0.0
    msg = tokens * m.hidden * a.act_bytes
    return 2 * m.layers * allreduce_time(msg, tp, gpu, a)


# ------------------------------------------------------------------- prefill

@dataclass(frozen=True)
class PrefillResult:
    prompt_len: int
    flops: float
    compute_s: float
    comm_s: float
    ttft_s: float
    intensity: float       # FLOPs per weight-byte read (arithmetic intensity)
    tok_per_s: float


def prefill(m: ModelSpec, gpu: GPUSpec, prompt_len: int, tp: int = 1,
            a: Assumptions = Assumptions()) -> PrefillResult:
    f = prefill_flops(m, prompt_len)
    peak = gpu.peak_flops(a.weight_dtype) * tp * a.mfu_prefill
    compute = f / peak
    comm = tp_comm_time(m, gpu, tp, prompt_len, a)
    bytes_moved = m.param_count() * a.w_bytes + prompt_len * m.kv_bytes_per_token(a.kv_bytes)
    return PrefillResult(
        prompt_len=prompt_len, flops=f, compute_s=compute, comm_s=comm,
        ttft_s=compute + comm, intensity=f / bytes_moved,
        tok_per_s=prompt_len / (compute + comm),
    )


# -------------------------------------------------------------------- decode

@dataclass(frozen=True)
class DecodeResult:
    batch: int
    context: int
    tp: int
    compute_s: float
    memory_s: float
    comm_s: float
    tpot_s: float          # time per output token (per step)
    floor_s: float         # inviolable weight-read floor at raw HBM bandwidth
    bound: str             # "memory" | "compute"
    tok_per_s: float       # whole-replica generation throughput (batch/tpot)
    tok_per_s_per_gpu: float
    intensity: float       # FLOPs / HBM byte for the step
    energy_per_tok_j: float

    def cost_per_mtok(self, dollars_per_gpu_hr: float) -> float:
        gpu_seconds_per_tok = self.tp * self.tpot_s / self.batch
        return dollars_per_gpu_hr / 3600.0 * gpu_seconds_per_tok * 1e6


def decode(m: ModelSpec, gpu: GPUSpec, batch: int, context: int, tp: int = 1,
           a: Assumptions = Assumptions()) -> DecodeResult:
    # compute: every token multiplies the active params
    flops = batch * decode_flops_per_token(m, context)
    compute = flops / (gpu.peak_flops(a.weight_dtype) * tp * a.mfu_decode)

    # memory: weights read once per step (MoE: expert coverage grows with
    # batch), KV read per sequence
    weight_bytes = m.weight_read_params(batch) * a.w_bytes
    kv_read = batch * context * m.kv_bytes_per_token(a.kv_bytes)
    hbm = gpu.hbm_bw * tp
    memory = (weight_bytes + kv_read) / (hbm * a.hbm_efficiency)

    comm = tp_comm_time(m, gpu, tp, batch, a)
    tpot = max(compute, memory) + comm

    floor = (m.matmul_params_decode() * a.w_bytes) / hbm  # raw-BW weight floor
    tok_s = batch / tpot
    return DecodeResult(
        batch=batch, context=context, tp=tp,
        compute_s=compute, memory_s=memory, comm_s=comm, tpot_s=tpot,
        floor_s=floor, bound="compute" if compute >= memory else "memory",
        tok_per_s=tok_s, tok_per_s_per_gpu=tok_s / tp,
        intensity=flops / (weight_bytes + kv_read),
        energy_per_tok_j=gpu.tdp_w * tp * tpot / batch,
    )


def critical_batch(m: ModelSpec, gpu: GPUSpec, context: int,
                   a: Assumptions = Assumptions()) -> float:
    """Closed-form batch size where decode flips memory-bound -> compute-bound.

    Solve B*f/(peak*mfu) = (W + B*kv*ctx)/(BW*eff) for B (TP cancels; dense
    weight traffic assumed, so exact for dense models):

        B* = (W / (BW*eff)) / ( f/(peak*mfu) - kv*ctx/(BW*eff) )

    Returns inf when the KV-read slope exceeds the compute slope — then extra
    batch adds memory traffic as fast as FLOPs and decode never goes
    compute-bound at this context length.

    MoE models get a numerical solve instead: weight traffic itself grows with
    batch (expert coverage) until all experts are resident, so the closed form
    with active-only weights would understate the crossover.
    """
    f = decode_flops_per_token(m, context)
    kv = m.kv_bytes_per_token(a.kv_bytes) * context
    bw = gpu.hbm_bw * a.hbm_efficiency
    compute_slope = f / (gpu.peak_flops(a.weight_dtype) * a.mfu_decode)
    if compute_slope <= kv / bw:
        return math.inf
    if not m.is_moe:
        W = m.matmul_params_decode() * a.w_bytes
        return (W / bw) / (compute_slope - kv / bw)
    # MoE: bisect compute(B) - memory(B) with batch-dependent weight reads
    def gap(b: float) -> float:
        mem = (m.weight_read_params(int(math.ceil(b))) * a.w_bytes + b * kv) / bw
        return b * compute_slope - mem
    lo, hi = 1.0, 1.0
    while gap(hi) < 0:
        hi *= 2
        if hi > 1e9:
            return math.inf
    for _ in range(80):
        mid = (lo + hi) / 2
        if gap(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ----------------------------------------------------------------- KV budget

@dataclass(frozen=True)
class KVBudget:
    weight_bytes: float
    usable_kv_bytes: float
    kv_bytes_per_token: int
    max_tokens: int
    max_seqs_at_context: int
    context: int


def kv_budget(m: ModelSpec, gpu: GPUSpec, context: int, tp: int = 1,
              a: Assumptions = Assumptions()) -> KVBudget:
    """How many tokens of KV fit after weights + reserve, across the TP group.
    Weights are TOTAL stored params (all MoE experts live in HBM)."""
    weights = m.param_count() * a.w_bytes
    usable = gpu.hbm_bytes * tp * (1.0 - a.kv_reserve_frac) - weights
    per_tok = m.kv_bytes_per_token(a.kv_bytes)
    max_tokens = max(0, int(usable / per_tok))
    return KVBudget(
        weight_bytes=weights, usable_kv_bytes=max(0.0, usable),
        kv_bytes_per_token=per_tok, max_tokens=max_tokens,
        max_seqs_at_context=max_tokens // max(1, context), context=context,
    )


# ------------------------------------------------------------- TP scaling

def tp_scaling(m: ModelSpec, gpu: GPUSpec, batch: int, context: int,
               tps=(1, 2, 4, 8), a: Assumptions = Assumptions()):
    """Decode TPOT and scaling efficiency across TP degrees at fixed batch.

    Efficiency := (TPOT(1) / TPOT(tp)) / tp — 1.0 means the extra GPUs cut
    per-token latency perfectly linearly (ideal weight/KV sharding, free comm).
    """
    base = decode(m, gpu, batch, context, tp=1, a=a)
    out = []
    for tp in tps:
        d = decode(m, gpu, batch, context, tp=tp, a=a)
        eff = (base.tpot_s / d.tpot_s) / tp
        out.append((tp, d, eff))
    return out


# ----------------------------------------------- end-to-end / benchmark shape

@dataclass(frozen=True)
class ThroughputPoint:
    batch: int
    isl: int
    osl: int
    round_s: float
    prefill_s: float
    decode_s: float
    output_tok_per_s: float
    output_tok_per_s_per_gpu: float


def benchmark_throughput(m: ModelSpec, gpu: GPUSpec, tp: int, isl: int, osl: int,
                         a: Assumptions = Assumptions(),
                         max_batch: int = 2048,
                         batch: int | None = None) -> ThroughputPoint:
    """Steady-state max-throughput estimate for an ISL/OSL benchmark.

    Continuous batching is approximated as conservation of work: a round of B
    requests costs (batched prefill of B*ISL tokens) + (OSL decode steps at
    batch B, average context ISL + OSL/2). B defaults to the KV-capacity limit
    at full context, capped at max_batch. This deliberately ignores scheduler
    gaps, chunked-prefill interference, and latency SLOs — it is the airspeed
    the airframe allows, not the airline schedule.
    """
    if batch is None:
        cap = kv_budget(m, gpu, isl + osl, tp, a).max_seqs_at_context
        batch = max(1, min(max_batch, cap))
    # batched prefill: linear FLOPs scale with B*ISL; attention term per request
    f = (2.0 * m.matmul_params_prefill() * isl + 2.0 * m.vocab * m.hidden
         + float(m.layers) * m.n_heads * isl * isl * (m.qk_dim + m.v_dim)) * batch
    peak = gpu.peak_flops(a.weight_dtype) * tp * a.mfu_prefill
    pre = f / peak + tp_comm_time(m, gpu, tp, batch * isl, a)
    avg_ctx = isl + osl // 2
    dec = osl * decode(m, gpu, batch, avg_ctx, tp, a).tpot_s
    total = pre + dec
    out_tok_s = batch * osl / total
    return ThroughputPoint(batch=batch, isl=isl, osl=osl, round_s=total,
                           prefill_s=pre, decode_s=dec,
                           output_tok_per_s=out_tok_s,
                           output_tok_per_s_per_gpu=out_tok_s / tp)


def with_dtype(a: Assumptions, weight_dtype: str, kv_dtype: str | None = None) -> Assumptions:
    return replace(a, weight_dtype=weight_dtype, kv_dtype=kv_dtype or weight_dtype)
