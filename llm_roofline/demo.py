"""Guided demo: the napkin, worked end to end."""

from __future__ import annotations

from .analytics import Assumptions, critical_batch, decode, kv_budget, prefill
from .gpus import get_gpu
from .models import get_model
from .report import run_validation, table


def _hdr(s: str):
    print()
    print(s)
    print("=" * len(s))


def run_demo() -> None:
    a = Assumptions()
    m, g = get_model("llama-3-70b"), get_gpu("h100")
    tp = 4

    _hdr("1. The napkin: how fast CAN Llama-3-70B decode on 4x H100 (BF16)?")
    W = m.matmul_params_decode() * a.w_bytes
    print(f"   weights read per token: {W / 1e9:.1f} GB, sharded over {tp} GPUs")
    print(f"   HBM: {g.hbm_gbps / 1000:.2f} TB/s per GPU x {a.hbm_efficiency:.0%} efficiency")
    print(f"   floor = {W / 1e9:.1f} GB / (4 x 3.35 TB/s) = "
          f"{W / (tp * g.hbm_bw) * 1e3:.2f} ms/token (raw), "
          f"{W / (tp * g.hbm_bw * a.hbm_efficiency) * 1e3:.2f} ms at 90% eff")
    d = decode(m, g, batch=1, context=8192, tp=tp, a=a)
    print(f"   + KV read @ ctx 8192 + 2 all-reduces/layer over NVLink:")
    print(f"   model TPOT = {d.tpot_s * 1e3:.2f} ms  ->  {1 / d.tpot_s:,.0f} tok/s "
          f"single-stream  (bound: {d.bound})")
    p = prefill(m, g, 8192, tp, a)
    print(f"   TTFT for an 8k prompt: {p.ttft_s * 1e3:.0f} ms "
          f"({p.flops / 1e12:.0f} TFLOP at MFU {a.mfu_prefill:.0%})")

    _hdr("2. Batching buys throughput until the roofline knee")
    rows = []
    for b in (1, 8, 32, 128, 256):
        r = decode(m, g, b, 8192, tp, a)
        rows.append([b, r.bound, f"{r.tpot_s * 1e3:.2f}",
                     f"{r.tok_per_s_per_gpu:,.0f}", f"{r.energy_per_tok_j:.2f}"])
    print(table(rows, ["batch", "bound", "TPOT ms", "tok/s/GPU", "J/tok"]))
    bc = critical_batch(m, g, 8192, a)
    print(f"   closed-form crossover batch at ctx 8192: "
          f"{'inf — KV reads keep decode memory-bound forever' if bc == float('inf') else f'~{bc:.0f}'}")

    _hdr("3. KV cache is the real capacity currency")
    fp8 = Assumptions(weight_dtype="fp8", kv_dtype="fp8")
    rows = []
    for key, gpu_key, tp_, aa in [("llama-3-70b", "h100", 4, a),
                                  ("llama-3-70b", "h200", 4, a),
                                  ("deepseek-v3", "h200", 8, fp8)]:
        mm, gg = get_model(key), get_gpu(gpu_key)
        bud = kv_budget(mm, gg, 8192, tp_, aa)
        rows.append([f"{mm.name} @ {tp_}x {gg.name} ({aa.weight_dtype})",
                     f"{mm.kv_bytes_per_token(aa.kv_bytes) / 1024:.1f}",
                     f"{bud.max_tokens / 1e6:.2f}M", bud.max_seqs_at_context])
    print(table(rows, ["config", "KV KiB/tok", "KV tokens", "seqs @ 8k"]))
    print("   (DeepSeek-V3 shown FP8 — it is FP8-native and does not fit 8x141 GB in BF16.)")
    print("   DeepSeek-V3 is 9.5x Llama-70B's size but caches 4.7x FEWER KV bytes")
    print("   per token — that is MLA's compressed latent (512+64 dims/layer) at work.")

    _hdr("4. FP8 halves both rooflines' pain")
    for dt in ("bf16", "fp8"):
        aa = Assumptions(weight_dtype=dt, kv_dtype=dt)
        r = decode(m, g, 1, 8192, tp, aa)
        print(f"   {dt:>4s}: TPOT {r.tpot_s * 1e3:5.2f} ms  "
              f"({1 / r.tpot_s:,.0f} tok/s single-stream)")

    _hdr("5. Honesty check: prediction vs published measurements")
    print(run_validation())
    print()
    print("Errors of tens of percent are expected and documented — the napkin has")
    print("no scheduler, no kernel overheads, and an assumed MFU. See VALIDATION.md.")
