"""Analytics invariants: the decode floor, the crossover, TP behavior."""

import math

import pytest

from llm_roofline.analytics import (Assumptions, critical_batch, decode,
                                    kv_budget, prefill)
from llm_roofline.gpus import GPUS, get_gpu
from llm_roofline.models import MODELS, get_model

A = Assumptions()


@pytest.mark.parametrize("mkey", list(MODELS))
@pytest.mark.parametrize("gkey", list(GPUS))
def test_decode_floor_inviolable(mkey, gkey):
    """Sanity tier: TPOT can never beat weight_bytes / raw HBM bandwidth."""
    m, g = MODELS[mkey], GPUS[gkey]
    for a in (A, Assumptions(weight_dtype="fp8", kv_dtype="fp8"),
              Assumptions(mfu_prefill=1.0, mfu_decode=1.0, hbm_efficiency=1.0)):
        for tp in (1, 4):
            floor = m.matmul_params_decode() * a.w_bytes / (g.hbm_bw * tp)
            for batch in (1, 16, 256):
                d = decode(m, g, batch, 4096, tp, a)
                assert d.tpot_s >= floor * (1 - 1e-12), (
                    f"{m.name}/{g.name} tp{tp} b{batch}: predicted "
                    f"{d.tpot_s * 1e3:.3f} ms beats the physical floor "
                    f"{floor * 1e3:.3f} ms")
                assert d.floor_s == pytest.approx(floor)


def test_crossover_matches_numerical_sweep():
    """Closed-form critical batch == where compute and memory times cross."""
    # ctx 512: short enough that the KV-read slope does not swamp the compute
    # slope (at ctx 2048 with BF16 KV, Llama-3-8B decode on H100 is *never*
    # compute-bound — itself a finding the closed form makes obvious).
    m, g = get_model("llama-3-8b"), get_gpu("h100")
    ctx = 512
    bc = critical_batch(m, g, ctx, A)
    assert math.isfinite(bc)
    # at the closed-form batch, compute ~= memory
    d = decode(m, g, int(round(bc)), ctx, 1, A)
    assert d.compute_s == pytest.approx(d.memory_s, rel=0.05)
    # bracketing: memory-bound just below, compute-bound just above
    assert decode(m, g, max(1, int(bc * 0.8)), ctx, 1, A).bound == "memory"
    assert decode(m, g, int(bc * 1.25) + 1, ctx, 1, A).bound == "compute"


def test_moe_crossover_accounts_for_expert_traffic():
    """MoE weight reads grow with batch, so the true crossover is far past the
    naive active-params closed form — and self-consistent at the solution."""
    fp8 = Assumptions(weight_dtype="fp8", kv_dtype="fp8")
    m, g = get_model("deepseek-v3"), get_gpu("h200")
    bc = critical_batch(m, g, 8192, fp8)
    assert math.isfinite(bc)
    d = decode(m, g, int(round(bc)), 8192, 1, fp8)
    assert d.compute_s == pytest.approx(d.memory_s, rel=0.05)
    # naive closed form with active-only weights (what a dense model would use)
    f = 2.0 * m.matmul_params_decode() + 2.0 * m.layers * m.n_heads * 8192 * (m.qk_dim + m.v_dim)
    bw = g.hbm_bw * fp8.hbm_efficiency
    naive = (m.matmul_params_decode() / bw) / (
        f / (g.peak_flops("fp8") * fp8.mfu_decode)
        - m.kv_bytes_per_token(1) * 8192 / bw)
    assert bc > 3 * naive


def test_crossover_infinite_at_huge_context():
    """When per-sequence KV reads outweigh per-token FLOP cost, extra batch
    never makes decode compute-bound."""
    m, g = get_model("llama-3-70b"), get_gpu("a100")
    assert critical_batch(m, g, 1_000_000, A) == math.inf


def test_tp1_equals_no_comm_baseline():
    m, g = get_model("llama-3-70b"), get_gpu("h100")
    d = decode(m, g, 8, 8192, tp=1, a=A)
    assert d.comm_s == 0.0
    assert d.tpot_s == pytest.approx(max(d.compute_s, d.memory_s))
    p = prefill(m, g, 8192, tp=1, a=A)
    assert p.comm_s == 0.0 and p.ttft_s == pytest.approx(p.compute_s)


def test_tp_speedup_sublinear_but_real():
    m, g = get_model("llama-3-70b"), get_gpu("h100")
    t1 = decode(m, g, 1, 8192, tp=1, a=A).tpot_s
    t4 = decode(m, g, 1, 8192, tp=4, a=A).tpot_s
    assert t4 < t1                 # TP helps memory-bound decode
    assert t4 > t1 / 4             # but never ideally (all-reduce cost)


def test_prefill_quadratic_term_matters_at_long_context():
    m, g = get_model("llama-3-8b"), get_gpu("h100")
    short = prefill(m, g, 2048, 1, A)
    lin_frac_short = (2 * m.matmul_params_prefill() * 2048) / short.flops
    long = prefill(m, g, 131_072, 1, A)
    lin_frac_long = (2 * m.matmul_params_prefill() * 131_072) / long.flops
    assert lin_frac_short > 0.95   # attention negligible at 2k
    assert lin_frac_long < 0.75    # attention dominant share by 128k


def test_kv_budget_accounting():
    m, g = get_model("llama-3-70b"), get_gpu("h100")
    b = kv_budget(m, g, context=8192, tp=4, a=A)
    usable = 4 * 80e9 * 0.9 - m.param_count() * 2
    assert b.usable_kv_bytes == pytest.approx(usable)
    assert b.max_tokens == int(usable / m.kv_bytes_per_token(2))
    # 70B BF16 does not fit on one 80GB GPU
    assert kv_budget(m, g, 8192, tp=1, a=A).max_tokens == 0


def test_fp8_speeds_up_decode():
    m, g = get_model("llama-3-70b"), get_gpu("h100")
    bf16 = decode(m, g, 1, 8192, 4, Assumptions())
    fp8 = decode(m, g, 1, 8192, 4, Assumptions(weight_dtype="fp8", kv_dtype="fp8"))
    assert fp8.tpot_s < 0.6 * bf16.tpot_s  # ~halves the memory-bound time
