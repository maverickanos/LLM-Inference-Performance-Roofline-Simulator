"""Tier-1 validation: derived parameter counts must match published totals.

If the architecture algebra can't reproduce the published parameter count,
no downstream latency number deserves trust.
"""

import pytest

from llm_roofline.models import MODELS, get_model

TOL = 0.02  # 2%


@pytest.mark.parametrize("key", list(MODELS))
def test_total_params_match_published(key):
    m = MODELS[key]
    derived = m.param_count()
    err = abs(derived - m.published_params) / m.published_params
    assert err < TOL, (
        f"{m.name}: derived {derived / 1e9:.3f}B vs published "
        f"{m.published_params / 1e9:.3f}B ({err:.2%})")


@pytest.mark.parametrize("key", list(MODELS))
def test_active_params_match_published(key):
    m = MODELS[key]
    derived = m.active_param_count()
    err = abs(derived - m.published_active_params) / m.published_active_params
    assert err < TOL, (
        f"{m.name}: derived active {derived / 1e9:.3f}B vs published "
        f"{m.published_active_params / 1e9:.3f}B ({err:.2%})")


def test_llama3_exact_to_half_percent():
    # the dense Llama-3 counts are known exactly; hold them tighter
    for key, pub in [("llama-3-8b", 8_030_261_248), ("llama-3-70b", 70_553_706_496)]:
        m = get_model(key)
        assert abs(m.param_count() - pub) / pub < 0.005


def test_moe_active_accounting():
    mix = get_model("mixtral-8x7b")
    # active = total - (8-2) inactive experts per layer
    inactive = 6 * 3 * mix.hidden * mix.moe_intermediate * mix.layers
    assert mix.param_count() - mix.active_param_count() == inactive
    # DeepSeek-V3: 256 routed - 8 active per MoE layer inactive
    ds = get_model("deepseek-v3")
    inactive = (256 - 8) * 3 * ds.hidden * ds.moe_intermediate * (ds.layers - 3)
    assert ds.param_count() - ds.active_param_count() == inactive
    assert ds.active_param_count() < 0.06 * ds.param_count()


def test_moe_weight_read_grows_with_batch_then_saturates():
    mix = get_model("mixtral-8x7b")
    r1 = mix.weight_read_params(1)   # 2 experts/layer touched
    r2 = mix.weight_read_params(2)   # 4 experts/layer touched
    rbig = mix.weight_read_params(10_000)
    assert r1 == mix.matmul_params_decode()
    assert r1 < r2 < rbig
    # saturates once all 8 experts are touched (batch * top_k >= 8, i.e. B=4)
    assert rbig == mix.weight_read_params(4)
    full = mix.param_count() - mix.embed_params()
    assert abs(rbig - full) / full < 0.001
