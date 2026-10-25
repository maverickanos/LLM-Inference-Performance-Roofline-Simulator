"""KV-cache formulas: GQA vs hand calculation, MLA compression."""

from llm_roofline.models import get_model


def test_gqa_kv_matches_hand_calc():
    # Llama-3-70B: 2 (K and V) * 80 layers * 8 kv-heads * 128 dim * 2 bytes
    m = get_model("llama-3-70b")
    assert m.kv_bytes_per_token(2) == 2 * 80 * 8 * 128 * 2 == 327_680
    # Mistral-7B: 2 * 32 * 8 * 128 * 2
    assert get_model("mistral-7b").kv_bytes_per_token(2) == 131_072


def test_fp8_kv_halves():
    m = get_model("llama-3-70b")
    assert m.kv_bytes_per_token(1) * 2 == m.kv_bytes_per_token(2)


def test_mla_kv_hand_calc():
    # DeepSeek-V3 caches c_KV (512) + rope key (64) per layer:
    # 61 * (512 + 64) * 2 bytes = 70,272 bytes/token
    ds = get_model("deepseek-v3")
    assert ds.kv_bytes_per_token(2) == 61 * (512 + 64) * 2 == 70_272


def test_mla_kv_much_smaller_than_gqa():
    """The headline MLA claim: a 671B model caches far less KV per token than
    a 70B GQA model — and enormously less than naive MHA would."""
    ds = get_model("deepseek-v3")
    llama = get_model("llama-3-70b")
    assert ds.param_count() > 9 * llama.param_count()
    assert ds.kv_bytes_per_token(2) < 0.25 * llama.kv_bytes_per_token(2)
    # vs hypothetical MHA at DeepSeek's own dims: 2*61*128*128*2 bytes/token
    naive_mha = 2 * ds.layers * ds.n_heads * ds.head_dim * 2
    assert ds.kv_bytes_per_token(2) < naive_mha / 50
