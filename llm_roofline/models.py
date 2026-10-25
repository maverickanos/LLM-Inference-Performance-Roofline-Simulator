"""Model architecture database.

Every architecture is entered from its public Hugging Face config.json (URL in
each entry, verified 2026-08-15). Parameter counts are NOT entered by hand —
they are DERIVED from the architecture (embeddings, GQA/MLA attention, MLP or
MoE experts, norms) and tested against the published totals in
tests/test_params.py, which is the tier-1 validation of this whole tool: if
the napkin can't reproduce the parameter count, nothing downstream deserves
trust.

MLA (DeepSeek-V3) KV cache: with Multi-head Latent Attention only the
compressed KV latent c_KV (kv_lora_rank = 512) plus the decoupled RoPE key
(qk_rope_head_dim = 64) is cached per token per layer — 576 elements/layer —
instead of 2 * n_kv_heads * head_dim. See DeepSeek-V2 paper §2.1
(arXiv:2405.04434) and DeepSeek-V3 technical report (arXiv:2412.19437).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    hidden: int
    layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate: int          # dense-MLP intermediate size
    vocab: int
    tie_embeddings: bool = False
    attn_bias: bool = False    # QKV projection bias (Qwen2 family)
    # --- MoE (0 experts => dense model) ---
    n_routed_experts: int = 0
    n_shared_experts: int = 0
    top_k: int = 0
    moe_intermediate: int = 0
    n_dense_layers: int | None = None  # leading layers with a dense MLP (DeepSeek-V3: 3)
    # --- MLA (DeepSeek) ---
    mla: bool = False
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    # --- published ground truth (for validation only, never used in math) ---
    published_params: float = 0.0
    published_active_params: float = 0.0
    source: str = ""

    # ------------------------------------------------------------------ MoE
    @property
    def is_moe(self) -> bool:
        return self.n_routed_experts > 0

    @property
    def _dense_layers(self) -> int:
        if not self.is_moe:
            return self.layers
        return self.n_dense_layers or 0

    @property
    def _moe_layers(self) -> int:
        return self.layers - self._dense_layers

    # ------------------------------------------------------- per-layer parts
    def attn_params_per_layer(self) -> int:
        if self.mla:
            qk_head = self.qk_nope_head_dim + self.qk_rope_head_dim
            q = self.hidden * self.q_lora_rank + self.q_lora_rank * self.n_heads * qk_head
            kv_a = self.hidden * (self.kv_lora_rank + self.qk_rope_head_dim)
            kv_b = self.kv_lora_rank * self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)
            o = self.n_heads * self.v_head_dim * self.hidden
            norms = self.q_lora_rank + self.kv_lora_rank
            return q + kv_a + kv_b + o + norms
        q = self.hidden * self.n_heads * self.head_dim
        kv = 2 * self.hidden * self.n_kv_heads * self.head_dim
        o = self.n_heads * self.head_dim * self.hidden
        bias = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim if self.attn_bias else 0
        return q + kv + o + bias

    def _expert_params(self) -> int:
        return 3 * self.hidden * self.moe_intermediate  # gate, up, down

    def dense_mlp_params(self) -> int:
        return 3 * self.hidden * self.intermediate

    def moe_mlp_params_per_layer(self, active_only: bool) -> int:
        router = self.hidden * self.n_routed_experts + self.n_routed_experts
        n = (self.top_k if active_only else self.n_routed_experts) + self.n_shared_experts
        return router + n * self._expert_params()

    # ------------------------------------------------------------- totals
    def embed_params(self) -> int:
        return self.vocab * self.hidden

    def lm_head_params(self) -> int:
        return 0 if self.tie_embeddings else self.vocab * self.hidden

    def _count(self, active_only: bool) -> int:
        total = self.embed_params() + self.lm_head_params()
        total += self.layers * self.attn_params_per_layer()
        total += self._dense_layers * self.dense_mlp_params()
        total += self._moe_layers * self.moe_mlp_params_per_layer(active_only)
        total += self.layers * 2 * self.hidden + self.hidden  # RMSNorms
        return total

    def param_count(self) -> int:
        """Total stored parameters (all experts)."""
        return self._count(active_only=False)

    def active_param_count(self) -> int:
        """Parameters activated per token (top-k + shared experts only)."""
        return self._count(active_only=True)

    # ----------------------------------------------------- inference algebra
    def matmul_params_decode(self) -> int:
        """Params multiplied per decode token: active params minus the input
        embedding lookup (a table read, not a GEMM). The LM head IS a GEMM and
        runs every decode step, so it stays (even when tied)."""
        n = self.active_param_count() - self.embed_params()
        if self.tie_embeddings:
            n += self.vocab * self.hidden  # tied head still multiplies
        return n

    def matmul_params_prefill(self) -> int:
        """Per-token GEMM params during prefill: as decode, minus the LM head
        (production servers compute logits for the last position only)."""
        return self.matmul_params_decode() - self.vocab * self.hidden

    def weight_read_params(self, batch: int) -> int:
        """Expected parameters READ from HBM per decode step at a given batch.

        Dense models: independent of batch. MoE: a batch of B tokens touches
        up to min(n_routed, B * top_k) routed experts per layer, so the
        per-step weight traffic grows with batch until every expert is resident
        in the working set — this is why MoE batching erodes the usual
        'weights amortize over batch' win. Upper-bound (no routing collisions).
        """
        n = self.matmul_params_decode()
        if self.is_moe:
            touched = min(self.n_routed_experts, batch * self.top_k)
            extra = (touched - self.top_k) * self._expert_params()
            n += max(0, extra) * self._moe_layers
        return n

    # --------------------------------------------------------------- KV cache
    def kv_elems_per_token(self) -> int:
        if self.mla:
            # cache c_KV latent + decoupled rope key only (arXiv:2405.04434 §2.1)
            return self.layers * (self.kv_lora_rank + self.qk_rope_head_dim)
        return self.layers * 2 * self.n_kv_heads * self.head_dim

    def kv_bytes_per_token(self, kv_bytes: int = 2) -> int:
        return self.kv_elems_per_token() * kv_bytes

    # attention-score dims used for FLOP accounting
    @property
    def qk_dim(self) -> int:
        return (self.qk_nope_head_dim + self.qk_rope_head_dim) if self.mla else self.head_dim

    @property
    def v_dim(self) -> int:
        return self.v_head_dim if self.mla else self.head_dim


_MODELS = [
    ModelSpec(
        key="llama-3-8b", name="Llama-3-8B",
        hidden=4096, layers=32, n_heads=32, n_kv_heads=8, head_dim=128,
        intermediate=14336, vocab=128256,
        published_params=8.030e9, published_active_params=8.030e9,
        source="https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json",
    ),
    ModelSpec(
        key="llama-3-70b", name="Llama-3-70B",
        hidden=8192, layers=80, n_heads=64, n_kv_heads=8, head_dim=128,
        intermediate=28672, vocab=128256,
        published_params=70.554e9, published_active_params=70.554e9,
        source="https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json",
    ),
    ModelSpec(
        # Same trunk as Llama-3.1/3.3-70B (only rope/context differ), so it is
        # the architecture used for the Llama-2-70B MLPerf anchor too.
        key="llama-2-70b", name="Llama-2-70B",
        hidden=8192, layers=80, n_heads=64, n_kv_heads=8, head_dim=128,
        intermediate=28672, vocab=32000,
        published_params=68.977e9, published_active_params=68.977e9,
        source="https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json",
    ),
    ModelSpec(
        key="mistral-7b", name="Mistral-7B-v0.1",
        hidden=4096, layers=32, n_heads=32, n_kv_heads=8, head_dim=128,
        intermediate=14336, vocab=32000,
        published_params=7.242e9, published_active_params=7.242e9,
        source="https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json",
    ),
    ModelSpec(
        key="qwen2.5-32b", name="Qwen2.5-32B",
        hidden=5120, layers=64, n_heads=40, n_kv_heads=8, head_dim=128,
        intermediate=27648, vocab=152064, attn_bias=True,
        published_params=32.5e9, published_active_params=32.5e9,
        source="https://huggingface.co/Qwen/Qwen2.5-32B/blob/main/config.json",
    ),
    ModelSpec(
        key="mixtral-8x7b", name="Mixtral-8x7B",
        hidden=4096, layers=32, n_heads=32, n_kv_heads=8, head_dim=128,
        intermediate=14336, vocab=32000,
        n_routed_experts=8, n_shared_experts=0, top_k=2, moe_intermediate=14336,
        n_dense_layers=0,
        published_params=46.703e9, published_active_params=12.9e9,
        source="https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json",
    ),
    ModelSpec(
        key="deepseek-v3", name="DeepSeek-V3",
        hidden=7168, layers=61, n_heads=128, n_kv_heads=128, head_dim=128,
        intermediate=18432, vocab=129280,
        n_routed_experts=256, n_shared_experts=1, top_k=8, moe_intermediate=2048,
        n_dense_layers=3,
        mla=True, q_lora_rank=1536, kv_lora_rank=512,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        # 671B total / 37B active per the technical report (arXiv:2412.19437);
        # excludes the 14B MTP module (see README_WEIGHTS.md in the repo).
        published_params=671e9, published_active_params=37e9,
        source="https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json",
    ),
]

MODELS: dict[str, ModelSpec] = {m.key: m for m in _MODELS}
_ALIASES = {
    "llama3-8b": "llama-3-8b", "llama3-70b": "llama-3-70b",
    "llama2-70b": "llama-2-70b", "mixtral": "mixtral-8x7b",
    "qwen-2.5-32b": "qwen2.5-32b", "dsv3": "deepseek-v3",
}


def get_model(key: str) -> ModelSpec:
    k = key.lower()
    k = _ALIASES.get(k, k)
    if k not in MODELS:
        raise KeyError(f"unknown model {key!r}; available: {', '.join(MODELS)}")
    return MODELS[k]
