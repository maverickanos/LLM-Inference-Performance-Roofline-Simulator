# Validation

A performance model that never meets a measurement is a vibe. This file is the
contract: what the model gets exactly right, what it gets approximately right,
and the bound it can never violate. All URLs were retrieved and checked on
**2026-08-15**. Reproduce every number below with:

```
./run.sh test                     # tiers 1 and 3 (automated)
python -m llm_roofline validate   # tier 2 table
```

## Tier 1 — Exact: derived parameter counts vs published totals

Parameter counts are **derived from architecture algebra** (embeddings, GQA/MLA
attention, MLP or MoE experts, norms), never entered by hand. If the algebra
can't reproduce the published count, nothing downstream deserves trust.
Automated in `tests/test_params.py` (tolerance 2%; Llama-3 held to 0.5%).

| Model | Published | Derived | Error |
|---|---|---|---|
| Llama-3-8B | 8.030 B | 8.030 B | +0.00% |
| Llama-3-70B | 70.554 B | 70.554 B | −0.00% |
| Llama-2-70B | 68.977 B | 68.977 B | −0.00% |
| Mistral-7B | 7.242 B | 7.242 B | −0.00% |
| Qwen2.5-32B | 32.5 B | 32.764 B | +0.81% |
| Mixtral-8x7B (total / active) | 46.703 / 12.9 B | 46.703 / 12.880 B | −0.00% / −0.16% |
| DeepSeek-V3 (total / active) | 671 / 37 B | 671.026 / 37.552 B | +0.00% / +1.49% |

DeepSeek-V3 is the demanding case: reproducing 671 B requires getting MLA's
low-rank Q/KV projections, 256 routed + 1 shared experts across 58 MoE layers,
3 leading dense layers, and untied embeddings all simultaneously right
(architecture per the [technical report, arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
and the [HF config](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json);
the 671 B figure excludes the 14 B MTP module, per the repo's README_WEIGHTS).

## Tier 2 — Anchored: prediction vs published measured throughput

Assumptions for every row: FP8 weights + FP8 KV, MFU 0.5 (prefill and decode),
90% achievable HBM bandwidth, 10% HBM reserve, idealized ring all-reduce with
5 µs/op latency. Batch size is set by the model's own KV-capacity limit
(capped at 2048). Predicted and measured are **output tokens/s for the whole
GPU group**.

| Benchmark (measured) | Measured tok/s | Predicted | Error | Source |
|---|---|---|---|---|
| Llama-3.3-70B FP8, 2×H100 SXM, ISL/OSL 128/128, TensorRT-LLM | 6,092 | 6,584 | **+8%** | [TRT-LLM perf overview](https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html) |
| Llama-3.3-70B FP8, 2×H200 SXM, ISL/OSL 2048/2048, TensorRT-LLM | 3,776 | 5,786 | **+53%** | [TRT-LLM perf overview](https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html) |
| Llama-3.1-8B FP8, 1×H100 SXM, ISL/OSL 128/128, TensorRT-LLM | 26,401 | 33,990 | **+29%** | [TRT-LLM perf overview](https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html) |
| Llama-3.1-8B FP8, 1×H100 SXM, ISL/OSL 2048/128, TensorRT-LLM | 3,276 | 3,498 | **+7%** | [TRT-LLM perf overview](https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html) |
| Llama-2-70B FP8, MLPerf Inference v4.1 offline, 8×H200 (34,864 tok/s → 4,358/GPU) | 4,358 | 3,443 | **−21%** | [NVIDIA MLPerf v4.1 blog](https://developer.nvidia.com/blog/nvidia-blackwell-platform-sets-new-llm-inference-records-in-mlperf-inference-v4-1) |

Notes, in the interest of honesty:

- The TensorRT-LLM perf-overview page is a living document; the numbers above
  are what it showed on 2026-08-15. Llama-3.3-70B and Llama-3.1-70B share the
  Llama-3-70B trunk modeled here (they differ in RoPE/context config only).
- **Why the model over-predicts (+7% to +53%):** it assumes work conservation —
  no scheduler gaps, no chunked-prefill interference, no attention-kernel
  inefficiency at long context, perfectly overlapped continuous batching, and a
  flat MFU of 0.5 regardless of GEMM shape. The +53% row is exactly where those
  assumptions bite hardest: 2048/2048 means long-context attention (worse
  kernel efficiency than the dense-GEMM MFU assumed) and heavy KV pressure at
  the model's own optimistic max batch. Real servers also cannot always run at
  the KV-capacity batch limit.
- **Why the MLPerf row under-predicts (−21%):** the cited page does not publish
  the dataset's mean sequence lengths, so ISL/OSL are approximated as
  1024/1024; MLPerf offline submissions are also heavily tuned (custom
  scheduling, FP8 attention) and exceed the flat 50%-MFU assumption. Treat this
  row as the loosest anchor.
- The napkin's honest claim is therefore: **right shape, right bounds, tens of
  percent absolute error.** For a few-line analytical model with two assumed
  efficiency constants, landing within 8–53% of tuned production stacks across
  a 8B–70B, H100/H200, short/long-context grid is the point — and the failure
  modes are exactly where the model says its assumptions are weakest.

## Tier 3 — Sanity: the inviolable decode floor

Memory-bound decode can never beat

```
TPOT >= active_weight_bytes / (TP × raw_HBM_bandwidth)
```

— reading the weights once per token at full theoretical bandwidth with zero
KV, zero compute, and zero communication. `tests/test_analytics.py::test_decode_floor_inviolable`
sweeps every model × every GPU × {BF16, FP8} × TP {1,4} × batch {1,16,256} —
including a deliberately absurd configuration with MFU = 1.0 and HBM efficiency
= 1.0 — and asserts no predicted TPOT ever crosses the floor. If a future
change makes the model claim 70 GB of weights can be read in less than
70 GB / bandwidth, the suite fails.

Also automated: closed-form critical batch matches the numerical
compute/memory crossover; TP=1 exactly equals the no-communication baseline;
MoE expert-coverage weight traffic saturates at all experts resident; MLA KV
per token (70,272 B for 671B-param DeepSeek-V3) is ~4.7× smaller than GQA KV
of a 70B model (327,680 B) — each against hand-computed values.
