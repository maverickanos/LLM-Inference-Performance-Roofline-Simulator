"""Terminal report + validation table."""

from __future__ import annotations

from .analytics import (Assumptions, benchmark_throughput, critical_batch,
                        decode, kv_budget, prefill)
from .gpus import get_gpu
from .models import get_model


def _fmt(x: float, unit: str = "") -> str:
    if x != x or x in (float("inf"),):
        return "inf"
    for div, suf in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(x) >= div:
            return f"{x / div:,.2f}{suf}{unit}"
    return f"{x:,.2f}{unit}"


def table(rows: list[list[str]], header: list[str]) -> str:
    cols = [header] + rows
    widths = [max(len(str(r[i])) for r in cols) for i in range(len(header))]
    def line(r):
        return "  ".join(str(c).rjust(w) for c, w in zip(r, widths))
    sep = "  ".join("-" * w for w in widths)
    return "\n".join([line(header), sep] + [line(r) for r in rows])


def run_report(model_key: str, gpu_key: str, tp: int, batches: list[int],
               context: int, a: Assumptions, cost_per_gpu_hr: float | None,
               outdir: str | None) -> str:
    m, g = get_model(model_key), get_gpu(gpu_key)
    lines: list[str] = []
    w = lines.append

    w(f"llm-roofline — {m.name} on {tp}x {g.name}")
    w(f"weights {a.weight_dtype.upper()}, KV {a.kv_dtype.upper()}, "
      f"MFU prefill {a.mfu_prefill:.0%} / decode {a.mfu_decode:.0%}, "
      f"HBM eff {a.hbm_efficiency:.0%}, context {context}")
    w("")

    total_p = m.param_count()
    active_p = m.active_param_count()
    kvb = m.kv_bytes_per_token(a.kv_bytes)
    w(f"params: {total_p / 1e9:.2f}B total / {active_p / 1e9:.2f}B active"
      f"   weights: {total_p * a.w_bytes / 1e9:.1f} GB"
      f"   KV: {kvb / 1024:.1f} KiB/token")

    bud = kv_budget(m, g, context, tp, a)
    w(f"KV budget @ ctx {context}: {bud.usable_kv_bytes / 1e9:.1f} GB free -> "
      f"{_fmt(float(bud.max_tokens))} tokens -> {bud.max_seqs_at_context} concurrent seqs")

    p = prefill(m, g, context, tp, a)
    w(f"prefill {context} tokens: {p.flops / 1e12:.1f} TFLOP, "
      f"intensity {p.intensity:,.0f} FLOP/B  ->  TTFT {p.ttft_s * 1e3:.1f} ms "
      f"({p.tok_per_s / 1e3:.1f}k tok/s)")
    bc = critical_batch(m, g, context, a)
    w(f"critical decode batch (memory->compute crossover): "
      f"{'inf (never compute-bound at this ctx)' if bc != bc or bc == float('inf') else f'~{bc:.0f}'}")
    w("")

    hdr = ["batch", "bound", "TPOT ms", "compute ms", "memory ms", "comm ms",
           "tok/s/GPU", "J/tok"]
    if cost_per_gpu_hr:
        hdr.append("$/Mtok")
    rows = []
    for b in batches:
        d = decode(m, g, b, context, tp, a)
        row = [b, d.bound, f"{d.tpot_s * 1e3:.2f}", f"{d.compute_s * 1e3:.2f}",
               f"{d.memory_s * 1e3:.2f}", f"{d.comm_s * 1e3:.2f}",
               f"{d.tok_per_s_per_gpu:,.0f}", f"{d.energy_per_tok_j:.3f}"]
        if cost_per_gpu_hr:
            row.append(f"{d.cost_per_mtok(cost_per_gpu_hr):.2f}")
        rows.append(row)
    w(table(rows, hdr))
    w("")
    d1 = decode(m, g, 1, context, tp, a)
    w(f"single-stream: TTFT {p.ttft_s * 1e3:.1f} ms, TPOT {d1.tpot_s * 1e3:.2f} ms "
      f"({1 / d1.tpot_s:,.0f} tok/s); weight-read floor {d1.floor_s * 1e3:.2f} ms "
      f"(raw HBM bound — nothing beats this without smaller weights or more GPUs)")

    if outdir:
        from .plots import plot_batch_curves, plot_roofline, plot_tp_scaling
        paths = [
            plot_roofline(m, g, tp, context, batches, a, outdir),
            plot_batch_curves(m, g, tp, context, batches, a, outdir),
            plot_tp_scaling(m, g, batches[len(batches) // 2], context, a, outdir),
        ]
        w("")
        w("plots: " + ", ".join(paths))
    return "\n".join(lines)


# ------------------------------------------------------------------ validation

# Tier-2 anchors: published MEASURED numbers, each verified at its URL on
# 2026-08-15. Throughputs are output tokens/s for the whole GPU group.
ANCHORS = [
    dict(model="llama-3-70b", gpu="h100", tp=2, isl=128, osl=128,
         measured=6092.0,
         what="Llama-3.3-70B FP8, 2x H100, ISL/OSL 128/128 (TensorRT-LLM perf overview)",
         url="https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html"),
    dict(model="llama-3-70b", gpu="h200", tp=2, isl=2048, osl=2048,
         measured=3776.0,
         what="Llama-3.3-70B FP8, 2x H200, ISL/OSL 2048/2048 (TensorRT-LLM perf overview)",
         url="https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html"),
    dict(model="llama-3-8b", gpu="h100", tp=1, isl=128, osl=128,
         measured=26401.0,
         what="Llama-3.1-8B FP8, 1x H100, ISL/OSL 128/128 (TensorRT-LLM perf overview)",
         url="https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html"),
    dict(model="llama-3-8b", gpu="h100", tp=1, isl=2048, osl=128,
         measured=3276.0,
         what="Llama-3.1-8B FP8, 1x H100, ISL/OSL 2048/128 (TensorRT-LLM perf overview)",
         url="https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html"),
    dict(model="llama-2-70b", gpu="h200", tp=1, isl=1024, osl=1024,
         measured=34864.0 / 8,
         what="Llama-2-70B FP8, MLPerf Inf v4.1 offline, 8x H200 = 34,864 tok/s "
              "-> 4,358/GPU (ISL/OSL approximated as 1024/1024; the cited page "
              "does not publish the dataset's mean lengths)",
         url="https://developer.nvidia.com/blog/nvidia-blackwell-platform-sets-new-llm-inference-records-in-mlperf-inference-v4-1"),
]


def run_validation(a: Assumptions | None = None) -> str:
    a = a or Assumptions(weight_dtype="fp8", kv_dtype="fp8")
    rows = []
    for c in ANCHORS:
        m, g = get_model(c["model"]), get_gpu(c["gpu"])
        pred = benchmark_throughput(m, g, c["tp"], c["isl"], c["osl"], a)
        err = (pred.output_tok_per_s - c["measured"]) / c["measured"] * 100.0
        rows.append([c["what"][:52], f"{c['measured']:,.0f}",
                     f"{pred.output_tok_per_s:,.0f}", f"{err:+.0f}%"])
    out = ["Tier-2 validation: model prediction vs published measured throughput",
           f"(assumptions: FP8 weights+KV, MFU {a.mfu_prefill:.0%}, "
           f"HBM eff {a.hbm_efficiency:.0%}; URLs in VALIDATION.md)", "",
           table(rows, ["benchmark", "measured tok/s", "predicted", "error"])]
    return "\n".join(out)
