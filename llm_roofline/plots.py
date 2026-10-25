"""Roofline and scaling plots (matplotlib, static PNG, light theme).

Palette/chrome follow a validated accessible palette: series colors
blue #2a78d6 / orange #eb6834 / aqua #1baf7a (first three slots validate
all-pairs for color-vision deficiency), ink #0b0b0b, secondary #52514e,
muted #898781, hairline grid #e1e0d9, axis #c3c2b7, surface #fcfcfb.
Structural lines (ceilings, floors, ideals) wear ink/muted grays so color
follows the data entity, not decoration. One axis per panel, always.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analytics import (Assumptions, critical_batch, decode,
                        decode_flops_per_token, prefill)
from .gpus import GPUSpec
from .models import ModelSpec

INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"


def _style(ax, title: str):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=SECONDARY)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)


def _fig(nrows=1, height=4.6):
    fig, axes = plt.subplots(nrows, 1, figsize=(8.2, height * nrows), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _legend(ax):
    leg = ax.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY)
    return leg


def _save(fig, outdir: str, name: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roofline(m: ModelSpec, gpu: GPUSpec, tp: int, context: int,
                  batches, a: Assumptions, outdir: str) -> str:
    """Classic roofline (per-GPU): attainable TFLOP/s vs arithmetic intensity,
    with the prefill operating point and the decode operating points per batch."""
    fig, ax = _fig(height=5.2)
    _style(ax, f"Roofline — {m.name} on {tp}x {gpu.name} "
               f"({a.weight_dtype.upper()} weights, ctx {context})")

    ai = np.logspace(-0.5, 4.2, 400)
    bw = gpu.hbm_bw  # raw bandwidth ceiling, per GPU
    peak_bf16 = gpu.bf16_tflops * 1e12
    roof_bf16 = np.minimum(peak_bf16, ai * bw)
    ax.plot(ai, roof_bf16 / 1e12, color=INK, linewidth=2,
            label=f"BF16 roof ({gpu.bf16_tflops:.0f} TF, {gpu.hbm_gbps / 1000:.2f} TB/s)")
    if gpu.fp8_tflops:
        roof_fp8 = np.minimum(gpu.fp8_tflops * 1e12, ai * bw)
        ax.plot(ai, roof_fp8 / 1e12, color=MUTED, linewidth=2, linestyle="--",
                label=f"FP8 roof ({gpu.fp8_tflops:.0f} TF)")

    # decode operating points: achieved FLOP/s per GPU vs achieved intensity
    xs, ys = [], []
    for b in batches:
        d = decode(m, gpu, b, context, tp, a)
        step_flops = b * decode_flops_per_token(m, context)
        xs.append(d.intensity)
        ys.append(step_flops / d.tpot_s / tp / 1e12)  # achieved FLOP/s per GPU
    ax.plot(xs, ys, color=ORANGE, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=5,
            label=f"decode, batch {batches[0]}-{batches[-1]}")
    ax.annotate(f"B={batches[0]}", (xs[0], ys[0]), textcoords="offset points",
                xytext=(6, -12), fontsize=8, color=SECONDARY)
    ax.annotate(f"B={batches[-1]}", (xs[-1], ys[-1]), textcoords="offset points",
                xytext=(6, 6), fontsize=8, color=SECONDARY)

    p = prefill(m, gpu, context, tp, a)
    peak = gpu.peak_flops(a.weight_dtype)
    ax.plot([p.intensity], [peak * a.mfu_prefill / 1e12], color=BLUE, marker="o",
            markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.2,
            linestyle="none", zorder=6, label=f"prefill (L={context})")
    ax.annotate(f"prefill @ MFU {a.mfu_prefill:.0%}",
                (p.intensity, peak * a.mfu_prefill / 1e12),
                textcoords="offset points", xytext=(-10, 10),
                fontsize=8, color=SECONDARY, ha="right")

    knee = peak_bf16 / bw
    ax.annotate(f"machine balance\n{knee:.0f} FLOP/byte", (knee, gpu.bf16_tflops),
                textcoords="offset points", xytext=(-8, -34), fontsize=8,
                color=MUTED, ha="right")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP / HBM byte)", color=SECONDARY, fontsize=9)
    ax.set_ylabel("attainable TFLOP/s per GPU", color=SECONDARY, fontsize=9)
    _legend(ax)
    return _save(fig, outdir, "roofline.png")


def plot_batch_curves(m: ModelSpec, gpu: GPUSpec, tp: int, context: int,
                      batches, a: Assumptions, outdir: str) -> str:
    """TPOT and per-GPU throughput vs batch (two panels, shared x, one axis each)."""
    res = [decode(m, gpu, b, context, tp, a) for b in batches]
    fig, (ax1, ax2) = _fig(nrows=2, height=3.4)

    _style(ax1, f"Decode latency vs batch — {m.name}, {tp}x {gpu.name}, ctx {context}")
    ax1.plot(batches, [r.tpot_s * 1e3 for r in res], color=BLUE, linewidth=2,
             marker="o", markersize=5, markeredgecolor=SURFACE, label="TPOT (model)")
    floor = res[0].floor_s * 1e3
    ax1.axhline(floor, color=MUTED, linewidth=1.5, linestyle=":",
                label=f"weight-read floor {floor:.2f} ms")
    bc = critical_batch(m, gpu, context, a)
    if np.isfinite(bc) and batches[0] < bc < batches[-1]:
        ax1.axvline(bc, color=AXIS, linewidth=1.2, linestyle="--")
        ax1.annotate(f"critical batch ~{bc:.0f}\nmemory->compute bound",
                     (bc, ax1.get_ylim()[1]), fontsize=8, color=SECONDARY,
                     textcoords="offset points", xytext=(6, -24))
    ax1.set_xscale("log", base=2)
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel("ms per token step", color=SECONDARY, fontsize=9)
    _legend(ax1)

    _style(ax2, "Generation throughput vs batch")
    ax2.plot(batches, [r.tok_per_s_per_gpu for r in res], color=ORANGE, linewidth=2,
             marker="o", markersize=5, markeredgecolor=SURFACE,
             label="tokens/s per GPU (model)")
    if np.isfinite(bc) and batches[0] < bc < batches[-1]:
        ax2.axvline(bc, color=AXIS, linewidth=1.2, linestyle="--")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("decode batch size (concurrent sequences)", color=SECONDARY, fontsize=9)
    ax2.set_ylabel("tokens / s / GPU", color=SECONDARY, fontsize=9)
    _legend(ax2)
    return _save(fig, outdir, "batch_curves.png")


def plot_tp_scaling(m: ModelSpec, gpu: GPUSpec, batch: int, context: int,
                    a: Assumptions, outdir: str, tps=(1, 2, 4, 8)) -> str:
    """Per-token decode speedup vs TP degree against the ideal linear line."""
    base = decode(m, gpu, batch, context, tp=1, a=a)
    speed = [base.tpot_s / decode(m, gpu, batch, context, tp=t, a=a).tpot_s for t in tps]
    fig, ax = _fig(height=4.8)
    _style(ax, f"Tensor-parallel decode scaling — {m.name} on {gpu.name}, "
               f"batch {batch}, ctx {context}")
    ax.plot(tps, tps, color=MUTED, linewidth=1.5, linestyle="--", label="ideal (linear)")
    ax.plot(tps, speed, color=BLUE, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label="model (ring all-reduce)")
    for t, s in zip(tps, speed):
        ax.annotate(f"{s / t:.0%}", (t, s), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8.5, color=SECONDARY)
    ax.set_xticks(list(tps))
    ax.set_xlabel("tensor-parallel degree", color=SECONDARY, fontsize=9)
    ax.set_ylabel("TPOT speedup vs TP=1", color=SECONDARY, fontsize=9)
    ax.annotate("labels: parallel efficiency", (0.98, 0.02), xycoords="axes fraction",
                ha="right", fontsize=8, color=MUTED)
    _legend(ax)
    return _save(fig, outdir, "tp_scaling.png")
