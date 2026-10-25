"""CLI: python -m llm_roofline {report,demo,validate,list} ..."""

from __future__ import annotations

import argparse
import sys

from .analytics import Assumptions
from .gpus import GPUS
from .models import MODELS
from .report import run_report, run_validation


def _parse_batches(spec: str) -> list[int]:
    """'1..256' -> powers of two in range; '1,8,64' -> explicit list; '32' -> [32]."""
    if ".." in spec:
        lo, hi = (int(x) for x in spec.split("..", 1))
        out, b = [], 1
        while b <= hi:
            if b >= lo:
                out.append(b)
            b *= 2
        return out or [lo]
    return [int(x) for x in spec.split(",")]


def _assumptions(args) -> Assumptions:
    kw = dict(weight_dtype=args.dtype, kv_dtype=args.kv_dtype or args.dtype)
    if args.mfu is not None:
        kw.update(mfu_prefill=args.mfu, mfu_decode=args.mfu)
    return Assumptions(**kw)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="llm_roofline",
        description="Analytical roofline model of LLM inference on NVIDIA GPUs.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="full report + plots for one model/GPU config")
    r.add_argument("--model", required=True, help="e.g. llama-3-70b (see 'list')")
    r.add_argument("--gpu", required=True, help="e.g. h100 (see 'list')")
    r.add_argument("--tp", type=int, default=1, help="tensor-parallel degree")
    r.add_argument("--batch", default="1..256", help="'1..256' (pow2 sweep) or '1,8,64'")
    r.add_argument("--context", type=int, default=8192)
    r.add_argument("--dtype", default="bf16", choices=["bf16", "fp8"])
    r.add_argument("--kv-dtype", default=None, choices=["bf16", "fp8"])
    r.add_argument("--mfu", type=float, default=None,
                   help="override assumed MFU for both prefill and decode")
    r.add_argument("--cost-per-gpu-hr", type=float, default=None,
                   help="adds a $/Mtok column")
    r.add_argument("--out", default="results", help="PNG output dir ('' to skip plots)")

    sub.add_parser("demo", help="guided walkthrough of the model's headline claims")
    sub.add_parser("validate", help="prediction vs published measured numbers")
    sub.add_parser("list", help="available models and GPUs")

    args = p.parse_args(argv)

    if args.cmd == "report":
        print(run_report(args.model, args.gpu, args.tp, _parse_batches(args.batch),
                         args.context, _assumptions(args), args.cost_per_gpu_hr,
                         args.out or None))
    elif args.cmd == "validate":
        print(run_validation())
    elif args.cmd == "list":
        print("models:")
        for m in MODELS.values():
            extra = " (MoE)" if m.is_moe else ""
            extra += " (MLA)" if m.mla else ""
            print(f"  {m.key:14s} {m.param_count() / 1e9:7.1f}B total, "
                  f"{m.active_param_count() / 1e9:6.1f}B active{extra}")
        print("gpus:")
        for g in GPUS.values():
            fp8 = f"{g.fp8_tflops:.0f}" if g.fp8_tflops else "-"
            print(f"  {g.key:8s} {g.bf16_tflops:7.1f} TF bf16 / {fp8:>5s} TF fp8, "
                  f"{g.hbm_gb:.0f} GB @ {g.hbm_gbps / 1000:.2f} TB/s, {g.interconnect}")
    elif args.cmd == "demo":
        from .demo import run_demo
        run_demo()
    return 0


if __name__ == "__main__":
    sys.exit(main())
