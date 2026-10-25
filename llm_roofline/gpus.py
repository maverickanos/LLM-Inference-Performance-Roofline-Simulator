"""GPU spec database.

All tensor-core throughput numbers are DENSE TFLOPS. NVIDIA's headline
marketing numbers are usually the 2:4 structured-sparsity figures, which are
exactly 2x the dense rate; LLM inference GEMMs are dense, so the sparse
numbers are the wrong ones to plan with. Each entry cites the source used to
verify the dense figure (verified 2026-08-15).

`interconnect_gbps` is the per-GPU TOTAL bidirectional bandwidth (the number
NVIDIA quotes, e.g. "900 GB/s NVLink"). A ring all-reduce sends and receives
concurrently on one direction each, so the usable per-direction link rate is
interconnect_gbps / 2 (see analytics.allreduce_time).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSpec:
    key: str                 # registry key, e.g. "h100"
    name: str                # marketing name
    bf16_tflops: float       # dense BF16 tensor-core TFLOPS
    fp8_tflops: float | None  # dense FP8 tensor-core TFLOPS (None: no FP8 units)
    hbm_gb: float            # memory capacity, GB (1e9 bytes)
    hbm_gbps: float          # memory bandwidth, GB/s
    interconnect_gbps: float  # total bidirectional GPU-to-GPU bandwidth, GB/s
    interconnect: str        # "NVLink" | "PCIe"
    tdp_w: float             # board power, W
    source: str              # URL the dense numbers were verified against

    def peak_flops(self, dtype: str) -> float:
        """Dense peak in FLOP/s for a weight dtype ("bf16" | "fp8").

        If FP8 weights are requested on a GPU without FP8 tensor cores
        (A100), math falls back to BF16 rate: weights are stored in 8 bits
        but dequantized to BF16 for the GEMM (weight-only-quant kernels).
        """
        if dtype == "fp8" and self.fp8_tflops is not None:
            return self.fp8_tflops * 1e12
        if dtype in ("bf16", "fp8"):
            return self.bf16_tflops * 1e12
        raise ValueError(f"unknown dtype {dtype!r} (expected 'bf16' or 'fp8')")

    @property
    def hbm_bytes(self) -> float:
        return self.hbm_gb * 1e9

    @property
    def hbm_bw(self) -> float:  # bytes/s
        return self.hbm_gbps * 1e9


_GPUS = [
    GPUSpec(
        key="a100", name="A100-SXM-80GB",
        # Datasheet: BF16 624 TFLOPS "with sparsity" -> 312 dense. No FP8 units
        # (Ampere; INT8 only). HBM2e 80 GB @ 2,039 GB/s; NVLink 600 GB/s; 400 W SXM.
        # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-a4-nvidia-1485612-r12-web.pdf
        bf16_tflops=312.0, fp8_tflops=None,
        hbm_gb=80.0, hbm_gbps=2039.0,
        interconnect_gbps=600.0, interconnect="NVLink", tdp_w=400.0,
        source="https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-a4-nvidia-1485612-r12-web.pdf",
    ),
    GPUSpec(
        key="h100", name="H100-SXM",
        # Datasheet: BF16 1,979 TFLOPS and FP8 3,958 TFLOPS are the sparsity
        # figures -> dense 989 / 1,979. HBM3 80 GB @ 3.35 TB/s; NVLink 900 GB/s; 700 W.
        # https://www.nvidia.com/en-us/data-center/h100/ (H100 SXM column)
        bf16_tflops=989.0, fp8_tflops=1979.0,
        hbm_gb=80.0, hbm_gbps=3350.0,
        interconnect_gbps=900.0, interconnect="NVLink", tdp_w=700.0,
        source="https://www.nvidia.com/en-us/data-center/h100/",
    ),
    GPUSpec(
        key="h200", name="H200-SXM",
        # Same GH100 compute silicon as H100-SXM (989 BF16 / 1,979 FP8 dense);
        # upgrade is memory: HBM3e 141 GB @ 4.8 TB/s. NVLink 900 GB/s; 700 W.
        # https://www.nvidia.com/en-us/data-center/h200/
        bf16_tflops=989.0, fp8_tflops=1979.0,
        hbm_gb=141.0, hbm_gbps=4800.0,
        interconnect_gbps=900.0, interconnect="NVLink", tdp_w=700.0,
        source="https://www.nvidia.com/en-us/data-center/h200/",
    ),
    GPUSpec(
        key="b200", name="B200-SXM",
        # DGX B200 datasheet: 72 PFLOPS FP8 "training" across 8 GPUs is the
        # sparse figure -> 9 PF/GPU sparse -> 4,500 TFLOPS dense FP8; BF16 is
        # half that (2,250 dense). Shipping spec is 180 GB HBM3e per GPU
        # (early announcements said 192 GB) @ ~7.7 TB/s; NVLink5 1.8 TB/s; 1,000 W.
        # https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet
        # https://www.runpod.io/articles/guides/nvidia-b200 (180 GB, 1000 W confirmation)
        bf16_tflops=2250.0, fp8_tflops=4500.0,
        hbm_gb=180.0, hbm_gbps=7700.0,
        interconnect_gbps=1800.0, interconnect="NVLink", tdp_w=1000.0,
        source="https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet",
    ),
    GPUSpec(
        key="l40s", name="L40S",
        # Datasheet: FP16/BF16 362.05 TFLOPS dense (733 with sparsity), FP8 733
        # dense (1,466 with sparsity). NOTE: Ada quotes tensor rates with FP16
        # accumulate; FP32-accumulate GEMMs run at half these rates.
        # GDDR6 48 GB @ 864 GB/s; no NVLink (PCIe Gen4 x16, 64 GB/s bidir); 350 W.
        # https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413
        # https://lenovopress.lenovo.com/lp1812.pdf (dense vs sparsity table)
        bf16_tflops=362.05, fp8_tflops=733.0,
        hbm_gb=48.0, hbm_gbps=864.0,
        interconnect_gbps=64.0, interconnect="PCIe", tdp_w=350.0,
        source="https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413",
    ),
    GPUSpec(
        key="rtx4090", name="RTX 4090",
        # Ada whitepaper (App. A): FP16 tensor with FP32 accumulate 165.2 TFLOPS
        # dense (330.3 sparse); FP8 with FP32 accumulate 330.3 dense (660.6
        # sparse). FP16-accumulate paths are 2x but lossy for LLM accumulations.
        # GDDR6X 24 GB @ 1,008 GB/s; no NVLink (PCIe Gen4 x16); 450 W.
        # https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
        bf16_tflops=165.2, fp8_tflops=330.3,
        hbm_gb=24.0, hbm_gbps=1008.0,
        interconnect_gbps=64.0, interconnect="PCIe", tdp_w=450.0,
        source="https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf",
    ),
]

GPUS: dict[str, GPUSpec] = {g.key: g for g in _GPUS}
_ALIASES = {
    "a100-sxm-80gb": "a100", "h100-sxm": "h100", "h200-sxm": "h200",
    "b200-sxm": "b200", "4090": "rtx4090", "rtx-4090": "rtx4090",
}


def get_gpu(key: str) -> GPUSpec:
    k = key.lower()
    k = _ALIASES.get(k, k)
    if k not in GPUS:
        raise KeyError(f"unknown GPU {key!r}; available: {', '.join(GPUS)}")
    return GPUS[k]
