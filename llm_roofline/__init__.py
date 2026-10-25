"""llm-roofline: first-principles analytical performance model of LLM
inference on NVIDIA datacenter GPUs."""

from .analytics import (Assumptions, benchmark_throughput, critical_batch,
                        decode, kv_budget, prefill, tp_scaling)
from .gpus import GPUS, GPUSpec, get_gpu
from .models import MODELS, ModelSpec, get_model

__version__ = "0.1.0"
__all__ = [
    "Assumptions", "benchmark_throughput", "critical_batch", "decode",
    "kv_budget", "prefill", "tp_scaling", "GPUS", "GPUSpec", "get_gpu",
    "MODELS", "ModelSpec", "get_model", "__version__",
]
