# LLM Inference Performance & Roofline Simulator

A Python-based analytical simulator for estimating Large Language Model (LLM) inference performance across GPU architectures, model configurations, and deployment strategies.

The project uses hardware specifications, model architecture, memory bandwidth, and communication costs to estimate latency, throughput, KV-cache capacity, energy per token, and serving cost. It applies roofline analysis to identify compute-bound and memory-bound workloads without requiring physical GPU hardware.

> **Project basis:** This repository is based on [llm-roofline by Azra Bano](https://github.com/azrabano23/llm-roofline). Preserve the original license and attribution when redistributing or publishing modifications.

## Overview

LLM inference performance depends on more than raw GPU compute. Model size, memory bandwidth, sequence length, batching, and multi-GPU communication all affect end-to-end latency and throughput.

This simulator provides an analytical approach to studying these trade-offs before running hardware benchmarks.

### Key objectives

- Estimate Transformer prefill and autoregressive decode performance.
- Identify compute, memory-bandwidth, and communication bottlenecks.
- Evaluate batch sizes, context lengths, precision formats, and tensor parallelism.
- Estimate KV-cache capacity and concurrent sequence limits.
- Compare theoretical performance bounds with published benchmark measurements.
- Visualize hardware utilization and performance scaling.

## Features

### 1. Analytical inference modelling

Estimate key inference metrics using model and hardware specifications:

- Time to First Token (TTFT).
- Time per Output Token (TPOT).
- Tokens per second and throughput per GPU.
- KV-cache memory consumption and capacity.
- Energy per token and estimated serving cost.

### 2. Roofline performance analysis

Use arithmetic intensity, compute throughput, and memory bandwidth to understand workload limitations.

- Estimate theoretical compute and memory performance bounds.
- Analyze how batching changes arithmetic intensity.
- Identify compute-bound and memory-bound operating regions.
- Compare model configurations against hardware capabilities.

### 3. Hardware and model configuration

Evaluate different combinations of supported GPU architectures, model families, numerical precision, and tensor-parallel configurations.

Supported GPU configurations include A100-SXM-80GB, H100-SXM, H200-SXM, B200, L40S, and RTX 4090.

Supported models include Llama-3-8B, Llama-3-70B, Llama-2-70B, Mistral-7B, Qwen2.5-32B, Mixtral-8x7B, and DeepSeek-V3.

Refer to the source configuration files for exact supported options.

### 4. Batch-size and scaling analysis

Sweep batch sizes to examine their impact on latency, throughput, and memory requirements.

Analyze tensor-parallel scaling to understand how distributing model weights across GPUs changes compute requirements, memory traffic, and collective communication overhead.

### 5. Validation and testing

The project includes checks against published model parameter counts, comparisons with published inference benchmarks, and tests that enforce theoretical memory-bandwidth performance bounds.

Predictions are analytical estimates rather than guarantees of real-world performance.

## Methodology

The simulator combines model-level workload estimates with hardware-level performance constraints.

### Decode latency

Autoregressive decoding repeatedly reads model weights and accesses the KV cache. At low batch sizes, memory bandwidth often limits performance.

A simplified lower bound is:

\`\`\`text
Decode time >= Weight bytes / Effective memory bandwidth
\`\`\`

### Compute-bound versus memory-bound performance

The roofline model estimates attainable performance from arithmetic intensity and hardware limits.

\`\`\`text
Attainable FLOP/s <= min(
    Peak compute throughput,
    Memory bandwidth × Arithmetic intensity
)
\`\`\`

### Prefill and decode

- **Prefill:** Processes the input prompt and is often compute-bound for sufficiently large workloads.
- **Decode:** Generates output tokens sequentially and is often memory-bound at low batch sizes.

Actual behavior depends on model architecture, sequence length, precision, kernels, and deployment configuration.

### Multi-GPU performance

Tensor parallelism distributes model computation and memory requirements across GPUs. Communication introduces additional overhead, so scaling is not perfectly linear.

The simulator estimates communication costs using an idealized ring all-reduce model.

## Technology Stack

- **Language:** Python
- **Numerical computing:** NumPy
- **Visualization:** Matplotlib
- **Testing:** Python test suite
- **Tools:** Git, command-line interface

The analytical model can run without a physical GPU. NumPy and Matplotlib are the primary runtime dependencies described by the project.

## Getting Started

Clone the repository and navigate to its directory:

\`\`\`bash
git clone https://github.com/azrabano23/llm-roofline.git
cd llm-roofline
\`\`\`

Run the available examples and tests:

\`\`\`bash
./run.sh demo
./run.sh test
./run.sh report
\`\`\`

Generate a report for a particular configuration:

\`\`\`bash
python -m llm_roofline report \\
    --model llama-3-70b \\
    --gpu h100 \\
    --tp 4 \\
    --batch 1..256 \\
    --context 8192
\`\`\`

Validate predictions against the project's benchmark references:

\`\`\`bash
python -m llm_roofline validate
\`\`\`

List supported models and hardware:

\`\`\`bash
python -m llm_roofline list
\`\`\`

Consult the repository's scripts and CLI help for the current command options.

## Example: Performance Analysis

Consider evaluating a 70-billion-parameter Transformer model on multiple H100 GPUs.

The analysis can investigate:

1. How decode latency changes with batch size.
2. How memory bandwidth constrains token generation.
3. How context length affects KV-cache memory consumption.
4. How tensor parallelism changes per-token latency.
5. How communication overhead affects multi-GPU scaling.

The resulting estimates and plots help explain performance trends and identify configurations worth benchmarking on real hardware.

## Limitations

This simulator is an analytical approximation of LLM inference performance.

- Model FLOP utilization is an assumption rather than a prediction of kernel efficiency.
- Kernel launch overhead, detailed attention-kernel behavior, and scheduler effects are not fully modelled.
- Communication estimates use an idealized ring all-reduce model.
- Real serving throughput depends on continuous batching, scheduling policies, workload distribution, and service-level objectives.
- MoE routing and multi-GPU deployment are represented using simplified assumptions.
- Hardware specifications and benchmark comparisons depend on the accuracy and applicability of their source data.

Analytical estimates should be validated against real measurements before making production capacity-planning decisions.

## References

- Williams, Waterman, and Patterson. *Roofline: An Insightful Visual Performance Model for Multicore Architectures*. CACM, 2009.
- Pope et al. *Efficiently Scaling Transformer Inference*. https://arxiv.org/abs/2211.05102
- Agrawal et al. *Vidur: A Large-Scale Simulation Framework for LLM Inference*. https://arxiv.org/abs/2405.05465
- NVIDIA TensorRT-LLM performance overview: https://nvidia.github.io/TensorRT-LLM/performance/perf-overview.html
- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437

## License and Attribution

This project is based on the existing [llm-roofline repository](https://github.com/azrabano23/llm-roofline).

Retain the original license, copyright notices, and required attribution. Clearly document any modifications, experiments, or additional features you implement.

## Future Improvements

Potential extensions for further development include:

- Calibration against locally measured inference benchmarks.
- Sensitivity analysis for memory bandwidth and compute throughput.
- Additional communication models for multi-GPU deployments.
- Comparative analysis of quantization and KV-cache precision.
- Automated configuration sweeps with reproducible result reports.
