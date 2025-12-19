# Examples

This directory contains examples demonstrating how to use the MLPerf Inference Endpoint Benchmarking System.

## Available Examples

### [01_LocalBenchmark](01_LocalBenchmark/)

Local model benchmarking with a small HuggingFace model, demonstrating custom DataLoader and event hooks.

### [02_ServerBenchmarking](02_ServerBenchmarking/)

Benchmarking a real-world model served via open-source serving systems such as [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang).

### [03_BenchmarkComparison](03_BenchmarkComparison/)

Compare `inference-endpoint` with [vLLM's](https://github.com/vllm-project/vllm) benchmarking tool.

## Getting Help

- For general usage: See main [README](../README.md)
- For development: See [DEVELOPMENT.md](../docs/DEVELOPMENT.md)
- For CLI reference: See [CLI_QUICK_REFERENCE.md](../docs/CLI_QUICK_REFERENCE.md)
