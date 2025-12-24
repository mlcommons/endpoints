# Echo Server Benchmark Comparison

This example compares the performance of `inference-endpoint` and `vLLM benchmark_serving` tools using a local echo server. The echo server responds instantly, which isolates the benchmarking tool overhead from actual inference latency.

## Prerequisites

1. **inference-endpoint** installed in your environment
2. **vLLM venv** set up from example 03:
   ```bash
   cd ../03_BenchmarkComparison
   ./setup_vllm_venv.sh
   ```

## Usage

### Basic usage (30,000 requests, streaming mode)

```bash
python benchmark_echo_server.py
```

### With custom settings

```bash
# 10,000 requests with 8 workers
python benchmark_echo_server.py --num-prompts 10000 --workers 8

# Non-streaming mode
python benchmark_echo_server.py --no-stream

# Verbose output
python benchmark_echo_server.py --verbose
```

### Full options

```bash
python benchmark_echo_server.py --help
```

## What it measures

Since the echo server responds instantly (sub-millisecond), this benchmark primarily measures:

1. **Request throughput (req/s)** - How fast the benchmarking tool can issue and process requests
2. **Client overhead** - Time spent in HTTP connection handling, SSE parsing, etc.
3. **TTFT** - Time to first token (mostly client/network overhead)
4. **TPOT** - Time per output token (should be near-zero for echo server)

## Example Output

```
==========================================================================================
ECHO SERVER BENCHMARK COMPARISON
==========================================================================================
Metric                              | Inference-Endpoint     | vLLM Benchmark
------------------------------------------------------------------------------------------
Duration (s)                        | 4.50                   | 4.52
Request Throughput (req/s)          | 6666.67                | 6637.17
Output Token Throughput (tok/s)     | 59333.33               | 59054.12
------------------------------------------------------------------------------------------
Mean TTFT (ms)                      | 0.15                   | 0.12
Median TTFT (ms)                    | 0.10                   | 0.08
P99 TTFT (ms)                       | 0.45                   | 0.42
------------------------------------------------------------------------------------------
Mean TPOT (ms)                      | 0.01                   | 0.01
Median TPOT (ms)                    | 0.01                   | 0.01
P99 TPOT (ms)                       | 0.02                   | 0.02
==========================================================================================

✓ inference-endpoint is 0.4% faster (6667 vs 6637 req/s)
```

## Notes

- The echo server echoes back the user's message as the response
- Both tools use the Qwen/Qwen2.5-0.5B-Instruct tokenizer for token counting
- Results may vary based on system load and available CPU cores
