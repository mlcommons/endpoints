# Benchmark Comparison Example

Compare `inference-endpoint` with vLLM's benchmarking tool using identical prompts.

## Prerequisites

**Install vLLM**:

```bash
pip install vllm
```

**Running inference server** (OpenAI-compatible):

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000
```

## Usage

```bash
cd examples/03_BenchmarkComparison
python compare_with_vllm.py --model "Qwen/Qwen2.5-0.5B-Instruct" --endpoint http://localhost:8000
```

### Options

| Option                | Description                      | Default                 |
| --------------------- | -------------------------------- | ----------------------- |
| `--model`, `-m`       | Model name (required)            | -                       |
| `--num-prompts`, `-n` | Number of prompts                | 100                     |
| `--endpoint`          | Server URL                       | `http://localhost:8000` |
| `--max-output-tokens` | Max output tokens                | 2000                    |
| `--timeout`           | Timeout in seconds               | 900                     |
| `--workers`           | Number of workers                | 1                       |
| `--verbose`, `-v`     | Show full output from each run   | -                       |
| `--dry`               | Print commands without executing | -                       |

### Example

```bash
python compare_with_vllm.py \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --num-prompts 200 \
    --max-output-tokens 1000
```

## Output

The script runs both benchmarks and displays a comparison table:

```

$ python examples/03_BenchmarkComparison/compare_with_vllm.py --model Qwen/Qwen2.5-0.5B-Instruct --num-prompts 10000

====================================================================================================
Metric                              | Inference Endpoint        | vLLM Benchmark
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
Test Duration (s)                   | 284.28                    | 300.69
----------------------------------------------------------------------------------------------------
Throughput (req/s)                  | 35.18                     | 33.26
Total Generated Tokens              | 4446263                   | 4626060
Output Token Throughput (tok/s)     | 15640.65                  | 15384.64
----------------------------------------------------------------------------------------------------
Mean TTFT (ms)                      | 137112.88                 | 146093.86
Median TTFT (ms)                    | 137092.46                 | 145656.40
P99 TTFT (ms)                       | 270902.92                 | 281810.49
----------------------------------------------------------------------------------------------------
Mean TPOT (ms)                      | 15.85                     | 15.60
Median TPOT (ms)                    | 15.56                     | 15.61
P99 TPOT (ms)                       | 36.47                     | 23.49
----------------------------------------------------------------------------------------------------
Mean ITL (ms)                       | 15.85                     | 15.42
Median ITL (ms)                     | 15.56                     | 12.17
P99 ITL (ms)                        | 36.47                     | 35.96
----------------------------------------------------------------------------------------------------
Mean Output Length (tokens)         | 444                       | 462
Median Output Length (tokens)       | 401                       | 406
P99 Output Length (tokens)          | 2000                      | 2000
====================================================================================================

```
