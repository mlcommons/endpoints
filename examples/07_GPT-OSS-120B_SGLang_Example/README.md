# GPT-OSS-120B SGLang Example

This example demonstrates how to benchmark the GPT-OSS-120B model using SGLang as the inference backend with three evaluation datasets:

- **GPQA (Graduate-Level Google-Proof Q&A)**: Diamond subset for testing reasoning capabilities
- **AIME 2025**: Mathematical problem-solving benchmark
- **LiveCodeBench**: Coding evaluation benchmark

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [1. Setting Up the SGLang Endpoint](#1-setting-up-the-sglang-endpoint)
  - [2. Setting Up LiveCodeBench](#2-setting-up-livecodebench)
- [Running the Benchmark](#running-the-benchmark)
- [Evaluation Scripts](#evaluation-scripts)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.12+
- CUDA-capable GPU(s) with sufficient VRAM (recommended: 8x H200 or B200 GPUs)
- SGLang installed (see setup instructions below)
- Git
- pip

## Setup Instructions

### 1. Setting Up the SGLang Endpoint

You have two options for setting up the GPT-OSS-120B model with SGLang:

#### Option A: Using MLCommons MLPerf Inference Reference Implementation

The official MLPerf Inference reference implementation for GPT-OSS-120B provides detailed instructions for model setup, data preparation, and server deployment.

1. **Clone the MLCommons Inference Repository**:

   ```bash
   git clone https://github.com/mlcommons/inference.git
   cd inference/language/gpt-oss-120b
   ```

2. **Follow the Setup Instructions**:

   - Review the README at [https://github.com/mlcommons/inference/tree/master/language/gpt-oss-120b](https://github.com/mlcommons/inference/tree/master/language/gpt-oss-120b)
   - Download the model weights
   - Set up the required dependencies
   - Configure the environment

3. **Launch the SGLang Server**:
   Follow the instructions in the MLPerf reference implementation to start the SGLang server. The typical command looks like:
   ```bash
   ./sglang/run_server.sh \
        --model_path /path/to/gpt-oss-120b/model/ \
        --dp <Number of GPUs> \
        --stream_interval 100
   ```

#### Option B: Direct SGLang Installation

If you already have the model weights or prefer a direct approach, follow the [instructions from SGLang](https://docs.sglang.io/basic_usage/gpt_oss.html) on how to set up and deploy GPT-OSS. Make sure it is set to port 30000.

### 2. Setting Up LiveCodeBench

LiveCodeBench has a few security concerns and dependency conflicts, so it is recommended to run LiveCodeBench via the
containerized workflow.

Follow the instructions in the [LiveCodeBench README](../../src/inference_endpoint/evaluation/livecodebench/README.md#running-the-container)

#### Non-containerized run (NOT RECOMMENDED)

If you prefer to run `lcb-service` standalone without the docker container, do the following:

```bash
# Enter your venv for inference-endpoint
source /path/to/inference-endpoint/venv/bin/activate

# Downgrade Huggingface Datasets to 3.6.0
pip install datasets==3.6.0

# Install other dependencies
pip install fastapi==0.128.0 uvicorn[standard]==0.40.0

# Enable cli-based calling of LCBServe
export ALLOW_LCB_LOCAL_EVAL=true
```

After these steps, the `LiveCodeBenchScorer` will fallback to running `lcb_serve` as a subprocess on the host.

## Running the Benchmark

### Main Benchmark Suite

The `run.py` script runs all three benchmarks (GPQA, AIME25, and LiveCodeBench) in sequence:

```bash
python run.py \
    --report-dir ./results \
    --num-repeats 1 \
    --min-duration 10 \
    --max-duration 600
```

**Arguments**:

- `--report-dir`: Directory to save benchmark results (default: `sglang_accuracy_report`)
- `--num-repeats`: Number of times to repeat each dataset (default: 1)
- `--min-duration`: Minimum benchmark duration in seconds (default: 10)
- `--max-duration`: Maximum benchmark duration in seconds (default: 600)
- `--force-regenerate`: Force regeneration of datasets even if they exist

### Expected Output

The benchmark will display:

- Progress bars for each dataset evaluation
- Pass@1 scores for each benchmark:
  - GPQA Diamond accuracy
  - AIME25 problem-solving accuracy
  - LiveCodeBench coding accuracy

Results are saved to the specified report directory with detailed event logs and metrics.

## Evaluation Scripts

Individual evaluation scripts are provided for running each benchmark separately. These can be run after run.py
is complete and the report directory has been generated.

### GPQA Evaluation

```bash
python eval_gpqa.py \
    --dataset-path datasets/gpqa/diamond/gpqa_diamond.parquet \
    --report-dir sglang_accuracy_report
```

### AIME25 Evaluation

```bash
python eval_aime.py \
    --dataset-path datasets/aime25/aime25.parquet \
    --report-dir sglang_accuracy_report
```

### LiveCodeBench Evaluation

```bash
python eval_livecodebench.py \
    --dataset-path datasets/livecodebench/release_v6/livecodebench_release_v6.parquet \
    --report-dir sglang_accuracy_report \
    --lcb-version release_v6 \
    --timeout 60
```

**Additional Arguments**:

- `--lcb-version`: LiveCodeBench version tag (default: `release_v6`)
- `--timeout`: Timeout in seconds for each test execution (default: 60)

## Configuration

### Endpoint Configuration

The default endpoint configuration is in `run.py`:

```python
SGLANG_SERVER_HOST = "localhost"
SGLANG_SERVER_PORT = 30000
SGLANG_ENDPOINT = f"http://{SGLANG_SERVER_HOST}:{SGLANG_SERVER_PORT}/generate"
```

To use a different endpoint, modify these constants or edit the script.

### Worker Configuration

The HTTP client uses 4 workers by default:

```python
http_config = HTTPClientConfig(
    endpoint_urls=[SGLANG_ENDPOINT],
    num_workers=4,
    api_type="sglang",
)
```

Adjust `num_workers` based on your workload and server capacity.

## Troubleshooting

### Server Connection Issues

**Problem**: Cannot connect to SGLang server

**Solutions**:

- Verify the server is running: `curl http://localhost:30000/health`
- Check firewall settings if using a remote server
- Ensure the port number matches in both server and client configurations

### Out of Memory Errors

**Problem**: CUDA out of memory errors

**Solutions**:

- Increase the tensor parallelism size (`--tp-size`)
- Reduce batch size in the generation config
- Use GPUs with more VRAM
- Check for memory leaks with `nvidia-smi`

### LiveCodeBench Dependency Conflicts

**Problem**: Dependency version conflicts with `datasets` package

**Solutions**:

- Use a separate virtual environment for LiveCodeBench
- Ensure `datasets==3.6.0` is installed (required by LiveCodeBench)
- Check that the LiveCodeBench installation completed successfully

### Slow Inference Speed

**Problem**: Benchmark takes too long to complete

**Solutions**:

- Verify GPU utilization with `nvidia-smi`
- Check server logs for bottlenecks
- Increase `num_workers` in the HTTP client configuration
- Consider using FlashInfer or other optimizations in SGLang

### Dataset Generation Errors

**Problem**: Errors during dataset loading

**Solutions**:

- Use `--force-regenerate` to regenerate datasets from scratch
- Check internet connection for downloading Hugging Face datasets
- Verify sufficient disk space for dataset caching
- Check Hugging Face credentials if datasets are gated

## Additional Resources

- **SGLang Documentation**: [https://sgl-project.github.io/](https://sgl-project.github.io/)
- **MLPerf Inference GPT-OSS-120B**: [https://github.com/mlcommons/inference/tree/master/language/gpt-oss-120b](https://github.com/mlcommons/inference/tree/master/language/gpt-oss-120b)
- **LiveCodeBench**: [https://github.com/LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)
- **GPQA Dataset**: [https://huggingface.co/datasets/Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa)
