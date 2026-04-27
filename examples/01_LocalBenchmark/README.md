# Local Benchmark Example

Demonstrates benchmarking a locally-hosted HuggingFace model using custom components.

## Overview

This example shows how to:

- Create a custom `DataLoader` with hardcoded prompts
- Implement a synchronous `SampleIssuer` for local models
- Use event hooks to monitor benchmark progress
- Configure runtime settings with MLCommons rulesets

## Prerequisites

```bash
# Install PyTorch and Transformers
bash install_torch.sh

# Or manually
uv pip install torch transformers
```

## Usage

```bash
# Non-streaming mode (faster, no TTFT metrics)
uv run python run_tinyllm.py

# Streaming mode (enables TTFT metrics)
uv run python run_tinyllm.py --streaming
```

## Output

Results are saved to `tinyllm_benchmark_report/` directory containing:

- `result_summary.json` - Performance metrics
- `outputs.jsonl` - Generated responses
- `runtime_settings.json` - Configuration used

## Notes

- Uses TinyLlama-1.1B-Chat model (~1.1GB) - popular small model with 4M+ downloads/month
- Demonstrates framework components, not production benchmarking
- For remote endpoints, use the CLI commands instead (see [CLI_QUICK_REFERENCE.md](../../docs/CLI_QUICK_REFERENCE.md))
