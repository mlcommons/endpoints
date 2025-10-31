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
pip install torch transformers
```

## Usage

```bash
# Non-streaming mode (faster, no TTFT metrics)
python run_tinyllm.py

# Streaming mode (enables TTFT metrics)
python run_tinyllm.py --streaming
```

## Output

Results are saved to `tinyllm_benchmark_report/` directory containing:

- `result_summary.json` - Performance metrics
- `outputs.jsonl` - Generated responses
- `runtime_settings.json` - Configuration used

## Notes

- Uses TinyLLM model (~15MB) for quick testing
- Demonstrates framework components, not production benchmarking
- For remote endpoints, use the CLI commands instead (see [CLI_QUICK_REFERENCE.md](../../docs/CLI_QUICK_REFERENCE.md))
