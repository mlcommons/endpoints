# Dataset Preset Testing

Unit tests for dataset preset transforms. These tests verify that presets correctly transform dataset columns without requiring end-to-end benchmark runs.

## Quick Start

```bash
# Run all preset tests
pytest tests/unit/dataset_manager/test_dataset_presets.py -v

# Run tests for a specific dataset
pytest tests/unit/dataset_manager/test_dataset_presets.py::TestCNNDailyMailPresets -v

# Exclude slow tests (Harmonize transform requires transformers)
pytest tests/unit/dataset_manager/test_dataset_presets.py -m "not slow" -v
```

## Preset Coverage

| Dataset       | Presets                         | Tests |
| ------------- | ------------------------------- | ----- |
| CNNDailyMail  | `llama3_8b`, `llama3_8b_sglang` | 6     |
| AIME25        | `gptoss`                        | 3     |
| GPQA          | `gptoss`                        | 3     |
| LiveCodeBench | `gptoss`                        | 3     |
| OpenOrca      | `llama2_70b`                    | 3     |

## Adding Tests for New Presets

When adding a new dataset preset, add a test class to `tests/unit/dataset_manager/test_dataset_presets.py`:

```python
import pandas as pd
import pytest
from inference_endpoint.dataset_manager.transforms import apply_transforms
from inference_endpoint.dataset_manager.predefined.my_dataset import MyDataset


class TestMyDatasetPresets:
    @pytest.fixture
    def sample_data(self):
        """Minimal sample data matching dataset schema."""
        return pd.DataFrame({
            "input_col1": ["value1"],
            "input_col2": ["value2"],
        })

    @pytest.fixture
    def transformed_data(self, sample_data):
        """Apply preset transforms to sample data."""
        transforms = MyDataset.PRESETS.my_preset()
        return apply_transforms(sample_data, transforms)

    def test_my_preset_instantiation(self):
        """Verify preset can be created."""
        transforms = MyDataset.PRESETS.my_preset()
        assert transforms is not None
        assert len(transforms) > 0

    def test_my_preset_transforms_apply(self, transformed_data):
        """Verify transforms apply without errors."""
        assert transformed_data is not None
        assert "prompt" in transformed_data.columns  # Expected output column

    def test_my_preset_output_format(self, transformed_data):
        """Verify output has expected format."""
        # Validate format-specific expectations
        assert len(transformed_data["prompt"][0]) > 0
```

If the preset uses `Harmonize` transform (requires `transformers` library), mark slow tests:

```python
@pytest.mark.slow
def test_my_preset_transforms_apply(self, transformed_data):
    # Test that requires transformers library
    pass
```

## Test Scope

✅ **Tests verify:**

- Preset instantiation
- Transform application without errors
- Required output columns exist
- Data is properly transformed

❌ **Tests do NOT verify:**

- Model inference accuracy
- API endpoint compatibility
- Throughput/latency metrics
- Full benchmark runs

See `src/inference_endpoint/dataset_manager/README.md` for dataset schema and preset creation details.
