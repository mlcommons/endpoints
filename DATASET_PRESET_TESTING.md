# Dataset Preset Testing Documentation

## Overview

This guide explains the unit testing solution for preset datasets in the MLPerf Inference Endpoint system. The tests verify that dataset transforms work correctly without requiring end-to-end benchmark runs or external compute resources.

## What Was Added

### 1. **Test File: `tests/unit/dataset_manager/test_dataset_presets.py`**

Comprehensive unit tests covering all dataset presets:

- **CNNDailyMail**: Tests for `llama3_8b` and `llama3_8b_sglang` presets
- **AIME25**: Tests for `gptoss` preset
- **GPQA**: Tests for `gptoss` preset
- **LiveCodeBench**: Tests for `gptoss` preset
- **OpenOrca**: Tests for `llama2_70b` preset

Each preset gets three types of tests:
1. **Instantiation test** - Verifies the preset can be created
2. **Transform application test** - Verifies transforms apply without errors
3. **Output validation test** - Verifies transforms produce expected output format

## Running the Tests

### Run all preset tests:
```bash
pytest tests/unit/dataset_manager/test_dataset_presets.py -v
```

### Run tests for a specific dataset:
```bash
pytest tests/unit/dataset_manager/test_dataset_presets.py::TestCNNDailyMailPresets -v
```

### Run a specific test:
```bash
pytest tests/unit/dataset_manager/test_dataset_presets.py::TestCNNDailyMailPresets::test_llama3_8b_transforms_apply -v
```

### Run with coverage:
```bash
pytest tests/unit/dataset_manager/test_dataset_presets.py --cov=src/inference_endpoint/dataset_manager --cov-report=html
```

## Test Structure

Each test class uses pytest fixtures to provide minimal sample data:

```python
@pytest.fixture
def sample_cnn_data(self):
    """Create minimal sample data matching CNN/DailyMail schema."""
    return pd.DataFrame({
        "article": ["..."],
        "highlights": ["..."],
    })
```

This approach:
- ✅ No external API calls or dataset downloads
- ✅ Tests run in <1 second (no network I/O)
- ✅ Minimal memory footprint
- ✅ Tests can run in CI/CD pipelines
- ✅ Simple to extend with new datasets

## Programmatic Dataset Usage (No YAML)

The schema reference documents how to use datasets without YAML configuration. See the `DATASET_SCHEMA_REFERENCE.md` for input/output column specifications.

### Load a dataset with preset programmatically:
```python
from inference_endpoint.dataset_manager.predefined.cnndailymail import CNNDailyMail

# Get transforms
transforms = CNNDailyMail.PRESETS.llama3_8b_sglang()

# Load dataset
dataset = CNNDailyMail.get_dataloader(transforms=transforms)

# Use in benchmark
sample = dataset.load_sample(0)
```

### Create and test custom dataset:
```python
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.dataset_manager.transforms import apply_transforms
import pandas as pd

# Create sample data
data = pd.DataFrame({
    "question": ["What is AI?"],
    "answer": ["Artificial Intelligence"]
})

# Get preset transforms
from inference_endpoint.dataset_manager.predefined.aime25 import AIME25
transforms = AIME25.PRESETS.gptoss()

# Apply transforms
result = apply_transforms(data, transforms)

# Verify
assert "prompt" in result.columns
assert len(result) == 1
```

## How Transform Tests Work

### Test Categories

1. **Instantiation Tests**
   - Verify preset functions can be called without errors
   - Ensure transforms are returned as a list
   - Quick smoke tests

2. **Application Tests**
   - Apply transforms to sample data
   - Verify output DataFrame has correct shape
   - Check that required output columns are created

3. **Validation Tests**
   - Verify transform output meets expected format
   - Check that data from source columns is properly embedded
   - Validate format-specific requirements (e.g., code delimiters, multiple choice format)

### Example Test Pattern

```python
def test_preset_name_transforms_apply(self, sample_data):
    """Test that transforms apply without errors."""
    # 1. Get the preset
    transforms = DatasetClass.PRESETS.preset_name()

    # 2. Apply to sample data
    result = apply_transforms(sample_data, transforms)

    # 3. Verify output
    assert result is not None
    assert len(result) == len(sample_data)
    assert "prompt" in result.columns  # or other expected column
```

## Extending the Tests

### Add a new dataset preset test:

1. **Create the test class** in `test_dataset_presets.py`:
```python
class TestNewDatasetPresets:
    """Test NewDataset presets."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data matching schema."""
        return pd.DataFrame({
            "column1": [...],
            "column2": [...],
        })

    def test_preset_name_instantiation(self):
        """Test preset can be instantiated."""
        transforms = NewDataset.PRESETS.preset_name()
        assert transforms is not None

    def test_preset_name_transforms_apply(self, sample_data):
        """Test transforms apply without errors."""
        transforms = NewDataset.PRESETS.preset_name()
        result = apply_transforms(sample_data, transforms)
        assert "prompt" in result.columns
```

2. **Import the dataset class** at the top:
```python
from inference_endpoint.dataset_manager.predefined.new_dataset import NewDataset
```

### Test when transforms change:

Since tests apply actual transforms to sample data, any change to a preset's transforms will automatically be caught:

```bash
# Run tests before making changes to preset
pytest tests/unit/dataset_manager/test_dataset_presets.py -v

# Modify src/inference_endpoint/dataset_manager/predefined/cnndailymail/presets.py
# Tests will catch any breaking changes:
pytest tests/unit/dataset_manager/test_dataset_presets.py::TestCNNDailyMailPresets -v
```

## What These Tests Don't Cover

These are **unit tests** for transforms, not end-to-end benchmark tests:

- ❌ Network latency or throughput metrics
- ❌ Model inference accuracy
- ❌ Full dataset loading (only sample rows)
- ❌ API endpoint responses
- ❌ External service dependencies

These require separate integration tests or actual benchmark runs.

## Integration with CI/CD

Add to your CI pipeline:

```yaml
# Example GitHub Actions or similar
- name: Test Dataset Presets
  run: |
    pytest tests/unit/dataset_manager/test_dataset_presets.py \
      -v \
      --cov=src/inference_endpoint/dataset_manager \
      --cov-report=json
```

## Key Benefits

✅ **Fast** - Tests run in <5 seconds with no external dependencies
✅ **Reliable** - No flakiness from network calls or dataset availability
✅ **Maintainable** - Clear test structure, easy to extend
✅ **Coverage** - Catches transform regressions automatically
✅ **No resources** - Works with no GPU/compute, only CPU
✅ **Development friendly** - Run locally before committing

## Example Usage Scenarios

### Scenario 1: Verify transform changes don't break presets
```bash
# After modifying transforms.py:
pytest tests/unit/dataset_manager/test_dataset_presets.py -v
```

### Scenario 2: Test new preset implementation
```python
# In your preset function:
def new_preset() -> list[Transform]:
    return [Transform1(), Transform2()]

# Add unit test:
def test_new_preset_transforms_apply(self, sample_data):
    transforms = DatasetClass.PRESETS.new_preset()
    result = apply_transforms(sample_data, transforms)
    assert "expected_column" in result.columns
```

### Scenario 3: Validate dataset before full benchmark run
```bash
# Quick validation using pytest
pytest tests/unit/dataset_manager/test_dataset_presets.py -v
```

## Troubleshooting

### Test import errors
```bash
# Ensure src directory is in PYTHONPATH (from repo root)
export PYTHONPATH=./src:$PYTHONPATH
pytest tests/unit/dataset_manager/test_dataset_presets.py
```

### Missing dataset dependencies
Some presets may require optional tokenizers (e.g., Harmonize transform requires transformers).
Run with:
```bash
pytest tests/unit/dataset_manager/test_dataset_presets.py -m "not slow" -v
```

### Debugging a specific test
```bash
pytest tests/unit/dataset_manager/test_dataset_presets.py::TestClass::test_method -vvs
```

## Next Steps

1. **Run the tests** to verify your current setup:
   ```bash
   pytest tests/unit/dataset_manager/test_dataset_presets.py -v
   ```

2. **Add to pre-commit** to catch regressions automatically:
   ```bash
   pre-commit run pytest
   ```

3. **Extend tests** when adding new dataset presets or transforms
