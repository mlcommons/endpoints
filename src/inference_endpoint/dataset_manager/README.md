# Dataset Manager

The `dataset_manager` module provides a flexible, composable system for loading, transforming, and preparing benchmark datasets for inference endpoint evaluation. It separates the concepts of **datasets** and **transforms** to enable reusable, modular data processing pipelines.

## Table of Contents

- [Core Concepts](#core-concepts)
  - [What is a Dataset?](#what-is-a-dataset)
  - [What is a Transform?](#what-is-a-transform)
  - [When to Create a New Dataset](#when-to-create-a-new-dataset)
- [Architecture](#architecture)
  - [Transform Pipeline](#transform-pipeline)
  - [Model-Specific vs Adapter-Specific Transforms](#model-specific-vs-adapter-specific-transforms)
- [Usage](#usage)
  - [Using Predefined Datasets](#using-predefined-datasets)
  - [Loading Custom Datasets](#loading-custom-datasets)
  - [Creating New Datasets](#creating-new-datasets)
  - [Creating Model Presets](#creating-model-presets)
- [Transform Types](#transform-types)
- [Examples](#examples)

## Core Concepts

### What is a Dataset?

A **dataset** in the context of Inference Endpoint is defined as:

> A set of keyed data with a set of **critical keys**. Critical keys define the core components of the data that are ingested by the model. Combinations of critical keys should be unique within the dataset.

**Critical keys** are the essential columns that define the dataset's identity. Non-critical keys (metadata, auxiliary information) can be retrieved by searching for rows in the original dataset and matching the critical keys.

**Example:** In the GPQA dataset, the critical keys are:

- `question`: The question text
- `choices`: The 4 multiple-choice options

Non-critical keys like `domain` and `subdomain` provide metadata but don't uniquely identify the sample.

### What is a Transform?

A **transform** is an operation that is uniformly applied to all rows of a dataset. A key property of transforms is:

> **Transforms must be reversible** if a copy of the original dataset is provided.

This reversibility principle means:

- **User prompt formatting** is reversible with an equivalent Regex extraction pattern
- **Dropping non-critical columns** is reversible if the original dataset is provided
- **Inserting static columns** is reversible by dropping those columns
- **Harmonizing data** (e.g., converting to model-specific formats) is reversible with a Regex pattern

This ensures that transforms don't fundamentally change the dataset's identity—they only reformat it for specific use cases.

### When to Create a New Dataset

A new `Dataset` subclass should be created when:

> A dataset variant **cannot be transformed** from an existing dataset with a set of transforms.

If different formatting or preprocessing can be achieved through transforms, it should remain the same base dataset with different transform presets.

**Example:** `AIME25` with different prompt formats for different models (GPT-OSS, Llama2, etc.) should be the same dataset with different transform presets, not separate dataset classes.

## Architecture

### Transform Pipeline

The transform system uses a pipeline architecture where transforms are applied sequentially to a pandas DataFrame. The pipeline consists of two main stages:

1. **Model-Specific Transforms** (Dataset + Model)

   - Applied first in the pipeline
   - Generate standardized columns (e.g., `prompt`, `user_prompt`)
   - Dataset-dependent and model-dependent
   - Defined in dataset preset modules (e.g., `aime25/presets.py`)

2. **Adapter-Specific Transforms** (Adapter)
   - Applied after model-specific transforms
   - Consume standardized columns from stage 1
   - Prepare data in the exact format required by the API
   - Adapter-dependent but dataset-agnostic
   - Defined in adapter classes (e.g., `openai_adapter.py`, `sglang/adapter.py`)

### Model-Specific vs Adapter-Specific Transforms

The architecture prevents a "cross-product explosion" of transforms by separating concerns:

```
┌─────────────────────────────────────────────────────────────┐
│  Dataset + Model Transforms (Preset)                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ UserPromptFormatter: {question} → "prompt" column    │  │
│  │ Other model-specific transforms...                    │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Adapter Transforms                                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Harmonize: "prompt" → "input_tokens"                  │  │
│  │ AddStaticColumns: Add API metadata                    │  │
│  │ ColumnFilter: Keep only API-required columns         │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Key Benefits:**

- **Reusability:** Adapter transforms work with any dataset that provides the expected columns
- **Maintainability:** Adding a new model requires only one preset per dataset
- **Clarity:** Clear separation between dataset formatting and API formatting

**Constraint:** Adapter-specific transforms **require certain columns** (e.g., `prompt`) to be present in the input DataFrame. These columns must be created by the end of the model-specific transform pipeline.

## Usage

### Using Predefined Datasets

Predefined datasets use a naming convention that separates the dataset from its model-specific transforms:

```
<dataset_name>::<model_preset>
```

**Example:**

```yaml
datasets:
  - name: "aime25::gptoss"
    type: "accuracy"
```

This specifies:

- Base dataset: `aime25`
- Model preset: `gptoss` (found in `aime25/presets.py`)

The adapter-specific transforms are automatically applied based on the `api_type` or `adapter` specified in the configuration.

### Loading Custom Datasets

For custom datasets not in the predefined registry, use `Dataset.load_from_file()`:

```python
from inference_endpoint.dataset_manager import Dataset
from inference_endpoint.dataset_manager.dataset import DatasetFormat
from inference_endpoint.dataset_manager.transforms import UserPromptFormatter

# Load with transforms
dataset = Dataset.load_from_file(
    file_path="path/to/dataset.parquet",
    format=DatasetFormat.PARQUET,
    transforms=[
        UserPromptFormatter(
            user_prompt_format="Question: {question}\nAnswer:",
            output_column="prompt"
        )
    ]
)
```

**Supported Formats:**

- `.csv` - CSV files with headers
- `.parquet` - Apache Parquet files
- `.json` - JSON files
- `.jsonl` - JSON Lines files
- `huggingface` - HuggingFace datasets

### Creating New Datasets

To create a new predefined dataset:

1. **Create a subclass of `Dataset`** with a unique `dataset_id`:

```python
from inference_endpoint.dataset_manager import Dataset
from pathlib import Path
import pandas as pd

class MyDataset(Dataset, dataset_id="my_dataset"):
    COLUMN_NAMES = ["question", "answer"]  # Critical keys
    PRESETS = presets  # Import from presets.py

    @classmethod
    def generate(
        cls,
        datasets_dir: Path,
        seed: int = 0,
        max_samples: int | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        # Load and process the dataset
        # Return a DataFrame with the required columns
        pass
```

2. **Create model presets** in `presets.py`:

```python
# my_dataset/presets.py
from inference_endpoint.dataset_manager.transforms import (
    Transform,
    UserPromptFormatter,
)

def gptoss() -> list[Transform]:
    return [
        UserPromptFormatter(
            user_prompt_format="{question}\nPlease reason step by step.",
        ),
    ]

def llama2() -> list[Transform]:
    return [
        UserPromptFormatter(
            user_prompt_format="<s>[INST] {question} [/INST]",
        ),
    ]
```

3. **Register in `__init__.py`**:

```python
# my_dataset/__init__.py
from ...dataset import Dataset, load_from_huggingface
from . import presets

class MyDataset(Dataset, dataset_id="my_dataset"):
    # ... implementation
```

### Creating Model Presets

Model presets define dataset + model specific transforms. Each preset function should:

1. Return a list of transforms
2. Generate standardized columns expected by adapters (typically `prompt`)
3. Be named descriptively (e.g., `gptoss()`, `llama2()`, `claude()`)

```python
def my_model() -> list[Transform]:
    return [
        UserPromptFormatter(
            user_prompt_format="Custom format: {question}",
            output_column="prompt"
        ),
        # Additional model-specific transforms...
    ]
```

## Transform Types

### Base Classes

- **`Transform`**: Abstract base class for all transforms. Operates on entire DataFrames.
- **`RowProcessor`**: Special transform that processes each row individually. Automatically handles iteration.

### Built-in Transforms

#### `UserPromptFormatter`

Formats user prompts from DataFrame columns using a format string.

```python
UserPromptFormatter(
    user_prompt_format="{question}\nAnswer: {choices}",
    output_column="prompt"
)
```

#### `Harmonize`

Converts prompts to OpenAI Harmony-compatible tokenized format.

```python
Harmonize(
    tokenizer_name="openai/gpt-oss-120b",
    encoding_name="HARMONY_GPT_OSS",
    prompt_column="prompt",
    tokenized_column="input_tokens"
)
```

#### `AddStaticColumns`

Adds columns with constant values (useful for API metadata).

```python
AddStaticColumns({"max_tokens": 1024, "temperature": 0.0})
```

#### `ColumnFilter`

Keeps only specified columns (allow-list).

```python
ColumnFilter(
    required_columns=["input_tokens"],
    optional_columns=["system"]
)
```

#### `ColumnRemap`

Remaps column names with support for fuzzy matching.

```python
ColumnRemap(
    remap={
        ("user_prompt", "question", "input"): "prompt",
        "answer": "ground_truth"
    }
)
```

#### `MakeAdapterCompatible`

Special transform that searches for common prompt column names and standardizes them.

```python
MakeAdapterCompatible()  # Automatically remaps to "prompt" and "system"
```

### Performance Optimization

The `apply_transforms()` function automatically **fuses consecutive `RowProcessor` transforms** into a single pass over the DataFrame to minimize iteration overhead.

```python
transforms = [
    UserPromptFormatter(...),      # RowProcessor
    AddColumnFromRow(...),          # RowProcessor
    # ↑ These two are fused into one pass
    AddStaticColumns(...),          # Regular Transform - breaks fusion
    ColumnFilter(...),              # Regular Transform
]
```

## Examples

### Example 1: Using AIME25 with GPT-OSS

```yaml
# In benchmark config
datasets:
  - name: "aime25::gptoss"
    type: "accuracy"
    accuracy_config:
      eval_method: "pass_at_1"
      ground_truth: "answer"
      extractor: "boxed_math_extractor"
```

This applies:

1. **Model preset** (`gptoss`): Formats question with reasoning instruction
2. **Adapter transforms** (from SGLang adapter): Harmonizes prompt, adds metadata, filters columns

### Example 2: Custom Dataset with Transforms

```python
from inference_endpoint.dataset_manager import Dataset
from inference_endpoint.dataset_manager.dataset import DatasetFormat
from inference_endpoint.dataset_manager.transforms import (
    UserPromptFormatter,
    AddStaticColumns,
    ColumnFilter,
)

# Load custom dataset
dataset = Dataset.load_from_file(
    file_path="my_questions.jsonl",
    format=DatasetFormat.JSONL,
    transforms=[
        UserPromptFormatter(
            user_prompt_format="Q: {question}\nA:",
            output_column="prompt"
        ),
        AddStaticColumns({"max_tokens": 512}),
        ColumnFilter(required_columns=["prompt", "answer"])
    ]
)

# Load and prepare for inference
dataset.load(api_type=APIType.SGLANG)
```

### Example 3: Creating a New Transform

```python
from inference_endpoint.dataset_manager.transforms import RowProcessor

class ExtractYear(RowProcessor):
    """Extract year from a date string."""

    def __init__(self, date_column: str, output_column: str = "year"):
        self.date_column = date_column
        self.output_column = output_column

    def process_row(self, row: dict[str, Any]) -> dict[str, Any]:
        date_str = row[self.date_column]
        row[self.output_column] = int(date_str.split("-")[0])
        return row
```

## Design Rationale

The separation of datasets and transforms provides several benefits:

1. **No Dataset Explosion:** Instead of creating `AIME25_GPT_OSS`, `AIME25_Llama2`, `AIME25_Claude`, etc., we have one `AIME25` dataset with multiple presets.

2. **Composability:** Transforms can be mixed and matched. Adapter transforms work with any dataset that provides the required columns.

3. **Maintainability:** Adding a new model requires updating only the preset transforms, not creating new dataset classes.

4. **Clarity:** Users see both the base dataset (`aime25`) and the transformations (`gptoss`) in the config, making it clear what data is being used and how it's being prepared.

5. **Reversibility:** Since transforms are reversible, the original dataset data is preserved conceptually. Transforms only change the format, not the fundamental content.

## Directory Structure

```
dataset_manager/
├── __init__.py              # Module exports
├── README.md                # This file
├── dataset.py               # Dataset base classes and loaders
├── factory.py               # Dataset loader factory
├── transforms.py            # Transform classes
└── predefined/              # Predefined datasets
    ├── aime25/
    │   ├── __init__.py      # AIME25 dataset class
    │   └── presets.py       # Model-specific presets
    ├── gpqa/
    │   ├── __init__.py
    │   └── presets.py
    ├── livecodebench/
    │   ├── __init__.py
    │   ├── presets.py
    │   └── lcb_serve.py
    └── ...
```

## Contributing

When contributing new datasets or transforms:

1. **New Datasets:** Follow the pattern in `predefined/aime25/`

   - Create a new directory under `predefined/`
   - Implement the `Dataset` subclass with `generate()` method
   - Define `COLUMN_NAMES` (critical keys)
   - Create `presets.py` with model-specific transforms

2. **New Transforms:** Add to `transforms.py`

   - Inherit from `Transform` or `RowProcessor`
   - Implement clear docstrings
   - Consider reversibility principle

3. **New Model Presets:** Add to the relevant `presets.py`
   - Return a list of transforms
   - Ensure output includes expected columns for adapters (e.g., `prompt`)
