# Dataset Manager — Design Spec

> Loads benchmark datasets from local files and HuggingFace sources and applies ordered transform pipelines to produce request-ready samples for the load generator.

**Component specs:** [async_utils](../async_utils/Design.md) · [commands](../commands/Design.md) · [config](../config/Design.md) · [core](../core/Design.md) · **dataset_manager** · [endpoint_client](../endpoint_client/Design.md) · [evaluation](../evaluation/Design.md) · [load_generator](../load_generator/Design.md) · [metrics](../metrics/Design.md) · [openai](../openai/Design.md) · [plugins](../plugins/Design.md) · [profiling](../profiling/Design.md) · [sglang](../sglang/Design.md) · [testing](../testing/Design.md) · [utils](../utils/Design.md)

---

## Overview

`dataset_manager/` loads benchmark datasets from various sources and applies transformation
pipelines to produce request-ready samples. It decouples dataset format (how data is stored)
from model and adapter requirements (how data must be shaped).

## Responsibilities

- Load samples from JSONL, JSON, CSV, Parquet, and HuggingFace sources
- Apply ordered transform pipelines to adapt raw rows to API format
- Provide a uniform `Dataset` interface regardless of source or format
- Register built-in (predefined) datasets by name for ruleset use

## Component Map

```
DataLoaderFactory
      |
      +-- format -> DatafileLoader subclass
      |                      (jsonl / json / csv / parquet / hf)
      |                  |
      |                  v
      |           raw DataFrame
      |                  |
      +-- transforms -> Transform pipeline
                             |
                             v
                        Dataset  (load_sample / num_samples)
```

## Public Interface

### `Dataset`

Concrete base class. Subclasses register themselves in `Dataset.PREDEFINED` via
`__init_subclass__`.

```python
class Dataset:
    PREDEFINED: ClassVar[dict[str, type["Dataset"]]]  # name → subclass registry

    def load_sample(self, index: int) -> Any: ...
    def num_samples(self) -> int: ...

    repeats: int = 1
    # When repeats > 1, the dataset wraps around after num_samples()
```

`load_sample()` typically returns a `dict`, but the return type is `Any` — dataset schemas vary
widely and are not enforced at the base class level.

### `DataLoaderFactory`

```python
class DataLoaderFactory:
    @staticmethod
    def create_loader(config: DatasetConfig, num_repeats: int = 1, **kwargs) -> Dataset
```

`config` is the `Dataset` Pydantic model from `config/schema.py`; it carries path, format,
parser/remap config, and dataset name. Format is inferred from file extension when
`config.format` is not set:

- `.jsonl` → `JSONL`
- `.json` → `JSON`
- `.csv` → `CSV`
- `.parquet` → `PARQUET`
- explicit `format=huggingface` → `HF`

Presets (e.g. `"gpqa::Qwen/Qwen3-8B"`) are encoded in `config.name` as a `"::"` split — the
factory resolves them to a predefined dataset class with a model-specific transform stack.

### `Transform` (abstract base)

```python
class Transform(ABC):
    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

Transforms are composed in order; each receives the output of the previous.

## Built-in Transforms

| Transform               | Purpose                                                |
| ----------------------- | ------------------------------------------------------ |
| `ColumnRemap`           | Rename columns (e.g. `question` -> `prompt`)           |
| `UserPromptFormatter`   | Apply format string to produce the `prompt` column     |
| `MakeAdapterCompatible` | Ensure columns match what `HttpRequestAdapter` expects |

## Predefined Datasets

Registered in `dataset.py` under `Dataset.PREDEFINED`. Referenced by name in rulesets and YAML
configs. Each predefined dataset ships with default transforms for supported model families.

| Name                        | Source        | Notes                                      |
| --------------------------- | ------------- | ------------------------------------------ |
| `aime25`                    | AIME 2025     | Math reasoning                             |
| `gpqa`                      | GPQA Diamond  | Science QA                                 |
| `cnndailymail`              | CNN/DailyMail | Summarization                              |
| `open_orca`                 | OpenOrca      | General instruction                        |
| `livecodebench`             | LiveCodeBench | Code generation; requires additional setup |
| `shopify_product_catalogue` | Shopify       | E-commerce Q&A (q3vl)                      |
| `random`                    | Synthetic     | Generated prompts for throughput testing   |

## Preset System

A preset string like `"gpqa::Qwen/Qwen3-8B"` resolves to a predefined dataset with a
model-specific transform stack pre-applied. This is used by rulesets to ensure consistent
prompt formatting across submissions.

## Design Decisions

**Transforms are separate from datasets**

The same raw dataset can be used with different models (each with different prompt templates) or
different API adapters (OpenAI vs SGLang). Keeping transforms out of the dataset class means
neither the dataset nor the adapter has to know about the other.

**Format inference from extension**

Reducing friction for CLI users is a priority. Specifying `--dataset my_data.jsonl` should just
work. For non-standard sources such as HuggingFace datasets, callers can set the dataset
`format` explicitly in YAML or in the repeatable `--dataset ...,format=huggingface` string.

**`load_sample()` returns a dict, not a typed struct**

Dataset schemas vary widely (different columns, optional fields). A dict interface avoids a
proliferation of dataset-specific types while still being easily introspectable and debuggable.
The adapter layer (`openai/openai_adapter.py`) is responsible for reading the expected keys.

**`repeats` for issuing more samples than the dataset size**

When `n_samples_to_issue > num_samples()`, the dataset wraps. Index arithmetic (`index %
num_samples()`) is handled by the Dataset base class. This avoids duplicating the logic in every
scheduler.

## Integration Points

| Consumer                           | Usage                                                         |
| ---------------------------------- | ------------------------------------------------------------- |
| `load_generator/load_generator.py` | Calls `load_sample(index)` for each scheduled query           |
| `config/rulesets/mlcommons/`       | References predefined datasets by name                        |
| `commands/benchmark/`              | Constructs dataset via `DataLoaderFactory` from CLI/YAML args |
