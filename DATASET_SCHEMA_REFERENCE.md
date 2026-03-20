# Dataset Preset Schema Reference

Quick reference for the columns and transformations of each preset dataset.

## Preset Summary Table

| Dataset | Preset | Input Columns | Output Columns | Transform Type |
|---------|--------|---------------|----------------|---|
| **CNNDailyMail** | `llama3_8b` | `article` | `prompt`, `chat_template` | UserPromptFormatter + AddStaticColumns |
| **CNNDailyMail** | `llama3_8b_sglang` | `article` | `prompt`, `input_tokens` | UserPromptFormatter + Harmonize |
| **AIME25** | `gptoss` | `question` | `prompt` | UserPromptFormatter |
| **GPQA** | `gptoss` | `question`, `choice1-4` | `prompt` | UserPromptFormatter |
| **LiveCodeBench** | `gptoss` | `question`, `starter_code` | `prompt` | UserPromptFormatter |
| **OpenOrca** | `llama2_70b` | `question`, `system_prompt` | `prompt`, `system` | ColumnRemap |

---

## Detailed Schema Definitions

### CNNDailyMail Dataset

#### Preset: `llama3_8b`

**Input:**
```python
{
    "article": str,        # Full news article text
    "highlights": str,     # Summary/highlights (for eval only)
}
```

**Output:**
```python
{
    "article": str,
    "highlights": str,
    "prompt": str,                    # Formatted prompt with article
    "chat_template": str,             # Custom Jinja template
}
```

**Sample test data:**
```python
pd.DataFrame({
    "article": ["CNN reported today that markets are up."],
    "highlights": ["Markets up"],
})
```

**Transformation:**
- Formats prompt: `"Summarize the following news article in {max_new_tokens} tokens...\n\nArticle:\n{article}\n\nSummary:"`
- Adds custom chat template for Llama 3.1-8b tokenization

---

#### Preset: `llama3_8b_sglang`

**Input:**
```python
{
    "article": str,        # Full news article text
    "highlights": str,     # (optional) for eval
}
```

**Output:**
```python
{
    "article": str,
    "highlights": str,
    "prompt": str,
    "input_tokens": list,  # Tokenized prompt
}
```

**Sample test data:**
```python
pd.DataFrame({
    "article": ["CNN reported today that markets are up."],
    "highlights": ["Markets up"],
})
```

**Transformation:**
- Formats prompt same as `llama3_8b`
- Tokenizes using Harmonize transform (requires `transformers` library)
- Produces `input_tokens` column with token IDs

---

### AIME25 Dataset

#### Preset: `gptoss`

**Input:**
```python
{
    "question": str,       # Math problem statement
    "answer": int or str,  # Correct answer (for eval)
}
```

**Output:**
```python
{
    "question": str,
    "answer": int or str,
    "prompt": str,         # Formatted with boxed answer instruction
}
```

**Sample test data:**
```python
pd.DataFrame({
    "question": ["If x + 2 = 5, what is x?"],
    "answer": [3],
})
```

**Transformation:**
- Formats prompt: `"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}. "`
- Instructs model to format answer in `\boxed{answer}` format

---

### GPQA Dataset

#### Preset: `gptoss`

**Input:**
```python
{
    "question": str,       # Multiple choice question
    "choice1": str,        # Option A
    "choice2": str,        # Option B
    "choice3": str,        # Option C
    "choice4": str,        # Option D
    "correct_choice": str, # "A", "B", "C", or "D" (for eval)
}
```

**Output:**
```python
{
    "question": str,
    "choice1": str,
    "choice2": str,
    "choice3": str,
    "choice4": str,
    "correct_choice": str,
    "prompt": str,         # Formatted with all choices and instructions
}
```

**Sample test data:**
```python
pd.DataFrame({
    "question": ["What is the capital of France?"],
    "choice1": ["Paris"],
    "choice2": ["London"],
    "choice3": ["Berlin"],
    "choice4": ["Madrid"],
    "correct_choice": ["A"],
})
```

**Transformation:**
- Formats prompt:
  ```
  {question}

  (A) {choice1}
  (B) {choice2}
  (C) {choice3}
  (D) {choice4}

  Express your final answer as the corresponding option 'A', 'B', 'C', or 'D'.
  ```

---

### LiveCodeBench Dataset

#### Preset: `gptoss`

**Input:**
```python
{
    "question": str,       # Coding problem description
    "starter_code": str,   # Initial code template
}
```

**Output:**
```python
{
    "question": str,
    "starter_code": str,
    "prompt": str,         # Formatted with code delimiters
}
```

**Sample test data:**
```python
pd.DataFrame({
    "question": ["Write a function that returns the sum of two numbers."],
    "starter_code": ["def add(a, b):\n    pass"],
})
```

**Transformation:**
- Formats prompt with instructions and starter code in ```python``` delimiters:
  ```
  You are a python coding expert that solves problems step-by-step.
  ...
  {question}
  ### Format: ...
  ```python
  {starter_code}
  ```
  ```

---

### OpenOrca Dataset

#### Preset: `llama2_70b`

**Input:**
```python
{
    "question": str,       # User query
    "system_prompt": str,  # System instruction
    "response": str,       # Expected response (for eval)
}
```

**Output:**
```python
{
    "response": str,
    "prompt": str,         # Remapped from "question"
    "system": str,         # Remapped from "system_prompt"
}
```

**Sample test data:**
```python
pd.DataFrame({
    "question": ["What is machine learning?"],
    "system_prompt": ["You are an AI expert."],
    "response": ["Machine learning is..."],
})
```

**Transformation:**
- Column remapping: `question` → `prompt`, `system_prompt` → `system`
- Preserves data, only renames columns

---

## Creating Test Data for New Presets

When adding tests for a new preset, follow this pattern:

1. **Identify input columns** from the preset's `UserPromptFormatter` or other transforms
2. **Create minimal fixture** with required columns:
   ```python
   @pytest.fixture
   def sample_data(self):
       return pd.DataFrame({
           "col1": ["value1"],
           "col2": ["value2"],
       })
   ```
3. **Test instantiation:**
   ```python
   def test_preset_instantiation(self):
       transforms = DatasetClass.PRESETS.preset_name()
       assert transforms is not None
   ```
4. **Test application:**
   ```python
   def test_preset_transforms_apply(self, sample_data):
       transforms = DatasetClass.PRESETS.preset_name()
       result = apply_transforms(sample_data, transforms)
       assert "expected_output_column" in result.columns
   ```
5. **Test output format:**
   ```python
   def test_preset_output_format(self, sample_data):
       transforms = DatasetClass.PRESETS.preset_name()
       result = apply_transforms(sample_data, transforms)
       assert "formatting_characteristic" in result["output_col"]
   ```

---

## Testing Transforms Without Datasets

To test transforms without downloading full datasets, create minimal DataFrames:

```python
import pandas as pd
from inference_endpoint.dataset_manager.transforms import apply_transforms
from inference_endpoint.dataset_manager.predefined.cnndailymail import CNNDailyMail

# Minimal sample data
data = pd.DataFrame({
    "article": ["Short article text"],
    "highlights": ["Summary"],
})

# Get preset
transforms = CNNDailyMail.PRESETS.llama3_8b_sglang()

# Apply
result = apply_transforms(data, transforms)

# Verify
print("Columns:", result.columns.tolist())
print("Prompt length:", len(result["prompt"][0]))
```

---

## Quick Validation Script

To validate all presets quickly without downloading datasets, run the test suite:

```bash
pytest tests/unit/dataset_manager/test_dataset_presets.py -v
```

This runs 20 tests across all 6 presets in <5 seconds with no external dependencies.

---

## Running Tests for Specific Datasets

```bash
# Test CNN/DailyMail only
pytest tests/unit/dataset_manager/test_dataset_presets.py::TestCNNDailyMailPresets -v

# Test AIME25 only
pytest tests/unit/dataset_manager/test_dataset_presets.py::TestAIME25Presets -v

# Test all transform application (not instantiation)
pytest -k "transforms_apply" tests/unit/dataset_manager/test_dataset_presets.py -v
```

---

## Column Name Conventions

### Prompt-like columns (can be fuzzy remapped)
- `prompt` - Main input to model
- `user_prompt` - User's query
- `question` - Question to answer
- `input` - Generic input
- `input_text` - Text input
- `problem` - Problem statement
- `query` - Search/database query

### System instruction columns
- `system` - System prompt
- `system_prompt` - System instruction
- `instruction` - Task instruction

### Output columns (from models)
- `response` - Model response
- `answer` - Answer to question
- `highlights` - Text summary
- `output` - Generic output

See `MakeAdapterCompatible` transform for fuzzy column remapping logic.
