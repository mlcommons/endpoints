# Evaluator and Scorer Architecture

## Overview

The evaluation system is designed with a clean separation of concerns between **Extractors**, **Scorers**, and **Evaluators**:

- **Extractor**: Extracts specific patterns (e.g., ABCD answers) from raw model outputs using regex
- **Scorer**: Compares extracted outputs against ground truth and returns numeric score(s)
- **Evaluator**: Orchestrates the evaluation process, using an Extractor and Scorer to process all samples and aggregate results

## File Structure

```
evaluators/
├── base.py              # Abstract base classes: Extractor, Scorer, Evaluator
├── scorers.py           # Concrete Scorer implementations: PassAt1Scorer, RougeScorer
├── README.md            # This file
└── MCQLetterChoice/
    └── entrypoint.py    # MCQ-specific evaluator and scorer
```

- **base.py**: Contains abstract base classes and common extractors (like `ABCDExtractor`)
- **scorers.py**: Contains concrete implementations of scorers for common metrics
- **[Domain]/** directories: Domain-specific evaluators and scorers (e.g., MCQLetterChoice)

## Architecture

```
Raw Model Output → Extractor → Extracted Answer → Scorer → Score(s) → Evaluator → Aggregated Metric
```

### Scorer Class

The `Scorer` is an abstract base class (defined in `base.py`) responsible for comparing outputs against ground truths. It returns either:

- A single numeric value (e.g., 0 or 1 for pass@1 scoring)
- A dictionary of metric names to numeric values (e.g., `{"rouge1": 0.8, "rouge2": 0.6}`)

#### Built-in Scorers (in `scorers.py`)

1. **PassAt1Scorer**: Implements pass@1 scoring (exact string match)

   - Returns: `1` if exact match, `0` otherwise
   - Use case: Multiple choice questions, exact answer matching

2. **RougeScorer**: Production-ready multi-metric scorer using Google's `rouge-score` library
   - Returns: `{"rouge1": float, "rouge2": float, "rougeL": float}` (F1 scores)
   - Use case: Text generation, summarization tasks
   - Parameters: `use_stemmer` (default: True) for Porter stemmer normalization

### Evaluator Class

The `Evaluator` uses an `Extractor` and `Scorer` to:

1. Extract outputs from model responses
2. Score each extracted output against ground truth
3. Aggregate scores (typically by averaging)

## Usage Examples

### Example 1: Multiple Choice Question Evaluation (Single Metric)

```python
from pathlib import Path
from inference_endpoint.accuracy.evaluators.base import ABCDExtractor, Evaluator
from inference_endpoint.accuracy.evaluators.scorers import PassAt1Scorer
from inference_endpoint.accuracy.datasets.base import AccuracyDataset

# Create evaluator with extractor and scorer
evaluator = Evaluator(
    extractor=ABCDExtractor,
    scorer=PassAt1Scorer(),
    accuracy_dataset=my_dataset,
    dataset_dir=Path("data/gpqa")
)

# Extract outputs from model responses
evaluator.extract_outputs(
    outputs_file=Path("outputs.jsonl"),
    save_to=Path("extracted.csv"),
    uuid_map=uuid_mapping,
    num_samples=100
)

# Calculate aggregated score
score = evaluator.calculate_score(Path("extracted.csv"))
print(f"Accuracy: {score:.2%}")  # e.g., "Accuracy: 85.50%"
```

### Example 2: Custom Scorer (Single Metric)

```python
from inference_endpoint.accuracy.evaluators.base import ABCDExtractor, Evaluator, Scorer

class CaseInsensitiveScorer(Scorer):
    """Scorer that performs case-insensitive matching."""

    def score_sample(self, sample_output: str, ground_truth: str) -> int:
        return 1 if sample_output.lower() == ground_truth.lower() else 0

# Use in evaluator
evaluator = Evaluator(
    extractor=ABCDExtractor,
    scorer=CaseInsensitiveScorer(),
    accuracy_dataset=my_dataset,
    dataset_dir=Path("data/")
)
```

### Example 3: Multi-Metric Scorer

```python
from pathlib import Path
from inference_endpoint.accuracy.evaluators.base import Evaluator
from inference_endpoint.accuracy.evaluators.scorers import RougeScorer

# Use ROUGE scorer for text generation tasks
# RougeScorer uses Google's rouge-score library for accurate computation
evaluator = Evaluator(
    extractor=my_text_extractor,
    scorer=RougeScorer(use_stemmer=True),  # use_stemmer=True for better matching
    accuracy_dataset=summarization_dataset,
    dataset_dir=Path("data/summaries")
)

# Calculate scores
scores = evaluator.calculate_score(Path("extracted.csv"))
# Returns: {"rouge1": 0.82, "rouge2": 0.65, "rougeL": 0.78}

print(f"ROUGE-1 F1: {scores['rouge1']:.2%}")
print(f"ROUGE-2 F1: {scores['rouge2']:.2%}")
print(f"ROUGE-L F1: {scores['rougeL']:.2%}")
```

### Example 4: Code Execution Scorer

```python
import subprocess
from pathlib import Path
from inference_endpoint.accuracy.evaluators.base import Scorer

class CodeExecutionScorer(Scorer):
    """Scorer that executes generated code and checks if tests pass."""

    def score_sample(self, sample_output: str, ground_truth: str) -> int:
        """
        Execute code and run test suite.

        Args:
            sample_output: Generated code
            ground_truth: Path to test file or test script
        """
        # Write code to temporary file
        with open("temp_code.py", "w") as f:
            f.write(sample_output)

        try:
            # Run tests in container or subprocess
            result = subprocess.run(
                ["python", "temp_code.py", "--test", ground_truth],
                capture_output=True,
                timeout=5
            )
            # Return 1 if all tests pass, 0 otherwise
            return 1 if result.returncode == 0 else 0
        except (subprocess.TimeoutExpired, Exception):
            return 0
```

## Creating Custom Evaluators

For domain-specific evaluation logic, subclass `Evaluator`:

```python
from inference_endpoint.accuracy.evaluators.base import ABCDExtractor, Evaluator
from inference_endpoint.accuracy.evaluators.MCQLetterChoice.entrypoint import (
    MCQLetterChoiceScorer,
    normalize_choice
)

class CustomMCQEvaluator(Evaluator):
    """Custom evaluator with domain-specific normalization."""

    def __init__(self, *args, **kwargs):
        # Create custom scorer with your logic
        scorer = MCQLetterChoiceScorer()
        super().__init__(
            extractor=ABCDExtractor,
            scorer=scorer,
            *args,
            **kwargs
        )
```

### Extending Existing Scorers

You can also extend existing scorers for domain-specific logic. For example,
`MCQLetterChoiceScorer` extends `PassAt1Scorer` to add normalization:

```python
from inference_endpoint.accuracy.evaluators.scorers import PassAt1Scorer

class MCQLetterChoiceScorer(PassAt1Scorer):
    """Extends PassAt1Scorer with answer normalization."""

    def score_sample(self, sample_output: str, ground_truth: str) -> int:
        # Normalize the extracted answer to 'choiceN' format
        extracted_answer = normalize_choice(sample_output)
        # Use parent's exact match scoring
        return super().score_sample(extracted_answer, ground_truth)
```

This approach promotes code reuse and composition.

## Pass@k Convention

This implementation follows [Artificial Analysis's pass@k convention](https://artificialanalysis.ai/methodology/intelligence-benchmarking), where:

- **pass@1**: The model gets exactly one attempt to produce the correct answer
- Score is 1 if correct on first attempt, 0 otherwise
- Final metric is the average across all samples

For pass@k where k > 1, you would need to modify the scorer to handle multiple attempts per sample.

## Design Principles

1. **Separation of Concerns**: Extract, Score, and Aggregate are separate responsibilities
2. **Extensibility**: Easy to add new scorers for different evaluation metrics
3. **Flexibility**: Support both single-value and multi-value metrics
4. **Type Safety**: Use type hints for better IDE support and error catching
5. **Composability**: Mix and match extractors and scorers as needed

## Migration from Old API

Old API (deprecated):

```python
class MyEvaluator(Evaluator):
    def evaluate_sample(self, sample_output: str, ground_truth: str) -> int:
        return 1 if sample_output == ground_truth else 0
```

New API (create custom scorer in separate file or inline):

```python
from inference_endpoint.accuracy.evaluators.base import Evaluator, Scorer

class MyScorer(Scorer):
    def score_sample(self, sample_output: str, ground_truth: str) -> int:
        return 1 if sample_output == ground_truth else 0

class MyEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            extractor=MyExtractor,
            scorer=MyScorer(),
            *args,
            **kwargs
        )
```

Or use built-in scorers from `scorers.py`:

```python
from inference_endpoint.accuracy.evaluators.base import Evaluator
from inference_endpoint.accuracy.evaluators.scorers import PassAt1Scorer

class MyEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(
            extractor=MyExtractor,
            scorer=PassAt1Scorer(),
            *args,
            **kwargs
        )
```

## Testing

When writing tests for custom scorers:

```python
from inference_endpoint.accuracy.evaluators.base import Scorer

class CaseInsensitiveScorer(Scorer):
    def score_sample(self, sample_output: str, ground_truth: str) -> int:
        return 1 if sample_output.lower() == ground_truth.lower() else 0

def test_custom_scorer():
    scorer = CaseInsensitiveScorer()

    # Test exact match
    assert scorer.score_sample("A", "A") == 1

    # Test case insensitivity
    assert scorer.score_sample("a", "A") == 1

    # Test mismatch
    assert scorer.score_sample("A", "B") == 0
```

When testing built-in scorers:

```python
from inference_endpoint.accuracy.evaluators.scorers import PassAt1Scorer, RougeScorer

def test_pass_at_1_scorer():
    scorer = PassAt1Scorer()
    assert scorer.score_sample("correct", "correct") == 1
    assert scorer.score_sample("wrong", "correct") == 0

def test_rouge_scorer():
    # Test with stemmer enabled (default)
    scorer = RougeScorer(use_stemmer=True)
    scores = scorer.score_sample("the cat sat", "the cat sat on mat")

    # Verify all expected metrics are present
    assert "rouge1" in scores
    assert "rouge2" in scores
    assert "rougeL" in scores

    # Verify scores are valid F1 values
    assert 0.0 <= scores["rouge1"] <= 1.0
    assert 0.0 <= scores["rouge2"] <= 1.0
    assert 0.0 <= scores["rougeL"] <= 1.0

    # Test perfect match
    perfect_scores = scorer.score_sample("identical text", "identical text")
    assert perfect_scores["rouge1"] == 1.0
    assert perfect_scores["rouge2"] == 1.0
    assert perfect_scores["rougeL"] == 1.0
```

When testing evaluators, ensure you test the full pipeline including extraction and aggregation.
