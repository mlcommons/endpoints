# Evaluator Unit Tests

This directory contains comprehensive unit tests for the evaluator system using fake/mock data.

## Test Files

### `test_scorers.py`

Tests for concrete Scorer implementations:

- **`TestPassAt1Scorer`**: Tests for exact-match scoring

  - Exact matches return 1
  - Mismatches return 0
  - Case sensitivity
  - Whitespace handling
  - Edge cases (empty strings, special characters, unicode)

- **`TestRougeScorer`**: Tests for ROUGE multi-metric scoring
  - Returns dictionary with rouge1, rouge2, rougeL
  - Perfect matches score 1.0
  - No overlap scores 0.0
  - Partial overlap produces intermediate scores
  - Stemmer effectiveness
  - Case insensitivity
  - Punctuation handling
  - Parametrized test cases

**Total Test Cases**: ~35 tests

### `test_extractors.py`

Tests for Extractor implementations:

- **`TestABCDExtractor`**: Tests for ABCD answer extraction
  - Simple letter extraction (A, B, C, D)
  - Various formats: "Answer: A", "**Answer:** B", "(C)", "[D]"
  - LaTeX formats: `\boxed{A}`, `\textbf{B}`
  - Markdown formatting
  - Case insensitivity
  - Multi-sentence responses
  - No valid answer returns None
  - Real-world GPT-style responses
  - Parametrized extraction cases

**Total Test Cases**: ~20 tests

### `test_mcq_evaluator.py`

Tests for MCQ-specific components:

- **`TestNormalizeChoice`**: Tests for answer normalization

  - Letter to choiceN format conversion
  - Whitespace handling
  - Formatting removal (parentheses, brackets, periods)
  - Custom options support
  - Invalid input handling

- **`TestMCQLetterChoiceScorer`**: Tests for MCQ scorer
  - Correct/incorrect answer scoring
  - Format handling
  - Inheritance from PassAt1Scorer
  - Batch scoring scenarios
  - Realistic MCQ evaluation scenarios
  - Edge cases (all same answer, mixed formatting)

**Total Test Cases**: ~25 tests

### `test_evaluator_integration.py`

Integration tests for the full Evaluator pipeline:

- **`TestEvaluatorWithPassAt1`**: Tests Evaluator with PassAt1Scorer

  - Perfect accuracy (100%)
  - Zero accuracy (0%)
  - Partial accuracy (50%)
  - Empty file handling
  - File not found errors

- **`TestEvaluatorWithRougeScorer`**: Tests Evaluator with RougeScorer

  - Returns dictionary of metrics
  - Perfect and partial matches
  - Aggregation of multiple samples

- **`TestEvaluatorExtractOutputs`**: Tests output extraction

  - Basic extraction from JSONL
  - No match handling (empty string)
  - Row offset support
  - UUID mapping

- **`TestEvaluatorEdgeCases`**: Tests edge cases
  - Large datasets (100+ samples)
  - Mixed empty and valid outputs
  - Error handling

**Total Test Cases**: ~15 tests

## Running the Tests

### Run all evaluator tests:

```bash
pytest tests/unit/accuracy/evaluators/ -v
```

### Run specific test file:

```bash
pytest tests/unit/accuracy/evaluators/test_scorers.py -v
```

### Run specific test class:

```bash
pytest tests/unit/accuracy/evaluators/test_scorers.py::TestPassAt1Scorer -v
```

### Run specific test:

```bash
pytest tests/unit/accuracy/evaluators/test_scorers.py::TestPassAt1Scorer::test_exact_match_returns_one -v
```

### Run with coverage:

```bash
pytest tests/unit/accuracy/evaluators/ --cov=inference_endpoint.accuracy.evaluators --cov-report=html
```

## Test Coverage

The test suite covers:

- ✅ All Scorer implementations (PassAt1Scorer, RougeScorer)
- ✅ All Extractor implementations (ABCDExtractor)
- ✅ MCQ-specific components (normalize_choice, MCQLetterChoiceScorer)
- ✅ Full Evaluator pipeline (extraction + scoring + aggregation)
- ✅ Edge cases and error conditions
- ✅ Integration between components
- ✅ File I/O operations (using temporary files)
- ✅ Both single-metric and multi-metric scorers

**Total Test Cases**: ~95 comprehensive tests

## Test Data

All tests use **fake/mock data** - no real datasets or model outputs are required. This includes:

- Mock CSV files created with `tempfile`
- Mock JSONL output files
- Mock datasets with predefined ground truths
- Synthetic text for ROUGE scoring
- Fake model responses for extraction testing

## Design Principles

1. **Isolation**: Each test is independent and uses temporary files
2. **Clarity**: Clear test names describing what is being tested
3. **Coverage**: Both happy paths and edge cases
4. **Fixtures**: Reusable setup in `setup_method()`
5. **Parametrization**: Use `@pytest.mark.parametrize` for multiple similar cases
6. **Mocking**: Use mocks for external dependencies (datasets, file I/O)
7. **Cleanup**: Automatic cleanup of temporary files using try/finally

## Adding New Tests

When adding new scorer or evaluator implementations:

1. Create tests in the appropriate file
2. Follow the existing naming conventions
3. Test both happy paths and edge cases
4. Use parametrization for similar test cases
5. Use temporary files for file I/O tests
6. Mock external dependencies
7. Ensure cleanup of resources

Example template:

```python
class TestMyNewScorer:
    """Test cases for MyNewScorer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = MyNewScorer()

    def test_basic_functionality(self):
        """Test basic scoring behavior."""
        score = self.scorer.score_sample("output", "ground_truth")
        assert isinstance(score, (int, float, dict))

    def test_edge_case(self):
        """Test edge case behavior."""
        score = self.scorer.score_sample("", "")
        assert score is not None
```

## Dependencies

Tests require:

- `pytest` - Test framework
- `pytest-cov` (optional) - Coverage reporting
- Standard library: `tempfile`, `csv`, `pathlib`
- Project dependencies: `orjson` (for JSONL parsing)
- `rouge-score` - For RougeScorer tests

Install test dependencies:

```bash
pip install -r requirements/test.txt  # If available
# or
pip install pytest pytest-cov
```
