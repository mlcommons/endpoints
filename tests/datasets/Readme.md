This directory contains datasets used primarily for testing.

## `ds_samples.jsonl`

Extracted from the deep-seek preprocessed dataset by taking a sample from each of the sources `{math500, aime1983, livecodebench, gpqa, mmlu_pro}`.

The dataset has the following columns:

```
['dataset', 'ground_truth', 'ref_accuracy', 'ref_extracted_answer',
       'ref_output', 'text_input', 'metric', 'question', 'tok_ref_output',
       'tok_ref_output_len', 'tok_input', 'tok_input_len',
       'templated_text_input']
```

## `squad_pruned`

A pruned version of the [SQuAD v1.1](https://github.com/rajpurkar/SQuAD-explorer/raw/master/dataset/dev-v1.1.json) dataset.
The Huggingface squad dataset (used here) has the following columns : `'id', 'title', 'context', 'question', 'answers'`
Note that the HF version is a flattened version of the original squad dataset - each question/answer is a flattened version of `(data['data'][0]['paragraphs'][0])['qas'][0].keys()`
The pruned version holds 50 samples from each slice (training and validation)

## `dummy_1k.jsonl`

**Purpose:** Dummy dataset for local CLI testing (NOT real benchmark data).

**Samples:** 1000

**Format:** JSONL (one JSON object per line with columns: `text_input`, `ref_output`)

**Content:**

- 10 prompt templates (stories, explanations, poems, descriptions, etc.)
- 10 topics (AI, quantum computing, renewable energy, space, biotech, etc.)
- Rotated to create 1000 unique prompts with case numbers for variation

**Generation:**

```bash
uv run python scripts/create_dummy_dataset.py
```

**Example Prompts:**

```
Write a short story about artificial intelligence (case 0)
Explain the concept of quantum computing (case 1) in simple terms
Create a poem about renewable energy (case 2)
Describe space exploration (case 3) in detail
```

**Use Cases:**

- Testing CLI commands locally with echo server
- Quick smoke tests before production deployment
- Validating configuration changes
- Development and debugging without large datasets

**Usage:**

```bash
# Test offline benchmark
uv run inference-endpoint benchmark offline \
  --endpoints http://localhost:8765 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.jsonl

# Test probe
uv run inference-endpoint probe \
  --endpoints http://localhost:8765 \
  --model Qwen/Qwen3-8B \
  --requests 10
```

See `docs/LOCAL_TESTING.md` for complete testing guide.

### Candidates

- CNN / DailyMail v3.0.0
- OpenOrca, GSM8K, MBXP
