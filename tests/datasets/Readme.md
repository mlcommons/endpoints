This directory contains datasets used primarily for testing.

## `ds_samples.pkl`

Extracted from the deep-seek preprocessed dataset by taking a sample from each of the sources `{math500, aime1983, livecodebench, gpqa, mmlu_pro}`.
It maintains the same pickled format as the source. Code to regenerate:

```
def write_unique_samples(src_file_path: str, dst_file_path: str):
    """"
    Utility function - writes the unique samples to a pickle file.
    """
    with open(src_file_path, 'rb') as file:
        data = pickle.load(file)

    dataset_sources = {'math500', 'aime1983', 'livecodebench', 'gpqa', 'mmlu_pro'}
    samples = pandas.DataFrame(columns=data.columns)
    for dataset_source in dataset_sources:
        filtered = data[data['dataset']==dataset_source]
        samples = pandas.concat([samples ,filtered.iloc[[0]]], ignore_index=True)

    with open(dst_file_path, 'wb') as file:
        pickle.dump(samples, file)
```

The dataset has the following columns:

```
['dataset', 'ground_truth', 'ref_accuracy', 'ref_extracted_answer',
       'ref_output', 'text_input', 'metric', 'question', 'tok_ref_output',
       'tok_ref_output_len', 'tok_input', 'tok_input_len',
       'templated_text_input']
```

## `squad_pickled`

A pruned version of the [SQuAD v1.1](https://github.com/rajpurkar/SQuAD-explorer/raw/master/dataset/dev-v1.1.json) dataset.
The Huggingface squad dataset (used here) has the following columns : `'id', 'title', 'context', 'question', 'answers'`
Note that the HF version is a flattened version of the original squad dataset - each question/answer is a flattened version of `(data['data'][0]['paragraphs'][0])['qas'][0].keys()`
The pruned version holds 50 samples from each slice (training and validation)

## `dummy_1k.pkl`

**Purpose:** Dummy dataset for local CLI testing (NOT real benchmark data).

**Samples:** 1000

**Format:** Pickle (pandas DataFrame with columns: `text_input`, `ref_output`)

**Size:** ~132 KB

**Content:**

- 10 prompt templates (stories, explanations, poems, descriptions, etc.)
- 10 topics (AI, quantum computing, renewable energy, space, biotech, etc.)
- Rotated to create 1000 unique prompts with case numbers for variation

**Generation:**

```bash
python scripts/create_dummy_dataset.py
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
inference-endpoint benchmark offline \
  --endpoint http://localhost:8765 \
  --model Qwen/Qwen3-8B \
  --dataset tests/datasets/dummy_1k.pkl

# Test probe
inference-endpoint probe \
  --endpoint http://localhost:8765 \
  --model Qwen/Qwen3-8B \
  --requests 10
```

See `docs/LOCAL_TESTING.md` for complete testing guide.

## `gpqa_sample.pkl`

**Purpose:** Sample GPQA dataset for testing evaluation pipeline.

**Samples:** 10 (from gpqa_diamond variant)

**Format:** Pickle (pandas DataFrame)

**Columns:** `question`, `text_input`, `ground_truth`, `dataset`, `domain`, `subdomain`

**Content:**
- Graduate-level multiple choice questions (A/B/C/D)
- Questions formatted with shuffled options
- Ground truth as single letter (A/B/C/D)

**Generation:**

```bash
python -m inference_endpoint.eval.dataset_generation.generate_gpqa \
  --output tests/datasets/gpqa_sample.pkl \
  --variant diamond \
  --num-samples 10 \
  --seed 42
```

**Note:** Requires HuggingFace authentication (`huggingface-cli login`) to access the gated GPQA dataset.

**Use Cases:**
- Testing GPQA evaluator
- Testing eval command
- Testing pass@k calculation
- Unit and integration tests

### Candidates

- CNN / DailyMail v3.0.0
- OpenOrca, GSM8K, MBXP
