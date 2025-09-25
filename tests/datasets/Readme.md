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

### Candidates

- CNN / DailyMail v3.0.0
- OpenOrca, GSM8K, MBXP
