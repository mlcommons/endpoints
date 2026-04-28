# Running Endpoints with Llama Models

This example covers benchmarking two Llama models:

- **[Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)** — CNN/DailyMail summarization, offline and online modes
- **[Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)** — Open Orca, online (Poisson) mode

---

## Llama-3.1-8B-Instruct

### Dataset

The Llama3.1-8B benchmark uses the [cnn/dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset. When using the provided config files, the validation split is downloaded automatically by specifying the dataset name as `- name: cnn_dailymail::llama3_8b`.

For post-training quantization calibration, use the [cnn-dailymail-calibration-list](https://github.com/mlcommons/inference/blob/v4.0/calibration/CNNDailyMail/calibration-list.txt):

```bash
curl -OL https://raw.githubusercontent.com/mlcommons/inference/v4.0/calibration/CNNDailyMail/calibration-list.txt
uv run python download_cnndm.py --save-dir data --calibration-ids-file calibration-list.txt --split train
```

### Environment

```bash
export HF_TOKEN=<your Hugging Face token>
export HF_HOME=<path to your hf_home, e.g. ~/.cache/huggingface>
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
hf download $MODEL_NAME
```

### Launch the server

**Note:** To generate outputs matching MLPerf submissions from legacy loadgen, apply a custom chat template (handled automatically by the `cnn_dailymail::llama3_8b` preset). The `--trust-request-chat-template` flag is required. **Security warning:** this flag allows execution of request-provided chat templates and should only be used in trusted environments. Do not enable it on publicly exposed endpoints.

```bash
docker run --runtime nvidia --gpus all \
    -v ${HF_HOME}:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest --model ${MODEL_NAME} --trust-request-chat-template
```

### Offline mode

```bash
uv run inference-endpoint benchmark from-config -c offline_llama3_8b_cnn.yaml --timeout 600
```

### Online mode

```bash
uv run inference-endpoint benchmark from-config -c online_llama3_8b_cnn.yaml --timeout 600
```

---

## Llama-2-70b-chat-hf

### Dataset

Download the preprocessed Open Orca dataset from the MLCommons R2 bucket. Navigate to your desired download directory and run:

```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    https://inference.mlcommons-storage.org/metadata/llama-2-70b-open-orca-dataset.uri
```

The dataset will be downloaded automatically to `./open_orca`. Additional instructions for downloading the model and dataset are in the [Reference Implementation for llama2-70b](https://github.com/mlcommons/inference/tree/master/language/llama2-70b).

### Environment

Go to [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) and request access, then create a HuggingFace access token with read permissions.

```bash
export MODEL_NAME=meta-llama/Llama-2-70b-chat-hf
export HF_TOKEN=<your Hugging Face token>
export HF_HOME=<path to your hf_home, e.g. ~/.cache/huggingface>
hf download $MODEL_NAME
```

### Accuracy evaluation setup (optional)

Accuracy evaluation requires additional packages. Skip this for performance-only runs.

```bash
uv pip install nltk evaluate rouge_score
uv run python -c 'import nltk; nltk.download("punkt"); nltk.download("punkt_tab")'
```

### Launch the server

```bash
docker run --runtime nvidia --gpus all \
    -v ${HF_HOME}:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest --model ${MODEL_NAME} --gpu-memory-utilization 0.95
```

### Online mode

```bash
uv run inference-endpoint benchmark from-config -c online_llama2_70b_orca.yaml --timeout 600
```
