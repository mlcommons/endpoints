# Running Endpoints with Qwen3-VL-235B-A22B on Shopify Product Catalogue

This document describes how to perform MLPerf Q3VL benchmarking using the inference endpoints with [Qwen3-VL-235B-A22B-instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct) model and [Shopify's Product Catalogue dataset](https://huggingface.co/datasets/Shopify/product-catalogue) for multimodal product taxonomy classification. 

## Get Dataset

The Shopify Product Catalogue dataset is loaded from HuggingFace and will be generated automatically on first run. Images are converted to base64 for storage.

```
# Dataset is auto-downloaded from https://huggingface.co/datasets/Shopify/product-catalogue
# No manual download required - DataLoaderFactory handles it
```

## Get Model

Use the public quantized MLPerf checkpoint:

```
export MODEL_NAME=Qwen/Qwen3-VL-235B-A22B-Instruct
export HF_TOKEN=<your Hugging Face token>  # Optional for public model; may help with rate limits
hf download $MODEL_NAME
```

The model is available at [Qwen3-VL-235B-A22B-instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct) — no access request required.

**Note:** The Shopify Product Catalogue includes `ground_truth_category`, `ground_truth_brand`, and `ground_truth_is_secondhand` from the HuggingFace dataset. For accuracy evaluation, use the `shopify_category_f1` scorer which computes hierarchical F1 for category taxonomy (matches [MLCommons Q3VL evaluation](https://github.com/mlcommons/inference/blob/master/multimodal/qwen3-vl/src/mlperf_inf_mm_q3vl/evaluation.py)).

To add accuracy evaluation, include an accuracy dataset alongside the performance dataset:

```yaml
datasets:
  - name: shopify_product_catalogue::q3vl
    type: "performance"
    force: true
  - name: shopify_product_catalogue::q3vl
    type: "accuracy"
    force: true
    accuracy_config:
      eval_method: "shopify_category_f1"
      ground_truth: "ground_truth_category"
      extractor: "identity_extractor" # Required by benchmark; scorer parses JSON internally
      num_repeats: 1
```

## Benchmark Qwen3-VL-235B-A22B using a config file

Prepare the environment:

```
export MODEL_NAME=Qwen/Qwen3-VL-235B-A22B-Instruct
export HF_TOKEN=<your Hugging Face token>  # Optional for public model
export HF_HOME=<path to HuggingFace cache, e.g. ~/.cache/huggingface>
```

Launch the vLLM server (vision model requires appropriate GPU resources):

```
docker run --runtime nvidia --gpus all \
  -p 8000:8000 \
  --ipc=host \
  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  --env "VLLM_HTTP_TIMEOUT_KEEP_ALIVE=3600" \
  --env "VLLM_ENGINE_READY_TIMEOUT_S=3600" \
  -v ${HF_HOME}:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model ${MODEL_NAME} \
  --tensor-parallel-size 4 \
  --max-model-len=32768 \
  --async-scheduling \
  --max-num-seqs 1024 \
  --limit-mm-per-prompt.video 0
```

Run the benchmark:

```
inference-endpoint benchmark from-config -c examples/08_Qwen3-VL-235B-A22B_Example/offline_qwen3_vl_235b_a22b_shopify.yaml --timeout 600
```

This config uses `test_mode: "acc"` for accuracy-only (hierarchical F1). Change to `"both"` for perf+acc or `"perf"` for perf-only.
