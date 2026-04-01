# Running Endpoints with Llama2-70B

This document describes how to perform MLPerf Llama2-70B benchmarking using the inference endpoints. Additional instructions to download the model and dataset are provided in [Reference Implementation for llama2-70b](https://github.com/mlcommons/inference/tree/master/language/llama2-70b)

## (Optional) Get Dataset

You can use the MLCommons R2 Downloader to download the preprocessed dataset from a Cloudflare R2 bucket (more information about the MLC R2 Downloader, including how to run it on Windows, can be found [here](https://inference.mlcommons-storage.org)).

Navigate in the terminal to your desired download directory and run the following command to download the dataset:

```
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://inference.mlcommons-storage.org/metadata/llama-2-70b-open-orca-dataset.uri
```

Dataset will be downloaded automatically to ./open_orca before benchmark if not downloaded previously.

## Get Model

First go to [meta-llama/Llama2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) and make a request, sign in to HuggingFace (if you don't have account, you'll need to create one). Create an access token with read permissions.

Set the following environment variables, then download the model to the local HuggingFace cache. Setting `HF_HOME` before downloading ensures the model lands in the same directory that Docker will mount, so the container reuses the cached weights instead of downloading them at startup:

```
export MODEL_NAME=meta-llama/Llama-2-70b-chat-hf
export HF_TOKEN=<your Hugging Face token>
export HF_HOME=<Path to your hf_home, usually /USERNAME/.cache/huggingface>
hf download $MODEL_NAME
```

# Prepare for accuracy evaluation

Accuracy evaluation for MLPerf Llama2-70B requires additional setup. Before running the benchmark, run the following:

```
python3 -m pip install nltk evaluate rouge_score
python3 -c 'import nltk; nltk.download("punkt"); nltk.download("punkt_tab")'
```

These steps are not needed when doing performance-only runs.

# Benchmark Llama2-70b using a config file

To run [llama2-70b](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) on a single Nvidia-H200 GPU, launch the vLLM Docker container. The `-v ${HF_HOME}:/root/.cache/huggingface` mount makes the locally downloaded model available inside the container:

```
docker run --runtime nvidia --gpus all \
    -v ${HF_HOME}:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest --model ${MODEL_NAME} --gpu_memory_utilization 0.95
```

And launch the benchmark using the config file `online_llama2_70b_orca.yaml`

```
inference-endpoint benchmark from-config -c examples/06_Llama2-70B_Example/online_llama2_70b_orca.yaml --timeout 600
```
