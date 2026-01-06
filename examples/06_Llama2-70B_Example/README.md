# Running Endpoints with Llama2-70B

This document describes how to perform MLPerf Llama2-70B benchmarking using the inference endpoints. Instructions to download the model and dataset are provided in [Reference Implementation for llama2-70b](https://github.com/mlcommons/inference/tree/master/language/llama2-70b)

## Get Model

First go to [meta-llama/Llama2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) and make a request, sign in to HuggingFace (if you don't have account, you'll need to create one). Create an access token with read permissions

```
export MODEL_NAME=meta-llama/Llama-2-70b-chat-hf
export HF_TOKEN=<your Hugging Face token>
hf download $MODEL_NAME
```

# Benchmark Llama2-70b using a config file

To run [llama2-70b](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) on a single Nvidia-H200 GPU, we first prepare the environment:

```
export MODEL_NAME=meta-llama/Llama-2-70b-chat-hf
export HF_TOKEN=<your Hugging Face token>
```

Launch docker container:

```
docker run --runtime nvidia --gpus all     -v ${HF_HOME}:/root/.cache/huggingface     --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN"     -p 8000:8000     --ipc=host     vllm/vllm-openai:latest --model ${MODEL_NAME} --gpu_memory_utilization 0.95
```

And launch the benchmark using the config file `online_llama2_70b_orca.yaml`

```
inference-endpoint benchmark from-config -c examples/06_Llama2-70B_Example/online_llama2_70b_orca.yaml --timeout 600
```
