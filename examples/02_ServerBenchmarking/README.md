# Benchmarking a HF model via vLLM or SgLang

This document describes how we can benchmark an inference server using the inference endpoints.

## Model

We are going to use [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) from Huggingface to demonstrate how to benchmark vLLM and SgLang via inference endpoints.

## Launch the server

The following environment variables are used by the commands below to make the scripts easier to run

```
export HF_TOKEN=<your huggingface token>
export HF_HOME=<Path to your hf_home, usually /USERNAME/.cache/huggingface>
export MODEL_NAME=<model to run, for instance meta-llama/Llama-3.1-8B-Instruct>
```

It is convenient to download the model prior to launch so that the container can reuse the model instead of having to download it post-launch. This can be done via `hf download $MODEL_NAME`. The models downloaded can be verified via `hf cache scan`

### [vLLM](https://github.com/vllm-project/vllm)

We can launch the latest docker image for vllm using the command below:

```
docker run --runtime nvidia --gpus all -v ${HF_HOME}:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" -p 8000:8000 --ipc=host vllm/vllm-openai:latest  --model ${MODEL_NAME}

```

### [SgLang](https://github.com/sgl-project/sglang)

For SgLang, we use a similar docker command:

```
docker run --gpus all --shm-size 32g --net host -v ${HF_HOME}:/root/.cache/huggingface --env HF_TOKEN=${HF_TOKEN} --ipc=host lmsysorg/sglang:latest python3 -m sglang.launch_server --model-path ${MODEL_NAME} --host 0.0.0.0 --port 8000 --tp-size 1 --enable-metrics
```

### [Enroot](https://github.com/NVIDIA/enroot):

On some platforms, docker is replaced by enroot to provide virtualization. The following steps describe how to launch vLLM using enroot - SgLang instructions are similar:

```
enroot import docker://vllm/vllm-openai:latest
enroot start -e HF_TOKEN=$HF_TOKEN -m $HF_HOME:/root/.cache/huggingface vllm+vllm-openai+latest.sqsh  --model ${MODEL_NAME}$
```

## Launching the client

Once the server is up and running, we can send requests to the endpoint by passing in the endpoint address via `-e` as well as the model name

```
inference-endpoint benchmark offline -e http://localhost:8000 -d tests/datasets/dummy_1k.pkl  --model ${MODEL_NAME}
```

# Using a config file

To run [llama2-70b](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) on a single Nvidia-H200 GPU, we first prepare the environment:

```
export MODEL_NAME=meta-llama/Llama-2-70b-chat-hf
export HF_TOKEN=<your hugging face token>
hf download $MODEL_NAME

```

Launch docker container:

```
docker run --runtime nvidia --gpus all     -v ${HF_HOME}:/root/.cache/huggingface     --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN"     -p 8000:8000     --ipc=host     vllm/vllm-openai:latest --model ${MODEL_NAME} --gpu_memory_utilization 0.95

```

And launch the benchmark using the config file `online_llama2_70b_cnn.yaml`. Note that you will need to export the [cnn/dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset via

```
from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
dataset["train"].to_json("cnn_dailymail_train.json")
```

And then launch the example template.

```
inference-endpoint benchmark from-config -c src/inference_endpoint/config/templates/online_llama2_70b_cnn.yaml --timeout 600

```
