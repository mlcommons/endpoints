# Running Endpoints with [Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

It is recommended to use a config file such as [online_llama3_8b_cnn.yaml](online_llama3_8b_cnn.yaml) to run the benchmark.

## Download dataset (Only needed if quantizing the model)

The Llama3.1-8B benchmark uses the [cnn/dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset (for summarization). If using a config (such as provided) to run the benchmark, the (validation) dataset is downloaded automatically by specifying dataset name as `- name: cnn_dailymail::llama3_8b # or cnn_dailymail::llama3_8b_sglang` under the `dataset` entry.

For post-training quantization, users can use the [cnn-dailymail-calibration-list](https://github.com/mlcommons/inference/blob/v4.0/calibration/CNNDailyMail/calibration-list.txt) to select samples for the calibration.

```
curl -OL https://raw.githubusercontent.com/mlcommons/inference/v4.0/calibration/CNNDailyMail/calibration-list.txt
python download_cnndm.py --save-dir data --calibration-ids-file calibration-list.txt --split train
```

## Launch the server

We provide instructions below for using either vLLM or SGLang endpoints.

The following environment variables are used by the commands below to make the scripts easier to run

```
export HF_TOKEN=<your Hugging Face token>
export HF_HOME=<Path to your hf_home, usually /USERNAME/.cache/huggingface>
export MODEL_NAME=<model to run, for instance meta-llama/Llama-3.1-8B-Instruct>
```

It is convenient to download the model prior to launch so that the container can reuse the model instead of having to download it post-launch. This can be done via `hf download $MODEL_NAME`. The models downloaded can be verified via `hf cache scan`

### [vLLM](https://github.com/vllm-project/vllm) (Using NVIDIA GPUs for demo)

**Note**: To generate same outputs as the ones produced from submissions with legacy loadgen, we need to apply a custom chat template (this is taken care of automatically by the cnn-dailymail dataset preset). The flag `--trust-request-chat-template` is also required for this behavior. **Security warning:** `--trust-request-chat-template` allows execution of request-provided chat templates and should only be used in trusted environments or when all requests are controlled by the benchmark harness/preset. Do not enable this flag on publicly exposed endpoints receiving untrusted traffic.

We can launch the latest docker image for vllm using the command below:

```
docker run --runtime nvidia --gpus all -v ${HF_HOME}:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model ${MODEL_NAME} --trust-request-chat-template
```

### [SGLang](https://github.com/sgl-project/sglang)

- First build the container and start the endpoint

```
# Clone the SGLang repository
SGLANG_VER=3f9fc8b848365a5797a44856854e3e6f00a60dd0 # Latest tested
git clone https://github.com/sgl-project/sglang.git
cd sglang/docker && git checkout $SGLANG_VER

# Build the docker image
docker build -t sglang-cpu:latest -f xeon.Dockerfile .

# Initiate a docker container
docker run -it --privileged --ipc=host --network=host -v /dev/shm:/dev/shm -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 -e "HF_TOKEN=<secret>" --name sglang-cpu-server sglang-cpu:latest /bin/bash

# Start sglang endpoint
docker exec -u root -w /workspace sglang-cpu-server /bin/bash -lc python3 -m sglang.launch_server \
    --model-path $MODEL_NAME \
    --served-model-name meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --device cpu \
    --max-running-requests 64 \
    --max-total-tokens 131072 \
    --chunked-prefill-size 8192 \
    --max-prefill-tokens 32768 \
    --mem-fraction-static 0.9 \
    --disable-piecewise-cuda-graph \
    --disable-radix-cache \
    --host 127.0.0.1 \
    --port 8080 2>&1 | tee server.log"
```

## Start benchmark

Make sure the [`inference-endpoint`](https://github.com/mlcommons/endpoints/tree/main?tab=readme-ov-file#installation) is installed and activated

**Note** Double-check the config file for correct parameters such as the model name in the config

- Launch the benchmark with config yaml (For performance only, remove the accuracy dataset entry in the `online_llama3_8b_cnn.yaml`)

### vLLM endpoint targets

- To run Offline mode

```
inference-endpoint benchmark from-config -c offline_llama3_8b_cnn.yaml
```

- To run Online mode

```
inference-endpoint benchmark from-config -c online_llama3_8b_cnn.yaml
```

### SGLang endpoint targets

- To run the offline benchmark:

```
inference-endpoint benchmark from-config -c offline_llama3_8b_cnn_sglang_api.yaml
```
