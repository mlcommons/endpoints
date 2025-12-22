# Running Endpoints with Llama2-70B

This document describes how to perform MLPerf Llama2-70B benchmarking using the inference endpoints. Instructions to download the model and dataset are provided below, but users are welcome to visit [Reference Implementation for llama2-70b](https://github.com/mlcommons/inference/tree/master/language/llama2-70b) for more methods to download the model and dataset.

## Get Model

First go to [meta-llama/Llama2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) and make a request, sign in to HuggingFace (if you don't have account, you'll need to create one). Create an access token with read permissions

```
export MODEL_NAME=meta-llama/Llama-2-70b-chat-hf
export HF_TOKEN=<your Hugging Face token>
hf download $MODEL_NAME
```

## Get Dataset

You can use Rclone to download the preprocessed dataset from a Cloudflare R2 bucket.

To run Rclone on Windows, you can download the executable [here](https://rclone.org/install/#windows).
To install Rclone on Linux/macOS/BSD systems, run:

```
sudo -v ; curl https://rclone.org/install.sh | sudo bash
```

Once Rclone is installed, run the following command to authenticate with the bucket:

```
rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
```

You can then navigate in the terminal to your desired download directory and run the following command to download the dataset:

```
rclone copy mlc-inference:mlcommons-inference-wg-public/open_orca ./open_orca -P
# Unzip the dataset files
sudo apt install -y gzip
cd ./open_orca && gzip -d *.gz && cd ..
```

The rest of the code assumes that the dataset files are located in ./open_orca. Please modify the datasets/path filed in the .yaml files if they are in a different location.

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
