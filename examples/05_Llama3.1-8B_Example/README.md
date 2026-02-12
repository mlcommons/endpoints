# Running Endpoints with [Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

It is recommended to use a config file such as [online_llama3_8b_cnn.yaml](online_llama3_8b_cnn.yaml) to run the benchmark.

## [Optional] Download dataset

The Llama3.1-8B benchmark uses the [cnn/dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset (for summarization). If using a config (such as provided) to run the benchmark, the (validation) dataset is downloaded automatically by specifying dataset name as `- name: cnn_dailymail::llama3_8b` under the `dataset` entry.

- For post-training quantization, users can use the [cnn-dailymail-calibration-list](https://github.com/mlcommons/inference/blob/v4.0/calibration/CNNDailyMail/calibration-list.txt) to select samples for the calibration.

```
curl -OL https://raw.githubusercontent.com/mlcommons/inference/v4.0/calibration/CNNDailyMail/calibration-list.txt
python download_cnndm.py --save-dir data --calibration-ids-file calibration-list.txt --split train
```

## Launch the server

The following environment variables are used by the commands below to make the scripts easier to run

```
export HF_TOKEN=<your Hugging Face token>
export HF_HOME=<Path to your hf_home, usually /USERNAME/.cache/huggingface>
export MODEL_NAME=<model to run, for instance meta-llama/Llama-3.1-8B-Instruct>
```

It is convenient to download the model prior to launch so that the container can reuse the model instead of having to download it post-launch. This can be done via `hf download $MODEL_NAME`. The models downloaded can be verified via `hf cache scan`

### [vLLM](https://github.com/vllm-project/vllm)

**Note**: To generate same outputs as the ones produced from submissions with legacy loadgen, we need to apply a custom chat template (this is taken care of automatically by the cnn-dailymail dataset preset). The flag `--trust-request-chat-template` is also required.

We can launch the latest docker image for vllm using the command below:

```
docker run --runtime nvidia --gpus all -v ${HF_HOME}:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model ${MODEL_NAME} --trust-request-chat-template
```

### To run Offline mode

**Note** Double-check the config file for correct parameters such as the model name in the config

- Launch the benchmark with config yaml

```
inference-endpoint benchmark from-config -c offline_llama3_8b_cnn.yaml --timeout 600
```

### To run Online mode

**Note** Double-check the config file for correct parameters

- Launch the benchmark with config yaml (For performance only, remove the accuracy dataset entry in the `online_llama3_8b_cnn.yaml`)

```
inference-endpoint benchmark from-config -c online_llama3_8b_cnn.yaml --timeout 600
```
