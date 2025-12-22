# Running Endpoints with [Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

## Download dataset

The Llama3.1-8B benchmark uses the [cnn/dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) dataset (for summarization). Download, modify the input prompt and save it using the following command:

```
python download_cnndm.py --save-dir data --split validation
# Processed data will be saved at data/cnn_dailymail_validation.json
```

- To generate calibration dataset, users can use the [cnn-dailymail-calibration-list](https://github.com/mlcommons/inference/blob/v4.0/calibration/CNNDailyMail/calibration-list.txt)

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

We can launch the latest docker image for vllm using the command below:

```
docker run --runtime nvidia --gpus all -v ${HF_HOME}:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" -p 8000:8000 --ipc=host vllm/vllm-openai:latest  --model ${MODEL_NAME}

```

### To run Offline mode

**Note** Double-check the config file for correct parameters

- Launch the benchmark with config yaml

```
inference-endpoint benchmark from-config -c offline_llama3_8b_cnn.yaml --timeout 600
```

### To run Online mode

**Note** Double-check the config file for correct parameters

- Launch the benchmark with config yaml

```
inference-endpoint benchmark from-config -c online_llama3_8b_cnn.yaml --timeout 600
```
