## Getting dataset:

The dataset can be obtained from the LLM task-force which is in the process of finalizing the contents of the dataset for both performance and accuracy. The dataset is in a parquet format.

## Launch server:

Common configs:

```
export HF_HOME=<your_hf_home_dir>
export HF_TOKEN=<your_hf_token>
export MODEL_NAME=openai/gpt-oss-120b

```

`vLLM` can be launched via:

```
docker run --runtime nvidia --gpus all -v ${HF_HOME}:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" -p 8000:8000 --ipc=host vllm/vllm-openai:latest --model ${MODEL_NAME} --gpu_memory_utilization 0.95
```

`SGLang` can be launched via:

```
docker run --runtime nvidia --gpus all --net host  -v ${HF_HOME}:/root/.cache/huggingface     --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN"     --ipc=host lmsysorg/sglang:latest python3 -m sglang.launch_server --model-path ${MODEL_NAME} --host 0.0.0.0  --port 3000 --data-parallel-size=1 --max-running-requests 512 --mem-fraction-static 0.85 --chunked-prefill-size 16384 --ep-size=1 --enable-metrics --stream-interval 500
```

## Launch benchmark:

```
inference-endpoint benchmark from-config -c examples/04_GPTOSS120B_Example/gptoss_120b_example.yaml --timeout 6000
```

## vllm bench:

`vllm bench serve` provided support for custom datasets only via the `jsonl` format. We can convert the parquet files to `jsonl` via the following script:

```
import pandas as pd

parquet_file = 'examples/04_GPTOSS120B_Example/data/perf_eval_ref.parquet'
json_file = 'examples/04_GPTOSS120B_Example/data/perf_eval_ref.jsonl'

# 1. Read the original file
df = pd.read_parquet(parquet_file)

# 2. Rename the column(s)
# Use a dictionary mapping old names to new names
df = df.rename(columns={'prompt': 'raw_prompt'})
df = df.rename(columns={'text_input': 'prompt'})

# 3. Write the renamed DataFrame to a new file
df.to_json(json_file, orient='records', lines=True)

```

Note that it also renames the column from `text_input` to `prompt` as the custom dataloader requires the `jsonl` to have the pre-processed prompt under that name.
We can launch the benchmarking command but it has to be pointed to the `completions` endpoint instead of the `chat-completions` endpoint as the prompt is preprocessed. While the numbers generated cannot be directly compared to inference-endpoint (which uses the `chat-completion` endpoint), it can provide a good reference for relative performance given the output token distribution.

```
vllm bench serve   --backend vllm   --model ${MODEL_NAME}   --endpoint /v1/completions   --dataset-name custom --dataset-path /home/scratch.rkaleem_gpu/datasets/gpt-oss/v3/acc/acc_eval_inputs.jsonl  --custom-output-len 2000 --num-prompts 6396 --max-concurrency 512 --save-result --save-detailed

```

## Debugging:

[mitmproxy](https://www.mitmproxy.org/) is a tool that can help debug HTTP requests and responses to understand the differences in payload for different scenarios. For our use case, we would like to be able to inspect the HTTP requests and responses between the benchmarking client and the server. We can run `mitmproxy` in a reverse-proxy mode as below:

```
mitmproxy -p 8001  --mode reverse:http://localhost:8000/
```

This launches `mitmproxy` at port `8081` and forwards it to `8000` on the local machine. Now our server (`vLLM` or `SGLang`) can run on port `8000` and our client will send requests to `8001` which will be logged and forwarded to the server. The client will receive the response back transparently with the responses being logged as well. This allows us to inspect the exact
