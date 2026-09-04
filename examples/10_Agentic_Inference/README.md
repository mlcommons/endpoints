# Multi-Turn Agentic Benchmark

This example runs agentic inference conversations through an OpenAI-compatible endpoint. The client preserves conversation order, sends one in-flight turn per active conversation, and adds `X-Session-ID: <conversation_id>` on every request so a router can keep a conversation on the same backend.

## Dataset

Use flat JSONL with one row per message. Rows for each `conversation_id` must be contiguous and ordered by increasing `turn`.

```jsonl
{"conversation_id":"c1","turn":1,"role":"user","system":"...","content":"...","tools":[...],"delay_seconds":0.4}
{"conversation_id":"c1","turn":2,"role":"assistant","tool_calls":[...]}
{"conversation_id":"c1","turn":3,"role":"tool","tool_results":[...],"delay_seconds":1.2}
{"conversation_id":"c1","turn":4,"role":"assistant","content":"..."}
```

Required fields are `conversation_id`, `turn`, and `role`. User rows normally include `content`; agentic rows can also include `system`, `tools`, `tool_calls`, `tool_results`, `reasoning_content`, and `delay_seconds`.

The official MLPerf dataset is available from [MLCommons storage](https://endpoints.mlcommons-storage.org/index.html#mlperf-agentic-inference). The dataset SHA-256 is `1beb24c882122df96571cf11b390acbea388944038bc55c78b891475459014ae`. Submitters must use this dataset unchanged for official submissions.

Place the dataset under `examples/10_Agentic_Inference/datasets/` or point the YAML at another accessible JSONL path.

## Supported Models

The Agentic Inference benchmark can support any model. The current iteration of the MLPerf Inference benchmark accepts official submissions for the following three models:

| Model                  | Architecture                   | Parameters               | Context          |
| ---------------------- | ------------------------------ | ------------------------ | ---------------- |
| Kimi K3                | MoE + KDA/Gated MLA            | 2.8T total / 104B active | 1,048,576 tokens |
| Qwen3.6-35B-A3B        | MoE + Gated DeltaNet/Attention | 35B total / 3B active    | 262,144 tokens   |
| DeepSeek-V4-Pro (DSV4) | MoE + hybrid CSA/HCA + mHC     | 1.6T total / 49B active  | 1,048,576 tokens |

Reference implementations and runnable examples for all three models are provided below.

## Start A Server

To run the benchmark, expose one of the supported models through an OpenAI-compatible API endpoint. The serving framework is not prescribed: submitters may use vLLM, SGLang, TensorRT-LLM, or another serving framework as long as it provides an OpenAI-compatible endpoint.

The following SGLang commands are reference examples. Adjust model paths, parallelism, ports, and memory settings for your hardware.

### Kimi K3

See the [SGLang Kimi-K3 recipe](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3) for model-specific deployment guidance.

```bash
sglang serve \
  --model-path moonshotai/Kimi-K3 \
  --served-model-name moonshotai/Kimi-K3 \
  --tp-size 8 \
  --trust-remote-code \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --host 0.0.0.0 \
  --port 8000
```

### Qwen3.6-35B-A3B

See the [SGLang Qwen3.6 recipe](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.6) for model-specific deployment guidance.

```bash
sglang serve \
  --model-path Qwen/Qwen3.6-35B-A3B \
  --served-model-name Qwen/Qwen3.6-35B-A3B \
  --port 30000 \
  --host 0.0.0.0 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --weight-loader-prefetch-checkpoints \
  --mem-fraction-static 0.95
```

### DSV4

See the [SGLang DeepSeek-V4 recipe](https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4) for hardware-specific deployment guidance.

```bash
sglang serve \
  --model-path deepseek-ai/DeepSeek-V4-Pro-0813 \
  --served-model-name deepseek-ai/DeepSeek-V4-Pro-0813 \
  --tp-size 8 \
  --trust-remote-code \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --host 0.0.0.0 \
  --port 30000
```

`--model-path` is the checkpoint loaded by the server. It can be a local path visible to the server container or a Hugging Face model ID, depending on your SGLang environment. `--served-model-name` is the OpenAI model name exposed to clients; set `model_params.name` in the YAML to the same value.

## Client YAML

Runnable example configs are provided for [Kimi K3](kimi_agentic_benchmark.yaml), [Qwen3.6-35B-A3B](qwen_agentic_benchmark.yaml), and [DSV4](dsv4_agentic_benchmark.yaml).

Some key client features specific to the Agentic Inference benchmark are described below.

### Salting Mechanism

`agentic_inference.enable_salt` must be set to `true` for official submissions.

When `agentic_inference.enable_salt: true`, the strategy adds a short deterministic `[salt: ...]` marker before the system prompt for the trajectory repeat and another after the system prompt for the conversation. Each salt is four hex characters. This restricts kv-cache reuse to:

1. Fully allowed within a trajectory.
2. System prompt allowed within same iteration of the dataset.
3. Disallowed across multiple iterations of dataset.

### Inline Accuracy

Submitters must enable inline accuracy for official submissions by setting `accuracy_config.eval_method: agentic_inference_inline` on the performance dataset. The benchmark then scores the generated `events.jsonl` during finalization and writes `scores.json` under `report_dir`. The scorer uses the loaded agentic inference dataset as ground truth, matches completed assistant responses back to their conversation/turn ids, and compares them with the expected assistant turns embedded in the dataset. It does not issue a separate accuracy phase.

### Tail Management

Agentic inference benchmarks can have a long tail because different users receive trajectories with very different turn counts, delays, and generated lengths. In large runs this tail can last up to an hour after steady-state work has already ended, so the benchmark separates the performance window from the remaining accuracy/logging drain.

The benchmark stops performance tracking when the first active user finishes its final assigned trajectory. It emits `STOP_PERFORMANCE_TRACKING` at that point to avoid measuring the tail. Turns issued before this event remain in the performance window even if they finish later; turns issued after it are excluded from performance metrics.

For official submissions, submitters must set `agentic_inference.stop_issuing_on_first_user_complete` to `false` so the client finishes already-started trajectories for accuracy. During optimization, set it to `true` to stop issuing future turns at the performance boundary and shorten the tail.

### SWE-bench Accuracy

Submitters must enable SWE-bench accuracy for official submissions. The Kimi K3, Qwen3.6-35B-A3B, and DSV4 example YAML files include the required SWE-bench accuracy dataset. The benchmark framework skips its built-in endpoint phase for the SWE-bench dataset. Instead, `SWEBenchScorer` submits the run to a native SWE-bench service. The service host owns Docker, `mini-swe-agent`, and the `swebench` evaluation harness, and it drives requests to the configured endpoint.

Keep `accuracy_config.num_repeats: 1`: the scorer performs one external evaluation run per benchmark. Optional `accuracy_config.extras.subset` and `split` are used consistently for dataset loading, preflight, and scoring.

`accuracy_config.extras.swebench_service_url` points the benchmark client to the service. Service mode follows the LiveCodeBench-style external-service convention for heavyweight evaluation work and supports exactly one endpoint URL in `endpoint_config.endpoints`; that URL must be reachable from the service host. Treat the service host as trusted infrastructure: it receives the endpoint URL and optional endpoint API key needed to run mini-swe-agent. Start the service with `--auth-token` and set `accuracy_config.extras.swebench_service_auth_token`. Only isolated local development should use the explicit `--allow-unauthenticated` override.

`accuracy_config.extras.workers` sets the agent run's parallelism (`--workers`). If unset, it defaults to the load pattern's `target_concurrency` (for `concurrency`/`agentic_inference` patterns), else 10. `max_eval_workers` (default 10, `--max_workers`) sets the eval harness's parallelism.

Qwen tool-call runs should set `accuracy_config.extras.swebench_template: qwen_tools`. The selected packaged template also activates the service's `QwenToolsModel` through mini-swe-agent's `model_class` hook.

If SWE-bench evaluation is needed, start the service with the following command on a host that has Docker:

```bash
uv run --project src/inference_endpoint/evaluation/swebench_service \
  python -m swebench_service --host 0.0.0.0 --port 18080 \
  --auth-token "$SWEBENCH_SERVICE_AUTH_TOKEN"
```

#### Build ARM64 SWE-bench Images

[`build_and_push.py`](build_and_push.py) builds and pushes the native ARM64 images for the first 200 pinned SWE-bench Verified tasks. The script validates the task list, applies the required ARM compatibility fixes, and skips images that already exist in the destination registry.

Run it on a native ARM64 machine with Docker after logging in to a registry where you have push access:

```bash
python3 -m venv /tmp/swebench-arm64-venv
/tmp/swebench-arm64-venv/bin/pip install "swebench==4.1.0"
docker login registry.example.com

REGISTRY=registry.example.com/group/project \
  /tmp/swebench-arm64-venv/bin/python \
  examples/10_Agentic_Inference/build_and_push.py
```

Set `WORKERS` to change the default build and push concurrency of `16`. Images are tagged `v4.1.0-arm64`, and interrupted runs can be resumed with the same command.

## Run The Client

Update the first `datasets` entry (`name` and `path`), `model_params.name`, and `endpoint_config.endpoints` as needed. Then select the matching model config and run it from the repo root:

```bash
CONFIG=examples/10_Agentic_Inference/qwen_agentic_benchmark.yaml
# For Kimi, use examples/10_Agentic_Inference/kimi_agentic_benchmark.yaml.
# For DSV4, use examples/10_Agentic_Inference/dsv4_agentic_benchmark.yaml.

# PERF (default): agentic performance and inline scoring; skips SWE-bench.
uv run inference-endpoint benchmark from-config --config "$CONFIG"

# BOTH: agentic performance followed by SWE-bench.
uv run inference-endpoint benchmark from-config --config "$CONFIG" --mode both

# ACC: SWE-bench only; skips the agentic performance dataset.
uv run inference-endpoint benchmark from-config --config "$CONFIG" --mode acc
```

The default `PERF` mode does not load, preflight, or submit external evaluation scorers. Use `--mode both` or `--mode acc` whenever SWE-bench should run.

See `accuracy/RUNBOOK.md` for preconditions, sanity checks, and common failure modes.

## Official Submission Rules

The YAML configuration gives submitters flexibility to configure the benchmark for their systems, but official submissions must follow the requirements below. All MLPerf Endpoint Benchmark rules apply in full. Where a requirement below is more specific to the Agentic Inference benchmark, the more specific requirement takes precedence for this benchmark.

### Sampling Parameters and Thinking Flags

Submitters must not modify the sampling parameters or thinking flags. The model-specific values in the example YAML files are authoritative and are repeated below for completeness:

For Kimi K3:

- `temperature: 1.0`
- `top_p: 1.0`
- `max_new_tokens: 8192`

For Qwen3.6-35B-A3B:

- `temperature: 1.0`
- `top_k: 20`
- `top_p: 0.95`
- `repetition_penalty: 1.0`
- `presence_penalty: 1.5`
- `max_new_tokens: 8192`
- `chat_template_kwargs.preserve_thinking: true`

For DSV4:

- `temperature: 1.0`
- `top_k: 0`
- `top_p: 0.95`
- `max_new_tokens: 8192`
- `streaming: "on"`
- `chat_template_kwargs.thinking: true`
- `chat_template_kwargs.reasoning_effort: max`
- `chat_template_kwargs.preserve_thinking: true`

Any sampling parameter or thinking flag not listed above for the selected model must be omitted. Submitters must not introduce additional sampling parameters or thinking flags.

`preserve_thinking` ensures that reasoning tokens from previous turns are not stripped by the chat template before the input is sent to the inference engine. Because chat-template processing is a server-side property, the client can only request this behavior by sending the flag. Popular serving frameworks, including SGLang, vLLM, and TensorRT-LLM, honor this flag; however, each submitter is responsible for verifying that their server is compliant. The chat template must not omit reasoning tokens from any previous turn.

### Dataset Size

For official submissions, `agentic_inference.num_trajectories_to_issue` must be a positive integer multiple of the total dataset size. The official dataset contains 613 trajectories, so valid values are `613`, `1226`, `1839`, and so on.

Submitters must also enable salting and inter-turn delays by setting:

- `agentic_inference.enable_salt: true`
- `agentic_inference.inject_tool_delay: true`

For official submissions, `agentic_inference.stop_issuing_on_first_user_complete` must be set to `false` so the client finishes already-started trajectories for accuracy after the performance window ends. Setting it to `true` stops issuing future turns at the performance boundary and may be used only for faster optimization or debugging runs. Runs with this setting enabled are not valid official submissions.

### Accuracy

Official submissions must enable both inline accuracy and SWE-bench accuracy. Configure the performance dataset with `accuracy_config.eval_method: agentic_inference_inline`, and configure the SWE-bench accuracy dataset with `accuracy_config.eval_method: swe_bench_scorer`. For SWE-bench, `accuracy_config.extras.num_instances` must be set to `200`. When using the example `online` configs, run with `--mode both` so performance, inline accuracy, and SWE-bench accuracy are all executed.

Qwen3.6-35B-A3B submissions must set `accuracy_config.extras.swebench_template: qwen_tools`. Kimi K3 submissions must omit `accuracy_config.extras.swebench_template`. The DSV4 SWE-bench template requirement is TBD.

Every Kimi K3 and Qwen3.6-35B-A3B submitted Pareto point must satisfy all of the model-specific accuracy thresholds below. For these models, SWE-bench accuracy is evaluated using mean-of-N: average one SWE-bench accuracy result from each of the [four mandatory regions](https://github.com/mlcommons/endpoints_policies/blob/main/endpoints_rules.md#54-regions-of-interest) (`N = 4`), then compare that mean with the model-specific SWE-bench threshold below. The DSV4 accuracy thresholds and SWE-bench evaluation policy are TBD.

| Metric             |          Kimi K3 |  Qwen3.6-35B-A3B | DSV4 |
| ------------------ | ---------------: | ---------------: | ---: |
| Inline accuracy    |      `>= 58.32%` |      `>= 55.86%` |  TBD |
| OSL per-turn mean  | `390-475` tokens | `355-434` tokens |  TBD |
| SWE-bench accuracy |       `>= 93.5%` |         `>= 69%` |  TBD |

### Approved Checkpoints and Speculative-Decoding Heads

Under the MLPerf Endpoint Benchmark rules, submitters must use only the approved model checkpoints and speculative-decoding heads listed below in their submissions. New artifacts may be added to this list only after approval from the Agentic Inference taskforce.

#### Kimi K3

Approved model checkpoints:

- [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
- [nvidia/Kimi-K3-NVFP4](https://huggingface.co/nvidia/Kimi-K3-NVFP4)

Approved speculative-decoding heads:

- [RadixArk/Kimi-K3-DSpark](https://huggingface.co/RadixArk/Kimi-K3-DSpark)
- [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark)

#### DSV4

Approved model checkpoints:

- [deepseek-ai/DeepSeek-V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813)
- [nvidia/DeepSeek-V4-Pro-0813-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4-Pro-0813-NVFP4)

Both approved DSV4 checkpoints include bundled DSpark speculative-decoding heads.

#### Qwen3.6-35B-A3B

Approved model checkpoints:

- [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) (contains both W4A16 and W4A4 scales)

Both approved Qwen checkpoints include their native MTP speculative-decoding heads.
