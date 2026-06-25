# Multi-Turn Agentic Benchmark

This example runs agentic inference conversations through an OpenAI-compatible
endpoint. The client preserves conversation order, sends one in-flight turn per
active conversation, and adds `X-Session-ID: <conversation_id>` on every request
so a router can keep a conversation on the same backend.

## Dataset

Use flat JSONL with one row per message. Rows for each `conversation_id` must be
contiguous and ordered by increasing `turn`.

```jsonl
{"conversation_id":"c1","turn":1,"role":"user","system":"...","content":"...","tools":[...],"delay_seconds":0.4}
{"conversation_id":"c1","turn":2,"role":"assistant","tool_calls":[...]}
{"conversation_id":"c1","turn":3,"role":"tool","tool_results":[...],"delay_seconds":1.2}
{"conversation_id":"c1","turn":4,"role":"assistant","content":"..."}
```

Required fields are `conversation_id`, `turn`, and `role`. User rows normally
include `content`; agentic rows can also include `system`, `tools`,
`tool_calls`, `tool_results`, `reasoning_content`, and `delay_seconds`.

Place the dataset under `examples/10_Agentic_Inference/datasets/` or point the YAML at
another accessible JSONL path.

## Start A Server

Start an SGLang OpenAI-compatible server. This is the standard recipe used for
throughput replays; adjust `--model-path`, `--tp`, and `--port` for your node.

```bash
python3 -m sglang.launch_server \
  --model-path /path/to/Kimi-K2.6 \
  --served-model-name kimi-k2.6 \
  --tp 8 \
  --trust-remote-code \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --host 0.0.0.0 \
  --port 8000
```

`--model-path` is the checkpoint loaded by the server. It can be a local path
visible to the server container or a Hugging Face model id, depending on your
SGLang environment. `--served-model-name` is the OpenAI model name exposed to
clients; set `model_params.name` in the YAML to the same value.

To enable Kimi Eagle3 speculative decoding in SGLang, use the
[`nvidia/Kimi-K2.6-Eagle3`](https://huggingface.co/nvidia/Kimi-K2.6-Eagle3)
draft checkpoint. Add the following optional flags to the server command and set
`--speculative-draft-model-path` to that Eagle3 checkpoint path as visible
inside the server container:

```bash
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path /path/to/Kimi-K2.6-Eagle3 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

## Client YAML

The runnable config is
`examples/10_Agentic_Inference/kimi_agentic_benchmark.yaml`.

### Fields

- `name`: human-readable run name written to reports and logs. Change this when
  creating a distinct benchmark config.
- `version`: config version label for this example.
- `type`: scheduler mode for the run.
- `model_params.name`: model name sent in each OpenAI request. Set this to the
  model name served by the endpoint.
- `model_params.temperature`: sampling temperature sent to the server.
- `model_params.top_p`: nucleus sampling value sent to the server.
- `model_params.max_new_tokens`: per-turn generation cap.
- `model_params.chat_template_kwargs.thinking`: Kimi chat-template option.
- `model_params.chat_template_kwargs.preserve_thinking`: preserves
  reasoning content in the rendered prompt.
- First dataset `name`: label used in benchmark outputs. Change this to match
  the dataset variant being run.
- First dataset `type`: dataset role for this entry.
- First dataset `path`: JSONL dataset path to run. Set this to a real local or
  mounted dataset path, for example `/path/to/agentic_combined.jsonl`.
- First dataset `accuracy_config.eval_method`: scorer used during finalization.
  `agentic_inference_inline` scores the performance replay outputs without issuing a
  separate accuracy phase.
- First dataset `agentic_inference.enable_salt`: applies deterministic salt
  markers when issuing conversation instances so repeats do not reuse KV cache
  by accident.
- First dataset `agentic_inference.inject_tool_delay`: honors positive
  `delay_seconds` values from the dataset before issuing user/tool turns.
- First dataset `agentic_inference.num_trajectories_to_issue`: total number of
  trajectories to start. Change this to scale runtime.
- First dataset `agentic_inference.stop_issuing_on_first_user_complete`: controls only
  whether the client keeps issuing after the measurement window ends. Performance
  tracking always stops when the first concurrency slot finishes a trajectory and
  there is no next trajectory left to assign. If this field is `true`, the client
  stops issuing future turns at that point and drains already in-flight turns. If
  this field is `false`, the client keeps replaying already-started active
  trajectories to completion for accuracy/log coverage, but those later-issued
  turns are outside the performance measurement window.
- `settings.runtime.min_duration_ms`: minimum run duration. Agentic inference replay
  completion is controlled by trajectory budget and active conversation drain.
- `settings.load_pattern.type`: enables conversation-aware issuing.
- `settings.load_pattern.target_concurrency`: maximum active conversations. Each
  active conversation has at most one in-flight request. Change this for the
  target concurrency of the run.
- `settings.client.warmup_connections`: disables pre-warmed HTTP sockets.
- `settings.client.max_idle_time`: connection idle lifetime in seconds.
- `endpoint_config.endpoints`: server URL list. Replace with the endpoint URLs
  for the run.
- `endpoint_config.api_type`: selects the endpoint protocol and route.
- `report_dir`: output directory for events, snapshots, scores, and reports.
  Change this per run so outputs are not overwritten.

### Benchmark Invariants

For official Kimi agentic benchmark runs, keep these values fixed:

- `version: "1.0"`
- `type: "online"`
- `model_params.temperature: 1.0`
- `model_params.top_p: 0.95`
- `model_params.max_new_tokens: 8192`
- `model_params.chat_template_kwargs.thinking: true`
- `model_params.chat_template_kwargs.preserve_thinking: true`
- First dataset `type: performance`
- First dataset `accuracy_config.eval_method: agentic_inference_inline`
- `settings.runtime.min_duration_ms: 0`
- `settings.load_pattern.type: agentic_inference`
- `settings.client.warmup_connections: 0`
- `settings.client.max_idle_time: 0.5`
- `endpoint_config.api_type: openai`

The agentic inference dataset required defaults are:

- First dataset `agentic_inference.enable_salt: true`
- First dataset `agentic_inference.inject_tool_delay: true`
- First dataset `agentic_inference.stop_issuing_on_first_user_complete: false`

Set `agentic_inference.num_trajectories_to_issue` to an integer multiple of the
dataset trajectory count so each repeat has the same representation. Use
`agentic_inference.stop_issuing_on_first_user_complete: true` only for faster
optimization/debug runs, not official benchmark runs.

### Salting Mechanism

When `agentic_inference.enable_salt: true`, the strategy adds a short deterministic
`[salt: ...]` marker before the system prompt for the trajectory repeat and
another after the system prompt for the conversation. Each salt is four hex characters.
This restricts kv-cache reuse to:

1. Fully allowed within a trajectory.
2. System prompt allowed within same iteration of the dataset.
3. Disallowed across multiple iterations of dataset.

### Inline Accuracy

When `accuracy_config.eval_method: agentic_inference_inline` is set on the performance
dataset, the benchmark scores the generated `events.jsonl` during finalization
and writes `scores.json` under `report_dir`. The scorer uses the loaded
agentic inference dataset as ground truth, matches completed assistant responses back
to their conversation/turn ids, and compares them with the expected assistant
turns embedded in the dataset. It does not issue a separate accuracy phase.

### Tail Management

Agentic inference benchmarks can have a long tail because different users receive
trajectories with very different turn counts, delays, and generated lengths. In
large runs this tail can last up to an hour after steady-state work has already
ended, so the benchmark separates the performance window from the remaining
accuracy/logging drain.

The benchmark stops performance tracking when the first active user finishes its
final assigned trajectory. It emits `STOP_PERFORMANCE_TRACKING` at that point to
avoid measuring the tail. Turns issued before this event remain in the
performance window even if they finish later; turns issued after it are excluded
from performance metrics.

For final submissions, keep
`agentic_inference.stop_issuing_on_first_user_complete: false` so the client finishes
already-started trajectories for accuracy. During optimization, set it to `true`
to stop issuing future turns at the performance boundary and shorten the tail.

## Run The Client

Update the first `datasets` entry (`name` and `path`), `model_params.name`, and
`endpoint_config.endpoints` as needed, then run:

```bash
uv run inference-endpoint benchmark from-config \
  --config examples/10_Agentic_Inference/kimi_agentic_benchmark.yaml
```

## SWE-bench Accuracy

`swe_bench_accuracy.yaml` runs the SWE-bench accuracy evaluation alongside a
minimal performance dataset. The benchmark framework skips its built-in
accuracy phase for this dataset; instead, `SWEBenchScorer` shells out to
`mini-swe-agent` and the `swebench` evaluation harness, and that external flow
drives requests to the configured endpoint.

Keep `accuracy_config.num_repeats: 1`: the scorer performs one external
evaluation run per benchmark. Optional `accuracy_config.extras.subset` and
`split` are used consistently for dataset loading, preflight, and scoring.

The isolated `uv` environment for those tools lives in `accuracy/`. Sync it
once before running:

```bash
cd examples/10_Agentic_Inference/accuracy
uv sync
```

Then run the benchmark from the repo root:

```bash
uv run inference-endpoint benchmark from-config \
  --config examples/10_Agentic_Inference/swe_bench_accuracy.yaml
```

See `accuracy/RUNBOOK.md` for preconditions, sanity checks, and common failure
modes.
