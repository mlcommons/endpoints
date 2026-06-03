# Multi-Turn Agentic Benchmark

This example runs multi-turn agentic conversations through an OpenAI-compatible
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

Place the dataset under `examples/09_MultiTurn/datasets/` or point the YAML at
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

## Client YAML

The runnable config is
`examples/09_MultiTurn/kimi_agentic_benchmark.yaml`.

### Fields

- `name`: human-readable run name written to reports and logs. Change this when
  creating a distinct benchmark config.
- `version: "1.0"`: config version label for this example. Final submission:
  keep `"1.0"`.
- `type: "online"`: runs through the online scheduler. Final submission: keep
  `"online"`.
- `model_params.name`: model name sent in each OpenAI request. Set this to the
  model name served by the endpoint.
- `model_params.temperature: 1.0`: sampling temperature sent to the server.
  Final submission: keep `1.0`.
- `model_params.top_p: 0.95`: nucleus sampling value sent to the server. Final
  submission: keep `0.95`.
- `model_params.max_new_tokens: 8192`: per-turn generation cap. Final
  submission: keep `8192`.
- `model_params.chat_template_kwargs.thinking: true`: Kimi chat-template option.
  Final submission: keep `true`.
- `model_params.chat_template_kwargs.preserve_thinking: true`: preserves
  reasoning content in the rendered prompt. Final submission: keep `true`.
- First dataset `name`: label used in benchmark outputs. Change this to match
  the dataset variant being run.
- First dataset `type: performance`: multi-turn replay is the performance
  dataset. Final submission: keep `performance`.
- First dataset `path`: JSONL dataset path to run. Replace the placeholder with
  the local or mounted dataset path.
- First dataset `multi_turn.enable_salt: true`: adds deterministic salt markers
  to conversation instances so repeats do not reuse KV cache by accident. Final
  submission: keep `true`.
- First dataset `multi_turn.inject_tool_delay: true`: honors positive
  `delay_seconds` values from the dataset before issuing user/tool turns. Final
  submission: keep `true`.
- First dataset `multi_turn.inline_accuracy: true`: scores the generated
  `events.jsonl` inline after the run. Final submission: keep `true`.
- First dataset `multi_turn.num_trajectories_to_issue: 990`: total number of
  trajectories to start. Change this to scale runtime. For final runs, use an
  integer multiple of the dataset trajectory count so each repeat has the same
  representation.
- First dataset `multi_turn.stop_issuing_on_first_user_complete`: controls only
  whether the client keeps issuing after the measurement window ends. Performance
  tracking always stops when the first concurrency slot finishes a trajectory and
  there is no next trajectory left to assign. If this field is `true`, the client
  stops issuing future turns at that point and drains already in-flight turns. If
  this field is `false`, the client keeps replaying already-started active
  trajectories to completion for accuracy/log coverage, but those later-issued
  turns are outside the performance measurement window. Final submission: keep
  `false`; use `true` only for faster optimization/debug runs.
- `settings.runtime.min_duration_ms: 0`: minimum run duration. Multi-turn replay
  completion is controlled by trajectory budget and active conversation drain.
- `settings.load_pattern.type: multi_turn`: enables conversation-aware issuing.
  Final submission: keep `multi_turn`.
- `settings.load_pattern.target_concurrency`: maximum active conversations. Each
  active conversation has at most one in-flight request. Change this for the
  target concurrency of the run.
- `settings.client.warmup_connections: 0`: disables pre-warmed HTTP sockets.
- `settings.client.max_idle_time: 0.5`: connection idle lifetime in seconds.
- `endpoint_config.endpoints`: server URL list. Replace with the endpoint URLs
  for the run.
- `endpoint_config.api_type: openai`: use `/v1/chat/completions`. Final
  submission: keep `openai`.
- `report_dir`: output directory for events, snapshots, scores, and reports.
  Change this per run so outputs are not overwritten.

### Salting Mechanism

When `multi_turn.enable_salt: true`, the client adds a short deterministic
`[salt: ...]` marker before the system prompt for the dataset repeat and another
after the system prompt for the conversation. Each salt is four hex characters.
This restricts kv-cache reuse to:

1. Fully allowed within a trajectory.
2. System prompt allowed within same iteration of the dataset.
3. Disallowed across multiple iterations of dataset.

### Inline Accuracy

When `multi_turn.inline_accuracy: true`, the benchmark scores the generated
`events.jsonl` during finalization and writes `scores.json` under `report_dir`.
Inline accuracy requires the performance dataset path because expected assistant
turns are read from the same JSONL used for replay. The scorer matches completed
assistant responses back to their conversation/turn ids and compares them with
the expected assistant turns embedded in the dataset.

### Tail Management

Multi-turn benchmarks can have a long tail because different users receive
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
`multi_turn.stop_issuing_on_first_user_complete: false` so the client finishes
already-started trajectories for accuracy. During optimization, set it to `true`
to stop issuing future turns at the performance boundary and shorten the tail.

## Run The Client

Update the first `datasets` entry (`name` and `path`), `model_params.name`, and
`endpoint_config.endpoints` as needed, then run:

```bash
uv run inference-endpoint benchmark from-config \
  --config examples/09_MultiTurn/kimi_agentic_benchmark.yaml
```
