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

Fields:

- `name`: human-readable run name written to reports and logs.
- `version`: config version label for this example.
- `type: online`: runs through the online scheduler. Final submission: do not
  modify.
- `model_params.name`: model name sent in each OpenAI request. Keep it aligned
  with the served model name.
- `model_params.temperature`: sampling temperature sent to the server. Final
  submission: do not modify.
- `model_params.top_p`: nucleus sampling value sent to the server. Final
  submission: do not modify.
- `model_params.max_new_tokens`: per-turn generation cap. Final submission: do
  not modify.
- `model_params.chat_template_kwargs.thinking`: Kimi chat-template option.
  Final submission: do not modify.
- `model_params.chat_template_kwargs.preserve_thinking`: Kimi chat-template
  option that preserves reasoning content in the rendered prompt. Final
  submission: do not modify.
- First dataset `name`: label used in benchmark outputs.
- First dataset `type: performance`: multi-turn replay is the performance
  dataset. Final submission: do not modify.
- First dataset `path`: JSONL dataset path to run.
- First dataset `multi_turn.turn_timeout_s`: per-turn deadline in seconds. A
  timeout aborts remaining turns in that conversation.
- First dataset `multi_turn.enable_salt`: appends a deterministic cache salt to
  repeated conversation instances so repeats do not reuse KV cache by accident.
  Final submission: do not modify.
- First dataset `multi_turn.inject_tool_delay`: when true, honors positive
  `delay_seconds` values from the dataset before issuing user/tool turns. Final
  submission: do not modify.
- First dataset `multi_turn.inline_accuracy`: when true, scores the generated
  `events.jsonl` inline after the run. Final submission: do not modify.
- First dataset `multi_turn.num_trajectories_to_issue`: total number of
  trajectories to start. If it is larger than the dataset trajectory count, the
  dataset is repeated in order with repeat-specific logical conversation ids.
- First dataset `multi_turn.stop_issuing_on_first_user_complete`: controls only
  whether the client keeps issuing after the measurement window ends. Performance
  tracking always stops when the first concurrency slot finishes a trajectory and
  there is no next trajectory left to assign. If this field is `true`, the client
  stops issuing future turns at that point and drains already in-flight turns. If
  this field is `false`, the client keeps replaying already-started active
  trajectories to completion for accuracy/log coverage, but those later-issued
  turns are outside the performance measurement window. Final submission: set to
  `false` for valid accuracy; use `true` only for faster optimization/debug runs.
- `settings.runtime.min_duration_ms`: minimum run duration. For multi-turn
  replay, completion is primarily controlled by trajectory budget and active
  conversation drain.
- `settings.load_pattern.type: multi_turn`: enables conversation-aware issuing.
- `settings.load_pattern.target_concurrency`: maximum active conversations.
  Each active conversation has at most one in-flight request.
- `settings.client.warmup_connections: 0`: disables pre-warmed HTTP sockets.
- `settings.client.max_idle_time`: connection idle lifetime in seconds.
- `endpoint_config.endpoints`: server URL list.
- `endpoint_config.api_type: openai`: use `/v1/chat/completions`. Final
  submission: do not modify.
- `report_dir`: output directory for events, snapshots, scores, and reports.

Performance measurement is based on issue time. A turn issued before
`STOP_PERFORMANCE_TRACKING` is counted even if it completes after the stop event.
A turn issued after that event is not counted in performance metrics.

## Run The Client

Update the first `datasets` entry (`name` and `path`), `model_params.name`, and
`endpoint_config.endpoints` as needed, then run:

```bash
uv run inference-endpoint benchmark from-config \
  --config examples/09_MultiTurn/kimi_agentic_benchmark.yaml
```
