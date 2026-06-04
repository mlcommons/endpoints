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

Key fields:

- `type: online`: runs the benchmark through the online scheduler.
- `model_params.name`: model name sent in each OpenAI request. Keep it aligned
  with the served model name.
- `model_params.temperature`, `top_p`, `max_new_tokens`: sampling settings sent
  to the server. `max_new_tokens` is large because agent turns can be long.
- `model_params.chat_template_kwargs`: Kimi-specific template options for
  reasoning preservation.
- First `datasets` entry `name`: label used in benchmark outputs.
- First `datasets` entry `type: performance`: multi-turn datasets are replayed as
  performance datasets.
- First `datasets` entry `path`: JSONL dataset path to run.
- First `datasets` entry `multi_turn.turn_timeout_s`: per-turn deadline. A
  timeout aborts the remaining turns in that conversation.
- First `datasets` entry `multi_turn.enable_salt`: appends a deterministic cache
  salt to each conversation system prompt.
- First `datasets` entry `multi_turn.inject_tool_delay`: honors positive
  `delay_seconds` values from client turns before issuing those turns.
- `settings.runtime.min_duration_ms`: minimum run duration. With no max duration
  override, the run finishes when the dataset is exhausted.
- `settings.load_pattern.type: multi_turn`: enables conversation-aware issuing.
- `settings.load_pattern.target_concurrency`: maximum active conversations.
  Each active conversation has at most one in-flight request.
- `settings.client.warmup_connections: 0`: avoids stale pre-warmed sockets with
  servers that close idle connections quickly.
- `settings.client.max_idle_time`: connection idle lifetime.
- `endpoint_config.endpoints`: server URL list.
- `endpoint_config.api_type: openai`: use `/v1/chat/completions`.
- `report_dir`: output directory for events, snapshots, and reports.

## Run The Client

Update the first `datasets` entry (`name` and `path`), `model_params.name`, and
`endpoint_config.endpoints` as needed, then run:

```bash
uv run inference-endpoint benchmark from-config \
  --config examples/09_MultiTurn/kimi_agentic_benchmark.yaml
```
