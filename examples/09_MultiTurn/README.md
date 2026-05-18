# Multi-Turn Conversation Benchmarking Examples

This directory contains examples for benchmarking conversational AI workloads with multi-turn conversation support.

## Overview

Multi-turn conversation benchmarking enables testing realistic conversational AI scenarios where each turn depends on previous responses. The system maintains conversation history and enforces turn sequencing to simulate real-world multi-turn interactions.

## Dataset Format

Multi-turn datasets use JSONL format with the following structure:

```jsonl
{"conversation_id": "c1", "turn": 1, "role": "user", "content": "...", "system": "..."}
{"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "..."}
{"conversation_id": "c1", "turn": 3, "role": "user", "content": "..."}
```

### Required Fields

- `conversation_id`: Unique identifier for each conversation
- `turn`: Turn number within conversation (1-indexed)
- `role`: Speaker role ("user" or "assistant")
- `content`: Message content

### Optional Fields

- `system`: System prompt (typically only on first user turn)
- `model`: Model name override for this turn
- `max_new_tokens`: Maximum tokens to generate for this turn

### Validation Rules

1. All rows for a given `conversation_id` must appear **consecutively** in the file (no interleaving
   with rows from other conversations). File-order within a conversation does not matter — the
   loader sorts by the `turn` column when building conversation history.
   The flat-row format is intentional: it enables row-by-row streaming without loading entire
   conversations into memory first.
2. Conversations must follow a valid role sequence:
   - Plain chat: `user → assistant → user → ...`
   - Agentic: `user → assistant (with tool_calls) → tool (tool_results list; parallel results merged) → [assistant (with tool_calls) → tool]* → assistant → user → ...`
3. First turn must be "user" role
4. Turn numbers must be sequential (1, 2, 3, ...)
5. Each conversation must have at least one turn

## Agentic (Tool-Sequence) Datasets

For agentic workloads where the model dispatches tools, the dataset must include tool-call
metadata in flat-row JSONL form: one row per message, with all rows for a conversation grouped
contiguously.

### Agentic Row Fields

The extra fields supported beyond plain user/assistant:

| Row role                                   | Extra fields                                                       |
| ------------------------------------------ | ------------------------------------------------------------------ |
| `assistant` with tool calls                | `tool_calls: [{id, type, function: {name, arguments}}]`            |
| `tool` results (single or merged parallel) | `tool_results: [{tool_call_id, content}, ...]`                     |
| `user` or `tool` turns                     | `tools: [...]` (OpenAI tool definitions forwarded to the endpoint) |

Example rows from an agentic dataset:

```jsonl
{"conversation_id": "sim_001", "turn": 1, "role": "user", "content": "Fix the bug in foo.py", "system": "You are a coding agent.", "tools": [...]}
{"conversation_id": "sim_001", "turn": 2, "role": "assistant", "tool_calls": [{"id": "functions.bash:0", "type": "function", "function": {"name": "bash", "arguments": "{\"cmd\": \"cat foo.py\"}"}}]}
{"conversation_id": "sim_001", "turn": 3, "role": "tool", "tool_results": [{"tool_call_id": "functions.bash:0", "content": "def foo():\n    return 1/0"}], "tools": [...]}
{"conversation_id": "sim_001", "turn": 4, "role": "assistant", "content": "The bug is a ZeroDivisionError. Here is the fix: ..."}
```

### Running agentic benchmarks

Update the `name` and `path` fields in the config file for the local dataset you want to run,
then start the benchmark:

```bash
inference-endpoint benchmark from-config \
    --config examples/09_MultiTurn/kimi_agentic_benchmark.yaml
```

---

## Configuration

### Basic Configuration

```yaml
datasets:
  - name: agentic_coding
    type: performance
    path: /path/to/agentic_dataset.jsonl
    multi_turn:
      turn_timeout_s: 300.0
      inject_tool_delay: true

settings:
  load_pattern:
    type: multi_turn
    target_concurrency: 8 # ← Required for multi_turn load pattern
```

### Concurrency Control

The `target_concurrency` field is **required** for the `multi_turn` load pattern and controls the maximum number of conversations active simultaneously (each active conversation has at most one in-flight turn):

```yaml
settings:
  load_pattern:
    type: multi_turn
    target_concurrency: 8 # ← Limit to 8 concurrent requests
```

**Behavior**:

- With `target_concurrency`: At most `target_concurrency` conversations are active simultaneously; each active conversation has exactly one in-flight turn at any time.
- Turn sequencing is preserved: turn N+1 is issued only after turn N's response arrives.

**Use cases**:

- **Prevent endpoint overload**: Control request rate to busy endpoints
- **Large-scale testing**: Benchmark 1000+ conversations without overwhelming system
- **Resource management**: Stay within port limits, memory constraints

**Example**: 100 conversations with `target_concurrency: 8`

```
t=0:   Start 8 conversations, issue turn-1 for each (8 in-flight)
t=0.5: Turn-1 of conv A completes → issue turn-2 of conv A (still 8 in-flight)
t=1.0: All turns of conv B complete → start conv 9, issue its turn-1 (still 8 in-flight)
...    Maintains at most 8 active conversations
```

### Turn Timeout

Configure the maximum time allowed between issuing a turn and receiving its response:

```yaml
multi_turn:
  turn_timeout_s: 300.0 # 5 minutes
```

If a turn does not receive a response within `turn_timeout_s` seconds, that turn is marked failed and all remaining turns in the same conversation are aborted (subsequent turns depend on the timed-out response). The event is logged as a warning.

## Running Multi-Turn Benchmarks

### Using Configuration File

```bash
inference-endpoint benchmark from-config \
  --config examples/09_MultiTurn/kimi_agentic_benchmark.yaml
```

### Viewing Results

Multi-turn benchmarks produce per-turn metrics:

- **Per-turn metrics**: Latency, TTFT, TPOT for each individual turn
- **Per-conversation metrics**: Total conversation latency, conversations per second _(planned — not yet implemented)_

**Note**: Multi-turn datasets are only supported as performance datasets. Using a multi-turn dataset as an accuracy dataset (`type: accuracy`) is not yet supported and will raise an error at startup.

Results are stored in the configured `report_dir`; per-turn records include
`conversation_id` and `turn` for conversation-level filtering.

## Architecture Notes

### Key Components

- **ConversationManager**: Tracks conversation state and message history
- **MultiTurnStrategy**: Enforces turn sequencing within conversations
- **MultiTurnDataset**: Validates and structures multi-turn data

### Turn Sequencing

The system ensures that:

1. Turn N+1 cannot be issued until turn N completes
2. Message history is included in subsequent requests
3. Concurrent conversations are supported (in independent mode)

### Memory Considerations

Each conversation maintains message history in memory. For large-scale benchmarks with long conversations:

- Memory usage: ~1KB per turn (approximate)
- 1000 conversations × 10 turns = ~10MB

## Troubleshooting

### "Conversation has invalid role sequence"

**Cause**: Conversation doesn't follow a valid role sequence.

**Fix**: For plain chat, ensure the dataset alternates between user and assistant:

```
user -> assistant -> user -> assistant -> ...
```

For agentic datasets, make sure the input JSONL is already in the valid flat-row sequence:

```
user -> assistant (tool_calls) -> tool -> [assistant (tool_calls) -> tool]* -> assistant -> user -> ...
```

**Note**: Parallel tool results from a single dispatch must be **merged into
one row** with a `tool_results` list, not represented as multiple consecutive
`tool` rows. The validator rejects consecutive `tool` rows.

### "Turn timed out"

**Cause**: A turn did not receive a response within `turn_timeout_s` seconds after it was issued.

**Fixes**:

- Increase `turn_timeout_s` in configuration
- Check endpoint performance
- Verify endpoint is responding

### Single-turn benchmarks unaffected

Multi-turn logic is only activated when a `multi_turn:` block is present in the dataset configuration. Existing single-turn benchmarks continue to work unchanged with zero performance overhead.

## Future Enhancements

Planned features:

- [ ] Poisson conversation arrival mode
- [ ] Per-conversation metrics in reporting (total conversation latency, conversations per second)
- [ ] Conversation-level latency percentiles
- [ ] Dynamic conversation branching
