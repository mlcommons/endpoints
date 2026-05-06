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
   with rows from other conversations). Turns within a conversation must be in order.
   The flat-row format is intentional: it enables row-by-row streaming without loading entire
   conversations into memory first.
2. Conversations must follow a valid role sequence:
   - Plain chat: `user → assistant → user → ...`
   - Agentic: `user → assistant (with tool_calls) → tool → [tool | assistant (with tool_calls)]* → assistant → user → ...`
3. First turn must be "user" role
4. Turn numbers must be sequential (1, 2, 3, ...)
5. Each conversation must have at least one turn

## Agentic (Tool-Sequence) Datasets

For agentic workloads where the model dispatches tools, the dataset must include tool-call
metadata. The source format for these datasets is a **snapshot JSONL** — each line contains the
full conversation history at a particular checkpoint. The benchmarker requires **flat-row JSONL**
(one row per message), so a conversion step is needed first.

### Source snapshot format

Each line in the source file represents one snapshot of a conversation:

```json
{
  "conversation_id": "sim_001",
  "conversation_idx": 5,
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "tool_calls": [{"id": "...", "type": "function", "function": {"name": "bash", "arguments": "{\"cmd\": \"ls\"}"}}]},
    {"role": "tool", "tool_call_id": "...", "content": "file1.txt\nfile2.txt"},
    {"role": "assistant", "content": "Done."}
  ],
  "tools": [...],
  "metadata": {}
}
```

Multiple snapshots may exist per `conversation_id` (one per `conversation_idx`); only the
highest-indexed snapshot per conversation is used.

### Converting to flat-row format

The following commands convert each source snapshot file to the flat-row format required by the benchmarker.
Run from the repo root:

```bash
# First argument: input snapshot JSONL; second argument: output flat-row JSONL
python scripts/convert_agentic_snapshot.py \
    /path/to/agentic_coding_dataset.jsonl \
    examples/09_MultiTurn/datasets/agentic_coding_flat.jsonl \
    --verify

python scripts/convert_agentic_snapshot.py \
    /path/to/agentic_workflow_dataset.jsonl \
    examples/09_MultiTurn/datasets/agentic_workflow_flat.jsonl \
    --verify
```

The `datasets/` directory under `examples/09_MultiTurn/` is a placeholder; run the conversion
commands above to populate it before benchmarking.

The `--verify` flag cross-checks every client turn's message history against the source snapshot
and exits with code 1 if any mismatch is found. The script also:

- Collapses consecutive `user` messages into one (keeps turn sequencing clean)
- Merges consecutive `tool` messages for the same assistant dispatch into a single row with a
  `tool_results` list (so all parallel results are sent together in one API call)

### Flat-row format after conversion

The extra fields supported beyond plain user/assistant:

| Row role                         | Extra fields                                                       |
| -------------------------------- | ------------------------------------------------------------------ |
| `assistant` with tool calls      | `tool_calls: [{id, type, function: {name, arguments}}]`            |
| `tool` single result             | `tool_call_id: <str>`, `content: <str>`                            |
| `tool` parallel results (merged) | `tool_results: [{tool_call_id, content}, ...]`                     |
| `user` or `tool` turns           | `tools: [...]` (OpenAI tool definitions forwarded to the endpoint) |

Example rows from a converted agentic dataset:

```jsonl
{"conversation_id": "sim_001", "turn": 1, "role": "user", "content": "Fix the bug in foo.py", "system": "You are a coding agent.", "tools": [...]}
{"conversation_id": "sim_001", "turn": 2, "role": "assistant", "tool_calls": [{"id": "functions.bash:0", "type": "function", "function": {"name": "bash", "arguments": "{\"cmd\": \"cat foo.py\"}"}}]}
{"conversation_id": "sim_001", "turn": 3, "role": "tool", "tool_call_id": "functions.bash:0", "content": "def foo():\n    return 1/0", "tools": [...]}
{"conversation_id": "sim_001", "turn": 4, "role": "assistant", "content": "The bug is a ZeroDivisionError. Here is the fix: ..."}
```

### Running agentic benchmarks

After converting the datasets, update the `path` field in the config files and run:

```bash
inference-endpoint benchmark from-config \
    --config examples/09_MultiTurn/agentic_coding_benchmark.yaml

inference-endpoint benchmark from-config \
    --config examples/09_MultiTurn/agentic_workflow_benchmark.yaml
```

---

## Configuration

### Basic Configuration

```yaml
datasets:
  - name: customer_support
    type: performance
    path: examples/09_MultiTurn/customer_support_conversations.jsonl
    multi_turn:
      turn_timeout_s: 300.0

settings:
  load_pattern:
    type: multi_turn
    target_concurrency: 32 # ← Required for multi_turn load pattern
```

### Concurrency Control

The `target_concurrency` field is **required** for the `multi_turn` load pattern and controls the maximum number of in-flight requests across all conversations:

```yaml
settings:
  load_pattern:
    type: multi_turn
    target_concurrency: 32 # ← Limit to 32 concurrent requests
```

**Behavior**:

- With `target_concurrency`: Limits total in-flight requests across all conversations
- Combines with turn sequencing: Turn N+1 still waits for turn N, AND waits for available slot

**Use cases**:

- **Prevent endpoint overload**: Control request rate to busy endpoints
- **Large-scale testing**: Benchmark 1000+ conversations without overwhelming system
- **Resource management**: Stay within port limits, memory constraints

**Example**: 100 conversations with `target_concurrency: 32`

```
t=0:   Issue first 32 turn-1s (concurrency limit reached)
t=0.5: Turn-1 completes → issue next turn-1 (slot filled)
t=1.0: Turn-1 completes → issue turn-2 of completed conv (slot filled)
...    Maintains ~32 in-flight across all conversations
```

### Turn Timeout

Configure maximum wait time for previous turn completion:

```yaml
multi_turn:
  turn_timeout_s: 300.0 # 5 minutes
```

If a turn times out waiting for the previous turn, it will be skipped and logged as a warning.

## Running Multi-Turn Benchmarks

### Using Configuration File

```bash
inference-endpoint benchmark from-config \
  --config examples/09_MultiTurn/multi_turn_benchmark.yaml
```

### Viewing Results

Multi-turn benchmarks produce both per-turn and per-conversation metrics:

- **Per-turn metrics**: Latency, TTFT, TPOT for each individual turn
- **Per-conversation metrics**: Total conversation latency, conversations per second

Results are stored in the configured `report_dir` with conversation metadata included in the events log (`events.jsonl`).

## Example Datasets

### customer_support_conversations.jsonl

Simple customer support conversations demonstrating basic multi-turn interactions:

- 3 conversations
- 2-4 turns per conversation
- Customer support agent system prompt

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

For agentic datasets, use the conversion script (`scripts/convert_agentic_snapshot.py`) to
produce a properly sequenced flat-row file. The valid agentic sequence is:

```
user -> assistant (tool_calls) -> tool -> [tool | assistant (tool_calls)]* -> assistant -> user -> ...
```

### "Turn timed out waiting for prev turn"

**Cause**: Previous turn took longer than `turn_timeout_s` to complete.

**Fixes**:

- Increase `turn_timeout_s` in configuration
- Check endpoint performance
- Verify endpoint is responding

### Single-turn benchmarks unaffected

Multi-turn logic is only activated when a `multi_turn:` block is present in the dataset configuration. Existing single-turn benchmarks continue to work unchanged with zero performance overhead.

## Future Enhancements

Planned features:

- [ ] Poisson conversation arrival mode implementation
- [ ] Per-conversation metrics in reporting
- [ ] Conversation-level latency percentiles
- [ ] Dynamic conversation branching
