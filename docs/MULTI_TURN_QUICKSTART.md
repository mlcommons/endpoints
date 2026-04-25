# Multi-Turn Conversation Benchmarking - Quick Start Guide

## Quick Start in 5 Minutes

### 1. Prepare Your Dataset

Create a JSONL file with your conversations. All rows for a given `conversation_id` must appear
**consecutively** in the file (no interleaving with other conversations):

```jsonl
{"conversation_id": "c1", "turn": 1, "role": "user", "content": "Hello!", "system": "You are a helpful assistant"}
{"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "Hi! How can I help?"}
{"conversation_id": "c1", "turn": 3, "role": "user", "content": "What's 2+2?"}
{"conversation_id": "c1", "turn": 4, "role": "assistant", "content": "2+2 equals 4."}
```

**Rules**:

- Alternate between "user" and "assistant" roles
- Start with "user" role
- Sequential turn numbers (1, 2, 3, ...)
- Same `conversation_id` for all turns in a conversation
- All rows for the same `conversation_id` must be grouped together

### 2. Create Configuration File

Save as `multi_turn_config.yaml`:

```yaml
name: "my-multi-turn-benchmark"
version: "1.0"
type: "online"

model_params:
  name: "your-model-name"
  temperature: 0.7
  max_new_tokens: 256

datasets:
  - name: my_conversations
    type: performance
    path: path/to/your/conversations.jsonl
    multi_turn: # ← Presence of this block enables multi-turn mode
      mode: independent # ← Per-conv pipelines; no cross-conv turn barrier
      turn_timeout_s: 300 # ← Max wait for prev turn

settings:
  load_pattern:
    type: multi_turn # ← Use multi-turn scheduler
    target_concurrency: 32 # ← Required: max concurrent requests

  client:
    workers: 4

endpoint_config:
  endpoints:
    - "http://your-endpoint:8000"
  api_type: openai

report_dir: logs/my_multi_turn_benchmark
```

Results are written to `report_dir` (here: `logs/my_multi_turn_benchmark/`).

### 3. Run Benchmark

```bash
inference-endpoint benchmark from-config --config multi_turn_config.yaml
```

That's it! Your benchmark will now:

- ✅ Enforce turn ordering (turn N+1 waits for turn N)
- ✅ Include conversation history in each request
- ✅ Track per-turn and per-conversation metrics
- ✅ Log all turns with conversation metadata

---

## Understanding Results

After the benchmark completes, check the directory configured via `report_dir`:

### Events Log

The `events.jsonl` file contains one JSON record per line:

- Standard fields: `sample_uuid`, `event_type`, `timestamp_ns`
- **New fields**: `conversation_id`, `turn_number`

Query examples:

```bash
# All events for a specific conversation
grep '"conversation_id": "c1"' logs/my_multi_turn_benchmark/events.jsonl

# With jq for structured output
jq 'select(.conversation_id == "c1") | {conversation_id, turn_number, event_type, timestamp_ns}' \
  logs/my_multi_turn_benchmark/events.jsonl
```

### Metrics

Currently available:

- **Per-turn metrics**: Latency, TTFT, TPOT for each turn
- **Conversation tracking**: All events tagged with conversation_id

_Note: Per-conversation aggregation (e.g., "conversations/sec") is coming in a future update._

---

## Conversation Modes Explained

### Independent Mode (Default)

```yaml
mode: independent
```

**Behavior**:

- Issues turn-1 of ALL conversations at t=0
- Then sequences turns within each conversation independently
- Maximum parallelism and throughput

**Use for**: Realistic production load where short conversations finish while long ones are still running.
For single-conversation debugging, use `mode: independent` with `target_concurrency: 1`.
Note: unlike the plain `ConcurrencyScheduler`, multi-turn + `target_concurrency: 1` still enforces
per-conversation turn ordering — turn N+1 waits for turn N even at concurrency 1.

**Example timeline**:

```
t=0:    conv1-turn1, conv2-turn1, conv3-turn1 (all at once)
t=0.5:  conv1-turn2 (after conv1-turn1 completes)
t=0.7:  conv2-turn2 (after conv2-turn1 completes)
t=0.8:  conv1-turn3 (after conv1-turn2 completes)
...
```

---

## Concurrency Control

`target_concurrency` is **required** for the `multi_turn` load pattern. It limits the maximum number of in-flight requests across all conversations and prevents endpoint overload when many conversations run simultaneously.

```yaml
settings:
  load_pattern:
    type: multi_turn
    target_concurrency: 32 # ← Limit to 32 concurrent requests
```

**Sizing guide**:

- Small (< 50 convs): `target_concurrency: 32`
- Medium (50-500 convs): `target_concurrency: 64`
- Large (500+ convs): `target_concurrency: 96` or higher

---

## Common Configurations

### Recommended: With Concurrency Control

```yaml
multi_turn:
  mode: independent

settings:
  load_pattern:
    type: multi_turn
    target_concurrency: 32 # ← Prevents overload
  client:
    workers: 8

datasets:
  - samples: 100
```

### High Throughput Testing

```yaml
multi_turn:
  mode: independent
  turn_timeout_s: 600

settings:
  load_pattern:
    type: multi_turn
    target_concurrency: 96
  client:
    workers: 16 # More workers for parallel conversations
```

### Long Conversations

```yaml
multi_turn:
  mode: independent
  turn_timeout_s: 1800 # 30 minutes for slow responses

settings:
  load_pattern:
    type: multi_turn
    target_concurrency: 32
```

---

## Troubleshooting

### "Conversation has invalid role sequence"

**Problem**: Your dataset doesn't follow a valid role sequence.

**Fix**: Check your JSONL. Valid sequences:

- Plain chat: `user → assistant → user → assistant → ...`
- Agentic (tool-use): `user → assistant → tool → assistant → tool → ... → user`

Conversations may also end with a `tool` row (the model's response to the final tool call is the benchmark target).

### "Rows for conversation X are not consecutive"

**Problem**: Rows for the same `conversation_id` are interleaved with rows from other conversations.

**Fix**: Sort your JSONL so all rows for each conversation appear together.

### "Turn timed out waiting for prev turn"

**Problem**: Previous turn took longer than `turn_timeout_s`.

**Fixes**:

1. Increase `turn_timeout_s` in config
2. Check if your endpoint is slow or unresponsive
3. Look for errors in the endpoint logs

### Dataset not loading

**Problem**: MultiTurnDataset not recognized.

**Fix**: Ensure `multi_turn:` block is present in the dataset config. The file format
is auto-detected from the `.jsonl` extension — no `format` field is needed:

```yaml
datasets:
  - path: your_file.jsonl
    multi_turn:
      mode: independent
```

---

## Example Datasets

### Simple 2-Turn Conversation

```jsonl
{"conversation_id": "c1", "turn": 1, "role": "user", "content": "Hi"}
{"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "Hello!"}
```

### With System Prompt

```jsonl
{"conversation_id": "c1", "turn": 1, "role": "user", "content": "Who won?", "system": "You are a sports expert"}
{"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "The Lakers won."}
```

### Multiple Conversations

```jsonl
{"conversation_id": "c1", "turn": 1, "role": "user", "content": "Hi"}
{"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "Hello!"}
{"conversation_id": "c2", "turn": 1, "role": "user", "content": "Hey"}
{"conversation_id": "c2", "turn": 2, "role": "assistant", "content": "Hi there!"}
```

### With Model Override

```jsonl
{"conversation_id": "c1", "turn": 1, "role": "user", "content": "Summarize this", "model": "gpt-4"}
{"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "Here's the summary..."}
```

---

## Testing Your Setup

### 1. Use the Example Dataset

```bash
cd examples/09_MultiTurn
inference-endpoint benchmark from-config --config multi_turn_benchmark.yaml
```

### 2. Check the Logs

```bash
cat logs/multi_turn_test/benchmark.log
# Look for: "Turn X of conversation_id issued"
```

### 3. Verify Event Recording

```bash
# List all unique conversation IDs in the events log
jq -r '.conversation_id' logs/multi_turn_test/events.jsonl | sort -u
# Should show your conversation IDs
```

---

## Tips & Best Practices

### Dataset Design

- **Keep conversations realistic**: 2-10 turns typical
- **Test edge cases**: 1-turn conversations, very long conversations
- **Include system prompts**: Helps model understand context

### Performance

- **Workers**: Set `workers` = number of concurrent conversations
- **Timeout**: Set `turn_timeout_s` = 2x your longest expected turn latency
- **Memory**: ~1KB per turn, plan accordingly for large datasets

### Debugging

- **Start small**: Test with 1-2 conversations first
- **Single conversation**: Use `mode: independent` with `target_concurrency: 1`
- **Check events.jsonl**: Verify turn ordering with `jq`

---

## More Information

- **Full Documentation**: See `examples/09_MultiTurn/README.md`
- **Architecture**: See `AGENTS.md` (Multi-Turn section)

---

## Checklist

Before running your first multi-turn benchmark:

- [ ] Dataset follows format (user/assistant alternation, or agentic user→assistant→tool sequences)
- [ ] All rows for each conversation_id are grouped together
- [ ] Config has `multi_turn:` block in the dataset section
- [ ] Config has `load_pattern.type: multi_turn`
- [ ] Endpoint is running and reachable
- [ ] File uses `.jsonl` extension (format is auto-detected)
- [ ] Conversation IDs are unique per conversation
- [ ] Turn numbers are sequential (1, 2, 3, ...)

Happy benchmarking!
