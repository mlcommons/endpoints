# Multi-Turn Conversation Benchmarking - Quick Start Guide

## 🚀 Quick Start in 5 Minutes

### 1. Prepare Your Dataset

Create a JSONL file with your conversations:

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
    format: jsonl
    multi_turn:
      enabled: true # ← Enable multi-turn mode
      mode: parallel # ← All conv turn-1 at once
      turn_timeout_s: 300 # ← Max wait for prev turn

settings:
  load_pattern:
    type: multi_turn # ← Use multi-turn scheduler
    target_concurrency: 32 # ← OPTIONAL: limit concurrent requests

  client:
    workers: 4

endpoint_config:
  endpoints:
    - "http://your-endpoint:8000"
  api_type: openai

report_dir: logs/my_multi_turn_benchmark
```

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

## 📊 Understanding Results

After the benchmark completes, check `logs/my_multi_turn_benchmark/`:

### Events Database

The `events.db` SQLite database includes:

- Standard fields: sample_uuid, event_type, timestamp_ns
- **New fields**: conversation_id, turn_number

Query example:

```sql
SELECT conversation_id, turn_number, event_type, timestamp_ns
FROM events
WHERE conversation_id = 'c1'
ORDER BY turn_number;
```

### Metrics

Currently available:

- **Per-turn metrics**: Latency, TTFT, TPOT for each turn
- **Conversation tracking**: All events tagged with conversation_id

_Note: Per-conversation aggregation (e.g., "conversations/sec") is coming in a future update._

---

## 🎯 Conversation Modes Explained

### Parallel Mode (Default)

```yaml
mode: parallel
```

**Behavior**:

- Issues turn-1 of ALL conversations at t=0
- Then sequences turns within each conversation
- Maximum parallelism and throughput

**Use for**: Stress testing with many concurrent conversations

**Example timeline**:

```
t=0:    conv1-turn1, conv2-turn1, conv3-turn1 (all at once)
t=0.5:  conv1-turn2 (after conv1-turn1 completes)
t=0.7:  conv2-turn2 (after conv2-turn1 completes)
t=0.8:  conv1-turn3 (after conv1-turn2 completes)
...
```

### Sequential Mode

```yaml
mode: sequential
```

**Behavior**:

- Complete conversation 1 entirely
- Then start conversation 2
- One conversation at a time

**Use for**: Controlled testing, debugging, single-user simulation

**Example timeline**:

```
t=0:   conv1-turn1
t=0.5: conv1-turn2
t=1.0: conv1-turn3
t=1.5: conv2-turn1  ← Only after conv1 fully complete
t=2.0: conv2-turn2
...
```

### Poisson Mode (Planned)

```yaml
mode: poisson
conversations_per_second: 10.0
```

**Behavior**:

- Start conversations with Poisson arrival
- Sequence turns within each
- Realistic user arrival patterns

**Status**: Not yet implemented (falls back to PARALLEL)

---

## 🎛️ Concurrency Control (NEW!)

For benchmarks with **> 50 conversations**, use `target_concurrency` to prevent endpoint overload:

```yaml
settings:
  load_pattern:
    type: multi_turn
    target_concurrency: 32 # ← Limit to 32 concurrent requests
```

**Why?** Without this, PARALLEL mode issues ALL turn-1s at once (could be 100+), overwhelming your endpoint.

**Rule of thumb**:

- Small (< 50 convs): No limit needed
- Medium (50-500 convs): `target_concurrency: 32`
- Large (500+ convs): `target_concurrency: 64`

**See**: `HYBRID_SCHEDULER_GUIDE.md` for detailed guide

---

## 🔧 Common Configurations

### Recommended: With Concurrency Control

```yaml
multi_turn:
  enabled: true
  mode: parallel

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
  enabled: true
  mode: parallel
  turn_timeout_s: 600

settings:
  client:
    workers: 16 # More workers for parallel conversations
```

### Controlled Testing

```yaml
multi_turn:
  enabled: true
  mode: sequential
  turn_timeout_s: 300

settings:
  client:
    workers: 1 # Single worker for sequential testing
```

### Long Conversations

```yaml
multi_turn:
  enabled: true
  mode: parallel
  turn_timeout_s: 1800 # 30 minutes for slow responses
```

---

## ❓ Troubleshooting

### "Conversation has invalid role sequence"

**Problem**: Your dataset doesn't alternate between user/assistant.

**Fix**: Check your JSONL - must be: user, assistant, user, assistant, ...

### "Turn timed out waiting for prev turn"

**Problem**: Previous turn took longer than `turn_timeout_s`.

**Fixes**:

1. Increase `turn_timeout_s` in config
2. Check if your endpoint is slow or unresponsive
3. Look for errors in the endpoint logs

### Dataset not loading

**Problem**: MultiTurnDataset not recognized.

**Fix**: Ensure `format: jsonl` is specified in config:

```yaml
datasets:
  - path: your_file.jsonl
    format: jsonl # ← Required for JSONL
```

---

## 📝 Example Datasets

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

## 🧪 Testing Your Setup

### 1. Use the Example Dataset

```bash
cd examples/multi_turn
inference-endpoint benchmark from-config --config multi_turn_benchmark.yaml
```

### 2. Check the Logs

```bash
cat logs/multi_turn_test/benchmark.log
# Look for: "Turn X of conversation_id issued"
```

### 3. Verify Event Recording

```bash
sqlite3 logs/multi_turn_test/events.db
sqlite> SELECT DISTINCT conversation_id FROM events;
# Should show your conversation IDs
```

---

## 💡 Tips & Best Practices

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
- **Use sequential mode**: Easier to debug than parallel
- **Check events.db**: Verify turn ordering in database

---

## 🔗 More Information

- **Full Documentation**: See `examples/multi_turn/README.md`
- **Implementation Details**: See `MULTI_TURN_IMPLEMENTATION_SUMMARY.md`
- **Architecture**: See `AGENTS.md` (Multi-Turn section)

---

## ✅ Checklist

Before running your first multi-turn benchmark:

- [ ] Dataset follows format (alternating user/assistant roles)
- [ ] Config has `multi_turn.enabled: true`
- [ ] Config has `load_pattern.type: multi_turn`
- [ ] Endpoint is running and reachable
- [ ] `format: jsonl` specified for JSONL datasets
- [ ] Conversation IDs are unique per conversation
- [ ] Turn numbers are sequential (1, 2, 3, ...)

Happy benchmarking! 🚀
