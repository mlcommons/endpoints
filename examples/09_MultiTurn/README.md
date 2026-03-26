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

1. Conversations must alternate between "user" and "assistant" roles
2. First turn must be "user" role
3. Turn numbers must be sequential (1, 2, 3, ...)
4. Each conversation must have at least one turn

## Configuration

### Basic Configuration

```yaml
datasets:
  - name: customer_support
    type: performance
    path: examples/multi_turn/customer_support_conversations.jsonl
    format: jsonl
    multi_turn:
      enabled: true
      mode: parallel
      turn_timeout_s: 300.0

settings:
  load_pattern:
    type: multi_turn
```

### Concurrency Control (Optional)

The multi-turn scheduler supports **optional concurrency limiting** to control the maximum number of in-flight requests across all conversations:

```yaml
settings:
  load_pattern:
    type: multi_turn
    target_concurrency: 32  # ← Limit to 32 concurrent requests
```

**Behavior**:
- Without `target_concurrency`: Unlimited concurrency (all turn-1s issue at t=0 in PARALLEL mode)
- With `target_concurrency`: Limits total in-flight requests across all conversations
- Combines with turn sequencing: Turn N+1 still waits for turn N, AND waits for available slot

**Use cases**:
- 🎯 **Prevent endpoint overload**: Control request rate to busy endpoints
- 🎯 **Large-scale testing**: Benchmark 1000+ conversations without overwhelming system
- 🎯 **Resource management**: Stay within port limits, memory constraints

**Example**: 100 conversations with `target_concurrency: 32`
```
t=0:   Issue first 32 turn-1s (concurrency limit reached)
t=0.5: Turn-1 completes → issue next turn-1 (slot filled)
t=1.0: Turn-1 completes → issue turn-2 of completed conv (slot filled)
...    Maintains ~32 in-flight across all conversations
```

### Conversation Modes

#### Parallel Mode (Default)
Issues turn-1 of all conversations simultaneously (or up to concurrency limit), then sequences turns within each conversation.

```yaml
multi_turn:
  enabled: true
  mode: parallel

# Optional: add concurrency control
settings:
  load_pattern:
    type: multi_turn
    target_concurrency: 32
```

**Use case**: Maximum throughput testing with multiple concurrent conversations

#### Sequential Mode
Completes conversation 1, then conversation 2, etc.

```yaml
multi_turn:
  enabled: true
  mode: sequential
```

**Use case**: Controlled testing with one conversation at a time

#### Poisson Mode
Starts conversations with Poisson arrival, sequences turns within each conversation.

```yaml
multi_turn:
  enabled: true
  mode: poisson
  conversations_per_second: 10.0
```

**Use case**: Realistic arrival patterns (not yet implemented)

### Turn Timeout

Configure maximum wait time for previous turn completion:

```yaml
multi_turn:
  enabled: true
  turn_timeout_s: 300.0  # 5 minutes
```

If a turn times out waiting for the previous turn, it will be skipped and logged as a warning.

## Running Multi-Turn Benchmarks

### Using Configuration File

```bash
inference-endpoint benchmark from-config \
  --config examples/multi_turn/multi_turn_benchmark.yaml
```

### Viewing Results

Multi-turn benchmarks produce both per-turn and per-conversation metrics:

- **Per-turn metrics**: Latency, TTFT, TPOT for each individual turn
- **Per-conversation metrics**: Total conversation latency, conversations per second

Results are stored in the configured `report_dir` with conversation metadata included in the events database.

## Example Datasets

### customer_support_conversations.jsonl

Simple customer support conversations demonstrating basic multi-turn interactions:
- 3 conversations
- 2-4 turns per conversation
- Customer support agent system prompt

## Architecture Notes

### Key Components

- **ConversationManager**: Tracks conversation state and message history
- **MultiTurnScheduler**: Enforces turn sequencing within conversations
- **ConversationSample**: Sample with conversation metadata
- **MultiTurnDataset**: Validates and structures multi-turn data

### Turn Sequencing

The system ensures that:
1. Turn N+1 cannot be issued until turn N completes
2. Message history is included in subsequent requests
3. Concurrent conversations are supported (in parallel mode)

### Memory Considerations

Each conversation maintains message history in memory. For large-scale benchmarks with long conversations:
- Memory usage: ~1KB per turn (approximate)
- 1000 conversations × 10 turns = ~10MB

## Troubleshooting

### "Conversation has invalid role sequence"

**Cause**: Conversation doesn't alternate between user and assistant roles.

**Fix**: Ensure dataset follows the alternating pattern:
```
user -> assistant -> user -> assistant -> ...
```

### "Turn timed out waiting for prev turn"

**Cause**: Previous turn took longer than `turn_timeout_s` to complete.

**Fixes**:
- Increase `turn_timeout_s` in configuration
- Check endpoint performance
- Verify endpoint is responding

### Single-turn benchmarks unaffected

Multi-turn logic is only activated when `multi_turn.enabled: true` in the dataset configuration. Existing single-turn benchmarks continue to work unchanged with zero performance overhead.

## Future Enhancements

Planned features:
- [ ] Poisson conversation arrival mode implementation
- [ ] Per-conversation metrics in reporting
- [ ] Conversation-level latency percentiles
- [ ] Support for tool/function calls in conversations
- [ ] Dynamic conversation branching
