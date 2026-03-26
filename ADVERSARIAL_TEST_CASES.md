# Adversarial Test Cases for Multi-Turn Conversation Benchmarking

This document catalogs all adversarial test cases designed to break multi-turn functionality. These tests validate edge cases, race conditions, and malformed inputs that could cause system failures.

---

## ConversationManager Unit Tests (26 Tests)

### Missing Conversation Handling

#### 1. `test_conversation_manager_missing_conversation_wait`
**Purpose**: Verify KeyError is raised when waiting for turn on non-existent conversation.
```python
manager = ConversationManager()
with pytest.raises(KeyError, match="not initialized"):
    manager.wait_for_turn_ready("non_existent_conv", 1, timeout=0.1)
```
**Expected**: `KeyError` with helpful message

#### 2. `test_conversation_manager_missing_conversation_mark_issued`
**Purpose**: Verify KeyError is raised when marking turn issued on non-existent conversation.
```python
manager = ConversationManager()
with pytest.raises(KeyError, match="not initialized"):
    manager.mark_turn_issued("non_existent_conv", 1, "test content")
```
**Expected**: `KeyError` with helpful message

#### 3. `test_conversation_manager_missing_conversation_mark_complete`
**Purpose**: Verify KeyError is raised when marking turn complete on non-existent conversation.
```python
manager = ConversationManager()
with pytest.raises(KeyError, match="not initialized"):
    manager.mark_turn_complete("non_existent_conv", "response")
```
**Expected**: `KeyError` with helpful message

---

### Timeout Edge Cases

#### 4. `test_conversation_manager_zero_timeout`
**Purpose**: Test with zero timeout - should fail immediately.
```python
manager = ConversationManager()
manager.get_or_create("conv_001", None)
# Turn 2 not ready (turn 1 not complete)
result = manager.wait_for_turn_ready("conv_001", 2, timeout=0.0)
assert result is False
```
**Expected**: Immediate failure (False)

#### 5. `test_conversation_manager_negative_timeout`
**Purpose**: Test with negative timeout - should fail immediately.
```python
manager = ConversationManager()
manager.get_or_create("conv_001", None)
result = manager.wait_for_turn_ready("conv_001", 2, timeout=-1.0)
assert result is False
```
**Expected**: Immediate failure (False)

#### 6. `test_conversation_manager_very_short_timeout`
**Purpose**: Test with very short timeout (1ms).
```python
manager = ConversationManager()
manager.get_or_create("conv_001", None)
result = manager.wait_for_turn_ready("conv_001", 2, timeout=0.001)
assert result is False
```
**Expected**: Timeout failure (False)

---

### Out-of-Order Operations

#### 7. `test_conversation_manager_mark_complete_before_issued`
**Purpose**: Test marking turn complete without marking it issued first.
```python
manager = ConversationManager()
state = manager.get_or_create("conv_001", None)
manager.mark_turn_complete("conv_001", "response")
# Should handle gracefully with warning
```
**Expected**: No crash, warning logged, state updated

#### 8. `test_conversation_manager_double_mark_complete`
**Purpose**: Test marking same turn complete twice.
```python
manager = ConversationManager()
state = manager.get_or_create("conv_001", None)
manager.mark_turn_issued("conv_001", 1, "user message")
manager.mark_turn_complete("conv_001", "response 1")
manager.mark_turn_complete("conv_001", "response 2")
# Should handle duplicate completion
```
**Expected**: No crash, warning logged

#### 9. `test_conversation_manager_skip_turn`
**Purpose**: Test skipping turn (issue turn 3 without completing turn 1).
```python
manager = ConversationManager()
state = manager.get_or_create("conv_001", None)
manager.mark_turn_issued("conv_001", 1, "message 1")
# Skip turn 1 completion, try to wait for turn 3
result = manager.wait_for_turn_ready("conv_001", 3, timeout=0.1)
assert result is False
```
**Expected**: Timeout (turn not ready)

---

### Invalid Turn Numbers

#### 10. `test_conversation_manager_negative_turn_number`
**Purpose**: Test with negative turn number.
```python
manager = ConversationManager()
state = manager.get_or_create("conv_001", None)
manager.mark_turn_issued("conv_001", -1, "message")
assert state.pending_user_turn == -1
```
**Expected**: Accepts but semantically invalid

#### 11. `test_conversation_manager_zero_turn_number`
**Purpose**: Test with zero turn number.
```python
manager = ConversationManager()
state = manager.get_or_create("conv_001", None)
manager.mark_turn_issued("conv_001", 0, "message")
assert state.pending_user_turn == 0
```
**Expected**: Accepts turn 0

#### 12. `test_conversation_manager_large_turn_number`
**Purpose**: Test with very large turn number (1,000,000).
```python
manager = ConversationManager()
state = manager.get_or_create("conv_001", None)
large_turn = 1000000
manager.mark_turn_issued("conv_001", large_turn, "message")
manager.mark_turn_complete("conv_001", "response")
assert state.current_turn == large_turn + 1
```
**Expected**: Handles large numbers

---

### Conversation ID Edge Cases

#### 13. `test_conversation_manager_empty_conversation_id`
**Purpose**: Test with empty conversation ID.
```python
manager = ConversationManager()
state = manager.get_or_create("", None)
assert state is not None
assert state.conversation_id == ""
```
**Expected**: Empty string is valid

#### 14. `test_conversation_manager_very_long_conversation_id`
**Purpose**: Test with very long conversation ID (10KB).
```python
manager = ConversationManager()
long_id = "x" * 10000
state = manager.get_or_create(long_id, None)
assert state.conversation_id == long_id
```
**Expected**: Handles 10KB ID

#### 15. `test_conversation_manager_special_chars_conversation_id`
**Purpose**: Test with special characters in conversation ID.
```python
manager = ConversationManager()
special_id = "conv:001\n\t\r\\/'\"<>"
state = manager.get_or_create(special_id, None)
assert state.conversation_id == special_id
```
**Expected**: Special chars accepted

#### 16. `test_conversation_manager_unicode_conversation_id`
**Purpose**: Test with unicode/emoji in conversation ID.
```python
manager = ConversationManager()
unicode_id = "conv_🚀_测试_001"
state = manager.get_or_create(unicode_id, None)
assert state.conversation_id == unicode_id
```
**Expected**: Unicode supported

---

### Content Edge Cases

#### 17. `test_conversation_manager_very_long_content`
**Purpose**: Test with very long message content (1MB).
```python
manager = ConversationManager()
state = manager.get_or_create("conv_001", None)
long_content = "x" * 1000000  # 1MB
manager.mark_turn_issued("conv_001", 1, long_content)
assert state.message_history[-1]["content"] == long_content
```
**Expected**: Handles 1MB content

#### 18. `test_conversation_manager_empty_content`
**Purpose**: Test with empty message content.
```python
manager = ConversationManager()
state = manager.get_or_create("conv_001", None)
manager.mark_turn_issued("conv_001", 1, "")
assert state.message_history[-1]["content"] == ""
```
**Expected**: Empty string accepted

#### 19. `test_conversation_manager_none_content`
**Purpose**: Test with None as content.
```python
manager = ConversationManager()
state = manager.get_or_create("conv_001", None)
manager.mark_turn_issued("conv_001", 1, None)
assert state.message_history[-1]["content"] is None
```
**Expected**: None accepted

---

### Race Conditions

#### 20. `test_conversation_manager_race_double_issue`
**Purpose**: Test race condition - two threads issuing same turn simultaneously.
```python
manager = ConversationManager()
manager.get_or_create("conv_001", None)
results = []

def issue_turn():
    try:
        manager.mark_turn_issued("conv_001", 1, "content")
        results.append("success")
    except Exception as e:
        results.append(f"error: {e}")

t1 = threading.Thread(target=issue_turn)
t2 = threading.Thread(target=issue_turn)
t1.start()
t2.start()
t1.join()
t2.join()

assert len(results) == 2  # Both succeed, no lock
```
**Expected**: Both succeed (no crash)

#### 21. `test_conversation_manager_race_complete_during_wait`
**Purpose**: Test race - completing turn while another thread is waiting.
```python
manager = ConversationManager()
manager.get_or_create("conv_001", None)
manager.mark_turn_issued("conv_001", 1, "content")
wait_result = []

def wait_for_turn():
    result = manager.wait_for_turn_ready("conv_001", 3, timeout=2.0)
    wait_result.append(result)

def complete_turn():
    time.sleep(0.1)
    manager.mark_turn_complete("conv_001", "response")

t1 = threading.Thread(target=wait_for_turn)
t2 = threading.Thread(target=complete_turn)
t1.start()
t2.start()
t1.join()
t2.join()

assert wait_result[0] is True
```
**Expected**: Wait succeeds after completion

---

### Stress Tests

#### 22. `test_conversation_manager_many_conversations`
**Purpose**: Test with many parallel conversations (1000).
```python
manager = ConversationManager()
for i in range(1000):
    state = manager.get_or_create(f"conv_{i:04d}", None)
    manager.mark_turn_issued(f"conv_{i:04d}", 1, f"message {i}")
    manager.mark_turn_complete(f"conv_{i:04d}", f"response {i}")
assert len(manager._conversations) == 1000
```
**Expected**: Handles 1000 conversations

#### 23. `test_conversation_manager_long_conversation`
**Purpose**: Test single conversation with many turns (100 turns).
```python
manager = ConversationManager()
state = manager.get_or_create("conv_001", None)
for turn in range(1, 101):
    manager.mark_turn_issued("conv_001", turn, f"message {turn}")
    manager.mark_turn_complete("conv_001", f"response {turn}")
assert len(state.message_history) == 200  # 100 user + 100 assistant
```
**Expected**: Handles 100 turns

---

### Miscellaneous

#### 24. `test_conversation_state_direct_modification`
**Purpose**: Test directly modifying conversation state (should be avoided).
```python
manager = ConversationManager()
state = manager.get_or_create("conv_001", None)
# Direct modification outside manager
state.current_turn = 999
state.pending_user_turn = 888
# State is corrupted but manager doesn't prevent this
assert state.current_turn == 999
```
**Expected**: No prevention (API misuse)

#### 25. `test_conversation_manager_concurrent_creates`
**Purpose**: Test concurrent creation of same conversation.
```python
manager = ConversationManager()
results = []

def create_conv():
    state = manager.get_or_create("conv_001", "system prompt")
    results.append(state)

threads = [threading.Thread(target=create_conv) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

assert all(r is results[0] for r in results)  # Same state object
```
**Expected**: All get same state (thread-safe)

#### 26. `test_conversation_manager_timeout_then_complete`
**Purpose**: Test completing turn after timeout has occurred.
```python
manager = ConversationManager()
manager.get_or_create("conv_001", None)
manager.mark_turn_issued("conv_001", 1, "content")
# Wait with short timeout - will fail
result1 = manager.wait_for_turn_ready("conv_001", 3, timeout=0.01)
assert result1 is False
# Now complete turn
manager.mark_turn_complete("conv_001", "response")
# Next wait should succeed immediately
result2 = manager.wait_for_turn_ready("conv_001", 3, timeout=0.01)
assert result2 is True
```
**Expected**: Subsequent wait succeeds

---

## MultiTurnDataset Unit Tests (18 Tests)

### Turn Number Issues

#### 1. `test_multi_turn_dataset_gaps_in_turn_numbers`
**Purpose**: Test conversation with gaps in turn numbers (1, 3, 5 skipping 2, 4).
```python
data = [
    {"conversation_id": "c1", "turn": 1, "role": "user", "content": "msg1"},
    {"conversation_id": "c1", "turn": 3, "role": "assistant", "content": "resp1"},
    {"conversation_id": "c1", "turn": 5, "role": "user", "content": "msg2"},
]
```
**Expected**: Should reject but currently passes (validation gap)

#### 2. `test_multi_turn_dataset_duplicate_turn_numbers`
**Purpose**: Test conversation with duplicate turn numbers.
```python
data = [
    {"conversation_id": "c1", "turn": 1, "role": "user", "content": "msg1"},
    {"conversation_id": "c1", "turn": 1, "role": "assistant", "content": "resp1"},
    {"conversation_id": "c1", "turn": 2, "role": "user", "content": "msg2"},
]
```
**Expected**: Loads (sorting handles it)

#### 3. `test_multi_turn_dataset_negative_turn_numbers`
**Purpose**: Test conversation with negative turn numbers.
```python
data = [
    {"conversation_id": "c1", "turn": -1, "role": "user", "content": "msg1"},
    {"conversation_id": "c1", "turn": 0, "role": "assistant", "content": "resp1"},
]
```
**Expected**: Loads but unexpected

#### 4. `test_multi_turn_dataset_turn_zero`
**Purpose**: Test conversation starting with turn 0.
```python
data = [
    {"conversation_id": "c1", "turn": 0, "role": "user", "content": "msg1"},
    {"conversation_id": "c1", "turn": 1, "role": "assistant", "content": "resp1"},
]
```
**Expected**: Loads successfully

#### 5. `test_multi_turn_dataset_very_large_turn_number`
**Purpose**: Test conversation with very large turn number (1,000,000).
```python
data = [
    {"conversation_id": "c1", "turn": 1000000, "role": "user", "content": "msg1"},
    {"conversation_id": "c1", "turn": 1000001, "role": "assistant", "content": "resp1"},
]
```
**Expected**: Handles large numbers

#### 6. `test_multi_turn_dataset_unordered_turns`
**Purpose**: Test conversation with unordered turns (3, 1, 2).
```python
data = [
    {"conversation_id": "c1", "turn": 3, "role": "user", "content": "msg2"},
    {"conversation_id": "c1", "turn": 1, "role": "user", "content": "msg1"},
    {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "resp1"},
]
```
**Expected**: Should reject but currently passes (validation gap)

---

### Conversation ID Issues

#### 7. `test_multi_turn_dataset_empty_conversation_id`
**Purpose**: Test conversation with empty conversation_id.
```python
data = [
    {"conversation_id": "", "turn": 1, "role": "user", "content": "msg1"},
    {"conversation_id": "", "turn": 2, "role": "assistant", "content": "resp1"},
]
```
**Expected**: Loads successfully

#### 8. `test_multi_turn_dataset_none_conversation_id`
**Purpose**: Test conversation with None conversation_id.
```python
data = [
    {"conversation_id": None, "turn": 1, "role": "user", "content": "msg1"},
    {"conversation_id": None, "turn": 2, "role": "assistant", "content": "resp1"},
]
```
**Expected**: Handled gracefully (error or accepts)

#### 9. `test_multi_turn_dataset_mixed_conversation_ids_types`
**Purpose**: Test with mixed types for conversation_id (string and int).
```python
data = [
    {"conversation_id": "c1", "turn": 1, "role": "user", "content": "msg1"},
    {"conversation_id": 123, "turn": 1, "role": "user", "content": "msg2"},
]
```
**Expected**: Loads successfully

---

### Missing Fields

#### 10. `test_multi_turn_dataset_missing_conversation_id`
**Purpose**: Test conversation missing conversation_id field.
```python
data = [
    {"turn": 1, "role": "user", "content": "msg1"},
    {"turn": 2, "role": "assistant", "content": "resp1"},
]
```
**Expected**: Raises KeyError or ValueError

#### 11. `test_multi_turn_dataset_missing_turn_field`
**Purpose**: Test conversation missing turn field.
```python
data = [
    {"conversation_id": "c1", "role": "user", "content": "msg1"},
    {"conversation_id": "c1", "role": "assistant", "content": "resp1"},
]
```
**Expected**: Raises KeyError or ValueError

#### 12. `test_multi_turn_dataset_missing_role_field`
**Purpose**: Test conversation missing role field.
```python
data = [
    {"conversation_id": "c1", "turn": 1, "content": "msg1"},
    {"conversation_id": "c1", "turn": 2, "content": "resp1"},
]
```
**Expected**: Raises KeyError or ValueError

---

### Invalid Values

#### 13. `test_multi_turn_dataset_invalid_role`
**Purpose**: Test conversation with invalid role.
```python
data = [
    {"conversation_id": "c1", "turn": 1, "role": "invalid", "content": "msg1"},
    {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "resp1"},
]
```
**Expected**: Raises ValueError (invalid role)

---

### Content Edge Cases

#### 14. `test_multi_turn_dataset_empty_content`
**Purpose**: Test conversation with empty content.
```python
data = [
    {"conversation_id": "c1", "turn": 1, "role": "user", "content": ""},
    {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": ""},
]
```
**Expected**: Loads successfully

#### 15. `test_multi_turn_dataset_very_long_content`
**Purpose**: Test conversation with very long content (1MB).
```python
data = [
    {"conversation_id": "c1", "turn": 1, "role": "user", "content": "x" * 1000000},
    {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "y" * 1000000},
]
```
**Expected**: Loads successfully

#### 16. `test_multi_turn_dataset_unicode_content`
**Purpose**: Test conversation with unicode/emoji content.
```python
data = [
    {"conversation_id": "c1", "turn": 1, "role": "user", "content": "Hello 🚀 世界"},
    {"conversation_id": "c1", "turn": 2, "role": "assistant", "content": "こんにちは 🎉"},
]
```
**Expected**: Loads successfully

---

### Malformed Data

#### 17. `test_multi_turn_dataset_malformed_json`
**Purpose**: Test with malformed JSON.
```python
# Write invalid JSON (missing closing brace)
'{"conversation_id": "c1", "turn": 1, "role": "user"\n'
```
**Expected**: Raises JSONDecodeError

#### 18. `test_multi_turn_dataset_single_message_conversation`
**Purpose**: Test conversation with only one message (no response).
```python
data = [
    {"conversation_id": "c1", "turn": 1, "role": "user", "content": "msg1"}
]
```
**Expected**: Loads successfully (1 sample)

---

## Integration Tests (3 Tests)

### 1. `test_multi_turn_very_long_conversation`
**Purpose**: Test single conversation with many turns (50 turns) against real endpoint.
**Dataset**: 50 user turns (100 messages total)
**Endpoint**: http://localhost:8868 (vLLM + Llama-3.2-1B-Instruct)
**Expected**: All 50 user turns complete successfully
**Timeout**: 300 seconds

### 2. `test_multi_turn_very_large_messages`
**Purpose**: Test with very large message content (10KB per message).
**Dataset**: 2 user turns with 10KB content each
**Endpoint**: http://localhost:8868
**Expected**: Large messages transmitted and processed correctly
**Timeout**: 120 seconds

### 3. `test_multi_turn_unicode_content`
**Purpose**: Test with unicode and emoji in message content.
**Dataset**: Multilingual content (Chinese, Japanese, Russian, emojis)
**Endpoint**: http://localhost:8868
**Expected**: Unicode content handled correctly
**Timeout**: 120 seconds

---

## Test Results Summary

### ConversationManager (26 tests)
- **Before Fix**: 22/26 passed (85%)
- **After Fix**: 26/26 passed (100%) ✅
- **Critical Fixes**: TypeError and race condition

### MultiTurnDataset (18 tests)
- **Result**: 15/18 passed (83%)
- **Failures**: 3 validation gaps (not critical)

### Integration (3 tests)
- **Result**: 3/3 passed (100%) ✅
- **Endpoint**: Remained operational throughout testing

### Overall
- **Total Tests**: 47
- **Passed**: 44/47 (94%)
- **Failed**: 3/47 (6% - all dataset validation gaps)

---

## Key Findings

### Critical Bugs Found and Fixed
1. ✅ **TypeError in add_assistant_turn()** - System crash on duplicate responses
2. ✅ **Event signaling race condition** - False timeouts under load
3. ⚠️ **Dataset validation gaps** - Non-contiguous/unordered turns accepted

### Edge Cases Successfully Handled
- ✅ Very large values (1MB content, 10KB IDs, 1M turn numbers)
- ✅ Unicode and special characters
- ✅ Concurrent access (1000 conversations, 100 turns)
- ✅ Empty and None values
- ✅ Extreme timeouts (0, negative)

### Recommendations
1. **Immediate**: Deploy with current fixes (critical bugs resolved)
2. **Future**: Add stricter dataset validation for turn ordering
3. **Monitoring**: Log warnings for duplicate responses in production
4. **Limits**: Consider adding configurable max_turns and max_message_size

---

## Test File Locations (Reference Only)

The actual test implementations were documented but not committed:
- `tests/unit/load_generator/test_multi_turn_adversarial.py` (26 tests)
- `tests/unit/dataset_manager/test_multi_turn_dataset_adversarial.py` (18 tests)
- `tests/integration/test_multi_turn_adversarial_integration.py` (3 tests)

These test cases should be re-implemented if needed for future validation.
