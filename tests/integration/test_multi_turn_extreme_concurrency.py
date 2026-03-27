# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extreme concurrency stress test for multi-turn conversations."""

import concurrent.futures
import time
from collections import defaultdict

import pytest

# NOTE: Full integration test disabled - requires benchmark infrastructure
# Keeping unit tests below for ConversationManager stress testing


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.run_explicitly
def test_conversation_manager_extreme_concurrency_unit():
    """Unit test: ConversationManager with 4096 concurrent operations.

    Tests thread safety and scalability without hitting real endpoint.
    """
    from inference_endpoint.load_generator.conversation_manager import (
        ConversationManager,
    )

    print("\n=== Testing ConversationManager with 4096 conversations ===")

    manager = ConversationManager()
    num_conversations = 4096
    turns_per_conversation = 5

    # Phase 1: Create all conversations in parallel
    print("\nPhase 1: Creating 4096 conversations in parallel...")
    start_create = time.time()

    def create_conversation(conv_idx):
        conv_id = f"conv_{conv_idx:04d}"
        state = manager.get_or_create(conv_id, "test system")
        return conv_id, state

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = [
            executor.submit(create_conversation, i) for i in range(num_conversations)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    create_time = time.time() - start_create
    print(f"Created {len(results)} conversations in {create_time:.2f}s")
    print(f"Rate: {len(results) / create_time:.0f} conversations/sec")

    # Phase 2: Process turns - parallel across conversations, sequential within each
    print(f"\nPhase 2: Processing {turns_per_conversation} turns per conversation...")
    print("Mode: Parallel across conversations, sequential within each conversation")

    errors = []
    turn_times = []

    def process_conversation(conv_idx):
        """Process all turns for one conversation sequentially."""
        conv_id = f"conv_{conv_idx:04d}"
        local_times = []
        local_errors = []

        for turn in range(1, turns_per_conversation + 1):
            try:
                turn_start = time.time()

                # Issue turn
                manager.mark_turn_issued(conv_id, turn, f"message {turn}")

                # Simulate processing delay (1-5ms)
                time.sleep(0.001 + (hash(conv_id + str(turn)) % 5) / 1000.0)

                # Complete turn
                manager.mark_turn_complete(conv_id, f"response {turn}")

                turn_time = time.time() - turn_start
                local_times.append(turn_time)
            except Exception as e:
                local_errors.append(str(e))

        return local_times, local_errors

    print(f"Total turn operations: {num_conversations * turns_per_conversation}")

    start_turns = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        futures = [
            executor.submit(process_conversation, i) for i in range(num_conversations)
        ]

        for future in concurrent.futures.as_completed(futures):
            times, errs = future.result()
            turn_times.extend(times)
            errors.extend(errs)

    turns_time = time.time() - start_turns

    print(f"\nCompleted {len(turn_times)} turns in {turns_time:.2f}s")
    print(f"Rate: {len(turn_times) / turns_time:.0f} turns/sec")
    print(f"Errors: {len(errors)}")

    if turn_times:
        print("\nTurn latency statistics:")
        print(f"  Min: {min(turn_times) * 1000:.2f}ms")
        print(f"  Max: {max(turn_times) * 1000:.2f}ms")
        print(f"  Mean: {sum(turn_times) / len(turn_times) * 1000:.2f}ms")
        print(f"  P50: {sorted(turn_times)[len(turn_times) // 2] * 1000:.2f}ms")
        print(f"  P95: {sorted(turn_times)[int(len(turn_times) * 0.95)] * 1000:.2f}ms")
        print(f"  P99: {sorted(turn_times)[int(len(turn_times) * 0.99)] * 1000:.2f}ms")

    # Phase 3: Verify state correctness
    print("\nPhase 3: Verifying conversation states...")

    verification_errors = []

    def verify_conversation(conv_idx):
        conv_id = f"conv_{conv_idx:04d}"
        state = manager._conversations.get(conv_id)

        if state is None:
            return f"{conv_id}: State not found"

        expected_turn = turns_per_conversation + 1
        if state.current_turn != expected_turn:
            return f"{conv_id}: Expected turn {expected_turn}, got {state.current_turn}"

        expected_messages = (
            turns_per_conversation * 2 + 1
        )  # system + (user + assistant) * N
        if len(state.message_history) != expected_messages:
            return (
                f"{conv_id}: Expected {expected_messages} messages, "
                f"got {len(state.message_history)}"
            )

        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = [
            executor.submit(verify_conversation, i) for i in range(num_conversations)
        ]
        for future in concurrent.futures.as_completed(futures):
            error = future.result()
            if error:
                verification_errors.append(error)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nConversations: {num_conversations}")
    print(f"Turns per conversation: {turns_per_conversation}")
    print(f"Total turn operations: {len(turn_times)}")
    print(f"Total time: {create_time + turns_time:.2f}s")

    print(f"\nErrors during execution: {len(errors)}")
    if errors[:5]:
        print("First 5 errors:")
        for err in errors[:5]:
            print(f"  - {err}")

    print(f"\nVerification errors: {len(verification_errors)}")
    if verification_errors[:5]:
        print("First 5 verification errors:")
        for err in verification_errors[:5]:
            print(f"  - {err}")

    # Memory usage
    total_conversations = len(manager._conversations)
    print("\nMemory stats:")
    print(f"  Total conversations in manager: {total_conversations}")

    # Sample conversation state size
    if manager._conversations:
        sample_conv = list(manager._conversations.values())[0]
        print(f"  Messages per conversation: {len(sample_conv.message_history)}")

    # Assert success
    assert len(errors) == 0, f"Had {len(errors)} execution errors"
    assert (
        len(verification_errors) == 0
    ), f"Had {len(verification_errors)} verification errors"

    print("\n✅ Extreme concurrency unit test PASSED")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.run_explicitly
def test_conversation_manager_race_conditions_stress():
    """Stress test: Concurrent operations on same conversations.

    Tests race conditions when multiple threads operate on the same
    conversation simultaneously (issue, complete, wait).
    """
    import threading

    from inference_endpoint.load_generator.conversation_manager import (
        ConversationManager,
    )

    print("\n=== Testing race conditions with 1024 conversations, high contention ===")

    manager = ConversationManager()
    num_conversations = 1024
    operations_per_conversation = 100
    num_threads = 128

    # Create all conversations
    for i in range(num_conversations):
        manager.get_or_create(f"conv_{i:04d}", "test")

    errors = defaultdict(list)
    operations_completed = {"issue": 0, "complete": 0, "wait": 0}
    lock = threading.Lock()

    def worker(worker_id):
        """Each worker performs random operations on random conversations."""
        import random

        local_errors = defaultdict(list)
        local_ops = {"issue": 0, "complete": 0, "wait": 0}

        for _ in range(operations_per_conversation):
            conv_idx = random.randint(0, num_conversations - 1)
            conv_id = f"conv_{conv_idx:04d}"
            op_type = random.choice(["issue", "complete", "wait"])

            try:
                if op_type == "issue":
                    turn = random.randint(1, 10)
                    manager.mark_turn_issued(conv_id, turn, f"msg {turn}")
                    local_ops["issue"] += 1

                elif op_type == "complete":
                    manager.mark_turn_complete(conv_id, "response")
                    local_ops["complete"] += 1

                elif op_type == "wait":
                    turn = random.randint(1, 10)
                    manager.wait_for_turn_ready(conv_id, turn, timeout=0.001)
                    local_ops["wait"] += 1

            except Exception as e:
                local_errors[type(e).__name__].append(str(e))

        with lock:
            for op, count in local_ops.items():
                operations_completed[op] += count
            for err_type, err_msgs in local_errors.items():
                errors[err_type].extend(err_msgs)

    print(f"\nStarting {num_threads} threads...")
    print(f"Operations per conversation: {operations_per_conversation}")
    print(f"Total operations: {num_threads * operations_per_conversation}")

    start = time.time()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = time.time() - start

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    total_ops = sum(operations_completed.values())
    print(f"\nTotal operations completed: {total_ops}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Rate: {total_ops / elapsed:.0f} ops/sec")

    print("\nOperations by type:")
    for op_type, count in sorted(operations_completed.items()):
        print(f"  {op_type}: {count}")

    print("\nErrors by type:")
    for err_type, err_msgs in sorted(errors.items()):
        print(f"  {err_type}: {len(err_msgs)} occurrences")
        if err_msgs[:3]:
            for msg in err_msgs[:3]:
                print(f"    - {msg[:100]}")

    # Verify no critical errors
    critical_errors = [e for e in errors.keys() if e not in ["KeyError"]]
    assert len(critical_errors) == 0, f"Critical errors: {critical_errors}"

    print("\n✅ Race condition stress test PASSED")
