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

import random
import threading
import time
from typing import Any

import pytest
from inference_endpoint import metrics
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import (
    ConversationMode,
    LoadPattern,
    LoadPatternType,
    MultiTurnConfig,
)
from inference_endpoint.load_generator.conversation_manager import ConversationManager
from inference_endpoint.load_generator.sample import SampleEventHandler
from inference_endpoint.load_generator.scheduler import (
    BLOCK_ON_PREVIOUS_TURN,
    MultiTurnScheduler,
    WithoutReplacementSampleOrder,
)


@pytest.fixture
def multi_turn_dataset_metadata() -> dict[str, Any]:
    """Sample multi-turn dataset metadata for 3 conversations with 3 user turns each.

    Uses absolute turn numbering (matching actual dataset format):
    - Odd turns (1, 3, 5): user messages (these are the samples)
    - Even turns (2, 4, 6): assistant responses (not in samples, but part of conversation flow)
    """
    return {
        "samples": [
            # First user turn for each conversation (turn 1)
            {"conversation_id": "conv_001", "turn": 1},
            {"conversation_id": "conv_002", "turn": 1},
            {"conversation_id": "conv_003", "turn": 1},
            # Second user turn for each conversation (turn 3, after assistant turn 2)
            {"conversation_id": "conv_001", "turn": 3},
            {"conversation_id": "conv_002", "turn": 3},
            {"conversation_id": "conv_003", "turn": 3},
            # Third user turn for each conversation (turn 5, after assistant turn 4)
            {"conversation_id": "conv_001", "turn": 5},
            {"conversation_id": "conv_002", "turn": 5},
            {"conversation_id": "conv_003", "turn": 5},
        ],
        "num_conversations": 3,
        "max_turns_per_conv": 5,  # 5 because last user turn is turn 5 (then turn 6 would be assistant)
    }


@pytest.fixture
def multi_turn_config_parallel() -> MultiTurnConfig:
    """Multi-turn config for parallel mode."""
    return MultiTurnConfig(
        enabled=True, mode=ConversationMode.PARALLEL, turn_timeout_s=5.0
    )


@pytest.fixture
def multi_turn_config_sequential() -> MultiTurnConfig:
    """Multi-turn config for sequential mode."""
    return MultiTurnConfig(
        enabled=True, mode=ConversationMode.SEQUENTIAL, turn_timeout_s=5.0
    )


@pytest.fixture
def multi_turn_runtime_settings(random_seed) -> RuntimeSettings:
    """Runtime settings for multi-turn scheduler tests."""
    return RuntimeSettings(
        metric_target=metrics.Throughput(100),
        reported_metrics=[],
        min_duration_ms=1000,
        max_duration_ms=10_000,
        n_samples_from_dataset=9,
        n_samples_to_issue=9,
        min_sample_count=9,
        rng_sched=random.Random(random_seed),
        rng_sample_index=random.Random(random_seed),
        load_pattern=LoadPattern(type=LoadPatternType.MULTI_TURN),
    )


@pytest.fixture
def multi_turn_runtime_settings_sequential(random_seed) -> RuntimeSettings:
    """Runtime settings for sequential multi-turn tests."""
    return RuntimeSettings(
        metric_target=metrics.Throughput(100),
        reported_metrics=[],
        min_duration_ms=1000,
        max_duration_ms=10_000,
        n_samples_from_dataset=9,
        n_samples_to_issue=9,
        min_sample_count=9,
        rng_sched=random.Random(random_seed),
        rng_sample_index=random.Random(random_seed),
        load_pattern=LoadPattern(type=LoadPatternType.MULTI_TURN),
    )


@pytest.fixture
def multi_turn_runtime_settings_with_concurrency(random_seed):
    """Runtime settings with concurrency control."""
    return RuntimeSettings(
        metric_target=metrics.Throughput(100),
        reported_metrics=[],
        min_duration_ms=1000,
        max_duration_ms=10_000,
        n_samples_from_dataset=9,
        n_samples_to_issue=9,
        min_sample_count=9,
        rng_sched=random.Random(random_seed),
        rng_sample_index=random.Random(random_seed),
        load_pattern=LoadPattern(type=LoadPatternType.MULTI_TURN, target_concurrency=2),
    )


@pytest.mark.unit
def test_multi_turn_scheduler_parallel_mode(
    multi_turn_runtime_settings, multi_turn_dataset_metadata, clean_sample_event_hooks
):
    """Test PARALLEL mode: all turn-1 at t=0, then sequence within each conversation."""
    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        multi_turn_runtime_settings,
        WithoutReplacementSampleOrder,
        conversation_manager,
        multi_turn_dataset_metadata,
        multi_turn_config_parallel,
    )

    # Test the schedule generation directly (without blocking)
    schedule = list(scheduler._parallel_schedule())

    # First 3 samples should be turn-1 (all conversations start simultaneously)
    first_three = schedule[:3]
    for s_idx, delay in first_three:
        sample_meta = multi_turn_dataset_metadata["samples"][s_idx]
        assert sample_meta["turn"] == 1
        assert delay == 0  # No delay for turn-1

    # Remaining samples should be turn>1 with BLOCK_ON_PREVIOUS_TURN
    remaining = schedule[3:]
    for s_idx, delay in remaining:
        sample_meta = multi_turn_dataset_metadata["samples"][s_idx]
        assert sample_meta["turn"] > 1
        assert delay == BLOCK_ON_PREVIOUS_TURN


@pytest.mark.unit
def test_multi_turn_scheduler_sequential_mode(
    multi_turn_runtime_settings_sequential,
    multi_turn_dataset_metadata,
    clean_sample_event_hooks,
):
    """Test SEQUENTIAL mode: complete conv1, then conv2, etc."""
    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        multi_turn_runtime_settings_sequential,
        WithoutReplacementSampleOrder,
        conversation_manager,
        multi_turn_dataset_metadata,
        multi_turn_config_sequential,
    )

    # Test the schedule generation directly (without blocking)
    schedule = list(scheduler._sequential_schedule())

    # First sample should be conv_001 turn-1 (first user turn) with delay=0
    s_idx, delay = schedule[0]
    assert (
        multi_turn_dataset_metadata["samples"][s_idx]["conversation_id"] == "conv_001"
    )
    assert multi_turn_dataset_metadata["samples"][s_idx]["turn"] == 1
    assert delay == 0

    # Second sample should be conv_001 turn-3 (second user turn) with BLOCK_ON_PREVIOUS_TURN
    s_idx, delay = schedule[1]
    assert (
        multi_turn_dataset_metadata["samples"][s_idx]["conversation_id"] == "conv_001"
    )
    assert multi_turn_dataset_metadata["samples"][s_idx]["turn"] == 3
    assert delay == BLOCK_ON_PREVIOUS_TURN

    # Third sample should be conv_001 turn-5 (third user turn) with BLOCK_ON_PREVIOUS_TURN
    s_idx, delay = schedule[2]
    assert (
        multi_turn_dataset_metadata["samples"][s_idx]["conversation_id"] == "conv_001"
    )
    assert multi_turn_dataset_metadata["samples"][s_idx]["turn"] == 5
    assert delay == BLOCK_ON_PREVIOUS_TURN

    # Verify all 3 conversations are processed in order
    conv_ids_seen = []
    for s_idx, _ in schedule:
        conv_id = multi_turn_dataset_metadata["samples"][s_idx]["conversation_id"]
        if conv_id not in conv_ids_seen:
            conv_ids_seen.append(conv_id)

    assert conv_ids_seen == ["conv_001", "conv_002", "conv_003"]


@pytest.mark.unit
def test_multi_turn_scheduler_poisson_mode_fallback(
    random_seed, multi_turn_dataset_metadata, clean_sample_event_hooks
):
    """Test POISSON mode falls back to PARALLEL (not yet implemented)."""
    runtime_settings = RuntimeSettings(
        metric_target=metrics.Throughput(100),
        reported_metrics=[],
        min_duration_ms=1000,
        max_duration_ms=10_000,
        n_samples_from_dataset=9,
        n_samples_to_issue=9,
        min_sample_count=9,
        rng_sched=random.Random(random_seed),
        rng_sample_index=random.Random(random_seed),
        load_pattern=LoadPattern(type=LoadPatternType.MULTI_TURN),
    )

    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        runtime_settings,
        WithoutReplacementSampleOrder,
        conversation_manager,
        multi_turn_dataset_metadata,
        multi_turn_config_parallel,
    )

    # Test the schedule generation directly (should fall back to parallel)
    schedule = list(scheduler._poisson_schedule())

    # Should behave like PARALLEL (all turn-1 first)
    first_three = schedule[:3]
    for s_idx, delay in first_three:
        sample_meta = multi_turn_dataset_metadata["samples"][s_idx]
        assert sample_meta["turn"] == 1
        assert delay == 0


@pytest.mark.unit
def test_multi_turn_scheduler_turn_blocking(
    multi_turn_runtime_settings,
    multi_turn_dataset_metadata,
    multi_turn_config_parallel,
    clean_sample_event_hooks,
):
    """Test turn blocking mechanism with actual conversation manager."""
    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        multi_turn_runtime_settings,
        WithoutReplacementSampleOrder,
        conversation_manager,
        multi_turn_dataset_metadata,
        multi_turn_config_parallel,
    )

    # Get iterator
    schedule_iter = iter(scheduler)

    # Consume first 3 samples (all turn-1) and create conversation states
    issued_samples = []
    for _ in range(3):
        s_idx, delay = next(schedule_iter)
        sample_meta = multi_turn_dataset_metadata["samples"][s_idx]
        issued_samples.append((s_idx, sample_meta))

        # Create conversation state and mark turn as issued
        conversation_manager.get_or_create(sample_meta["conversation_id"], None)
        conversation_manager.mark_turn_issued(
            sample_meta["conversation_id"],
            sample_meta["turn"],
            f"User message {sample_meta['turn']}",
        )

    # Next sample should be turn-2 which will block until turn-1 completes
    blocking_event = threading.Event()

    def consume_next_sample():
        """Thread that will block trying to get turn-2."""
        s_idx, delay = next(schedule_iter)
        blocking_event.set()

    consumer_thread = threading.Thread(target=consume_next_sample, daemon=True)
    consumer_thread.start()

    # Give thread time to hit the blocking wait
    time.sleep(0.2)
    assert not blocking_event.is_set()  # Should still be blocked

    # Complete one turn-1
    first_conv_id = issued_samples[0][1]["conversation_id"]
    conversation_manager.mark_turn_complete(first_conv_id, "Assistant response")

    # Thread should now unblock
    assert blocking_event.wait(timeout=2.0)
    consumer_thread.join(timeout=1.0)


@pytest.mark.unit
def test_multi_turn_scheduler_with_concurrency_control(
    multi_turn_runtime_settings_with_concurrency,
    multi_turn_dataset_metadata,
    multi_turn_config_parallel,
    clean_sample_event_hooks,
):
    """Test hybrid scheduler: turn blocking + concurrency control."""
    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        multi_turn_runtime_settings_with_concurrency,
        WithoutReplacementSampleOrder,
        conversation_manager,
        multi_turn_dataset_metadata,
        multi_turn_config_parallel,
    )

    # Verify concurrency control is enabled
    assert scheduler._condition is not None
    assert scheduler._target_concurrency == 2

    # Track issued samples
    issued_count = 0
    issued_lock = threading.Lock()
    max_inflight = 0

    # Track when each sample issues
    issued_events = [threading.Event() for _ in range(9)]

    def issue_worker():
        """Issues samples through scheduler."""
        nonlocal issued_count, max_inflight
        position = 0
        for _s_idx, _delay in scheduler:
            with issued_lock:
                issued_count += 1
                current_inflight = scheduler._inflight
                max_inflight = max(max_inflight, current_inflight)
                assert (
                    current_inflight <= 2
                ), f"Concurrency {current_inflight} exceeded limit 2"
            issued_events[position].set()
            position += 1

    issue_thread = threading.Thread(target=issue_worker, daemon=True)
    issue_thread.start()

    # First 2 samples should issue immediately (turn-1 of first 2 conversations)
    assert issued_events[0].wait(timeout=1.0)
    assert issued_events[1].wait(timeout=1.0)

    # Third sample should be blocked by concurrency limit
    time.sleep(0.2)
    assert not issued_events[2].is_set()

    # Release one slot by invoking the hook
    scheduler._release_slot()

    # Third sample should now issue
    assert issued_events[2].wait(timeout=1.0)

    # Release remaining slots to let all samples complete
    for _ in range(8):  # Release enough slots for remaining samples
        scheduler._release_slot()

    issue_thread.join(timeout=5.0)

    # Verify concurrency was respected
    assert max_inflight <= 2


@pytest.mark.unit
def test_multi_turn_scheduler_hook_based_release(
    multi_turn_runtime_settings_with_concurrency,
    multi_turn_dataset_metadata,
    multi_turn_config_parallel,
    clean_sample_event_hooks,
):
    """Test that completion hook releases concurrency slots."""
    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        multi_turn_runtime_settings_with_concurrency,
        WithoutReplacementSampleOrder,
        conversation_manager,
        multi_turn_dataset_metadata,
        multi_turn_config_parallel,
    )

    # Verify hook is registered
    assert len(SampleEventHandler.complete_hooks) == 1
    assert SampleEventHandler.complete_hooks[0] == scheduler._release_slot

    # Manually invoke hook
    _initial_inflight = 0
    scheduler._inflight = 1
    scheduler._release_slot()

    assert scheduler._inflight == 0


@pytest.mark.unit
def test_multi_turn_scheduler_timeout_handling(
    random_seed,
    multi_turn_dataset_metadata,
    multi_turn_config_parallel,
    clean_sample_event_hooks,
):
    """Test turn timeout handling when previous turn never completes."""
    runtime_settings = RuntimeSettings(
        metric_target=metrics.Throughput(100),
        reported_metrics=[],
        min_duration_ms=1000,
        max_duration_ms=10_000,
        n_samples_from_dataset=9,
        n_samples_to_issue=9,
        min_sample_count=9,
        rng_sched=random.Random(random_seed),
        rng_sample_index=random.Random(random_seed),
        load_pattern=LoadPattern(type=LoadPatternType.MULTI_TURN),
    )

    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        runtime_settings,
        WithoutReplacementSampleOrder,
        conversation_manager,
        multi_turn_dataset_metadata,
        multi_turn_config_parallel,
    )

    schedule_iter = iter(scheduler)

    # Consume first turn (turn-1) and complete all but first conversation
    all_conv_ids = []
    for _ in range(3):
        s_idx, delay = next(schedule_iter)
        sample_meta = multi_turn_dataset_metadata["samples"][s_idx]
        conv_id = sample_meta["conversation_id"]
        all_conv_ids.append(conv_id)

        # Create conversation state and mark turn as issued
        conversation_manager.get_or_create(conv_id, None)
        conversation_manager.mark_turn_issued(
            conv_id, sample_meta["turn"], "User message"
        )

        # Complete all except the first conversation
        if conv_id != all_conv_ids[0]:
            conversation_manager.mark_turn_complete(conv_id, "Assistant response")

    # Try to get turn-2 for first conversation (should timeout and skip)
    # The scheduler will try turn-2 of first conv, timeout, then continue to other conversations
    start = time.time()

    # Collect remaining samples (should skip first conv turn-2 after timeout)
    remaining_samples = []
    try:
        for _ in range(10):  # Try to get more samples (some may be skipped)
            s_idx, delay = next(schedule_iter)
            sample_meta = multi_turn_dataset_metadata["samples"][s_idx]
            remaining_samples.append(sample_meta)
    except StopIteration:
        pass

    elapsed = time.time() - start

    # Should have timed out at least once
    assert elapsed >= 0.25  # At least one timeout occurred

    # First conv turn-2 should have been skipped (not in remaining samples)
    first_conv_turn2_found = any(
        s["conversation_id"] == all_conv_ids[0] and s["turn"] == 2
        for s in remaining_samples
    )
    assert not first_conv_turn2_found  # Should have been skipped due to timeout


@pytest.mark.unit
def test_multi_turn_scheduler_no_concurrency_control(
    multi_turn_runtime_settings,
    multi_turn_dataset_metadata,
    multi_turn_config_parallel,
    clean_sample_event_hooks,
):
    """Test scheduler without concurrency control (unlimited)."""
    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        multi_turn_runtime_settings,
        WithoutReplacementSampleOrder,
        conversation_manager,
        multi_turn_dataset_metadata,
        multi_turn_config_parallel,
    )

    # Verify concurrency control is disabled
    assert scheduler._condition is None
    assert scheduler._target_concurrency is None

    # Test schedule generation (without blocking on turns)
    schedule = list(scheduler._parallel_schedule())
    assert len(schedule) == 9


@pytest.mark.unit
def test_multi_turn_scheduler_complete_conversation_flow(
    multi_turn_runtime_settings,
    multi_turn_dataset_metadata,
    multi_turn_config_parallel,
    clean_sample_event_hooks,
):
    """Test complete flow of one conversation from turn-1 to turn-3."""
    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        multi_turn_runtime_settings,
        WithoutReplacementSampleOrder,
        conversation_manager,
        multi_turn_dataset_metadata,
        multi_turn_config_parallel,
    )

    schedule_iter = iter(scheduler)

    # Track the first conversation
    first_conv_samples = []

    # Get first turn of first conversation
    s_idx, delay = next(schedule_iter)
    sample_meta = multi_turn_dataset_metadata["samples"][s_idx]
    conv_id = sample_meta["conversation_id"]
    first_conv_samples.append(sample_meta)

    # Create conversation state and complete turn 1
    conversation_manager.get_or_create(conv_id, None)
    conversation_manager.mark_turn_issued(conv_id, 1, "Turn 1 message")
    conversation_manager.mark_turn_complete(conv_id, "Turn 1 response")

    # Process other conversations' turn-1 (and create their states)
    other_conv_ids = []
    for _ in range(2):
        s_idx, delay = next(schedule_iter)
        sample_meta = multi_turn_dataset_metadata["samples"][s_idx]
        other_conv_ids.append(sample_meta["conversation_id"])
        conversation_manager.get_or_create(sample_meta["conversation_id"], None)
        conversation_manager.mark_turn_issued(
            sample_meta["conversation_id"], sample_meta["turn"], "User message"
        )
        conversation_manager.mark_turn_complete(
            sample_meta["conversation_id"], "Assistant response"
        )

    # Collect all remaining samples and process them
    while True:
        try:
            s_idx, delay = next(schedule_iter)
            sample_meta = multi_turn_dataset_metadata["samples"][s_idx]

            # Track samples from first conversation
            if sample_meta["conversation_id"] == conv_id:
                first_conv_samples.append(sample_meta)

                # Complete this turn for first conversation
                conversation_manager.mark_turn_issued(
                    conv_id, sample_meta["turn"], f"Turn {sample_meta['turn']} message"
                )
                conversation_manager.mark_turn_complete(
                    conv_id, f"Turn {sample_meta['turn']} response"
                )
            else:
                # Complete turns for other conversations
                conversation_manager.mark_turn_issued(
                    sample_meta["conversation_id"], sample_meta["turn"], "User message"
                )
                conversation_manager.mark_turn_complete(
                    sample_meta["conversation_id"], "Assistant response"
                )
        except StopIteration:
            break

    # Verify we got all 3 user turns of the first conversation (turns 1, 3, 5)
    assert len(first_conv_samples) == 3
    assert first_conv_samples[0]["turn"] == 1
    assert first_conv_samples[1]["turn"] == 3
    assert first_conv_samples[2]["turn"] == 5

    # Verify conversation state
    state = conversation_manager._conversations[conv_id]
    # After turn 5 (user) + turn 6 (assistant), current_turn is 6
    assert state.current_turn == 6
    assert len(state.message_history) == 6  # 3 user + 3 assistant


@pytest.mark.unit
def test_multi_turn_scheduler_multiple_waiters_same_turn(
    multi_turn_runtime_settings, multi_turn_dataset_metadata, clean_sample_event_hooks
):
    """Test multiple threads waiting for same turn (shouldn't happen in practice)."""
    conversation_manager = ConversationManager()
    scheduler1 = MultiTurnScheduler(
        multi_turn_runtime_settings,
        WithoutReplacementSampleOrder,
        conversation_manager,
        multi_turn_dataset_metadata,
    )
    scheduler2 = MultiTurnScheduler(
        multi_turn_runtime_settings,
        WithoutReplacementSampleOrder,
        conversation_manager,
        multi_turn_dataset_metadata,
    )

    # Issue turn-1
    schedule_iter1 = iter(scheduler1)
    s_idx, delay = next(schedule_iter1)
    sample_meta = multi_turn_dataset_metadata["samples"][s_idx]
    conv_id = sample_meta["conversation_id"]

    # Create conversation state and mark turn as issued
    conversation_manager.get_or_create(conv_id, None)
    conversation_manager.mark_turn_issued(conv_id, 1, "User message")

    # Skip to turn-2 in schedule (create states for other conversations)
    for _ in range(2):
        s_idx, delay = next(schedule_iter1)
        sample_meta = multi_turn_dataset_metadata["samples"][s_idx]
        conversation_manager.get_or_create(sample_meta["conversation_id"], None)
        conversation_manager.mark_turn_issued(
            sample_meta["conversation_id"], sample_meta["turn"], "User message"
        )
        conversation_manager.mark_turn_complete(
            sample_meta["conversation_id"], "Assistant response"
        )

    # Two threads try to wait for turn-2
    ready_count = 0
    ready_lock = threading.Lock()

    def wait_for_turn_2(sched_iter):
        nonlocal ready_count
        next(sched_iter)  # This will block waiting for turn-2
        with ready_lock:
            ready_count += 1

    thread1 = threading.Thread(
        target=wait_for_turn_2, args=(schedule_iter1,), daemon=True
    )

    # Second scheduler iterator
    schedule_iter2 = iter(scheduler2)
    for _ in range(3):  # Skip all turn-1
        s_idx, delay = next(schedule_iter2)

    thread2 = threading.Thread(
        target=wait_for_turn_2, args=(schedule_iter2,), daemon=True
    )

    thread1.start()
    thread2.start()

    time.sleep(0.2)

    # Complete turn-1 for first conversation
    conversation_manager.mark_turn_complete(conv_id, "Assistant response")

    # Both threads should wake up
    thread1.join(timeout=1.0)
    thread2.join(timeout=1.0)

    assert ready_count == 2
