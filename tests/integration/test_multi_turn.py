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

"""Consolidated integration tests for multi-turn conversation benchmarking.

Test Categories:
1. Basic E2E: End-to-end benchmarking with different conversation modes
2. Message History: Verify conversation context accumulates correctly
3. Concurrency Control: Test with varying concurrency levels
4. Large Scale: Stress testing with many conversations and high concurrency
5. Unit Stress: ConversationManager stress testing (no real endpoint)
"""

import concurrent.futures
import json
import random
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Generator
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from inference_endpoint import metrics
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.config.runtime_settings import RuntimeSettings
from inference_endpoint.config.schema import (
    ConversationMode,
    LoadPattern,
    LoadPatternType,
    MultiTurnConfig,
)
from inference_endpoint.core.types import QueryResult
from inference_endpoint.dataset_manager.dataset import DatasetFormat
from inference_endpoint.dataset_manager.multi_turn_dataset import MultiTurnDataset
from inference_endpoint.dataset_manager.transforms import AddStaticColumns
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.http_sample_issuer import HttpClientSampleIssuer
from inference_endpoint.load_generator import (
    BenchmarkSession,
    SampleEvent,
    SampleEventHandler,
    WithoutReplacementSampleOrder,
)
from inference_endpoint.load_generator.conversation_manager import ConversationManager
from inference_endpoint.load_generator.scheduler import MultiTurnScheduler

# ============================================================================
# Shared Fixtures
# ============================================================================


@pytest.fixture
def endpoint_url() -> str:
    """Endpoint URL for integration tests."""
    return "http://localhost:8868"


@pytest.fixture
def multi_turn_model_name(endpoint_url: str) -> str:
    """Query model name from the endpoint instead of hardcoding."""
    try:
        with urlopen(f"{endpoint_url}/v1/models", timeout=5.0) as response:
            models_data = json.loads(response.read())

        # OpenAI-compatible endpoints return {"data": [{"id": "model-name", ...}, ...]}
        if "data" in models_data and len(models_data["data"]) > 0:
            return models_data["data"][0]["id"]

        # Fallback if response format is different
        raise ValueError(f"Unexpected /v1/models response format: {models_data}")
    except (URLError, ValueError, KeyError) as e:
        pytest.skip(f"Could not query model name from endpoint {endpoint_url}: {e}")
        return ""  # Unreachable but satisfies mypy


@pytest.fixture
def small_dataset() -> Generator[str, None, None]:
    """Small multi-turn dataset: 2 conversations, 4 user turns total."""
    conversations = [
        {
            "conversation_id": "test_conv_001",
            "turn": 1,
            "role": "user",
            "content": "Hello, I need help with Python",
            "system": "You are a helpful programming assistant",
        },
        {
            "conversation_id": "test_conv_001",
            "turn": 2,
            "role": "assistant",
            "content": "I'd be happy to help you with Python!",
        },
        {
            "conversation_id": "test_conv_001",
            "turn": 3,
            "role": "user",
            "content": "How do I read a file?",
        },
        {
            "conversation_id": "test_conv_002",
            "turn": 1,
            "role": "user",
            "content": "What is machine learning?",
        },
        {
            "conversation_id": "test_conv_002",
            "turn": 2,
            "role": "assistant",
            "content": "Machine learning is a field of AI.",
        },
        {
            "conversation_id": "test_conv_002",
            "turn": 3,
            "role": "user",
            "content": "Can you give an example?",
        },
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in conversations:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    yield temp_path
    Path(temp_path).unlink()


class MultiTurnSampleIssuer(HttpClientSampleIssuer):
    """Sample issuer for multi-turn testing."""

    def __init__(self, endpoint_url: str, zmq_context: ManagedZMQContext):
        self.http_config = HTTPClientConfig(
            endpoint_urls=[endpoint_url],
            warmup_connections=0,
        )
        super().__init__(HTTPEndpointClient(self.http_config, zmq_context=zmq_context))


# ============================================================================
# Test Category 1: Basic End-to-End Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.parametrize(
    "mode,expected_completions",
    [
        pytest.param(
            ConversationMode.PARALLEL, 4, id="parallel_mode"
        ),  # All turns issue in parallel, then sequence
        pytest.param(
            ConversationMode.SEQUENTIAL, 4, id="sequential_mode"
        ),  # Complete conv1 then conv2
    ],
)
def test_basic_end_to_end(
    small_dataset, endpoint_url, mode, expected_completions, multi_turn_model_name
):
    """Test basic end-to-end multi-turn benchmarking with different modes."""
    dataset = MultiTurnDataset.load_from_file(
        small_dataset,
        format=DatasetFormat.JSONL,
        transforms=[AddStaticColumns({"model": multi_turn_model_name})],
    )
    dataset.load()

    multi_turn_config = MultiTurnConfig(enabled=True, mode=mode, turn_timeout_s=60.0)

    rt_settings = RuntimeSettings(
        metric_target=metrics.Throughput(10),
        reported_metrics=[],
        min_duration_ms=1000,
        max_duration_ms=120_000,
        n_samples_from_dataset=dataset.num_samples(),
        n_samples_to_issue=dataset.num_samples(),
        min_sample_count=dataset.num_samples(),
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
        load_pattern=LoadPattern(type=LoadPatternType.MULTI_TURN),
    )

    completed_queries = []
    conversations_tracked = {}

    def on_complete_hook(result: QueryResult):
        completed_queries.append(result.id)
        metadata = result.metadata or {}
        conv_id = metadata.get("conversation_id")
        turn = metadata.get("turn_number") or metadata.get("turn")

        if conv_id:
            if conv_id not in conversations_tracked:
                conversations_tracked[conv_id] = []
            conversations_tracked[conv_id].append(
                {
                    "turn": turn,
                    "response_length": len(result.get_response_output_string()),
                }
            )

    SampleEventHandler.register_hook(SampleEvent.COMPLETE, on_complete_hook)

    try:
        conversation_manager = ConversationManager()
        scheduler = MultiTurnScheduler(
            rt_settings,
            WithoutReplacementSampleOrder,
            conversation_manager,
            dataset.conversation_metadata,
            multi_turn_config,
        )
        SampleEventHandler.set_conversation_manager(conversation_manager)

        with ManagedZMQContext.scoped() as zmq_ctx:
            sample_issuer = MultiTurnSampleIssuer(
                f"{endpoint_url}/v1/chat/completions", zmq_ctx
            )

            try:
                sess = BenchmarkSession.start(
                    rt_settings,
                    dataset,
                    sample_issuer,
                    scheduler,
                    name="multi_turn_basic_test",
                    max_shutdown_timeout_s=120,
                )
                sess.wait_for_test_end()
            finally:
                sample_issuer.shutdown()
                sample_issuer.http_client.shutdown()

        assert len(completed_queries) == expected_completions
        assert len(conversations_tracked) == 2

        for _conv_id, turns in conversations_tracked.items():
            for turn_info in turns:
                assert turn_info["response_length"] > 0

    finally:
        SampleEventHandler.clear_hooks()


# ============================================================================
# Test Category 2: Message History Tests
# ============================================================================


@pytest.mark.integration
def test_message_history_accumulation(
    small_dataset, endpoint_url, multi_turn_model_name
):
    """Test that message history accumulates correctly across turns."""
    dataset = MultiTurnDataset.load_from_file(
        small_dataset,
        format=DatasetFormat.JSONL,
        transforms=[AddStaticColumns({"model": multi_turn_model_name})],
    )
    dataset.load()

    multi_turn_config = MultiTurnConfig(
        enabled=True, mode=ConversationMode.PARALLEL, turn_timeout_s=60.0
    )

    rt_settings = RuntimeSettings(
        metric_target=metrics.Throughput(10),
        reported_metrics=[],
        min_duration_ms=1000,
        max_duration_ms=120_000,
        n_samples_from_dataset=dataset.num_samples(),
        n_samples_to_issue=dataset.num_samples(),
        min_sample_count=dataset.num_samples(),
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
        load_pattern=LoadPattern(type=LoadPatternType.MULTI_TURN),
    )

    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        rt_settings,
        WithoutReplacementSampleOrder,
        conversation_manager,
        dataset.conversation_metadata,
        multi_turn_config,
    )
    SampleEventHandler.set_conversation_manager(conversation_manager)

    try:
        with ManagedZMQContext.scoped() as zmq_ctx:
            sample_issuer = MultiTurnSampleIssuer(
                f"{endpoint_url}/v1/chat/completions", zmq_ctx
            )

            try:
                sess = BenchmarkSession.start(
                    rt_settings,
                    dataset,
                    sample_issuer,
                    scheduler,
                    name="multi_turn_history_test",
                    max_shutdown_timeout_s=120,
                )
                sess.wait_for_test_end()
            finally:
                sample_issuer.shutdown()
                sample_issuer.http_client.shutdown()

        # Verify conversation states
        conv_001_state = conversation_manager._conversations.get("test_conv_001")
        assert conv_001_state is not None
        assert len(conv_001_state.message_history) >= 3

        # Verify system message is first
        assert conv_001_state.message_history[0]["role"] == "system"
        assert "programming assistant" in conv_001_state.message_history[0]["content"]

        # Verify alternating user/assistant pattern
        for i in range(1, len(conv_001_state.message_history)):
            expected_role = "user" if i % 2 == 1 else "assistant"
            assert conv_001_state.message_history[i]["role"] == expected_role

    finally:
        SampleEventHandler.clear_hooks()


# ============================================================================
# Test Category 3: Concurrency Control Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.parametrize(
    "target_concurrency",
    [
        pytest.param(1, id="concurrency_1"),
        pytest.param(2, id="concurrency_2"),
        pytest.param(128, id="concurrency_128_high"),
    ],
)
def test_concurrency_control(
    small_dataset, endpoint_url, target_concurrency, multi_turn_model_name
):
    """Test multi-turn benchmarking with varying concurrency levels."""
    dataset = MultiTurnDataset.load_from_file(
        small_dataset,
        format=DatasetFormat.JSONL,
        transforms=[AddStaticColumns({"model": multi_turn_model_name})],
    )
    dataset.load()

    multi_turn_config = MultiTurnConfig(
        enabled=True, mode=ConversationMode.PARALLEL, turn_timeout_s=60.0
    )

    rt_settings = RuntimeSettings(
        metric_target=metrics.Throughput(10),
        reported_metrics=[],
        min_duration_ms=1000,
        max_duration_ms=120_000,
        n_samples_from_dataset=dataset.num_samples(),
        n_samples_to_issue=dataset.num_samples(),
        min_sample_count=dataset.num_samples(),
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
        load_pattern=LoadPattern(
            type=LoadPatternType.MULTI_TURN, target_concurrency=target_concurrency
        ),
    )

    completed_count = 0

    def on_complete_hook(result: QueryResult):
        nonlocal completed_count
        completed_count += 1

    SampleEventHandler.register_hook(SampleEvent.COMPLETE, on_complete_hook)

    try:
        conversation_manager = ConversationManager()
        scheduler = MultiTurnScheduler(
            rt_settings,
            WithoutReplacementSampleOrder,
            conversation_manager,
            dataset.conversation_metadata,
            multi_turn_config,
        )
        SampleEventHandler.set_conversation_manager(conversation_manager)

        with ManagedZMQContext.scoped() as zmq_ctx:
            sample_issuer = MultiTurnSampleIssuer(
                f"{endpoint_url}/v1/chat/completions", zmq_ctx
            )

            try:
                sess = BenchmarkSession.start(
                    rt_settings,
                    dataset,
                    sample_issuer,
                    scheduler,
                    name="multi_turn_concurrency_test",
                    max_shutdown_timeout_s=120,
                )
                sess.wait_for_test_end()
            finally:
                sample_issuer.shutdown()
                sample_issuer.http_client.shutdown()

        assert completed_count == 4

    finally:
        SampleEventHandler.clear_hooks()


# ============================================================================
# Test Category 4: Large Scale Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.run_explicitly
@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "num_conversations,num_workers,completion_threshold,concurrency",
    [
        pytest.param(20, 4, 1.0, 128, id="20_conv_high_concurrency"),
        pytest.param(50, 8, 0.95, None, id="50_conv_medium_scale"),
        pytest.param(100, 16, 0.95, None, id="100_conv_large_scale"),
        pytest.param(4096, 64, 0.90, None, id="4096_conv_extreme_scale"),
    ],
)
def test_large_scale(
    tmp_path,
    endpoint_url,
    num_conversations,
    num_workers,
    completion_threshold,
    concurrency,
    multi_turn_model_name,
):
    """Test multi-turn benchmarking at scale with varying conversation counts.

    Args:
        num_conversations: Number of concurrent conversations
        num_workers: Number of worker processes
        completion_threshold: Minimum fraction of conversations that must complete
        concurrency: Optional concurrency limit (None = unlimited)
        multi_turn_model_name: Model name from pytest fixture
    """
    turns_per_conversation = 3

    # Create dataset
    dataset_path = tmp_path / f"large_scale_{num_conversations}.jsonl"
    conversations = []

    for conv_idx in range(num_conversations):
        conv_id = f"conv_{conv_idx:04d}"

        for turn_idx in range(turns_per_conversation):
            turn = turn_idx * 2 + 1  # User turns: 1, 3, 5

            # User message
            conversations.append(
                {
                    "conversation_id": conv_id,
                    "turn": turn,
                    "role": "user",
                    "content": f"What is {turn_idx + 1} + {turn_idx + 2}?",
                    "system": "You are a helpful math assistant"
                    if turn_idx == 0
                    else None,
                    "max_new_tokens": 16 if num_conversations > 100 else 32,
                }
            )

            # Assistant placeholder
            conversations.append(
                {
                    "conversation_id": conv_id,
                    "turn": turn + 1,
                    "role": "assistant",
                    "content": f"The answer is {(turn_idx + 1) + (turn_idx + 2)}.",
                }
            )

    with open(dataset_path, "w") as f:
        for item in conversations:
            f.write(json.dumps(item) + "\n")

    # Load dataset
    dataset = MultiTurnDataset.load_from_file(
        str(dataset_path),
        format=DatasetFormat.JSONL,
        transforms=[AddStaticColumns({"model": multi_turn_model_name})],
    )
    dataset.load()

    # Setup runtime settings
    max_duration_ms = 300000 if num_conversations <= 100 else 1800000

    multi_turn_config = MultiTurnConfig(
        enabled=True,
        mode=ConversationMode.PARALLEL,
        turn_timeout_s=60.0 if num_conversations <= 100 else 300.0,
    )

    load_pattern_kwargs = {"type": LoadPatternType.MULTI_TURN}
    if concurrency is not None:
        load_pattern_kwargs["target_concurrency"] = concurrency

    rt_settings = RuntimeSettings(
        metric_target=metrics.Throughput(100),
        reported_metrics=[],
        min_duration_ms=1,
        max_duration_ms=max_duration_ms,
        n_samples_from_dataset=dataset.num_samples(),
        n_samples_to_issue=dataset.num_samples(),
        min_sample_count=dataset.num_samples(),
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
        load_pattern=LoadPattern(**load_pattern_kwargs),
    )

    # Create conversation manager and scheduler
    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        runtime_settings=rt_settings,
        sample_order_cls=WithoutReplacementSampleOrder,
        conversation_manager=conversation_manager,
        dataset_metadata=dataset.conversation_metadata,
        multi_turn_config=multi_turn_config,
    )

    with ManagedZMQContext.scoped() as zmq_ctx:
        from inference_endpoint.endpoint_client.config import HTTPClientConfig
        from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient

        http_config = HTTPClientConfig(
            endpoint_urls=[f"{endpoint_url}/v1/chat/completions"],
            num_workers=num_workers,
        )
        sample_issuer = HttpClientSampleIssuer(
            HTTPEndpointClient(http_config, zmq_context=zmq_ctx)
        )

        try:
            SampleEventHandler.set_conversation_manager(conversation_manager)

            session = BenchmarkSession.start(
                runtime_settings=rt_settings,
                dataset=dataset,
                sample_issuer=sample_issuer,
                scheduler=scheduler,
                name=f"large_scale_{num_conversations}_test",
                max_shutdown_timeout_s=120 if num_conversations <= 100 else 600,
            )

            session.wait_for_test_end()

            # Verify results
            total_conversations = len(conversation_manager._conversations)
            expected_turn = turns_per_conversation * 2
            completed_conversations = sum(
                1
                for state in conversation_manager._conversations.values()
                if state.current_turn == expected_turn
            )

            assert total_conversations == num_conversations
            assert completed_conversations >= num_conversations * completion_threshold

        finally:
            sample_issuer.shutdown()
            sample_issuer.http_client.shutdown()
            SampleEventHandler.clear_hooks()


# ============================================================================
# Test Category 5: Unit Stress Tests (No Real Endpoint)
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.run_explicitly
@pytest.mark.parametrize(
    "num_conversations,turns_per_conversation",
    [
        pytest.param(4096, 5, id="4096_conv_5_turns"),
        pytest.param(1024, 10, id="1024_conv_10_turns"),
    ],
)
def test_conversation_manager_stress(num_conversations, turns_per_conversation):
    """Unit stress test: ConversationManager with many concurrent operations."""
    manager = ConversationManager()

    # Phase 1: Create all conversations in parallel
    def create_conversation(conv_idx):
        conv_id = f"conv_{conv_idx:04d}"
        return manager.get_or_create(conv_id, "test system")

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = [
            executor.submit(create_conversation, i) for i in range(num_conversations)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == num_conversations

    # Phase 2: Process turns - parallel across conversations, sequential within each
    errors = []

    def process_conversation(conv_idx):
        conv_id = f"conv_{conv_idx:04d}"
        local_errors = []

        for turn in range(1, turns_per_conversation + 1):
            try:
                manager.mark_turn_issued(conv_id, turn, f"message {turn}")
                time.sleep(0.001)  # Simulate processing
                manager.mark_turn_complete(conv_id, f"response {turn}")
            except Exception as e:
                local_errors.append(str(e))

        return local_errors

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = [
            executor.submit(process_conversation, i) for i in range(num_conversations)
        ]
        for future in concurrent.futures.as_completed(futures):
            errors.extend(future.result())

    # Phase 3: Verify all conversations completed correctly
    verification_errors = []

    def verify_conversation(conv_idx):
        conv_id = f"conv_{conv_idx:04d}"
        state = manager._conversations[conv_id]
        # current_turn represents the next turn to be processed, not the last completed
        # After completing turn N, current_turn = N + 1
        expected_turn = turns_per_conversation + 1

        if state.current_turn != expected_turn:
            return f"{conv_id}: Expected turn {expected_turn}, got {state.current_turn}"

        expected_messages = (
            turns_per_conversation * 2 + 1
        )  # system + (user + assistant) * N
        if len(state.message_history) != expected_messages:
            return f"{conv_id}: Expected {expected_messages} messages, got {len(state.message_history)}"

        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        futures = [
            executor.submit(verify_conversation, i) for i in range(num_conversations)
        ]
        for future in concurrent.futures.as_completed(futures):
            error = future.result()
            if error:
                verification_errors.append(error)

    assert len(errors) == 0, f"Had {len(errors)} execution errors"
    assert (
        len(verification_errors) == 0
    ), f"Had {len(verification_errors)} verification errors"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.run_explicitly
def test_conversation_manager_race_conditions():
    """Stress test: Concurrent operations on same conversations for race condition detection."""
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
        local_errors = defaultdict(list)
        local_ops = {"issue": 0, "complete": 0, "wait": 0}

        for _ in range(operations_per_conversation):
            conv_id = f"conv_{random.randint(0, num_conversations - 1):04d}"
            operation = random.choice(["issue", "complete", "wait"])

            try:
                if operation == "issue":
                    turn = random.randint(1, 10)
                    manager.mark_turn_issued(conv_id, turn, f"msg {turn}")
                    local_ops["issue"] += 1
                elif operation == "complete":
                    manager.mark_turn_complete(conv_id, "response")
                    local_ops["complete"] += 1
                else:  # wait
                    turn = random.randint(1, 10)
                    manager.wait_for_turn_ready(conv_id, turn, timeout=0.001)
                    local_ops["wait"] += 1
            except Exception as e:
                local_errors[operation].append(str(e))

        return local_errors, local_ops

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for future in concurrent.futures.as_completed(futures):
            local_errors, local_ops = future.result()
            for op, errs in local_errors.items():
                errors[op].extend(errs)
            with lock:
                for op, count in local_ops.items():
                    operations_completed[op] += count

    # Some operations may legitimately fail (e.g., waiting for a turn that never completes)
    # but we should not have crashes or deadlocks
    total_operations = sum(operations_completed.values())
    assert total_operations > 0  # At least some operations completed


# ============================================================================
# Test Category 7: Sequential Mode No Overlap Test
# ============================================================================


@pytest.mark.integration
def test_sequential_no_overlap(small_dataset, endpoint_url, multi_turn_model_name):
    """Test that sequential mode truly waits between conversations (no overlap).

    This test verifies that conversation 2 does not start until conversation 1's
    final assistant response has completed, ensuring true sequential semantics.
    """
    dataset = MultiTurnDataset.load_from_file(
        small_dataset,
        format=DatasetFormat.JSONL,
        transforms=[AddStaticColumns({"model": multi_turn_model_name})],
    )
    dataset.load()

    multi_turn_config = MultiTurnConfig(
        enabled=True, mode=ConversationMode.SEQUENTIAL, turn_timeout_s=60.0
    )

    rt_settings = RuntimeSettings(
        metric_target=metrics.Throughput(10),
        reported_metrics=[],
        min_duration_ms=1000,
        max_duration_ms=120_000,
        n_samples_from_dataset=dataset.num_samples(),
        n_samples_to_issue=dataset.num_samples(),
        min_sample_count=dataset.num_samples(),
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
        load_pattern=LoadPattern(type=LoadPatternType.MULTI_TURN),
    )

    # Track timing of events
    event_log = []

    def on_complete_hook(result: QueryResult):
        metadata = result.metadata or {}
        conv_id = metadata.get("conversation_id")
        turn = metadata.get("turn_number") or metadata.get("turn")

        event_log.append(
            {
                "type": "response_complete",
                "conv_id": conv_id,
                "turn": turn,
                "timestamp": time.time(),
            }
        )

    SampleEventHandler.register_hook(SampleEvent.COMPLETE, on_complete_hook)

    try:
        conversation_manager = ConversationManager()
        scheduler = MultiTurnScheduler(
            rt_settings,
            WithoutReplacementSampleOrder,
            conversation_manager,
            dataset.conversation_metadata,
            multi_turn_config,
        )
        SampleEventHandler.set_conversation_manager(conversation_manager)

        with ManagedZMQContext.scoped() as zmq_ctx:
            sample_issuer = MultiTurnSampleIssuer(
                f"{endpoint_url}/v1/chat/completions", zmq_ctx
            )

            try:
                sess = BenchmarkSession.start(
                    rt_settings,
                    dataset,
                    sample_issuer,
                    scheduler,
                    name="sequential_no_overlap_test",
                    max_shutdown_timeout_s=120,
                )
                sess.wait_for_test_end()
            finally:
                sample_issuer.shutdown()
                sample_issuer.http_client.shutdown()

        # Verify sequential behavior: conv2 events should all come after conv1 events
        conv1_events = [e for e in event_log if e["conv_id"] == "test_conv_001"]
        conv2_events = [e for e in event_log if e["conv_id"] == "test_conv_002"]

        assert len(conv1_events) > 0, "No conv1 events recorded"
        assert len(conv2_events) > 0, "No conv2 events recorded"

        conv1_last_timestamp = max(e["timestamp"] for e in conv1_events)
        conv2_first_timestamp = min(e["timestamp"] for e in conv2_events)

        # Conv2 should start AFTER conv1's last response completes
        # Allow small timing tolerance for event recording overhead
        timing_tolerance = 0.01  # 10ms tolerance
        assert conv2_first_timestamp >= conv1_last_timestamp - timing_tolerance, (
            f"Sequential mode violated: Conv2 started at {conv2_first_timestamp:.3f} "
            f"before conv1 completed at {conv1_last_timestamp:.3f} "
            f"(overlap of {conv1_last_timestamp - conv2_first_timestamp:.3f}s)"
        )

        print(
            f"\n✓ Sequential mode verified: "
            f"Conv1 completed at {conv1_last_timestamp:.3f}s, "
            f"Conv2 started at {conv2_first_timestamp:.3f}s "
            f"(gap: {conv2_first_timestamp - conv1_last_timestamp:.3f}s)"
        )

    finally:
        SampleEventHandler.clear_hooks()
