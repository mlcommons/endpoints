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

"""Integration tests for multi-turn conversation benchmarking."""

import json
import random
import tempfile
from collections.abc import Generator
from pathlib import Path

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


@pytest.fixture
def multi_turn_test_dataset() -> Generator[str, None, None]:
    """Create multi-turn conversation dataset for testing."""
    conversations = [
        # Conversation 1: 3 user turns
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
            "content": "I'd be happy to help you with Python! What do you need assistance with?",
        },
        {
            "conversation_id": "test_conv_001",
            "turn": 3,
            "role": "user",
            "content": "How do I read a file?",
        },
        # Conversation 2: 2 user turns
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
            "content": "Machine learning is a field of AI that enables systems to learn from data.",
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


@pytest.fixture(params=["http://localhost:8868"])
def endpoint_url(request) -> str:
    """Parameterized endpoint URL fixture."""
    return request.param


class MultiTurnSampleIssuer(HttpClientSampleIssuer):
    """Sample issuer for multi-turn testing."""

    def __init__(self, endpoint_url: str, zmq_context: ManagedZMQContext):
        self.http_config = HTTPClientConfig(
            endpoint_urls=[endpoint_url],
            warmup_connections=0,
        )
        super().__init__(HTTPEndpointClient(self.http_config, zmq_context=zmq_context))


@pytest.mark.integration
@pytest.mark.parametrize(
    "mode,expected_completions",
    [
        (ConversationMode.PARALLEL, 4),  # All user turns should complete
        (ConversationMode.SEQUENTIAL, 4),  # All user turns should complete
    ],
)
def test_multi_turn_end_to_end(
    multi_turn_test_dataset, endpoint_url, mode, expected_completions
):
    """Test end-to-end multi-turn benchmarking with different conversation modes."""
    # Load dataset
    dataset = MultiTurnDataset.load_from_file(
        multi_turn_test_dataset, format=DatasetFormat.JSONL
    )
    dataset.load()

    assert dataset.num_samples() == 4  # 4 user turns total

    # Create multi-turn config
    multi_turn_config = MultiTurnConfig(
        enabled=True,
        mode=mode,
        turn_timeout_s=60.0,
    )

    # Create runtime settings
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

    # Track results
    completed_queries = []
    conversations_tracked = {}

    def on_complete_hook(result: QueryResult):
        """Track completed queries."""
        response_text = result.get_response_output_string()
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
                    "response_length": len(response_text),
                }
            )

    SampleEventHandler.register_hook(SampleEvent.COMPLETE, on_complete_hook)

    try:
        # Create conversation manager and scheduler
        conversation_manager = ConversationManager()
        scheduler = MultiTurnScheduler(
            rt_settings,
            WithoutReplacementSampleOrder,
            conversation_manager,
            dataset.conversation_metadata,
            multi_turn_config,
        )
        SampleEventHandler.set_conversation_manager(conversation_manager)

        # Run benchmark
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
                    name="multi_turn_integration_test",
                    max_shutdown_timeout_s=120,
                )
                sess.wait_for_test_end()

            finally:
                if sample_issuer:
                    sample_issuer.shutdown()
                    sample_issuer.http_client.shutdown()

        # Assertions
        assert (
            len(completed_queries) == expected_completions
        ), f"Expected {expected_completions} completions, got {len(completed_queries)}"

        assert (
            len(conversations_tracked) == 2
        ), f"Expected 2 conversations, tracked {len(conversations_tracked)}"

        # Verify all responses have content
        for conv_id, turns in conversations_tracked.items():
            for turn_info in turns:
                assert (
                    turn_info["response_length"] > 0
                ), f"{conv_id} turn {turn_info['turn']} has empty response"

    finally:
        SampleEventHandler.clear_hooks()


@pytest.mark.integration
def test_multi_turn_message_history_accumulation(multi_turn_test_dataset, endpoint_url):
    """Test that message history accumulates correctly across turns."""
    dataset = MultiTurnDataset.load_from_file(
        multi_turn_test_dataset, format=DatasetFormat.JSONL
    )
    dataset.load()

    multi_turn_config = MultiTurnConfig(
        enabled=True,
        mode=ConversationMode.PARALLEL,
        turn_timeout_s=60.0,
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
                if sample_issuer:
                    sample_issuer.shutdown()
                    sample_issuer.http_client.shutdown()

        # Check conversation states
        conv_001_state = conversation_manager._conversations.get("test_conv_001")
        conv_002_state = conversation_manager._conversations.get("test_conv_002")

        assert conv_001_state is not None, "Conversation 001 not found"
        assert conv_002_state is not None, "Conversation 002 not found"

        # Conversation 001 should have: system + user + assistant (turn 1 complete)
        # If turn 3 also completed: + user + assistant
        assert (
            len(conv_001_state.message_history) >= 3
        ), f"Expected at least 3 messages in conv_001, got {len(conv_001_state.message_history)}"

        # Verify system message is first
        assert conv_001_state.message_history[0]["role"] == "system"
        assert "programming assistant" in conv_001_state.message_history[0]["content"]

        # Verify alternating user/assistant pattern
        for i in range(1, len(conv_001_state.message_history)):
            expected_role = "user" if i % 2 == 1 else "assistant"
            assert conv_001_state.message_history[i]["role"] == expected_role

    finally:
        SampleEventHandler.clear_hooks()


@pytest.mark.integration
@pytest.mark.parametrize("target_concurrency", [1, 2, 128])
def test_multi_turn_with_concurrency_control(
    multi_turn_test_dataset, endpoint_url, target_concurrency
):
    """Test multi-turn benchmarking with concurrency control.

    Tests with varying concurrency levels including high concurrency (128).
    With only 4 user turns in the dataset, high concurrency tests whether
    the system handles cases where concurrency limit exceeds workload size.
    """
    dataset = MultiTurnDataset.load_from_file(
        multi_turn_test_dataset, format=DatasetFormat.JSONL
    )
    dataset.load()

    multi_turn_config = MultiTurnConfig(
        enabled=True,
        mode=ConversationMode.PARALLEL,
        turn_timeout_s=60.0,
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
            type=LoadPatternType.MULTI_TURN,
            target_concurrency=target_concurrency,
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
                if sample_issuer:
                    sample_issuer.shutdown()
                    sample_issuer.http_client.shutdown()

        # Should complete all samples regardless of concurrency
        assert (
            completed_count == 4
        ), f"Expected 4 completions with concurrency={target_concurrency}, got {completed_count}"

    finally:
        SampleEventHandler.clear_hooks()


@pytest.mark.integration
def test_multi_turn_high_concurrency_large_dataset(endpoint_url):
    """Test multi-turn benchmarking with high concurrency (128) on a larger dataset.

    Creates 20 conversations with 3 user turns each (60 total user turns)
    to better stress-test high concurrency.
    """
    # Create larger dataset: 20 conversations × 3 user turns = 60 user turns
    conversations = []
    for conv_idx in range(20):
        conv_id = f"test_conv_{conv_idx:03d}"
        # Turn 1: user
        conversations.append(
            {
                "conversation_id": conv_id,
                "turn": 1,
                "role": "user",
                "content": f"Question {conv_idx}-1",
                "system": "You are a helpful assistant" if conv_idx == 0 else None,
            }
        )
        # Turn 2: assistant
        conversations.append(
            {
                "conversation_id": conv_id,
                "turn": 2,
                "role": "assistant",
                "content": f"Response {conv_idx}-1",
            }
        )
        # Turn 3: user
        conversations.append(
            {
                "conversation_id": conv_id,
                "turn": 3,
                "role": "user",
                "content": f"Question {conv_idx}-2",
            }
        )
        # Turn 4: assistant
        conversations.append(
            {
                "conversation_id": conv_id,
                "turn": 4,
                "role": "assistant",
                "content": f"Response {conv_idx}-2",
            }
        )
        # Turn 5: user
        conversations.append(
            {
                "conversation_id": conv_id,
                "turn": 5,
                "role": "user",
                "content": f"Question {conv_idx}-3",
            }
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for item in conversations:
            f.write(json.dumps(item) + "\n")
        temp_path = f.name

    try:
        # Load dataset
        dataset = MultiTurnDataset.load_from_file(temp_path, format=DatasetFormat.JSONL)
        dataset.load()

        assert dataset.num_samples() == 60  # 20 conversations × 3 user turns

        # Create multi-turn config
        multi_turn_config = MultiTurnConfig(
            enabled=True,
            mode=ConversationMode.PARALLEL,
            turn_timeout_s=60.0,
        )

        # Create runtime settings with high concurrency
        rt_settings = RuntimeSettings(
            metric_target=metrics.Throughput(10),
            reported_metrics=[],
            min_duration_ms=1000,
            max_duration_ms=180_000,  # 3 minutes for larger dataset
            n_samples_from_dataset=dataset.num_samples(),
            n_samples_to_issue=dataset.num_samples(),
            min_sample_count=dataset.num_samples(),
            rng_sched=random.Random(42),
            rng_sample_index=random.Random(42),
            load_pattern=LoadPattern(
                type=LoadPatternType.MULTI_TURN,
                target_concurrency=128,  # High concurrency
            ),
        )

        # Track results
        completed_count = 0
        conversations_completed = set()

        def on_complete_hook(result: QueryResult):
            nonlocal completed_count
            completed_count += 1
            metadata = result.metadata or {}
            conv_id = metadata.get("conversation_id")
            if conv_id:
                conversations_completed.add(conv_id)

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
                        name="multi_turn_high_concurrency_test",
                        max_shutdown_timeout_s=180,
                    )
                    sess.wait_for_test_end()

                finally:
                    if sample_issuer:
                        sample_issuer.shutdown()
                        sample_issuer.http_client.shutdown()

            # Assertions
            assert (
                completed_count == 60
            ), f"Expected 60 completions with high concurrency, got {completed_count}"
            assert (
                len(conversations_completed) == 20
            ), f"Expected 20 conversations, got {len(conversations_completed)}"

        finally:
            SampleEventHandler.clear_hooks()

    finally:
        Path(temp_path).unlink()
