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

"""Real integration tests with actual HTTP requests for multi-turn concurrency."""

import json
import random
import threading
import time

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
from inference_endpoint.dataset_manager.dataset import DatasetFormat
from inference_endpoint.dataset_manager.multi_turn_dataset import MultiTurnDataset
from inference_endpoint.endpoint_client.config import HTTPClientConfig
from inference_endpoint.endpoint_client.http_client import HTTPEndpointClient
from inference_endpoint.endpoint_client.http_sample_issuer import HttpClientSampleIssuer
from inference_endpoint.load_generator import (
    BenchmarkSession,
    SampleEventHandler,
    WithoutReplacementSampleOrder,
)
from inference_endpoint.load_generator.conversation_manager import ConversationManager
from inference_endpoint.load_generator.scheduler import MultiTurnScheduler


class MultiTurnSampleIssuer(HttpClientSampleIssuer):
    """Sample issuer for multi-turn testing."""

    def __init__(
        self, endpoint_url: str, zmq_context: ManagedZMQContext, num_workers: int = 8
    ):
        self.http_config = HTTPClientConfig(
            endpoint_urls=[endpoint_url],
            num_workers=num_workers,
        )
        super().__init__(HTTPEndpointClient(self.http_config, zmq_context=zmq_context))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.run_explicitly
@pytest.mark.timeout(0)  # Disable pytest timeout for long-running tests
@pytest.mark.parametrize(
    "num_conversations,num_workers,completion_threshold",
    [
        (50, 8, 0.95),  # Small scale: 150 requests, 8 workers, 95% threshold
        (100, 16, 0.95),  # Medium scale: 300 requests, 16 workers, 95% threshold
        (4096, 64, 0.90),  # Extreme scale: 12,288 requests, 64 workers, 90% threshold
    ],
    ids=["50_conversations", "100_conversations", "4096_conversations"],
)
def test_concurrent_conversations_real_endpoint(
    tmp_path, num_conversations, num_workers, completion_threshold
):
    """Test concurrent conversations with real HTTP requests to model endpoint.

    This is a TRUE integration test:
    - Creates N conversations × 3 turns = 3N user messages
    - Makes 3N actual HTTP requests to port 8868
    - Uses full benchmark infrastructure (workers, ZMQ, etc.)
    - Measures real end-to-end latency with model inference

    Args:
        tmp_path: Pytest temporary directory fixture
        num_conversations: Number of concurrent conversations to test
        num_workers: Number of worker processes to use
        completion_threshold: Minimum fraction of conversations that must complete
    """
    endpoint_url = "http://localhost:8868/v1/chat/completions"
    turns_per_conversation = 3

    print("\n" + "=" * 70)
    print(f"REAL INTEGRATION TEST: {num_conversations} Concurrent Conversations")
    print("=" * 70)

    # Create dataset
    dataset_path = tmp_path / f"concurrent_{num_conversations}.jsonl"
    conversations = []

    total_requests = num_conversations * turns_per_conversation
    print(
        f"\nCreating dataset: {num_conversations} conversations × {turns_per_conversation} turns"
    )
    print(f"This will result in {total_requests} HTTP requests to the model endpoint")

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

            # Assistant placeholder (dataset format requirement)
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

    print(f"Dataset created: {len(conversations)} lines")
    print(f"User messages (HTTP requests): {total_requests}")

    # Load dataset
    dataset = MultiTurnDataset.load_from_file(
        str(dataset_path), format=DatasetFormat.JSONL
    )
    dataset.load()

    print("\nDataset loaded:")
    print(f"  Total samples: {dataset.num_samples()}")
    print(f"  Conversations: {dataset.conversation_metadata['num_conversations']}")
    print(f"  Max turns: {dataset.conversation_metadata['max_turns_per_conv']}")

    # Setup runtime settings
    max_duration_ms = 300000 if num_conversations <= 100 else 1800000
    turn_timeout_s = 60.0 if num_conversations <= 100 else 300.0

    multi_turn_config = MultiTurnConfig(
        enabled=True,
        mode=ConversationMode.PARALLEL,
        turn_timeout_s=turn_timeout_s,
    )

    runtime_settings = RuntimeSettings(
        metric_target=metrics.Throughput(100),
        reported_metrics=[],
        min_duration_ms=1,
        max_duration_ms=max_duration_ms,
        n_samples_from_dataset=dataset.num_samples(),
        n_samples_to_issue=dataset.num_samples(),
        min_sample_count=dataset.num_samples(),
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
        load_pattern=LoadPattern(type=LoadPatternType.MULTI_TURN),
    )

    print("\nBenchmark configuration:")
    print(f"  Workers: {num_workers}")
    print(f"  Mode: {multi_turn_config.mode}")
    print(f"  Samples to issue: {runtime_settings.n_samples_to_issue}")
    print(f"  Turn timeout: {turn_timeout_s}s")
    print(f"  Max duration: {max_duration_ms / 1000}s")

    # Create conversation manager and scheduler
    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        runtime_settings=runtime_settings,
        sample_order_cls=WithoutReplacementSampleOrder,
        conversation_manager=conversation_manager,
        dataset_metadata=dataset.conversation_metadata,
        multi_turn_config=multi_turn_config,
    )

    print("\n" + "=" * 70)
    print("Starting benchmark...")
    if num_conversations > 100:
        print("⚠️  This will take approximately 10-15 minutes")
    print("=" * 70)

    start_time = time.time()

    with ManagedZMQContext.scoped() as zmq_ctx:
        # Create sample issuer
        sample_issuer = MultiTurnSampleIssuer(
            endpoint_url, zmq_ctx, num_workers=num_workers
        )

        try:
            # Inject conversation manager into event handler
            SampleEventHandler.set_conversation_manager(conversation_manager)

            # Start benchmark session
            session = BenchmarkSession.start(
                runtime_settings=runtime_settings,
                dataset=dataset,
                sample_issuer=sample_issuer,
                scheduler=scheduler,
                name=f"concurrent_{num_conversations}_test",
                max_shutdown_timeout_s=120 if num_conversations <= 100 else 600,
            )

            # Progress monitoring for large tests
            if num_conversations > 100:
                stop_monitor = threading.Event()

                def progress_monitor():
                    while not stop_monitor.is_set():
                        time.sleep(10)
                        if not stop_monitor.is_set():
                            elapsed = time.time() - start_time
                            completed = len(
                                [
                                    s
                                    for s in conversation_manager._conversations.values()
                                    if s.current_turn >= 2
                                ]
                            )
                            print(
                                f"  [{elapsed:.0f}s] Conversations with ≥1 turn complete: {completed}/{num_conversations}"
                            )

                monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
                monitor_thread.start()

            # Wait for completion
            print("\nBenchmark running...")
            session.wait_for_test_end()

            # Stop progress monitor if it was started
            if num_conversations > 100:
                stop_monitor.set()
                monitor_thread.join(timeout=1)

            elapsed_time = time.time() - start_time

            print("\n" + "=" * 70)
            print("BENCHMARK COMPLETED")
            print("=" * 70)

            # Get statistics from conversation manager
            total_conversations = len(conversation_manager._conversations)

            print("\nConversation Manager Statistics:")
            print(f"  Total conversations tracked: {total_conversations}")

            # Verify all conversations completed
            completed_conversations = 0
            incomplete_conversations = []

            # After N user turns (1, 3, 5, ..., 2N-1) and N assistant turns (2, 4, 6, ..., 2N),
            # current_turn should be 2N (last assistant turn completed)
            expected_turn = turns_per_conversation * 2  # After 3 turns: turn 6
            for conv_id, state in conversation_manager._conversations.items():
                if state.current_turn == expected_turn:
                    completed_conversations += 1
                else:
                    incomplete_conversations.append(
                        f"{conv_id}: turn {state.current_turn} (expected {expected_turn})"
                    )

            print(
                f"  Completed conversations: {completed_conversations}/{total_conversations}"
            )

            if incomplete_conversations and len(incomplete_conversations) <= 10:
                print(f"\nIncomplete conversations: {len(incomplete_conversations)}")
                for incomplete in incomplete_conversations[:10]:
                    print(f"    - {incomplete}")

            # Performance metrics
            print("\nPerformance Metrics:")
            print(
                f"  Total time: {elapsed_time:.2f}s ({elapsed_time / 60:.1f} minutes)"
            )
            print(f"  HTTP requests made: {total_requests}")
            print(f"  Throughput: {total_requests / elapsed_time:.2f} requests/sec")
            print(
                f"  Average latency: {elapsed_time / total_requests * 1000:.2f}ms per request"
            )

            # Turn distribution (for large tests)
            if num_conversations > 100:
                turn_distribution = {}
                for state in conversation_manager._conversations.values():
                    turn_distribution[state.current_turn] = (
                        turn_distribution.get(state.current_turn, 0) + 1
                    )

                print("\nTurn distribution:")
                for turn in sorted(turn_distribution.keys()):
                    count = turn_distribution[turn]
                    print(
                        f"    Turn {turn}: {count} conversations ({100 * count / total_conversations:.1f}%)"
                    )

            # Verify correctness
            print("\nVerification:")
            print(f"  ✓ Expected conversations: {num_conversations}")
            print(f"  ✓ Actual conversations: {total_conversations}")
            print(
                f"  ✓ Completed: {completed_conversations} ({100 * completed_conversations / num_conversations:.1f}%)"
            )

            assert (
                total_conversations == num_conversations
            ), f"Expected {num_conversations} conversations, got {total_conversations}"

            assert (
                completed_conversations >= num_conversations * completion_threshold
            ), f"Less than {completion_threshold * 100}% conversations completed: {completed_conversations}/{num_conversations}"

            print(
                f"\n✅ TEST PASSED ({completion_threshold * 100}% completion threshold)"
            )
            print("=" * 70)

        finally:
            # Cleanup
            sample_issuer.shutdown()
            sample_issuer.http_client.shutdown()
            SampleEventHandler.clear_hooks()
