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

"""Tests for sequential mode conversation ordering.

Verifies that sequential mode preserves dataset conversation order,
not lexicographic sorting of conversation IDs.
"""

import random

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
from inference_endpoint.load_generator.scheduler import (
    MultiTurnScheduler,
    WithoutReplacementSampleOrder,
)


@pytest.mark.unit
def test_sequential_mode_unsorted_conversation_ids():
    """Test sequential mode with non-alphabetically ordered conversation IDs.

    Dataset order: zebra, alpha, beta
    Lexicographic order: alpha, beta, zebra
    Should preserve: zebra, alpha, beta (dataset order)
    """
    # Dataset with conversations in specific order: zebra, alpha, beta
    metadata = {
        "samples": [
            {"index": 0, "conversation_id": "conv_zebra", "turn": 1},
            {"index": 1, "conversation_id": "conv_zebra", "turn": 3},
            {"index": 2, "conversation_id": "conv_alpha", "turn": 1},
            {"index": 3, "conversation_id": "conv_alpha", "turn": 3},
            {"index": 4, "conversation_id": "conv_beta", "turn": 1},
            {"index": 5, "conversation_id": "conv_beta", "turn": 3},
        ],
        "num_conversations": 3,
        "max_turns_per_conv": 3,
        "user_turns_per_conversation": {
            "conv_zebra": 2,
            "conv_alpha": 2,
            "conv_beta": 2,
        },
    }

    multi_turn_config = MultiTurnConfig(
        enabled=True, mode=ConversationMode.SEQUENTIAL, turn_timeout_s=60.0
    )

    rt_settings = RuntimeSettings(
        metric_target=metrics.Throughput(10),
        reported_metrics=[],
        min_duration_ms=1000,
        max_duration_ms=120_000,
        n_samples_from_dataset=6,
        n_samples_to_issue=6,
        min_sample_count=6,
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
        load_pattern=LoadPattern(type=LoadPatternType.MULTI_TURN),
    )

    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        rt_settings,
        WithoutReplacementSampleOrder,
        conversation_manager,
        metadata,
        multi_turn_config,
    )

    schedule = list(scheduler._sequential_schedule())

    # Extract conversation order from schedule
    conv_order = []
    for s_idx, _ in schedule:
        conv_id = metadata["samples"][s_idx]["conversation_id"]
        if conv_id not in conv_order:
            conv_order.append(conv_id)

    # Should preserve dataset order, NOT alphabetical
    assert conv_order == [
        "conv_zebra",
        "conv_alpha",
        "conv_beta",
    ], f"Expected dataset order [zebra, alpha, beta], got {conv_order}"
    # Should NOT be: ["conv_alpha", "conv_beta", "conv_zebra"]


@pytest.mark.unit
def test_sequential_mode_numeric_string_ids():
    """Test that numeric strings don't get lexicographically sorted.

    Dataset order: 100, 20, 3
    Lexicographic order: 100, 20, 3 (coincidentally same)
    Numeric order: 3, 20, 100
    Should preserve: 100, 20, 3 (dataset order)
    """
    metadata = {
        "samples": [
            {"index": 0, "conversation_id": "conv_100", "turn": 1},
            {"index": 1, "conversation_id": "conv_20", "turn": 1},
            {"index": 2, "conversation_id": "conv_3", "turn": 1},
        ],
        "num_conversations": 3,
        "max_turns_per_conv": 1,
        "user_turns_per_conversation": {"conv_100": 1, "conv_20": 1, "conv_3": 1},
    }

    multi_turn_config = MultiTurnConfig(
        enabled=True, mode=ConversationMode.SEQUENTIAL, turn_timeout_s=60.0
    )

    rt_settings = RuntimeSettings(
        metric_target=metrics.Throughput(10),
        reported_metrics=[],
        min_duration_ms=1000,
        max_duration_ms=120_000,
        n_samples_from_dataset=3,
        n_samples_to_issue=3,
        min_sample_count=3,
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
        load_pattern=LoadPattern(type=LoadPatternType.MULTI_TURN),
    )

    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        rt_settings,
        WithoutReplacementSampleOrder,
        conversation_manager,
        metadata,
        multi_turn_config,
    )

    schedule = list(scheduler._sequential_schedule())

    conv_order = []
    for s_idx, _ in schedule:
        conv_id = metadata["samples"][s_idx]["conversation_id"]
        if conv_id not in conv_order:
            conv_order.append(conv_id)

    # Dataset order: 100, 20, 3
    # Should NOT be numeric order: 3, 20, 100
    assert conv_order == ["conv_100", "conv_20", "conv_3"]


@pytest.mark.unit
def test_sequential_mode_reverse_alphabetical_order():
    """Test dataset with reverse alphabetical conversation order."""
    metadata = {
        "samples": [
            {"index": 0, "conversation_id": "conv_c", "turn": 1},
            {"index": 1, "conversation_id": "conv_b", "turn": 1},
            {"index": 2, "conversation_id": "conv_a", "turn": 1},
        ],
        "num_conversations": 3,
        "max_turns_per_conv": 1,
        "user_turns_per_conversation": {"conv_c": 1, "conv_b": 1, "conv_a": 1},
    }

    multi_turn_config = MultiTurnConfig(
        enabled=True, mode=ConversationMode.SEQUENTIAL, turn_timeout_s=60.0
    )

    rt_settings = RuntimeSettings(
        metric_target=metrics.Throughput(10),
        reported_metrics=[],
        min_duration_ms=1000,
        max_duration_ms=120_000,
        n_samples_from_dataset=3,
        n_samples_to_issue=3,
        min_sample_count=3,
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
        load_pattern=LoadPattern(type=LoadPatternType.MULTI_TURN),
    )

    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        rt_settings,
        WithoutReplacementSampleOrder,
        conversation_manager,
        metadata,
        multi_turn_config,
    )

    schedule = list(scheduler._sequential_schedule())

    conv_order = []
    for s_idx, _ in schedule:
        conv_id = metadata["samples"][s_idx]["conversation_id"]
        if conv_id not in conv_order:
            conv_order.append(conv_id)

    # Should be reverse alphabetical (dataset order), not sorted
    assert conv_order == ["conv_c", "conv_b", "conv_a"]
    # Should NOT be: ["conv_a", "conv_b", "conv_c"]


@pytest.mark.unit
def test_sequential_mode_single_conversation():
    """Test edge case with single conversation (no ordering needed)."""
    metadata = {
        "samples": [
            {"index": 0, "conversation_id": "only_conv", "turn": 1},
            {"index": 1, "conversation_id": "only_conv", "turn": 3},
        ],
        "num_conversations": 1,
        "max_turns_per_conv": 3,
        "user_turns_per_conversation": {"only_conv": 2},
    }

    multi_turn_config = MultiTurnConfig(
        enabled=True, mode=ConversationMode.SEQUENTIAL, turn_timeout_s=60.0
    )

    rt_settings = RuntimeSettings(
        metric_target=metrics.Throughput(10),
        reported_metrics=[],
        min_duration_ms=1000,
        max_duration_ms=120_000,
        n_samples_from_dataset=2,
        n_samples_to_issue=2,
        min_sample_count=2,
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
        load_pattern=LoadPattern(type=LoadPatternType.MULTI_TURN),
    )

    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        rt_settings,
        WithoutReplacementSampleOrder,
        conversation_manager,
        metadata,
        multi_turn_config,
    )

    schedule = list(scheduler._sequential_schedule())

    # Should schedule both turns
    assert len(schedule) == 2
    assert schedule[0][0] == 0  # First sample index
    assert schedule[1][0] == 1  # Second sample index


@pytest.mark.unit
def test_sequential_mode_uuid_like_ids():
    """Test with UUID-like conversation IDs that would sort incorrectly."""
    metadata = {
        "samples": [
            {
                "index": 0,
                "conversation_id": "uuid-zzz-9999-ffff-000000000001",
                "turn": 1,
            },
            {
                "index": 1,
                "conversation_id": "uuid-aaa-1111-0000-000000000002",
                "turn": 1,
            },
            {
                "index": 2,
                "conversation_id": "uuid-mmm-5555-7777-000000000003",
                "turn": 1,
            },
        ],
        "num_conversations": 3,
        "max_turns_per_conv": 1,
        "user_turns_per_conversation": {
            "uuid-zzz-9999-ffff-000000000001": 1,
            "uuid-aaa-1111-0000-000000000002": 1,
            "uuid-mmm-5555-7777-000000000003": 1,
        },
    }

    multi_turn_config = MultiTurnConfig(
        enabled=True, mode=ConversationMode.SEQUENTIAL, turn_timeout_s=60.0
    )

    rt_settings = RuntimeSettings(
        metric_target=metrics.Throughput(10),
        reported_metrics=[],
        min_duration_ms=1000,
        max_duration_ms=120_000,
        n_samples_from_dataset=3,
        n_samples_to_issue=3,
        min_sample_count=3,
        rng_sched=random.Random(42),
        rng_sample_index=random.Random(42),
        load_pattern=LoadPattern(type=LoadPatternType.MULTI_TURN),
    )

    conversation_manager = ConversationManager()
    scheduler = MultiTurnScheduler(
        rt_settings,
        WithoutReplacementSampleOrder,
        conversation_manager,
        metadata,
        multi_turn_config,
    )

    schedule = list(scheduler._sequential_schedule())

    conv_order = []
    for s_idx, _ in schedule:
        conv_id = metadata["samples"][s_idx]["conversation_id"]
        if conv_id not in conv_order:
            conv_order.append(conv_id)

    # Should preserve dataset order (zzz, aaa, mmm)
    assert conv_order == [
        "uuid-zzz-9999-ffff-000000000001",
        "uuid-aaa-1111-0000-000000000002",
        "uuid-mmm-5555-7777-000000000003",
    ]
