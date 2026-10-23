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

"""Balanced agentic scheduler: coverage, start alignment, ordering, tagging and secrecy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import pytest

from inference_endpoint.config.schema import AgenticInferenceConfig, LaneSchedulerConfig
from inference_endpoint.core.types import ErrorData, QueryResult
from inference_endpoint.dataset_manager.agentic_inference_dataset import (
    ConversationMetadata,
    ConversationSampleEntry,
)
from inference_endpoint.exceptions import InputValidationError
from inference_endpoint.load_generator.lane_scheduler import (
    LaneAgenticInferenceStrategy,
    lane_copies,
    split_points,
)

pytestmark = pytest.mark.unit

_SALT_RE = re.compile(r"^\[salt: ([0-9a-f]+)\]\n\n.*\n\n\[salt: ([0-9a-f]+)\]$", re.S)


def _metadata(
    conversations: dict[str, int], delay: float = 0.0
) -> ConversationMetadata:
    samples, messages, delays = [], {}, {}
    index = 0
    for conv_id, n_turns in conversations.items():
        for turn in range(1, n_turns + 1):
            samples.append(
                ConversationSampleEntry(
                    conversation_id=conv_id, turn=turn, sample_index=index
                )
            )
            messages[(conv_id, turn)] = [
                {"role": "system", "content": f"system {conv_id}"},
                {"role": "user", "content": f"turn {turn}"},
            ]
            if delay:
                delays[(conv_id, turn)] = delay
            index += 1
    return ConversationMetadata(
        samples=samples,
        num_conversations=len(conversations),
        max_turns_per_conv=max(conversations.values()),
        client_turns_per_conversation=dict(conversations),
        pre_built_messages_by_key=messages,
        delay_seconds_by_key=delays,
    )


# 4 coding conversations (long) and 12 workflow conversations (short).
_DATASET = {
    **{f"repo__issue-{i}": 9 + i for i in range(4)},
    **{f"sim_{i}": 2 + i % 3 for i in range(12)},
}


class LoopIssuer:
    """Completes every request after ``latency`` seconds on the running loop."""

    def __init__(self, latency: float = 0.0005, fail_every: int = 0):
        self.latency = latency
        self.fail_every = fail_every
        self.strategy: LaneAgenticInferenceStrategy | None = None
        self.records: list[dict] = []
        self.issued_count = 0
        self.uuid_to_index: dict[str, int] = {}
        self.uuid_to_conv_info: dict[str, tuple[str, int | None]] = {}
        self.completed_uuids: set[str] = set()

    def issue(
        self,
        sample_index,
        data_override=None,
        conversation_id="",
        turn=None,
        routing_key=None,
        tracked=True,
    ):
        query_id = f"q{self.issued_count}"
        self.issued_count += 1
        self.records.append(
            {
                "qid": query_id,
                "conv": conversation_id,
                "turn": turn,
                "tracked": tracked,
                "route": routing_key,
                "system": data_override["messages"][0]["content"],
            }
        )
        error = None
        if self.fail_every and self.issued_count % self.fail_every == 0:
            error = ErrorData(error_type="x", error_message="boom")
        assert self.strategy is not None
        asyncio.get_running_loop().call_later(
            self.latency,
            self.strategy.on_sample_complete,
            QueryResult(id=query_id, error=error),
        )
        return query_id

    def register_skipped(self, *args, **kwargs):
        raise AssertionError("lanes never register skipped turns")

    def mark_inflight_complete(self) -> None:
        pass

    def stop_performance_tracking(self) -> None:
        pass


def _config(**lanes) -> AgenticInferenceConfig:
    defaults = {"min_copies": 3, "warmup_s": 0.01}
    defaults.update(lanes)
    return AgenticInferenceConfig(
        enable_salt=True, scheduler="lanes", lanes=LaneSchedulerConfig(**defaults)
    )


def _run(
    tmp_path: Path, concurrency: int = 4, issuer: LoopIssuer | None = None, **lanes
) -> tuple[LoopIssuer, dict]:
    issuer = issuer or LoopIssuer()
    strategy = LaneAgenticInferenceStrategy(
        dataset_metadata=_metadata(_DATASET),
        agentic_inference_config=_config(**lanes),
        target_concurrency=concurrency,
        rng_sample_index=random.Random(42),
        report_dir=tmp_path,
    )
    issuer.strategy = strategy
    asyncio.run(asyncio.wait_for(strategy.execute(issuer), timeout=60))
    return issuer, json.loads((tmp_path / "lane_plan.json").read_text())


def _source(conv: str) -> tuple[str, str, int]:
    match = re.search(r"__(repeat|unmeasured)_(\d+)$", conv)
    return (
        (conv, "repeat", 1)
        if match is None
        else (conv[: match.start()], match.group(1), int(match.group(2)))
    )


def test_every_dataset_turn_is_measured_exactly_copies_times(tmp_path: Path) -> None:
    issuer, plan = _run(tmp_path)
    copies = plan["copies"]
    measured = Counter()
    for rec in issuer.records:
        if rec["tracked"]:
            source, kind, copy = _source(rec["conv"])
            assert kind == "repeat"
            measured[(source, copy, rec["turn"])] += 1
    expected = {
        (c, k, t)
        for c, n in _DATASET.items()
        for k in range(1, copies + 1)
        for t in range(1, n + 1)
    }
    assert set(measured) == expected
    assert set(measured.values()) == {1}
    assert (
        plan["measured_turns_completed"]
        == plan["measured_turns_expected"]
        == len(expected)
    )


def test_unmeasured_load_is_tagged_and_never_uses_measured_ids(tmp_path: Path) -> None:
    issuer, _plan = _run(tmp_path)
    unmeasured = [r for r in issuer.records if not r["tracked"]]
    assert unmeasured, "calibration and warm-up issue unmeasured load"
    assert all("__unmeasured_" in r["conv"] for r in unmeasured)
    assert all("__unmeasured_" not in r["conv"] for r in issuer.records if r["tracked"])


def test_salts_and_routing_keys_are_keyed_and_opaque(tmp_path: Path) -> None:
    issuer, plan = _run(tmp_path)
    by_conv: dict[str, set[tuple[str, str]]] = {}
    for rec in issuer.records:
        match = _SALT_RE.match(rec["system"])
        assert match is not None
        by_conv.setdefault(rec["conv"], set()).add(match.groups())
        assert len(rec["route"]) == 32 and _source(rec["conv"])[0] not in rec["route"]
    # A measured instance and its unmeasured duplicates never share a salt.
    salts = [salt for conv, pairs in by_conv.items() for salt in pairs]
    unmeasured = {
        s for conv, pairs in by_conv.items() if "__unmeasured_" in conv for s in pairs
    }
    measured = {
        s
        for conv, pairs in by_conv.items()
        if "__unmeasured_" not in conv
        for s in pairs
    }
    assert not unmeasured & measured
    assert len(salts) == len(set(salts))
    # The head salt depends only on the copy, so every conversation of a copy
    # shares the cached system-prompt prefix (the reference salt's shape); the
    # tail separates instances. A per-instance head made every conversation
    # re-prefill the shared system prompt and collapsed the prefix-cache hit rate.
    heads_by_copy: dict[object, set[str]] = {}
    tail_owner: dict[str, str] = {}
    for conv, pairs in by_conv.items():
        source, kind, copy = _source(conv)
        group = "unmeasured" if kind == "unmeasured" else copy
        for head, tail in pairs:
            heads_by_copy.setdefault(group, set()).add(head)
            assert tail_owner.setdefault(tail, conv) == conv
    assert all(len(heads) == 1 for heads in heads_by_copy.values())
    assert len({next(iter(h)) for h in heads_by_copy.values()}) == len(heads_by_copy)
    assert (
        plan["secret_sha256"]
        == hashlib.sha256(bytes.fromhex(plan["secret_hex"])).hexdigest()
    )
    commitment = json.loads((tmp_path / "lane_commitment.json").read_text())
    assert commitment["secret_sha256"] == plan["secret_sha256"]
    assert plan["secret_hex"] not in commitment.values()


def test_two_runs_draw_different_secret_plans(tmp_path: Path) -> None:
    _a, plan_a = _run(tmp_path / "a")
    _b, plan_b = _run(tmp_path / "b")
    assert plan_a["secret_hex"] != plan_b["secret_hex"]
    assert plan_a["copies"] == plan_b["copies"]


def test_failed_measured_turns_still_complete_the_run(tmp_path: Path) -> None:
    issuer, plan = _run(tmp_path, issuer=LoopIssuer(fail_every=7))
    assert plan["measured_turns_completed"] == plan["measured_turns_expected"]
    assert plan["measured_turns_failed"] > 0


def test_rejects_trajectory_count_that_disagrees_with_the_copy_rule(
    tmp_path: Path,
) -> None:
    cfg = AgenticInferenceConfig(
        enable_salt=True,
        scheduler="lanes",
        num_trajectories_to_issue=len(_DATASET),
        lanes=LaneSchedulerConfig(min_copies=3),
    )
    strategy = LaneAgenticInferenceStrategy(
        dataset_metadata=_metadata(_DATASET),
        agentic_inference_config=cfg,
        target_concurrency=4,
        report_dir=tmp_path,
    )
    issuer = LoopIssuer()
    issuer.strategy = strategy
    with pytest.raises(InputValidationError, match="measures 3 copies"):
        asyncio.run(strategy.execute(issuer))


def test_lanes_require_salt() -> None:
    with pytest.raises(ValueError, match="enable_salt"):
        AgenticInferenceConfig(scheduler="lanes")


def test_copy_rule() -> None:
    cfg = LaneSchedulerConfig()
    assert lane_copies(113, 32, cfg) == 4
    assert lane_copies(113, 256, cfg) == 4
    assert lane_copies(113, 384, cfg) == 6
    assert lane_copies(113, 768, cfg) == 12
    assert lane_copies(113, 1024, cfg) == 16


def test_measurement_starts_with_every_slot_mid_conversation(tmp_path: Path) -> None:
    """All slots switch to measured work at one instant from their next turn,
    so measurement neither ramps up from cold slots nor staggers by slot."""
    issuer, plan = _run(tmp_path, concurrency=4)
    assert plan["slots_measured_from_next_turn"] >= 3
    # Every slot's first measured request goes out within a couple of turn
    # latencies of the switch, far less than the measured phase itself.
    assert plan["measured_start_spread_s"] < 0.25 * plan["measured_phase_s"]
    first_measured = next(i for i, r in enumerate(issuer.records) if r["tracked"])
    assert all(not r["tracked"] for r in issuer.records[:first_measured])


def test_split_points_bound_every_chunk_and_balance_them() -> None:
    turns = [1.0] * 90
    cuts = split_points(turns, max_len_s=40.0)
    bounds = [0, *cuts, len(turns)]
    chunks = [sum(turns[a:b]) for a, b in zip(bounds, bounds[1:], strict=False)]
    assert len(chunks) == 3 and max(chunks) <= 40.0 and min(chunks) >= 29.0
    # Short enough already, or a single turn: never cut.
    assert split_points([1.0] * 10, 40.0) == []
    assert split_points([500.0], 40.0) == []


def test_long_conversations_are_measured_in_parts_without_losing_turns(
    tmp_path: Path,
) -> None:
    """Instances longer than an eighth of the measured phase are split into
    parts; every turn is still measured exactly once per copy."""
    # One copy over 8 slots: the measured phase is ~10 turns long, so the
    # 9-12 turn coding conversations exceed an eighth of it.
    issuer, plan = _run(
        tmp_path, concurrency=8, min_copies=1, coding_conversations_per_slot=0.5
    )
    assert plan["split_segments"] > 0
    measured = Counter(
        (_source(r["conv"])[0], _source(r["conv"])[2], r["turn"])
        for r in issuer.records
        if r["tracked"]
    )
    assert set(measured.values()) == {1}
    assert len(measured) == plan["measured_turns_expected"]
