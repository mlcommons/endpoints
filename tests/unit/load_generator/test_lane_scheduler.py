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

"""Stationary agentic scheduler: window size, entry state, deck order, tagging and secrecy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
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
    batch_mean_ci,
    fit_service,
    lane_passes,
    turn_sizes,
)

pytestmark = pytest.mark.unit

_SALT_RE = re.compile(r"^\[salt: ([0-9a-f]+)\]\n\n.*\n\n\[salt: ([0-9a-f]+)\]$", re.S)
_CONV_RE = re.compile(r"^(.*)__(repeat|unmeasured)_(\d+)$")


def _metadata(
    conversations: dict[str, int], delays: dict[str, float] | None = None
) -> ConversationMetadata:
    """Each turn's messages extend the previous turn's: the system prompt, then
    per earlier turn a user message and a recorded reply 40 x turn chars long."""
    samples, messages, delay_map = [], {}, {}
    index = 0
    for conv_id, n_turns in conversations.items():
        history = [{"role": "system", "content": f"system {conv_id}"}]
        for turn in range(1, n_turns + 1):
            samples.append(
                ConversationSampleEntry(
                    conversation_id=conv_id, turn=turn, sample_index=index
                )
            )
            history = [*history, {"role": "user", "content": f"turn {turn}"}]
            messages[(conv_id, turn)] = history
            history = [*history, {"role": "assistant", "content": "a" * 40 * turn}]
            if delays and conv_id in delays:
                delay_map[(conv_id, turn)] = delays[conv_id]
            index += 1
    return ConversationMetadata(
        samples=samples,
        num_conversations=len(conversations),
        max_turns_per_conv=max(conversations.values()),
        client_turns_per_conversation=dict(conversations),
        pre_built_messages_by_key=messages,
        delay_seconds_by_key=delay_map,
    )


# 4 coding conversations (long) and 12 workflow conversations (short).
_DATASET = {
    **{f"repo__issue-{i}": 9 + i for i in range(4)},
    **{f"sim_{i}": 2 + i % 3 for i in range(12)},
}
_DATASET_TURNS = sum(_DATASET.values())


class LoopIssuer:
    """Completes every request after ``latency`` seconds on the running loop;
    ``latency`` may be a function of (data_override, seconds since first issue)."""

    def __init__(self, latency=0.0005, fail_every: int = 0):
        self.latency = latency
        self.started: float | None = None
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
        source, kind, serial = _CONV_RE.match(conversation_id).groups()
        self.records.append(
            {
                "qid": query_id,
                "source": source,
                "kind": kind,
                "serial": int(serial),
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
        loop = asyncio.get_running_loop()
        if self.started is None:
            self.started = loop.time()
        latency = (
            self.latency(data_override, loop.time() - self.started)
            if callable(self.latency)
            else self.latency
        )
        loop.call_later(
            latency,
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


def _config(inject_tool_delay: bool = False, **lanes) -> AgenticInferenceConfig:
    defaults = {"warmup_s": 0.02, "passes": 3}
    defaults.update(lanes)
    return AgenticInferenceConfig(
        enable_salt=True,
        scheduler="lanes",
        inject_tool_delay=inject_tool_delay,
        lanes=LaneSchedulerConfig(**defaults),
    )


def _strategy(
    tmp_path: Path, concurrency: int, dataset=None, delays=None, **lanes
) -> LaneAgenticInferenceStrategy:
    return LaneAgenticInferenceStrategy(
        dataset_metadata=_metadata(dataset or _DATASET, delays),
        agentic_inference_config=_config(inject_tool_delay=bool(delays), **lanes),
        target_concurrency=concurrency,
        rng_sample_index=random.Random(42),
        report_dir=tmp_path,
    )


def _run(
    tmp_path: Path, concurrency: int = 4, issuer: LoopIssuer | None = None, **lanes
) -> tuple[LoopIssuer, dict]:
    issuer = issuer or LoopIssuer()
    strategy = _strategy(tmp_path, concurrency, **lanes)
    issuer.strategy = strategy
    asyncio.run(asyncio.wait_for(strategy.execute(issuer), timeout=60))
    return issuer, json.loads((tmp_path / "lane_plan.json").read_text())


def test_window_counts_whole_passes_issued_in_one_contiguous_stretch(
    tmp_path: Path,
) -> None:
    issuer, plan = _run(tmp_path)
    tracked = [i for i, r in enumerate(issuer.records) if r["tracked"]]
    assert plan["passes"] == 3
    assert len(tracked) == 3 * _DATASET_TURNS == plan["measured_turns_expected"]
    assert plan["measured_turns_completed"] == plan["measured_turns_expected"]
    # Nothing outside the window counts: once a turn goes out unmeasured after
    # the window opened, no later turn is counted again.
    assert all(issuer.records[i]["tracked"] for i in range(tracked[0], tracked[-1] + 1))
    assert all(
        r["kind"] == ("repeat" if r["tracked"] else "unmeasured")
        for r in issuer.records
    )
    # Each counted turn is one (instance, turn); an instance's counted turns are
    # consecutive, starting where the window found it.
    by_instance: dict[int, list[int]] = defaultdict(list)
    for r in issuer.records:
        if r["tracked"]:
            by_instance[r["serial"]].append(r["turn"])
    for turns in by_instance.values():
        assert turns == list(range(turns[0], turns[0] + len(turns)))
    assert plan["coverage"]["mean_times_counted"] == pytest.approx(3)


def test_slots_enter_mid_conversation_in_proportion_to_length(tmp_path: Path) -> None:
    """The stationary entry picks a conversation by its duration and a turn
    uniform in time, so slots start spread over every turn position, weighted
    to long conversations and to turns preceded by long tool delays."""
    dataset = {"long": 10, "short": 2}
    strategy = _strategy(tmp_path, 1, dataset=dataset, delays={"short": 63.0})
    strategy._load_sources()
    entries = Counter()
    n = 20000
    for _ in range(n):
        instance, cursor = strategy._stationary_entry()
        entries[(strategy._sources[instance.source].source_id, cursor)] += 1
    # Weights: long = 10 turns x 7 s = 70; short = 2 x (7 + 63) = 140.
    long_share = sum(v for (s, _), v in entries.items() if s == "long") / n
    assert long_share == pytest.approx(70 / 210, abs=0.02)
    # Uniform in time within "long": every turn equally likely, not just turn 1.
    long_turns = [entries[("long", k)] for k in range(10)]
    assert min(long_turns) > 0.7 * max(long_turns)


def test_later_conversations_come_from_a_shuffled_deck(tmp_path: Path) -> None:
    """After entry, every source is drawn once per pass whatever its length:
    starts per source differ by at most one deck."""
    issuer, plan = _run(tmp_path, concurrency=4)
    entry_serials = set(range(1, 5))
    starts = Counter(
        r["source"]
        for r in issuer.records
        if r["turn"] == 1 and r["serial"] not in entry_serials
    )
    assert set(starts) == set(_DATASET)
    assert max(starts.values()) - min(starts.values()) <= 1
    assert plan["decks_drawn"] >= 3


def test_salts_and_routing_keys_are_keyed_and_opaque(tmp_path: Path) -> None:
    issuer, plan = _run(tmp_path)
    by_instance: dict[int, set[tuple[str, str, str]]] = defaultdict(set)
    for rec in issuer.records:
        match = _SALT_RE.match(rec["system"])
        assert match is not None
        by_instance[rec["serial"]].add((*match.groups(), rec["route"]))
        assert len(rec["route"]) == 32 and rec["source"] not in rec["route"]
    # One salt and one routing key per instance for its whole life, including
    # across the window opening, so its prefix cache carries over.
    assert all(len(v) == 1 for v in by_instance.values())
    tails = [next(iter(v))[1] for v in by_instance.values()]
    routes = [next(iter(v))[2] for v in by_instance.values()]
    assert len(set(tails)) == len(tails) and len(set(routes)) == len(routes)
    # The head is shared per deck (cached system prompt), so far fewer heads
    # than instances.
    heads = {next(iter(v))[0] for v in by_instance.values()}
    assert len(heads) <= plan["decks_drawn"] + 1 < len(by_instance)
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


def test_failed_measured_turns_still_complete_the_run(tmp_path: Path) -> None:
    issuer, plan = _run(tmp_path, issuer=LoopIssuer(fail_every=7))
    assert plan["measured_turns_completed"] == plan["measured_turns_expected"]
    assert plan["measured_turns_failed"] > 0


def test_passes_follow_the_warmup_issue_rate(tmp_path: Path) -> None:
    """Without a fixed count the window covers about window_s at the rate seen
    in the warm-up; slower turns mean fewer passes."""
    fast, plan_fast = _run(tmp_path / "f", passes=None, warmup_s=0.2, window_s=0.5)
    slow, plan_slow = _run(
        tmp_path / "s",
        issuer=LoopIssuer(latency=0.004),
        passes=None,
        warmup_s=0.2,
        window_s=0.5,
    )
    assert plan_fast["passes"] > plan_slow["passes"] >= 1
    for plan in (plan_fast, plan_slow):
        assert plan["measured_turns_expected"] == plan["passes"] * _DATASET_TURNS


def test_pass_rule() -> None:
    assert lane_passes(1200, 240, 1) == 5
    assert lane_passes(1200, 241, 1) == 5
    assert lane_passes(1200, 239, 1) == 6
    assert lane_passes(1200, 3000, 1) == 1
    assert lane_passes(1200, 3000, 2) == 2
    assert lane_passes(1200, None, 3) == 3


def test_rejects_fixed_trajectory_count(tmp_path: Path) -> None:
    cfg = AgenticInferenceConfig(
        enable_salt=True,
        scheduler="lanes",
        num_trajectories_to_issue=len(_DATASET),
        lanes=LaneSchedulerConfig(passes=1),
    )
    strategy = LaneAgenticInferenceStrategy(
        dataset_metadata=_metadata(_DATASET),
        agentic_inference_config=cfg,
        target_concurrency=4,
        report_dir=tmp_path,
    )
    issuer = LoopIssuer()
    issuer.strategy = strategy
    with pytest.raises(InputValidationError, match="num_trajectories_to_issue"):
        asyncio.run(strategy.execute(issuer))


def test_lanes_require_salt() -> None:
    with pytest.raises(ValueError, match="enable_salt"):
        AgenticInferenceConfig(scheduler="lanes")


def _context_latency(data_override, _elapsed: float) -> float:
    """Latency grows with the request's context size."""
    chars = sum(len(m.get("content") or "") for m in data_override["messages"])
    return 0.0003 + 2e-6 * chars


def test_calibrated_entry_fits_then_reseeds_every_slot(tmp_path: Path) -> None:
    _issuer, plan = _run(
        tmp_path,
        issuer=LoopIssuer(latency=_context_latency),
        entry="calibrated",
        calibration_s=0.1,
        warmup_s=0.3,
    )
    fit = plan["calibration"]
    assert "error" not in fit and fit["samples"] >= 30
    assert fit["mean_turn_s"] > 0
    # Every slot moved to a calibrated entry before the run ended, and the
    # window still counts exact passes.
    assert plan["reseed_pending_at_end"] == 0 and plan["ready_s"] is not None
    assert plan["measured_turns_completed"] == plan["measured_turns_expected"]
    assert plan["measured_turns_expected"] == plan["passes"] * _DATASET_TURNS


def test_calibration_must_finish_inside_a_fixed_warmup() -> None:
    with pytest.raises(ValueError, match="calibration_s"):
        LaneSchedulerConfig(entry="calibrated", calibration_s=200, warmup_s=180)


def test_adaptive_warmup_opens_once_steady(tmp_path: Path) -> None:
    _issuer, plan = _run(
        tmp_path,
        issuer=LoopIssuer(latency=0.002),
        warmup="adaptive",
        warmup_interval_s=0.05,
        warmup_band=0.25,
        warmup_min_s=0.05,
        warmup_max_s=3.0,
    )
    assert plan["warmup_converged"] is True
    assert plan["warmup_s_actual"] < 3.0


def test_adaptive_warmup_gives_up_at_max_while_still_drifting(tmp_path: Path) -> None:
    """Latency that keeps rising never settles: open at warmup_max_s, flagged."""
    _issuer, plan = _run(
        tmp_path,
        issuer=LoopIssuer(latency=lambda _d, t: 0.002 * (1 + 8 * t)),
        warmup="adaptive",
        warmup_interval_s=0.05,
        warmup_band=0.05,
        warmup_min_s=0.05,
        warmup_max_s=0.6,
    )
    assert plan["warmup_converged"] is False
    assert plan["warmup_s_actual"] >= 0.6


def test_precision_stop_adds_passes_until_the_target_or_the_cap(tmp_path: Path) -> None:
    _a, capped = _run(
        tmp_path / "a",
        stop="precision",
        precision_target=1e-9,
        precision_batch_s=0.01,
        max_passes=3,
    )
    assert capped["passes"] == 3
    assert [d["extend"] for d in capped["pass_decisions"]] == [True, True, False]
    assert capped["measured_turns_completed"] == 3 * _DATASET_TURNS
    _b, met = _run(
        tmp_path / "b",
        stop="precision",
        precision_target=0.5,
        precision_batch_s=0.01,
        precision_min_batches=4,
        max_passes=60,
    )
    last = met["pass_decisions"][-1]
    assert met["passes"] < 60 and last["extend"] is False
    assert last["ci95_half_width_rel"] <= 0.5 and last["batches"] >= 4
    # Too few batches never stop the window, however tight the estimate looks.
    assert all(d["extend"] for d in met["pass_decisions"] if d["batches"] < 4)
    assert met["measured_turns_expected"] == met["passes"] * _DATASET_TURNS


def test_batch_mean_ci() -> None:
    s = 1_000_000_000
    flat = [(0, 10.0), (s, 10.0), (2 * s, 10.0)]
    assert batch_mean_ci(flat, 0, 3 * s, s) == (3, 10.0, 0.0)
    n, rate, half = batch_mean_ci([(0, 10.0), (s, 20.0)], 0, 2 * s, s)
    # mean 15, sd 7.07, t(1 df) 12.71: half-width 12.71 * 7.07 / sqrt(2) / 15.
    assert (n, rate) == (2, 15.0) and half == pytest.approx(4.237, abs=1e-3)
    assert batch_mean_ci(flat, 0, s, s)[1:] == (None, None)


def test_fit_service_recovers_coefficients_and_drops_negative_ones() -> None:
    rng = random.Random(0)
    rows = [(rng.uniform(0, 500), rng.uniform(0, 5000)) for _ in range(200)]
    exact = [(1.0 + 0.01 * o + 0.001 * p, o, p) for o, p in rows]
    assert fit_service(exact) == pytest.approx((1.0, 0.01, 0.001))
    # Latency that falls with prefill size is not physical: c is dropped.
    falling = [(1.0 + 0.01 * o - 0.0001 * p, o, p) for o, p in rows]
    assert fit_service(falling)[2] == 0.0
    assert fit_service(exact[:10]) is None


def test_turn_sizes_split_new_input_context_and_recorded_reply() -> None:
    system = {"role": "system", "content": "s" * 100}
    t1 = [system, {"role": "user", "content": "u" * 10}]
    t2 = [
        *t1,
        {"role": "assistant", "content": "a" * 30},
        {"role": "user", "content": "v" * 5},
    ]
    new, ctx, out = turn_sizes([t1, t2])
    assert new == (110, 35) and ctx == (110, 145) and out == (30.0, None)
