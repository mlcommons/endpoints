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

"""Balanced-lane agentic scheduler (``agentic_inference.scheduler: lanes``).

The queue scheduler hands conversations to slots from one shared start queue.
Every slot starts at turn 1 at the same moment (a ramp), and the measured
window closes when the queue runs dry while the longest conversations are
still running (a tail that is never counted). This scheduler removes both:

1. Calibration. All C slots run unmeasured load from a provisional plan. Once
   every slot's first (cold) request has completed, plus ``calibration_skip_s``,
   the mean service time of warm turns (issue to completion) is measured for
   coding and workflow turns separately for ``calibration_s``.
2. Plan. ``copies`` dataset copies (from a rule, not a user knob) are packed
   longest-first into C lanes of equal estimated length, where a turn's
   estimate is its recorded tool delay plus the measured service time of its
   domain. Each lane is a cycle in a key-shuffled order.
3. Run. Each slot enters its lane at an evenly spread phase, runs an
   unmeasured warm-up whose length is drawn from the secret key, measures
   exactly one lap, and then keeps the load up with unmeasured padding from its
   own lane until every measured turn has completed. A slot that finishes its
   lap early steals unstarted measured segments from the slot furthest behind.

Every dataset turn is measured exactly ``copies`` times. Unmeasured requests are
ISSUED with ``tracked=False`` and an ``__unmeasured_<n>`` conversation id, so no
metric, OSL statistic or accuracy score counts them. Each segment gets its own
keyed salt and an opaque routing key; the key is generated at run start, never
configured, and written to ``lane_plan.json`` only after the last request.
"""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import json
import logging
import math
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config.schema import AgenticInferenceConfig, LaneSchedulerConfig
from ..core.types import QueryResult
from ..dataset_manager.agentic_inference_dataset import ConversationMetadata
from ..exceptions import InputValidationError
from .agentic_inference_strategy import AgenticInferenceStrategy
from .conversation_manager import ConversationManager
from .strategy import PhaseIssuerProtocol

logger = logging.getLogger(__name__)

# Same rule AgenticInferenceInlineScorer uses to split workflow from coding turns.
_WORKFLOW_CONVERSATION_RE = re.compile(r"^sim_\d+$")
CODING = "coding"
WORKFLOW = "workflow"
# Planning guess for the provisional (calibration) plan only.
_PROVISIONAL_SERVICE_S = 2.0
# A domain with fewer warm samples than this falls back to the pooled mean.
_MIN_DOMAIN_SAMPLES = 30


def conversation_domain(conversation_id: str) -> str:
    """Return ``workflow`` for ``sim_<n>`` conversations, else ``coding``."""
    return WORKFLOW if _WORKFLOW_CONVERSATION_RE.match(conversation_id) else CODING


def lane_copies(
    n_coding_conversations: int, concurrency: int, cfg: LaneSchedulerConfig
) -> int:
    """Dataset copies measured: enough that each lane holds about
    ``coding_conversations_per_lane`` coding conversations, never fewer than
    ``min_copies``."""
    if n_coding_conversations == 0:
        return cfg.min_copies
    return max(
        cfg.min_copies,
        math.ceil(
            cfg.coding_conversations_per_lane * concurrency / n_coding_conversations
        ),
    )


class RunKey:
    """Per-run secret. Derives salts, routing keys and plan randomness.

    ``seed`` ties the plan family to the run's dataloader seed; ``secret``
    (256 random bits from the OS) makes it unpredictable to the server until
    ``lane_plan.json`` reveals it after the run.
    """

    def __init__(self, secret: bytes, seed: int):
        self.secret = secret
        self.seed = seed

    @classmethod
    def generate(cls, seed: int) -> RunKey:
        return cls(os.urandom(32), seed)

    @property
    def commitment(self) -> str:
        return hashlib.sha256(self.secret).hexdigest()

    def _subkey(self, purpose: str) -> bytes:
        return hashlib.blake2b(
            purpose.encode() + self.seed.to_bytes(8, "big", signed=False),
            key=self.secret,
            digest_size=32,
        ).digest()

    def digest(self, purpose: str, message: str, size: int = 8) -> str:
        return hashlib.blake2b(
            message.encode(), key=self._subkey(purpose), digest_size=size
        ).hexdigest()

    def rng(self, purpose: str) -> random.Random:
        return random.Random(int.from_bytes(self._subkey(purpose), "big"))


@dataclass(frozen=True, slots=True)
class SourceConversation:
    """One dataset conversation as the planner sees it."""

    source_id: str
    # (sample_index, turn) in turn order.
    turns: tuple[tuple[int, int], ...]
    # Tool delay before each turn (0 when tool delay is not injected).
    delays: tuple[float, ...]
    domain: str


@dataclass(slots=True)
class Segment:
    """A contiguous run of turns ``[first, stop)`` of one conversation instance."""

    source: int
    copy: int  # >= 1 for a measured instance, 0 for unmeasured load
    first: int
    stop: int
    measured: bool
    uid: str
    est_s: float
    logical_id: str = ""


@dataclass(slots=True)
class SlotProgram:
    """One slot's plan: its lane, where it enters, and where its lap starts."""

    lane: list[tuple[int, int]]  # (source index, copy) in cyclic order
    positions: list[tuple[int, int]]  # flattened (instance index, turn cursor)
    entry_pos: int
    lap_pos: int
    warmup_est_s: float
    warmup: deque[Segment] = field(default_factory=deque)
    measured: deque[Segment] = field(default_factory=deque)
    padding_pos: int = 0

    @property
    def n_turns(self) -> int:
        return len(self.positions)


class LanePlanner:
    """Builds slot programs. Pure apart from the RNG it is given."""

    def __init__(
        self,
        sources: list[SourceConversation],
        service_s: dict[str, float],
        uid_prefix: str,
    ):
        self._sources = sources
        self._service_s = service_s
        self._uid_prefix = uid_prefix
        self._uid_counter = 0

    def turn_est(self, source: int, cursor: int) -> float:
        src = self._sources[source]
        return src.delays[cursor] + self._service_s[src.domain]

    def instance_est(self, source: int) -> float:
        src = self._sources[source]
        return sum(src.delays) + self._service_s[src.domain] * len(src.turns)

    def _uid(self) -> str:
        self._uid_counter += 1
        return f"{self._uid_prefix}:{self._uid_counter}"

    def pack(
        self, copies: int, concurrency: int, rng: random.Random
    ) -> list[list[tuple[int, int]]]:
        """Longest-first onto the least-loaded lane, then shuffle each lane."""
        instances = [
            (s, k) for k in range(1, copies + 1) for s in range(len(self._sources))
        ]
        est = [self.instance_est(s) for s in range(len(self._sources))]
        instances.sort(key=lambda inst: (-est[inst[0]], inst[0], inst[1]))
        loads = [(0.0, j) for j in range(concurrency)]
        lanes: list[list[tuple[int, int]]] = [[] for _ in range(concurrency)]
        # Ties on load resolve to the lowest lane index.
        heapq.heapify(loads)
        for inst in instances:
            load, j = heapq.heappop(loads)
            lanes[j].append(inst)
            heapq.heappush(loads, (load + est[inst[0]], j))
        for lane in lanes:
            rng.shuffle(lane)
        return lanes

    def programs(
        self,
        lanes: list[list[tuple[int, int]]],
        rng: random.Random,
        warmup_min_s: float,
        warmup_max_s: float,
        measure: bool,
    ) -> list[SlotProgram | None]:
        """Entry point, warm-up and measured lap for every lane.

        ``measure=False`` builds a provisional plan whose slots only produce
        unmeasured load (calibration).
        """
        phases = [(j + 0.5) / len(lanes) for j in range(len(lanes))]
        rng.shuffle(phases)
        programs: list[SlotProgram | None] = []
        for j, lane in enumerate(lanes):
            if not lane:
                programs.append(None)
                continue
            positions: list[tuple[int, int]] = []
            cum: list[float] = []
            total = 0.0
            for inst_idx, (source, _copy) in enumerate(lane):
                for cursor in range(len(self._sources[source].turns)):
                    positions.append((inst_idx, cursor))
                    cum.append(total)
                    total += self.turn_est(source, cursor)
            warmup = rng.uniform(warmup_min_s, warmup_max_s) if measure else 0.0
            lap_at = phases[j] * total
            entry_at = (lap_at - warmup) % total if total > 0 else 0.0
            lap_pos = _position_at(cum, lap_at)
            entry_pos = _position_at(cum, entry_at)
            program = SlotProgram(
                lane=lane,
                positions=positions,
                entry_pos=entry_pos,
                lap_pos=lap_pos,
                warmup_est_s=warmup,
            )
            n = len(positions)
            if measure:
                warm_turns = (lap_pos - entry_pos) % n
                program.warmup.extend(
                    self.walk(program, entry_pos, warm_turns, measured=False)
                )
                program.measured.extend(self.walk(program, lap_pos, n, measured=True))
                program.padding_pos = lap_pos
            else:
                program.padding_pos = entry_pos
            programs.append(program)
        return programs

    def walk(
        self, program: SlotProgram, start_pos: int, n_turns: int, measured: bool
    ) -> list[Segment]:
        """Split ``n_turns`` cyclic lane positions from ``start_pos`` into segments
        at conversation-instance boundaries and at the lane wrap."""
        segments: list[Segment] = []
        n = program.n_turns
        pos = start_pos
        remaining = n_turns
        while remaining > 0:
            inst_idx, cursor = program.positions[pos]
            source, copy = program.lane[inst_idx]
            conv_len = len(self._sources[source].turns)
            # Turns left in this instance before its end or the lane wrap.
            span = min(conv_len - cursor, remaining, n - pos)
            stop = cursor + span
            segments.append(
                Segment(
                    source=source,
                    copy=copy if measured else 0,
                    first=cursor,
                    stop=stop,
                    measured=measured,
                    uid=self._uid(),
                    est_s=sum(self.turn_est(source, c) for c in range(cursor, stop)),
                )
            )
            remaining -= span
            pos = (pos + span) % n
        return segments

    def next_padding(self, program: SlotProgram) -> Segment:
        """The next unmeasured segment of the slot's own lane."""
        segment = self.walk(program, program.padding_pos, 1, measured=False)[0]
        # Extend to the end of the instance (walk stops after one turn).
        inst_idx, cursor = program.positions[program.padding_pos]
        source, _copy = program.lane[inst_idx]
        conv_len = len(self._sources[source].turns)
        stop = min(conv_len, cursor + (program.n_turns - program.padding_pos))
        segment.stop = stop
        segment.est_s = sum(self.turn_est(source, c) for c in range(cursor, stop))
        program.padding_pos = (program.padding_pos + (stop - cursor)) % program.n_turns
        return segment


def _position_at(cum: list[float], at: float) -> int:
    """Largest index whose cumulative start time is <= ``at``."""
    lo, hi = 0, len(cum) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cum[mid] <= at:
            lo = mid
        else:
            hi = mid - 1
    return lo


@dataclass(slots=True)
class _Slot:
    index: int
    program: SlotProgram | None = None
    phase: str = "calibration"  # calibration -> run
    current: Segment | None = None
    cursor: int = 0
    inflight: str | None = None
    delay_handle: asyncio.TimerHandle | None = None
    first_request_done: bool = False
    switch_pending: bool = False
    measured_est_s: float = 0.0  # estimate of unstarted measured segments
    lap_start_ns: int | None = None
    lap_end_ns: int | None = None
    measured_turns: int = 0
    lap_turns_planned: int = 0


@dataclass(slots=True)
class _Inflight:
    slot: int
    issue_ns: int
    measured: bool
    domain: str
    cold: bool
    conversation_id: str
    turn: int


class LaneAgenticInferenceStrategy(AgenticInferenceStrategy):
    """Agentic load strategy for ``agentic_inference.scheduler: lanes``."""

    def __init__(
        self,
        dataset_metadata: ConversationMetadata,
        agentic_inference_config: AgenticInferenceConfig,
        target_concurrency: int,
        rng_sample_index: random.Random | None = None,
        report_dir: Path | None = None,
        run_key: RunKey | None = None,
    ):
        super().__init__(
            conversation_manager=ConversationManager(),
            dataset_metadata=dataset_metadata,
            agentic_inference_config=agentic_inference_config,
            target_concurrency=target_concurrency,
            rng_sample_index=rng_sample_index,
        )
        if not target_concurrency or target_concurrency <= 0:
            raise InputValidationError(
                "scheduler: lanes requires a positive target_concurrency"
            )
        self._cfg = agentic_inference_config.lanes
        self._inject_delay = agentic_inference_config.inject_tool_delay
        self._concurrency = target_concurrency
        self._report_dir = report_dir
        seed = rng_sample_index.getrandbits(64) if rng_sample_index is not None else 0
        self._key = run_key if run_key is not None else RunKey.generate(seed)
        self._sources: list[SourceConversation] = []
        self._slots: list[_Slot] = []
        self._lane_inflight: dict[str, _Inflight] = {}
        self._planner: LanePlanner | None = None
        self._copies = 0
        self._measured_total = 0
        self._measured_completed = 0
        self._measured_failed = 0
        self._unmeasured_serial = 0
        self._halted = False
        self._calibrating = True
        self._cold_done_ns: int | None = None
        self._first_left = 0
        self._calib_samples: list[tuple[int, str, float]] = []
        self._service_s: dict[str, float] = {}
        self._calibration_counts: dict[str, int] = {}
        self._calibration_window_ns: tuple[int, int] | None = None
        self._steals: list[dict[str, Any]] = []
        self._run_started_ns = 0
        self._plan_ns: int | None = None
        self._halt_ns: int | None = None
        self._lanes_json: list[list[tuple[int, int]]] = []

    # ------------------------------------------------------------------ setup

    def _load_sources(self) -> None:
        conv_samples: dict[str, list[tuple[int, int]]] = {}
        for sample_meta in self._dataset_metadata.samples:
            assert sample_meta.sample_index is not None
            conv_samples.setdefault(sample_meta.conversation_id, []).append(
                (sample_meta.sample_index, sample_meta.turn)
            )
        delay_map = self._dataset_metadata.delay_seconds_by_key
        self._sources = []
        for source_id, turns in conv_samples.items():
            ordered = tuple(sorted(turns, key=lambda x: x[1]))
            delays = tuple(
                float(delay_map.get((source_id, turn), 0.0))
                if self._inject_delay
                else 0.0
                for _idx, turn in ordered
            )
            self._sources.append(
                SourceConversation(
                    source_id=source_id,
                    turns=ordered,
                    delays=delays,
                    domain=conversation_domain(source_id),
                )
            )
        self._base_convs = [(s.source_id, list(s.turns)) for s in self._sources]

    def _validate_copies(self) -> None:
        n_coding = sum(1 for s in self._sources if s.domain == CODING)
        self._copies = lane_copies(n_coding, self._concurrency, self._cfg)
        requested = self._num_trajectories_to_issue
        expected = self._copies * len(self._sources)
        if requested is not None and requested != expected:
            raise InputValidationError(
                f"scheduler: lanes measures {self._copies} copies at concurrency "
                f"{self._concurrency} ({expected} trajectories); set "
                f"num_trajectories_to_issue to {expected} or omit it (got {requested})"
            )
        self._measured_total = self._copies * sum(len(s.turns) for s in self._sources)

    # -------------------------------------------------------------- execution

    async def execute(self, phase_issuer: PhaseIssuerProtocol) -> int:
        self._phase_issuer = phase_issuer
        self._loop = asyncio.get_running_loop()
        self._all_done = asyncio.Event()
        self._error = None
        self._stopping = False
        self._run_started_ns = time.monotonic_ns()
        self._load_sources()
        if not self._sources:
            return phase_issuer.issued_count
        self._validate_salt_system_prompts()
        self._validate_copies()
        self._write_json(
            "lane_commitment.json",
            {"secret_sha256": self._key.commitment, "seed": self._key.seed},
        )
        logger.info(
            "Lane scheduler: %d copies over %d lanes; run key commitment sha256=%s",
            self._copies,
            self._concurrency,
            self._key.commitment,
        )

        provisional = LanePlanner(
            self._sources,
            {CODING: _PROVISIONAL_SERVICE_S, WORKFLOW: _PROVISIONAL_SERVICE_S},
            "cal",
        )
        rng = self._key.rng("calibration-plan")
        # Same copy count as the real plan, so every slot has load to calibrate with.
        lanes = provisional.pack(self._copies, self._concurrency, rng)
        programs = provisional.programs(lanes, rng, 0.0, 0.0, measure=False)
        self._planner = provisional
        self._slots = [
            _Slot(index=j, program=programs[j]) for j in range(self._concurrency)
        ]
        self._first_left = sum(1 for p in programs if p is not None)
        try:
            for slot in self._slots:
                if slot.program is not None:
                    self._advance(slot, first=True)
            if not self._lane_inflight and not self._has_pending_delays():
                return phase_issuer.issued_count
            await self._all_done.wait()
            if self._error is not None:
                raise self._error
            return phase_issuer.issued_count
        finally:
            for slot in self._slots:
                if slot.delay_handle is not None:
                    slot.delay_handle.cancel()
                    slot.delay_handle = None
            for handle in self._timeout_handles.values():
                handle.cancel()
            self._timeout_handles.clear()
            self._write_plan()

    def _has_pending_delays(self) -> bool:
        return any(slot.delay_handle is not None for slot in self._slots)

    def _finish_calibration(self, now_ns: int) -> None:
        assert self._cold_done_ns is not None
        start = self._cold_done_ns + int(self._cfg.calibration_skip_s * 1e9)
        end = start + int(self._cfg.calibration_s * 1e9)
        window = [
            (d, s) for issue_ns, d, s in self._calib_samples if start <= issue_ns < end
        ]
        pooled = [s for _d, s in window]
        pooled_mean = sum(pooled) / len(pooled) if pooled else _PROVISIONAL_SERVICE_S
        for domain in (CODING, WORKFLOW):
            values = [s for d, s in window if d == domain]
            self._calibration_counts[domain] = len(values)
            self._service_s[domain] = (
                sum(values) / len(values)
                if len(values) >= _MIN_DOMAIN_SAMPLES
                else pooled_mean
            )
        self._calibration_window_ns = (start, end)
        self._calib_samples.clear()
        self._calibrating = False
        self._plan_ns = now_ns

        planner = LanePlanner(self._sources, self._service_s, "run")
        rng = self._key.rng("plan")
        lanes = planner.pack(self._copies, self._concurrency, rng)
        programs = planner.programs(
            lanes, rng, self._cfg.warmup_min_s, self._cfg.warmup_max_s, measure=True
        )
        self._planner = planner
        self._lanes_json = lanes
        logger.info(
            "Lane scheduler: measured service time coding=%.3fs (n=%d) workflow=%.3fs (n=%d); plan built",
            self._service_s[CODING],
            self._calibration_counts[CODING],
            self._service_s[WORKFLOW],
            self._calibration_counts[WORKFLOW],
        )
        for slot in self._slots:
            program = programs[slot.index]
            slot.program = program
            slot.phase = "run"
            slot.current = None
            if program is None:
                if slot.delay_handle is not None:
                    slot.delay_handle.cancel()
                    slot.delay_handle = None
                continue
            slot.measured_est_s = sum(seg.est_s for seg in program.measured)
            slot.lap_turns_planned = program.n_turns
            if slot.inflight is not None:
                slot.switch_pending = True
            else:
                if slot.delay_handle is not None:
                    slot.delay_handle.cancel()
                    slot.delay_handle = None
                self._advance(slot, first=True)

    # ------------------------------------------------------------ slot logic

    def _logical_id(self, segment: Segment) -> str:
        source_id = self._sources[segment.source].source_id
        if segment.measured:
            return (
                source_id
                if segment.copy == 1
                else f"{source_id}__repeat_{segment.copy}"
            )
        self._unmeasured_serial += 1
        return f"{source_id}__unmeasured_{self._unmeasured_serial}"

    def _next_segment(self, slot: _Slot) -> Segment:
        program = slot.program
        assert program is not None and self._planner is not None
        if slot.phase == "run":
            if program.warmup:
                return program.warmup.popleft()
            if program.measured:
                segment = program.measured.popleft()
                slot.measured_est_s -= segment.est_s
                return segment
            if self._cfg.work_stealing and not self._halted:
                stolen = self._steal_for(slot)
                if stolen is not None:
                    return stolen
        return self._planner.next_padding(program)

    def _remaining_est(self, slot: _Slot) -> float:
        remaining = slot.measured_est_s
        segment = slot.current
        if segment is not None and segment.measured and self._planner is not None:
            remaining += sum(
                self._planner.turn_est(segment.source, c)
                for c in range(slot.cursor, segment.stop)
            )
        return remaining

    def _steal_for(self, thief: _Slot) -> Segment | None:
        """Take the last unstarted measured segment of the slot furthest behind,
        when moving it lets that slot finish no later than the thief."""
        victim: _Slot | None = None
        victim_remaining = 0.0
        for slot in self._slots:
            if slot is thief or slot.program is None or not slot.program.measured:
                continue
            remaining = self._remaining_est(slot)
            if remaining > victim_remaining:
                victim, victim_remaining = slot, remaining
        if victim is None:
            return None
        assert victim.program is not None
        candidate = victim.program.measured[-1]
        if candidate.est_s > victim_remaining - candidate.est_s:
            return None
        victim.program.measured.pop()
        victim.measured_est_s -= candidate.est_s
        self._steals.append(
            {
                "t_ns": time.monotonic_ns(),
                "thief": thief.index,
                "victim": victim.index,
                "source_id": self._sources[candidate.source].source_id,
                "copy": candidate.copy,
                "turns": [candidate.first, candidate.stop],
                "victim_remaining_est_s": round(victim_remaining, 3),
            }
        )
        return candidate

    def _advance(self, slot: _Slot, first: bool = False) -> None:
        """Schedule the slot's next turn (after its tool delay)."""
        if self._halted or self._stopping or slot.program is None:
            return
        segment = slot.current
        if segment is None or slot.cursor >= segment.stop:
            segment = self._next_segment(slot)
            segment.logical_id = self._logical_id(segment)
            slot.current = segment
            slot.cursor = segment.first
        delay = 0.0 if first else self._sources[segment.source].delays[slot.cursor]
        if delay > 0.0:
            assert self._loop is not None
            slot.delay_handle = self._loop.call_later(delay, self._issue_safely, slot)
        else:
            self._issue(slot)

    def _issue_safely(self, slot: _Slot) -> None:
        slot.delay_handle = None
        try:
            self._issue(slot)
        except Exception as exc:
            logger.exception("Lane scheduler: error issuing for slot %d", slot.index)
            self._error = exc
            if self._all_done is not None:
                self._all_done.set()

    def _issue(self, slot: _Slot) -> None:
        if self._halted or self._stopping:
            self._signal_done_if_idle()
            return
        segment = slot.current
        assert segment is not None and self._phase_issuer is not None
        source = self._sources[segment.source]
        idx, turn = source.turns[slot.cursor]
        cold = slot.cursor == segment.first
        query_id = self._phase_issuer.issue(
            idx,
            data_override=self._salted_override(source.source_id, turn, segment.uid),
            conversation_id=segment.logical_id,
            turn=turn,
            routing_key=self._key.digest("route", segment.uid, 16),
            tracked=segment.measured,
        )
        if query_id is None:
            self._request_stop_issuing()
            return
        now = time.monotonic_ns()
        slot.cursor += 1
        slot.inflight = query_id
        if segment.measured and slot.lap_start_ns is None:
            slot.lap_start_ns = now
        self._lane_inflight[query_id] = _Inflight(
            slot=slot.index,
            issue_ns=now,
            measured=segment.measured,
            domain=source.domain,
            cold=cold,
            conversation_id=segment.logical_id,
            turn=turn,
        )
        assert self._loop is not None
        self._timeout_handles[query_id] = self._loop.call_later(
            self._turn_timeout_s, self._handle_lane_timeout, query_id
        )

    def _salted_override(self, source_id: str, turn: int, uid: str) -> dict[str, Any]:
        messages = self._dataset_metadata.pre_built_messages_by_key.get(
            (source_id, turn)
        )
        if not messages:
            raise InputValidationError(
                f"scheduler: lanes requires pre-built messages; {source_id!r} turn {turn} has none"
            )
        salted = [dict(message) for message in messages]
        for message in salted:
            if message.get("role") == "system" and isinstance(
                message.get("content"), str
            ):
                head = self._key.digest("salt-head", uid)
                tail = self._key.digest("salt-tail", uid)
                message["content"] = (
                    f"[salt: {head}]\n\n{message['content']}\n\n[salt: {tail}]"
                )
                return {"messages": salted, "input_tokens": None, "token_ids": None}
        raise InputValidationError(
            f"scheduler: lanes requires a system prompt; {source_id!r} has none"
        )

    # ----------------------------------------------------------- completions

    def on_sample_complete(self, result: QueryResult) -> None:
        info = self._lane_inflight.pop(result.id, None)
        if info is None:
            logger.debug("Lane scheduler: dropping late response %s", result.id)
            return
        handle = self._timeout_handles.pop(result.id, None)
        if handle is not None:
            handle.cancel()
        try:
            self._complete(info, failed=result.error is not None)
        except Exception as exc:
            logger.exception("Lane scheduler: error completing %s", result.id)
            self._error = exc
            if self._all_done is not None:
                self._all_done.set()

    def _handle_lane_timeout(self, query_id: str) -> None:
        info = self._lane_inflight.pop(query_id, None)
        if info is None:
            return
        self._timeout_handles.pop(query_id, None)
        issuer = self._phase_issuer
        if issuer is not None and query_id in getattr(issuer, "uuid_to_index", {}):
            issuer.mark_inflight_complete()
            if hasattr(issuer, "completed_uuids"):
                issuer.completed_uuids.add(query_id)  # type: ignore[attr-defined]
            if hasattr(issuer, "uuid_to_conv_info"):
                issuer.uuid_to_conv_info.pop(query_id, None)  # type: ignore[attr-defined]
        logger.warning("Lane scheduler: turn timed out (query=%s)", query_id)
        self._publish_synthetic_failure(
            query_id,
            info.conversation_id,
            info.turn,
            error_type="TurnTimeout",
            error_message=f"turn timeout after {self._turn_timeout_s}s",
        )
        self._complete(info, failed=True)

    def _complete(self, info: _Inflight, failed: bool) -> None:
        now = time.monotonic_ns()
        slot = self._slots[info.slot]
        slot.inflight = None
        if not slot.first_request_done:
            slot.first_request_done = True
            self._first_left -= 1
            if self._first_left == 0 and self._cold_done_ns is None:
                self._cold_done_ns = now
        if self._calibrating and not info.cold and not failed:
            self._calib_samples.append(
                (info.issue_ns, info.domain, (now - info.issue_ns) / 1e9)
            )
        if info.measured:
            self._measured_completed += 1
            slot.measured_turns += 1
            slot.lap_end_ns = now
            if failed:
                self._measured_failed += 1
            if self._measured_completed >= self._measured_total:
                self._halt(now)
        if self._halted or self._stopping:
            self._signal_done_if_idle()
            return
        if (
            self._calibrating
            and self._cold_done_ns is not None
            and now
            >= self._cold_done_ns
            + int((self._cfg.calibration_skip_s + self._cfg.calibration_s) * 1e9)
        ):
            self._finish_calibration(now)
            if slot.switch_pending or slot.current is None:
                slot.switch_pending = False
                self._advance(slot, first=True)
            return
        if slot.switch_pending:
            slot.switch_pending = False
            self._advance(slot, first=True)
            return
        self._advance(slot)

    def _halt(self, now_ns: int) -> None:
        if self._halted:
            return
        self._halted = True
        self._halt_ns = now_ns
        for slot in self._slots:
            if slot.delay_handle is not None:
                slot.delay_handle.cancel()
                slot.delay_handle = None
        logger.info(
            "Lane scheduler: all %d measured turns complete", self._measured_total
        )

    def _request_stop_issuing(self) -> None:
        self._stopping = True
        for slot in self._slots:
            if slot.delay_handle is not None:
                slot.delay_handle.cancel()
                slot.delay_handle = None
        self._signal_done_if_idle()

    def _signal_done_if_idle(self) -> None:
        if not self._lane_inflight and self._all_done is not None:
            self._all_done.set()

    # ------------------------------------------------------------- metadata

    def _write_json(self, name: str, payload: dict[str, Any]) -> None:
        if self._report_dir is None:
            return
        self._report_dir.mkdir(parents=True, exist_ok=True)
        (self._report_dir / name).write_text(json.dumps(payload, indent=2) + "\n")

    def plan_metadata(self) -> dict[str, Any]:
        """Everything needed to rebuild and audit the plan, key included."""
        lap_starts = [s.lap_start_ns for s in self._slots if s.lap_start_ns is not None]
        lap_ends = [s.lap_end_ns for s in self._slots if s.lap_end_ns is not None]
        lap_s = [
            (s.lap_end_ns - s.lap_start_ns) / 1e9
            for s in self._slots
            if s.lap_start_ns is not None and s.lap_end_ns is not None
        ]
        return {
            "scheduler": "lanes",
            "secret_hex": self._key.secret.hex(),
            "secret_sha256": self._key.commitment,
            "seed": self._key.seed,
            "concurrency": self._concurrency,
            "copies": self._copies,
            "measured_turns_expected": self._measured_total,
            "measured_turns_completed": self._measured_completed,
            "measured_turns_failed": self._measured_failed,
            "config": self._cfg.model_dump(),
            "service_s": self._service_s,
            "calibration_samples": self._calibration_counts,
            "clock": "time.monotonic_ns, same clock as events.jsonl",
            "run_started_ns": self._run_started_ns,
            "cold_done_ns": self._cold_done_ns,
            "calibration_window_ns": list(self._calibration_window_ns)
            if self._calibration_window_ns
            else None,
            "plan_ns": self._plan_ns,
            "halt_ns": self._halt_ns,
            "lap_start_spread_s": (max(lap_starts) - min(lap_starts)) / 1e9
            if lap_starts
            else None,
            "lap_finish_spread_s": (max(lap_ends) - min(lap_ends)) / 1e9
            if lap_ends
            else None,
            "sum_lap_s": sum(lap_s),
            "steals": self._steals,
            "slots": [
                {
                    "slot": s.index,
                    "lane": [
                        [self._sources[src].source_id, copy]
                        for src, copy in self._lanes_json[s.index]
                    ]
                    if self._lanes_json
                    else [],
                    "entry_pos": s.program.entry_pos if s.program else None,
                    "lap_pos": s.program.lap_pos if s.program else None,
                    "warmup_est_s": round(s.program.warmup_est_s, 3)
                    if s.program
                    else None,
                    "lap_turns_planned": s.lap_turns_planned,
                    "measured_turns": s.measured_turns,
                    "lap_start_ns": s.lap_start_ns,
                    "lap_end_ns": s.lap_end_ns,
                }
                for s in self._slots
            ],
        }

    def _write_plan(self) -> None:
        try:
            self._write_json("lane_plan.json", self.plan_metadata())
        except Exception:
            logger.exception("Lane scheduler: could not write lane_plan.json")
