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

"""Balanced agentic scheduler (``agentic_inference.scheduler: lanes``).

The queue scheduler hands conversations to slots from one shared start queue.
Every slot starts at turn 1 at the same moment (a ramp), and the measured
window closes when the queue runs dry while the longest conversations are
still running (a tail that is never counted). This scheduler removes both and
measures every dataset turn exactly ``copies`` times:

1. Warm-up (unmeasured, ``warmup_s``). Each of the C slots enters a randomly
   chosen conversation at a random turn, so the server sees the context-length
   and workload mix of steady state from the first request, and keeps drawing
   random conversations from turn 1. Warm turns give the per-turn service time
   of coding and workflow turns.
2. Measurement start (one clock instant for every slot). Each slot's current
   conversation becomes measured from its next turn as copy ``k`` of that
   source; the turns it already ran go back into the measured pool as their
   own segment. Every other conversation instance (``copies`` per source) is in
   the pool. Measurement therefore starts with every slot in flight at a random
   point of a conversation: no ramp.
3. Run. A slot that finishes a measured segment takes the next one from the
   shared pool, so no slot idles while measured work is left. The pool order
   is random (from the secret key) with one constraint: an instance of
   estimated length ``L`` is drawn to start uniformly in ``[0, T - L]``, where
   ``T`` is the estimated measured time. Long SWE conversations therefore never
   start so late that they finish after the rest, while the mix stays close to
   uniform. Once the pool is empty, idle slots run unmeasured padding so the
   server keeps C requests in flight until the last measured turn completes.

Unmeasured requests are ISSUED with ``tracked=False`` and an
``__unmeasured_<n>`` conversation id, so no metric, OSL statistic or accuracy
score counts them. The salt's head depends only on the copy (shared by every
conversation of that copy, so the common system prompt stays cached, as with
the queue scheduler's salt); its tail is unique per segment. The key is
generated at run start, never configured, and written to ``lane_plan.json``
only after the last request.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass
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
# Per-turn service-time estimate before any warm turn has completed.
_DEFAULT_SERVICE_S = 5.0
# Fewer warm samples than this keep a flat (context-independent) estimate.
_MIN_WARM_SAMPLES = 30
# No measured segment is estimated longer than this fraction of the measured
# phase; longer conversation instances are measured as several segments. The
# pool order is then uniformly random (the workload mix stays steady), and a
# segment that starts last still finishes soon after the pool empties.
_MAX_SEGMENT_FRACTION = 1 / 8


def _context_chars(messages: list[dict[str, Any]]) -> int:
    """Prompt size of one request in characters: every message's text fields."""
    total = 0
    for message in messages:
        for key in ("content", "reasoning_content", "tool_calls"):
            value = message.get(key)
            if isinstance(value, str):
                total += len(value)
            elif value:
                total += len(json.dumps(value))
    return total


def conversation_domain(conversation_id: str) -> str:
    """Return ``workflow`` for ``sim_<n>`` conversations, else ``coding``."""
    return WORKFLOW if _WORKFLOW_CONVERSATION_RE.match(conversation_id) else CODING


def lane_copies(
    n_coding_conversations: int, concurrency: int, cfg: LaneSchedulerConfig
) -> int:
    """Dataset copies measured: about ``coding_conversations_per_slot`` coding
    conversations per slot, never fewer than ``min_copies``. Enough measured
    work that the run is long compared with the longest conversation."""
    if n_coding_conversations == 0:
        return cfg.min_copies
    return max(
        cfg.min_copies,
        math.ceil(
            cfg.coding_conversations_per_slot * concurrency / n_coding_conversations
        ),
    )


def split_points(turn_estimates: list[float], max_len_s: float) -> list[int]:
    """Cut positions that divide a run of turns into the fewest chunks of about
    equal estimated length, none longer than ``max_len_s`` (a single turn is
    never cut). Returns the interior cut indices, ascending."""
    total = sum(turn_estimates)
    if max_len_s <= 0 or total <= max_len_s or len(turn_estimates) < 2:
        return []
    chunks = min(len(turn_estimates), math.ceil(total / max_len_s))
    cuts: list[int] = []
    acc = 0.0
    for i, est in enumerate(turn_estimates[:-1], start=1):
        acc += est
        if acc >= total * (len(cuts) + 1) / chunks:
            cuts.append(i)
            if len(cuts) == chunks - 1:
                break
    return cuts


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
    """One dataset conversation."""

    source_id: str
    # (sample_index, turn) in turn order.
    turns: tuple[tuple[int, int], ...]
    # Tool delay before each turn (0 when tool delay is not injected).
    delays: tuple[float, ...]
    # Prompt size of each turn in characters (service time grows with it).
    context_chars: tuple[int, ...]
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
    logical_id: str = ""


@dataclass(slots=True)
class _Slot:
    index: int
    current: Segment | None = None
    cursor: int = 0
    inflight: str | None = None
    delay_handle: asyncio.TimerHandle | None = None
    first_request_done: bool = False
    measured_turns: int = 0
    first_measured_issue_ns: int | None = None
    last_measured_complete_ns: int | None = None


@dataclass(slots=True)
class _Inflight:
    slot: int
    issue_ns: int
    measured: bool
    domain: str
    context_chars: int
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
        self._source_weights: list[float] = []
        self._slots: list[_Slot] = []
        self._lane_inflight: dict[str, _Inflight] = {}
        self._pool: deque[Segment] = deque()
        self._pool_order: list[list[Any]] = []
        self._copies = 0
        self._measured_total = 0
        self._measured_completed = 0
        self._measured_failed = 0
        self._unmeasured_serial = 0
        self._uid_counter = 0
        self._measuring = False
        self._halted = False
        # Warm (context_chars, service seconds) samples, and the fitted model
        # service = intercept + slope * context_chars.
        self._warm_samples: list[tuple[int, float]] = []
        self._service_intercept_s = _DEFAULT_SERVICE_S
        self._service_per_char_s = 0.0
        self._warm_count = 0
        self._horizon_s = 0.0
        self._converted = 0
        self._splits = 0
        self._run_started_ns = 0
        self._measure_ns: int | None = None
        self._pool_empty_ns: int | None = None
        self._halt_ns: int | None = None
        self._entry_rng = self._key.rng("entry")
        self._padding_rng = self._key.rng("padding")
        self._measure_timer: asyncio.TimerHandle | None = None

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
            messages_by_key = self._dataset_metadata.pre_built_messages_by_key
            context_chars = tuple(
                _context_chars(messages_by_key.get((source_id, turn)) or [])
                for _idx, turn in ordered
            )
            self._sources.append(
                SourceConversation(
                    source_id=source_id,
                    turns=ordered,
                    delays=delays,
                    context_chars=context_chars,
                    domain=conversation_domain(source_id),
                )
            )
        self._base_convs = [(s.source_id, list(s.turns)) for s in self._sources]
        # Steady-state entry: a slot is in a source in proportion to the time
        # that source takes, at a turn uniform over its turns.
        self._source_weights = [
            self._estimate(i, 0, len(s.turns)) for i, s in enumerate(self._sources)
        ]

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

    def _estimate(self, source: int, first: int, stop: int) -> float:
        src = self._sources[source]
        return (
            sum(src.delays[first:stop])
            + self._service_intercept_s * (stop - first)
            + self._service_per_char_s * sum(src.context_chars[first:stop])
        )

    def _fit_service_model(self) -> None:
        """Least-squares service = intercept + slope * context_chars over warm
        turns. Long-context turns cost more; a per-domain mean would make long
        SWE conversations look short and start them too late."""
        samples = self._warm_samples
        self._warm_count = len(samples)
        if len(samples) < _MIN_WARM_SAMPLES:
            if samples:
                self._service_intercept_s = sum(s for _x, s in samples) / len(samples)
            return
        n = len(samples)
        mx = sum(x for x, _s in samples) / n
        my = sum(s for _x, s in samples) / n
        sxx = sum((x - mx) ** 2 for x, _s in samples)
        sxy = sum((x - mx) * (s - my) for x, s in samples)
        slope = max(0.0, sxy / sxx) if sxx > 0 else 0.0
        self._service_per_char_s = slope
        self._service_intercept_s = max(0.0, my - slope * mx)

    def _segment(
        self, source: int, copy: int, first: int, stop: int, measured: bool
    ) -> Segment:
        self._uid_counter += 1
        return Segment(
            source=source,
            copy=copy,
            first=first,
            stop=stop,
            measured=measured,
            uid=f"seg:{self._uid_counter}",
        )

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
            "Lane scheduler: %d copies, %d measured turns over %d slots; "
            "run key commitment sha256=%s",
            self._copies,
            self._measured_total,
            self._concurrency,
            self._key.commitment,
        )
        self._slots = [_Slot(index=j) for j in range(self._concurrency)]
        try:
            for slot in self._slots:
                slot.current = self._entry_segment()
                slot.cursor = self._entry_cursor(slot.current)
                self._advance(slot, first=True)
            self._measure_timer = self._loop.call_later(
                self._cfg.warmup_s, self._start_measurement_safely
            )
            await self._all_done.wait()
            if self._error is not None:
                raise self._error
            return phase_issuer.issued_count
        finally:
            if self._measure_timer is not None:
                self._measure_timer.cancel()
            for slot in self._slots:
                if slot.delay_handle is not None:
                    slot.delay_handle.cancel()
                    slot.delay_handle = None
            for handle in self._timeout_handles.values():
                handle.cancel()
            self._timeout_handles.clear()
            self._write_plan()

    def _entry_segment(self) -> Segment:
        source = self._entry_rng.choices(
            range(len(self._sources)), weights=self._source_weights
        )[0]
        segment = self._segment(
            source, 0, 0, len(self._sources[source].turns), measured=False
        )
        segment.logical_id = self._logical_id(segment)
        return segment

    def _entry_cursor(self, segment: Segment) -> int:
        return self._entry_rng.randrange(segment.stop)

    def _start_measurement_safely(self) -> None:
        self._measure_timer = None
        try:
            self._start_measurement()
        except Exception as exc:
            logger.exception("Lane scheduler: could not start measurement")
            self._error = exc
            if self._all_done is not None:
                self._all_done.set()

    def _start_measurement(self) -> None:
        if self._halted or self._stopping:
            return
        now = time.monotonic_ns()
        self._measure_ns = now
        self._fit_service_model()
        self._warm_samples.clear()

        rng = self._key.rng("measure")
        free = {s: list(range(1, self._copies + 1)) for s in range(len(self._sources))}
        heads: list[Segment] = []
        running_est = 0.0
        # Every slot measures from its next turn: its current conversation
        # becomes copy k of that source, and the turns it already ran go back
        # into the pool as their own measured segment.
        for slot in self._slots:
            segment = slot.current
            if (
                segment is None
                or slot.cursor >= segment.stop
                or not free[segment.source]
            ):
                # Nothing to convert (its conversation ends with the in-flight
                # turn, or every copy of the source is already taken): drop the
                # unmeasured conversation so the slot takes measured work next.
                slot.current = None
                if slot.delay_handle is not None:
                    slot.delay_handle.cancel()
                    slot.delay_handle = None
                continue
            copy = free[segment.source].pop(rng.randrange(len(free[segment.source])))
            remainder = self._segment(
                segment.source, copy, slot.cursor, segment.stop, measured=True
            )
            remainder.logical_id = self._logical_id(remainder)
            slot.current = remainder
            running_est += self._estimate(segment.source, slot.cursor, segment.stop)
            if slot.cursor > 0:
                heads.append(
                    self._segment(segment.source, copy, 0, slot.cursor, measured=True)
                )
            self._converted += 1
        items = heads + [
            self._segment(s, copy, 0, len(self._sources[s].turns), measured=True)
            for s, copies in free.items()
            for copy in copies
        ]
        estimates = [self._estimate(seg.source, seg.first, seg.stop) for seg in items]
        self._horizon_s = (running_est + sum(estimates)) / self._concurrency
        max_len = _MAX_SEGMENT_FRACTION * self._horizon_s
        pool: list[Segment] = []
        for seg in items:
            per_turn = [
                self._estimate(seg.source, c, c + 1) for c in range(seg.first, seg.stop)
            ]
            bounds = (
                [seg.first]
                + [seg.first + cut for cut in split_points(per_turn, max_len)]
                + [seg.stop]
            )
            if len(bounds) == 2:
                pool.append(seg)
                continue
            self._splits += len(bounds) - 2
            for first, stop in zip(bounds, bounds[1:], strict=False):
                pool.append(
                    self._segment(seg.source, seg.copy, first, stop, measured=True)
                )
        rng.shuffle(pool)
        self._pool = deque(pool)
        self._pool_order = [
            [self._sources[seg.source].source_id, seg.copy, seg.first, seg.stop]
            for seg in self._pool
        ]
        self._measuring = True
        logger.info(
            "Lane scheduler: measurement started; service = %.3fs + %.3fs per 10k "
            "prompt chars (n=%d warm turns); %d slots measure from their next turn, "
            "%d segments pooled (%d cuts in long instances), estimated %.0fs",
            self._service_intercept_s,
            self._service_per_char_s * 1e4,
            self._warm_count,
            self._converted,
            len(self._pool),
            self._splits,
            self._horizon_s,
        )
        # A slot whose conversation ended with its in-flight turn, or that is
        # waiting out a tool delay before a new unmeasured conversation, starts
        # measured work now rather than finishing the unmeasured one.
        for slot in self._slots:
            if slot.inflight is None and slot.delay_handle is None:
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

    def _next_segment(self) -> Segment:
        if self._measuring and self._pool:
            segment = self._pool.popleft()
            if not self._pool:
                self._pool_empty_ns = time.monotonic_ns()
        else:
            source = self._padding_rng.randrange(len(self._sources))
            segment = self._segment(
                source, 0, 0, len(self._sources[source].turns), measured=False
            )
        segment.logical_id = self._logical_id(segment)
        return segment

    def _advance(self, slot: _Slot, first: bool = False) -> None:
        """Schedule the slot's next turn (after its tool delay)."""
        if self._halted or self._stopping:
            return
        segment = slot.current
        if segment is None or slot.cursor >= segment.stop:
            segment = self._next_segment()
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
            data_override=self._salted_override(
                source.source_id, turn, segment.copy, segment.uid
            ),
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
        if segment.measured and slot.first_measured_issue_ns is None:
            slot.first_measured_issue_ns = now
        self._lane_inflight[query_id] = _Inflight(
            slot=slot.index,
            issue_ns=now,
            measured=segment.measured,
            domain=source.domain,
            context_chars=source.context_chars[slot.cursor - 1],
            cold=cold,
            conversation_id=segment.logical_id,
            turn=turn,
        )
        assert self._loop is not None
        self._timeout_handles[query_id] = self._loop.call_later(
            self._turn_timeout_s, self._handle_lane_timeout, query_id
        )

    def _salted_override(
        self, source_id: str, turn: int, copy: int, uid: str
    ) -> dict[str, Any]:
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
                # Same shape as the queue scheduler's salt: the head depends only
                # on the copy, so every conversation of a copy shares the cached
                # system-prompt prefix (500 workflow conversations share one
                # ~127k-character prompt). The tail is unique per segment, so
                # conversation histories never reuse another instance's cache.
                head = self._key.digest("salt-head", str(copy))
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
        first = not slot.first_request_done
        slot.first_request_done = True
        if not self._measuring and not info.cold and not first and not failed:
            self._warm_samples.append((info.context_chars, (now - info.issue_ns) / 1e9))
        if info.measured:
            self._measured_completed += 1
            slot.measured_turns += 1
            slot.last_measured_complete_ns = now
            if failed:
                self._measured_failed += 1
            if self._measured_completed >= self._measured_total:
                self._halt(now)
        if self._halted or self._stopping:
            self._signal_done_if_idle()
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
        """Everything needed to audit the run, key included."""
        starts = [
            s.first_measured_issue_ns
            for s in self._slots
            if s.first_measured_issue_ns is not None
        ]
        ends = [
            s.last_measured_complete_ns
            for s in self._slots
            if s.last_measured_complete_ns is not None
        ]

        def seconds(a: int | None, b: int | None) -> float | None:
            return (b - a) / 1e9 if a is not None and b is not None else None

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
            "service_model": {
                "intercept_s": self._service_intercept_s,
                "per_10k_chars_s": self._service_per_char_s * 1e4,
                "warm_samples": self._warm_count,
            },
            "clock": "time.monotonic_ns, same clock as events.jsonl",
            "run_started_ns": self._run_started_ns,
            "measure_ns": self._measure_ns,
            "pool_empty_ns": self._pool_empty_ns,
            "halt_ns": self._halt_ns,
            "estimated_measured_s": self._horizon_s,
            "measured_phase_s": seconds(self._measure_ns, self._halt_ns),
            "tail_after_pool_empty_s": seconds(self._pool_empty_ns, self._halt_ns),
            "slots_measured_from_next_turn": self._converted,
            "split_segments": self._splits,
            "measured_start_spread_s": (max(starts) - min(starts)) / 1e9
            if starts
            else None,
            "measured_finish_spread_s": (max(ends) - min(ends)) / 1e9 if ends else None,
            "pool_order": self._pool_order,
            "slots": [
                {
                    "slot": s.index,
                    "measured_turns": s.measured_turns,
                    "first_measured_issue_ns": s.first_measured_issue_ns,
                    "last_measured_complete_ns": s.last_measured_complete_ns,
                }
                for s in self._slots
            ],
        }

    def _write_plan(self) -> None:
        try:
            self._write_json("lane_plan.json", self.plan_metadata())
        except Exception:
            logger.exception("Lane scheduler: could not write lane_plan.json")
