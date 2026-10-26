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

"""Stationary agentic scheduler (``agentic_inference.scheduler: lanes``).

The queue scheduler starts every conversation at turn 1 at the same moment (a
ramp of short prompts that the window counts) and stops counting when its
start queue runs dry while the longest conversations still run (late
long-context turns that the window never counts). This scheduler starts the
system in its steady state and measures a fixed amount of work by issue time:

1. Stationary start (unmeasured, ``warmup_s``). Each of the C slots enters a
   conversation drawn with probability proportional to its estimated length,
   at a point uniform in time within it (the equilibrium state of a renewal
   process). The server therefore sees the steady-state mix of conversation
   types and context lengths from the first request; the warm-up only warms
   the prefix cache.
2. Steady load. A slot that finishes a conversation takes the next one from a
   shuffled deck of the whole dataset (one deck per pass), so the mix stays at
   its steady value for the whole run. No conversation is preferred by length.
3. Measured window. From the end of the warm-up, every turn issued is counted
   until ``passes`` x (dataset turns) have been issued, so the counted samples
   are a whole number of dataset passes. ``passes`` follows from the warm-up
   issue rate: enough passes to cover ``window_s``. Later turns are issued
   unmeasured so the counted turns complete under the same load, and the run
   stops once every counted turn has completed. Conversations still running
   at that point are not continued: there is no tail to wait for.

Three switches (``lanes.*``; defaults reproduce the behaviour above), so their
effect can be measured separately:

- ``entry: calibrated`` replaces the flat per-turn service prior with a fit
  ``latency = a + b*output + c*prefill`` over the first ``calibration_s`` of
  completions, then moves each slot to a fresh entry drawn with the fitted
  weights as its current request completes.
- ``warmup: adaptive`` opens the window once the per-interval completion rate
  and median latency stop changing, instead of after a fixed ``warmup_s``.
- ``stop: precision`` adds passes until the batch-means 95% confidence
  interval of throughput is narrower than ``precision_target``.

``lane_plan.json`` records the fit, the warm-up intervals, each pass-boundary
decision and the interval at close for every run.

Every turn of the dataset is counted ``passes`` times on average; the exact
count per turn varies because the window cuts conversations at its edges.
Counted turns use the conversation id ``<source>__repeat_<n>`` and others
``<source>__unmeasured_<n>``, where ``n`` numbers conversation instances, so no
metric, OSL statistic or accuracy score counts unmeasured work. The salt's head
depends only on the deck (shared by every conversation of a pass, so the common
system prompt stays cached, as with the queue scheduler's salt); its tail and
the routing key are unique per instance and never change mid-conversation.
The key is generated at run start, never configured, and written to
``lane_plan.json`` only after the last request.
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
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.schema import AgenticInferenceConfig
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
# Per-turn service-time prior for the stationary entry weights (entry: prior).
# Only the relative weight of service against recorded tool delay matters;
# GB300 C768 measured 6.3-7.2 s. entry: calibrated replaces it with a fit.
_PRIOR_SERVICE_S = 7.0
# Fewer warm completions than this keep the prior.
_MIN_FIT_SAMPLES = 30
# Two-sided 95% Student t quantiles by degrees of freedom (normal beyond 30).
_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31}
_T95.update({9: 2.26, 10: 2.23, 12: 2.18, 15: 2.13, 20: 2.09, 30: 2.04})


def conversation_domain(conversation_id: str) -> str:
    """Return ``workflow`` for ``sim_<n>`` conversations, else ``coding``."""
    return WORKFLOW if _WORKFLOW_CONVERSATION_RE.match(conversation_id) else CODING


def lane_passes(window_s: float, pass_s: float | None, min_passes: int) -> int:
    """Dataset passes to count: enough to cover ``window_s`` at ``pass_s`` per
    pass, never fewer than ``min_passes``. Without a rate, ``min_passes``."""
    if not pass_s or pass_s <= 0:
        return min_passes
    return max(min_passes, math.ceil(window_s / pass_s))


def _t95(df: int) -> float:
    for k in sorted(_T95):
        if df <= k:
            return _T95[k]
    return 1.96


def batch_mean_ci(
    events: list[tuple[int, float]], start_ns: int, end_ns: int, batch_ns: int
) -> tuple[int, float | None, float | None]:
    """Batch means over whole batches of ``events`` (time_ns, amount) in
    [start_ns, end_ns). Returns (batches, mean amount per second, 95% CI
    half-width relative to the mean); the last two are None below 2 batches."""
    n = (end_ns - start_ns) // batch_ns
    if n < 2:
        return int(n), None, None
    sums = [0.0] * n
    for t, amount in events:
        b = (t - start_ns) // batch_ns
        if 0 <= b < n:
            sums[b] += amount
    mean = statistics.fmean(sums)
    if mean <= 0:
        return int(n), 0.0, None
    half = _t95(n - 1) * statistics.stdev(sums) / math.sqrt(n)
    return int(n), mean / (batch_ns / 1e9), half / mean


def fit_service(
    samples: list[tuple[float, float, float]],
) -> tuple[float, float, float] | None:
    """Least squares ``latency = a + b*output + c*prefill`` with non-negative
    coefficients (a feature whose coefficient comes out negative is dropped and
    the rest refitted). ``samples`` are (latency_s, output size, prefill size)."""
    if len(samples) < _MIN_FIT_SAMPLES:
        return None
    import numpy as np

    y = np.array([s[0] for s in samples])
    x = np.array([[1.0, s[1], s[2]] for s in samples])
    active = [0, 1, 2]
    while active:
        coef, *_ = np.linalg.lstsq(x[:, active], y, rcond=None)
        negative = [f for f, c in zip(active, coef, strict=True) if c < 0]
        if not negative:
            out = [0.0, 0.0, 0.0]
            for f, c in zip(active, coef, strict=True):
                out[f] = float(c)
            return out[0], out[1], out[2]
        active.remove(negative[0])
    return None


def _message_chars(message: dict[str, Any]) -> int:
    content = message.get("content")
    size = len(content) if isinstance(content, str) else len(json.dumps(content))
    if message.get("tool_calls"):
        size += len(json.dumps(message["tool_calls"]))
    return size


def turn_sizes(
    messages_by_turn: list[list[dict[str, Any]]],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[float | None, ...]]:
    """Per client turn of one conversation: (new input chars since the
    previous turn, full context chars, recorded assistant reply chars or None
    for the last turn). Each turn's messages extend the previous turn's."""
    new, ctx, out = [], [], []
    total = 0
    for k, messages in enumerate(messages_by_turn):
        prev = len(messages_by_turn[k - 1]) if k else 0
        added = sum(_message_chars(m) for m in messages[prev:])
        total += added
        new.append(added)
        ctx.append(total)
        if k + 1 < len(messages_by_turn):
            reply = messages_by_turn[k + 1][len(messages) :]
            out.append(
                float(
                    sum(
                        _message_chars(m) for m in reply if m.get("role") == "assistant"
                    )
                )
            )
        else:
            out.append(None)
    return tuple(new), tuple(ctx), tuple(out)


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
    domain: str
    # Filled for entry: calibrated (see turn_sizes); empty otherwise.
    new_chars: tuple[int, ...] = ()
    ctx_chars: tuple[int, ...] = ()
    out_chars: tuple[float | None, ...] = ()


@dataclass(slots=True)
class _Instance:
    """One run of a source conversation on a slot."""

    source: int
    deck: int  # pass the instance was drawn from; 0 for stationary-entry instances
    serial: int
    uid: str


@dataclass(slots=True)
class _Slot:
    index: int
    current: _Instance | None = None
    cursor: int = 0
    inflight: str | None = None
    delay_handle: asyncio.TimerHandle | None = None
    cold: bool = False  # next issue is the first turn of a mid-conversation entry


@dataclass(slots=True)
class _Inflight:
    slot: int
    counted: bool
    conversation_id: str
    turn: int
    issue_ns: int
    # Prefill size (full context for a cold entry, else new input) and the
    # recorded reply size, for the entry: calibrated fit.
    prefill_chars: float
    recorded_out: float | None


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
        self._dataset_turns = 0
        self._slots: list[_Slot] = []
        self._lane_inflight: dict[str, _Inflight] = {}
        self._entry_rng = self._key.rng("entry")
        self._deck_rng = self._key.rng("deck")
        self._deck: list[int] = []
        self._deck_index = 0
        self._serial = 0
        self._measuring = False
        self._closed = False
        self._halted = False
        self._passes = 0
        self._pass_s: float | None = None
        self._target = 0
        self._counted_issued = 0
        self._counted_completed = 0
        self._counted_failed = 0
        self._counted_per_turn: Counter[tuple[int, int]] = Counter()
        self._run_started_ns = 0
        # Warm-up completions: (complete_ns, latency_s).
        self._warm_completions: list[tuple[int, float]] = []
        # entry: calibrated -- (latency_s, output chars, prefill chars, recorded chars).
        self._fit_samples: list[tuple[float, float, float, float | None]] = []
        self._fit: dict[str, Any] | None = None
        self._entry_weights: tuple[list[float], list[list[float]]] | None = None
        self._reseed_pending: set[int] = set()
        self._reseeding = False
        self._ready_ns: int | None = None  # entry (or reseed) settled
        self._warmup_intervals: list[dict[str, Any]] = []
        self._warmup_converged: bool | None = None
        # Window completions (all requests): (complete_ns, output chars).
        self._window_output: list[tuple[int, float]] = []
        self._pass_decisions: list[dict[str, Any]] = []
        self._precision_at_close: dict[str, Any] | None = None
        self._warmup_timer: asyncio.TimerHandle | None = None
        self._measure_ns: int | None = None
        self._window_end_ns: int | None = None
        self._halt_ns: int | None = None
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
            self._sources.append(
                SourceConversation(
                    source_id=source_id,
                    turns=ordered,
                    delays=delays,
                    domain=conversation_domain(source_id),
                    **self._sizes(source_id, ordered),
                )
            )
        self._base_convs = [(s.source_id, list(s.turns)) for s in self._sources]
        self._dataset_turns = sum(len(s.turns) for s in self._sources)

    def _sizes(
        self, source_id: str, ordered: tuple[tuple[int, int], ...]
    ) -> dict[str, tuple[Any, ...]]:
        if self._cfg.entry != "calibrated":
            return {}
        messages = self._dataset_metadata.pre_built_messages_by_key
        new, ctx, out = turn_sizes(
            [messages.get((source_id, turn)) or [] for _idx, turn in ordered]
        )
        return {"new_chars": new, "ctx_chars": ctx, "out_chars": out}

    def _validate_config(self) -> None:
        if self._num_trajectories_to_issue is not None:
            raise InputValidationError(
                "scheduler: lanes counts whole dataset passes chosen from the "
                "measured issue rate; omit num_trajectories_to_issue "
                f"(got {self._num_trajectories_to_issue})"
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
        self._validate_config()
        self._write_json(
            "lane_commitment.json",
            {"secret_sha256": self._key.commitment, "seed": self._key.seed},
        )
        logger.info(
            "Lane scheduler: stationary start over %d slots, %d dataset turns per "
            "pass; run key commitment sha256=%s",
            self._concurrency,
            self._dataset_turns,
            self._key.commitment,
        )
        self._slots = [_Slot(index=j) for j in range(self._concurrency)]
        self._ready_ns = self._run_started_ns
        try:
            for slot in self._slots:
                self._enter(slot)
            if self._cfg.entry == "calibrated":
                self._measure_timer = self._loop.call_later(
                    self._cfg.calibration_s, self._safely, self._calibrate
                )
            if self._cfg.warmup == "fixed":
                self._warmup_timer = self._loop.call_later(
                    self._cfg.warmup_s, self._safely, self._open_after_fixed_warmup
                )
            else:
                self._warmup_timer = self._loop.call_later(
                    self._cfg.warmup_interval_s, self._safely, self._warmup_tick
                )
            await self._all_done.wait()
            if self._error is not None:
                raise self._error
            return phase_issuer.issued_count
        finally:
            for timer in (self._measure_timer, self._warmup_timer):
                if timer is not None:
                    timer.cancel()
            for slot in self._slots:
                if slot.delay_handle is not None:
                    slot.delay_handle.cancel()
                    slot.delay_handle = None
            for handle in self._timeout_handles.values():
                handle.cancel()
            self._timeout_handles.clear()
            self._write_plan()

    def _safely(self, step: Any) -> None:
        try:
            step()
        except Exception as exc:
            logger.exception("Lane scheduler: %s failed", step.__name__)
            self._error = exc
            if self._all_done is not None:
                self._all_done.set()

    def _new_instance(self, source: int, deck: int) -> _Instance:
        self._serial += 1
        return _Instance(
            source=source, deck=deck, serial=self._serial, uid=f"i{self._serial}"
        )

    def _enter(self, slot: _Slot) -> None:
        """Put the slot at a stationary entry point and issue its turn."""
        slot.current, slot.cursor = self._stationary_entry()
        slot.cold = True
        self._issue(slot)

    def _stationary_entry(self) -> tuple[_Instance, int]:
        """A conversation in proportion to its length, at a point uniform in
        time: turn ``k`` with weight (service + tool delay before it)."""
        if self._entry_weights is not None:
            conv_weights, turn_weights = self._entry_weights
        else:
            conv_weights = [
                _PRIOR_SERVICE_S * len(s.turns) + sum(s.delays) for s in self._sources
            ]
            turn_weights = None
        source = self._entry_rng.choices(
            range(len(self._sources)), weights=conv_weights
        )[0]
        src = self._sources[source]
        cursor = self._entry_rng.choices(
            range(len(src.turns)),
            weights=(
                turn_weights[source]
                if turn_weights is not None
                else [_PRIOR_SERVICE_S + d for d in src.delays]
            ),
        )[0]
        return self._new_instance(source, deck=0), cursor

    def _next_instance(self) -> _Instance:
        if not self._deck:
            self._deck_index += 1
            self._deck = list(range(len(self._sources)))
            self._deck_rng.shuffle(self._deck)
        return self._new_instance(self._deck.pop(), deck=self._deck_index)

    def _calibrate(self) -> None:
        """entry: calibrated -- fit service time per turn, then reseed slots."""
        self._measure_timer = None
        if self._measuring or self._halted or self._stopping:
            return
        samples, self._fit_samples = self._fit_samples, []
        fit = fit_service([(lat, out, pre) for lat, out, pre, _ in samples])
        recorded = [(out, rec) for _lat, out, _pre, rec in samples if rec]
        rec_total = sum(rec for _out, rec in recorded)
        ratio = sum(out for out, _rec in recorded) / rec_total if rec_total else 1.0
        self._fit = {"samples": len(samples), "output_ratio": ratio}
        if fit is None:
            self._fit["error"] = "too few warm completions; kept the prior"
            logger.warning(
                "Lane scheduler: calibration skipped (%d samples)", len(samples)
            )
            return
        a, b, c = fit
        self._fit.update(intercept_s=a, s_per_output_char=b, s_per_prefill_char=c)
        known = [o for s in self._sources for o in s.out_chars if o is not None]
        global_out = statistics.fmean(known) if known else 0.0
        turn_weights = []
        for src in self._sources:
            src_known = [o for o in src.out_chars if o is not None]
            fill = statistics.fmean(src_known) if src_known else global_out
            turn_weights.append(
                [
                    max(0.0, a + b * ratio * (fill if out is None else out) + c * new)
                    + delay
                    for out, new, delay in zip(
                        src.out_chars, src.new_chars, src.delays, strict=True
                    )
                ]
            )
        self._entry_weights = ([sum(w) for w in turn_weights], turn_weights)
        self._fit["mean_turn_s"] = statistics.fmean(
            w - d
            for src, ws in zip(self._sources, turn_weights, strict=True)
            for w, d in zip(ws, src.delays, strict=True)
        )
        logger.info("Lane scheduler: calibrated entry %s; reseeding slots", self._fit)
        self._ready_ns = None
        self._reseeding = True
        self._reseed_pending = {slot.index for slot in self._slots}
        for slot in self._slots:
            if slot.delay_handle is not None:
                slot.delay_handle.cancel()
                slot.delay_handle = None
                self._reseed(slot)

    def _reseed(self, slot: _Slot) -> None:
        self._reseed_pending.discard(slot.index)
        if (
            self._ready_ns is None
            and len(self._reseed_pending) <= self._concurrency // 100
        ):
            self._ready_ns = time.monotonic_ns()
        if not self._reseed_pending:
            self._reseeding = False
        self._enter(slot)

    def _open_after_fixed_warmup(self) -> None:
        self._warmup_timer = None
        now = time.monotonic_ns()
        # Issue rate over the second half of the warm-up (the first half holds
        # the cold prefills of the stationary entry), and after any reseed.
        since = now - int(self._cfg.warmup_s * 1e9 / 2)
        if self._ready_ns is not None:
            since = max(since, self._ready_ns)
        self._start_window(since)

    def _warmup_tick(self) -> None:
        """warmup: adaptive -- open once completion rate and median latency
        have been steady for warmup_stable_intervals intervals."""
        self._warmup_timer = None
        if self._measuring or self._halted or self._stopping:
            return
        cfg = self._cfg
        now = time.monotonic_ns()
        interval_ns = int(cfg.warmup_interval_s * 1e9)
        latencies = [lat for t, lat in self._warm_completions if t >= now - interval_ns]
        settled = self._ready_ns is not None and now - interval_ns >= self._ready_ns
        self._warmup_intervals.append(
            {
                "t_s": (now - self._run_started_ns) / 1e9,
                "completions": len(latencies),
                "median_latency_s": statistics.median(latencies) if latencies else None,
                "settled": settled,
            }
        )
        k = cfg.warmup_stable_intervals
        recent = self._warmup_intervals[-(k + 1) :]
        stable = (
            self._ready_ns is not None
            and (now - self._ready_ns) / 1e9 >= cfg.warmup_min_s
            and len(recent) == k + 1
            and all(r["settled"] and r["completions"] for r in recent)
            and all(
                abs(cur[key] / prev[key] - 1.0) <= cfg.warmup_band
                for prev, cur in zip(recent, recent[1:], strict=False)
                for key in ("completions", "median_latency_s")
            )
        )
        if stable or (now - self._run_started_ns) / 1e9 >= cfg.warmup_max_s:
            self._warmup_converged = stable
            self._start_window(now - k * interval_ns)
            return
        assert self._loop is not None
        self._warmup_timer = self._loop.call_later(
            cfg.warmup_interval_s, self._safely, self._warmup_tick
        )

    def _start_window(self, rate_since_ns: int) -> None:
        if self._halted or self._stopping:
            return
        now = time.monotonic_ns()
        recent = sum(1 for t, _lat in self._warm_completions if t >= rate_since_ns)
        span_s = (now - rate_since_ns) / 1e9
        if recent and span_s > 0:
            self._pass_s = self._dataset_turns * span_s / recent
        if self._cfg.stop == "precision":
            self._passes = self._cfg.min_passes
        elif self._cfg.passes is not None:
            self._passes = self._cfg.passes
        else:
            self._passes = lane_passes(
                self._cfg.window_s, self._pass_s, self._cfg.min_passes
            )
        self._warm_completions.clear()
        self._target = self._passes * self._dataset_turns
        self._measure_ns = now
        self._measuring = True
        logger.info(
            "Lane scheduler: window open; %s s per pass from the warm-up, counting "
            "%d passes (%d turns), stop=%s",
            "-" if self._pass_s is None else f"{self._pass_s:.0f}",
            self._passes,
            self._target,
            self._cfg.stop,
        )

    def _window_ci(self, end_ns: int) -> dict[str, Any]:
        assert self._measure_ns is not None
        n, rate, half = batch_mean_ci(
            self._window_output,
            self._measure_ns,
            end_ns,
            int(self._cfg.precision_batch_s * 1e9),
        )
        return {"batches": n, "output_chars_per_s": rate, "ci95_half_width_rel": half}

    def _extend_window(self, now_ns: int) -> bool:
        """At a pass boundary: True to count one more pass (stop: precision)."""
        ci = self._window_ci(now_ns)
        decision = {"passes": self._passes, **ci}
        extend = (
            self._cfg.stop == "precision"
            and self._passes < self._cfg.max_passes
            and (
                ci["ci95_half_width_rel"] is None
                or ci["batches"] < self._cfg.precision_min_batches
                or ci["ci95_half_width_rel"] > self._cfg.precision_target
            )
        )
        decision["extend"] = extend
        self._pass_decisions.append(decision)
        if extend:
            self._passes += 1
            self._target += self._dataset_turns
        return extend

    # ------------------------------------------------------------ slot logic

    def _advance(self, slot: _Slot) -> None:
        """Schedule the slot's next turn after its tool delay; a finished
        conversation is replaced by the next one from the deck."""
        if self._halted or self._stopping:
            return
        instance = slot.current
        if instance is None or slot.cursor >= len(self._sources[instance.source].turns):
            slot.current = self._next_instance()
            slot.cursor = 0
        assert slot.current is not None
        delay = self._sources[slot.current.source].delays[slot.cursor]
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
        instance = slot.current
        assert instance is not None and self._phase_issuer is not None
        source = self._sources[instance.source]
        idx, turn = source.turns[slot.cursor]
        counted = self._measuring and not self._closed
        conversation_id = f"{source.source_id}__{'repeat' if counted else 'unmeasured'}_{instance.serial}"
        query_id = self._phase_issuer.issue(
            idx,
            data_override=self._salted_override(
                source.source_id, turn, instance.deck, instance.uid
            ),
            conversation_id=conversation_id,
            turn=turn,
            routing_key=self._key.digest("route", instance.uid, 16),
            tracked=counted,
        )
        if query_id is None:
            self._request_stop_issuing()
            return
        now = time.monotonic_ns()
        if counted:
            self._counted_issued += 1
            self._counted_per_turn[(instance.source, slot.cursor)] += 1
            if self._counted_issued >= self._target and not self._extend_window(now):
                self._closed = True
                self._window_end_ns = now
                self._precision_at_close = self._window_ci(now)
                logger.info(
                    "Lane scheduler: window closed after %d counted turns (%d passes)",
                    self._counted_issued,
                    self._passes,
                )
        prefill = recorded = 0.0
        if source.new_chars:
            sizes = source.ctx_chars if slot.cold else source.new_chars
            prefill = float(sizes[slot.cursor])
            recorded = source.out_chars[slot.cursor]
        slot.cold = False
        slot.cursor += 1
        slot.inflight = query_id
        self._lane_inflight[query_id] = _Inflight(
            slot=slot.index,
            counted=counted,
            conversation_id=conversation_id,
            turn=turn,
            issue_ns=now,
            prefill_chars=prefill,
            recorded_out=recorded,
        )
        assert self._loop is not None
        self._timeout_handles[query_id] = self._loop.call_later(
            self._turn_timeout_s, self._handle_lane_timeout, query_id
        )

    def _salted_override(
        self, source_id: str, turn: int, deck: int, uid: str
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
                # on the deck, so every conversation of a pass shares the cached
                # system-prompt prefix (500 workflow conversations share one
                # ~127k-character prompt). The tail is unique per instance, so
                # conversation histories never reuse another instance's cache.
                head = self._key.digest("salt-head", str(deck))
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
        output_chars = 0.0
        if result.error is None:
            try:
                output_chars = float(len(result.get_response_output_string()))
            except Exception:
                output_chars = 0.0
        try:
            self._complete(
                info, failed=result.error is not None, output_chars=output_chars
            )
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

    def _complete(
        self, info: _Inflight, failed: bool, output_chars: float = 0.0
    ) -> None:
        now = time.monotonic_ns()
        slot = self._slots[info.slot]
        slot.inflight = None
        if self._measuring:
            self._window_output.append((now, output_chars))
        else:
            latency_s = (now - info.issue_ns) / 1e9
            self._warm_completions.append((now, latency_s))
            if (
                self._cfg.entry == "calibrated"
                and self._fit is None
                and not failed
                and self._sources[0].new_chars
            ):
                self._fit_samples.append(
                    (latency_s, output_chars, info.prefill_chars, info.recorded_out)
                )
        if info.counted:
            self._counted_completed += 1
            if failed:
                self._counted_failed += 1
            if self._closed and self._counted_completed >= self._target:
                self._halt(now)
        if self._halted or self._stopping:
            self._signal_done_if_idle()
            return
        if slot.index in self._reseed_pending:
            self._reseed(slot)
            return
        self._advance(slot)

    def _halt(self, now_ns: int) -> None:
        """Every counted turn has completed: stop issuing and drain."""
        if self._halted:
            return
        self._halted = True
        self._halt_ns = now_ns
        for slot in self._slots:
            if slot.delay_handle is not None:
                slot.delay_handle.cancel()
                slot.delay_handle = None
        logger.info("Lane scheduler: all %d counted turns complete", self._target)
        self._signal_done_if_idle()

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

    def _coverage(self) -> dict[str, Any]:
        counts = [
            self._counted_per_turn.get((i, k), 0)
            for i, s in enumerate(self._sources)
            for k in range(len(s.turns))
        ]
        if not counts:
            return {}
        mean = statistics.fmean(counts)
        return {
            "dataset_turns": len(counts),
            "mean_times_counted": mean,
            "min_times_counted": min(counts),
            "max_times_counted": max(counts),
            "cv_times_counted": statistics.pstdev(counts) / mean if mean else None,
            "turns_never_counted": sum(1 for c in counts if c == 0),
        }

    def plan_metadata(self) -> dict[str, Any]:
        """Everything needed to audit the run, key included."""

        def seconds(a: int | None, b: int | None) -> float | None:
            return (b - a) / 1e9 if a is not None and b is not None else None

        return {
            "scheduler": "lanes",
            "design": "stationary start, fixed counted passes",
            "secret_hex": self._key.secret.hex(),
            "secret_sha256": self._key.commitment,
            "seed": self._key.seed,
            "concurrency": self._concurrency,
            "passes": self._passes,
            "pass_s_from_warmup": self._pass_s,
            "measured_turns_expected": self._target,
            "measured_turns_completed": self._counted_completed,
            "measured_turns_failed": self._counted_failed,
            "config": self._cfg.model_dump(),
            "clock": "time.monotonic_ns, same clock as events.jsonl",
            "run_started_ns": self._run_started_ns,
            "measure_ns": self._measure_ns,
            "window_end_ns": self._window_end_ns,
            "halt_ns": self._halt_ns,
            # Issue-time span of the counted turns (the throughput denominator)
            # and the time until the last of them completed.
            "issue_window_s": seconds(self._measure_ns, self._window_end_ns),
            "measured_phase_s": seconds(self._measure_ns, self._halt_ns),
            "drain_after_window_s": seconds(self._window_end_ns, self._halt_ns),
            "decks_drawn": self._deck_index,
            "conversation_instances": self._serial,
            "coverage": self._coverage(),
            "warmup_s_actual": seconds(self._run_started_ns, self._measure_ns),
            "warmup_converged": self._warmup_converged,
            "warmup_intervals": self._warmup_intervals,
            "calibration": self._fit,
            "ready_s": seconds(self._run_started_ns, self._ready_ns),
            "reseed_pending_at_end": len(self._reseed_pending),
            "pass_decisions": self._pass_decisions,
            "precision_at_close": self._precision_at_close,
        }

    def _write_plan(self) -> None:
        try:
            self._write_json("lane_plan.json", self.plan_metadata())
        except Exception:
            logger.exception("Lane scheduler: could not write lane_plan.json")
