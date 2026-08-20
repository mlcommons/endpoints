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

"""Run-scoped abort machinery for the benchmark orchestrator.

``PerfPhaseTimeout`` bounds the PERFORMANCE phase (``runtime.max_duration_ms``);
``RunWatchdog`` is the whole-run deadline (``settings.timeouts.run_timeout_s``);
``SigintGovernor`` is the run's one Ctrl-C policy. All owned by
``commands/benchmark/execute.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import time
import types
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from inference_endpoint.load_generator.session import BenchmarkSession, PhaseType

if TYPE_CHECKING:
    from inference_endpoint.commands.benchmark.pipeline import MetricsPipeline

logger = logging.getLogger(__name__)


class SigintGovernor:
    """The run's single SIGINT policy — installed ONCE by ``run_benchmark``.

    One ``signal.signal`` handler covers the entire run (setup, session,
    metrics drain, finalize) instead of window-scoped install/remove pairs,
    whose gaps are exactly where a ^C used to slip through as a raw
    KeyboardInterrupt and abort teardown half-way.

    One behavior, keystroke-count-independent (so runners that forward the
    terminal's group SIGINT, like ``uv run``, need no special handling):
    - ^C with no live run (sync setup, finalization after the loop returned,
      between audit phases): nothing to stop gracefully — raise
      KeyboardInterrupt immediately (default behavior, exit 130).
    - ^C with a live run: graceful — ``session.stop()``; the stopped run
      publishes INTERRUPTED+ENDED, services drain, artifacts land as
      state=interrupted, then ``run_benchmark`` raises for exit 130. A
      teardown grace timer is armed: if the metrics drain has not finished
      within ``TEARDOWN_GRACE_S``, the aggregator is SIGTERMed (its handler
      writes a best-effort INTERRUPTED snapshot; ``terminate_all`` escalates
      to SIGKILL) so a wedged drain can never hang the abort.
    - Any repeat ^C: logged no-op — the stop is already in flight and the
      grace timer bounds the teardown.
    """

    TEARDOWN_GRACE_S = 30.0
    """Seconds after a ^C before a still-running metrics drain is abandoned."""

    def __init__(self) -> None:
        self.interrupted = False
        self._session: BenchmarkSession | None = None
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_grace_expiry: Callable[[], None] | None = None
        self._grace_handle: asyncio.TimerHandle | None = None

    def bind_task(
        self, task: asyncio.Task | None, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Bind the run coroutine's task — the live-run gate for the graceful path."""
        self._task = task
        self._loop = loop

    def bind_session(
        self, session: BenchmarkSession, on_grace_expiry: Callable[[], None]
    ) -> None:
        self._session = session
        self._on_grace_expiry = on_grace_expiry

    def cancel_grace(self) -> None:
        """Disarm the teardown grace timer (drain finished on its own)."""
        if self._grace_handle is not None:
            self._grace_handle.cancel()
            self._grace_handle = None

    def _stop_gracefully(self) -> None:
        """Runs on the loop: stop the session and bound the teardown."""
        assert self._session is not None and self._loop is not None
        self._session.stop()
        if self._on_grace_expiry is not None and self._grace_handle is None:

            def _expire() -> None:
                logger.warning(
                    "Teardown did not finish within %.0fs of ^C — abandoning "
                    "the metrics drain",
                    self.TEARDOWN_GRACE_S,
                )
                assert self._on_grace_expiry is not None
                self._on_grace_expiry()

            self._grace_handle = self._loop.call_later(self.TEARDOWN_GRACE_S, _expire)

    def __call__(self, signum: int, frame: types.FrameType | None) -> None:
        if self.interrupted:
            # Stop already in flight; the grace timer bounds the teardown. A
            # forwarded duplicate delivery (uv run) lands here harmlessly too.
            logger.warning("SIGINT again: shutdown already in progress")
            return
        self.interrupted = True
        if (
            self._session is None
            or self._task is None
            or self._task.done()
            or self._loop is None
            or not self._loop.is_running()
        ):
            # No live run to stop gracefully. call_soon_threadsafe on a
            # stopped loop would queue session.stop and never run it —
            # silently swallowing the ^C.
            raise KeyboardInterrupt
        logger.warning("SIGINT received: stopping benchmark gracefully")
        # A signal handler runs at an arbitrary bytecode boundary — possibly
        # mid-event-loop-iteration. Don't mutate asyncio state (Event.set,
        # call_later) from here; hand the stop to the loop, the one asyncio
        # entry point documented as signal-handler safe.
        self._loop.call_soon_threadsafe(self._stop_gracefully)


@contextlib.contextmanager
def sigint_policy(governor: SigintGovernor) -> Iterator[None]:
    """Install ``governor`` as the SIGINT handler; restore the previous one.

    Stays passive (installs nothing) when ``signal.getsignal`` returns
    ``None`` — a C-installed handler Python cannot represent and
    ``signal.signal`` would refuse to accept back — or when installation
    raises ValueError (not the main thread; embedded use). Restoration
    happens on exit, after the caller's own finally blocks, so a repeat ^C
    during cleanup still hits the governor's no-op instead of the default
    handler.
    """
    prev = signal.getsignal(signal.SIGINT)
    if prev is None:
        yield
        return
    try:
        signal.signal(signal.SIGINT, governor)
    except ValueError:
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, prev)


class PerfPhaseTimeout:
    """Session-stop timer that bounds the PERFORMANCE phase only.

    ``max_duration_ms`` is a safety cap on the performance phase. The timer is
    armed when the performance phase starts and cancelled as soon as any later
    phase starts, so it can never truncate a subsequent accuracy phase: a
    combined perf+accuracy run must let accuracy finish regardless of how long
    perf ran.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        max_duration_ms: int | None,
        on_timeout: Callable[[], None],
    ) -> None:
        self._loop = loop
        self._max_duration_ms = max_duration_ms
        self._on_timeout = on_timeout
        self._handle: asyncio.TimerHandle | None = None

    def on_phase_start(self, phase_type: PhaseType) -> None:
        self.cancel()
        if phase_type == PhaseType.PERFORMANCE and self._max_duration_ms is not None:
            self._handle = self._loop.call_later(
                self._max_duration_ms / 1000.0, self._on_timeout
            )

    def cancel(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None


class RunWatchdog:
    """Whole-run deadline timer for ``settings.timeouts.run_timeout_s``.

    Armed before the metrics pipeline starts (so service-launch and
    endpoint-connect stalls are bounded) and kept armed through the metrics
    drain (so a stuck aggregator drain is bounded too). On fire, once the
    session exists: stop the session (its run unwinds and publishes ENDED, so
    the event logger — spared the SIGTERM — flushes and exits) and SIGTERM
    the aggregator, whose handler immediately writes the INTERRUPTED final
    snapshot with whatever stats it holds at that instant (``publish_final``
    is first-wins, so INTERRUPTED stays authoritative). If the aggregator
    ignores the SIGTERM (wedged/unschedulable), the same teardown grace as
    the ^C path SIGTERM→SIGKILLs every service child so the drain's
    wait-for-exit unblocks — the deadline is a hard bound. Before the session
    exists (service launch / endpoint connect still pending), stopping
    nothing would let those awaits run out their own readiness timeouts past
    the deadline — so the orchestration task is cancelled instead, which
    unwinds them promptly and lets ``MetricsPipeline.__aexit__`` kill the
    services. ``run_benchmark`` raises ``ExecutionError`` after finalization
    whenever ``fired`` is set, so a timed-out run always fails loudly even if
    a still-draining aggregator finalized COMPLETE first.
    """

    TEARDOWN_GRACE_S = SigintGovernor.TEARDOWN_GRACE_S

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        deadline: float | None,
        pipe: MetricsPipeline,
    ) -> None:
        self.fired = False
        self._session: BenchmarkSession | None = None
        self._task: asyncio.Task | None = None
        self._pipe = pipe
        self._loop = loop
        self._escalation: asyncio.TimerHandle | None = None
        self._handle = (
            loop.call_later(max(0.0, deadline - time.monotonic()), self._fire)
            if deadline is not None
            else None
        )

    def bind_task(self, task: asyncio.Task | None) -> None:
        """Bind the orchestration task — the pre-session cancellation target."""
        self._task = task

    def bind_session(self, session: BenchmarkSession) -> None:
        """Late-bind the session: it is created after the timer is armed.

        A deadline that already fired stops the session immediately, so no
        load is ever issued past it — the caller still runs the stopped
        session so STARTED/ENDED flow (the event logger exits only on ENDED)
        and the INTERRUPTED artifacts get written.
        """
        self._session = session
        if self.fired:
            session.stop()

    def _fire(self) -> None:
        self.fired = True
        logger.error(
            "Run timeout reached; aborting run — report will be marked " "INTERRUPTED."
        )
        if self._session is None:
            # Still in service launch / endpoint connect: cancel the
            # orchestration task so those awaits unwind now instead of
            # running out their own readiness timeouts past the deadline.
            # No load was issued, so there are no artifacts to preserve;
            # _run_benchmark_async translates the unwind into the run-timeout
            # ExecutionError.
            if self._task is not None:
                self._task.cancel()
            return
        self._session.stop()
        self._pipe.terminate_metrics_aggregator()
        # A wedged aggregator ignores the SIGTERM; escalate so the deadline
        # stays a hard bound (cancelled by cancel() when the drain finishes).
        self._escalation = self._loop.call_later(
            self.TEARDOWN_GRACE_S, self._pipe.abandon_drain
        )

    def cancel(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        if self._escalation is not None:
            self._escalation.cancel()
            self._escalation = None
