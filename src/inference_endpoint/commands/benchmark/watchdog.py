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
import logging
import time
from collections.abc import Callable
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

    Semantics:
    - ^C with no live run task (sync setup): nothing to stop gracefully —
      raise KeyboardInterrupt immediately (default behavior, exit 130).
    - First ^C: graceful — ``session.stop()``; the stopped run publishes
      INTERRUPTED+ENDED, services drain (including the metrics-tokenization
      backlog), artifacts land as state=interrupted, then ``run_benchmark``
      raises for exit 130.
    - Any further ^C: FORCE QUIT — the run task is cancelled, so the
      teardown (metrics drain included) is abandoned; the pipeline
      ``__aexit__`` kills the service children (the SIGTERMed aggregator
      writes a best-effort INTERRUPTED snapshot without finishing its
      drain), tmpfs is salvaged, exit 130.
    """

    def __init__(self) -> None:
        self.interrupted = False
        self.forced = False
        self._session: BenchmarkSession | None = None
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_task(
        self, task: asyncio.Task | None, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Bind the run coroutine's task — the force-quit cancellation target."""
        self._task = task
        self._loop = loop

    def bind_session(self, session: BenchmarkSession) -> None:
        self._session = session

    def __call__(self, signum: int, frame: object) -> None:
        if self.interrupted:
            self.forced = True
            logger.warning(
                "SIGINT again: force quit — abandoning teardown/metrics drain"
            )
            if (
                self._task is not None
                and not self._task.done()
                and self._loop is not None
                and self._loop.is_running()
            ):
                # Cancelling the run task unwinds its finallys: the pipeline
                # __aexit__ kills the service children and tmpfs is salvaged.
                self._loop.call_soon_threadsafe(self._task.cancel)
                return
            raise KeyboardInterrupt
        self.interrupted = True
        if self._session is None or self._loop is None:
            raise KeyboardInterrupt
        logger.warning(
            "SIGINT received: stopping benchmark gracefully (^C again to force)"
        )
        # A signal handler runs at an arbitrary bytecode boundary — possibly
        # mid-event-loop-iteration. Don't mutate asyncio state (Event.set,
        # Task.cancel) from here; hand session.stop to the loop, the one
        # asyncio entry point documented as signal-handler safe.
        self._loop.call_soon_threadsafe(self._session.stop)


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
    is first-wins, so INTERRUPTED stays authoritative). Before the session
    exists (service launch / endpoint connect still pending), stopping
    nothing would let those awaits run out their own readiness timeouts past
    the deadline — so the orchestration task is cancelled instead, which
    unwinds them promptly and lets ``MetricsPipeline.__aexit__`` kill the
    services. ``run_benchmark`` raises ``ExecutionError`` after finalization
    whenever ``fired`` is set, so a timed-out run always fails loudly even if
    a still-draining aggregator finalized COMPLETE first.
    """

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

    def cancel(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
