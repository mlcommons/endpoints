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

"""Run timers and abort machinery for the benchmark orchestrator.

``PerfPhaseTimeout`` bounds the PERFORMANCE phase (``runtime.max_duration_ms``);
``RunWatchdog`` enforces ``settings.timeouts.run_timeout_s``;
``SigintGovernor`` (+ ``sigint_policy``) is the run's one Ctrl-C policy.
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


class SigintGovernor:
    """The run's single SIGINT policy — installed ONCE per run.

    One handler covers the whole run (setup, session, drain, finalize); the
    window-scoped install/remove pairs it replaces were exactly where a ^C
    could slip through as a raw KeyboardInterrupt mid-teardown. Behavior is
    keystroke-count-independent (group-SIGINT forwarders like ``uv run`` need
    no special handling): no live run -> raise KeyboardInterrupt (exit 130);
    live run -> graceful ``session.stop()`` plus a teardown grace timer that
    abandons a still-wedged metrics drain (SIGTERM -> SIGKILL); any repeat ^C
    is a logged no-op.
    """

    def __init__(self, interrupted_teardown_grace_s: float | None) -> None:
        self.interrupted = False
        self._interrupted_teardown_grace_s = interrupted_teardown_grace_s
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
        if (
            self._on_grace_expiry is not None
            and self._interrupted_teardown_grace_s is not None
            and self._grace_handle is None
        ):

            def _expire() -> None:
                logger.warning(
                    "Teardown did not finish within %.0fs of ^C — abandoning "
                    "the metrics drain",
                    self._interrupted_teardown_grace_s,
                )
                assert self._on_grace_expiry is not None
                self._on_grace_expiry()

            self._grace_handle = self._loop.call_later(
                self._interrupted_teardown_grace_s, _expire
            )

    def __call__(self, signum: int, frame: types.FrameType | None) -> None:
        if self.interrupted:
            # Stop already in flight; the grace timer bounds the teardown.
            logger.warning("SIGINT again: shutdown already in progress")
            return
        self.interrupted = True
        if (
            self._task is None
            or self._task.done()
            or self._loop is None
            or not self._loop.is_running()
        ):
            # No live run at all (sync setup/finalize, between audit phases);
            # call_soon_threadsafe on a stopped loop would silently swallow
            # the ^C.
            raise KeyboardInterrupt
        if self._session is None:
            # Run task live but no session yet (service launch / endpoint
            # connect): cancel the task — same as the watchdog's pre-session
            # fire — so the pipeline __aexit__ kills the service children
            # instead of a raw raise orphaning them. _run_benchmark_async
            # maps the unwind back to KeyboardInterrupt.
            self._loop.call_soon_threadsafe(self._task.cancel)
            return
        logger.warning("SIGINT received: stopping benchmark gracefully")
        # Signal handlers run at arbitrary bytecode boundaries: hand the stop
        # to the loop via its one signal-safe entry point.
        self._loop.call_soon_threadsafe(self._stop_gracefully)


@contextlib.contextmanager
def sigint_policy(governor: SigintGovernor) -> Iterator[None]:
    """Install ``governor`` as the SIGINT handler; restore the previous one.

    Passive when ``getsignal`` returns ``None`` (a C-installed handler that
    ``signal.signal`` refuses back) or off the main thread. Restores on exit,
    after the caller's finally blocks, so a repeat ^C during cleanup still
    hits the governor's no-op.
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


class RunWatchdog:
    """Whole-run deadline timer for ``settings.timeouts.run_timeout_s``.

    Armed before the pipeline starts and kept armed through the metrics
    drain. On fire with a session: stop it (ENDED still flows, the event
    logger flushes) and SIGTERM the aggregator, whose handler writes the
    INTERRUPTED final snapshot; if the aggregator ignores the SIGTERM, the
    teardown grace SIGTERM->SIGKILLs the children so the deadline stays a
    hard bound. Before the session exists the orchestration task is
    cancelled instead, and ``MetricsPipeline.__aexit__`` kills the services.
    ``run_benchmark`` raises whenever ``fired`` is set.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        deadline: float | None,
        pipe: MetricsPipeline,
        interrupted_teardown_grace_s: float | None,
    ) -> None:
        self.fired = False
        self._session: BenchmarkSession | None = None
        self._task: asyncio.Task | None = None
        self._pipe = pipe
        self._loop = loop
        self._interrupted_teardown_grace_s = interrupted_teardown_grace_s
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
        """Late-bind the session; a deadline that already fired stops it now.

        The caller still runs the stopped session so STARTED/ENDED flow and
        the INTERRUPTED artifacts get written.
        """
        self._session = session
        if self.fired:
            session.stop()

    def _fire(self) -> None:
        self.fired = True
        logger.error(
            "Run timeout reached; aborting run — report will be marked INTERRUPTED."
        )
        if self._session is None:
            # Still in service launch / endpoint connect: cancel the task so
            # those awaits unwind now; _run_benchmark_async translates the
            # unwind into the run-timeout ExecutionError.
            if self._task is not None:
                self._task.cancel()
            return
        self._session.stop()
        self._pipe.terminate_metrics_aggregator()
        # A wedged aggregator ignores the SIGTERM; always escalate so
        # run_timeout_s stays a hard bound — a null ^C-grace must not soften
        # it (cancelled when the drain finishes on its own).
        grace = self._interrupted_teardown_grace_s
        self._escalation = self._loop.call_later(
            grace if grace is not None else 30.0, self._pipe.abandon_drain
        )

    def cancel(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
        if self._escalation is not None:
            self._escalation.cancel()
            self._escalation = None
