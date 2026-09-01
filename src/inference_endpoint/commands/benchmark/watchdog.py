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

"""Benchmark timeout and signal controllers."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import time
import types
from collections.abc import Callable, Iterator

from inference_endpoint.load_generator.session import BenchmarkSession, PhaseType

logger = logging.getLogger(__name__)


def _force_exit_process_group() -> None:
    os.killpg(os.getpgrp(), signal.SIGKILL)


class _PerfPhaseTimeout:
    """Bound PERFORMANCE issuing without aborting later phases.

    The timer is cancelled when any later phase starts, so the issue-duration
    cap cannot truncate an accuracy phase.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        max_issue_duration_ms: int | None,
        on_timeout: Callable[[], None],
    ) -> None:
        self._loop = loop
        self._max_duration_ms = max_issue_duration_ms
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
    def __init__(self) -> None:
        self.interrupted = False
        self._session: BenchmarkSession | None = None
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._abort_event: asyncio.Event | None = None

    def bind_task(
        self, task: asyncio.Task | None, loop: asyncio.AbstractEventLoop
    ) -> None:
        self._task = task
        self._loop = loop
        self._session = None
        self._abort_event = None

    def bind_session(
        self, session: BenchmarkSession, abort_event: asyncio.Event
    ) -> None:
        self._session = session
        self._abort_event = abort_event

    def __call__(self, signum: int, frame: types.FrameType | None) -> None:
        if self.interrupted:
            _force_exit_process_group()
            return
        self.interrupted = True
        if (
            self._task is None
            or self._task.done()
            or self._loop is None
            or not self._loop.is_running()
        ):
            raise KeyboardInterrupt
        if self._session is None:
            self._loop.call_soon_threadsafe(self._task.cancel)
            return
        assert self._abort_event is not None
        self._loop.call_soon_threadsafe(self._session.stop)
        self._loop.call_soon_threadsafe(self._abort_event.set)


@contextlib.contextmanager
def sigint_policy(governor: SigintGovernor) -> Iterator[None]:
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
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        deadline: float | None,
        abort_event: asyncio.Event,
    ) -> None:
        self.fired = False
        self._session: BenchmarkSession | None = None
        self._task: asyncio.Task | None = None
        self._abort_event = abort_event
        self._handle = (
            loop.call_later(max(0.0, deadline - time.monotonic()), self._fire)
            if deadline is not None
            else None
        )

    def bind_task(self, task: asyncio.Task | None) -> None:
        self._task = task

    def bind_session(self, session: BenchmarkSession) -> None:
        self._session = session
        if self.fired:
            session.stop()
            self._abort_event.set()

    def _fire(self) -> None:
        self.fired = True
        logger.error("Run timeout reached; aborting run")
        if self._session is not None:
            self._session.stop()
            self._abort_event.set()
        elif self._task is not None:
            self._task.cancel()

    def cancel(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None
