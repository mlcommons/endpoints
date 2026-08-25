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

"""SigintGovernor (graceful stop + teardown grace) and _PerfPhaseTimeout."""

from __future__ import annotations

import asyncio
import contextlib
import signal
from unittest.mock import MagicMock

import pytest
from inference_endpoint.commands.benchmark.watchdog import (
    SigintGovernor,
    _PerfPhaseTimeout,
    sigint_policy,
)
from inference_endpoint.load_generator.session import PhaseType


def _fire(gov: SigintGovernor) -> None:
    gov(signal.SIGINT, None)


@pytest.mark.unit
class TestSigintGovernor:
    def test_unbound_sigint_raises_keyboard_interrupt(self):
        gov = SigintGovernor()
        with pytest.raises(KeyboardInterrupt):
            _fire(gov)
        assert gov.interrupted

    @pytest.mark.asyncio
    async def test_presession_sigint_cancels_run_task(self):
        """^C after the task is bound but before the session exists must
        cancel the task (unwinding kills the service children via the
        pipeline __aexit__) — never raise raw and orphan them."""
        gov = SigintGovernor()
        run_task = asyncio.create_task(asyncio.sleep(30))
        await asyncio.sleep(0)
        gov.bind_task(run_task, asyncio.get_running_loop())

        _fire(gov)  # no raise: session not bound yet

        assert gov.interrupted
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=2.0)
        assert run_task.cancelled()

    def test_sigint_after_loop_returned_raises_immediately(self):
        """A ^C during sync finalization must not be swallowed.

        After ``run_until_complete`` returns, the session/task/loop stay
        bound but the loop is stopped — ``call_soon_threadsafe`` would queue
        ``session.stop`` on it and never run it.
        """
        gov = SigintGovernor()
        session = MagicMock()

        async def run_phase() -> None:
            gov.bind_task(asyncio.current_task(), asyncio.get_running_loop())
            gov.bind_session(session, asyncio.Event())

        asyncio.run(run_phase())

        with pytest.raises(KeyboardInterrupt):
            _fire(gov)
        assert gov.interrupted
        session.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_second_sigint_force_exits(self, monkeypatch):
        force_exit = MagicMock()
        monkeypatch.setattr(
            "inference_endpoint.commands.benchmark.watchdog._force_exit_process_group",
            force_exit,
        )
        gov = SigintGovernor()
        session = MagicMock()
        abort_event = asyncio.Event()
        gov.bind_task(asyncio.current_task(), asyncio.get_running_loop())
        gov.bind_session(session, abort_event)

        _fire(gov)
        await asyncio.sleep(0)

        session.stop.assert_called_once()
        assert abort_event.is_set()
        force_exit.assert_not_called()

        _fire(gov)
        force_exit.assert_called_once()


@pytest.mark.unit
class TestSigintPolicy:
    def test_installs_and_restores_on_exception(self):
        gov = SigintGovernor()
        prev = signal.getsignal(signal.SIGINT)
        with pytest.raises(RuntimeError):
            with sigint_policy(gov):
                assert signal.getsignal(signal.SIGINT) is gov
                raise RuntimeError("boom")
        assert signal.getsignal(signal.SIGINT) is prev

    def test_unrepresentable_c_handler_stays_untouched(self, monkeypatch):
        """getsignal()->None (C-installed handler): install nothing at all."""
        gov = SigintGovernor()
        monkeypatch.setattr(signal, "getsignal", lambda signum: None)
        install_spy = MagicMock()
        monkeypatch.setattr(signal, "signal", install_spy)
        with sigint_policy(gov):
            pass
        install_spy.assert_not_called()


@pytest.mark.unit
class TestPerfPhaseTimeout:
    """The max_issue_duration_ms cap bounds only the performance phase and never
    truncates a subsequent accuracy phase (regression: a combined
    perf+accuracy run was guillotined mid-accuracy because the perf timer
    was never cancelled). Exercised against the real running loop.
    """

    @pytest.mark.asyncio
    async def test_cap_fires_after_max_issue_duration(self):
        fired = asyncio.Event()
        timeout = _PerfPhaseTimeout(asyncio.get_running_loop(), 20, fired.set)

        timeout.on_phase_start(PhaseType.PERFORMANCE)

        await asyncio.wait_for(fired.wait(), timeout=2.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "max_issue_duration_ms, phases",
        [
            pytest.param(None, [PhaseType.PERFORMANCE], id="no-max-duration"),
            pytest.param(
                20,
                [PhaseType.WARMUP, PhaseType.ACCURACY],
                id="non-performance-phases",
            ),
            pytest.param(
                20,
                [PhaseType.PERFORMANCE, PhaseType.ACCURACY],
                id="accuracy-disarms-performance",
            ),
        ],
    )
    async def test_never_armed(self, max_issue_duration_ms, phases):
        fired = asyncio.Event()
        timeout = _PerfPhaseTimeout(
            asyncio.get_running_loop(), max_issue_duration_ms, fired.set
        )

        for phase_type in phases:
            timeout.on_phase_start(phase_type)

        await asyncio.sleep(0.1)
        assert not fired.is_set()
