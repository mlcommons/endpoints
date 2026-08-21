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

"""SigintGovernor (graceful stop + teardown grace) and PerfPhaseTimeout."""

from __future__ import annotations

import asyncio
import signal
from unittest.mock import MagicMock

import pytest
from inference_endpoint.commands.benchmark.watchdog import (
    PerfPhaseTimeout,
    SigintGovernor,
    sigint_policy,
)
from inference_endpoint.load_generator.session import PhaseType


def _fire(gov: SigintGovernor) -> None:
    gov(signal.SIGINT, None)


@pytest.mark.unit
class TestSigintGovernor:
    def test_unbound_sigint_raises_keyboard_interrupt(self):
        gov = SigintGovernor(interrupted_teardown_grace_s=30.0)
        with pytest.raises(KeyboardInterrupt):
            _fire(gov)
        assert gov.interrupted

    def test_sigint_after_loop_returned_raises_immediately(self):
        """A ^C during sync finalization must not be swallowed.

        After ``run_until_complete`` returns, the session/task/loop stay
        bound but the loop is stopped — ``call_soon_threadsafe`` would queue
        ``session.stop`` on it and never run it.
        """
        gov = SigintGovernor(interrupted_teardown_grace_s=30.0)
        session = MagicMock()

        async def run_phase() -> None:
            gov.bind_task(asyncio.current_task(), asyncio.get_running_loop())
            gov.bind_session(session, MagicMock())

        asyncio.run(run_phase())

        with pytest.raises(KeyboardInterrupt):
            _fire(gov)
        assert gov.interrupted
        session.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_live_sigint_stops_session_and_arms_grace(self):
        gov = SigintGovernor(interrupted_teardown_grace_s=30.0)
        session = MagicMock()
        on_grace = MagicMock()
        gov.bind_task(asyncio.current_task(), asyncio.get_running_loop())
        gov.bind_session(session, on_grace)

        _fire(gov)
        await asyncio.sleep(0)  # run the queued call_soon_threadsafe

        assert gov.interrupted
        session.stop.assert_called_once()
        assert gov._grace_handle is not None
        on_grace.assert_not_called()  # armed, not fired

    @pytest.mark.asyncio
    async def test_repeat_sigint_is_a_noop(self):
        """Any repeat ^C (incl. a forwarded duplicate under `uv run`) is silent."""
        gov = SigintGovernor(interrupted_teardown_grace_s=30.0)
        session = MagicMock()
        gov.bind_task(asyncio.current_task(), asyncio.get_running_loop())
        gov.bind_session(session, MagicMock())

        _fire(gov)
        _fire(gov)
        _fire(gov)
        await asyncio.sleep(0)

        session.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_grace_expiry_fires_callback_once(self):
        """The grace fires exactly once, even after repeat ^C deliveries."""
        gov = SigintGovernor(interrupted_teardown_grace_s=0.02)
        session = MagicMock()
        fired = asyncio.Event()
        on_grace = MagicMock(side_effect=fired.set)
        gov.bind_task(asyncio.current_task(), asyncio.get_running_loop())
        gov.bind_session(session, on_grace)

        _fire(gov)
        _fire(gov)  # repeat ^C must not arm a second timer
        await asyncio.wait_for(fired.wait(), timeout=2.0)
        await asyncio.sleep(0.05)  # room for an (incorrect) second fire

        assert on_grace.call_count == 1

    @pytest.mark.asyncio
    async def test_cancel_grace_disarms_pending_timer(self):
        gov = SigintGovernor(interrupted_teardown_grace_s=0.02)
        session = MagicMock()
        on_grace = MagicMock()
        gov.bind_task(asyncio.current_task(), asyncio.get_running_loop())
        gov.bind_session(session, on_grace)

        _fire(gov)
        await asyncio.sleep(0)
        gov.cancel_grace()  # the drain finished on its own
        await asyncio.sleep(0.1)  # 5x the grace: a leaked timer would fire

        on_grace.assert_not_called()


@pytest.mark.unit
class TestSigintPolicy:
    def test_installs_and_restores_previous_handler(self):
        gov = SigintGovernor(interrupted_teardown_grace_s=30.0)
        prev = signal.getsignal(signal.SIGINT)
        with sigint_policy(gov):
            assert signal.getsignal(signal.SIGINT) is gov
        assert signal.getsignal(signal.SIGINT) is prev

    def test_restores_on_exception(self):
        gov = SigintGovernor(interrupted_teardown_grace_s=30.0)
        prev = signal.getsignal(signal.SIGINT)
        with pytest.raises(RuntimeError):
            with sigint_policy(gov):
                raise RuntimeError("boom")
        assert signal.getsignal(signal.SIGINT) is prev

    def test_unrepresentable_c_handler_stays_untouched(self, monkeypatch):
        """getsignal()->None (C-installed handler): install nothing at all."""
        gov = SigintGovernor(interrupted_teardown_grace_s=30.0)
        monkeypatch.setattr(signal, "getsignal", lambda signum: None)
        install_spy = MagicMock()
        monkeypatch.setattr(signal, "signal", install_spy)
        with sigint_policy(gov):
            pass
        install_spy.assert_not_called()


@pytest.mark.unit
class TestPerfPhaseTimeout:
    """The max_duration_ms cap bounds only the performance phase and never
    truncates a subsequent accuracy phase (regression: a combined
    perf+accuracy run was guillotined mid-accuracy because the perf timer
    was never cancelled). Exercised against the real running loop.
    """

    @pytest.mark.asyncio
    async def test_cap_fires_after_max_duration(self):
        fired = asyncio.Event()
        timeout = PerfPhaseTimeout(asyncio.get_running_loop(), 20, fired.set)

        timeout.on_phase_start(PhaseType.PERFORMANCE)

        await asyncio.wait_for(fired.wait(), timeout=2.0)

    @pytest.mark.asyncio
    async def test_accuracy_phase_start_disarms_pending_perf_cap(self):
        fired = asyncio.Event()
        timeout = PerfPhaseTimeout(asyncio.get_running_loop(), 20, fired.set)

        timeout.on_phase_start(PhaseType.PERFORMANCE)
        timeout.on_phase_start(PhaseType.ACCURACY)

        await asyncio.sleep(0.1)  # 5x the cap: a leaked timer would have fired
        assert not fired.is_set()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "max_duration_ms, phases",
        [
            pytest.param(None, [PhaseType.PERFORMANCE], id="no-max-duration"),
            pytest.param(
                20,
                [PhaseType.WARMUP, PhaseType.ACCURACY],
                id="non-performance-phases",
            ),
        ],
    )
    async def test_never_armed(self, max_duration_ms, phases):
        fired = asyncio.Event()
        timeout = PerfPhaseTimeout(
            asyncio.get_running_loop(), max_duration_ms, fired.set
        )

        for phase_type in phases:
            timeout.on_phase_start(phase_type)

        await asyncio.sleep(0.1)
        assert not fired.is_set()

    @pytest.mark.asyncio
    async def test_cancel_is_idempotent_and_disarms(self):
        fired = asyncio.Event()
        timeout = PerfPhaseTimeout(asyncio.get_running_loop(), 20, fired.set)

        timeout.cancel()  # no handle yet — must not raise
        timeout.on_phase_start(PhaseType.PERFORMANCE)
        timeout.cancel()
        timeout.cancel()

        await asyncio.sleep(0.1)
        assert not fired.is_set()
