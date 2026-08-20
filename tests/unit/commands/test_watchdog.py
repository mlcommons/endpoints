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

"""SigintGovernor state machine: graceful vs force vs no-live-run paths."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import signal
import time
from unittest.mock import MagicMock

import pytest
from inference_endpoint.commands.benchmark.watchdog import (
    PerfPhaseTimeout,
    SigintGovernor,
)
from inference_endpoint.load_generator.session import PhaseType


def _fire(gov: SigintGovernor) -> None:
    gov(signal.SIGINT, None)


def _distinct_fire(gov: SigintGovernor) -> None:
    """A ^C outside the duplicate-delivery window (a deliberate press)."""
    gov._last_accepted_monotonic = float("-inf")
    _fire(gov)


@pytest.mark.unit
class TestSigintGovernor:
    def test_first_sigint_after_loop_returned_raises_immediately(self):
        """A ^C during sync finalization must not be swallowed.

        After ``run_until_complete`` returns, the session/task/loop stay
        bound but the loop is stopped — ``call_soon_threadsafe`` would queue
        ``session.stop`` on it and never run it.
        """
        gov = SigintGovernor()
        session = MagicMock()

        async def run_phase() -> None:
            gov.bind_task(asyncio.current_task(), asyncio.get_running_loop())
            gov.bind_session(session)

        asyncio.run(run_phase())

        with pytest.raises(KeyboardInterrupt):
            _fire(gov)
        assert gov.interrupted
        session.stop.assert_not_called()

    def test_second_distinct_sigint_after_loop_returned_raises(self):
        """The force path with a finished task escalates to KeyboardInterrupt."""
        gov = SigintGovernor()
        session = MagicMock()

        async def run_phase() -> None:
            gov.bind_task(asyncio.current_task(), asyncio.get_running_loop())
            gov.bind_session(session)

        asyncio.run(run_phase())

        with pytest.raises(KeyboardInterrupt):
            _fire(gov)
        with pytest.raises(KeyboardInterrupt):
            _distinct_fire(gov)
        assert gov.forced

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bound", [True, False], ids=["bound", "unbound"])
    @pytest.mark.parametrize(
        "deliveries",
        [
            seq
            for n in (1, 2, 3)
            for seq in itertools.product(("distinct", "dup"), repeat=n)
            # A duplicate before any accepted delivery cannot occur: the dedup
            # window opens on the first accepted ^C.
            if seq[0] == "distinct"
        ],
        ids="-".join,
    )
    async def test_delivery_sequences_exhaustive(self, bound, deliveries):
        """Every bind-state x delivery-sequence (length <= 3), exhaustively.

        Contract: an accepted, distinct delivery is never silently dropped —
        it schedules a graceful stop (first, bound), cancels the live run
        task (second, bound), or raises KeyboardInterrupt (unbound). Only
        duplicate deliveries inside the window are silent. Escalation to
        force happens on exactly the second accepted delivery.
        """
        gov = SigintGovernor()
        session = MagicMock()
        run_task = asyncio.create_task(asyncio.sleep(30))
        await asyncio.sleep(0)  # let the child task start
        if bound:
            gov.bind_task(run_task, asyncio.get_running_loop())
            gov.bind_session(session)

        accepted = 0
        try:
            for kind in deliveries:
                if kind == "dup":
                    # Inside the duplicate window of the previous delivery.
                    gov._last_accepted_monotonic = time.monotonic()
                else:
                    gov._last_accepted_monotonic = float("-inf")
                    accepted += 1
                if kind == "distinct" and not bound:
                    with pytest.raises(KeyboardInterrupt):
                        gov(signal.SIGINT, None)
                else:
                    gov(signal.SIGINT, None)  # silent: bound or deduped

            assert gov.interrupted
            assert gov.forced == (accepted >= 2)
            if bound:
                await asyncio.sleep(0)  # run queued call_soon_threadsafe work
                assert session.stop.call_count == 1
                if accepted >= 2:
                    # Force path cancelled the run task; let it settle.
                    with contextlib.suppress(asyncio.CancelledError):
                        await asyncio.wait_for(run_task, timeout=2.0)
                    assert run_task.cancelled()
                else:
                    assert not run_task.done()
            else:
                session.stop.assert_not_called()
                assert not run_task.done()
        finally:
            run_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await run_task


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
