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
import signal
from unittest.mock import MagicMock

import pytest
from inference_endpoint.commands.benchmark.watchdog import SigintGovernor


def _fire(gov: SigintGovernor) -> None:
    gov(signal.SIGINT, None)


def _distinct_fire(gov: SigintGovernor) -> None:
    """A ^C outside the duplicate-delivery window (a deliberate press)."""
    gov._last_accepted_monotonic = float("-inf")
    _fire(gov)


@pytest.mark.unit
class TestSigintGovernor:
    def test_unbound_first_sigint_raises_keyboard_interrupt(self):
        gov = SigintGovernor()
        with pytest.raises(KeyboardInterrupt):
            _fire(gov)
        assert gov.interrupted
        assert not gov.forced

    @pytest.mark.asyncio
    async def test_live_run_first_sigint_stops_session_gracefully(self):
        gov = SigintGovernor()
        session = MagicMock()
        gov.bind_task(asyncio.current_task(), asyncio.get_running_loop())
        gov.bind_session(session)

        _fire(gov)
        await asyncio.sleep(0)  # let the queued call_soon_threadsafe run

        assert gov.interrupted
        assert not gov.forced
        session.stop.assert_called_once()

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
    async def test_duplicate_delivery_within_window_is_dropped(self):
        """One keystroke forwarded by a runner (uv run) must count once."""
        gov = SigintGovernor()
        session = MagicMock()
        gov.bind_task(asyncio.current_task(), asyncio.get_running_loop())
        gov.bind_session(session)

        _fire(gov)
        _fire(gov)  # forwarded duplicate, inside the window
        await asyncio.sleep(0)

        assert gov.interrupted
        assert not gov.forced
        session.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_second_distinct_sigint_cancels_live_run_task(self):
        gov = SigintGovernor()
        session = MagicMock()
        gov.bind_task(asyncio.current_task(), asyncio.get_running_loop())
        gov.bind_session(session)

        _fire(gov)
        _distinct_fire(gov)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.sleep(5)

        assert gov.forced
