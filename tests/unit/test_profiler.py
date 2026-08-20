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

"""Unit tests for the line_profiler module."""

import asyncio
import io
import os
from unittest import mock

import inference_endpoint.profiling.line_profiler as line_profiler
import pytest
from inference_endpoint.profiling import (
    get_stats,
    is_enabled,
    pause,
    print_stats,
    profile,
    resume,
)
from inference_endpoint.profiling.line_profiler import (
    ENV_VAR_ENABLE_LINE_PROFILER,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def restore_profiler_singleton():
    """Restore the module-level singleton after any test that replaces it.

    The module's public API (``profile``, ``print_stats``, ...) is bound to
    the singleton created at import time; tests that reset ``_instance`` and
    re-init under a patched env must not leak that replacement (or a live C
    profiler) into other tests.
    """
    original = line_profiler.ProfilerState._instance
    yield
    current = line_profiler.ProfilerState._instance
    if current is not None and current is not original:
        current.shutdown()
    line_profiler.ProfilerState._instance = original


@pytest.fixture
def enabled_profiler():
    """A fresh, enabled ProfilerState under ENABLE_LINE_PROFILER=1."""
    with mock.patch.dict(os.environ, {ENV_VAR_ENABLE_LINE_PROFILER: "1"}):
        line_profiler.ProfilerState._instance = None
        yield line_profiler.ProfilerState()


class TestProfilerState:
    """Test the ProfilerState singleton."""

    def test_singleton_pattern(self):
        state1 = line_profiler.ProfilerState()
        state2 = line_profiler.ProfilerState()
        assert state1 is state2

    def test_profiler_disabled_by_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            line_profiler.ProfilerState._instance = None
            state = line_profiler.ProfilerState()
            assert not state.enabled
            assert state.profiler is None

    def test_profiler_enabled_with_env_var(self, enabled_profiler):
        # Only check the enabled flag: line_profiler might not be installed.
        assert enabled_profiler.enabled


class TestProfileDecorators:
    """Test the profile decorator on both sync and async functions."""

    @pytest.mark.skipif(is_enabled(), reason="Test only runs when profiler disabled")
    def test_profile_decorator_sync_when_disabled(self):
        @profile
        def test_func(x):
            return x * 2

        # When disabled, decorator is a no-op and the function is unchanged.
        assert test_func(5) == 10
        assert test_func.__name__ == "test_func"

    @pytest.mark.skipif(is_enabled(), reason="Test only runs when profiler disabled")
    def test_profile_decorator_async_when_disabled(self):
        @profile
        async def test_async_func(x):
            await asyncio.sleep(0)
            return x * 2

        assert asyncio.run(test_async_func(5)) == 10
        assert test_async_func.__name__ == "test_async_func"

    def test_profile_decorator_sync_when_enabled(self, enabled_profiler):
        @enabled_profiler.profile
        def test_func(x):
            return x * 2

        assert test_func(5) == 10

    def test_profile_decorator_async_when_enabled(self, enabled_profiler):
        @enabled_profiler.profile
        async def test_async_func(x):
            await asyncio.sleep(0)
            return x * 2

        assert asyncio.run(test_async_func(5)) == 10


class TestProfilerMethods:
    """Test profiler utility methods."""

    @pytest.mark.skipif(is_enabled(), reason="Test only runs when profiler disabled")
    def test_print_stats_when_disabled(self):
        output = io.StringIO()
        print_stats(stream=output)
        assert output.getvalue() == ""

    @pytest.mark.skipif(is_enabled(), reason="Test only runs when profiler disabled")
    def test_get_stats_when_disabled(self):
        assert get_stats() == ""

    def test_pause_resume_methods(self):
        # Must not raise, enabled or not.
        resume()
        pause()

    def test_print_stats_no_output_when_no_functions(self, enabled_profiler):
        if enabled_profiler.profiler is None:
            pytest.skip("line_profiler not installed")
        assert len(enabled_profiler.profiler.functions) == 0

        output = io.StringIO()
        enabled_profiler.print_stats(stream=output, prefix="Test")

        assert output.getvalue() == ""


class TestProfilerCleanup:
    """Shutdown must always tear the C profiler down, exactly once."""

    def test_shutdown_handles_multiple_calls(self, enabled_profiler):
        enabled_profiler.shutdown()
        enabled_profiler.shutdown()
        enabled_profiler.shutdown()
        assert enabled_profiler.profiler is None

    def test_shutdown_after_print_stats_still_tears_down(self, enabled_profiler):
        """print_stats() marks stats printed; shutdown() must still teardown."""

        @enabled_profiler.profile
        def traced(x):
            return x + 1

        traced(1)
        enabled_profiler.print_stats(stream=io.StringIO())
        assert enabled_profiler._stats_printed is True

        enabled_profiler.shutdown()
        assert enabled_profiler.profiler is None

    def test_shutdown_tears_down_when_output_destination_fails(self, enabled_profiler):
        """A failing stats dump must still leave the C profiler disabled."""

        @enabled_profiler.profile
        def traced(x):
            return x + 1

        traced(1)
        with mock.patch.object(
            enabled_profiler,
            "_print_stats_to_destination",
            side_effect=OSError("disk full"),
        ):
            enabled_profiler.shutdown()
        assert enabled_profiler.profiler is None
