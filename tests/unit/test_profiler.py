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


@pytest.fixture(autouse=True)
def cleanup_profiler():
    """Ensure profiler is cleaned up after each test."""
    yield

    # Clean up after test
    if (
        line_profiler.ProfilerState._instance
        and line_profiler.ProfilerState._instance.profiler
    ):
        try:
            line_profiler.ProfilerState._instance.pause()
            # Clear any accumulated stats
            line_profiler.ProfilerState._instance._stats_printed = False
        except Exception:
            pass


class TestProfilerState:
    """Test the ProfilerState singleton."""

    def test_singleton_pattern(self):
        """Test that ProfilerState follows singleton pattern."""
        state1 = line_profiler.ProfilerState()
        state2 = line_profiler.ProfilerState()
        assert state1 is state2

    def test_profiler_disabled_by_default(self):
        """Test profiler is disabled when ENABLE_LINE_PROFILER is not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            # Force re-initialization
            line_profiler.ProfilerState._instance = None
            state = line_profiler.ProfilerState()
            assert not state.enabled
            assert state.profiler is None

    def test_profiler_enabled_with_env_var(self):
        """Test profiler is enabled when ENABLE_LINE_PROFILER=1."""
        with mock.patch.dict(os.environ, {ENV_VAR_ENABLE_LINE_PROFILER: "1"}):
            # Force re-initialization
            line_profiler.ProfilerState._instance = None
            try:
                state = line_profiler.ProfilerState()
                # Only check enabled flag, as line_profiler might not be installed
                assert state.enabled
            finally:
                # Reset for other tests
                line_profiler.ProfilerState._instance = None


class TestProfileDecorators:
    """Test the profile decorator on both sync and async functions."""

    @pytest.mark.skipif(is_enabled(), reason="Test only runs when profiler disabled")
    def test_profile_decorator_sync_when_disabled(self):
        """Test profile decorator returns original sync function when disabled."""

        @profile
        def test_func(x):
            return x * 2

        # When disabled, decorator should be no-op
        assert test_func(5) == 10
        # Function should be unchanged
        assert test_func.__name__ == "test_func"

    @pytest.mark.skipif(is_enabled(), reason="Test only runs when profiler disabled")
    def test_profile_decorator_async_when_disabled(self):
        """Test profile decorator returns original async function when disabled."""

        @profile
        async def test_async_func(x):
            await asyncio.sleep(0)
            return x * 2

        # When disabled, decorator should be no-op
        result = asyncio.run(test_async_func(5))
        assert result == 10
        # Function should be unchanged
        assert test_async_func.__name__ == "test_async_func"

    def test_profile_decorator_sync_when_enabled(self):
        """Test profile decorator wraps sync function when enabled."""
        with mock.patch.dict(os.environ, {ENV_VAR_ENABLE_LINE_PROFILER: "1"}):
            # Force re-initialization
            line_profiler.ProfilerState._instance = None

            # Import after setting env var
            from inference_endpoint.profiling.line_profiler import ProfilerState

            state = ProfilerState()

            @state.profile
            def test_func(x):
                return x * 2

            result = test_func(5)
            assert result == 10

    def test_profile_decorator_async_when_enabled(self):
        """Test profile decorator wraps async function when enabled."""
        with mock.patch.dict(os.environ, {ENV_VAR_ENABLE_LINE_PROFILER: "1"}):
            # Force re-initialization
            line_profiler.ProfilerState._instance = None

            # Import after setting env var
            from inference_endpoint.profiling.line_profiler import ProfilerState

            state = ProfilerState()

            @state.profile
            async def test_async_func(x):
                await asyncio.sleep(0)
                return x * 2

            result = asyncio.run(test_async_func(5))
            assert result == 10


class TestProfilerMethods:
    """Test profiler utility methods."""

    @pytest.mark.skipif(is_enabled(), reason="Test only runs when profiler disabled")
    def test_print_stats_when_disabled(self):
        """Test print_stats does nothing when profiler is disabled."""
        output = io.StringIO()
        print_stats(stream=output)
        assert output.getvalue() == ""

    @pytest.mark.skipif(is_enabled(), reason="Test only runs when profiler disabled")
    def test_get_stats_when_disabled(self):
        """Test get_stats returns empty string when profiler is disabled."""
        stats = get_stats()
        assert stats == ""

    def test_pause_resume_methods(self):
        """Test pause/resume methods don't crash when called."""
        # Should not raise any exceptions
        resume()
        pause()

    def test_print_stats_with_prefix(self):
        """Test print_stats with a prefix."""
        output = io.StringIO()
        print_stats(stream=output, prefix="Test Worker")

        # When disabled, should produce no output
        if not is_enabled():
            assert output.getvalue() == ""
        else:
            # When enabled, should have the prefix in output
            output_str = output.getvalue()
            if output_str:  # Only check if there's output
                assert (
                    "Test Worker - LINE PROFILER RESULTS" in output_str
                    or "Test Worker" in output_str
                )

    def test_print_stats_no_output_when_no_functions(self):
        """Test print_stats produces no output when no functions have been profiled."""
        with mock.patch.dict(os.environ, {ENV_VAR_ENABLE_LINE_PROFILER: "1"}):
            line_profiler.ProfilerState._instance = None
            state = line_profiler.ProfilerState()

            if state.profiler:
                # Ensure no functions are profiled (fresh profiler)
                assert len(state.profiler.functions) == 0

                output = io.StringIO()
                state.print_stats(stream=output, prefix="Test")

                # Should produce no output when no functions are profiled
                assert output.getvalue() == ""


class TestProfilerCleanup:
    """Test profiler cleanup behavior."""

    def test_shutdown_handles_multiple_calls(self):
        """Test that shutdown can be called multiple times safely."""
        with mock.patch.dict(os.environ, {ENV_VAR_ENABLE_LINE_PROFILER: "1"}):
            line_profiler.ProfilerState._instance = None
            state = line_profiler.ProfilerState()

            # Should not raise an exception when called multiple times
            state.shutdown()
            state.shutdown()
            state.shutdown()
