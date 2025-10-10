# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Parameterized tests for common profiler functionality across all profilers."""

import asyncio
import os
from unittest import mock

import pytest

from .conftest import check_profiler_library_available

# Check availability
LINE_PROFILER_AVAILABLE = check_profiler_library_available("line_profiler")
YAPPI_AVAILABLE = check_profiler_library_available("yappi")
PYINSTRUMENT_AVAILABLE = check_profiler_library_available("pyinstrument")

# Profiler configurations: (env_var, module_path, class_name, has_marker, available)
PROFILER_CONFIGS = [
    pytest.param(
        "ENABLE_LINE_PROFILER",
        "inference_endpoint.profiling.line_profiler",
        "LineProfiler",
        None,
        LINE_PROFILER_AVAILABLE,
        id="line_profiler",
    ),
    pytest.param(
        "ENABLE_PYINSTRUMENT",
        "inference_endpoint.profiling.pyinstrument_profiler",
        "PyinstrumentProfiler",
        "_pyinstrument_profiled",
        PYINSTRUMENT_AVAILABLE,
        id="pyinstrument",
    ),
    pytest.param(
        "ENABLE_YAPPI",
        "inference_endpoint.profiling.yappi_profiler",
        "YappiProfiler",
        "_yappi_profiled",
        YAPPI_AVAILABLE,
        id="yappi",
    ),
]


class TestCommonProfilerBehavior:
    """Test behavior common to all profilers using parameterization."""

    @pytest.mark.parametrize(
        "env_var,module_path,class_name,marker,available", PROFILER_CONFIGS
    )
    def test_profiler_disabled_by_default(
        self, env_var, module_path, class_name, marker, available
    ):
        """Test all profilers are disabled by default."""
        if not available:
            pytest.skip(f"{class_name} not installed")

        # Ensure env var is not set from previous tests
        os.environ.pop(env_var, None)

        # Import the class
        module = __import__(module_path, fromlist=[class_name])
        ProfilerClass = getattr(module, class_name)

        profiler = ProfilerClass()
        assert not profiler.enabled

    @pytest.mark.parametrize(
        "env_var,module_path,class_name,marker,available", PROFILER_CONFIGS
    )
    def test_profiler_enabled_with_env_var(
        self, env_var, module_path, class_name, marker, available, temp_profile_dir
    ):
        """Test all profilers enable when their env var is set."""
        if not available:
            pytest.skip(f"{class_name} not installed")

        os.environ[env_var] = "1"
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        module = __import__(module_path, fromlist=[class_name])
        ProfilerClass = getattr(module, class_name)

        profiler = ProfilerClass()
        assert profiler.enabled
        assert profiler._session_active

        profiler.stop()

        # Clean up
        del os.environ[env_var]
        del os.environ["PROFILER_OUT_DIR"]

    @pytest.mark.parametrize(
        "env_var,module_path,class_name,marker,available", PROFILER_CONFIGS
    )
    def test_profile_decorator_preserves_function_name(
        self, env_var, module_path, class_name, marker, available, temp_profile_dir
    ):
        """Test @profile decorator preserves function names across all profilers."""
        if not available:
            pytest.skip(f"{class_name} not installed")

        os.environ[env_var] = "1"
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        module = __import__(module_path, fromlist=[class_name])
        ProfilerClass = getattr(module, class_name)

        profiler = ProfilerClass()

        @profiler.profile
        def test_func(x):
            return x * 2

        # Function name preserved
        assert test_func.__name__ == "test_func"
        # Function works
        assert test_func(5) == 10
        # Marker attribute set (if applicable)
        if marker:
            assert hasattr(test_func, marker)
            assert getattr(test_func, marker) is True

        profiler.stop()

        # Clean up
        del os.environ[env_var]
        del os.environ["PROFILER_OUT_DIR"]

    @pytest.mark.parametrize(
        "env_var,module_path,class_name,marker,available", PROFILER_CONFIGS
    )
    def test_profile_async_preserves_coroutine(
        self, env_var, module_path, class_name, marker, available, temp_profile_dir
    ):
        """Test @profile decorator preserves async function properties."""
        if not available:
            pytest.skip(f"{class_name} not installed")

        os.environ[env_var] = "1"
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        module = __import__(module_path, fromlist=[class_name])
        ProfilerClass = getattr(module, class_name)

        profiler = ProfilerClass()

        @profiler.profile
        async def async_func(x):
            await asyncio.sleep(0.001)
            return x * 2

        # Function name preserved
        assert async_func.__name__ == "async_func"
        # Still a coroutine
        assert asyncio.iscoroutinefunction(async_func)
        # Function works
        result = asyncio.run(async_func(5))
        assert result == 10

        profiler.stop()

        # Clean up
        del os.environ[env_var]
        del os.environ["PROFILER_OUT_DIR"]

    @pytest.mark.parametrize(
        "env_var,module_path,class_name,marker,available", PROFILER_CONFIGS
    )
    def test_start_stop_lifecycle(
        self, env_var, module_path, class_name, marker, available, temp_profile_dir
    ):
        """Test start/stop lifecycle for all profilers."""
        if not available:
            pytest.skip(f"{class_name} not installed")

        os.environ[env_var] = "1"
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        module = __import__(module_path, fromlist=[class_name])
        ProfilerClass = getattr(module, class_name)

        profiler = ProfilerClass()
        assert profiler._session_active

        # Stop
        profiler.stop()
        assert not profiler._session_active

        # Restart
        profiler.start()
        assert profiler._session_active

        # Stop again
        profiler.stop()

        # Clean up
        del os.environ[env_var]
        del os.environ["PROFILER_OUT_DIR"]

    @pytest.mark.parametrize(
        "env_var,module_path,class_name,marker,available", PROFILER_CONFIGS
    )
    def test_shutdown_idempotent(
        self, env_var, module_path, class_name, marker, available, temp_profile_dir
    ):
        """Test shutdown is safe to call multiple times."""
        if not available:
            pytest.skip(f"{class_name} not installed")

        os.environ[env_var] = "1"
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        module = __import__(module_path, fromlist=[class_name])
        ProfilerClass = getattr(module, class_name)

        profiler = ProfilerClass()

        # Multiple shutdowns should not crash
        for _ in range(3):
            profiler.shutdown()

        # Clean up
        del os.environ[env_var]
        del os.environ["PROFILER_OUT_DIR"]

    @pytest.mark.parametrize(
        "env_var,module_path,class_name,marker,available", PROFILER_CONFIGS
    )
    def test_check_subprocess_spawn_handles_pid_change(
        self, env_var, module_path, class_name, marker, available, temp_profile_dir
    ):
        """Test check_subprocess_spawn reinitializes on PID change."""
        if not available:
            pytest.skip(f"{class_name} not installed")

        os.environ[env_var] = "1"
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        module = __import__(module_path, fromlist=[class_name])
        ProfilerClass = getattr(module, class_name)

        profiler = ProfilerClass()
        original_pid = profiler._original_pid

        # Simulate fork
        with mock.patch("os.getpid", return_value=original_pid + 999):
            profiler.check_subprocess_spawn()

            # Should update to new PID
            assert profiler._original_pid == original_pid + 999
            # Should reset state
            assert not profiler._stats_written

        profiler.stop()

        # Clean up
        del os.environ[env_var]
        del os.environ["PROFILER_OUT_DIR"]

    @pytest.mark.parametrize(
        "env_var,module_path,class_name,marker,available", PROFILER_CONFIGS
    )
    def test_check_subprocess_spawn_noop_when_disabled(
        self, env_var, module_path, class_name, marker, available
    ):
        """Test check_subprocess_spawn is no-op when disabled."""
        if not available:
            pytest.skip(f"{class_name} not installed")

        module = __import__(module_path, fromlist=[class_name])
        ProfilerClass = getattr(module, class_name)

        profiler = ProfilerClass()  # Disabled by default

        # Should not raise
        profiler.check_subprocess_spawn()


class TestCommonEdgeCases:
    """Test edge cases common to all profilers."""

    @pytest.mark.parametrize(
        "env_var,module_path,class_name,marker,available", PROFILER_CONFIGS
    )
    def test_stop_when_not_started(
        self, env_var, module_path, class_name, marker, available, temp_profile_dir
    ):
        """Test stop is safe when profiler not started."""
        if not available:
            pytest.skip(f"{class_name} not installed")

        os.environ[env_var] = "1"
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        module = __import__(module_path, fromlist=[class_name])
        ProfilerClass = getattr(module, class_name)

        profiler = ProfilerClass()

        # Stop multiple times
        profiler.stop()
        profiler.stop()
        profiler.stop()

        # Clean up
        del os.environ[env_var]
        del os.environ["PROFILER_OUT_DIR"]

    @pytest.mark.parametrize(
        "env_var,module_path,class_name,marker,available", PROFILER_CONFIGS
    )
    def test_start_when_already_started(
        self, env_var, module_path, class_name, marker, available, temp_profile_dir
    ):
        """Test start is no-op when already started."""
        if not available:
            pytest.skip(f"{class_name} not installed")

        os.environ[env_var] = "1"
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        module = __import__(module_path, fromlist=[class_name])
        ProfilerClass = getattr(module, class_name)

        profiler = ProfilerClass()
        assert profiler._session_active

        # Start again
        profiler.start()
        assert profiler._session_active

        profiler.stop()

        # Clean up
        del os.environ[env_var]
        del os.environ["PROFILER_OUT_DIR"]

    @pytest.mark.parametrize(
        "env_var,module_path,class_name,marker,available", PROFILER_CONFIGS
    )
    def test_shutdown_without_profiling_data(
        self, env_var, module_path, class_name, marker, available, temp_profile_dir
    ):
        """Test shutdown without any profiling data doesn't crash."""
        if not available:
            pytest.skip(f"{class_name} not installed")

        os.environ[env_var] = "1"
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        module = __import__(module_path, fromlist=[class_name])
        ProfilerClass = getattr(module, class_name)

        profiler = ProfilerClass()

        # Stop immediately
        profiler.stop()

        # Shutdown should handle no data gracefully
        profiler.shutdown()

        # Clean up
        del os.environ[env_var]
        del os.environ["PROFILER_OUT_DIR"]
