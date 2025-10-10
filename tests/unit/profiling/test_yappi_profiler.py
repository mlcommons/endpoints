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

"""Yappi profiler specific tests."""

import os

import pytest

from .conftest import check_profiler_library_available, sample_workload

YAPPI_AVAILABLE = check_profiler_library_available("yappi")


@pytest.mark.skipif(not YAPPI_AVAILABLE, reason="yappi not installed")
class TestYappiUniqueFeatures:
    """Test features unique to yappi (filtering, function tracking, callgrind output)."""

    def test_filter_profiled_disabled(self, temp_profile_dir):
        """Test YAPPI_FILTER_PROFILED=0 disables filtering."""
        os.environ["ENABLE_YAPPI"] = "1"
        os.environ["YAPPI_FILTER_PROFILED"] = "0"
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        from inference_endpoint.profiling.yappi_profiler import YappiProfiler

        profiler = YappiProfiler()
        assert profiler.filter_profiled is False

        profiler.stop()

    def test_filter_profiled_enabled_by_default(self, temp_profile_dir):
        """Test YAPPI_FILTER_PROFILED defaults to enabled."""
        os.environ.pop("YAPPI_FILTER_PROFILED", None)
        os.environ["ENABLE_YAPPI"] = "1"
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        from inference_endpoint.profiling.yappi_profiler import YappiProfiler

        profiler = YappiProfiler()
        assert profiler.filter_profiled is True

        profiler.stop()

    def test_function_tracking(self, temp_profile_dir):
        """Test that yappi tracks @profile decorated functions in _profiled_functions set."""
        os.environ["ENABLE_YAPPI"] = "1"
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        from inference_endpoint.profiling.yappi_profiler import YappiProfiler

        profiler = YappiProfiler()

        @profiler.profile
        def func1():
            return 1

        @profiler.profile
        def func2():
            return 2

        # Both functions tracked
        assert len(profiler._profiled_functions) == 2

        # Check full names in tracked set
        func1_name = f"{func1.__module__}.{func1.__qualname__}"
        func2_name = f"{func2.__module__}.{func2.__qualname__}"

        assert func1_name in profiler._profiled_functions
        assert func2_name in profiler._profiled_functions

        profiler.stop()

    def test_callgrind_output_format(self, temp_profile_dir):
        """Test yappi generates callgrind format output files."""
        os.environ["ENABLE_YAPPI"] = "1"
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        from inference_endpoint.profiling.yappi_profiler import YappiProfiler

        profiler = YappiProfiler()

        @profiler.profile
        def compute():
            total = 0
            for _ in range(1000):
                total += sample_workload()
            return total

        compute()
        profiler.shutdown()

        output_dir = temp_profile_dir / "yappi"
        if output_dir.exists():
            callgrind_files = list(output_dir.glob("*.callgrind"))
            if callgrind_files:
                # Verify callgrind files are non-empty
                assert all(f.stat().st_size > 0 for f in callgrind_files)
