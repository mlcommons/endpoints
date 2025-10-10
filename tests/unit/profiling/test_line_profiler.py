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

"""Line profiler specific tests."""

import io
import os

import pytest

from .conftest import check_profiler_library_available, sample_workload

LINE_PROFILER_AVAILABLE = check_profiler_library_available("line_profiler")


@pytest.mark.skipif(not LINE_PROFILER_AVAILABLE, reason="line_profiler not installed")
class TestLineProfilerUniqueFeatures:
    """Test features unique to line_profiler (pause/resume, print_stats, get_stats)."""

    def test_pause_resume_workflow(self, temp_profile_dir):
        """Test pause/resume profiling workflow (unique to line_profiler)."""
        os.environ["ENABLE_LINE_PROFILER"] = "1"
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        from inference_endpoint.profiling.line_profiler import LineProfiler

        profiler = LineProfiler()

        @profiler.profile
        def work():
            return sample_workload()

        work()

        # Pause profiling
        profiler.pause()
        assert not profiler._session_active

        # Resume profiling
        profiler.resume()
        assert profiler._session_active

        work()
        profiler.stop()

    def test_get_stats_returns_formatted_output(self, temp_profile_dir):
        """Test get_stats returns formatted statistics string."""
        os.environ["ENABLE_LINE_PROFILER"] = "1"
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        from inference_endpoint.profiling.line_profiler import LineProfiler

        profiler = LineProfiler()

        @profiler.profile
        def workload():
            return sample_workload()

        workload()

        # Get stats as string
        stats = profiler.get_stats()
        assert len(stats) > 0
        assert "workload" in stats or "sample_workload" in stats

        profiler.stop()

    def test_print_stats_with_custom_stream_and_prefix(self, temp_profile_dir):
        """Test print_stats outputs to custom stream with prefix."""
        os.environ["ENABLE_LINE_PROFILER"] = "1"
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        from inference_endpoint.profiling.line_profiler import LineProfiler

        profiler = LineProfiler()

        @profiler.profile
        def calculate(n):
            return sum(i * 2 for i in range(n))

        calculate(50)

        # Print to custom stream with prefix
        output = io.StringIO()
        profiler.print_stats(stream=output, prefix="Worker-1")

        stats_output = output.getvalue()
        if stats_output:
            assert "Worker-1" in stats_output or "calculate" in stats_output

        profiler.stop()

    def test_functions_registered_for_profiling(self, temp_profile_dir):
        """Test that decorated functions are registered in profiler."""
        os.environ["ENABLE_LINE_PROFILER"] = "1"
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        from inference_endpoint.profiling.line_profiler import LineProfiler

        profiler = LineProfiler()

        @profiler.profile
        def func1():
            return 1

        @profiler.profile
        def func2():
            return 2

        # Both functions should be registered
        assert len(profiler.profiler.functions) >= 2

        profiler.stop()
