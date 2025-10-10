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

"""Unit tests for the profiling module __init__.py (unified API)."""

import os


class TestProfilingUnifiedAPI:
    """Test the unified profiling API works correctly."""

    def test_profile_decorator_usage(self):
        """Test @profile decorator can be used without errors."""
        from inference_endpoint.profiling import profile

        @profile
        def compute(x):
            return x * 2

        # Should work regardless of profiler state
        assert compute(5) == 10
        assert compute.__name__ == "compute"

    def test_profiler_prevent_init_clears_environment(self):
        """Test profiler_prevent_init clears profiling environment variables."""
        # Set profiling env vars
        os.environ["ENABLE_LINE_PROFILER"] = "1"
        os.environ["ENABLE_PYINSTRUMENT"] = "1"
        os.environ["ENABLE_YAPPI"] = "1"
        os.environ["ENABLE_LOOP_STATS"] = "1"

        from inference_endpoint.profiling import profiler_prevent_init

        profiler_prevent_init()

        # All should be cleared
        assert "ENABLE_LINE_PROFILER" not in os.environ
        assert "ENABLE_PYINSTRUMENT" not in os.environ
        assert "ENABLE_YAPPI" not in os.environ
        assert "ENABLE_LOOP_STATS" not in os.environ
