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

"""Pyinstrument profiler specific tests."""

import os

import pytest

from .conftest import check_profiler_library_available

PYINSTRUMENT_AVAILABLE = check_profiler_library_available("pyinstrument")


@pytest.mark.skipif(not PYINSTRUMENT_AVAILABLE, reason="pyinstrument not installed")
class TestPyinstrumentUniqueFeatures:
    """Test features unique to pyinstrument (statistical sampling, async_mode)."""

    def test_marker_attribute_set(self, temp_profile_dir):
        """Test that pyinstrument sets _pyinstrument_profiled marker."""
        os.environ["ENABLE_PYINSTRUMENT"] = "1"
        os.environ["PROFILER_OUT_DIR"] = str(temp_profile_dir)

        from inference_endpoint.profiling.pyinstrument_profiler import (
            PyinstrumentProfiler,
        )

        profiler = PyinstrumentProfiler()

        @profiler.profile
        def compute(x):
            return x * 2

        # Marker attribute should be set
        assert hasattr(compute, "_pyinstrument_profiled")
        assert compute._pyinstrument_profiled is True

        profiler.stop()
