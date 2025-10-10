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

"""Shared fixtures and utilities for profiling tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_profile_dir():
    """Provide a temporary directory for profiler output."""
    # Clear cached output directory before test
    from inference_endpoint.profiling import profiler_utils

    profiler_utils._run_output_dir = None

    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

    # Clear cache after test
    profiler_utils._run_output_dir = None


def sample_workload():
    """A sample workload function to profile."""
    total = 0
    for i in range(1000):
        total += i * 2
    return total


def check_profiler_library_available(library_name: str) -> bool:
    """Check if a profiler library is installed."""
    try:
        __import__(library_name)
        return True
    except ImportError:
        return False
