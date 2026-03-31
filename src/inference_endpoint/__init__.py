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

"""
MLPerf Inference Endpoint Benchmarking System

A high-performance benchmarking tool for LLM endpoints with 50k QPS capability.
"""

import os as _os

# Suppress transformers "None of PyTorch…" advisory — we only use tokenizers.
# Must be set before any submodule triggers `import transformers`.
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
del _os

# Core imports - public API types
from .core.types import Query, QueryId, QueryResult  # noqa: E402

__version__ = "0.1.0"
__author__ = "MLPerf Inference Endpoint Team"
__description__ = "High-performance LLM endpoint benchmarking system"

__all__ = [
    "Query",
    "QueryResult",
    "QueryId",
]
