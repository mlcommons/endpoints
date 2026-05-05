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

"""Tests for MetricsTable and trigger plumbing.

The legacy tests in this file targeted the KVStore-backed table and have
not yet been migrated to the registry-based table introduced by
``metrics_pubsub_design_v5.md``. They are skipped at module load.
"""

import pytest

pytest.skip(
    reason=(
        "TODO: migrate to registry-based MetricsTable tests, tracked in "
        "metrics_pubsub_design_v5.md test impact section"
    ),
    allow_module_level=True,
)
