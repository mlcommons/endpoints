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

"""End-to-end MetricsAggregator tests.

The legacy E2E tests asserted against ``InMemoryKVStore`` snapshots; the
registry/publisher refactor replaces that surface entirely. Skipped at
module load pending rewrite as a pub/sub round-trip test.
"""

import pytest

pytest.skip(
    reason=(
        "TODO: migrate to pub/sub round-trip tests, tracked in "
        "metrics_pubsub_design_v5.md test impact section"
    ),
    allow_module_level=True,
)
