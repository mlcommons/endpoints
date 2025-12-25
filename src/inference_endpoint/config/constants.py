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

"""Global constants and mappings for the inference endpoint package."""

# Mapping from endpoints results keys to MLPerf loadgen and submission checker supported keys
# This ensures compatibility when generating user.conf and mlperf_log_details.txt for submission checker
# Format: {"endpoints_key": "loadgen_key"}
ENDPOINTS_TO_LOADGEN_KEY_MAPPING = {
    "n_samples_from_dataset": "qsl_reported_performance_count",
    # "n_samples_to_issue": "",
    # "total_samples_to_issue": "",
    "max_duration_ms": "effective_max_duration_ms",
    "min_duration_ms": "effective_min_duration_ms",
    "min_sample_count": "effective_min_query_count",
}
