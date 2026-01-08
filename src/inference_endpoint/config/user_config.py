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

from dataclasses import dataclass


@dataclass
class UserConfig:
    """Any value set to None will default to the value in the ruleset."""

    user_metric_target: float | None = (
        None  # Used as the baseline value for the metric to initialize the load generator with.
    )
    min_duration_ms: int | None = None
    max_duration_ms: int | None = None
    ds_subset_size: int | None = (
        None  # If set, will use min(ds_subset_size, dataset_size). Otherwise, defaults to dataset_size.
    )
    total_sample_count: int | None = (
        None  # If set, this is the total number of samples loadgen will send to the SUT over the course of the test.
    )
    # Otherwise, this is calculated based on ds_subset_size, user_metric_target, and min_duration_ms.
    min_sample_count: int | None = (
        None  # If total_sample_count is set, this is ignored. Otherwise, this is the minimum number of samples loadgen will
    )
    # send to the SUT over the course of the test.
