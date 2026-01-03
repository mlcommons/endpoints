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

"""Asyncio utilities for the MLPerf Inference Endpoint Benchmarking System."""

import asyncio


def create_eager_loop() -> asyncio.AbstractEventLoop:
    """Create event loop with eager task factory for better performance.

    The eager task factory immediately starts task execution rather than
    scheduling it for later, which can provide better performance for
    I/O-bound workloads.

    Returns:
        asyncio.AbstractEventLoop: A new event loop with eager task factory configured.
    """
    loop = asyncio.new_event_loop()
    loop.set_task_factory(asyncio.eager_task_factory)
    return loop
