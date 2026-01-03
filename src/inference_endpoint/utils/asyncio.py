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
import os

import uvloop


def create_eager_loop(
    slow_callback_duration: float | None = None,
) -> asyncio.AbstractEventLoop:
    """Create uvloop event loop with eager task factory.

    The eager task factory immediately starts task execution rather than scheduling it for later,
    which can provide better performance for I/O-bound workloads.

    Args:
        slow_callback_duration: If set, enables debug mode and logs callbacks exceeding
            this duration in seconds. Can also be set via ASYNCIO_SLOW_CALLBACK_MS env var.

    Returns:
        asyncio.AbstractEventLoop: A uvloop event loop with eager task factory configured.
    """
    loop = uvloop.new_event_loop()
    loop.set_task_factory(asyncio.eager_task_factory)

    # Check env var for slow callback threshold (in milliseconds)
    if slow_callback_duration is None:
        env_ms = os.environ.get("ASYNCIO_SLOW_CALLBACK_MS")
        if env_ms:
            slow_callback_duration = float(env_ms) / 1000.0

    if slow_callback_duration is not None:
        loop.set_debug(True)
        loop.slow_callback_duration = slow_callback_duration

    return loop
