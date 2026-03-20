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

"""Async runner with uvloop and eager_task_factory."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import TypeVar

import uvloop

T = TypeVar("T")


def run_async(coro: Coroutine[object, object, T]) -> T:
    """Run a coroutine with uvloop and eager_task_factory.

    Creates a fresh event loop per invocation. This is the standard way for
    synchronous CLI command handlers to execute async logic.
    """
    with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
        runner.get_loop().set_task_factory(asyncio.eager_task_factory)  # type: ignore[arg-type]
        return runner.run(coro)
