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


"""Manages asyncio and event loop creation and bookkeeping."""

import asyncio
import threading
from collections import namedtuple
from functools import partial
from typing import Literal

import uvloop

from inference_endpoint.utils import SingletonMixin

ManagedLoop = namedtuple("ManagedLoop", ["loop", "thread"])


class LoopManager(SingletonMixin):
    """Manages asyncio event loops within the main process.

    This class is a singleton - any new constructor calls will return the same
    instance as any other.

    Event loops are named and can be created and accessed by name.
    By default, there is always a "default" loop running on the main thread.
    Any other loops created will be launched on separate threads, and as such
    any tasks that are enqueued should be done so with `call_soon_threadsafe` or
    any other threadsafe methods.

    This also means that other threads can still enqueue tasks onto the main thread's
    event loop, which is useful for notifying shutdowns or synchronizing states.
    """

    def __init__(self, default_backend: str = "uvloop"):
        if getattr(self, "_initialized", False):
            return
        self.loops: dict[str, ManagedLoop] = {}

        # Create default loop
        # MyPy doesn't behave well with Literal
        self.create_loop(name="default", backend=default_backend)  # type: ignore[arg-type]
        self._initialized = True

    @property
    def default_loop(self) -> asyncio.AbstractEventLoop:
        """Get the default event loop.

        Returns:
            The default event loop.
        """
        managed = self.loops.get("default")
        if managed is None:
            raise RuntimeError("Default loop not found")
        return managed.loop

    def create_loop(
        self,
        name: str,
        backend: Literal["uvloop", "asyncio"] = "uvloop",
        task_factory_mode: Literal["eager", "lazy"] = "eager",
    ) -> asyncio.AbstractEventLoop:
        """Create a new event loop.

        Args:
            name: The name of the loop.
            backend: The backend to use for the loop.
            task_factory_mode: The mode to use for the task factory.

        Returns:
            The new event loop.
        """
        if name in self.loops:
            return self.loops[name].loop

        if backend == "uvloop":
            loop = uvloop.new_event_loop()
        elif backend == "asyncio":
            loop = asyncio.new_event_loop()
        else:
            raise ValueError(f"Invalid backend: {backend}")

        if task_factory_mode == "eager":
            loop.set_task_factory(asyncio.eager_task_factory)

        if name == "default":
            asyncio.set_event_loop(loop)
            self.loops[name] = ManagedLoop(loop=loop, thread=None)
        else:
            thread = threading.Thread(
                target=loop.run_forever,
                daemon=True,
                name=f"ManagedEventLoop-{name}",
            )
            self.loops[name] = ManagedLoop(loop=loop, thread=thread)
            thread.start()
        return loop

    def get_loop(self, name: str) -> asyncio.AbstractEventLoop:
        """Get a loop by name.

        Args:
            name: The name of the loop.

        Returns:
            The event loop.
        """
        return self.loops[name].loop

    def stop_loop(self, name: str, immediate: bool = False) -> None:
        """Stop a loop by name. The main event loop cannot be stopped.

        Args:
            name: The name of the loop.
            immediate: Whether to stop the loop immediately. If False, the loop will be stopped
                after all pending tasks are completed.
        """
        if name == "default":
            raise ValueError("The main event loop cannot be stopped.")

        loop: asyncio.AbstractEventLoop = self.get_loop(name)
        if immediate:
            loop.stop()
            self.loops.pop(name)
        else:
            loop.call_soon_threadsafe(loop.stop)
            self.default_loop.call_soon_threadsafe(partial(self.loops.pop, name))
