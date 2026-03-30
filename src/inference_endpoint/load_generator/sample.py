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

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..core.types import QueryResult, StreamChunk

logger = logging.getLogger(__name__)


class SampleEvent(str, Enum):
    """Event types for sample lifecycle hooks.

    Used by _SampleEventHandler for hook registration. These are local to the
    hook system and separate from core.record.SampleEventType which is used
    for the event publishing pipeline.
    """

    COMPLETE = "complete"
    FIRST_CHUNK = "first_chunk_received"
    NON_FIRST_CHUNK = "non_first_chunk_received"


class Sample:
    """Represents a sample/query to be sent to an inference endpoint.

    A Sample encapsulates the request data and provides a unique identifier for
    tracking through the benchmark lifecycle. It enforces immutability to prevent
    accidental modification during benchmarking.

    Immutability rules:
    - UUID is immutable once set (on creation)
    - Data can be set once from None to a value, then immutable
    - This allows delayed data loading while maintaining safety

    Memory optimization:
    - Uses __slots__ to reduce memory overhead
    - UUID as hex string (32 chars) instead of UUID object

    Attributes:
        uuid: Unique hex string identifier for this sample (32 characters).
        data: Request payload (dict, typically with prompt/model/params).
              Can be None initially and set once.

    Example:
        >>> sample = Sample({"prompt": "Hello", "model": "gpt-4"})
        >>> sample.uuid  # '8f3d2a1b...' (32 char hex)
        >>> sample.data["prompt"]  # 'Hello'
    """

    __slots__ = ["uuid", "data"]

    def __init__(self, data: Any):
        """Initialize sample with data and generate unique ID.

        Args:
            data: Request data to send to endpoint. Can be None if data
                 will be loaded later, but can only be set once.
        """
        # 128-bit UUID might be a little overkill for our use case, we can investigate slimming down memory usage
        self.uuid = uuid.uuid4().hex
        self.data = data

    def __setattr__(self, name: str, value: Any):
        if not hasattr(self, name) or (name == "data" and self.data is None):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(f"Sample is immutable - cannot set attribute: {name}")


class _SampleEventHandler:
    """Contains handlers for SampleEvents given a sample UUID. This is also to avoid needing other classes
    to do their own bookkeeping for Sample objects, which can be discarded once they are issued, as long as
    their UUIDs are saved.

    This class is a singleton rather than a class method mainly because it needs to hold some state (i.e. hooks)

    A user can register hooks to any event type, and will be run in the order they were registered.
    A valid hook is a callable that takes a single argument, representing the response object (StreamChunk or QueryResult).

    A simple example use-case of a hook is to update a progress bar on-completion of a sample.

    NOTE: Hook lists are not thread-safe. Hooks must be registered before the benchmark
    starts (single-threaded setup phase). This is a known limitation; _SampleEventHandler
    is being deprecated in favor of the pub-sub EventLoggerService.
    """

    __slots__ = ["first_chunk_hooks", "non_first_chunk_hooks", "complete_hooks"]

    SINGLETON = None
    _initialized = False

    def __new__(cls):
        if cls.SINGLETON is None:
            cls.SINGLETON = super().__new__(cls)
        return cls.SINGLETON

    def __init__(self):
        if _SampleEventHandler._initialized:
            return
        _SampleEventHandler._initialized = True

        self.first_chunk_hooks = []
        self.non_first_chunk_hooks = []
        self.complete_hooks = []

    def register_hook(
        self,
        event_type: SampleEvent,
        hook: Callable[[StreamChunk], None] | Callable[[QueryResult], None],
    ) -> None:
        if event_type == SampleEvent.FIRST_CHUNK:
            self.first_chunk_hooks.append(hook)
        elif event_type == SampleEvent.NON_FIRST_CHUNK:
            self.non_first_chunk_hooks.append(hook)
        elif event_type == SampleEvent.COMPLETE:
            self.complete_hooks.append(hook)
        else:
            raise ValueError(f"Invalid event type: {event_type}")

    def clear_hooks(self, ev_type: SampleEvent | None = None) -> None:
        if ev_type is None:
            self.first_chunk_hooks.clear()
            self.non_first_chunk_hooks.clear()
            self.complete_hooks.clear()
        elif ev_type == SampleEvent.FIRST_CHUNK:
            self.first_chunk_hooks.clear()
        elif ev_type == SampleEvent.NON_FIRST_CHUNK:
            self.non_first_chunk_hooks.clear()
        elif ev_type == SampleEvent.COMPLETE:
            self.complete_hooks.clear()

    def stream_chunk_complete(self, chunk: StreamChunk) -> None:
        """Handle completion of a streaming chunk.

        Called when a chunk arrives from a streaming response. Invokes
        registered hooks for first/non-first chunks.

        Args:
            chunk: StreamChunk containing response data and metadata.
        """
        assert isinstance(chunk, StreamChunk), f"Invalid chunk type: {type(chunk)}"

        if chunk.metadata.get("first_chunk", False):
            hooks = self.first_chunk_hooks
        else:
            hooks = self.non_first_chunk_hooks

        for hook in hooks:
            hook(chunk)

    def query_result_complete(self, result: QueryResult) -> None:
        """Handle completion of a query (success or failure).

        Called when a query finishes (with response or error). Invokes
        registered completion hooks.

        Args:
            result: QueryResult containing response data or error information.
        """
        assert isinstance(result, QueryResult), f"Invalid result type: {type(result)}"

        if result.error is not None:
            logger.error(f"Error in request {result.id}: {result.error}")

        for hook in self.complete_hooks:
            hook(result)


@dataclass
class IssuedSample:
    """Contains data about a sample that has been issued to the inference endpoint.

    SampleIssuer is not allowed to know the actual sample index of the data to prevent cheating
    and response caching. This class contains metadata about the sample for bookkeeping by the
    LoadGenerator and BenchmarkSession.
    """

    sample: Sample
    index: int
    issue_timestamp_ns: int


SampleEventHandler = _SampleEventHandler()
