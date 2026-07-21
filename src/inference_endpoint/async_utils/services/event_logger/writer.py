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

"""Writer base class for event records."""

import time
from abc import ABC, abstractmethod

from inference_endpoint.core.record import EventRecord


class RecordWriter(ABC):
    """Abstract base class for writing event records.

    Supports two independent auto-flush triggers: a count-based interval
    (after every N records) and a time-based latency bound (after this many
    seconds since the last flush). The time-based trigger keeps low-rate
    streams durable on disk without waiting to accumulate a full count batch.
    """

    def __init__(
        self,
        *args,
        flush_interval: int | None = None,
        max_flush_latency_s: float | None = None,
    ):
        """Initialize the writer.

        Args:
            flush_interval: If set, flush after every this many records written.
                None means no count-based flushing.
            max_flush_latency_s: If set, flush on the next write() once this many
                seconds have elapsed since the last flush, even if flush_interval
                has not been reached. Bounds on-disk staleness for low-rate
                streams. None means no time-based flushing.
        """
        self._flush_interval = flush_interval
        self._max_flush_latency_s = max_flush_latency_s
        self._n_since_last_flush = 0
        self._last_flush_monotonic = time.monotonic()

    def write(self, record: EventRecord) -> None:
        """Write a record and optionally flush based on count or elapsed time."""
        self._write_record(record)
        self._n_since_last_flush += 1
        if self._n_since_last_flush == 0:
            return
        if (
            self._flush_interval is not None
            and self._n_since_last_flush >= self._flush_interval
        ):
            self.flush()
        elif (
            self._max_flush_latency_s is not None
            and time.monotonic() - self._last_flush_monotonic
            >= self._max_flush_latency_s
        ):
            self.flush()

    @abstractmethod
    def _write_record(self, record: EventRecord) -> None:
        """Write an event record. Subclasses must implement this method."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def close(self) -> None:
        """Close the writer and release resources."""
        raise NotImplementedError("Subclasses must implement this method.")

    def flush(self) -> None:
        """Flush the writer to ensure all data is written to the underlying storage.

        Also resets the flush-interval count and the elapsed-time baseline so the
        next auto-flush happens after another N records or another latency window
        (whether flush was triggered by count, time, or manually).
        """
        self._n_since_last_flush = 0
        self._last_flush_monotonic = time.monotonic()
