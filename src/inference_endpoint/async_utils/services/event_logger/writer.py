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

from abc import ABC, abstractmethod

from inference_endpoint.core.record import EventRecord


class RecordWriter(ABC):
    """Abstract base class for writing event records.

    Supports an optional flush interval: after every N records written via
    write(), the writer is automatically flushed.
    """

    def __init__(self, *args, flush_interval: int | None = None):
        """Initialize the writer.

        Args:
            flush_interval: If set, flush after every this many records written.
                None means no automatic flushing.
        """
        self._flush_interval = flush_interval
        self._n_since_last_flush = 0

    def write(self, record: EventRecord) -> None:
        """Write a record and optionally flush based on flush_interval."""
        self._write_record(record)
        self._n_since_last_flush += 1
        if (
            self._flush_interval is not None
            and self._n_since_last_flush >= self._flush_interval
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

        Also resets the flush-interval count so the next flush happens after
        another N records (whether flush was triggered by the interval or manually).
        """
        self._n_since_last_flush = 0
