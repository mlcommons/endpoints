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

"""Writer base class and file-based implementations for event records."""

from abc import ABC, abstractmethod
from pathlib import Path

import msgspec
from inference_endpoint.async_utils.transport.record import EventRecord, EventType


class RecordWriter(ABC):
    """Abstract base class for writing event records.

    Supports an optional flush interval: after every N records written via
    write_record(), the writer is automatically flushed.
    """

    def __init__(self, *args, flush_interval: int | None = None):
        """Initialize the writer.

        Args:
            flush_interval: If set, flush after every this many records written.
                None means no automatic flushing.
        """
        self._flush_interval = flush_interval
        self._n_since_last_flush = 0

    def write_record(self, record: EventRecord) -> None:
        """Write a record and optionally flush based on flush_interval."""
        self.write(record)
        self._n_since_last_flush += 1
        if (
            self._flush_interval is not None
            and self._n_since_last_flush >= self._flush_interval
        ):
            self.flush()

    @abstractmethod
    def write(self, record: EventRecord) -> None:
        """Write an event record."""
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


class FileWriter(RecordWriter):
    """Writer for writing event records to a file."""

    def __init__(
        self,
        file_path: Path,
        mode: str = "w",
        flush_interval: int | None = None,
        **kwargs: object,
    ):
        super().__init__(flush_interval=flush_interval)
        self.file_path = Path(file_path)
        # No idea what the 'IO' type MyPy thinks this is, apparently even io.IOBase does not work, so just ignore.
        self.file_obj = self.file_path.open(mode=mode)  # type: ignore[assignment]

    def close(self) -> None:
        if self.file_obj is not None:
            try:
                self.flush()
                self.file_obj.close()
            except (OSError, FileNotFoundError):
                # File may already be closed or I/O error on close (e.g. disk full).
                pass
            finally:
                self.file_obj = None  # type: ignore[assignment]

    def record_to_line(self, record: EventRecord) -> str:
        """Convert an event record to a line of text."""
        raise NotImplementedError("Subclasses must implement this method.")

    def write(self, record: EventRecord) -> None:
        if self.file_obj is not None:
            self.file_obj.write(self.record_to_line(record) + "\n")

    def flush(self) -> None:
        if self.file_obj is not None:
            self.file_obj.flush()
        super().flush()


class JSONLWriter(FileWriter):
    """Writes to a JSONL file."""

    extension = ".jsonl"

    def __init__(self, file_path: Path, *args, **kwargs):
        super().__init__(file_path.with_suffix(self.extension), *args, **kwargs)

        # EventRecords are msgspec structs so we can use the built-in JSON encoder
        self.encoder = msgspec.json.Encoder(enc_hook=EventType.encode_hook)

    def record_to_line(self, record: EventRecord) -> str:
        return self.encoder.encode(record).decode("utf-8")
