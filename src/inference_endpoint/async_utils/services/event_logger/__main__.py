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


"""EventPublisherService subscriber for logging event records.

Currently supported:
    - JSONL file

Planned:
    - SQL-compatible Database
"""

import argparse
import asyncio
import os
from abc import ABC, abstractmethod
from pathlib import Path

import msgspec
from inference_endpoint.async_utils.loop_manager import LoopManager
from inference_endpoint.async_utils.transport.record import (
    ErrorEventType,
    EventRecord,
    EventType,
    SessionEventType,
)
from inference_endpoint.async_utils.transport.zmq.context import ManagedZMQContext
from inference_endpoint.async_utils.transport.zmq.pubsub import ZmqEventRecordSubscriber


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


def _is_error_event(record: EventRecord) -> bool:
    """True if the record is an error event (should not be dropped after ENDED)."""
    return isinstance(record.event_type, ErrorEventType)


class EventLoggerService(ZmqEventRecordSubscriber):
    """Event logger service for logging event records.

    When SessionEventType.ENDED is received, the service stops accepting
    further events (except Error events), closes writers, and stops the event loop.
    Writers are only closed after the current batch is fully processed, so error
    events that appear in the same batch after ENDED are still written.
    """

    def __init__(
        self,
        log_dir: Path,
        *args,
        writer_classes: tuple[type[RecordWriter], ...] = (JSONLWriter,),
        flush_interval: int | None = 100,
        shutdown_event: asyncio.Event | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._shutdown_received = False
        self._shutdown_event = shutdown_event

        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)

        if not log_dir.is_dir():
            raise NotADirectoryError(f"Log directory {log_dir} is not a directory")

        if not os.access(log_dir, os.W_OK):
            raise PermissionError(f"Log directory {log_dir} is not writable")

        self.writers: list[RecordWriter] = []
        for writer_class in writer_classes:
            self.writers.append(
                writer_class(log_dir / "events", flush_interval=flush_interval)
            )

    def _write_record_to_writers(self, record: EventRecord) -> None:
        """Write a single record to all writers (uses write_record for flush-on-interval)."""
        for writer in self.writers:
            writer.write_record(record)

    def _close_writers_and_stop(self) -> None:
        """Flush and close all writers, clear the list, then request loop stop."""
        for writer in self.writers:
            writer.flush()
            writer.close()
        self.writers.clear()
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._request_stop)

    async def process(self, records: list[EventRecord]) -> None:
        saw_shutdown = False
        for record in records:
            if self._shutdown_received and not _is_error_event(record):
                continue
            if record.event_type == SessionEventType.ENDED:
                self._shutdown_received = True
                saw_shutdown = True
            self._write_record_to_writers(record)
        if saw_shutdown:
            self._close_writers_and_stop()

    def _request_stop(self) -> None:
        """Close this subscriber and signal shutdown (or stop the loop if no shutdown_event)."""
        self.close()
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        elif self.loop is not None and self.loop.is_running():
            self.loop.stop()

    def close(self) -> None:
        for writer in self.writers:
            writer.close()
        self.writers.clear()
        super().close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Event logger service")
    parser.add_argument("--log-dir", type=Path, required=True, help="Log directory")
    parser.add_argument(
        "--socket-address",
        type=str,
        required=True,
        help="ZMQ socket address to connect to",
    )
    args = parser.parse_args()

    shutdown_event = asyncio.Event()
    loop = LoopManager().default_loop
    with ManagedZMQContext.scoped(socket_dir=args.log_dir.parent) as zmq_ctx:
        logger = EventLoggerService(
            args.log_dir,
            args.socket_address,
            zmq_ctx,
            loop,
            topics=None,  # Subscribe to all topics for logging
            shutdown_event=shutdown_event,
        )

        loop.call_soon_threadsafe(logger.start)
        await shutdown_event.wait()


if __name__ == "__main__":
    LoopManager().default_loop.run_until_complete(main())
