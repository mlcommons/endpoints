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

from __future__ import annotations

import atexit
import contextlib
import dataclasses
import logging
import multiprocessing
import queue
import shutil
import sqlite3
import threading
import time
import uuid
from functools import partial
from pathlib import Path
from typing import ClassVar

import orjson

from ..load_generator.events import Event, SampleEvent, SessionEvent
from ..profiling import profile
from ..utils import byte_quantity_to_str

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def sqlite3_cursor(path: Path):
    """Context manager for SQLite cursor that properly handles connection lifecycle.

    Args:
        path: Path to the SQLite database file.

    Yields:
        A SQLite cursor object.
    """
    conn = sqlite3.connect(str(path))
    cursor = conn.cursor()
    try:
        yield cursor, conn
    finally:
        cursor.close()
        conn.close()


@dataclasses.dataclass
class EventRow:
    sample_uuid: str
    event_type: Event
    timestamp_ns: int

    @staticmethod
    def to_table_query() -> str:
        return "CREATE TABLE IF NOT EXISTS events (sample_uuid VARCHAR(32), event_type VARCHAR(32), timestamp_ns INTEGER)"

    @staticmethod
    def insert_query() -> str:
        return "INSERT INTO events (sample_uuid, event_type, timestamp_ns) VALUES (?, ?, ?)"

    def to_insert_params(self) -> tuple[str, str, int]:
        return (
            self.sample_uuid,
            self.event_type.value,
            self.timestamp_ns,
        )


def register_cleanup(file_path: str):
    if multiprocessing.parent_process() is not None:
        return
    atexit.register(partial(Path(file_path).unlink, missing_ok=True))
    logger.debug(f"Registered at-exit cleanup for {file_path}")


class EventRecorderSingletonViolation(RuntimeError):
    """Raised when an attempt is made to create a second EventRecorder while one is already active.

    This is to prevent:
        - Multiple writer connections to the same database
        - Potential writes to the wrong event database if multiple are open
    """

    pass


class EventRecorder:
    """Records events to a shared memory database, which can be accessed across multiple processes.

    An optional session id can be provided to connect to an existing database. If the database does not exist, it will first check if /dev/shm has enough free space to
    create a new database.

    This class uses a dedicated writer thread to handle all SQLite operations, making it thread-safe.
    Events are queued via record_event() and processed asynchronously by the writer thread.

    Only 1 EventRecorder can be actively writing events at a time.
    """

    LIVE: EventRecorder | None = None

    _created_session_dbs: ClassVar[set[str]] = set()

    # Sentinel objects for queue control
    _STOP_SENTINEL: ClassVar[object] = object()
    _FORCE_COMMIT_SENTINEL: ClassVar[object] = object()

    def __init__(
        self,
        session_id: str | None = None,
        txn_buffer_size: int = 1000,
        min_memory_req_bytes: int = 512 * 1024 * 1024,
        notify_idle: threading.Event | None = None,
        close_timeout_s: float = 10.0,
    ):
        """Creates a new EventRecorder.

        Args:
            session_id: Optional session id to connect to an existing database. If not provided, a new database will be created.
            txn_buffer_size: The number of events to buffer before committing to the database. (Default: 1000)
            min_memory_req_bytes: The minimum amount of free space (in bytes) in /dev/shm required to create a new database. (Default: 1GB)
            notify_idle: Optional threading.Event. If provided, EventRecorder will set when the number of inflight samples is 0.
            close_timeout_s: The timeout in seconds to wait for the writer thread to finish processing when calling close(). (Default: 10.0)
        """
        if session_id is None:
            session_id = uuid.uuid4().hex

        self.session_id = session_id

        if self.connection_name not in EventRecorder._created_session_dbs:
            register_cleanup(self.connection_name)
            register_cleanup(self.outputs_path)
            EventRecorder._created_session_dbs.add(self.connection_name)

        if not Path(self.connection_name).parent.exists():
            raise FileNotFoundError(
                "Cannot create shm db, POSIX shm dir at /dev/shm does not exist"
            )

        if not Path(self.connection_name).exists():
            # If we're creating a new db, we require a minimum of 1GB of shared memory
            logging.debug(f"Creating new events db at {self.connection_name}")
            shm_stats = shutil.disk_usage("/dev/shm")
            logging.debug(
                f"/dev/shm usage stats: total={shm_stats.total}B, free={shm_stats.free}B"
            )

            min_memory_req_str = byte_quantity_to_str(min_memory_req_bytes)
            if shm_stats.total < min_memory_req_bytes:
                raise MemoryError(
                    f"A minimum of {min_memory_req_str} of total space in /dev/shm is required. Use --shm-size={min_memory_req_str} in `docker run` if using docker."
                )

            if shm_stats.free < min_memory_req_bytes:
                free_space_str = byte_quantity_to_str(shm_stats.free)
                raise MemoryError(
                    f"A minimum of {min_memory_req_str} of free space in /dev/shm is required, but only {free_space_str} is free. Please free up space or increase the /dev/shm size limit."
                )

        # Queue for thread-safe event recording
        self.event_queue: queue.Queue = queue.Queue()
        self.txn_buffer_size = txn_buffer_size

        # Writer thread management
        self.writer_thread: threading.Thread | None = None
        self._writer_started = False
        self.close_timeout_s = close_timeout_s

        self.notify_idle = notify_idle
        self.n_inflight_samples = 0
        self.should_check_idle = False

    @property
    def connection_name(self) -> Path:
        # To support accessing in multiple processes, we store the db in /dev/shm
        # Otherwise, using mode=memory&cache=shared only works within the same process
        return EventRecorder.db_path(self.session_id)

    @property
    def outputs_path(self) -> Path:
        return Path(self.connection_name).with_suffix(".outputs.jsonl")

    @staticmethod
    def db_path(session_id: str) -> Path:
        """Helper method to figure out the path of a session's database without creating an EventRecorder instance.

        Args:
            session_id: The session id.

        Returns:
            The path to the session's database.
        """
        return Path(f"/dev/shm/mlperf_testsession_{session_id}.db")

    def _writer_loop(self):
        """Writer thread loop that processes events from the queue and commits them to the database.

        This method runs in a dedicated thread and owns the SQLite connection and cursor.
        It processes events from the queue, buffering them until the buffer is full or a force commit is requested.
        """
        logging.debug(f"Writer thread started for {self.connection_name}")

        with (
            sqlite3_cursor(self.connection_name) as (cur, conn),
            self.outputs_path.open("w") as outputs,
        ):
            # Initialize the database table
            cur.execute(EventRow.to_table_query())
            conn.commit()

            event_buffer = []
            output_buffer = []

            insert_query = EventRow.insert_query()

            def commit_buffer():
                """Helper to commit and clear the event buffer."""
                if event_buffer:
                    cur.executemany(insert_query, event_buffer)
                    conn.commit()
                    event_buffer.clear()

                    # Write outputs to JSONL
                    if output_buffer:
                        for output in output_buffer:
                            outputs.write(orjson.dumps(output).decode("utf-8") + "\n")
                        output_buffer.clear()
                        outputs.flush()

            while True:
                try:
                    # Get item from queue, blocking until available
                    item = self.event_queue.get(timeout=1.0)
                except queue.Empty:
                    # Timeout - continue loop to check for stop condition
                    continue

                # Check for sentinel values
                should_commit = False
                if item is EventRecorder._STOP_SENTINEL:
                    # Commit any remaining events before stopping
                    if event_buffer:
                        logging.debug(
                            f"Writer thread stopping - committing final {len(event_buffer)} transactions"
                        )
                    should_commit = True
                elif item is EventRecorder._FORCE_COMMIT_SENTINEL:
                    # Force commit current buffer
                    if event_buffer:
                        logging.debug(
                            f"Force committing {len(event_buffer)} transactions"
                        )
                    should_commit = True
                else:
                    # Regular event - add to buffer
                    event_buffer.append(item[:-1])
                    if item[-1] is not None:
                        if item[1] == SampleEvent.FIRST_CHUNK.value:
                            # In post-processing, we use this to validate that the first chunk is the response output is the same as the data in the FIRST_CHUNK_RECEIVED event
                            output_buffer.append(
                                {"s_uuid": item[0], "first_chunk": item[-1]}
                            )
                        elif item[1] == SampleEvent.COMPLETE.value:
                            output_data = item[-1]
                            if not isinstance(output_data, list | tuple | str):
                                raise TypeError(
                                    f"QueryResult.response_output should be a list or tuple or str, but got {type(output_data)}"
                                )
                            output_buffer.append(
                                {
                                    "s_uuid": item[0],
                                    "output": output_data,
                                }
                            )
                        elif item[1] == SessionEvent.ERROR.value:
                            output_buffer.append(
                                {
                                    "s_uuid": item[0],
                                    "error_type": item[1],
                                    "error_message": item[-1],
                                }
                            )
                    should_commit = len(event_buffer) >= self.txn_buffer_size

                # Commit if buffer is full
                if should_commit:
                    logging.debug(
                        f"Committing {len(event_buffer)} transactions (max buffer size: {self.txn_buffer_size})"
                    )
                    commit_buffer()
                self.event_queue.task_done()

                if (
                    self.should_check_idle
                    and self.notify_idle is not None
                    and self.n_inflight_samples == 0
                    and self.event_queue.empty()
                ):
                    self.notify_idle.set()

                if item is EventRecorder._STOP_SENTINEL:
                    break
        logging.debug(f"Writer thread stopped for {self.connection_name}")

    def _start_writer_thread(self):
        """Starts the writer thread if not already started."""
        if EventRecorder.LIVE is not None:
            raise EventRecorderSingletonViolation(
                f"EventRecorder {EventRecorder.LIVE.session_id} is already active, cannot open {self.session_id}"
            )
        EventRecorder.LIVE = self

        if self._writer_started:
            logging.debug("Writer thread already started")
            return

        logging.debug(f"Starting writer thread for {self.connection_name}")
        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"EventRecorder-Writer-{self.session_id}",
            daemon=False,
        )
        self.writer_thread.start()
        self._writer_started = True

    def wait_for_writes(self, force_commit: bool = True):
        """Blocks until all queued events are processed.

        Args:
            force_commit: Whether to force commit the current buffer immediately. (Default: True)
        """
        if not self._writer_started:
            return

        if force_commit:
            self.event_queue.put(self._FORCE_COMMIT_SENTINEL)
        self.event_queue.join()

    @profile
    @classmethod
    def record_event(
        cls,
        ev_type: Event,
        timestamp_ns: int,
        sample_uuid: str = "",
        force_commit: bool = False,
        assert_active: bool = True,
        output: str | None = None,
    ) -> bool:
        """Records an event by pushing it to the queue for the writer thread to process.

        This method is thread-safe and can be called from multiple threads simultaneously.
        The actual database write happens asynchronously in the writer thread.

        Args:
            ev_type (Event): The type of event to record.
            timestamp_ns (int): The timestamp in nanoseconds of the event.
            sample_uuid (str): The sample uuid of the event.
            force_commit (bool): Whether to force commit the current buffer immediately.
            assert_active (bool): Whether to raise an exception if no EventRecorder is active.
                                  If False, this method will return False. (Default: True)
            output (str): The output to record associated with the event. Should only be used
                          for SampleEvent.COMPLETE events. (Default: None)
        Returns:
            bool: True if the event was recorded, False otherwise. If assert_active is True,
            this method will always return True or raise an exception.
        """
        if EventRecorder.LIVE is None:
            if assert_active:
                raise EventRecorderSingletonViolation(
                    "No EventRecorder is active, cannot record event"
                )
            return False

        rec_inst = EventRecorder.LIVE

        if not rec_inst._writer_started:
            raise RuntimeError(
                "Writer thread not started - Users should use `with EventRecorder(...)` to ensure the writer thread is started"
            )

        # Update inflight sample tracking
        if ev_type == SessionEvent.LOADGEN_ISSUE_CALLED:
            rec_inst.n_inflight_samples += 1
        elif ev_type == SampleEvent.COMPLETE:
            rec_inst.n_inflight_samples -= 1

        if rec_inst.n_inflight_samples < 0:
            raise RuntimeError(
                f"Number of inflight samples is negative: {rec_inst.n_inflight_samples}"
            )

        # Push event to queue for writer thread to process
        rec_inst.event_queue.put((sample_uuid, ev_type.value, timestamp_ns, output))

        # If force commit requested, send sentinel
        if force_commit:
            rec_inst.event_queue.put(EventRecorder._FORCE_COMMIT_SENTINEL)
        return True

    def close(self):
        """Closes the EventRecorder and stops the writer thread.

        This method signals the writer thread to stop, waits for it to finish processing
        all queued events, and then joins the thread.
        """
        if EventRecorder.LIVE is not self:
            raise EventRecorderSingletonViolation(
                f"EventRecorder {self.session_id} is not active, cannot close"
            )
        EventRecorder.LIVE = None

        if not self._writer_started:
            logging.debug("Writer thread was never started, nothing to close")
            return

        logging.debug("Stopping writer thread...")
        # Send stop sentinel to writer thread
        self.event_queue.put(self._STOP_SENTINEL)

        # Wait for the writer thread to finish
        if self.writer_thread is not None:
            self.writer_thread.join(timeout=self.close_timeout_s)
            if self.writer_thread.is_alive():
                n_pending = self.event_queue.qsize()
                raise RuntimeError(
                    f"Writer thread did not stop within timeout for {self.connection_name}. {n_pending} events pending."
                )
            else:
                logging.debug(
                    f"Writer thread stopped successfully for {self.connection_name}"
                )
        self._writer_started = False
        self.writer_thread = None

    def __enter__(self):
        """Context manager entry - starts the writer thread."""
        if not self._writer_started:
            self._start_writer_thread()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit - stops the writer thread."""
        self.close()


def record_exception(
    exc_value: Exception | str,
    sample_uuid: str | None = None,
):
    """Records an exception as an event to the current event recorder.

    This will force commit the existing event buffer immediately to ensure the error is surfaced
    as soon as possible for any monitoring.

    Args:
        exc_value: The exception to record, or a string error message.
        sample_uuid: The sample uuid to record the error for.
    """
    if EventRecorder.LIVE is None:
        return
    EventRecorder.record_event(
        SessionEvent.ERROR,
        time.monotonic_ns(),
        sample_uuid=sample_uuid,
        output=str(exc_value),
        force_commit=True,
    )
