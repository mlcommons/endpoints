from __future__ import annotations

import atexit
import contextlib
import dataclasses
import importlib
import logging
import math
import multiprocessing
import queue
import shutil
import sqlite3
import threading
import uuid
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

from ..load_generator.events import Event, SampleEvent, SessionEvent
from ..profiling import profile
from ..utils import byte_quantity_to_str

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def sqlite3_cursor(path: str):
    """Context manager for SQLite cursor that properly handles connection lifecycle.

    Args:
        path: Path to the SQLite database file.

    Yields:
        A SQLite cursor object.
    """
    conn = sqlite3.connect(path)
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


def register_cleanup(shm_path: str):
    if multiprocessing.parent_process() is not None:
        return
    atexit.register(partial(Path(shm_path).unlink, missing_ok=True))
    logger.debug(f"Registered at-exit cleanup for {shm_path}")


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
        min_memory_req_bytes: int = 1024 * 1024 * 1024,
        notify_idle: threading.Event | None = None,
    ):
        """Creates a new EventRecorder.

        Args:
            session_id: Optional session id to connect to an existing database. If not provided, a new database will be created.
            txn_buffer_size: The number of events to buffer before committing to the database. (Default: 1000)
            min_memory_req_bytes: The minimum amount of free space (in bytes) in /dev/shm required to create a new database. (Default: 1GB)
            notify_idle: Optional threading.Event. If provided, EventRecorder will set when the number of inflight samples is 0.
        """
        if session_id is None:
            session_id = uuid.uuid4().hex

        self.session_id = session_id

        if self.connection_name not in EventRecorder._created_session_dbs:
            register_cleanup(self.connection_name)
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

        self.notify_idle = notify_idle
        self.n_inflight_samples = 0
        self.should_check_idle = False

    @property
    def connection_name(self):
        # To support accessing in multiple processes, we store the db in /dev/shm
        # Otherwise, using mode=memory&cache=shared only works within the same process
        return EventRecorder.db_path(self.session_id)

    @staticmethod
    def db_path(session_id: str):
        """Helper method to figure out the path of a session's database without creating an EventRecorder instance.

        Args:
            session_id: The session id.

        Returns:
            The path to the session's database.
        """
        return f"/dev/shm/mlperf_testsession_{session_id}.db"

    def _writer_loop(self):
        """Writer thread loop that processes events from the queue and commits them to the database.

        This method runs in a dedicated thread and owns the SQLite connection and cursor.
        It processes events from the queue, buffering them until the buffer is full or a force commit is requested.
        """
        logging.debug(f"Writer thread started for {self.connection_name}")

        with sqlite3_cursor(self.connection_name) as (cur, conn):
            # Initialize the database table
            cur.execute(EventRow.to_table_query())
            conn.commit()

            event_buffer = []

            insert_query = EventRow.insert_query()

            def commit_buffer():
                """Helper to commit and clear the event buffer."""
                if event_buffer:
                    cur.executemany(insert_query, event_buffer)
                    conn.commit()
                    event_buffer.clear()

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
                    event_buffer.append(item)
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
        rec_inst.event_queue.put((sample_uuid, ev_type.value, timestamp_ns))

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
            self.writer_thread.join(timeout=10.0)
            if self.writer_thread.is_alive():
                raise RuntimeError(
                    f"Writer thread did not stop within timeout for {self.connection_name}"
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


@dataclasses.dataclass
class MetricRow:
    sample_uuid: str
    metric_type: str
    metric_value: float


@dataclasses.dataclass(frozen=True)
class RollupQueryTable:
    """Represents a table that is the result of a roll-up query.
    This class lazily converts tuples to MetricRow objects on-access to reduce unnecessary overhead.

    The columns are assumed to be (sample_uuid, metric_value). If a roll-up query returns different columns,
    define a subclass and override the __getitem__ method.
    """

    metric_type: str
    from_query: str
    rows: list[tuple[Any, ...]]

    def __getitem__(self, index: int) -> MetricRow:
        if index >= len(self.rows):
            raise IndexError(f"Index {index} out of range for {self.metric_type}")
        return MetricRow(self.rows[index][0], self.metric_type, self.rows[index][1])

    def __len__(self) -> int:
        return len(self.rows)

    def to_histogram(self, n_buckets: int = 20) -> tuple[list[int], list[int]]:
        """Returns a histogram of the metrics values.

        The buckets are uniformly sized, distributed between the min and max values.
        Bucket lower and upper bounds are integers, and the upper bound is exclusive.
        When determining the lower bound, we round down to the nearest integer.

        Args:
            n_buckets: The number of buckets to create.

        Returns:
            A tuple of lists, the first list is the buckets, the second list is the counts.
        """
        min_value = min(self.rows, key=lambda x: x[1])[1]
        max_value = max(self.rows, key=lambda x: x[1])[1]
        diff = max_value - min_value
        step = diff / n_buckets
        buckets = []
        counts = []
        for i in range(n_buckets):
            lower = int(min_value + i * step)
            buckets.append(lower)
            counts.append(0)

        # 2 options here:
        # O(N * M) where N is the number of rows and M is the number of buckets
        # O(N log N) where N is the number of rows
        if math.log(len(self.rows), 2) < n_buckets:
            for row in self.rows:
                for i in range(n_buckets):
                    if i == n_buckets - 1:
                        counts[i] += 1
                        break
                    elif row[1] < buckets[i + 1]:
                        counts[i] += 1
                        break
        else:
            bucket_idx = 0
            for row in sorted(self.rows, key=lambda x: x[1]):
                while (
                    bucket_idx < (n_buckets - 1) and row[1] >= buckets[bucket_idx + 1]
                ):
                    bucket_idx += 1
                counts[bucket_idx] += 1
        return buckets, counts


class MetricsReporter:
    """Derives metrics from events via rollup queries. This is a *read only* client."""

    def __init__(
        self,
        connection_name: str,
        client_type: str = "duckdb",
        intermediate_chunks_logged: bool = False,
    ):
        """
        Creates a new MetricsReporter.

        Args:
            connection_name: The path to the database to connect to.
            client_type: The client type to use to connect to the database. Choices: ["duckdb", "sqlite"] (Default: "duckdb")
            intermediate_chunks_logged: Whether to assume that there are intermediate chunks logged per sample, or if only the first and last chunks are logged.
                                        This is used to select faster queries to calculate TPOT (Default: False)
        """
        self.connection_name = connection_name
        self.client_type = client_type
        self.intermediate_chunks_logged = intermediate_chunks_logged
        self.is_closed = True

    def init_connection(self):
        if not self.is_closed:
            logging.debug(f"Connection already initialized at {self.connection_name}")
            return

        if self.client_type == "duckdb":
            logging.debug(f"Initializing duckdb connection at {self.connection_name}")
            if importlib.util.find_spec("duckdb") is None:
                raise ImportError("duckdb is not installed")
            duckdb = importlib.import_module("duckdb")
            # Install sqlite extension
            self.conn = duckdb.connect()

            logging.debug("Installing sqlite extension for duckdb")
            self.conn.install_extension("sqlite")
            self.conn.load_extension("sqlite")

            logging.debug(
                f"Attaching {self.connection_name} to duckdb in read-only mode"
            )
            self.conn.execute(
                f"ATTACH '{self.connection_name}' AS sqlite_db (TYPE sqlite, READ_ONLY)"
            )
            self.conn.execute("USE sqlite_db")

            self.cur_ = (
                self.conn
            )  # duckdb calls execute() on connection, there is no cursor object
        elif self.client_type == "sqlite":
            logging.debug(
                f"Initializing read-only sqlite connection at {self.connection_name}"
            )
            self.conn = sqlite3.connect(
                f"file:{self.connection_name}?mode=ro", uri=True
            )
            self.cur_ = self.conn.cursor()
        else:
            raise ValueError(f"Invalid client type: {self.client_type}")
        self.is_closed = False

    @profile
    def derive_metric(self, query: str, metric_type: str) -> RollupQueryTable:
        res = self.cur_.execute(query)
        logging.debug(f"Roll-up for {metric_type}. Running query: {query}")
        return RollupQueryTable(metric_type, query, res.fetchall())

    def derive_TTFT(self) -> RollupQueryTable:
        return self.derive_metric(
            f"""
            SELECT
                sample_uuid,
                MAX(CASE WHEN event_type = '{SampleEvent.FIRST_CHUNK.value}' THEN timestamp_ns END) -
                MAX(CASE WHEN event_type = '{SessionEvent.LOADGEN_ISSUE_CALLED.value}' THEN timestamp_ns END) AS ttft
            FROM events
            WHERE event_type IN ('{SessionEvent.LOADGEN_ISSUE_CALLED.value}', '{SampleEvent.FIRST_CHUNK.value}')
            GROUP BY sample_uuid
            HAVING COUNT(DISTINCT event_type) = 2
            """,
            "ttft",
        )

    def derive_TPOT(self) -> RollupQueryTable:
        if self.intermediate_chunks_logged:
            query = f"""
            WITH chunk_events AS (
                SELECT
                    sample_uuid,
                    event_type,
                    timestamp_ns,
                    LAG(timestamp_ns) OVER (
                        PARTITION BY sample_uuid
                        ORDER BY timestamp_ns
                    ) AS prev_timestamp
                FROM events
                WHERE event_type IN ('{SampleEvent.FIRST_CHUNK.value}', '{SampleEvent.NON_FIRST_CHUNK.value}')
            )
            SELECT
                sample_uuid,
                (timestamp_ns - prev_timestamp) AS tpot
            FROM chunk_events
            WHERE prev_timestamp IS NOT NULL
            ORDER BY sample_uuid, timestamp_ns
            """
        else:
            query = f"""
            SELECT
                sample_uuid,
                MAX(CASE WHEN event_type = '{SampleEvent.NON_FIRST_CHUNK.value}' THEN timestamp_ns END) -
                MAX(CASE WHEN event_type = '{SampleEvent.FIRST_CHUNK.value}' THEN timestamp_ns END) AS tpot
            FROM events
            WHERE event_type IN ('{SampleEvent.FIRST_CHUNK.value}', '{SampleEvent.NON_FIRST_CHUNK.value}')
            GROUP BY sample_uuid
            HAVING COUNT(DISTINCT event_type) = 2
            """
        return self.derive_metric(query, "tpot")

    @profile
    def get_sample_statuses(self) -> dict[int, str]:
        """Returns a dictionary with the following keys:
        - "total_sent" (int): The total number of samples sent
        - "completed" (int): The number of samples completed
        - "in_flight" (int): The number of samples in flight
        """
        statuses = self.cur_.execute(f"""
        SELECT
            COUNT(DISTINCT CASE WHEN event_type = '{SessionEvent.LOADGEN_ISSUE_CALLED.value}' THEN sample_uuid END) AS request_sent_count,
            COUNT(DISTINCT CASE WHEN event_type = '{SampleEvent.COMPLETE.value}' THEN sample_uuid END) AS complete_count
        FROM events
        """).fetchone()

        return {
            "total_sent": statuses[0],
            "completed": statuses[1],
            "in_flight": statuses[0] - statuses[1],
        }

    def close(self):
        if self.is_closed:
            logging.debug("Connection is already closed, skipping close")
            return
        self.is_closed = True

        if self.cur_ is not self.conn:
            self.cur_.close()
        self.conn.close()

    def __enter__(self):
        if self.is_closed:
            self.init_connection()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
