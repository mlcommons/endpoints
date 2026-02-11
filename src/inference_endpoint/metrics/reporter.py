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

from __future__ import annotations

import csv
import dataclasses
import functools
import importlib
import logging
import numbers
import os
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Iterable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import orjson

from ..load_generator.events import SampleEvent, SessionEvent
from ..profiling import profile
from ..utils import monotime_to_datetime

if TYPE_CHECKING:
    from transformers import Tokenizer


class TPOTReportingMode(str, Enum):
    """TPOT (Time Per Output Token) reporting mode.

    - REQUEST_WEIGHTED: Each request contributes one entry to TPOT calculation (default)
    - TOKEN_WEIGHTED: Each token contributes to TPOT calculation (weighted by token count)
    """

    REQUEST_WEIGHTED = "request_weighted"
    TOKEN_WEIGHTED = "token_weighted"


class SampleUUIDNotFoundError(Exception):
    def __init__(self, uuid: str, datasource: str):
        super().__init__(f"Sample UUID {uuid} not found in {datasource}")


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
    """A string describing the metric being computed and rolled up."""

    from_query: str
    """If provided, the query that was used to generate the table."""

    rows: list[tuple[Any, ...]]
    """The rows of the table, each a tuple of values."""

    repeats: list[int] | None = None
    """If provided, this means the rows are condensed by consecutive duplicates. `repeats`
    represents the number of times each row should be repeated."""

    _sorted_vals: np.ndarray = dataclasses.field(init=False)
    _by_uuid: dict[str, list[int]] = dataclasses.field(init=False)

    def __post_init__(self):
        if self.repeats is not None:
            if len(self.repeats) != len(self.rows):
                raise IndexError(
                    f"Length of repeats {len(self.repeats)} does not match length of rows {len(self.rows)}"
                )
            if not isinstance(self.repeats, np.ndarray):
                object.__setattr__(
                    self, "repeats", np.array(self.repeats, dtype=np.int64)
                )
            else:
                object.__setattr__(self, "repeats", self.repeats.astype(np.int64))

        # Metrics are always differences between integer nanosecond timestamps
        # Some might be 32-bit and 64 bit, so we force np.int64 here
        if self.repeats is None:
            sorted_vals = np.array([row[1] for row in self.rows], dtype=np.int64)
            sorted_vals.sort()
        else:
            arr = np.array(
                [(self.rows[i][1], self.repeats[i]) for i in range(len(self.rows))],
                dtype=np.int64,
            )
            sorted_vals = arr[arr[:, 0].argsort()]
        object.__setattr__(self, "_sorted_vals", sorted_vals)

        # Pre-compute a dictionary to map sample UUIDs to values
        by_uuid = defaultdict(list)
        for i, (s_uuid, value) in enumerate(self.rows):
            if self.repeats is not None:
                value = (value, self.repeats[i])
            by_uuid[s_uuid].append(value)
        object.__setattr__(self, "_by_uuid", by_uuid)

    def __getitem__(self, index: int) -> MetricRow:
        """Returns the MetricRow at the given index / row number in the table.

        Returns:
            MetricRow: The MetricRow at the given index / row number in the table.
        """
        length = len(self)
        if index >= length:
            raise IndexError(f"Index {index} out of range for {self.metric_type}")

        while index < 0:
            index += length

        if self.repeats is None:
            return MetricRow(self.rows[index][0], self.metric_type, self.rows[index][1])
        else:
            passed = 0
            for i, repeat in enumerate(self.repeats):
                next_row_start = passed + repeat
                if index < next_row_start:
                    return MetricRow(self.rows[i][0], self.metric_type, self.rows[i][1])
                else:
                    passed = next_row_start
            # This should never happen if our index validation is correct
            raise IndexError(f"Index {index} out of range for {self.metric_type}")

    def __len__(self) -> int:
        if self.repeats is None:
            return len(self.rows)
        else:
            return int(sum(self.repeats))

    def filter_uuid(self, uuid: str, only_first: bool = False) -> Any:
        """Returns the values for the given sample UUID.

        Args:
            uuid: The sample UUID to filter by.
            only_first: Whether to only return the first value for the sample UUID.
        Returns:
            The values for the given sample UUID as a tuple. If only_first is True,
            returns the first value directly, unless no values are found, in which
            case None is returned.
        """
        values = self._by_uuid[uuid]

        # Expand values if there are counts
        if self.repeats is not None:
            if only_first:  # If we only want the first value, we don't need to expand
                return values[0][0]

            expanded_values = []
            for value, count in values:
                expanded_values.extend([value] * count)
            values = expanded_values

        if only_first:
            if len(values) == 0:
                return None
            return values[0]
        return tuple(values)

    def __contains__(self, uuid: str) -> bool:
        """Returns True if the given sample UUID is in the table."""
        return uuid in self._by_uuid

    def summarize(
        self,
        percentiles: Iterable[float] = (99.9, 99, 97, 95, 90, 80, 75, 50, 25, 10, 5, 1),
    ) -> dict[str, float]:
        if len(self._sorted_vals) == 0:
            return {
                "total": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "avg": 0.0,
                "std_dev": 0.0,
                "percentiles": {str(p): 0.0 for p in percentiles},
                "histogram": {
                    "buckets": [],
                    "counts": [],
                },
            }
        else:
            # Note values are sorted, we can avoid using np.max and np.min
            # Need to convert to default Python types since orjson doesn't support numpy dtypes
            if self.repeats is None:
                values = self._sorted_vals
                counts = np.ones(self._sorted_vals.shape, dtype=self._sorted_vals.dtype)
            else:
                values = self._sorted_vals[:, 0]
                counts = self._sorted_vals[:, 1]

            total = int((values * counts).sum())
            minimum = int(values[0])
            maximum = int(values[-1])
            median = self.percentile(50)
            avg = float(np.average(values, weights=counts))
            if self.repeats is None:
                std_dev = float(np.std(values))
            else:
                deviations_squared = (values - avg) ** 2
                std_dev = float(
                    np.sqrt(np.sum(deviations_squared * counts) / counts.sum())
                )
            summary = {
                "total": total,
                "min": minimum,
                "max": maximum,
                "median": median,
                "avg": avg,
                "std_dev": std_dev,
                "percentiles": {
                    str(p): v for p, v in self.percentile(percentiles).items()
                },
            }

            # Add histogram
            buckets, counts = self.to_histogram(n_buckets=10)
            summary["histogram"] = {
                "buckets": buckets,
                "counts": counts,
            }
            return summary

    def to_histogram(
        self,
        n_buckets: int = 20,
        convert_to_native_types: bool = True,
    ) -> tuple[list[tuple[float, float]], list[int]] | tuple[np.ndarray, np.ndarray]:
        """Returns a histogram of the metrics values.

        The returned buckets are uniformly sized, distributed between the min and max values, with an
        inclusive lower bound and exclusive upper bound.

        Args:
            n_buckets: The number of buckets to create. Alternatively, any valid argument for `bins` in `np.histogram`
                       can be provided.
            convert_to_native_types: Whether to convert the buckets and counts to native Python types.
                                     If False, returns numpy arrays. (Default: True)

        Returns:
            A tuple of lists, the first list is the buckets, the second list is the counts.
            If convert_to_native_types is False, returns a numpy arrays instead.
        """
        if self.repeats is None:
            values = self._sorted_vals
            repeats = None
        else:
            values = self._sorted_vals[:, 0]
            repeats = self._sorted_vals[:, 1]

        # Derive bins from values
        bounds = np.histogram_bin_edges(values, bins=n_buckets)

        counts, _ = np.histogram(values, bins=bounds, weights=repeats)
        if not convert_to_native_types:
            buckets = np.zeros((len(bounds) - 1, 2), dtype=bounds.dtype)
            buckets[:, 0] = bounds[:-1]
            buckets[:, 1] = bounds[1:]
            return buckets, counts

        buckets = [
            (float(bounds[i]), float(bounds[i + 1])) for i in range(len(bounds) - 1)
        ]
        return buckets, counts.tolist()

    def percentile(
        self,
        percentile: float | list[float] | tuple[float, ...],
        interpolate_strategy: str = "linear",
    ) -> float | dict[float, float]:
        """Compute the percentile(s) of the metric values.
        The value returned is the value of the metric at the index marking the percentile,
        not an interpolated value.

        Args:
            percentile: The percentile(s) to compute. If a single value, returns a single value. If a list of values, returns a dictionary of values.
            interpolate_strategy: The percentile interpolation string to use for numpy.percentile. See
                                  https://numpy.org/doc/2.2/reference/generated/numpy.percentile.html to see what interpolation methods are available.
                                  (Default: "linear")

        Returns:
            A single value if a single percentile is provided, a dictionary of values if a list of percentiles is provided.
        """
        if not isinstance(percentile, (numbers.Number | Iterable)):
            raise TypeError(
                f"percentile must be a number or an iterable of numbers, got {type(percentile)}"
            )

        if isinstance(percentile, Iterable):
            if len(percentile) == 0:
                return {}

            if not isinstance(percentile[0], numbers.Number):
                raise TypeError(
                    f"percentile must be an iterable of numbers, got Iterable[{type(percentile[0])}]"
                )

        if self.repeats is None:
            perc_values = np.percentile(
                self._sorted_vals,
                percentile,
                overwrite_input=False,
                method=interpolate_strategy,
            )
        else:
            values = self._sorted_vals[:, 0]
            counts = self._sorted_vals[:, 1]
            perc_values = np.percentile(
                values,
                percentile,
                weights=counts,
                overwrite_input=False,
                method="inverted_cdf",
            )

        if isinstance(percentile, numbers.Number):
            return float(perc_values)
        else:
            return {p: float(v) for p, v in zip(percentile, perc_values, strict=False)}


@dataclasses.dataclass(frozen=True)
class Report:
    """Represents a summarized report of metrics"""

    version: str
    git_sha: str | None
    test_started_at: int
    n_samples_issued: int
    n_samples_completed: int
    duration_ns: int

    # For the following metrics, the key is a rollup statistic (i.e. mean, median, etc.)
    ttft: dict[str, float]
    tpot: dict[str, float]
    latency: dict[str, float]
    output_sequence_lengths: dict[str, int]
    tpot_reporting_mode: TPOTReportingMode = TPOTReportingMode.REQUEST_WEIGHTED

    @functools.cached_property
    def qps(self) -> float | None:
        """Calculates the queries (or samples) per second (QPS) based on actual throughput.

        This is the actual throughput: total completed samples divided by test duration.
        If duration is 0, which shouldn't happen in practice, returns None.

        Returns:
            The QPS or None if duration is 0.
        """
        if not self.duration_ns:
            return None
        return float(self.n_samples_completed / (self.duration_ns / 1e9))

    @functools.cached_property
    def tps(self) -> float | None:
        """Calculates the tokens per second based on the output sequence lengths and duration.

        Returns:
            The tokens per second or None if duration is 0.
        """
        if not self.duration_ns:
            return None
        if not self.output_sequence_lengths:
            return None
        return float(self.output_sequence_lengths["total"] / (self.duration_ns / 1e9))

    def to_json(self, save_to: os.PathLike | None = None) -> str:
        """Returns a JSON string representation of the report.

        Args:
            save_to: If provided, saves the serialized JSON to the given path.

        Returns:
            The JSON string representation of the report.
        """
        d = dataclasses.asdict(self)
        d["qps"] = self.qps
        d["tps"] = self.tps
        json_str = orjson.dumps(
            d, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
        ).decode("utf-8")
        if save_to is not None:
            with Path(save_to).open("w") as f:
                f.write(json_str)
        return json_str

    @staticmethod
    def _display_metric(
        metric_dict,
        fn: Callable[[str], None] = print,
        unit: str = "",
        max_bar_length: int = 30,
        scale_factor: float = 1.0,
        newline: str = "",
    ) -> None:
        """Displays a metric dictionary in a human-readable format.

        Args:
            metric_dict: The metric dictionary to display.
            fn: The function to call to print a string, such as logging.info, file.write, etc. (Default: `print`)
            unit: The string representing the unit of the metric
            max_bar_length: The maximum length of the bar to display for the histogram
            scale_factor: The factor to scale metric values by. (Default: 1.0)
            newline: The newline character to append to each line. Set to "\\n" for file.write. (Default: "")
        """
        for name, key in [
            ("Min", "min"),
            ("Max", "max"),
            ("Median", "median"),
            ("Avg.", "avg"),
            ("Std Dev.", "std_dev"),
        ]:
            fn(f"  {name}: {metric_dict[key] * scale_factor:.2f} {unit}{newline}")
        fn(f"\n  Histogram:{newline}")

        # Display histogram
        buckets = metric_dict["histogram"]["buckets"]
        counts = metric_dict["histogram"]["counts"]

        if len(buckets) > 0:
            bucket_strs = []
            for lower, upper in buckets:
                if upper is None:
                    bucket_strs.append(f"  {lower * scale_factor:.2f}+")
                else:
                    bucket_strs.append(
                        f"  [{lower * scale_factor:.2f}, {upper * scale_factor:.2f})"
                    )

            normalize_factor = max_bar_length / max(counts)
            max_bucket_str_len = max(len(s) for s in bucket_strs)

            for bucket_str, count in zip(bucket_strs, counts, strict=False):
                bar_length = int(count * normalize_factor)
                fn(
                    f"  {bucket_str:>{max_bucket_str_len}} |{'#' * bar_length} {count}{newline}"
                )

            fn(f"\n  Percentiles:{newline}")
            max_percentile_str_len = max(
                len(str(p)) for p in metric_dict["percentiles"].keys()
            )
            for percentile, value in metric_dict["percentiles"].items():
                fn(
                    f"  {percentile:>{max_percentile_str_len}}: {value * scale_factor:.2f} {unit}{newline}"
                )

    def display(
        self,
        fn: Callable[[str], None] = print,
        summary_only: bool = False,
        newline: str = "",
    ) -> None:
        """Displays the report in a human-readable format.

        Args:
            fn: The function to call to print a string, such as logging.info, file.write, etc. (Default: `print`)
            newline: The newline character to append to each line. Set to "\\n" for file.write. (Default: "")
        """

        fn(f"----------------- Summary -----------------{newline}")
        fn(f"Version: {self.version}{newline}")
        if self.git_sha:
            fn(f"Git SHA: {self.git_sha}{newline}")
        # Approximate absolute time of the test started at using monotime_to_datetime from utils.py
        test_started_at_approx = monotime_to_datetime(self.test_started_at)
        fn(
            f"Test started at: {self.test_started_at}, approx: ({test_started_at_approx.strftime('%Y-%m-%d %H:%M:%S')}){newline}"
        )
        fn(f"Total samples issued: {self.n_samples_issued}{newline}")
        fn(f"Total samples completed: {self.n_samples_completed}{newline}")
        if self.duration_ns is not None:
            fn(f"Duration: {self.duration_ns / 1e9:.2f} seconds{newline}")
        else:
            fn(f"Duration: N/A (no performance samples were issued){newline}")

        if self.qps is not None:
            fn(f"QPS: {self.qps:.2f}{newline}")
        else:
            fn(f"QPS: N/A (no performance samples were issued){newline}")

        if self.tps is not None:
            fn(f"TPS: {self.tps:.2f}{newline}")

        if summary_only:
            fn(f"----------------- End of Summary -----------------{newline}")
            return

        fn(f"\n\n------------------- Latency Breakdowns -------------------{newline}")
        if len(self.latency) > 0 and self.ttft == 0:
            fn(
                f"WARNING: Non-streaming-based Issuer used. TTFT metrics cannot be calculated{newline}"
            )

        for section_name, metric_dict, unit, scale_factor in [
            ("TTFT", self.ttft, "ms", 1e-6),
            (f"TPOT ({self.tpot_reporting_mode.value})", self.tpot, "ms", 1e-6),
            ("Latency", self.latency, "ms", 1e-6),
            ("Output sequence lengths", self.output_sequence_lengths, "tokens", 1.0),
        ]:
            if metric_dict is None or len(metric_dict) == 0:
                continue
            fn(f"{section_name}:{newline}")
            Report._display_metric(
                metric_dict,
                fn=fn,
                unit=unit,
                scale_factor=scale_factor,
                newline=newline,
            )
            fn(f"\n{newline}")


def _output_sequence_to_str(output_sequence: str | list[str]) -> str | None:
    if isinstance(output_sequence, list):
        return "".join(output_sequence)
    elif isinstance(output_sequence, str):
        return output_sequence
    else:
        return None


def output_sequence_from_data(
    data_bytes: bytes,
    join_chunks: bool = True,
) -> tuple[str | list[str] | None, str | list[str] | None]:
    """Parse the data column from a COMPLETE event and extract output and reasoning sequences.

    The data column is expected to be a JSON-encoded byte string. The decoded value can be:
    - A string: treated as the output sequence directly
    - A dictionary with 'output' key (required) and optionally 'reasoning' key
      - Both 'output' and 'reasoning' can be either strings or lists of strings
      - If a list of strings, they will be joined together

    Args:
        data_bytes: The raw bytes from the database 'data' column
        join_chunks: Whether to join the chunks into a single string if the data values are lists of strings
                    (Default: True)
    Returns:
        A tuple of (output_sequence, reasoning_sequence), where each is a string (if join_chunks is True),
        list of strings (if join_chunks is False) or None.
        If the data cannot be decoded or is invalid, returns (None, None).
    """
    if data_bytes is None or len(data_bytes) == 0:
        return None, None

    try:
        decoded_data = orjson.loads(data_bytes)
    except (orjson.JSONDecodeError, TypeError):
        logging.warning("Failed to decode data bytes")
        return None, None

    output, reasoning = None, None
    if isinstance(decoded_data, str):
        # If decoded value is a string, it's the output sequence
        output = decoded_data
    elif isinstance(decoded_data, dict):
        # If decoded value is a dict, extract 'output' and optionally 'reasoning'
        if "output" not in decoded_data:
            logging.warning("Dictionary data missing required 'output' key")
            return None, None

        # Extract output - can be string or list of strings
        output = (
            _output_sequence_to_str(decoded_data["output"])
            if join_chunks
            else decoded_data["output"]
        )
        if output is None:
            logging.warning(f"Output field has unexpected type: {type(output)}")
            return None, None

        # Extract reasoning if present - can be string or list of strings
        if "reasoning" in decoded_data:
            reasoning = (
                _output_sequence_to_str(decoded_data["reasoning"])
                if join_chunks
                else decoded_data["reasoning"]
            )
    else:
        logging.warning(f"Decoded data has unexpected type: {type(decoded_data)}")
        return None, None
    return output, reasoning


class MetricsReporter:
    """Derives metrics from events via rollup queries. This is a *read only* client."""

    def __init__(
        self,
        connection_name: os.PathLike,
        client_type: str = "duckdb",
    ):
        """
        Creates a new MetricsReporter.

        Args:
            connection_name: The path to the database to connect to.
            client_type: The client type to use to connect to the database. Choices: ["duckdb", "sqlite"] (Default: "duckdb")
        """
        self.connection_name = Path(connection_name)
        self.client_type = client_type
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
            # duckdb doesn't inherit proxy variables from environment
            # Try setting proxy explicitly if environment variables are not enough
            proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY")
            if proxy:
                logging.debug(f"Setting http_proxy to {proxy} for duckdb")
                # Use parameterized query to safely set http_proxy
                self.conn.execute("SET http_proxy=?", [proxy])
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

    @functools.cached_property
    def stop_performance_tracking_timestamp_ns(self) -> float:
        """Returns the timestamp_ns of the STOP_PERFORMANCE_TRACKING event.

        This method is cached to prevent re-derivation. If the event is not found,
        returns positive infinity, since this indicates that the performance run is probably still
        running, or the test was killed before it could complete.

        Returns:
            float: The timestamp_ns of STOP_PERFORMANCE_TRACKING event, or float('inf') if not found.
        """
        result = self.cur_.execute(f"""
        SELECT timestamp_ns
        FROM events
        WHERE event_type = '{SessionEvent.STOP_PERFORMANCE_TRACKING.value}'
        LIMIT 1
        """).fetchone()

        if result is None:
            logging.warning(
                "No STOP_PERFORMANCE_TRACKING event found, performance run not yet complete"
            )
            return float("inf")
        return float(result[0])

    @profile
    def derive_metric(self, query: str, metric_type: str) -> RollupQueryTable:
        res = self.cur_.execute(query)
        logging.debug(f"Roll-up for {metric_type}. Running query: {query}")
        return RollupQueryTable(metric_type, query, res.fetchall())

    def derive_TTFT(self) -> RollupQueryTable:
        stop_ts = self.stop_performance_tracking_timestamp_ns

        # Build the HAVING clause conditionally to handle infinity
        if stop_ts != float("inf"):
            before_stop_ts_clause = f"""
            HAVING COUNT(DISTINCT event_type) = 2
                AND MAX(CASE WHEN event_type = '{SessionEvent.LOADGEN_ISSUE_CALLED.value}' THEN timestamp_ns END) < {stop_ts}
            """
        else:
            before_stop_ts_clause = """
            HAVING COUNT(DISTINCT event_type) = 2
            """

        return self.derive_metric(
            f"""
            SELECT
                sample_uuid,
                MAX(CASE WHEN event_type = '{SampleEvent.FIRST_CHUNK.value}' THEN timestamp_ns END) -
                MAX(CASE WHEN event_type = '{SessionEvent.LOADGEN_ISSUE_CALLED.value}' THEN timestamp_ns END) AS ttft
            FROM events
            WHERE event_type IN ('{SessionEvent.LOADGEN_ISSUE_CALLED.value}', '{SampleEvent.FIRST_CHUNK.value}')
            GROUP BY sample_uuid
            {before_stop_ts_clause}
            """,
            "ttft",
        )

    def dump_all_to_csv(self, csv_path: Path):
        logging.debug(f"Dumping to CSV at {csv_path}")
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            query = """
            SELECT
                sample_uuid,
                timestamp_ns,
                event_type
            FROM events
            """
            rows = self.cur_.execute(query).fetchall()
            writer.writerows(rows)
            logging.debug(f"Written rows {len(rows)} to {csv_path}")

    def derive_duration(self, check_malformed: bool = True) -> float | None:
        """Calculates the total test duration.

        If STOP_PERFORMANCE_TRACKING event exists:
            - This method will return T_(last_perf_sample) - T_(test_started) where:
                - T_(test_started) is the timestamp of the TEST_STARTED event and
                - T_(last_perf_sample) is the timestamp of the latest COMPLETE event present
                  whose sample_uuid has a corresponding LOADGEN_ISSUE_CALLED event before
                  the STOP_PERFORMANCE_TRACKING event.
            - If for some reason, no samples were issued before the STOP_PERFORMANCE_TRACKING event,
              such as in the case of running an accuracy-only test, then this method will return None.

        If STOP_PERFORMANCE_TRACKING does not exist:
            - This method will return the max(timestamp_ns) - T_(test_started) where:
                - T_(test_started) is the timestamp of the TEST_STARTED event and
                - max(timestamp_ns) is the largest timestamp_ns in the events database.
            - An error is raised if TEST_ENDED is present, but not the event associated with max(timestamp_ns)

        If `check_malformed` is False, no checks for the error-conditions above are performed. This is useful
        to in cases where the latency of this method matters, and we would like to avoid executing extra queries.
        In this case, the caller can periodically set check_malformed to True to perform verification in intervals.

        Args:
            check_malformed: Whether to check for malformed events. (Default: True)

        Raises:
            RuntimeError: If TEST_STARTED is not present or occurs more than once
            RuntimeError: If TEST_ENDED exists but is not the maximum timestamp_ns
            RuntimeError: If more than one TEST_ENDED event exists

        Returns:
            float: The duration in nanoseconds, None if no performance samples were issued.
        """
        # Validate TEST_STARTED exists exactly once
        test_started_result = self.cur_.execute(f"""
        SELECT COUNT(*) AS n_starts, MAX(timestamp_ns) AS start_ts
        FROM events
        WHERE event_type = '{SessionEvent.TEST_STARTED.value}'
        """).fetchone()

        n_test_started = test_started_result[0]
        test_started_ts = test_started_result[1]

        # Return None early if no TEST_STARTED event to avoid errors in duration calculations
        if test_started_ts is None or n_test_started == 0:
            if check_malformed:
                raise RuntimeError("TEST_STARTED event not found in database")
            return None

        if check_malformed and n_test_started > 1:
            raise RuntimeError(
                f"Multiple TEST_STARTED events found - {n_test_started} events"
            )

        # Check if STOP_PERFORMANCE_TRACKING event exists
        stop_ts = self.stop_performance_tracking_timestamp_ns

        if stop_ts != float("inf"):
            # Build list of sample_uuids with LOADGEN_ISSUE_CALLED before stop_ts
            # Then find the max timestamp_ns of any event from those sample_uuids
            max_perf_ts_result = self.cur_.execute(f"""
            SELECT MAX(timestamp_ns) AS max_perf_ts
            FROM events
            WHERE sample_uuid IN (
                SELECT DISTINCT sample_uuid
                FROM events
                WHERE event_type = '{SessionEvent.LOADGEN_ISSUE_CALLED.value}'
                AND timestamp_ns < {stop_ts}
            )
            AND event_type = '{SampleEvent.COMPLETE.value}'
            """).fetchone()

            max_perf_ts = max_perf_ts_result[0]
            if max_perf_ts is None:
                # No samples were issued before stop_ts
                return None

            return float(max_perf_ts - test_started_ts)
        else:
            # No STOP_PERFORMANCE_TRACKING, use max timestamp_ns in database
            # Get max timestamp in database
            max_ts_result = self.cur_.execute("""
            SELECT MAX(timestamp_ns) AS max_ts
            FROM events
            """).fetchone()
            max_ts = max_ts_result[0]

            if check_malformed:
                # Validate TEST_ENDED constraints
                test_ended_result = self.cur_.execute(f"""
                SELECT COUNT(*) AS n_ends, MAX(timestamp_ns) AS end_ts
                FROM events
                WHERE event_type = '{SessionEvent.TEST_ENDED.value}'
                """).fetchone()

                n_test_ended = test_ended_result[0]
                test_ended_ts = test_ended_result[1]

                if n_test_ended > 1:
                    raise RuntimeError(
                        f"Multiple TEST_ENDED events found - {n_test_ended} events"
                    )

                # If TEST_ENDED exists, it must be the maximum timestamp
                if n_test_ended == 1 and test_ended_ts != max_ts:
                    raise RuntimeError(
                        f"TEST_ENDED exists (timestamp_ns={test_ended_ts}) but is not the maximum timestamp in database (max={max_ts})"
                    )

            if max_ts is None:
                return None

            return float(max_ts - test_started_ts)

    def derive_sample_latency(self) -> RollupQueryTable:
        """Calculates the end-to-end latency for each sample from issue to completion.

        Returns:
            RollupQueryTable: A table containing per-sample latencies in nanoseconds.
        """
        stop_ts = self.stop_performance_tracking_timestamp_ns

        # HAVING clause is different if there is a STOP_PERFORMANCE_TRACKING event
        if stop_ts != float("inf"):
            before_stop_ts_clause = f"""
            HAVING COUNT(DISTINCT event_type) = 2
                AND MAX(CASE WHEN event_type = '{SessionEvent.LOADGEN_ISSUE_CALLED.value}' THEN timestamp_ns END) < {stop_ts}
            """
        else:
            before_stop_ts_clause = """
            HAVING COUNT(DISTINCT event_type) = 2
            """

        return self.derive_metric(
            f"""
            SELECT
                sample_uuid,
                MAX(CASE WHEN event_type = '{SampleEvent.COMPLETE.value}' THEN timestamp_ns END) -
                MAX(CASE WHEN event_type = '{SessionEvent.LOADGEN_ISSUE_CALLED.value}' THEN timestamp_ns END) AS latency
            FROM events
            WHERE event_type IN ('{SessionEvent.LOADGEN_ISSUE_CALLED.value}', '{SampleEvent.COMPLETE.value}')
            GROUP BY sample_uuid
            {before_stop_ts_clause}
            """,
            "sample_latency",
        )

    @profile
    def get_sample_statuses(self) -> dict[int, str]:
        """Returns a dictionary with the following keys:
        - "total_sent" (int): The total number of samples sent
        - "completed" (int): The number of samples completed
        - "in_flight" (int): The number of samples in flight
        """
        stop_ts = self.stop_performance_tracking_timestamp_ns

        # Build WHERE clause to filter samples issued before stop_ts
        where_clause = ""
        if stop_ts != float("inf"):
            where_clause = f"""
            WHERE sample_uuid IN (
                SELECT sample_uuid FROM events
                WHERE event_type = '{SessionEvent.LOADGEN_ISSUE_CALLED.value}'
                AND timestamp_ns < {stop_ts}
            )
            """

        statuses = self.cur_.execute(f"""
        SELECT
            COUNT(DISTINCT CASE WHEN event_type = '{SessionEvent.LOADGEN_ISSUE_CALLED.value}' THEN sample_uuid END) AS request_sent_count,
            COUNT(DISTINCT CASE WHEN event_type = '{SampleEvent.COMPLETE.value}' THEN sample_uuid END) AS complete_count
        FROM events
        {where_clause}
        """).fetchone()

        return {
            "total_sent": statuses[0],
            "completed": statuses[1],
            "in_flight": statuses[0] - statuses[1],
        }

    def get_error_count(self) -> int:
        return self.cur_.execute(f"""
        SELECT
            COUNT(*) AS error_count
        FROM events
        WHERE event_type = '{SessionEvent.ERROR.value}'
        """).fetchone()[0]

    def get_sample_outputs(
        self, performance_only: bool = True
    ) -> list[tuple[str, bytes]]:
        """Query for COMPLETE events with their data column.

        Args:
            performance_only: Whether to only include samples that are in the performance window. (Default: True)

        Returns:
            A list of tuples containing (sample_uuid, data_bytes) for each COMPLETE event.
            Returns an empty list if no COMPLETE events are found.
        """
        stop_ts = self.stop_performance_tracking_timestamp_ns

        # Build WHERE clause to filter samples issued before STOP_PERFORMANCE_TRACKING
        if performance_only and stop_ts != float("inf"):
            before_stop_ts_clause = f"""
            AND sample_uuid IN (
                SELECT sample_uuid FROM events
                WHERE event_type = '{SessionEvent.LOADGEN_ISSUE_CALLED.value}'
                AND timestamp_ns < {stop_ts}
            )
            """
        else:
            before_stop_ts_clause = ""

        # Query for COMPLETE events with their data column
        query_result = self.cur_.execute(f"""
            SELECT sample_uuid, data
            FROM events
            WHERE event_type = '{SampleEvent.COMPLETE.value}'
            {before_stop_ts_clause}
        """).fetchall()

        return query_result

    @profile
    def get_output_sequence_lengths(
        self, tokenizer: Tokenizer
    ) -> RollupQueryTable | None:
        """Returns a RollupQueryTable representing per-sample output sequence lengths based on a Tokenizer.

        Reads output data from the 'data' column of COMPLETE events in the database.

        Args:
            tokenizer: A Tokenizer object from HuggingFace

        Returns:
            RollupQueryTable: A table containing per-sample output sequence lengths, or None if no complete events found.
        """
        query_result = self.get_sample_outputs()

        rows = []
        for sample_uuid, data_bytes in query_result:
            output_sequence, reasoning_sequence = output_sequence_from_data(data_bytes)

            if output_sequence is None:
                continue

            # Concatenate reasoning and output if reasoning exists
            if reasoning_sequence is not None:
                full_sequence = f"{reasoning_sequence} {output_sequence}"
            else:
                full_sequence = output_sequence

            # Tokenize and calculate length
            output_tokens = tokenizer.tokenize(full_sequence)
            rows.append((sample_uuid, len(output_tokens)))

        if not rows:
            return None

        return RollupQueryTable("output_sequence_length", None, rows)

    @profile
    def derive_TPOT(
        self,
        tokenizer: Tokenizer,
        ttft_rollup: RollupQueryTable | None = None,
        sample_latency_rollup: RollupQueryTable | None = None,
        condense_table: bool = True,
        reporting_mode: TPOTReportingMode = TPOTReportingMode.REQUEST_WEIGHTED,
    ) -> RollupQueryTable | None:
        """Derives the TPOT metric from the text outputs, ttft, and sample latencies.

        Roughly, if a sample UUID `X` has a TTFT of `a`, a total latency of `b`, and an output sequence `S`,
        then `X` will contribute `len(tokenize(S)) - 1` entries in the table, each with the value:
             `(b - a) / len(tokenize(S) - 1)`
        If the sample was completed in non-streaming mode however, then `a` is assumed to be 0, and `X` will
        instead contribute `len(tokenize(S))` entries, each with the value: `b / len(tokenize(S))`

        TPOT tracks the time it takes for each token after the first to be generated (in streaming mode). Since
        the client does not have direct visibility into the endpoint / server-under-test, we have to estimate it,
        assuming that in an ideal scenario, each token outputed in the output text took the same amount of
        time.

        Args:
            tokenizer: A Tokenizer object from HuggingFace, used to calculate the number of tokens in a sequence
            ttft_rollup: Precomputed TTFT RollupQueryTable. If not provided, will be derived via self.derive_TTFT()
            sample_latency_rollup: Precomputed sample latency RollupQueryTable. If not provided, will be derived via self.derive_sample_latency()
            condense_table: Whether to condense the table by not storing individual token times, but rather just keeping the average time per token
                            and number of tokens per sample UUID. This is only supported if reporting_mode is TOKEN_WEIGHTED.
                            If reporting_mode is REQUEST_WEIGHTED, each sample only contributes one entry to the table. (Default: True)
            reporting_mode: TPOT reporting mode (REQUEST_WEIGHTED or TOKEN_WEIGHTED). (Default: REQUEST_WEIGHTED)
        """
        if ttft_rollup is None:
            ttft_rollup = self.derive_TTFT()

        # If no TTFT data available, TPOT cannot be calculated accurately for streaming mode
        if len(ttft_rollup) == 0:
            return None

        if sample_latency_rollup is None:
            sample_latency_rollup = self.derive_sample_latency()

        # Query for COMPLETE events with their data column
        query_result = self.get_sample_outputs()

        if not query_result:
            return None

        rows = []
        if condense_table and reporting_mode == TPOTReportingMode.TOKEN_WEIGHTED:
            repeats = []
        else:
            repeats = None

        for sample_uuid, data_bytes in query_result:
            if data_bytes is None or len(data_bytes) == 0:
                continue

            # Extract output from decoded data
            # For TPOT calculation, we need the output to be a list of chunks (streaming mode) with at least 2
            # elements
            output_sequence, reasoning_sequence = output_sequence_from_data(
                data_bytes, join_chunks=False
            )
            if not isinstance(output_sequence, list):
                continue

            all_chunks = output_sequence
            if isinstance(reasoning_sequence, list):
                all_chunks.extend(reasoning_sequence)

            # For TPOT, we need streaming data (list of chunks with at least 2 elements)
            if len(all_chunks) < 2:
                continue

            # Skip samples that are not in the filtered rollups (i.e., issued after STOP_PERFORMANCE_TRACKING)
            if sample_uuid not in sample_latency_rollup:
                continue

            # Output can be in one of two formats depending on the issuer:
            # 1. A list of all chunks (i.e. ['chunk1', 'chunk2', ...])
            # 2. A 2 item list of ['chunk1', 'chunk2chunk3...']
            # Both of these are valid as we only need to distinguish the first chunk for the purposes of TPOT calculation.
            # The choice is up to the issuer implementation depending on performance considerations.

            # Join list elements to get the non-first chunk text
            if len(all_chunks) > 2:
                non_first_chunk = "".join(str(chunk) for chunk in all_chunks[1:])
            else:
                non_first_chunk = str(all_chunks[1])

            if len(non_first_chunk) == 0:
                # Possible malformed output data where empty string is included as a non-first chunk
                continue

            non_first_tokens = tokenizer.tokenize(non_first_chunk)
            n_non_first_tokens = len(non_first_tokens)

            latency = sample_latency_rollup.filter_uuid(sample_uuid, only_first=True)
            if latency is None:
                raise SampleUUIDNotFoundError(sample_uuid, "events record")

            ttft = ttft_rollup.filter_uuid(sample_uuid, only_first=True)
            if ttft is None:
                # Non-streaming mode for this sample - error
                raise RuntimeError(
                    f"No TTFT found for sample {sample_uuid} in streaming mode"
                )

            avg_tpot = (latency - ttft) / n_non_first_tokens

            if condense_table:
                rows.append((sample_uuid, avg_tpot))
                if reporting_mode == TPOTReportingMode.TOKEN_WEIGHTED:
                    repeats.append(n_non_first_tokens)
            else:
                # Entries are tuples, and are such immutable. We can use list multiplication for performance
                repeat_fac = (
                    1
                    if reporting_mode == TPOTReportingMode.REQUEST_WEIGHTED
                    else n_non_first_tokens
                )
                rows.extend([(sample_uuid, avg_tpot)] * repeat_fac)

        if not rows:
            return None

        return RollupQueryTable("tpot", None, rows, repeats=repeats)

    def close(self):
        if self.is_closed:
            logging.debug("Connection is already closed, skipping close")
            return
        self.is_closed = True

        if self.cur_ is not self.conn:
            self.cur_.close()
        self.conn.close()

    def dump_to_json(self, json_path: Path):
        """
        Dumps all events to a JSONL file, including decoded output data from the 'data' column.
        Each line in the output file is a valid JSON object.
        """

        with json_path.open("w", encoding="utf-8", newline="") as f:
            query_result = self.cur_.execute(
                "SELECT sample_uuid, event_type, timestamp_ns, data FROM events"
            )
            while True:
                if hasattr(query_result, "fetchmany"):
                    rows = query_result.fetchmany(1000)
                else:
                    rows = query_result.fetchall()

                if not rows:
                    break

                for sample_uuid, event_type, timestamp_ns, data_bytes in rows:
                    value = ""

                    # For events with data, decode and extract the relevant value
                    if data_bytes is not None and len(data_bytes) > 0:
                        if event_type == SampleEvent.COMPLETE.value:
                            # For COMPLETE, use helper method to extract output sequence
                            output_seq, reasoning_seq = output_sequence_from_data(
                                data_bytes
                            )
                            if output_seq is not None:
                                if reasoning_seq is not None:
                                    value = f"[reasoning: {reasoning_seq}] {output_seq}"
                                else:
                                    value = output_seq
                        elif event_type in (
                            SampleEvent.FIRST_CHUNK.value,
                            SessionEvent.ERROR.value,
                        ):
                            # For other event types, just decode and stringify
                            try:
                                decoded_data = orjson.loads(data_bytes)
                                value = str(decoded_data) if decoded_data else ""
                            except (orjson.JSONDecodeError, TypeError) as e:
                                value = f"<DECODE_ERROR: {e}>"

                    approx_datetime_str = monotime_to_datetime(timestamp_ns).isoformat()

                    json_obj = {
                        "sample_uuid": sample_uuid,
                        "event_type": event_type,
                        "timestamp_ns": timestamp_ns,
                        "approx_datetime_str": approx_datetime_str,
                        "value": value,
                    }
                    # Use orjson.dumps for each line
                    f.write(
                        orjson.dumps(json_obj, option=orjson.OPT_SORT_KEYS).decode(
                            "utf-8"
                        )
                        + "\n"
                    )

    def __enter__(self):
        if self.is_closed:
            self.init_connection()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _get_version_info(self) -> dict[str, str | None]:
        """Extract version info from TEST_STARTED event data.

        Returns:
            Dictionary with 'version' and 'git_sha' keys.
        """
        query = f"""
        SELECT data FROM events
        WHERE event_type = '{SessionEvent.TEST_STARTED.value}'
        LIMIT 1
        """
        result = self.cur_.execute(query).fetchone()
        if result and result[0]:
            try:
                return orjson.loads(result[0])
            except Exception:
                pass
        return {"version": "unknown", "git_sha": None}

    def get_test_started_at(self) -> int | None:
        """Gets the timestamp of the TEST_STARTED event.

        Returns:
            int: The timestamp of the TEST_STARTED event.
        """
        query = f"""
        SELECT timestamp_ns FROM events
        WHERE event_type = '{SessionEvent.TEST_STARTED.value}'
        LIMIT 1"""
        result = self.cur_.execute(query).fetchone()
        if result and result[0]:
            return result[0]
        return None

    def create_report(
        self,
        tokenizer: Tokenizer | None = None,
        tpot_reporting_mode: TPOTReportingMode = TPOTReportingMode.REQUEST_WEIGHTED,
    ) -> Report:
        """Creates a Report object from the metrics.

        Args:
            tokenizer: A Tokenizer object from HuggingFace. If provided, output sequence lengths will be calculated.
            tpot_reporting_mode: TPOT reporting mode (REQUEST_WEIGHTED or TOKEN_WEIGHTED). (Default: REQUEST_WEIGHTED)

        Returns:
            Report: A Report object containing the metrics.
        """
        test_started_at = self.get_test_started_at()
        if test_started_at is None:
            raise RuntimeError("TEST_STARTED event not found in database")

        sample_statuses = self.get_sample_statuses()
        ttft_rollup = self.derive_TTFT()
        sample_latency_rollup = self.derive_sample_latency()
        output_sequence_lengths = None
        tpot_summary = None
        if tokenizer is not None:
            osl_rollup = self.get_output_sequence_lengths(tokenizer)
            if osl_rollup is not None:
                output_sequence_lengths = osl_rollup.summarize()

            # Only calculate TPOT if TTFT data is available (streaming mode)
            if len(ttft_rollup) > 0:
                tpot_rollup = self.derive_TPOT(
                    tokenizer,
                    ttft_rollup=ttft_rollup,
                    sample_latency_rollup=sample_latency_rollup,
                    reporting_mode=tpot_reporting_mode,
                )
                if tpot_rollup is not None:
                    tpot_summary = tpot_rollup.summarize()

        if len(ttft_rollup) == 0:
            ttft_summary = None
        else:
            ttft_summary = ttft_rollup.summarize()

        # Extract version information
        version_info = self._get_version_info()

        return Report(
            version=version_info.get("version", "unknown"),
            git_sha=version_info.get("git_sha"),
            test_started_at=test_started_at,
            n_samples_issued=sample_statuses["total_sent"],
            n_samples_completed=sample_statuses["completed"],
            duration_ns=self.derive_duration(),
            ttft=ttft_summary,
            tpot=tpot_summary,
            latency=sample_latency_rollup.summarize(),
            output_sequence_lengths=output_sequence_lengths,
            tpot_reporting_mode=tpot_reporting_mode,
        )
