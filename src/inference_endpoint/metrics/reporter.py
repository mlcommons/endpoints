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

import dataclasses
import functools
import importlib
import logging
import numbers
import os
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import orjson

from ..load_generator.events import SampleEvent, SessionEvent
from ..profiling import profile

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
            return int(self.repeats.sum())

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
        percentiles: Iterable[float] = (99.9, 99, 95, 90, 80, 75, 50, 25, 10, 5, 1),
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

    @functools.cached_property
    def e2e_sample_latency_sec(self) -> float:
        """Calculates the end-to-end total latency across all samples in the test in seconds.

        Returns:
            The end-to-end total latency across all samples in the test in seconds.
        """
        return float(self.latency["total"] / 1e9)

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
        d["e2e_sample_latency_sec"] = self.e2e_sample_latency_sec
        json_str = orjson.dumps(d).decode("utf-8")
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
    ) -> None:
        """Displays a metric dictionary in a human-readable format.

        Args:
            metric_dict: The metric dictionary to display.
            fn: The function to call to print a string, such as logging.info, file.write, etc. (Default: `print`)
            unit: The string representing the unit of the metric
            max_bar_length: The maximum length of the bar to display for the histogram
            scale_factor: The factor to scale metric values by. (Default: 1.0)
        """
        for name, key in [
            ("Min", "min"),
            ("Max", "max"),
            ("Median", "median"),
            ("Avg.", "avg"),
            ("Std Dev.", "std_dev"),
        ]:
            fn(f"  {name}: {metric_dict[key] * scale_factor:.2f} {unit}")
        fn("\n  Histogram:")

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
                fn(f"  {bucket_str:>{max_bucket_str_len}} |{'#' * bar_length} {count}")

            fn("\n  Percentiles:")
            max_percentile_str_len = max(
                len(str(p)) for p in metric_dict["percentiles"].keys()
            )
            for percentile, value in metric_dict["percentiles"].items():
                fn(
                    f"  {percentile:>{max_percentile_str_len}}: {value * scale_factor:.2f} {unit}"
                )

    def display(
        self,
        fn: Callable[[str], None] = print,
        show_e2e_sample_latency: bool = False,
    ) -> None:
        """Displays the report in a human-readable format.

        Args:
            fn: The function to call to print a string, such as logging.info, file.write, etc. (Default: `print`)
            show_e2e_sample_latency: Whether to show the end-to-end sample latency. (Default: False)
        """

        fn("----------------- Summary -----------------")
        fn(f"Total samples issued: {self.n_samples_issued}")
        fn(f"Total samples completed: {self.n_samples_completed}")
        fn(f"Duration: {self.duration_ns / 1e9:.2f} seconds")
        if show_e2e_sample_latency:
            fn(
                f"Total time spent waiting on samples: {self.e2e_sample_latency_sec} seconds"
            )
        fn(f"QPS: {self.qps:.2f}")

        if self.tps is not None:
            fn(f"TPS: {self.tps:.2f}")

        fn("\n\n------------------- Latency Breakdowns -------------------")
        if len(self.latency) > 0 and self.ttft == 0:
            fn(
                "WARNING: Non-streaming-based Issuer used. TTFT metrics cannot be calculated"
            )

        for section_name, metric_dict, unit, scale_factor in [
            ("TTFT", self.ttft, "ms", 1e-6),
            (f"TPOT ({self.tpot_reporting_mode.value})", self.tpot, "ms", 1e-6),
            ("Latency", self.latency, "ms", 1e-6),
            ("Output sequence lengths", self.output_sequence_lengths, "tokens", 1.0),
        ]:
            if metric_dict is None or len(metric_dict) == 0:
                continue
            fn(f"{section_name}:")
            Report._display_metric(
                metric_dict,
                fn=fn,
                unit=unit,
                scale_factor=scale_factor,
            )
            fn("\n")


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
        self.outputs_path = self.connection_name.with_suffix(".outputs.jsonl")
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

    def derive_duration(self) -> float:
        """Calculates the total test duration as the difference between TEST_ENDED and TEST_STARTED events.

        Returns:
            float: The duration in nanoseconds.
        """
        n_time_bounds = self.cur_.execute(f"""
        SELECT
            COUNT(DISTINCT CASE WHEN event_type = '{SessionEvent.TEST_STARTED.value}' THEN timestamp_ns END) AS n_starts,
            COUNT(DISTINCT CASE WHEN event_type = '{SessionEvent.TEST_ENDED.value}' THEN timestamp_ns END) AS n_ends
        FROM events
        """).fetchone()
        if n_time_bounds != (1, 1):
            raise RuntimeError(
                f"Multiple TEST_STARTED or TEST_ENDED events found - {n_time_bounds[0]} START events, {n_time_bounds[1]} END events"
            )

        # Must have a dummy sample_uuid column, as RollupQueryTable uses it as a primary key
        rollup = self.derive_metric(
            f"""
            SELECT
                '' as sample_uuid,
                MAX(CASE WHEN event_type = '{SessionEvent.TEST_ENDED.value}' THEN timestamp_ns END) -
                MAX(CASE WHEN event_type = '{SessionEvent.TEST_STARTED.value}' THEN timestamp_ns END) AS duration
            FROM events
            WHERE event_type IN ('{SessionEvent.TEST_STARTED.value}', '{SessionEvent.TEST_ENDED.value}')
            HAVING COUNT(DISTINCT event_type) = 2
            """,
            "duration",
        )
        if len(rollup) == 0:
            return None  # Test did not complete, return None
        elif len(rollup) > 1:
            raise RuntimeError("Malformed query result - Only 1 row expected")
        return rollup[0].metric_value

    def derive_sample_latency(self) -> RollupQueryTable:
        """Calculates the end-to-end latency for each sample from issue to completion.

        Returns:
            RollupQueryTable: A table containing per-sample latencies in nanoseconds.
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
            HAVING COUNT(DISTINCT event_type) = 2
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

    def get_error_count(self) -> int:
        return self.cur_.execute(f"""
        SELECT
            COUNT(*) AS error_count
        FROM events
        WHERE event_type = '{SessionEvent.ERROR.value}'
        """).fetchone()[0]

    def read_output_rows(self) -> Iterator[tuple[str, str]]:
        """Iterator to load and read lines from the outputs file, decoding each line as JSON and yielding the sample_uuid and output.

        Returns:
            Iterator[tuple[str, str]]: An iterator of tuples containing the sample_uuid and output.
        """
        with self.outputs_path.open("r") as outputs:
            for line in outputs:
                data = orjson.loads(line)
                if "output" not in data:
                    continue
                yield data["s_uuid"], data["output"]

    @profile
    def get_output_sequence_lengths(
        self, tokenizer: Tokenizer
    ) -> RollupQueryTable | None:
        """Returns a RollupQueryTable representing per-sample output sequence lengths based on a Tokenizer.

        If no outputs file is found, returns None.

        Args:
            tokenizer: A Tokenizer object from HuggingFace

        Returns:
            RollupQueryTable: A table containing per-sample output sequence lengths.
        """
        if not self.outputs_path.exists():
            return None

        rows = []
        for sample_uuid, output in self.read_output_rows():
            if isinstance(output, list):
                output = "".join(output)
            output_tokens = tokenizer.tokenize(output)
            rows.append((sample_uuid, len(output_tokens)))
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
        if not self.outputs_path.exists():
            return None

        if ttft_rollup is None:
            ttft_rollup = self.derive_TTFT()

        # If no TTFT data available, TPOT cannot be calculated accurately for streaming mode
        if len(ttft_rollup) == 0:
            return None

        if sample_latency_rollup is None:
            sample_latency_rollup = self.derive_sample_latency()

        rows = []
        if condense_table and reporting_mode == TPOTReportingMode.TOKEN_WEIGHTED:
            repeats = []
        else:
            repeats = None

        for sample_uuid, output in self.read_output_rows():
            if not isinstance(output, list):  # JSON always deserializes to list
                continue
            elif len(output) < 2:
                continue

            # Output can be in one of two formats depending on the issuer:
            # 1. A list of all chunks (i.e. ['chunk1', 'chunk2', ...])
            # 2. A 2 item list of ['chunk1', 'chunk2chunk3...']
            # Both of these are valid as we only need to distinguish the first chunk for the purposes of TPOT calculation.
            # The choice is up to the issuer implementation depending on performance considerations.
            if len(output) > 2:
                non_first_chunk = "".join(output[1:])
            else:
                non_first_chunk = output[1]

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
        return RollupQueryTable("tpot", None, rows, repeats=repeats)

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
        return Report(
            n_samples_issued=sample_statuses["total_sent"],
            n_samples_completed=sample_statuses["completed"],
            duration_ns=self.derive_duration(),
            ttft=ttft_summary,
            tpot=tpot_summary,
            latency=sample_latency_rollup.summarize(),
            output_sequence_lengths=output_sequence_lengths,
            tpot_reporting_mode=tpot_reporting_mode,
        )
