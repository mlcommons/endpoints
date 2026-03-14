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

"""Metric emitters for the metrics aggregator service."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import msgspec


class MetricEmitter(ABC):
    """Base class for metric emitters."""

    @abstractmethod
    def emit(self, sample_uuid: str, metric_name: str, value: int | float) -> None:
        """Emit a metric value for a sample."""
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered metrics to the underlying store."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Flush and release resources."""
        raise NotImplementedError


class _MetricRecord(msgspec.Struct, gc=False):  # type: ignore[call-arg]
    sample_uuid: str
    metric_name: str
    value: int | float
    timestamp_ns: int


class JsonlMetricEmitter(MetricEmitter):
    """Writes metrics as JSONL lines to a file.

    Each line is a JSON object: {"sample_uuid": ..., "metric_name": ..., "value": ..., "timestamp_ns": ...}
    """

    def __init__(self, file_path: Path, flush_interval: int = 100) -> None:
        self._file_path = file_path.with_suffix(".jsonl")
        self._file = self._file_path.open("w")
        self._encoder = msgspec.json.Encoder()
        self._flush_interval = flush_interval
        self._n_since_flush = 0

    def emit(self, sample_uuid: str, metric_name: str, value: int | float) -> None:
        record = _MetricRecord(
            sample_uuid=sample_uuid,
            metric_name=metric_name,
            value=value,
            timestamp_ns=time.monotonic_ns(),
        )
        self._file.write(self._encoder.encode(record).decode("utf-8") + "\n")
        self._n_since_flush += 1
        if self._n_since_flush >= self._flush_interval:
            self.flush()

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()
        self._n_since_flush = 0

    def close(self) -> None:
        if self._file is not None:
            try:
                self.flush()
                self._file.close()
            except (OSError, FileNotFoundError):
                pass
            finally:
                self._file = None  # type: ignore[assignment]
