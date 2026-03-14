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

"""Per-sample metrics table for the metrics aggregator service."""

from __future__ import annotations

import logging

import msgspec

logger = logging.getLogger(__name__)


# gc=False: audit 2026-03: output_chunks list is only appended with str values,
# never with the SampleRow itself or anything referencing it. No cycles possible.
class SampleRow(msgspec.Struct, gc=False):  # type: ignore[call-arg]
    """Per-sample timing and text accumulation for metric computation.

    Stores raw timestamps (int nanoseconds, or None if not yet received)
    and accumulated text for tokenization-based metrics.

    AT-RISK (gc=False): Has mutable container field `output_chunks`. Any change
    that stores this struct (or something referencing it) inside `output_chunks`
    must be audited; if so, remove gc=False.
    """

    sample_uuid: str
    issued_ns: int | None = None
    complete_ns: int | None = None
    recv_first_ns: int | None = None
    last_recv_ns: int | None = None
    client_send_ns: int | None = None
    client_resp_done_ns: int | None = None
    prompt_text: str | None = None
    first_chunk_text: str | None = None
    output_chunks: list[str] = msgspec.field(default_factory=list)

    def ttft_ns(self) -> int | None:
        """Time to first token: recv_first - issued."""
        if self.recv_first_ns is not None and self.issued_ns is not None:
            return self.recv_first_ns - self.issued_ns
        return None

    def sample_latency_ns(self) -> int | None:
        """End-to-end latency: complete - issued."""
        if self.complete_ns is not None and self.issued_ns is not None:
            return self.complete_ns - self.issued_ns
        return None

    def request_duration_ns(self) -> int | None:
        """HTTP request duration: client_resp_done - client_send."""
        if self.client_resp_done_ns is not None and self.client_send_ns is not None:
            return self.client_resp_done_ns - self.client_send_ns
        return None

    def output_text(self) -> str:
        """Full output text from accumulated streaming chunks."""
        return "".join(self.output_chunks)


class MetricsTable:
    """Stores in-flight sample rows.

    Rows are created on ISSUED and removed on COMPLETE (after metrics are emitted).
    """

    def __init__(self) -> None:
        self._in_flight: dict[str, SampleRow] = {}

    def create_row(self, sample_uuid: str) -> SampleRow:
        if sample_uuid in self._in_flight:
            logger.warning(
                "Duplicate ISSUED for sample %s, possibly due to retry - skipping",
                sample_uuid,
            )
            return self._in_flight[sample_uuid]
        row = SampleRow(sample_uuid=sample_uuid)
        self._in_flight[sample_uuid] = row
        return row

    def get_row(self, sample_uuid: str) -> SampleRow | None:
        return self._in_flight.get(sample_uuid)

    def remove_row(self, sample_uuid: str) -> SampleRow | None:
        return self._in_flight.pop(sample_uuid, None)

    def __len__(self) -> int:
        return len(self._in_flight)
