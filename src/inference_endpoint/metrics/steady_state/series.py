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

from collections.abc import Callable
from dataclasses import dataclass, field

import msgspec
import msgspec.json

from inference_endpoint.core.record import (
    EventRecord,
    EventType,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import TextModelOutput

_DECODER = msgspec.json.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)


@dataclass(slots=True)
class SuperPassRollup:
    index: int
    n_issued: int = 0
    first_issue_ns: int = -1
    last_issue_ns: int = -1
    ttft_ns: list[float] = field(default_factory=list)
    latency_ns: list[float] = field(default_factory=list)
    tpot_ns: list[float] = field(default_factory=list)
    out_tokens: int = 0


def super_pass_size(dataset_size: int, concurrency: int) -> int:
    if dataset_size <= 0 or concurrency <= 0:
        raise ValueError("dataset_size and concurrency must be positive")
    s = -(-concurrency // dataset_size)  # ceil division
    return dataset_size * s


def build_super_pass_series(
    events_path: str,
    dataset_size: int,
    concurrency: int,
    count_tokens: Callable[[list[str]], list[int]] | None = None,
) -> list[SuperPassRollup]:
    """Bucket performance-tracked samples into super-passes by issue order.

    Each sample's TTFT/latency is attributed to the super-pass its ISSUED event
    fell into; completions after STOP_PERFORMANCE_TRACKING still count for rows
    that exist. TPOT/out_tokens are populated only when ``count_tokens`` is given.
    """
    sp_samples = super_pass_size(dataset_size, concurrency)
    series: list[SuperPassRollup] = []
    # uuid -> (super_pass_index, issue_ns, recv_first_ns|None, tpot_text|None)
    rows: dict[str, list] = {}
    tracking = False
    issue_counter = 0
    # buffered TPOT token counting (uuid-ordered)
    batch_uuids: list[str] = []
    batch_texts: list[str] = []
    # TPOT pairs a token count (deferred, batched) with the complete-recv delta;
    # the row is popped at COMPLETE, so stash (sp_idx, delta_ns) until flush.
    pending_tpot: dict[str, tuple[int, float]] = {}

    def _ensure(idx: int) -> SuperPassRollup:
        while len(series) <= idx:
            series.append(SuperPassRollup(index=len(series)))
        return series[idx]

    def flush_tpot() -> None:
        if not batch_texts or count_tokens is None:
            batch_uuids.clear()
            batch_texts.clear()
            return
        for uuid, cnt in zip(batch_uuids, count_tokens(batch_texts), strict=True):
            sp_idx, delta = pending_tpot.pop(uuid)
            if cnt > 0:
                series[sp_idx].tpot_ns.append(delta / cnt)
                series[sp_idx].out_tokens += cnt
        batch_uuids.clear()
        batch_texts.clear()

    with open(events_path, "rb") as f:
        for line in f:
            try:
                rec = _DECODER.decode(line)
            except (msgspec.DecodeError, NotImplementedError):
                continue
            et = rec.event_type
            if et is SessionEventType.START_PERFORMANCE_TRACKING:
                tracking = True
            elif et is SessionEventType.STOP_PERFORMANCE_TRACKING:
                tracking = False
            elif et is SampleEventType.ISSUED:
                if not tracking or not rec.sample_uuid:
                    continue
                existing = rows.get(rec.sample_uuid)
                if existing is not None:
                    existing[1] = rec.timestamp_ns  # retry: refresh issue ts only
                    continue
                sp_idx = issue_counter // sp_samples
                issue_counter += 1
                rows[rec.sample_uuid] = [sp_idx, rec.timestamp_ns, None, None]
                sp = _ensure(sp_idx)
                sp.n_issued += 1
                if sp.first_issue_ns < 0:
                    sp.first_issue_ns = rec.timestamp_ns
                sp.last_issue_ns = rec.timestamp_ns
            elif et is SampleEventType.RECV_FIRST:
                row = rows.get(rec.sample_uuid)
                if row is not None:
                    row[2] = rec.timestamp_ns
                    series[row[0]].ttft_ns.append(float(rec.timestamp_ns - row[1]))
            elif et is SampleEventType.COMPLETE:
                row = rows.pop(rec.sample_uuid, None)
                if row is None:
                    continue
                sp_idx, issue_ns, recv_ns, _ = row
                series[sp_idx].latency_ns.append(float(rec.timestamp_ns - issue_ns))
                if count_tokens is None or recv_ns is None:
                    continue
                data = rec.data
                if not isinstance(data, TextModelOutput) or data.tool_calls:
                    continue
                text = data.text_after_first_chunk()
                if text:
                    pending_tpot[rec.sample_uuid] = (
                        sp_idx,
                        float(rec.timestamp_ns - recv_ns),
                    )
                    batch_uuids.append(rec.sample_uuid)
                    batch_texts.append(text)
                    if len(batch_texts) >= 4096:
                        flush_tpot()
    flush_tpot()
    return series
