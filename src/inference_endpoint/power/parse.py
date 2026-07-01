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

"""Parse a power trace file into canonical samples.

Tolerant by design: malformed lines are dropped and counted, never raised, so a
flaky collector degrades gracefully instead of failing the benchmark.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from inference_endpoint.power.sources import ResolvedSource


@dataclass(frozen=True)
class PowerSample:
    ts_epoch_s: float
    label: str
    value: float


@dataclass(frozen=True)
class ParseResult:
    samples: list[PowerSample]
    dropped: int


def _parse_ts(raw: str) -> float:
    """Accept epoch seconds (float) or the nvidia-smi ``YYYY/MM/DD HH:MM:SS.mmm``."""
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
        pass
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"unparseable timestamp: {raw!r}")


def parse_trace(path: Path, src: ResolvedSource) -> ParseResult:
    """Read ``path`` per ``src``'s format/field-mapping into canonical samples."""
    if not path.exists():
        return ParseResult(samples=[], dropped=0)
    if src.fmt == "csv":
        return _parse_csv(path, src)
    return _parse_jsonl(path, src)


def _parse_jsonl(path: Path, src: ResolvedSource) -> ParseResult:
    samples: list[PowerSample] = []
    dropped = 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    dropped += 1
                    continue
                ts = _parse_ts(str(obj[src.ts_field]))
                value = float(obj[src.value_field])
                label = (
                    str(obj.get(src.label_field, "default"))
                    if src.label_field
                    else "default"
                )
                samples.append(PowerSample(ts, label, value))
            except (ValueError, KeyError, TypeError):
                dropped += 1
    return ParseResult(samples=samples, dropped=dropped)


def _parse_csv(path: Path, src: ResolvedSource) -> ParseResult:
    samples: list[PowerSample] = []
    dropped = 0
    # Field mapping is by integer index for headerless CSV (nvidia-smi), else by
    # column name. Split paths so the row type stays concrete for each.
    with path.open() as f:
        if src.csv_header:
            for named in csv.DictReader(f):
                try:
                    ts = _parse_ts(named[src.ts_field])
                    value = float(named[src.value_field])
                    label = (
                        str(named[src.label_field]) if src.label_field else "default"
                    )
                    samples.append(PowerSample(ts, label.strip(), value))
                except (ValueError, KeyError, TypeError):
                    dropped += 1
        else:
            try:
                ts_i = int(src.ts_field)
                val_i = int(src.value_field)
                lbl_i = int(src.label_field) if src.label_field else None
            except (ValueError, TypeError):
                return ParseResult(samples=[], dropped=0)
            for row in csv.reader(f):
                try:
                    ts = _parse_ts(row[ts_i])
                    value = float(row[val_i])
                    label = row[lbl_i].strip() if lbl_i is not None else "default"
                    # Prefix bare numeric labels only when the source asked for it
                    # (e.g. nvidia-smi GPU index 0 -> "gpu0"); generic sources
                    # keep their labels verbatim.
                    if src.label_prefix and label.isdigit():
                        label = f"{src.label_prefix}{label}"
                    samples.append(PowerSample(ts, label.strip(), value))
                except (ValueError, IndexError, TypeError):
                    dropped += 1
    return ParseResult(samples=samples, dropped=dropped)
