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

"""Phase-0 shared sample extractor for the steady-state studies.

One streaming pass over a run's ``events.jsonl`` produces a compact per-sample
Parquet table plus a sidecar ``run_meta.json``. The extractor stores RAW
timestamps only (``issue_ns``/``first_token_ns``/``complete_ns``) plus the
issue-order ``super_pass`` bucket; it bakes in NO window or drain flag, so each
downstream study defines its own steady-state window.

Bucketing mirrors ``inference_endpoint.metrics.steady_state.series``: samples are
attributed to a super-pass by ISSUED issue-order, and a duplicate ISSUED (a
retry) refreshes the issue timestamp WITHOUT advancing the issue counter.

``out_tokens`` is populated only from a reported completion-token count present
in the event payload. The COMPLETE payload is a ``TextModelOutput`` carrying
generated text but no reported token count, so ``out_tokens`` (and the derived
``tpot_ns``) are left null; studies tokenize on demand. The columns are still
emitted for a stable schema.

``dataset_idx`` is the sample's index into the source dataset, read from the
run's ``sample_idx_map.json`` (``{split: {uuid: idx}}``) — the map is keyed by
sample uuid, so it joins onto the extracted rows without touching the events.
This lets studies group repeated issuances of the same dataset sample (e.g. the
long-OSL hairball that trails into the drain is the same handful of samples
re-issued each pass). Unmapped rows, or a run with no map, get ``-1``.

Usage (run via uv):

    uv run --with pyarrow python scripts/steady_state_extract_samples.py \
        --events ~/skritch/.../C8/events.jsonl \
        --dataset-size 6396 --concurrency 8

Outputs ``samples.parquet`` + ``run_meta.json`` beside the events file unless
overridden. For a poisson run there is no concurrency N; pass ``--poisson`` with
``--concurrency 1`` so the super-pass bucket is a single dataset pass.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import msgspec
import msgspec.json
import pyarrow as pa
import pyarrow.parquet as pq
from inference_endpoint.core.record import (
    EventRecord,
    EventType,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.metrics.steady_state.series import super_pass_size

_DECODER = msgspec.json.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)

# Per-sample mutable row layout (indexed for hot-loop speed, no attr lookups).
_SP = 0  # super_pass bucket index
_ISSUE = 1  # issue_ns (latest ISSUED for this uuid; retries refresh)
_FIRST = 2  # first_token_ns (RECV_FIRST) or None
_COMPLETE = 3  # complete_ns (COMPLETE) or None
_OUT_TOKENS = 4  # reported completion-token count or None


def extract(events_path: str, dataset_size: int, concurrency: int) -> tuple[dict, dict]:
    """Stream ``events_path`` once → (columns dict, run-meta dict).

    Never materializes the file: iterates line by line. The only in-memory
    structure is the per-sample row map, bounded by the issued-sample count.
    """
    sp_samples = super_pass_size(dataset_size, concurrency)

    rows: dict[str, list] = {}
    order: list[str] = []  # issue order, for deterministic output rows
    tracking = False
    issue_counter = 0

    first_issue_ns = -1
    last_issue_ns = -1
    last_complete_ns = -1
    n_complete = 0

    with open(events_path, "rb") as f:
        for line in f:
            try:
                rec = _DECODER.decode(line)
            except (msgspec.DecodeError, NotImplementedError):
                continue
            et = rec.event_type
            if et is SampleEventType.ISSUED:
                if not tracking or not rec.sample_uuid:
                    continue
                existing = rows.get(rec.sample_uuid)
                if existing is not None:
                    # Retry: refresh issue timestamp only; do not advance counter.
                    existing[_ISSUE] = rec.timestamp_ns
                    continue
                sp_idx = issue_counter // sp_samples
                issue_counter += 1
                rows[rec.sample_uuid] = [sp_idx, rec.timestamp_ns, None, None, None]
                order.append(rec.sample_uuid)
                if first_issue_ns < 0:
                    first_issue_ns = rec.timestamp_ns
                last_issue_ns = rec.timestamp_ns
            elif et is SampleEventType.RECV_FIRST:
                row = rows.get(rec.sample_uuid)
                if row is not None:
                    row[_FIRST] = rec.timestamp_ns
            elif et is SampleEventType.COMPLETE:
                row = rows.get(rec.sample_uuid)
                if row is None:
                    continue
                if row[_COMPLETE] is None:
                    n_complete += 1
                row[_COMPLETE] = rec.timestamp_ns
                if rec.timestamp_ns > last_complete_ns:
                    last_complete_ns = rec.timestamp_ns
                # out_tokens: only a reported completion-token count qualifies.
                # The COMPLETE payload (TextModelOutput) carries no such count,
                # so this stays None; studies tokenize on demand.
            elif et is SessionEventType.START_PERFORMANCE_TRACKING:
                tracking = True
            elif et is SessionEventType.STOP_PERFORMANCE_TRACKING:
                tracking = False

    n_issued = len(order)
    uuids: list[str] = order
    super_pass: list[int] = [0] * n_issued
    issue_ns: list[int] = [0] * n_issued
    first_token_ns: list[int | None] = [None] * n_issued
    complete_ns: list[int | None] = [None] * n_issued
    ttft_ns: list[int | None] = [None] * n_issued
    lifetime_ns: list[int | None] = [None] * n_issued
    out_tokens: list[int | None] = [None] * n_issued
    tpot_ns: list[float | None] = [None] * n_issued

    for i, uuid in enumerate(order):
        sp, iss, first, comp, out = rows[uuid]
        super_pass[i] = sp
        issue_ns[i] = iss
        first_token_ns[i] = first
        complete_ns[i] = comp
        if first is not None:
            ttft_ns[i] = first - iss
        if comp is not None:
            lifetime_ns[i] = comp - iss
        if out is not None:
            out_tokens[i] = out
            if first is not None and comp is not None and out > 0:
                tpot_ns[i] = (comp - first) / out

    columns = {
        "uuid": uuids,
        "super_pass": super_pass,
        "issue_ns": issue_ns,
        "first_token_ns": first_token_ns,
        "complete_ns": complete_ns,
        "ttft_ns": ttft_ns,
        "lifetime_ns": lifetime_ns,
        "out_tokens": out_tokens,
        "tpot_ns": tpot_ns,
    }
    meta = {
        "N": concurrency,
        "dataset_size": dataset_size,
        "super_pass_size": sp_samples,
        "first_issue_ns": first_issue_ns if first_issue_ns >= 0 else None,
        "last_issue_ns": last_issue_ns if last_issue_ns >= 0 else None,
        "last_complete_ns": last_complete_ns if last_complete_ns >= 0 else None,
        "n_issued": n_issued,
        "n_complete": n_complete,
    }
    return columns, meta


def load_sample_idx_map(path: str) -> dict[str, int]:
    """Flatten ``sample_idx_map.json`` (``{split: {uuid: idx}}``) → ``{uuid: idx}``.

    Splits (performance/accuracy/...) are merged; a uuid appears in one split, so
    there is no cross-split collision in practice. Returns an empty dict for a
    missing or empty file.
    """
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    with open(path) as f:
        raw = json.load(f)
    merged: dict[str, int] = {}
    for value in raw.values():
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _schema() -> pa.Schema:
    return pa.schema(
        [
            ("uuid", pa.string()),
            ("dataset_idx", pa.int32()),
            ("super_pass", pa.int32()),
            ("issue_ns", pa.int64()),
            ("first_token_ns", pa.int64()),
            ("complete_ns", pa.int64()),
            ("ttft_ns", pa.int64()),
            ("lifetime_ns", pa.int64()),
            ("out_tokens", pa.int32()),
            ("tpot_ns", pa.float64()),
        ]
    )


def _write_parquet(columns: dict, out_path: str) -> None:
    table = pa.table(columns, schema=_schema())
    pq.write_table(table, out_path, compression="zstd")


def _write_meta(meta: dict, out_path: str) -> None:
    tmp = f"{out_path}.tmp"
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", required=True, help="path to events.jsonl")
    ap.add_argument("--dataset-size", type=int, required=True)
    ap.add_argument(
        "--concurrency",
        type=int,
        required=True,
        help="target concurrency N; for poisson pass 1 with --poisson",
    )
    ap.add_argument(
        "--poisson",
        action="store_true",
        help="poisson run: record N as null; super-pass = one dataset pass",
    )
    ap.add_argument("--note", default="", help="free-form provenance note for run_meta")
    ap.add_argument(
        "--sample-idx-map",
        default=None,
        help="path to sample_idx_map.json; default: beside the events file",
    )
    ap.add_argument("--out-parquet", default=None)
    ap.add_argument("--out-meta", default=None)
    args = ap.parse_args()

    events_path = args.events
    base = Path(events_path).parent
    out_parquet = args.out_parquet or str(base / "samples.parquet")
    out_meta = args.out_meta or str(base / "run_meta.json")
    map_path = args.sample_idx_map or str(base / "sample_idx_map.json")

    columns, meta = extract(events_path, args.dataset_size, args.concurrency)

    idx_map = load_sample_idx_map(map_path)
    columns["dataset_idx"] = [idx_map.get(u, -1) for u in columns["uuid"]]

    if args.poisson:
        meta["N"] = None
    if args.note:
        meta["note"] = args.note
    meta["out_tokens_present"] = any(v is not None for v in columns["out_tokens"])
    n_mapped = sum(1 for v in columns["dataset_idx"] if v >= 0)
    meta["sample_idx_map"] = map_path if idx_map else None
    meta["dataset_idx_mapped"] = n_mapped

    _write_parquet(columns, out_parquet)
    _write_meta(meta, out_meta)

    n_issued = meta["n_issued"]
    n_rows = len(columns["uuid"])
    assert n_rows == n_issued, f"row count {n_rows} != n_issued {n_issued}"
    n_first = sum(v is not None for v in columns["first_token_ns"])
    n_complete = meta["n_complete"]
    bad_lifetime = sum(1 for v in columns["lifetime_ns"] if v is not None and v < 0)
    print(
        json.dumps(
            {
                "events": events_path,
                "out_parquet": out_parquet,
                "out_meta": out_meta,
                "n_issued": n_issued,
                "n_rows": n_rows,
                "n_first_token": n_first,
                "n_complete": n_complete,
                "super_pass_size": meta["super_pass_size"],
                "negative_lifetime_rows": bad_lifetime,
                "out_tokens_present": meta["out_tokens_present"],
                "dataset_idx_mapped": meta["dataset_idx_mapped"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
