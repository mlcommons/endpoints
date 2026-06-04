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

"""Tests for the consumer-side final-snapshot read path in
``commands/benchmark/execute.py``.

The Report consumer reads ``final_snapshot.json`` as the primary source
and falls back to the pub/sub subscriber's ``latest`` only if the file
is missing (the aggregator was killed by an uncatchable signal before
its handler ran). These tests pin both branches plus the
malformed-file behavior, since this is the load-bearing path for the
"JSON file is the canonical Report source" architecture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    SessionState,
)
from inference_endpoint.commands.benchmark.execute import (
    _load_final_snapshot_from_disk,
)
from inference_endpoint.metrics.report import Report


def _snapshot_dict(
    *,
    state: str = SessionState.COMPLETE.value,
    n_pending_tasks: int = 0,
    n_completed: int = 5,
    duration_ns: int = 10_000_000_000,
) -> dict:
    """Build a minimal valid snapshot dict shaped like ``snapshot_to_dict``."""
    return {
        "counter": 1,
        "timestamp_ns": 12345,
        "state": state,
        "n_pending_tasks": n_pending_tasks,
        "metrics": [
            {
                "type": "counter",
                "name": "tracked_samples_completed",
                "value": n_completed,
            },
            {
                "type": "counter",
                "name": "tracked_samples_issued",
                "value": n_completed,
            },
            {
                "type": "counter",
                "name": "tracked_duration_ns",
                "value": duration_ns,
            },
            {
                "type": "counter",
                "name": "tracked_samples_failed",
                "value": 0,
            },
        ],
    }


@pytest.mark.unit
class TestLoadFinalSnapshotFromDisk:
    def test_returns_none_if_file_missing(self, tmp_path: Path):
        """SIGKILL / OOM-kill case: aggregator died before signal handler
        could write. Loader returns None so the caller can fall back to
        the live subscriber."""
        missing = tmp_path / "does_not_exist.json"
        assert _load_final_snapshot_from_disk(missing) is None

    def test_reads_valid_json_as_dict(self, tmp_path: Path):
        target = tmp_path / "final_snapshot.json"
        target.write_text(json.dumps(_snapshot_dict()))
        loaded = _load_final_snapshot_from_disk(target)
        assert loaded is not None
        assert loaded["state"] == SessionState.COMPLETE.value
        assert loaded["n_pending_tasks"] == 0

    def test_returns_none_on_malformed_json(self, tmp_path: Path, caplog):
        """A truncated / corrupt file MUST NOT crash the Report build —
        the caller falls back to the live subscriber and the report is
        marked incomplete. A warning is logged so the failure is visible."""
        target = tmp_path / "final_snapshot.json"
        target.write_bytes(b"{not valid json")
        with caplog.at_level("WARNING"):
            result = _load_final_snapshot_from_disk(target)
        assert result is None
        assert any("Failed to read final snapshot" in r.message for r in caplog.records)


@pytest.mark.unit
class TestReportFromLoadedSnapshot:
    """End-to-end: load JSON → build Report. Pins the
    state→complete-flag→display-warning contract that the consumer
    relies on across the three terminal states."""

    @pytest.mark.parametrize(
        "state, n_pending, expected_complete",
        [
            (SessionState.COMPLETE.value, 0, True),
            # Drain-timeout: COMPLETE state but tasks still pending.
            (SessionState.COMPLETE.value, 3, False),
            # Interrupted: signal-handler-written snapshot.
            (SessionState.INTERRUPTED.value, 0, False),
            (SessionState.INTERRUPTED.value, 7, False),
        ],
    )
    def test_report_complete_flag_matches_state_and_pending(
        self, tmp_path: Path, state: str, n_pending: int, expected_complete: bool
    ):
        target = tmp_path / "final_snapshot.json"
        target.write_text(
            json.dumps(_snapshot_dict(state=state, n_pending_tasks=n_pending))
        )
        loaded = _load_final_snapshot_from_disk(target)
        assert loaded is not None
        report = Report.from_snapshot(loaded)
        assert report.state == state
        assert report.complete is expected_complete

    def test_interrupted_display_surfaces_signal_warning(self, tmp_path: Path):
        """An INTERRUPTED snapshot loaded from disk produces a Report
        whose ``display()`` prominently calls out the signal-driven
        shutdown — so a user reading the output knows the data is
        partial, not just incomplete."""
        target = tmp_path / "final_snapshot.json"
        target.write_text(
            json.dumps(_snapshot_dict(state=SessionState.INTERRUPTED.value))
        )
        report = Report.from_snapshot(_load_final_snapshot_from_disk(target) or {})
        lines: list[str] = []
        report.display(fn=lines.append, summary_only=True)
        output = "\n".join(lines)
        # Must surface the signal cause explicitly.
        assert "interrupted" in output.lower()
        assert "SIGTERM" in output or "signal" in output.lower()

    def test_missing_file_path_fallback_yields_no_loaded_snapshot(self, tmp_path: Path):
        """The contract the caller in execute.py relies on: missing file
        → None → caller switches to live-snapshot fallback. This pins
        the precondition the fallback chain depends on."""
        result = _load_final_snapshot_from_disk(tmp_path / "nope.json")
        assert result is None
