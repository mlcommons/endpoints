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

import asyncio
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    SessionState,
)
from inference_endpoint.commands.benchmark.pipeline import (
    MetricsPipeline,
    _build_report_from_snapshot,
    _load_final_snapshot_from_disk,
)
from inference_endpoint.config.schema import LoadPatternType
from inference_endpoint.metrics.report import Report

# Dotted path for monkeypatch string targets — keeps a single import style (the
# module is imported once via ``from ... import`` and patched by path here).
_PIPE = "inference_endpoint.commands.benchmark.pipeline"


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


def _fake_config(*, poisson: bool = False, use_legacy: bool = False) -> SimpleNamespace:
    """Minimal stand-in exposing only what ``_build_report_from_snapshot`` reads."""
    return SimpleNamespace(
        settings=SimpleNamespace(
            load_pattern=SimpleNamespace(
                type=(
                    LoadPatternType.POISSON
                    if poisson
                    else LoadPatternType.MAX_THROUGHPUT
                ),
                use_legacy_loadgen_qps_metrics=use_legacy,
            ),
            runtime=SimpleNamespace(scheduler_random_seed=1, dataloader_random_seed=2),
            model_dump=lambda **kw: {"load_pattern": {}, "warmup": {}},
        )
    )


@pytest.mark.unit
class TestBuildReportFromSnapshot:
    """``_build_report_from_snapshot`` warning branches + swallow-to-None.

    ``Report.from_snapshot`` is stubbed so the branches under test (which key off
    the returned Report) are exercised without a fully valid snapshot payload.
    """

    def test_incomplete_report_logs_warning(self, monkeypatch, caplog):
        fake_report = SimpleNamespace(
            complete=False, state="complete", legacy_loadgen_window_duration_ns=None
        )
        monkeypatch.setattr(
            f"{_PIPE}.Report",
            SimpleNamespace(from_snapshot=lambda *a, **k: fake_report),
        )
        with caplog.at_level("WARNING"):
            out = _build_report_from_snapshot(
                _snapshot_dict(n_pending_tasks=3), _fake_config()
            )
        assert out is fake_report
        assert any("incomplete" in r.message.lower() for r in caplog.records)

    def test_legacy_loadgen_qps_warning(self, monkeypatch, caplog):
        fake_report = SimpleNamespace(
            complete=True,
            state="complete",
            legacy_loadgen_window_duration_ns=5_000_000_000,
        )
        monkeypatch.setattr(
            f"{_PIPE}.Report",
            SimpleNamespace(from_snapshot=lambda *a, **k: fake_report),
        )
        with caplog.at_level("WARNING"):
            out = _build_report_from_snapshot(
                _snapshot_dict(), _fake_config(poisson=True, use_legacy=True)
            )
        assert out is fake_report
        assert any("legacy" in r.message.lower() for r in caplog.records)

    def test_malformed_snapshot_swallowed_to_none(self, monkeypatch, caplog):
        def _boom(*a, **k):
            raise ValueError("bad snapshot")

        monkeypatch.setattr(f"{_PIPE}.Report", SimpleNamespace(from_snapshot=_boom))
        with caplog.at_level("WARNING"):
            out = _build_report_from_snapshot(_snapshot_dict(), _fake_config())
        assert out is None
        assert any(
            "Failed to build report from snapshot" in r.message for r in caplog.records
        )


def _make_pipe(tmp_path: Path) -> MetricsPipeline:
    return MetricsPipeline(
        MagicMock(),
        tokenizer_name=None,
        enable_streaming=False,
        event_log_dir=tmp_path / "events",
        metrics_output_dir=tmp_path / "metrics",
        loop=asyncio.get_event_loop(),
    )


@pytest.mark.unit
class TestDrainFallback:
    """``drain_and_build_report`` snapshot-sourcing fallback — the SIGKILL/OOM
    recovery path where ``final_snapshot.json`` never made it to disk."""

    @pytest.mark.asyncio
    async def test_falls_back_to_subscriber_when_no_disk_snapshot(
        self, tmp_path, monkeypatch, caplog
    ):
        pipe = _make_pipe(tmp_path)  # metrics dir has no final_snapshot.json
        pipe.publisher = MagicMock(buffered_count=0, pending_count=0)
        pipe._launcher = MagicMock()
        pipe.subscriber = MagicMock(latest=object())
        monkeypatch.setattr(f"{_PIPE}.snapshot_to_dict", lambda s: {"k": "v"})
        monkeypatch.setattr(
            f"{_PIPE}._build_report_from_snapshot", lambda d, c: "REPORT"
        )
        with caplog.at_level("WARNING"):
            report = await pipe.drain_and_build_report()
        assert report == "REPORT"
        assert any(
            "No final_snapshot.json on disk" in r.message for r in caplog.records
        )
        # Drained cleanly ⇒ publisher nulled (the signal __aexit__ reads).
        assert pipe.publisher is None

    @pytest.mark.asyncio
    async def test_no_snapshot_available_returns_none(self, tmp_path, caplog):
        pipe = _make_pipe(tmp_path)
        pipe.publisher = MagicMock(buffered_count=0, pending_count=0)
        pipe._launcher = MagicMock()
        pipe.subscriber = MagicMock(latest=None)  # no disk, no live snapshot
        with caplog.at_level("ERROR"):
            report = await pipe.drain_and_build_report()
        assert report is None
        assert any("No metrics snapshot available" in r.message for r in caplog.records)


@pytest.mark.unit
class TestAexitKillPolicy:
    """``__aexit__`` kills the service subprocesses iff the run never drained. A
    clean drain nulls ``pipe.publisher``; killing then would tear down a still-
    writing aggregator. The negative arm (clean drain ⇒ no kill) is the core
    invariant of the async-context-manager rewrite and must be pinned."""

    @pytest.mark.asyncio
    async def test_clean_drain_does_not_kill(self, tmp_path):
        pipe = _make_pipe(tmp_path)
        pipe._stack = contextlib.ExitStack()
        pipe._launcher = MagicMock()
        pipe.publisher = None  # drain_and_build_report nulled it on a clean drain
        pipe.subscriber = None
        await pipe.__aexit__(None, None, None)
        pipe._launcher.kill_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_never_drained_kills_once(self, tmp_path):
        pipe = _make_pipe(tmp_path)
        pipe._stack = contextlib.ExitStack()
        pipe._launcher = MagicMock()
        pipe.publisher = MagicMock()  # still set ⇒ setup/session error before drain
        pipe.subscriber = None
        await pipe.__aexit__(None, None, None)
        pipe._launcher.kill_all.assert_called_once()
