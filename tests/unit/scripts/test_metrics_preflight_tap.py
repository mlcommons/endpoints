# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the experiment-only GPT-OSS metrics preflight tap."""

# ruff: noqa: I001
# Keep import layout stable across the pinned pre-commit and local uv ruff.

from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    CounterStat,
    MetricsSnapshot,
    MetricsSnapshotCodec,
    SessionState,
)
from inference_endpoint.core.record import TOPIC_FRAME_SIZE

pytestmark = pytest.mark.unit


def _load_tap():
    path = Path("scratchpad/gptoss_nvl144_pr334_vvv_20260728/metrics_preflight_tap.py")
    spec = importlib.util.spec_from_file_location("metrics_preflight_tap", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tap = _load_tap()


def _write_process(
    proc_root: Path,
    *,
    pid: int,
    ppid: int,
    argv: list[str],
    status: str = "VmRSS:\t123 kB\nVmHWM:\t456 kB\n",
) -> None:
    proc_dir = proc_root / str(pid)
    proc_dir.mkdir()
    (proc_dir / "stat").write_text(f"{pid} (command with spaces) S {ppid} 0 0\n")
    (proc_dir / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    (proc_dir / "status").write_text(status)


def _snapshot(
    counter: int,
    *,
    state: SessionState = SessionState.LIVE,
    pending: int = 0,
    issued: int = 10,
    completed: int = 5,
) -> MetricsSnapshot:
    return MetricsSnapshot(
        counter=counter,
        timestamp_ns=counter * 100,
        state=state,
        n_pending_tasks=pending,
        metrics=[
            CounterStat("total_samples_issued", issued),
            CounterStat("total_samples_completed", completed),
            CounterStat("total_samples_failed", 0),
        ],
    )


class TestProcessDiscovery:
    def test_finds_only_aggregator_below_root(self, tmp_path: Path) -> None:
        proc = tmp_path / "proc"
        proc.mkdir()
        _write_process(proc, pid=100, ppid=1, argv=["benchmark"])
        _write_process(proc, pid=101, ppid=100, argv=["worker"])
        _write_process(
            proc,
            pid=102,
            ppid=101,
            argv=[
                "python",
                "-m",
                tap.AGGREGATOR_MODULE,
                "--socket-dir",
                "/dev/shm/zmq_a",
                "--metrics-socket=metrics_a",
            ],
        )
        _write_process(
            proc,
            pid=200,
            ppid=1,
            argv=["python", "-m", tap.AGGREGATOR_MODULE],
        )

        found = tap.find_aggregator_descendant(100, proc)

        assert found is not None
        assert found.pid == 102
        assert tap.parse_aggregator_socket_args(found.argv) == (
            "/dev/shm/zmq_a",
            "metrics_a",
        )
        assert (
            tap.metrics_ipc_address("/dev/shm/zmq_a", "metrics_a")
            == "ipc:///dev/shm/zmq_a/metrics_a"
        )

    def test_missing_socket_arg_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="metrics-socket"):
            tap.parse_aggregator_socket_args(["--socket-dir", "/tmp/x"])


class TestSampling:
    def test_reads_proc_cgroup_meminfo_and_tmpfs(self, tmp_path: Path) -> None:
        proc = tmp_path / "proc"
        proc.mkdir()
        _write_process(proc, pid=42, ppid=1, argv=["aggregator"])
        (proc / "42" / "cgroup").write_text("0::/job/step\n")
        (proc / "meminfo").write_text(
            "MemTotal:       10000 kB\nMemAvailable:    2500 kB\n"
        )

        cgroup_root = tmp_path / "cgroup"
        cgroup = cgroup_root / "job" / "step"
        cgroup.mkdir(parents=True)
        (cgroup / "memory.current").write_text("1000\n")
        (cgroup / "memory.peak").write_text("2000\n")
        (cgroup / "memory.max").write_text("3000\n")
        (cgroup / "memory.events").write_text("oom 2\noom_kill 1\n")

        events = tmp_path / "benchmark_1" / "events"
        events.mkdir(parents=True)
        (events / "events.jsonl").write_bytes(b"x" * 17)

        location = tap._find_cgroup(42, proc, cgroup_root)
        obs = tap.sample_memory(
            42,
            location,
            str(tmp_path / "benchmark_*" / "events" / "events.jsonl"),
            proc,
        )

        assert obs == tap.MemoryObservation(
            aggregator_alive=True,
            rss_kib=123,
            hwm_kib=456,
            cgroup_current_bytes=1000,
            cgroup_peak_bytes=2000,
            cgroup_max_bytes=3000,
            cgroup_oom=2,
            cgroup_oom_kill=1,
            mem_available_kib=2500,
            mem_total_kib=10000,
            tmpfs_event_files=1,
            tmpfs_events_bytes=17,
        )


class TestSnapshotsAndArtifacts:
    def test_decodes_frame_and_tracks_pending_memory_high_water(
        self, tmp_path: Path
    ) -> None:
        codec = MetricsSnapshotCodec()
        first = _snapshot(1, pending=3)
        second = _snapshot(
            4,
            state=SessionState.DRAINING,
            pending=7,
            issued=20,
            completed=20,
        )
        topic, payload = codec.encode(first)
        assert len(topic) == TOPIC_FRAME_SIZE
        assert tap.decode_metrics_frame(topic + payload, codec) == first

        stats = tap.MonitorStats(
            started_wall_ns=100,
            started_monotonic_ns=100,
            root_pid=1,
            aggregator_pid=42,
        )
        stats.observe_snapshot(first, 1_000_000_000)
        stats.observe_snapshot(second, 3_500_000_000)
        stats.observe_memory(
            tap.MemoryObservation(
                aggregator_alive=True,
                rss_kib=11,
                hwm_kib=12,
                cgroup_current_bytes=13,
                cgroup_peak_bytes=14,
                cgroup_max_bytes=15,
                cgroup_oom=0,
                cgroup_oom_kill=0,
                mem_available_kib=16,
                mem_total_kib=17,
                tmpfs_event_files=1,
                tmpfs_events_bytes=18,
            )
        )

        summary = stats.to_dict(ended_wall_ns=4_000_000_000, csv_path=tmp_path / "x")
        assert summary["published_pending_high_water"] == 7
        assert summary["pending_at_first_draining"] == 7
        assert summary["counter_gap_total"] == 2
        assert summary["counter_gap_max"] == 2
        assert summary["max_snapshot_gap_s"] == 2.5
        assert summary["aggregator_rss_high_water_kib"] == 11
        assert summary["tmpfs_events_high_water_bytes"] == 18
        assert summary["telemetry_capture_valid"] is True
        assert summary["telemetry_capture_failures"] == []

    def test_capture_gate_requires_rss_and_oom_counters(self) -> None:
        stats = tap.MonitorStats(
            started_wall_ns=100,
            started_monotonic_ns=100,
            root_pid=1,
            aggregator_pid=42,
            cgroup_version=2,
            snapshots_received=2,
            published_pending_high_water=7,
            aggregator_reported_hwm_high_water_kib=12,
            cgroup_memory_current_high_water_bytes=13,
            cgroup_memory_peak_high_water_bytes=14,
        )

        assert tap.telemetry_capture_failures(stats) == [
            "aggregator_rss_missing",
            "cgroup_oom_missing",
            "cgroup_oom_kill_missing",
        ]

        stats.cgroup_version = 1
        assert tap.telemetry_capture_failures(stats) == [
            "aggregator_rss_missing",
            "cgroup_oom_missing",
        ]

    def test_csv_finalization_and_atomic_summary(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "telemetry.csv"
        artifact = tap.AtomicCsv(csv_path, fsync_interval_s=0)
        artifact.open()
        row = dict.fromkeys(tap.CSV_FIELDS, "")
        row["row_kind"] = "memory"
        artifact.write(row)
        artifact.finalize()

        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["row_kind"] == "memory"
        assert not csv_path.with_suffix(".csv.part").exists()

        summary_path = tmp_path / "summary.json"
        payload = {"status": "complete"}
        from inference_endpoint.utils.atomic_write import atomic_write_bytes

        atomic_write_bytes(
            summary_path, (json.dumps(payload, sort_keys=True) + "\n").encode()
        )
        assert json.loads(summary_path.read_text()) == payload

    def test_missing_aggregator_is_nonzero_and_still_atomic(
        self, tmp_path: Path
    ) -> None:
        csv_path = tmp_path / "telemetry.csv"
        summary_path = tmp_path / "summary.json"
        args = tap._build_parser().parse_args(
            [
                "--root-pid",
                str(2**31 - 1),
                "--csv",
                str(csv_path),
                "--summary",
                str(summary_path),
                "--discover-timeout-s",
                "0",
            ]
        )

        exit_code, summary = tap.run(args)

        assert exit_code == 2
        assert summary["status"] == "aggregator_not_found"
        assert summary["telemetry_capture_valid"] is False
        assert "aggregator_not_found" in summary["telemetry_capture_failures"]
        assert csv_path.is_file()
        assert summary_path.is_file()
        assert not csv_path.with_suffix(".csv.part").exists()

    def test_end_to_end_discovers_and_taps_metrics_pub(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        socket_dir = tmp_path / "sockets"
        socket_dir.mkdir()
        socket_name = "metrics_test"
        child_code = """
import sys
import time
import zmq
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    CounterStat, MetricsSnapshot, MetricsSnapshotCodec, SessionState,
)

args = sys.argv[1:]
socket_dir = args[args.index("--socket-dir") + 1]
socket_name = args[args.index("--metrics-socket") + 1]
ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.setsockopt(zmq.LINGER, 0)
sock.bind(f"ipc://{socket_dir}/{socket_name}")
codec = MetricsSnapshotCodec()
time.sleep(0.5)
for i in range(1, 7):
    snap = MetricsSnapshot(
        counter=i,
        timestamp_ns=i,
        state=SessionState.LIVE,
        n_pending_tasks=i,
        metrics=[
            CounterStat("total_samples_issued", i),
            CounterStat("total_samples_completed", i),
            CounterStat("total_samples_failed", 0),
        ],
    )
    topic, payload = codec.encode(snap)
    sock.send(topic + payload)
    time.sleep(0.15)
sock.close(0)
ctx.term()
"""
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                child_code,
                tap.AGGREGATOR_MODULE,
                "--socket-dir",
                str(socket_dir),
                "--metrics-socket",
                socket_name,
            ]
        )
        observation = tap.MemoryObservation(
            aggregator_alive=True,
            rss_kib=100,
            hwm_kib=200,
            cgroup_current_bytes=300,
            cgroup_peak_bytes=400,
            cgroup_max_bytes=500,
            cgroup_oom=0,
            cgroup_oom_kill=0,
            mem_available_kib=600,
            mem_total_kib=700,
            tmpfs_event_files=0,
            tmpfs_events_bytes=0,
        )
        monkeypatch.setattr(tap, "sample_memory", lambda *args, **kwargs: observation)
        csv_path = tmp_path / "telemetry.csv"
        summary_path = tmp_path / "summary.json"
        args = tap._build_parser().parse_args(
            [
                "--root-pid",
                str(os.getpid()),
                "--csv",
                str(csv_path),
                "--summary",
                str(summary_path),
                "--discover-timeout-s",
                "5",
                "--discover-poll-s",
                "0.02",
                "--sample-interval-s",
                "0.05",
                "--poll-timeout-ms",
                "20",
                "--post-aggregator-exit-s",
                "0.1",
                "--fsync-interval-s",
                "0",
            ]
        )
        try:
            exit_code, summary = tap.run(args)
        finally:
            child.wait(timeout=5)

        assert exit_code == 0
        assert summary["status"] == "aggregator_exited"
        assert summary["telemetry_capture_valid"] is True
        assert summary["snapshots_received"] >= 2
        assert summary["published_pending_high_water"] >= 2
        assert summary["aggregator_reported_hwm_high_water_kib"] == 200
        assert summary["cgroup_memory_current_high_water_bytes"] == 300
        assert summary["cgroup_memory_peak_high_water_bytes"] == 400
        assert csv_path.is_file()
        assert summary_path.is_file()
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[-1]["row_kind"] == "terminal_memory"
