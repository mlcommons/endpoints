# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for ``inference_endpoint.utils.trace`` runtime helpers
(start_lag_task / start_snapshot_tap / teardown / path conventions /
emitter pipe-death). The dashboard aggregation lives in
``test_trace_dashboard.py``.
"""
# ruff: noqa: I001

from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
import time

import pytest

from inference_endpoint.utils import trace


# ---------------------------------------------------------------------------
# Path conventions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPathConventions:
    def test_fifo_path_is_per_pid_subdir(self) -> None:
        assert trace.fifo_path(12345) == "/tmp/endpoints_trace_12345/fifo"

    def test_snapshot_path_in_same_subdir(self) -> None:
        assert (
            trace.snapshot_sidecar_path(12345)
            == "/tmp/endpoints_trace_12345/snapshot.json"
        )

    def test_paths_share_per_pid_dir(self) -> None:
        # FIFO and snapshot must always be in the same per-pid dir so
        # one mkdir / one cleanup covers both.
        assert os.path.dirname(trace.fifo_path(7)) == os.path.dirname(
            trace.snapshot_sidecar_path(7)
        )


# ---------------------------------------------------------------------------
# emit_trace_id no-op guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNoOpGuard:
    def test_non_hex_id_safe_when_disabled(self) -> None:
        # Existing tests pass query ids like "q-1" / "q-stream"; the
        # emit_trace_id no-op guard must short-circuit before the hex
        # parse can raise ValueError.
        assert trace.is_enabled() is False
        trace.emit_trace_id(trace.Event.WRITTEN, "q-stream")
        trace.emit_trace_id(trace.Event.WRITTEN, "not-a-hex-string")

    def test_dashed_uuid_safe_when_disabled(self) -> None:
        trace.emit_trace_id(trace.Event.WRITTEN, "12345678-1234-1234-1234-123456789abc")


# ---------------------------------------------------------------------------
# enable_tracing / teardown
# ---------------------------------------------------------------------------


def _make_fifo_with_drain_thread() -> tuple[str, threading.Thread]:
    """Set up the convention layout (per-pid 0o700 dir + FIFO inside)
    that bootstrap() would normally create, plus a background reader
    so enable_tracing's blocking O_WRONLY open returns immediately."""
    path = trace.fifo_path(os.getpid())
    trace_dir = os.path.dirname(path)
    # Wipe any stale dir from a prior test in the same pid.
    if os.path.isdir(trace_dir):
        shutil.rmtree(trace_dir, ignore_errors=True)
    os.mkdir(trace_dir, 0o700)
    os.mkfifo(path, 0o600)
    trace._state.fifo_path = path  # so teardown unlinks like bootstrap does

    def _drain() -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            while True:
                if not os.read(fd, 4096):
                    return
        finally:
            os.close(fd)

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    time.sleep(0.05)
    return path, t


@pytest.mark.unit
class TestEnableTracing:
    def teardown_method(self) -> None:
        # Coroutines run synchronously here; an asyncio.run drives teardown.
        asyncio.run(trace.teardown())

    def test_no_op_on_missing_fifo(self) -> None:
        trace.enable_tracing("/tmp/this/does/not/exist")
        assert trace.is_enabled() is False

    def test_enable_then_teardown_idempotent(self) -> None:
        path, _ = _make_fifo_with_drain_thread()
        trace.enable_tracing(path)
        assert trace.is_enabled() is True
        # Calling enable_tracing again is a no-op (idempotent).
        trace.enable_tracing(path)
        assert trace.is_enabled() is True
        # First teardown disables; second teardown is harmless.
        asyncio.run(trace.teardown())
        assert trace.is_enabled() is False
        asyncio.run(trace.teardown())
        assert trace.is_enabled() is False


# ---------------------------------------------------------------------------
# Snapshot tap task
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSnapshotTap:
    def teardown_method(self) -> None:
        asyncio.run(trace.teardown())

    def test_tap_writes_atomic_json_then_teardown_cancels(self) -> None:
        path, _ = _make_fifo_with_drain_thread()
        trace.enable_tracing(path)
        snap_path = trace.snapshot_sidecar_path(os.getpid())

        async def _run() -> None:
            loop = asyncio.get_running_loop()
            provider_calls = {"n": 0}

            def provider() -> dict | None:
                provider_calls["n"] += 1
                return {"hello": provider_calls["n"]}

            trace.start_snapshot_tap(loop, provider, period_s=0.05)
            await asyncio.sleep(0.15)  # ≥ 2 ticks
            # File should exist with the latest provider payload.
            assert os.path.exists(snap_path)
            with open(snap_path) as f:
                blob = json.load(f)
            assert blob["hello"] >= 2
            assert provider_calls["n"] >= 2
            # teardown cancels the running task.
            await trace.teardown()

        asyncio.run(_run())
        assert trace.is_enabled() is False

    def test_provider_returning_none_skips_write(self) -> None:
        path, _ = _make_fifo_with_drain_thread()
        trace.enable_tracing(path)
        snap_path = trace.snapshot_sidecar_path(os.getpid())
        # Pre-remove any leftover sidecar.
        try:
            os.unlink(snap_path)
        except FileNotFoundError:
            pass

        async def _run() -> None:
            loop = asyncio.get_running_loop()
            trace.start_snapshot_tap(loop, lambda: None, period_s=0.05)
            await asyncio.sleep(0.12)
            assert not os.path.exists(snap_path)
            await trace.teardown()

        asyncio.run(_run())

    def test_start_when_disabled_is_no_op(self) -> None:
        async def _run() -> None:
            loop = asyncio.get_running_loop()
            # Tracing not enabled → no task is spawned, no exception.
            trace.start_snapshot_tap(loop, lambda: {"x": 1})
            await asyncio.sleep(0.01)
            await trace.teardown()

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Loop-lag task
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoopLagTask:
    def teardown_method(self) -> None:
        asyncio.run(trace.teardown())

    def test_start_when_disabled_is_no_op(self) -> None:
        async def _run() -> None:
            loop = asyncio.get_running_loop()
            trace.start_lag_task(loop)
            await asyncio.sleep(0.01)
            await trace.teardown()

        asyncio.run(_run())

    def test_start_when_enabled_creates_task(self) -> None:
        path, _ = _make_fifo_with_drain_thread()
        trace.enable_tracing(path)

        async def _run() -> None:
            loop = asyncio.get_running_loop()
            trace.start_lag_task(loop)
            # One task registered → teardown will cancel it.
            assert len(trace._state.tasks) == 1
            await trace.teardown()
            assert trace._state.tasks == []

        asyncio.run(_run())


# ---------------------------------------------------------------------------
# Teardown final-snapshot write
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTeardownFinalSnapshot:
    def teardown_method(self) -> None:
        asyncio.run(trace.teardown())

    def test_writes_final_snapshot_when_passed(self) -> None:
        path, _ = _make_fifo_with_drain_thread()
        trace.enable_tracing(path)
        snap_path = trace.snapshot_sidecar_path(os.getpid())
        try:
            os.unlink(snap_path)
        except FileNotFoundError:
            pass

        payload = {"final": True, "samples": 42}
        asyncio.run(trace.teardown(final_snapshot=payload))

        # File should reflect the passed dict, not whatever a tap
        # would have produced.
        with open(snap_path) as f:
            assert json.load(f) == payload

    def test_no_op_final_snapshot_when_disabled(self) -> None:
        # Without enable_tracing, fifo_path state isn't set; teardown
        # silently no-ops on the final-write path.
        asyncio.run(trace.teardown(final_snapshot={"ignored": True}))
        assert trace.is_enabled() is False


@pytest.mark.unit
class TestSyncCleanup:
    """``cleanup()`` runs from sync contexts (e.g. main.py's launcher
    finally block when bootstrap fired but the loop never started)."""

    def teardown_method(self) -> None:
        trace.cleanup()  # idempotent reset

    def test_idempotent_when_never_enabled(self) -> None:
        trace.cleanup()
        trace.cleanup()
        assert trace.is_enabled() is False

    def test_unlinks_fifo_and_disables_emitter(self) -> None:
        path, _ = _make_fifo_with_drain_thread()
        trace.enable_tracing(path)
        assert trace.is_enabled() is True
        assert os.path.exists(path)
        trace.cleanup()
        assert trace.is_enabled() is False
        assert not os.path.exists(path)
