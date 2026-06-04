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
import contextlib
import fcntl
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


@pytest.mark.unit
class TestAdaptiveSampling:
    """emit_trace_id drops events for non-sampled sids when _sample_shift
    > 0; at shift 0 (the default) everything passes."""

    def test_shift_zero_emits_everything(self, monkeypatch) -> None:
        seen: list[int] = []
        monkeypatch.setattr(trace, "emit_trace", lambda ev, sid: seen.append(sid))
        monkeypatch.setattr(trace, "_sample_shift", 0)
        for low in range(8):
            trace.emit_trace_id(trace.Event.ISSUED, f"{low:016x}" + "0" * 16)
        assert len(seen) == 8

    def test_shift_gates_by_low_bits(self, monkeypatch) -> None:
        # shift=2 → mask 0b11 → only sids with low 2 bits == 0 emit.
        seen: list[int] = []
        monkeypatch.setattr(trace, "emit_trace", lambda ev, sid: seen.append(sid))
        monkeypatch.setattr(trace, "_sample_shift", 2)
        for low in range(8):  # sid = low (top 48 bits zero)
            trace.emit_trace_id(trace.Event.ISSUED, f"{low:016x}" + "0" * 16)
        assert seen == [0, 4]  # 0,4 have low 2 bits clear; 1,2,3,5,6,7 gated

    def test_power_of_two_subset_is_consistent(self, monkeypatch) -> None:
        # A coarser sampler's kept set ⊆ a finer one's — so a request
        # fully traced at the max shift is emitted by every process.
        def kept(shift: int) -> set[int]:
            seen: list[int] = []
            monkeypatch.setattr(trace, "emit_trace", lambda ev, sid: seen.append(sid))
            monkeypatch.setattr(trace, "_sample_shift", shift)
            for low in range(64):
                trace.emit_trace_id(trace.Event.ISSUED, f"{low:016x}" + "0" * 16)
            return set(seen)

        assert kept(4).issubset(kept(2))
        assert kept(2).issubset(kept(0))


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
            pass  # no leftover sidecar from a prior run — nothing to remove

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
            pass  # no leftover sidecar from a prior run — nothing to remove

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


# ---------------------------------------------------------------------------
# _TraceEmitter (the SPSC ring → FIFO writer)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTraceEmitter:
    def teardown_method(self) -> None:
        # _die() mutates the module-level emit bindings; restore the disabled
        # defaults so a pipe-death test can't leak into later tests.
        trace.emit_trace = trace._noop
        trace._active_emitter = None

    def test_emit_then_flush_writes_decodable_frames(self) -> None:
        r, w = os.pipe()
        try:
            em = trace._TraceEmitter(w)
            em.flush()  # nothing buffered → must not write or block (early-out)
            em.emit(trace.Event.ISSUED, 0x1234)
            em.emit(trace.Event.COMPLETE, 0x5678)
            em.flush()
            data = os.read(r, trace.FRAME_SIZE * 2)
            assert len(data) == trace.FRAME_SIZE * 2
            ev0, sid0, ts0 = trace.PACKER.unpack_from(data, 0)
            ev1, sid1, _ = trace.PACKER.unpack_from(data, trace.FRAME_SIZE)
            assert (ev0, sid0) == (int(trace.Event.ISSUED), 0x1234)
            assert (ev1, sid1) == (int(trace.Event.COMPLETE), 0x5678)
            assert ts0 > 0  # monotonic_ns stamped at emit
            assert em._offset == 0  # reset after flush
        finally:
            os.close(r)
            with contextlib.suppress(OSError):
                os.close(w)

    def test_ring_overflow_drops_and_accounts(self) -> None:
        r, w = os.pipe()
        try:
            em = trace._TraceEmitter(w)
            em._offset = trace._BUF_CAPACITY  # simulate a full ring this cycle
            em.emit(trace.Event.ISSUED, 1)
            assert em.dropped_bytes() == trace.FRAME_SIZE
            assert em._offset == trace._BUF_CAPACITY  # unchanged — frame dropped
        finally:
            os.close(r)
            os.close(w)

    def test_eagain_when_reader_behind_drops_the_rest(self) -> None:
        # Non-blocking write end whose reader never drains: once the pipe
        # fills, os.write raises EAGAIN and flush() accounts the undelivered
        # tail (whole frames) as dropped rather than blocking. Size the emit
        # to 3x the *actual* pipe capacity so overflow is deterministic
        # regardless of the host's default/clamped pipe size.
        r, w = os.pipe()
        os.set_blocking(w, False)
        with contextlib.suppress(OSError):
            fcntl.fcntl(w, fcntl.F_SETPIPE_SZ, 4096)  # shrink to one page
        capacity = fcntl.fcntl(w, fcntl.F_GETPIPE_SZ)
        try:
            em = trace._TraceEmitter(w)
            for i in range(3 * capacity // trace.FRAME_SIZE):
                em.emit(trace.Event.ISSUED, i)
            em.flush()
            dropped = em.dropped_bytes()
            assert dropped > 0
            assert dropped % trace.FRAME_SIZE == 0  # frame-aligned tail, not torn
        finally:
            os.close(r)
            with contextlib.suppress(OSError):
                os.close(w)

    def test_pipe_death_marks_dead_and_disables(self) -> None:
        r, w = os.pipe()
        os.close(r)  # reader gone → write raises EPIPE (OSError, not EAGAIN)
        em = trace._TraceEmitter(w)
        trace._active_emitter = em  # so _die's global reset is observable
        em.emit(trace.Event.ISSUED, 1)
        em.flush()  # EPIPE → _die()
        assert em._dead is True
        assert trace.is_enabled() is False  # _die() cleared _active_emitter
        # Further calls are inert no-ops (fd already closed by _die).
        em.emit(trace.Event.COMPLETE, 2)
        em.flush()


# ---------------------------------------------------------------------------
# Adaptive-sampling policy (pure) + the tick loop that drives it
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSampleShiftPolicy:
    """Pure drop-driven sampling state machine: raise fast on any new drops
    (capped), ease off one step only after sustained clear (floored at 0)."""

    def test_raises_one_step_per_drop_tick_capped_at_max(self) -> None:
        m = trace._MAX_SAMPLE_SHIFT
        assert trace._next_sample_shift(0, 0, new_drops=True) == (1, 0)
        assert trace._next_sample_shift(3, 7, new_drops=True) == (4, 0)  # clear reset
        assert trace._next_sample_shift(m, 0, new_drops=True) == (m, 0)  # capped

    def test_recovers_only_after_sustained_clear_and_floors_at_zero(self) -> None:
        rec = trace._SAMPLE_RECOVER_TICKS
        assert trace._next_sample_shift(2, 0, new_drops=False) == (2, 1)  # accumulate
        assert trace._next_sample_shift(2, rec - 1, new_drops=False) == (
            1,
            0,
        )  # ease off
        assert trace._next_sample_shift(0, rec - 1, new_drops=False) == (
            0,
            rec,
        )  # floor


@pytest.mark.unit
class TestEmitLoopLag:
    """The per-process tick: a completed tick emits LOOP_LAG (tagged with the
    worker id), re-emits cumulative drops, and drives the sampling shift; a
    cancellation does a final drop-flush before returning."""

    def teardown_method(self) -> None:
        trace.emit_trace = trace._noop
        trace._active_emitter = None
        trace._sample_shift = 0

    class _StubEmitter:
        def __init__(self) -> None:
            self.flushed = 0

        def dropped_bytes(self) -> int:
            return 64  # non-zero, monotonically "new" vs the initial 0

        def flush(self) -> None:
            self.flushed += 1

    def test_completed_tick_emits_lag_and_drops_and_raises_shift(self) -> None:
        emitted: list[tuple[int, int]] = []
        stub = self._StubEmitter()

        async def _run() -> None:
            flushed = asyncio.Event()
            stub_flush = stub.flush

            def _flush() -> None:
                stub_flush()
                flushed.set()

            stub.flush = _flush  # type: ignore[method-assign]
            trace._active_emitter = stub  # type: ignore[assignment]
            trace.emit_trace = lambda event, sid: emitted.append((event, sid))
            task = asyncio.get_running_loop().create_task(
                trace.emit_loop_lag(7, period_s=0.001)
            )
            await asyncio.wait_for(flushed.wait(), timeout=2.0)  # ≥1 full tick
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        asyncio.run(_run())
        lag_sids = [s for ev, s in emitted if ev == int(trace.Event.LOOP_LAG)]
        assert lag_sids and all((s >> 56) & 0xFF == 7 for s in lag_sids)
        assert int(trace.Event.TRACE_DROPS) in [ev for ev, _ in emitted]
        assert 0 < trace._sample_shift <= trace._MAX_SAMPLE_SHIFT

    def test_cancellation_does_a_final_drop_flush(self) -> None:
        # A long period means no tick completes before we cancel, so the only
        # flush + drop-emit must come from the CancelledError handler.
        emitted: list[tuple[int, int]] = []
        stub = self._StubEmitter()

        async def _run() -> None:
            trace._active_emitter = stub  # type: ignore[assignment]
            trace.emit_trace = lambda event, sid: emitted.append((event, sid))
            task = asyncio.get_running_loop().create_task(
                trace.emit_loop_lag(0, period_s=30.0)
            )
            await asyncio.sleep(0)  # let it reach the long sleep
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        asyncio.run(_run())
        assert stub.flushed == 1  # exactly the cancel-handler flush
        events = [ev for ev, _ in emitted]
        assert int(trace.Event.TRACE_DROPS) in events  # final drop re-emit
        assert int(trace.Event.LOOP_LAG) not in events  # no tick completed


# ---------------------------------------------------------------------------
# bootstrap (verbose → level + trace setup)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBootstrap:
    """``bootstrap`` maps -v/-vv/-vvv to a log level and, at -vvv, stands up
    the FIFO + dashboard. We exercise the level mapping and the two early
    bail-outs (no dashboard script / dashboard died) which don't touch the
    process's stdout/stderr fds."""

    def teardown_method(self) -> None:
        trace.cleanup()
        shutil.rmtree(os.path.dirname(trace.fifo_path(os.getpid())), ignore_errors=True)

    @pytest.mark.parametrize(
        ("verbose", "level"), [(0, "INFO"), (1, "INFO"), (2, "DEBUG")]
    )
    def test_sub_trace_verbose_maps_to_level_without_enabling(
        self, verbose: int, level: str
    ) -> None:
        assert trace.bootstrap(verbose) == level
        assert trace.is_enabled() is False

    @pytest.mark.parametrize("dashboard_unavailable", ["missing", "died"])
    def test_vvv_without_a_live_dashboard_does_not_enable_tracing(
        self, monkeypatch, dashboard_unavailable: str
    ) -> None:
        # Both bail-outs (script absent / dashboard exited during the grace
        # window) must create the FIFO but leave tracing OFF — a blocking
        # O_WRONLY open with no reader would otherwise deadlock the run.
        if dashboard_unavailable == "missing":
            monkeypatch.setattr(trace, "_spawn_dashboard", lambda _path: None)
        else:

            class _DeadProc:
                returncode = 2

                def poll(self) -> int:
                    return 2  # already exited

            monkeypatch.setattr(trace, "_spawn_dashboard", lambda _path: _DeadProc())
            monkeypatch.setattr(trace, "_DASHBOARD_READY_S", 0.0)  # skip real sleep
        assert trace.bootstrap(3) == "TRACE"
        assert os.path.exists(trace.fifo_path(os.getpid()))
        assert trace.is_enabled() is False
