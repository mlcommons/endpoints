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

"""Tests for the snapshot wire schema and codec."""

from __future__ import annotations

import msgspec
import msgspec.msgpack
import pytest
from inference_endpoint.async_utils.services.metrics_aggregator.snapshot import (
    METRICS_SNAPSHOT_TOPIC,
    CounterStat,
    MetricsSnapshot,
    MetricsSnapshotCodec,
    SeriesStat,
    SessionState,
    snapshot_to_dict,
)
from inference_endpoint.core.record import TOPIC_FRAME_SIZE
from inference_endpoint.metrics.steady_state_diagnostics import (
    Anomaly,
    SteadyState,
    SteadyWindow,
    TpsBlock,
)


@pytest.mark.unit
class TestCounterStat:
    def test_roundtrip(self):
        stat = CounterStat(name="total_samples_issued", value=42)
        encoded = msgspec.msgpack.encode(stat)
        decoded = msgspec.msgpack.decode(encoded, type=CounterStat)
        assert decoded == stat

    def test_float_value(self):
        stat = CounterStat(name="duration_s", value=3.14)
        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(stat), type=CounterStat)
        assert decoded.value == pytest.approx(3.14)


@pytest.mark.unit
class TestSeriesStat:
    @pytest.mark.parametrize(
        "sum_sq",
        [
            55000,
            # Float exceeding uint64 max — would overflow msgpack if encoded as int.
            2.0 * (2**64 - 1),
        ],
    )
    def test_roundtrip(self, sum_sq):
        stat = SeriesStat(
            name="ttft_ns",
            count=5,
            total=500,
            min=50,
            max=150,
            sum_sq=sum_sq,
            percentiles={"50": 100.0, "99": 145.0},
            histogram=[((50.0, 100.0), 2), ((100.0, 150.0), 3)],
        )
        encoded = msgspec.msgpack.encode(stat)
        decoded = msgspec.msgpack.decode(encoded, type=SeriesStat)
        assert decoded == stat


@pytest.mark.unit
class TestMetricsSnapshot:
    def test_empty_metrics_roundtrip(self):
        snap = MetricsSnapshot(
            counter=1,
            timestamp_ns=1234,
            state=SessionState.LIVE,
            n_pending_tasks=0,
            metrics=[],
        )
        codec = MetricsSnapshotCodec()
        topic, payload = codec.encode(snap)
        assert topic == METRICS_SNAPSHOT_TOPIC
        assert len(topic) == TOPIC_FRAME_SIZE
        decoded = codec.decode(payload)
        assert decoded == snap

    def test_tagged_union_dispatch(self):
        """Decoder must produce the right concrete type per tag."""
        snap = MetricsSnapshot(
            counter=2,
            timestamp_ns=42,
            state=SessionState.COMPLETE,
            n_pending_tasks=3,
            metrics=[
                CounterStat(name="c1", value=10),
                SeriesStat(
                    name="s1",
                    count=1,
                    total=10,
                    min=10,
                    max=10,
                    sum_sq=100,
                    percentiles={"50": 10.0},
                    histogram=[((1.0, 10.0), 1)],
                ),
            ],
        )
        codec = MetricsSnapshotCodec()
        _, payload = codec.encode(snap)
        decoded = codec.decode(payload)
        assert isinstance(decoded.metrics[0], CounterStat)
        assert isinstance(decoded.metrics[1], SeriesStat)
        assert decoded.metrics[0].name == "c1"
        assert decoded.metrics[1].name == "s1"

    def test_steady_state_survives_the_wire_and_the_dict_form(self):
        verdict = SteadyState(
            found=True,
            reason=None,
            superpass_size=10,
            n_super_passes=4,
            warmup=0,
            window=SteadyWindow(
                sp_lo=0,
                sp_hi=4,
                n_super_passes=4,
                n_samples=40,
                start_ns=1,
                end_ns=2,
                plateau_index=0,
                n_plateaus=1,
                skipped_short=0,
            ),
            ttft=None,
            tpot=None,
            osl=None,
            latency=None,
            tps=TpsBlock(
                per_user=0.25, per_user_ci=[0.2, 0.3], system=1.5, system_ci=[1.4, 1.6]
            ),
            cov={},
            cov_basis=None,
            anomaly=Anomaly(
                detected=False,
                change_point_sp=None,
                delta_pct=0.0,
                pettitt=None,
                plateaus=[],
            ),
            short_window=None,
            global_trend={},
            drifting_up=["ttft_p50"],
        )
        snap = MetricsSnapshot(
            counter=3,
            timestamp_ns=7,
            state=SessionState.COMPLETE,
            n_pending_tasks=0,
            metrics=[],
            steady_state=verdict,
        )
        codec = MetricsSnapshotCodec()
        _, payload = codec.encode(snap)
        # The wire keeps the Struct; the dict form is builtins, since that is
        # what final_snapshot.json holds and what Report.from_snapshot converts.
        assert codec.decode(payload).steady_state == verdict
        assert snapshot_to_dict(snap)["steady_state"] == msgspec.to_builtins(verdict)

    def test_steady_state_defaults_to_none_for_older_frames(self):
        """The field is appended, so a frame without it still decodes."""
        codec = MetricsSnapshotCodec()
        payload = msgspec.msgpack.encode([4, 8, "live", 0, []])
        assert codec.decode(payload).steady_state is None

    def test_non_finite_floats_inside_the_verdict_are_scrubbed(self):
        """Without this, json.dumps(allow_nan=False) costs the run its snapshot."""
        snap = MetricsSnapshot(
            counter=5,
            timestamp_ns=9,
            state=SessionState.COMPLETE,
            n_pending_tasks=0,
            metrics=[],
            steady_state={"tps": {"system": float("nan")}, "ci": [float("inf"), 1.0]},
        )
        assert snapshot_to_dict(snap)["steady_state"] == {
            "tps": {"system": None},
            "ci": [None, 1.0],
        }

    def test_on_decode_error_drops_malformed(self):
        codec = MetricsSnapshotCodec()
        # Decode a clearly malformed payload (truncated msgpack)
        try:
            codec.decode(b"\xff\x00")
        except Exception as e:
            fallback = codec.on_decode_error(b"\xff\x00", e)
            assert fallback is None

    def test_on_decode_error_reraises_unknown(self):
        codec = MetricsSnapshotCodec()
        # Non-decode errors should propagate.
        with pytest.raises(RuntimeError):
            codec.on_decode_error(b"", RuntimeError("not a decode error"))


@pytest.mark.unit
class TestSessionStateTransitions:
    """The SessionState enum members are the only valid states the
    aggregator surfaces on the wire. Tests below pin the forward-only
    transition contract so a future enum addition / reorder doesn't
    silently break the consumer's drain-timeout detection
    (``state == COMPLETE and n_pending_tasks > 0``).
    """

    def test_all_states_are_string_serializable(self):
        # MetricsSnapshot encodes state as the enum *value* via msgspec's
        # str-Enum support. Each state must therefore round-trip as a
        # string literal — protects against accidental int-Enum reorder.
        for s in SessionState:
            assert isinstance(s.value, str)
            assert s.value == s

    def test_expected_member_set(self):
        # Pin the membership so a future addition is a deliberate review
        # decision, not an accident. Adding a new state requires updating
        # this test (and presumably the drain-timeout detection rule).
        assert {s.value for s in SessionState} == {
            "initialize",
            "live",
            "draining",
            "complete",
            "interrupted",
        }

    def test_forward_ordering_matches_declaration_order(self):
        # The aggregator transitions in declaration order; consumers can
        # rely on `list(SessionState)` for "did we move forward?" checks.
        # COMPLETE and INTERRUPTED are sibling terminal states — either is
        # reachable from DRAINING but not from each other.
        assert list(SessionState) == [
            SessionState.INITIALIZE,
            SessionState.LIVE,
            SessionState.DRAINING,
            SessionState.COMPLETE,
            SessionState.INTERRUPTED,
        ]

    @pytest.mark.parametrize(
        "state, n_pending, complete",
        [
            (SessionState.LIVE, 0, False),
            (SessionState.LIVE, 3, False),
            (SessionState.DRAINING, 5, False),
            (SessionState.COMPLETE, 0, True),
            # Drain-timeout case: COMPLETE arrived but tasks were
            # cancelled / abandoned. Consumer treats as incomplete.
            (SessionState.COMPLETE, 1, False),
            (SessionState.INITIALIZE, 0, False),
            # INTERRUPTED never satisfies the complete predicate, regardless
            # of n_pending_tasks — a kill-signal-triggered snapshot is
            # always partial data.
            (SessionState.INTERRUPTED, 0, False),
            (SessionState.INTERRUPTED, 7, False),
        ],
    )
    def test_complete_predicate(self, state, n_pending, complete):
        """``complete = state == COMPLETE and n_pending_tasks == 0``."""
        assert (state == SessionState.COMPLETE and n_pending == 0) is complete

    def test_initialize_round_trips_on_the_wire(self):
        # INITIALIZE is part of the wire schema; a snapshot tagged
        # INITIALIZE must decode back to the same state. (The aggregator
        # doesn't emit this state today — the tick task only starts on
        # the first STARTED — but the codec must still tolerate it for
        # future setup-phase ticks.)
        snap = MetricsSnapshot(
            counter=1,
            timestamp_ns=1,
            state=SessionState.INITIALIZE,
            n_pending_tasks=0,
            metrics=[],
        )
        codec = MetricsSnapshotCodec()
        _, payload = codec.encode(snap)
        decoded = codec.decode(payload)
        assert decoded.state == SessionState.INITIALIZE

    def test_interrupted_round_trips_on_the_wire(self):
        # INTERRUPTED is emitted by the signal handler's publish_final
        # call (SIGTERM/SIGINT). The msgpack wire form must round-trip.
        snap = MetricsSnapshot(
            counter=42,
            timestamp_ns=12345,
            state=SessionState.INTERRUPTED,
            n_pending_tasks=3,
            metrics=[CounterStat(name="c", value=7)],
        )
        codec = MetricsSnapshotCodec()
        _, msgpack_payload = codec.encode(snap)
        assert codec.decode(msgpack_payload).state == SessionState.INTERRUPTED
