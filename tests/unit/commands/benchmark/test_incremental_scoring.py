# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import msgspec.json
import pytest

from inference_endpoint.commands.benchmark.execute import _complete_uuids_in_event_log
from inference_endpoint.core.record import EventRecord, SampleEventType
from inference_endpoint.core.types import PromptData
from inference_endpoint.load_generator.session import PhaseResult, PhaseType


@pytest.mark.unit
def test_complete_uuids_in_event_log(tmp_path) -> None:
    events_path = tmp_path / "events.jsonl"
    complete = EventRecord(
        event_type=SampleEventType.COMPLETE,
        sample_uuid="uuid-a",
        data="answer-a",
    )
    other = EventRecord(
        event_type=SampleEventType.STARTED,
        sample_uuid="uuid-b",
        data=PromptData(prompt="hi"),
    )
    with events_path.open("w") as f:
        f.write(msgspec.json.encode(complete).decode() + "\n")
        f.write(msgspec.json.encode(other).decode() + "\n")

    assert _complete_uuids_in_event_log(events_path) == {"uuid-a"}


@pytest.mark.unit
def test_wait_for_phase_event_log_missing_file(tmp_path) -> None:
    from inference_endpoint.commands.benchmark.execute import _wait_for_phase_event_log

    phase = PhaseResult(
        name="aime25::deepseek_v4",
        phase_type=PhaseType.ACCURACY,
        uuid_to_index={"u1": 0},
        issued_count=1,
        start_time_ns=0,
        end_time_ns=1,
    )
    assert _wait_for_phase_event_log(tmp_path / "events.jsonl", phase, timeout_s=0.5) is False
