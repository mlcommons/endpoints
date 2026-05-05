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

"""Shared test doubles and factories for metrics aggregator tests.

NOTE: this conftest is in the process of being migrated to the
registry-based aggregator (metrics_pubsub_design_v5.md). The legacy
``InMemoryKVStore`` factories that previously lived here have been
removed; tests that depended on them are skipped pending rewrite. New
tests for ``snapshot.py``, ``registry.py``, and ``publisher.py`` are
self-contained and do not need helpers from this module.
"""

from __future__ import annotations

import asyncio

from inference_endpoint.core.record import (
    EventRecord,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import TextModelOutput

# ---------------------------------------------------------------------------
# Mock TokenizePool — still useful for tests that exercise async triggers
# directly.
# ---------------------------------------------------------------------------


class MockTokenizePool:
    """Mock TokenizePool that splits on whitespace with artificial async delay."""

    def __init__(self, delay: float = 0.01) -> None:
        self._delay = delay

    def token_count(self, text: str) -> int:
        return len(text.split())

    async def token_count_async(
        self, text: str, _loop: asyncio.AbstractEventLoop
    ) -> int:
        await asyncio.sleep(self._delay)
        return len(text.split())

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# EventRecord factories
# ---------------------------------------------------------------------------


def session_event(ev_type: SessionEventType, ts: int = 0) -> EventRecord:
    return EventRecord(event_type=ev_type, timestamp_ns=ts)


def sample_event(
    ev_type: SampleEventType, uuid: str, ts: int = 0, data=None
) -> EventRecord:
    return EventRecord(event_type=ev_type, timestamp_ns=ts, sample_uuid=uuid, data=data)


def text_output(s: str) -> TextModelOutput:
    return TextModelOutput(output=s)


def streaming_text(*chunks: str) -> TextModelOutput:
    return TextModelOutput(output=tuple(chunks))
