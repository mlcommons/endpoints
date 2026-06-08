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

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _schema() -> dict:
    with Path("scripts/multi_turn_dataset_schema.json").open() as fh:
        return json.load(fh)


def _row(delay_seconds):
    return {
        "conversation_id": "c1",
        "turn": 1,
        "role": "user",
        "content": "hello",
        "delay_seconds": delay_seconds,
    }


def test_delay_seconds_schema_declares_non_negative_number():
    prop = _schema()["definitions"]["generationParameters"]["properties"][
        "delay_seconds"
    ]

    assert prop["type"] == "number"
    assert prop["minimum"] == 0


@pytest.mark.parametrize(
    ("delay_seconds", "valid"),
    [(0, True), (1.25, True), (-5, False), ("not-a-number", False)],
)
def test_delay_seconds_validation(delay_seconds, valid):
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft7Validator(_schema())

    errors = list(validator.iter_errors(_row(delay_seconds)))

    assert (not errors) is valid
