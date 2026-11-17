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

"""Tests for the token-count basis vocabulary in ``core.types``."""

import logging
from typing import Any

import pytest
from inference_endpoint.core.types import (
    COMPLETION_TOKENS_KEY,
    PROMPT_TOKENS_KEY,
    TOKEN_HDR_HIGH,
    TokenCountSource,
    basis_key,
    put_server_usage,
    server_usage_count,
    server_usage_counts,
    well_formed_count,
)


@pytest.mark.unit
def test_other_basis():
    assert TokenCountSource.CLIENT.other is TokenCountSource.SERVER
    assert TokenCountSource.SERVER.other is TokenCountSource.CLIENT


@pytest.mark.unit
@pytest.mark.parametrize("primary", list(TokenCountSource))
def test_basis_key_suffixes_only_the_other_basis(primary):
    other = primary.other
    assert basis_key("osl", primary, primary=primary) == "osl"
    assert basis_key("osl", other, primary=primary) == f"osl_{other.value}"
    assert basis_key("tpot_ns", other, primary=primary) == f"tpot_ns_{other.value}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "value, expected",
    [
        (0, 0),
        (7, 7),
        (TOKEN_HDR_HIGH, TOKEN_HDR_HIGH),
        (TOKEN_HDR_HIGH + 1, None),
        (-1, None),
        (True, None),
        (False, None),
        (3.0, None),
        ("3", None),
        (None, None),
        ([], None),
    ],
)
def test_well_formed_count(value, expected):
    assert well_formed_count(value) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "usage, expected",
    [
        ({"prompt_tokens": 3, "completion_tokens": 5}, (3, 5)),
        ({"prompt_tokens": 3, "completion_tokens": "x"}, (3, None)),
        ({"prompt_tokens": 12.0}, (None, None)),
        ({}, (None, None)),
        (None, (None, None)),
        ([1, 2], (None, None)),
        ("usage", (None, None)),
    ],
)
def test_server_usage_counts(usage, expected):
    assert server_usage_counts(usage) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "usage, shown",
    [
        ("x", "'x' (type 'str')"),
        ([], "[] (type 'list')"),
        ({"prompt_tokens": 12.0}, "'prompt_tokens': 12.0 (type 'float')"),
        ({"completion_tokens": True}, "'completion_tokens': True (type 'bool')"),
        (
            {"prompt_tokens": 3, "completion_tokens": TOKEN_HDR_HIGH + 1},
            f"'completion_tokens': {TOKEN_HDR_HIGH + 1} (type 'int')",
        ),
    ],
)
def test_first_malformed_usage_warns_once_showing_value_and_type(
    usage_warning_unlatched, usage, shown
):
    server_usage_counts(usage)
    server_usage_counts(usage)
    server_usage_counts("another")
    (record,) = usage_warning_unlatched.records
    assert record.levelno == logging.WARNING
    assert shown in record.getMessage()


@pytest.mark.unit
def test_warning_shows_a_short_repr(usage_warning_unlatched):
    server_usage_counts({"prompt_tokens": "9" * 1000})
    (record,) = usage_warning_unlatched.records
    assert len(record.getMessage()) < 200


@pytest.mark.unit
@pytest.mark.parametrize(
    "value, expected, warns",
    [(5, 5, False), (None, None, False), (-1, None, True), (5.0, None, True)],
)
def test_server_usage_count_warns_only_on_non_null_rejects(
    usage_warning_unlatched, value, expected, warns
):
    assert server_usage_count(value, PROMPT_TOKENS_KEY) == expected
    assert bool(usage_warning_unlatched.records) is warns


@pytest.mark.unit
@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": None, "completion_tokens": None},
        {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    ],
    ids=["absent", "empty", "null-counts", "well-formed"],
)
def test_absent_or_well_formed_usage_never_warns(usage_warning_unlatched, usage):
    server_usage_counts(usage)
    assert usage_warning_unlatched.records == []


@pytest.mark.unit
def test_put_server_usage_writes_only_present_counts():
    md: dict[str, Any] = {}
    put_server_usage(md, 4, None)
    assert md == {PROMPT_TOKENS_KEY: 4}


@pytest.mark.unit
def test_put_server_usage_writes_both_counts():
    md: dict[str, Any] = {"other": 1}
    put_server_usage(md, 4, 9)
    assert md == {"other": 1, PROMPT_TOKENS_KEY: 4, COMPLETION_TOKENS_KEY: 9}


@pytest.mark.unit
def test_metadata_keys_are_the_openai_usage_field_names():
    assert PROMPT_TOKENS_KEY == "prompt_tokens"
    assert COMPLETION_TOKENS_KEY == "completion_tokens"
