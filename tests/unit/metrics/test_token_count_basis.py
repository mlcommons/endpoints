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

"""Tests for ``inference_endpoint.metrics.token_count_basis``."""

import json

import pytest
import yaml
from inference_endpoint.config.schema import BenchmarkConfig
from inference_endpoint.config.schema import TestType as BenchmarkTestType
from inference_endpoint.core.types import APIType, TextModelOutput, TokenCountSource
from inference_endpoint.metrics.token_count_basis import (
    ARCHIVED_RUN_BASIS,
    RunBasisError,
    ServerUsage,
    classify_server_usage,
    effective_token_count_source,
    missing_usage_message,
    read_server_usage,
    replay_basis,
    token_count_source_for_run,
)

from tests.test_helpers import SERVER_USAGE_CASES

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("prompt", "completion", "output", "expected"),
    [case[1:] for case in SERVER_USAGE_CASES],
    ids=[case[0] for case in SERVER_USAGE_CASES],
)
def test_classify_server_usage(prompt, completion, output, expected):
    assert classify_server_usage(prompt, completion, output) is expected


@pytest.mark.parametrize(
    ("prompt", "completion", "output", "expected"),
    [case[1:] for case in SERVER_USAGE_CASES],
    ids=[case[0] for case in SERVER_USAGE_CASES],
)
def test_read_server_usage(prompt, completion, output, expected):
    assert read_server_usage(prompt, completion, output) == (
        expected,
        completion if expected is ServerUsage.USABLE else None,
    )


@pytest.mark.parametrize(
    "output",
    [
        TextModelOutput(output="", reasoning="thinking"),
        TextModelOutput(output="", tool_calls=({"type": "function"},)),
        # A malformed tool-call payload is non-empty output, not a true zero.
        TextModelOutput(output="", tool_calls=(("not-a-dict",),)),
        TextModelOutput(output="", tool_calls=(({"index": ["unhashable"]},),)),
    ],
    ids=["reasoning", "tool_calls", "malformed_not_a_dict", "malformed_index"],
)
def test_zero_completion_beside_non_text_output_is_missing(output):
    assert classify_server_usage(10, 0, output) is ServerUsage.MISSING


@pytest.mark.parametrize("configured", list(TokenCountSource))
def test_effective_basis_forces_client_for_videogen(configured):
    assert (
        effective_token_count_source(configured, APIType.VIDEOGEN)
        is TokenCountSource.CLIENT
    )


@pytest.mark.parametrize("configured", list(TokenCountSource))
@pytest.mark.parametrize(
    "api_type", [t for t in APIType if t is not APIType.VIDEOGEN], ids=lambda t: t.value
)
def test_effective_basis_keeps_configured_otherwise(configured, api_type):
    assert effective_token_count_source(configured, api_type) is configured


_USAGE = "3 of 10 successful tracked samples returned no server token usage"
# The CLI flag, then the YAML key: `benchmark from-config` takes no such flag.
_FIX = (
    "set --token-count-source client if this server cannot report usage "
    "(YAML: settings.token_count_source: client)"
)
_EMPTY = (
    "had empty output — likely server errors or cut streams; "
    "client mode will not fix these"
)


@pytest.mark.parametrize(
    ("n_missing", "n_empty", "expected"),
    [
        (3, 0, f"{_USAGE}; {_FIX}"),
        (3, 3, f"{_USAGE}; 3 {_EMPTY}"),
        (3, 1, f"{_USAGE}; {_FIX}; 1 {_EMPTY}"),
    ],
)
def test_missing_usage_message_server_primary_is_the_fatal_text(
    n_missing, n_empty, expected
):
    assert (
        missing_usage_message(n_missing, n_empty, 10, primary=TokenCountSource.SERVER)
        == expected
    )


def test_missing_usage_message_by_primary():
    server = missing_usage_message(3, 0, 10, primary=TokenCountSource.SERVER)
    client = missing_usage_message(3, 0, 10, primary=TokenCountSource.CLIENT)
    assert "settings.token_count_source" in server
    assert "_server" in client
    assert "exit" not in client
    assert "3 of 10" in client


def _write_summary(run_dir, payload):
    (run_dir / "performance").mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    (run_dir / "performance" / "result_summary.json").write_text(text)


def _write_config(run_dir, payload):
    text = payload if isinstance(payload, str) else yaml.safe_dump(payload)
    (run_dir / "config.yaml").write_text(text)


def _config_yaml(source, api_type=APIType.OPENAI):
    return {
        "settings": {"token_count_source": source.value},
        "endpoint_config": {"api_type": api_type.value},
    }


@pytest.mark.parametrize("recorded", list(TokenCountSource))
def test_run_basis_reads_summary_first(tmp_path, recorded):
    _write_summary(tmp_path, {"token_count_source": recorded.value})
    _write_config(tmp_path, _config_yaml(recorded.other))
    assert token_count_source_for_run(tmp_path) is recorded


@pytest.mark.parametrize("recorded", list(TokenCountSource))
def test_run_basis_reads_the_given_summary(tmp_path, recorded):
    summary_path = tmp_path / "elsewhere" / "result_summary.json"
    summary_path.parent.mkdir()
    summary_path.write_text(json.dumps({"token_count_source": recorded.value}))
    _write_summary(tmp_path, {"token_count_source": recorded.other.value})
    assert token_count_source_for_run(tmp_path, summary_path=summary_path) is recorded


@pytest.mark.parametrize("configured", list(TokenCountSource))
def test_run_basis_given_summary_without_key_falls_back_to_config_yaml(
    tmp_path, configured
):
    summary_path = tmp_path / "elsewhere.json"
    summary_path.write_text(json.dumps({"tpot": {}}))
    _write_summary(tmp_path, {"token_count_source": configured.other.value})
    _write_config(tmp_path, _config_yaml(configured))
    assert token_count_source_for_run(tmp_path, summary_path=summary_path) is configured


@pytest.mark.parametrize("configured", list(TokenCountSource))
@pytest.mark.parametrize(
    "summary", [None, {"run_config": {}}], ids=["absent", "no_key"]
)
def test_run_basis_falls_back_to_config_yaml(tmp_path, configured, summary):
    if summary is not None:
        _write_summary(tmp_path, summary)
    _write_config(tmp_path, _config_yaml(configured))
    assert token_count_source_for_run(tmp_path) is configured


@pytest.mark.parametrize("configured", list(TokenCountSource))
def test_run_basis_config_yaml_videogen_is_client(tmp_path, configured):
    _write_config(tmp_path, _config_yaml(configured, APIType.VIDEOGEN))
    assert token_count_source_for_run(tmp_path) is TokenCountSource.CLIENT


@pytest.mark.parametrize("configured", list(TokenCountSource))
@pytest.mark.parametrize("api_type", [APIType.OPENAI, APIType.VIDEOGEN])
def test_run_basis_reads_what_a_run_writes(tmp_path, configured, api_type):
    config = BenchmarkConfig.create_default_config(BenchmarkTestType.OFFLINE)
    config = config.with_updates(
        settings=config.settings.model_copy(update={"token_count_source": configured}),
        endpoint_config=config.endpoint_config.model_copy(
            update={"api_type": api_type}
        ),
    )
    config.to_yaml_file(tmp_path / "config.yaml")
    assert token_count_source_for_run(tmp_path) is effective_token_count_source(
        configured, api_type
    )


def test_run_basis_archived_is_client(tmp_path, capsys):
    assert token_count_source_for_run(tmp_path) is ARCHIVED_RUN_BASIS
    assert "--token-count-source" in capsys.readouterr().err


def test_run_basis_config_without_recorded_basis_is_archived(tmp_path, capsys):
    _write_config(tmp_path, {"settings": {}, "endpoint_config": {}})
    assert token_count_source_for_run(tmp_path) is ARCHIVED_RUN_BASIS
    assert "--token-count-source" in capsys.readouterr().err


@pytest.mark.parametrize(
    "summary",
    ["{not json", "[1, 2]", json.dumps({"token_count_source": 3})],
    ids=["bad_json", "not_an_object", "not_a_string"],
)
def test_run_basis_unreadable_summary_raises(tmp_path, summary):
    _write_summary(tmp_path, summary)
    with pytest.raises(RunBasisError, match="--token-count-source"):
        token_count_source_for_run(tmp_path)


@pytest.mark.parametrize(
    "config",
    ["settings: [unclosed", "- a\n- b\n", {"settings": "x"}],
    ids=["bad_yaml", "not_a_mapping", "settings_not_a_mapping"],
)
def test_run_basis_unreadable_config_raises(tmp_path, config):
    _write_config(tmp_path, config)
    with pytest.raises(RunBasisError, match="--token-count-source"):
        token_count_source_for_run(tmp_path)


def test_run_basis_unknown_value_raises(tmp_path):
    _write_summary(tmp_path, {"token_count_source": "bogus"})
    with pytest.raises(RunBasisError, match="--token-count-source"):
        token_count_source_for_run(tmp_path)


@pytest.mark.parametrize(
    "config",
    [
        {"settings": {"token_count_source": "bogus"}},
        {
            "settings": {"token_count_source": TokenCountSource.SERVER.value},
            "endpoint_config": {"api_type": "bogus"},
        },
    ],
    ids=["bogus_source", "bogus_api_type"],
)
def test_run_basis_unknown_config_value_raises(tmp_path, config):
    _write_config(tmp_path, config)
    with pytest.raises(RunBasisError, match="--token-count-source"):
        token_count_source_for_run(tmp_path)


def _override_id(override):
    return "none" if override is None else override.value


@pytest.mark.parametrize("recorded", list(TokenCountSource))
@pytest.mark.parametrize("override", [None, *TokenCountSource], ids=_override_id)
def test_replay_basis_is_the_override_else_the_run_basis(tmp_path, recorded, override):
    _write_summary(tmp_path, {"token_count_source": recorded.value})
    basis = recorded if override is None else override
    assert replay_basis(tmp_path, override=override) == (recorded, basis)


@pytest.mark.parametrize("recorded", list(TokenCountSource))
def test_replay_basis_reads_the_given_summary(tmp_path, recorded):
    summary_path = tmp_path / "elsewhere.json"
    summary_path.write_text(json.dumps({"token_count_source": recorded.value}))
    assert replay_basis(tmp_path, override=None, summary_path=summary_path) == (
        recorded,
        recorded,
    )


def test_replay_basis_notes_an_archived_run_only_without_override(tmp_path, capsys):
    assert replay_basis(tmp_path, override=None) == (
        ARCHIVED_RUN_BASIS,
        ARCHIVED_RUN_BASIS,
    )
    assert "--token-count-source" in capsys.readouterr().err
    for override in TokenCountSource:
        assert replay_basis(tmp_path, override=override) == (
            ARCHIVED_RUN_BASIS,
            override,
        )
        assert capsys.readouterr().err == ""


@pytest.mark.parametrize("override", list(TokenCountSource))
def test_replay_basis_unreadable_run_basis_is_none_beside_an_override(
    tmp_path, override
):
    _write_summary(tmp_path, "{not json")
    assert replay_basis(tmp_path, override=override) == (None, override)
    with pytest.raises(RunBasisError, match="--token-count-source"):
        replay_basis(tmp_path, override=None)
