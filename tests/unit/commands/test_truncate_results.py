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

"""Unit tests for the truncate-results command."""

from __future__ import annotations

import hashlib
import json

import pytest
from inference_endpoint.commands.truncate_results import (
    TruncateConfig,
    execute_truncate,
    truncate_results_dict,
)


def _results(n: int) -> dict:
    return {
        "config": {"mode": "both"},
        "results": {"total": n, "successful": n, "qps": float(n)},
        "accuracy_scores": {"ds": {"score": 0.9}},
        "responses": {f"u{i}": f"response {i}" for i in range(n)},
        "errors": ["Sample u-err: boom"],
    }


@pytest.mark.unit
def test_keeps_first_n_full_and_hashes_every_response():
    src = _results(5)
    out = truncate_results_dict(src, keep_n=2)

    # First N kept verbatim, the rest dropped from `responses`.
    assert out["responses"] == {"u0": "response 0", "u1": "response 1"}
    # Every original response is provably accounted for via its sha256.
    assert out["truncation"]["response_hashes"] == {
        uid: hashlib.sha256(text.encode()).hexdigest()
        for uid, text in src["responses"].items()
    }
    assert out["truncation"] == {
        "responses_truncated": True,
        "hash_algorithm": "sha256",
        "n_responses_total": 5,
        "n_responses_kept": 2,
        "response_hashes": out["truncation"]["response_hashes"],
    }


@pytest.mark.unit
def test_preserves_non_response_sections():
    src = _results(5)
    out = truncate_results_dict(src, keep_n=2)
    for key in ("config", "results", "accuracy_scores", "errors"):
        assert out[key] == src[key]


@pytest.mark.unit
def test_does_not_mutate_input():
    src = _results(5)
    truncate_results_dict(src, keep_n=2)
    assert len(src["responses"]) == 5


@pytest.mark.unit
def test_keep_n_exceeding_total_keeps_all():
    out = truncate_results_dict(_results(3), keep_n=10)
    assert len(out["responses"]) == 3
    assert out["truncation"]["n_responses_kept"] == 3


@pytest.mark.unit
def test_passthrough_when_no_responses():
    perf_only = {"config": {"mode": "offline"}, "results": {"qps": 50.0}}
    out = truncate_results_dict(perf_only, keep_n=5)
    assert out == perf_only
    assert "truncation" not in out


@pytest.mark.unit
def test_execute_writes_truncated_copy_leaving_original(tmp_path):
    src = tmp_path / "results.json"
    src.write_text(json.dumps(_results(4)))

    execute_truncate(TruncateConfig(results=src, keep_n=1))

    out = json.loads((tmp_path / "results.truncated.json").read_text())
    assert len(out["responses"]) == 1
    assert out["truncation"]["n_responses_total"] == 4
    assert len(json.loads(src.read_text())["responses"]) == 4  # original intact


@pytest.mark.unit
def test_execute_in_place(tmp_path):
    src = tmp_path / "results.json"
    src.write_text(json.dumps(_results(4)))

    execute_truncate(TruncateConfig(results=src, keep_n=1, in_place=True))

    assert len(json.loads(src.read_text())["responses"]) == 1
    assert not (tmp_path / "results.truncated.json").exists()
