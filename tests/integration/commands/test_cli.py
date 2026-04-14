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

"""CLI integration tests.

Hypothesis fuzzing: auto-discovers all CLI flags from cyclopts
``assemble_argument_collection()`` and tests random combinations through
the parser. E2E tests verify all three execution modes against echo server.
"""

from __future__ import annotations

import enum
import json
from typing import Literal, get_args, get_origin

import pytest
from hypothesis import given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, composite
from inference_endpoint.commands.benchmark.cli import benchmark_app
from inference_endpoint.main import app
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Flag discovery — walks cyclopts argument tree at import time.
# Zero hardcoded knowledge: all flags, types, and valid values are derived
# from the Pydantic models via assemble_argument_collection().
# ---------------------------------------------------------------------------


def _enum_values(hint):
    """Extract string values from enum or Literal type hints."""
    if isinstance(hint, type) and issubclass(hint, enum.Enum):
        return [e.value for e in hint]
    if get_origin(hint) is Literal:
        return [a.value if isinstance(a, enum.Enum) else str(a) for a in get_args(hint)]
    return None


def _discover_flags(cmd_name: str) -> list[tuple[str, list[str]]]:
    """Discover (flag, [valid_values]) for every leaf CLI flag in a subcommand.

    Includes both primary names and aliases as separate entries so Hypothesis
    exercises them independently.
    """
    sub_app = benchmark_app.resolved_commands()[cmd_name]
    result = []
    type_vals: dict[type, list[str]] = {
        int: ["1", "10"],
        float: ["1.0", "10.0"],
        str: ["test-val"],
        bool: [],
    }
    for arg in sub_app.assemble_argument_collection():
        flags = [n for n in arg.names if n.startswith("--")]
        if not flags or arg.names == ("*",):
            continue
        if isinstance(arg.hint, type) and issubclass(arg.hint, BaseModel):
            continue
        enum_vals = _enum_values(arg.hint)
        vals: list[str] = (
            [str(v) for v in enum_vals] if enum_vals else type_vals.get(arg.hint, [])
        )
        result.append((flags[0], vals))
        # Aliases must also parse correctly — add them separately
        for alias in flags[1:]:
            result.append((alias, vals))
    return result


_OFFLINE_FLAGS = _discover_flags("offline")
_ONLINE_FLAGS = _discover_flags("online")

# ---------------------------------------------------------------------------
# Hypothesis strategies — build random CLI invocations from discovered flags.
# Covers all three modes: offline, online/poisson, online/concurrency.
# ---------------------------------------------------------------------------

_OFF = [
    "benchmark",
    "offline",
    "--endpoints",
    "http://h:80",
    "--model",
    "m",
    "--dataset",
    "d.pkl",
]
_ON_P = [
    "benchmark",
    "online",
    "--endpoints",
    "http://h:80",
    "--model",
    "m",
    "--dataset",
    "d.pkl",
    "--load-pattern",
    "poisson",
    "--target-qps",
    "100",
]
_ON_C = [
    "benchmark",
    "online",
    "--endpoints",
    "http://h:80",
    "--model",
    "m",
    "--dataset",
    "d.pkl",
    "--load-pattern",
    "concurrency",
    "--concurrency",
    "10",
]


def _build_tokens(
    draw: DrawFn, base: list[str], flags: list[tuple[str, list[str]]]
) -> list[str]:
    """Append 1-10 random flags (with valid values) to a base token list."""
    n = draw(st.integers(min_value=1, max_value=10))
    chosen = draw(
        st.lists(
            st.sampled_from(flags), min_size=n, max_size=n, unique_by=lambda x: x[0]
        )
    )
    tokens = list(base)
    for flag, vals in chosen:
        if flag in tokens:
            continue  # don't duplicate flags already in base
        tokens.extend([flag, draw(st.sampled_from(vals))] if vals else [flag])
    return tokens


@composite
def offline_tokens(draw: DrawFn) -> list[str]:
    """Random offline CLI invocation."""
    return _build_tokens(draw, _OFF, _OFFLINE_FLAGS)


@composite
def online_tokens(draw: DrawFn) -> list[str]:
    """Random online CLI invocation — randomly picks poisson or concurrency base."""
    return _build_tokens(draw, draw(st.sampled_from([_ON_P, _ON_C])), _ONLINE_FLAGS)


# ---------------------------------------------------------------------------
# Hypothesis parsing fuzz
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.schema_fuzz
@pytest.mark.slow
@hyp_settings(max_examples=2000, deadline=5000)
@given(tokens=offline_tokens())
def test_offline_cli_no_crash(tokens):
    """Random offline flag combos must parse or reject cleanly — never crash."""
    try:
        app.parse_args(tokens)
    except SystemExit:
        pass  # clean rejection (e.g. invalid value) is fine


@pytest.mark.integration
@pytest.mark.schema_fuzz
@pytest.mark.slow
@hyp_settings(max_examples=2000, deadline=5000)
@given(tokens=online_tokens())
def test_online_cli_no_crash(tokens):
    """2000 random online flag combos (poisson + concurrency) — no crashes."""
    try:
        app.parse_args(tokens)
    except SystemExit:
        pass  # clean rejection (e.g. missing required arg) is fine


# ---------------------------------------------------------------------------
# E2E: CLI tokens → echo server → results.json
# One test per execution mode: offline, poisson, concurrency.
# ---------------------------------------------------------------------------

_FAST = [
    "--workers",
    "1",
    "--client.warmup-connections",
    "0",
    "--client.max-connections",
    "10",
]


def _run(tokens: list[str]):
    """Invoke the full CLI pipeline, swallowing normal SystemExit(0)."""
    try:
        app.meta(tokens)
    except SystemExit as e:
        if e.code != 0:
            raise


def _bench(url, ds, tmp_path, *extra):
    """Run a benchmark via CLI and return parsed results.json."""
    _run(
        [
            *extra,
            "--endpoints",
            url,
            "--model",
            "test-model",
            "--dataset",
            ds,
            "--report-dir",
            str(tmp_path),
            *_FAST,
        ]
    )
    return json.loads((tmp_path / "results.json").read_text())


class TestE2E:
    """Full CLI → benchmark execution → echo server → results.json."""

    @pytest.mark.integration
    def test_offline(self, mock_http_echo_server, ds_dataset_path, tmp_path):
        """Offline (max_throughput): all queries at t=0."""
        r = _bench(
            mock_http_echo_server.url,
            ds_dataset_path,
            tmp_path,
            "benchmark",
            "offline",
            "--duration",
            "0",
            "--streaming",
            "off",
        )
        assert r["results"]["total"] > 0
        assert r["results"]["successful"] > 0

    @pytest.mark.integration
    def test_poisson(self, mock_http_echo_server, ds_dataset_path, tmp_path):
        """Online (poisson): sustained QPS with Poisson arrival distribution."""
        r = _bench(
            mock_http_echo_server.url,
            ds_dataset_path,
            tmp_path,
            "benchmark",
            "online",
            "--load-pattern",
            "poisson",
            "--target-qps",
            "50",
            "--duration",
            "2000",
        )
        assert r["results"]["total"] > 0

    @pytest.mark.integration
    def test_concurrency(self, mock_http_echo_server, ds_dataset_path, tmp_path):
        """Online (concurrency): fixed concurrent requests."""
        r = _bench(
            mock_http_echo_server.url,
            ds_dataset_path,
            tmp_path,
            "benchmark",
            "online",
            "--load-pattern",
            "concurrency",
            "--concurrency",
            "4",
            "--duration",
            "2000",
        )
        assert r["results"]["total"] > 0
