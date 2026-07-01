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

"""Power source registry.

A source builder turns a :class:`PowerConfig` into a :class:`ResolvedSource`
(a sidecar argv + how to read what it prints). Built-ins register themselves
below; users add their own with the ``@power_source`` decorator and feed it
arbitrary settings via ``cfg.options`` — no core edits, no entry-points:

    from inference_endpoint.power import power_source, ResolvedSource

    @power_source("my_pdu")
    def _build(cfg):
        return ResolvedSource(
            argv=["my-pdu-reader", cfg.options["rack"]],
            fmt="jsonl", value_kind="power_w",
            ts_field="ts", value_field="value", label_field="label",
            csv_header=False,
        )

For non-Python sources, the built-in ``command`` source runs any program that
prints the canonical JSONL contract — the process boundary is the plugin API.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from inference_endpoint.config.schema import PowerConfig


@dataclass(frozen=True)
class ResolvedSource:
    """A runnable sidecar plus how to read what it prints (all internal)."""

    argv: list[str]
    fmt: str  # "csv" (nvidia-smi only) | "jsonl"
    value_kind: str  # "power_w" | "energy_j"
    # Field/column mapping: integer-index strings for headerless CSV, key names
    # for JSONL.
    ts_field: str
    value_field: str
    label_field: str | None
    csv_header: bool
    # If set, bare numeric labels are prefixed with it (nvidia GPU 0 -> "gpu0").
    label_prefix: str | None = None


PowerSourceBuilder = Callable[[PowerConfig], ResolvedSource]
_REGISTRY: dict[str, PowerSourceBuilder] = {}


def power_source(name: str) -> Callable[[PowerSourceBuilder], PowerSourceBuilder]:
    """Register a builder under ``name`` (usable as ``power.source``)."""

    def decorator(fn: PowerSourceBuilder) -> PowerSourceBuilder:
        _REGISTRY[name] = fn
        return fn

    return decorator


def resolve(cfg: PowerConfig) -> ResolvedSource:
    """Look up and run the builder for ``cfg.source``. Caller guarantees enabled."""
    builder = _REGISTRY.get(cfg.source or "")
    if builder is None:
        raise ValueError(
            f"unknown power source {cfg.source!r}; available: {sorted(_REGISTRY)}"
        )
    return builder(cfg)


def _jsonl_source(argv: list[str], value_kind: str) -> ResolvedSource:
    """A source emitting the canonical JSONL contract {ts, value, label}."""
    return ResolvedSource(
        argv=argv,
        fmt="jsonl",
        value_kind=value_kind,
        ts_field="ts",
        value_field="value",
        label_field="label",
        csv_header=False,
    )


def _require(cfg: PowerConfig, key: str, source: str) -> Any:
    """Fetch a required ``options`` key or raise a clear error."""
    if key not in cfg.options:
        raise ValueError(f"power source {source!r} requires options.{key}")
    return cfg.options[key]


@power_source("nvidia_smi")
def _nvidia_smi(cfg: PowerConfig) -> ResolvedSource:
    """First reference source. NVIDIA-specific bits live here, not in the schema.

    options: gpu_indices (list[int], optional; default all visible GPUs).
    """
    argv = [
        # stdbuf line-buffers so tail samples aren't lost to block buffering
        # when stdout is a file and the sidecar is killed at phase end.
        "stdbuf",
        "-oL",
        "nvidia-smi",
        "--query-gpu=timestamp,index,power.draw",
        "--format=csv,noheader,nounits",
        "-lms",
        str(int(cfg.interval_s * 1000)),
    ]
    gpu_indices = cfg.options.get("gpu_indices")
    if gpu_indices:
        argv += ["-i", ",".join(str(i) for i in gpu_indices)]
    # Headerless CSV columns: timestamp(nvidia fmt), index, power.draw(W).
    return ResolvedSource(
        argv=argv,
        fmt="csv",
        value_kind="power_w",
        ts_field="0",
        value_field="2",
        label_field="1",
        csv_header=False,
        label_prefix="gpu",
    )


@power_source("prometheus")
def _prometheus(cfg: PowerConfig) -> ResolvedSource:
    """options: url (str), query (PromQL returning watts/joules)."""
    argv = [
        sys.executable,
        "-m",
        "inference_endpoint.power.prom_poll",
        "--url",
        str(_require(cfg, "url", "prometheus")),
        "--query",
        str(_require(cfg, "query", "prometheus")),
        "--interval",
        str(cfg.interval_s),
    ]
    return _jsonl_source(argv, cfg.value_kind)


@power_source("command")
def _command(cfg: PowerConfig) -> ResolvedSource:
    """options: argv (list[str]) that prints one canonical JSONL sample per line."""
    argv = _require(cfg, "argv", "command")
    return _jsonl_source(list(argv), cfg.value_kind)
