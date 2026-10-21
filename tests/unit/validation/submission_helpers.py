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

"""Builders for `results/<system>/<model>/r<N>` uploads, shared by the suite.

Every `r<N>` run directory carries the three artifacts `Point.from_node`
reads: `point.yaml` (§8.3's `concurrency` plus the optional `offline`
declaration), its own `system_desc.json` (§8.2.1's `max_supported_concurrency`
plus the optional `c_min_override`) and `result_summary.json` (§8.1's
`system_tps`). `Run` spells those facts. `OMITTED` leaves a fact out of its
document (the absent state), while any other value passes through verbatim so
a refusal test can write `"16"` or `true` where an integer is required.

`write_upload` writes one `Run` per run directory under `root` and returns
`Submission.from_directory(root)`, so tests exercise the real tree walk.
`write_cohort` writes a cohort directory: the scope files plus the
`valid_refs.yaml` (`STANDARD_REFS` unless a test declares its own table),
since `Cohort.from_directory` refuses a cohort without one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml
from inference_endpoint.validation.catalog import Catalog
from inference_endpoint.validation.refs import RefTable
from inference_endpoint.validation.submission import Submission

OMITTED: Final = object()
_RUN_FILES = ("point.yaml", "system_desc.json", "result_summary.json")


@dataclass(frozen=True)
class Run:
    """One `r<N>` run directory's declared facts, as the three documents say.

    Args:
        concurrency (Any): The run's `point.yaml` `concurrency` value. (Default: 16)
        max_supported_concurrency (Any): The run's `system_desc.json`
            `max_supported_concurrency` value. (Default: 8192)
        system_tps (Any): The run's `result_summary.json` `system_tps` value.
            (Default: 1.0)
        offline (Any): The run's `point.yaml` `offline` declaration.
            (Default: `OMITTED`)
        c_min_override (Any): The run's `system_desc.json` `c_min_override`
            declaration. (Default: `OMITTED`)
        dir_name (str | None): The run directory's name. `dir_name` defaults to
            `r<concurrency>`.
        config_extra (Mapping[str, Any]): Extra `point.yaml` keys. `*_extra` keys
            are merged last so a test can add or overwrite any document key beside the
            named facts. (Default: {})
        system_desc_extra (Mapping[str, Any]): Extra `system_desc.json` keys,
            merged last. (Default: {})
        summary_extra (Mapping[str, Any]): Extra `result_summary.json` keys,
            merged last. (Default: {})
    """

    concurrency: Any = 16
    max_supported_concurrency: Any = 8192
    system_tps: Any = 1.0
    offline: Any = OMITTED
    c_min_override: Any = OMITTED
    dir_name: str | None = None
    config_extra: Mapping[str, Any] = field(default_factory=dict)
    system_desc_extra: Mapping[str, Any] = field(default_factory=dict)
    summary_extra: Mapping[str, Any] = field(default_factory=dict)

    def name(self) -> str:
        """The run directory's `r<N>`-shaped name."""
        return self.dir_name if self.dir_name is not None else f"r{self.concurrency}"

    def documents(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """`(point.yaml, system_desc.json, result_summary.json)` contents to write."""
        config = {
            **{
                key: value
                for key, value in (
                    ("concurrency", self.concurrency),
                    ("offline", self.offline),
                )
                if value is not OMITTED
            },
            **self.config_extra,
        }
        system_desc = {
            **{
                key: value
                for key, value in (
                    ("max_supported_concurrency", self.max_supported_concurrency),
                    ("c_min_override", self.c_min_override),
                )
                if value is not OMITTED
            },
            **self.system_desc_extra,
        }
        summary = {
            **({} if self.system_tps is OMITTED else {"system_tps": self.system_tps}),
            **self.summary_extra,
        }
        return config, system_desc, summary


def write_run(run_dir: Path, run: Run) -> Path:
    """Write one run's three artifact documents under `run_dir`."""
    run_dir.mkdir(parents=True, exist_ok=True)
    config, system_desc, summary = run.documents()
    (run_dir / _RUN_FILES[0]).write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / _RUN_FILES[1]).write_text(json.dumps(system_desc), encoding="utf-8")
    (run_dir / _RUN_FILES[2]).write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def write_upload(
    root: Path,
    paretos: Mapping[tuple[str, str], Sequence[Run]],
    ruleset: Catalog | None = None,
) -> Submission:
    """Write each `(system, pareto)` run set below `root` and walk it as an upload.

    Args:
        root (Path): The upload root. The run sets are written below it.
        paretos (Mapping[tuple[str, str], Sequence[Run]]): The `paretos` keys are
            `(system_name, pareto_name)`. Each value is the pareto's ordered `Run` s,
            written to `results/<system>/<pareto>/<r<N>>/`.
        ruleset (Catalog | None): The `Catalog` folded into the walk's `ruleset`
            root. (Default: None)
    """
    writing: set[str] = set()
    for (system, pareto), runs in paretos.items():
        for run in runs:
            relative = f"{system}/{pareto}/{run.name()}"
            if relative in writing:
                raise ValueError(f"two runs share one directory: {relative}")
            writing.add(relative)
            write_run(root / "results" / system / pareto / run.name(), run)
    return Submission.from_directory(root, ruleset=ruleset)


# The suite's standard `valid_refs.yaml`: every field name its checks name,
# with the three absent states spelled per field -- `concurrency` /
# `system_desc.max_supported_concurrency` / `result_summary.system_tps` are
# required (their absence breaks a row loudly), `offline` declares §8.3's three
# named states (`allowlist`) with the absent-or-`none` state as `default:
# null`, and the rest are plain optionals
# (typed leaves, open `dict` subtrees, or named blocks).
STANDARD_REFS: Final = """
point:
  config:
    concurrency: { type: int, required: true }
    offline: { type: str, allowlist: [dedicated, elected, none], default: null }
    nodes_used: int
    warmup: { duration_s: int }
    steady_state: { window: dict }
    isl: dict
    osl: dict
    runtime_settings: dict
    dataset: dict
    speculative_decoding: dict
  system_desc:
    max_supported_concurrency: { type: int, required: true }
    c_min_override: int
    cooling: str
  result_summary:
    system_tps: { type: float, required: true }
    tps_per_user: float
    ttft_p90_ms: float
    tpot_p90_ms: float
    n_samples_issued: int
    n_samples_completed: int
    qps: float
    percentiles: dict
    dp_shortfall: dict
  accuracy_results: dict
pareto:
  system_desc:
    max_supported_concurrency: { type: int, required: true }
    c_min_override: int
    cooling: str
system:
  system_power: { total: float }
"""


def write_cohort(
    root: Path,
    *,
    refs: str | Mapping[str, Any] | None = STANDARD_REFS,
    **scope_files: str,
) -> Path:
    """A cohort directory from `scope -> YAML text` keyword arguments.

    Example:

        root = write_cohort(tmp / "cohort", point=CHECK_ONE)  # tmp/cohort: point.yaml beside valid_refs.yaml

    Args:
        root (Path): The cohort directory the scope files are written into.
        refs (str | Mapping[str, Any] | None): `refs=` overrides the declaration
            (raw YAML text or a mapping, per-cohort updatability). `refs=None` writes
            none, for the missing-table refusal pin. (Default: `STANDARD_REFS`)
        **scope_files (str): Each keyword argument maps a scope name to YAML
            text, written to `<scope>.yaml`.
    """
    root.mkdir(parents=True, exist_ok=True)
    if refs is not None:
        text = refs if isinstance(refs, str) else yaml.safe_dump(dict(refs))
        (root / RefTable.FILENAME).write_text(text, encoding="utf-8")
    for scope, text in scope_files.items():
        (root / f"{scope}.yaml").write_text(text, encoding="utf-8")
    return root
