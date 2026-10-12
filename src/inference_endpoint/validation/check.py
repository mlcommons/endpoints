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

"""Check blocks: a cohort's checks, decoded from YAML and validated in isolation.

One YAML file describes one check (`id`, prose `description`/`rule`/`fix`, a
`severity`, the node kind it applies to and its `when`/`assert` rows):

    id: concur_dur_round
    description: Concurrency asked for must be the per-system maximum.
    rule: "Upload Rules > Concurrency"
    fix: Ask for max_sync concurrency in exec_scenario.
    severity: fail
    root: point
    when:
      - over: sys.config.max_sync
        measure: count
        op: eq
        value: 1
    assert:
      - left: point.request.concurrency_achieved
        op: eq
        right: point.config.max_sync

`when` gates the rule (all rows must hold for the `assert` rows to speak);
`assert` is what the rule holds to. `Cohort.from_directory` gathers every
check in a cohort directory.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from inference_endpoint.validation.compare import Aggregate, Compare
from inference_endpoint.validation.tree import RuleScopes, _StrictLoader


class Severity(str, Enum):  # noqa: UP042 - repository wire-enum convention
    """What a violated `assert` row costs the submission.

    `fail` is a blocking finding; `warn` is reported and passes through.
    """

    FAIL = "fail"
    WARN = "warn"


def _decode_row(value: Any) -> Any:
    """YAML decode hook on every `when`/`assert` row, before model validation.

    `{"over": ..., "measure": "count", ...}` -> `Aggregate`; anything else with
    `left`/`op` -> `Compare`. A bare value raises ValueError: both row kinds
    are models, so a scalar cannot be either.
    """
    if isinstance(value, dict):
        if "measure" in value:
            return Aggregate(**value)
        return Compare(**value)
    raise ValueError(f"expected a `Compare` or `Aggregate` row, got {value!r}")


Row = Annotated[Compare | Aggregate, BeforeValidator(_decode_row)]


class Check(BaseModel):
    """One check block: its provenance and the conditions it evaluates."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    description: str
    rule: str
    fix: str
    severity: Severity
    root: RuleScopes
    when: list[Row] = Field(default_factory=list)
    # `assert` is a Python keyword: YAML uses it, the model field cannot.
    assert_: list[Row] = Field(default_factory=list, alias="assert")

    @model_validator(mode="after")
    def _check_shape(self) -> Check:
        if not (self.when or self.assert_):
            raise ValueError("check takes at least one `when` or `assert` row")
        return self


class Cohort:
    """Every check a cohort defines: one YAML per check under one directory.

    `Cohort.from_directory("examples/cohort")` -> that cohort's checks.
    A file that is not a check, a duplicate `id` or a missing directory raises:
    a load problem, not a report.
    """

    __slots__ = ("checks",)

    def __init__(self, checks: tuple[Check, ...]) -> None:
        self.checks = checks

    @classmethod
    def from_directory(cls, root: Path) -> Cohort:
        """The checks under `root`; each `*.yaml` is one `Check`, sorted by filename.

        Duplicate `id` raises ValueError: which file wins must never be a race.
        """
        if not root.is_dir():
            raise NotADirectoryError(root)
        checks = tuple(_load_check(path) for path in sorted(root.glob("*.yaml")))
        seen: set[str] = set()
        for check in checks:
            if check.id in seen:
                raise ValueError(f"duplicate check id {check.id!r}")
            seen.add(check.id)
        return cls(checks)


def _load_check(path: Path) -> Check:
    """The single `Check` a YAML file describes; parse problems name the file."""
    text = path.read_text(encoding="utf-8")
    data = yaml.load(text, Loader=_StrictLoader)
    try:
        return Check.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"{path.name} is not a check: {exc}") from exc
