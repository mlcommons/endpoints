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

A cohort directory holds one YAML file per scope it checks
(`system.yaml`/`submission.yaml`/`curve.yaml`/`point.yaml`; a missing file is
an empty scope). Each file is a mapping of check id to its block (prose
`description`/`rule`, optional `fix`, a `severity` and its `when`/`assert`
rows): the filename says the scope and the key says the id, so a block
declaring either is a load error. Two checks in `point.yaml`:

    concurrency_in_range:
      description: Each point's concurrency lands in a valid region.
      rule: "Endpoints Rules Section 5.4, Concurrency Regions."
      severity: warn
      assert:
        - left: point.config.concurrency
          op: in
          right: regions.coverage

    offline_ordering:
      description: The dedicated Offline run meets Section 5.7.2's two clauses.
      rule: "Endpoints Rules Section 5.7.2, Relationship to Maximum Supported
        Concurrency."
      fix: Either raise the dedicated run's system_tps to 0.98x the C_max
        point's, or elect the C_max point as Offline.
      severity: warn
      when:
        - left: point.config.offline
          op: eq
          right: dedicated
      assert:
        - left: derived.offline_ordering
          op: eq
          right: true

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
    `left`/`op` -> `Compare`. An already-built row passes through, and any
    construction failure (a non-identifier key, a shape error) re-raises as
    ValueError so `_load_check`'s net can name the file. A bare value raises
    too: both row kinds are models, so a scalar cannot be either.
    """
    if isinstance(value, Compare | Aggregate):
        return value
    if isinstance(value, dict):
        try:
            return Aggregate(**value) if "measure" in value else Compare(**value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bad row {value!r}: {exc}") from exc
    raise ValueError(f"expected a `Compare` or `Aggregate` row, got {value!r}")


Row = Annotated[Compare | Aggregate, BeforeValidator(_decode_row)]


class Check(BaseModel):
    """One check block: its provenance and the conditions it evaluates."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    description: str
    rule: str
    severity: Severity
    scope: RuleScopes
    fix: str | None = None
    needs: tuple[str, ...] = ()
    when: list[Row] = Field(default_factory=list)
    # `assert` is a Python keyword: YAML uses it, the model field cannot.
    assert_: list[Row] = Field(default_factory=list, alias="assert")

    @model_validator(mode="after")
    def _check_shape(self) -> Check:
        if not (self.when or self.assert_):
            raise ValueError("check takes at least one `when` or `assert` row")
        return self


class Cohort:
    """Every check a cohort defines: one YAML per scope under one directory.

    `Cohort.from_directory("examples/cohort")` -> that cohort's checks.
    A stray file in the directory, a check block declaring its own `id`/`scope`,
    a duplicate id or a missing directory raises: a load problem, not a report.
    """

    __slots__ = ("checks",)

    def __init__(self, checks: tuple[Check, ...]) -> None:
        self.checks = checks

    @classmethod
    def from_directory(cls, root: Path) -> Cohort:
        """The checks under `root`; scope files in `RuleScopes` order, then key order.

        Duplicate `id` raises ValueError: which file wins must never be a race.
        """
        if not root.is_dir():
            raise NotADirectoryError(root)
        stray = sorted(
            {path.name for path in root.iterdir()}
            - {f"{scope.value}.yaml" for scope in RuleScopes}
        )
        if stray:
            raise ValueError(
                f"{root} is not a cohort: unexpected {stray} "
                "(only system.yaml/submission.yaml/curve.yaml/point.yaml belong here)"
            )
        checks: list[Check] = []
        for scope in RuleScopes:
            checks.extend(_load_scope_checks(root / f"{scope.value}.yaml", scope))
        if not checks:
            raise ValueError(f"{root} holds no checks")
        seen: set[str] = set()
        for check in checks:
            if check.id in seen:
                raise ValueError(f"duplicate check id {check.id!r}")
            seen.add(check.id)
        return cls(tuple(checks))


def _load_scope_checks(path: Path, scope: RuleScopes) -> tuple[Check, ...]:
    """Every `Check` one scope file describes; the mapping key becomes the id.

    A missing file means an empty scope. Every parse problem names the file.
    """
    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.load(text, Loader=_StrictLoader)
        if data is None:
            raise ValueError("empty scope file")
        if not isinstance(data, dict):
            raise ValueError("expected a mapping of check id to check")
        return tuple(_check_from_entry(key, body, scope) for key, body in data.items())
    except FileNotFoundError:
        return ()
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"{path.name} is not a checks file: {exc}") from exc


def _check_from_entry(check_id: str, body: Any, scope: RuleScopes) -> Check:
    """One id-keyed block as a `Check`; the key and file name its id and scope."""
    if not isinstance(body, dict):
        raise ValueError(f"check {check_id!r} must be a mapping")
    claimed = sorted({"id", "scope"} & body.keys())
    if claimed:
        raise ValueError(
            f"check {check_id!r} defines its own {claimed}: the key is the id "
            "and the filename is the scope"
        )
    return Check.model_validate({**body, "id": check_id, "scope": scope.value})
