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
(`system.yaml`/`submission.yaml`/`pareto.yaml`/`point.yaml`. A missing file is
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

`when` gates the rule (all rows must hold for the `assert` rows to speak).
`assert` is what the rule holds to. `Cohort.from_directory` gathers every
check in a cohort directory.
"""

from __future__ import annotations

from enum import Enum
from functools import partial
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
from inference_endpoint.validation.refs import Ref, RefTable
from inference_endpoint.validation.regions import RegionKind
from inference_endpoint.validation.submission import DERIVATION_NAMES
from inference_endpoint.validation.tree import RuleScopes, StrictLoader

_SCOPE_DEPTH = {
    RuleScopes.SUBMISSION: 0,
    RuleScopes.SYSTEM: 1,
    RuleScopes.PARETO: 2,
    RuleScopes.POINT: 3,
}

_REGION_PATHS = {("coverage",)} | {("by_kind", kind.value) for kind in RegionKind}


def _validate_row_ref(scope: RuleScopes, ref: Ref) -> None:
    """Refuse a row naming no fact fetchable at `scope`. ValueError says why.

    The `derived`/`regions` folds exist per pareto, so only pareto- and
    point-scoped rows may name them. A row naming a finer scope than its own
    (`system.yaml` reading `point.` facts) would resolve nowhere at its nodes
    and is refused for the same reason. Field names under a document are
    checked against the cohort's `valid_refs.yaml` (`_checked_ref`).

    Args:
        scope (RuleScopes): The scope of the check whose row names `ref`.
        ref (Ref): The row's reference: `parts[0]` is the root, the rest name the
            fact under it.
    Raises:
        ValueError: When a `derived`/`regions` ref is named above pareto scope, a
            derivation or region fact is unknown, or the named fact resolves only below
            a node finer than the check's scope.
    """
    head = ref.parts[0]
    if head == "derived":
        if scope not in (RuleScopes.PARETO, RuleScopes.POINT):
            raise ValueError(
                f"check scoped {scope.value} cannot name {ref}: `derived` is a "
                "per-pareto fact fold (write the check in pareto.yaml/point.yaml)"
            )
        if len(ref.parts) != 2 or ref.parts[1] not in DERIVATION_NAMES:
            raise ValueError(f"{ref} names no derivation")
        return
    if head == "regions":
        if scope not in (RuleScopes.PARETO, RuleScopes.POINT):
            raise ValueError(
                f"check scoped {scope.value} cannot name {ref}: `regions` is a "
                "per-pareto fact fold (write the check in pareto.yaml/point.yaml)"
            )
        if tuple(ref.parts[1:]) not in _REGION_PATHS:
            raise ValueError(f"{ref} names no region fact")
        return
    if head == "ruleset":
        return  # ruleset facts are the catalog's own data keys
    kind = RuleScopes(head)
    if _SCOPE_DEPTH[kind] > _SCOPE_DEPTH[scope]:
        raise ValueError(
            f"check scoped {scope.value} cannot name {ref}: it resolves only "
            f"below a {kind.value} node (write the check in {kind.value}.yaml)"
        )


def _checked_ref(scope: RuleScopes, ref: Ref, table: RefTable) -> Ref:
    """`ref` checked at `scope`, annotated with the table's policy notes.

    The one place a row's names get checked: `table.validate_names` refuses a
    misspelled or over-deep fact (its "did you mean" hint included) and
    `table.annotate` fills `Ref.required`/`Ref.default`, the two notes the
    evaluation policy reads. Fold-root refs (`derived`/`regions`/`ruleset`)
    skip the table: their names are checker-owned and `_validate_row_ref` has
    already vetted them. One wrong name dies at cohort load rather than
    silently switching a `when`-gated check off against some submission.

    Args:
        scope (RuleScopes): The scope of the check whose row names `ref`.
        ref (Ref): The row's reference, before the table has annotated it.
        table (RefTable): The cohort's `valid_refs.yaml` ref table.
    Returns:
        Ref: `ref` with `required`/`default` filled in from the table. A
            fold-root ref comes back untouched.
    Raises:
        ValueError: When `_validate_row_ref` or the table refuses the ref: a
            wrong name dies at cohort load, never at run time.
    """
    _validate_row_ref(scope, ref)
    if ref.parts[0] in Ref.FOLD_ROOTS:
        return ref
    table.validate_names(ref.parts)
    return table.annotate(ref)


class Severity(str, Enum):  # noqa: UP042 - repository wire-enum convention
    """What a violated `assert` row costs the submission.

    `fail` is a blocking finding. `warn` is reported and passes through.
    """

    FAIL = "fail"
    WARN = "warn"


def _decode_row(value: Any) -> Any:
    """YAML decode hook on every `when`/`assert` row, before model validation.

    `{"over": ..., "aggregate": "count", ...}` -> `Aggregate`. Anything else with
    `left`/`op` -> `Compare`. An already-built row passes through, and any
    construction failure (a non-identifier key, a shape error) re-raises as
    ValueError so `Cohort.from_directory` can name the file. A bare value raises
    too: both row kinds are models, so a scalar cannot be either.

    Args:
        value (Any): The row as YAML decoded it: a mapping of the row's fields,
            or an already-built `Compare` /`Aggregate`.
    Returns:
        Compare | Aggregate: The row as a model: `Aggregate` when it carries a
            `aggregate` field, `Compare` otherwise.
    Raises:
        ValueError: When the mapping builds neither row model (a non-identifier
            key, a shape error), or when `value` is neither row kind.
    """
    if isinstance(value, Compare | Aggregate):
        return value
    if isinstance(value, dict):
        try:
            return Aggregate(**value) if "aggregate" in value else Compare(**value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bad row {value!r}: {exc}") from exc
    raise ValueError(f"expected a `Compare` or `Aggregate` row, got {value!r}")


Row = Annotated[Compare | Aggregate, BeforeValidator(_decode_row)]


class Check(BaseModel):
    """One check block: its provenance and the conditions it evaluates.

    Args:
        id (str): The check's id: the cohort YAML key, never a block-declared one.
        description (str): The human sentence the report shows for the finding.
        rule (str): Where in the Endpoints Rules the check comes from.
        severity (Severity): What a violated `assert` row costs the submission
            (`fail` blocks, `warn` passes through).
        scope (RuleScopes): The node kind the check evaluates at: the scope file's
            name, never a block-declared one.
        fix (str | None): How to resolve a failing report. (Default: None)
        needs (tuple[str, ...]): Upstream check ids that must vouch before this
            check's rows run. (Default: ())
        when (list[Row]): Gate predicates: the check evaluates only where all hold.
            (Default: [])
        assert_ (list[Row]): Condition predicates: each is one finding per instance.
            (Default: [])
    """

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

    `Cohort.from_directory("examples/cohort")` -> that cohort's checks and its
    `valid_refs.yaml` ref table (what its rows may name, and how an omitted
    fact reads). A hand-built `Cohort((check, ...))` carries no table
    (`refs is None`): its rows' names are checked only by resolution at run
    time.

    Args:
        checks (tuple[Check, ...]): The cohort's checks.
        refs (RefTable | None): The `valid_refs.yaml` ref table, or `None` for a
            hand-built cohort. (Default: None)
    """

    __slots__ = ("checks", "refs")

    def __init__(self, checks: tuple[Check, ...], refs: RefTable | None = None) -> None:
        self.checks = checks
        self.refs = refs

    @classmethod
    def from_directory(cls, root: Path) -> Cohort:
        """The checks under `root`. Scope files are read in `RuleScopes` order,
        then each file's checks in key order.

        Args:
            root (Path): The cohort directory: the scope files plus the
                `valid_refs.yaml` ref table.
        Returns:
            Cohort: The cohort's checks, every row's names checked against the
                table.
        Raises:
            NotADirectoryError: When `root` is not a directory.
            ValueError: When the directory holds a stray file, the ref table is
                missing, a scope file is not a mapping of check id to check, a check
                block declares its own `id` /`scope`, or no checks are found: a load
                problem, not a report. A duplicate `id` is refused here: which file wins
                must never be a race.
        """
        if not root.is_dir():
            raise NotADirectoryError(root)
        stray = sorted(
            {path.name for path in root.iterdir()}
            - {RefTable.FILENAME, *(f"{scope.value}.yaml" for scope in RuleScopes)}
        )
        if stray:
            raise ValueError(
                f"{root} is not a cohort: unexpected {stray} "
                f"(only {RefTable.FILENAME} and the scope files system.yaml/"
                "submission.yaml/pareto.yaml/point.yaml belong here)"
            )
        refs = RefTable.from_yaml(root / RefTable.FILENAME)
        checks: list[Check] = []
        for scope in RuleScopes:
            path = root / f"{scope.value}.yaml"
            try:
                text = path.read_text(encoding="utf-8")
                data = yaml.load(text, Loader=StrictLoader)
                if data is None:
                    raise ValueError("empty scope file")
                if not isinstance(data, dict):
                    raise ValueError("expected a mapping of check id to check")
                for check_id, body in data.items():
                    # One id-keyed block as a `Check`; the key and file name its
                    # id and scope.
                    if not isinstance(body, dict):
                        raise ValueError(f"check {check_id!r} must be a mapping")
                    claimed = sorted({"id", "scope"} & body.keys())
                    if claimed:
                        raise ValueError(
                            f"check {check_id!r} defines its own {claimed}: the key "
                            "is the id and the filename is the scope"
                        )
                    checks.append(
                        Check.model_validate(
                            {**body, "id": check_id, "scope": scope.value}
                        )
                    )
            except FileNotFoundError:
                continue  # a missing scope file means an empty scope
            except (OSError, UnicodeDecodeError, yaml.YAMLError, ValueError) as exc:
                raise ValueError(f"{path.name} is not a checks file: {exc}") from exc
        if not checks:
            raise ValueError(f"{root} holds no checks")
        seen: set[str] = set()
        for check in checks:
            if check.id in seen:
                raise ValueError(f"duplicate check id {check.id!r}")
            seen.add(check.id)
        for check in checks:
            for row in (*check.when, *check.assert_):
                row.map_refs(partial(_checked_ref, check.scope, table=refs))
        return cls(tuple(checks), refs)
