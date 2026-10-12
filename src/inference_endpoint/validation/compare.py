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

"""Condition rows: every check condition is a `Compare` or an `Aggregate`.

A row decodes from YAML and evaluates against the scoped facts of one node (a
mapping per scope root, e.g. `{"point": ...}`). Operands are literals or `Ref`
dotted references (decode hooks below). Which op does what lives in
`operations.py` (`Op` subclasses); which aggregation `measure` does what is the
`Aggregate` subclass family below. `Verdict.evaluate` wraps one row's own
`_outcome`, whose `why` is written by the row (the failing values, bounded), so
no downstream code has to reconstruct the message.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, model_validator

from inference_endpoint.validation.operations import Op, Predicate
from inference_endpoint.validation.ranges import Interval
from inference_endpoint.validation.sentinels import MISSING


@dataclass(frozen=True)
class Ref:
    """A dotted reference into the scoped facts, e.g. `point.result_summary.tps`."""

    # The scope roots that open a reference; a class constant, not a field.
    ROOTS = ("submission", "system", "curve", "point", "ruleset", "cohort")

    parts: tuple[str, ...]

    @classmethod
    def from_string(cls, value: str) -> Ref | None:
        """A `Ref` from a scope-rooted dotted string, else None for a literal.

        `point.result_summary.tps` -> `Ref(("point", "result_summary", "tps"))`,
        which locates that fact under the `point` scope.
        `result_summary.tps` -> None: not rooted in a valid scope object root,
        so YAML keeps it as the literal string. `point.` and `point` -> None too.
        """
        head, dot, rest = value.partition(".")
        if dot and head in cls.ROOTS and rest:
            return cls(tuple(value.split(".")))
        return None

    def resolve(self, scopes: Mapping[str, Any]) -> tuple[bool, Any]:
        """(found, value) of this reference: a missing key is absent, a stored null is a value.

        Against `{"point": {"tps": 1, "bad": None}}`:
        `point.tps` -> (True, 1); `point.nope` -> (False, None): no such fact;
        `point.bad` -> (True, None): the fact exists and is null.
        Scope roots read mappings or fact objects alike, so rows test against
        plain dicts here and typed nodes at run time.
        """
        obj: Any = scopes
        for part in self.parts:
            if isinstance(obj, Mapping):
                if part not in obj:
                    return False, None
                obj = obj[part]
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return False, None
        return True, obj

    def __str__(self) -> str:
        return ".".join(self.parts)


def _decode_operand(value: Any) -> Any:
    """YAML decode hook on every `Operand`, run at model-validate time.

    `"point.tps"` -> `Ref(("point", "tps"))`; `"tps"` or `7` -> the literal itself.
    """
    ref = Ref.from_string(value) if isinstance(value, str) else None
    return ref or value


Operand = Annotated[Any, BeforeValidator(_decode_operand)]
MeasureName = Literal["count", "sum", "mean", "min", "max", "any", "all"]
MeasureOp = Literal["eq", "ne", "gt", "ge", "lt", "le"]


def _show(value: Any, limit: int = 80) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class Compare(BaseModel):
    """One condition row: `op` applied to `left` and (per op) `right`/`lo`/`hi`.

    A YAML row this decodes from, inside a check's `when` or `assert` block:

        when:
          - left: point.result_summary.n_samples_completed
            op: eq
            right: point.result_summary.n_samples_issued

    Arity follows the op (`Op.slots` in `operations.py`): `exists`, `absent` and
    `not_null` take `left` alone; `between` takes `lo` and `hi`; the rest take
    `right`. A `right` given at all to a left-only op is a shape error.
    `abs_tol`/`rel_tol` reach `eq` and `ne` only.
    """

    model_config = ConfigDict(extra="forbid")

    op: str
    left: Operand
    right: Operand | None = None
    lo: Operand | None = None
    hi: Operand | None = None
    abs_tol: float = 0.0
    rel_tol: float = 0.0

    @model_validator(mode="after")
    def _check_shape(self) -> Compare:
        # Presence, not truthiness: `right: null` is a real operand (eq against
        # a literal null is legal), so arity is decided by field presence.
        want = set(Op(self.op).slots)
        used = {name for name in ("right", "lo", "hi") if name in self.model_fields_set}
        if used != want:
            raise ValueError(f"`{self.op}` takes `left` plus {sorted(want) or 'none'}")
        return self

    def _outcome(self, scopes: Mapping[str, Any]) -> tuple[bool, str]:
        """(held, why): presence rows read `left` directly; the rest compare operands.

        A missing reference fails the row (its `why` names it) except where the
        op asks about presence itself. `_show` bounds each failing value.
        """
        op = Op(self.op)
        if op.needs_found:
            found, value = (
                self.left.resolve(scopes)
                if isinstance(self.left, Ref)
                else (True, self.left)
            )
            held = op.holds(Predicate(value if found else MISSING, None))
            if held:
                return True, ""
            got = repr(value) if found else "absent"
            return False, f"expected {self.left} {op.label}, got {got}"

        out: list[Any] = []
        missing = []
        for side in (self.left, self.right, self.lo, self.hi):
            if isinstance(side, Ref):
                found, value = side.resolve(scopes)
                out.append(value)
                if not found:
                    missing.append(str(side))
            else:
                out.append(side)
        if missing:
            return False, f"missing {', '.join(missing)}"
        left, right, lo, hi = out
        if Op(self.op).slots == ("lo", "hi"):
            predicate = Predicate(left, Interval(lo, hi))
        else:
            predicate = Predicate(left, right)
        held = op.holds(predicate, self.abs_tol, self.rel_tol)
        if held:
            return True, ""
        # The message's operand form follows the row's slots: a literal `None`
        # on `right` is a real operand, never the interval placeholder.
        if Op(self.op).slots == ("lo", "hi"):
            target = f"{self.lo}..{self.hi}"
        else:
            target = self.right
        return False, (
            f"expected {self.left} {self.op} {target}, "
            f"got {_show(left)} vs {_show(right)}"
        )


class Aggregate(BaseModel):
    """One collection row: `measure` of `over`, compared to `value`.

    A YAML row this decodes from:

        when:
          - over: point.results.accuracy
            measure: count
            op: ge
            value: 1

    `over` reads a list, or the values of a mapping. `measure` picks the
    subclass registered for it (see `__new__`): `Count` over an empty collection
    is 0 and `AnyTrue` is False, while `Sum`, `Mean`, `Min`, `Max` and `AllTrue`
    fail with `no values` over empty rather than guessing (a vacuous `all` pass
    on an empty upload would be a silent false-pass). `op` is one of the scalar
    ops (`MeasureOp` above).
    """

    model_config = ConfigDict(extra="forbid")

    # kind -> subclass, filled by `__init_subclass__` on each measure family.
    registry: ClassVar[dict[str, type[Aggregate]]] = {}

    over: Operand
    measure: MeasureName
    op: MeasureOp
    value: Operand
    abs_tol: float = 0.0
    rel_tol: float = 0.0

    defines_empty: ClassVar[bool] = False

    def __new__(cls, **data: Any) -> Aggregate:
        """`Aggregate(measure="count", ...)` -> `Count(...)` over the registered family."""
        if cls is Aggregate:
            try:
                return super().__new__(cls.registry[data["measure"]])
            except KeyError:
                raise ValueError(f"unknown measure {data['measure']!r}") from None
        return super().__new__(cls)

    def __init_subclass__(cls, *, measure: str, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.registry[measure] = cls

    @model_validator(mode="after")
    def _check_shape(self) -> Aggregate:
        if "value" not in self.model_fields_set:
            raise ValueError("`Aggregate` takes `over`, `measure`, `op` and `value`")
        return self

    def run(self, items: list[Any]) -> Any:
        """The aggregation itself over `items`; a subclass owns one.

        `Count().run([1, 2])` -> 2. Called on an empty `items` raises
        NotImplementedError: the empty-collection policy sits in `_outcome`.
        """
        raise NotImplementedError

    def _outcome(self, scopes: Mapping[str, Any]) -> tuple[bool, str]:
        """(held, why) of `measure(over) op value`; `why` names the measured result."""
        resolved = []
        missing = []
        for side in (self.over, self.value):
            if isinstance(side, Ref):
                found, value = side.resolve(scopes)
                resolved.append(value)
                if not found:
                    missing.append(str(side))
            else:
                resolved.append(side)
        if missing:
            return False, f"missing {', '.join(missing)}"
        over, value = resolved
        items = (
            list(over.values())
            if isinstance(over, Mapping)
            else (
                list(over)
                if isinstance(over, Sequence) and not isinstance(over, str)
                else None
            )
        )
        if items is None:
            return False, f"expected {self.over} to be a collection, got {_show(over)}"
        if not items and not self.defines_empty:
            return False, f"expected {self.measure} of {self.over}, got no values"
        measured = self.run(items)
        held = Op(self.op).holds(Predicate(measured, value), self.abs_tol, self.rel_tol)
        if held:
            return True, ""
        return False, (
            f"expected {self.measure} of {self.over} {self.op} {self.value}, "
            f"got {_show(measured)} vs {_show(value)}"
        )


class Count(Aggregate, measure="count"):
    """`len(items)`: `Count().run([1, 2])` -> 2; empty -> 0."""

    defines_empty = True

    def run(self, items: list[Any]) -> Any:
        return len(items)


class Sum(Aggregate, measure="sum"):
    """`sum(items)`: `Sum().run([1, 2, 3])` -> 6."""

    def run(self, items: list[Any]) -> Any:
        return sum(items)


class Mean(Aggregate, measure="mean"):
    """`sum(items) / len(items)`: `Mean().run([1, 3])` -> 2.0."""

    def run(self, items: list[Any]) -> Any:
        return sum(items) / len(items)


class Min(Aggregate, measure="min"):
    """`min(items)`: `Min().run([2, 1, 3])` -> 1."""

    def run(self, items: list[Any]) -> Any:
        return min(items)


class Max(Aggregate, measure="max"):
    """`max(items)`: `Max().run([2, 1, 3])` -> 3."""

    def run(self, items: list[Any]) -> Any:
        return max(items)


class AnyTrue(Aggregate, measure="any"):
    """`any(items)` over truthiness: `AnyTrue().run([0, 5])` -> True; empty -> False."""

    defines_empty = True

    def run(self, items: list[Any]) -> Any:
        return any(items)


class AllTrue(Aggregate, measure="all"):
    """`all(items)` over truthiness: `AllTrue().run([1, 0])` -> False (empty fails in `_outcome`)."""

    def run(self, items: list[Any]) -> Any:
        return all(items)


@dataclass(frozen=True)
class Verdict:
    """Does a condition hold, and why not: the message is the row's own."""

    ok: bool
    why: str

    @classmethod
    def evaluate(cls, row: Compare | Aggregate, scopes: Mapping[str, Any]) -> Verdict:
        """Evaluate one condition row against a node's scoped facts."""
        held, why = row._outcome(scopes)
        return cls(held, why)

    @classmethod
    def evaluate_all(
        cls, rows: list[Compare | Aggregate], scopes: Mapping[str, Any]
    ) -> list[Verdict]:
        """Evaluate one condition list (a `when` or `assert` block) in order."""
        return [cls.evaluate(row, scopes) for row in rows]
