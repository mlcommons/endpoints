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

A row decodes from YAML and evaluates against the scoped fields of one node (a
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

from inference_endpoint.validation.operations import Op, Predicate, vet_pattern
from inference_endpoint.validation.ranges import Interval
from inference_endpoint.validation.sentinels import MISSING


@dataclass(frozen=True)
class Ref:
    """A dotted reference into the scoped fields, e.g. `point.result_summary.tps`."""

    # The scope roots that open a reference; a class constant, not a field.
    ROOTS = (
        "submission",
        "system",
        "curve",
        "point",
        "derived",
        "ruleset",
        "regions",
    )

    parts: tuple[str, ...]

    @classmethod
    def from_string(cls, value: str) -> Ref | None:
        """A `Ref` from a scope-rooted dotted string, else None for a literal.

        `point.result_summary.tps` -> `Ref(("point", "result_summary", "tps"))`,
        which locates that field under the `point` scope.
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
        `point.tps` -> (True, 1); `point.nope` -> (False, None): no such field;
        `point.bad` -> (True, None): the field exists and is null.
        Scope roots read nested mappings or node objects alike, so rows test
        against plain dicts here and typed nodes at run time.
        """
        obj: Any = scopes
        for part in self.parts:
            if isinstance(obj, Mapping):
                if part not in obj:
                    return False, None
                obj = obj[part]
            elif not part.startswith("_") and hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return False, None
        return True, obj

    def __str__(self) -> str:
        return ".".join(self.parts)


def _one_edit_apart(left: str, right: str) -> bool:
    """Whether the strings differ by exactly one insert/delete/substitute.

    Typo detector for scope roots (`submisson.` is one edit from `submission.`
    while `pointish.` is two): a dotted operand that would silently decode as a
    literal raises instead when its head is a near-miss of a root.
    """
    if left == right or abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right, strict=True)) == 1
    short, long = sorted((left, right), key=len)
    i = j = edits = 0
    while i < len(short) and j < len(long):
        if short[i] == long[j]:
            i += 1
        else:
            edits += 1
        j += 1
    return edits + (len(long) - j) == 1


def _near_root_misspelling(head: str) -> str | None:
    """The scope root one edit away from `head`, else None."""
    for root_name in Ref.ROOTS:
        if _one_edit_apart(head, root_name):
            return root_name
    return None


def _decode_operand(value: Any) -> Any:
    """YAML decode hook on every `Operand`, run at model-validate time.

    `"point.tps"` -> `Ref(("point", "tps"))` (a root-prefixed dotted string);
    `["ruleset", "models", "llama3.1-8b-v1"]` -> a `Ref` over raw segments, for
    keys that themselves contain dots (only a non-empty all-strings list headed
    by a scope root decodes this way: every other list is a literal collection);
    `{"literal": ...}` -> that value verbatim, for a string that would
    otherwise read as a reference. `"tps"` or `7` -> the literal itself. A
    dotted scalar whose head is one edit from a root raises rather than
    silently reading as a literal.
    """
    if isinstance(value, Ref):
        return value
    if (
        isinstance(value, list)
        and value
        and all(isinstance(part, str) and part for part in value)
    ):
        head = value[0]
        if head in Ref.ROOTS:
            return Ref(tuple(value))
        near = _near_root_misspelling(head)
        if near is not None:
            raise ValueError(
                f"{value!r} looks like a misspelled {near!r} reference; "
                'fix the root or write {"literal": ...} to mean the value'
            )
        return value
    if isinstance(value, dict) and set(value) == {"literal"}:
        return value["literal"]
    if isinstance(value, str):
        ref = Ref.from_string(value)
        if ref is not None:
            return ref
        head, dot, _ = value.partition(".")
        if dot:
            near = _near_root_misspelling(head)
            if near is not None:
                raise ValueError(
                    f"{value!r} looks like a misspelled {near!r} reference; "
                    'fix the root or write {"literal": ...} to mean the string'
                )
        return value
    return value


Operand = Annotated[Any, BeforeValidator(_decode_operand)]
MeasureName = Literal["count", "sum", "mean", "min", "max", "any", "all"]
MeasureOp = Literal["eq", "ne", "gt", "ge", "lt", "le"]


def _show(value: Any, limit: int = 80) -> str:
    # `Ref` renders its dotted form, not its dataclass repr: messages quote the
    # row's own operand syntax.
    text = str(value) if isinstance(value, Ref) else repr(value)
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
        op = Op(self.op)
        want = set(op.slots)
        used = {name for name in ("right", "lo", "hi") if name in self.model_fields_set}
        if used != want:
            raise ValueError(f"`{self.op}` takes `left` plus {sorted(want) or 'none'}")
        if self.op == "matches":
            if not (isinstance(self.right, str) and not isinstance(self.right, Ref)):
                raise ValueError("`matches` needs `right` as a literal regex string")
            vet_pattern(self.right)
        if not op.tolerant and (self.abs_tol or self.rel_tol):
            raise ValueError(
                f"`{self.op}` has no tolerance semantics (use `eq`/`ne` for abs_tol/rel_tol)"
            )
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
            got = _show(value) if found else "absent"
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
            target: Any = f"{self.lo}..{self.hi}"
        else:
            target = self.right
        return False, (
            f"expected {self.left} {self.op} {target}, "
            f"got {_show(left)} vs {_show(right)}"
        )

    def describe(self) -> str:
        """The row as prose: `left op right`, `left between lo..hi`, or `left op`.

        The row label every boundary message reuses: one shape per row kind, so
        a message never reaches for fields a row does not have.
        """
        op = Op(self.op)
        if op.slots == ("lo", "hi"):
            return f"{_show(self.left)} {self.op} {_show(self.lo)}..{_show(self.hi)}"
        if op.slots == ():
            return f"{_show(self.left)} {self.op}"
        return f"{_show(self.left)} {self.op} {_show(self.right)}"

    def operands(self) -> tuple[Any, ...]:
        """Every operand this row resolves; its `Ref`s live among them."""
        return (self.left, self.right, self.lo, self.hi)


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
                measure = data["measure"]
            except KeyError:
                raise ValueError("aggregate row needs a `measure`") from None
            try:
                return super().__new__(cls.registry[measure])
            except KeyError:
                raise ValueError(f"unknown measure {measure!r}") from None
        return super().__new__(cls)

    def __init_subclass__(cls, *, measure: str, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.registry[measure] = cls

    @model_validator(mode="after")
    def _check_shape(self) -> Aggregate:
        # The row must run its declared measure: `Count(measure="sum")` would
        # compute `len` while its messages claim `sum`.
        if Aggregate.registry.get(self.measure) is not type(self):
            raise ValueError(
                f"row computes {type(self).__name__} but declares measure {self.measure!r}"
            )
        op = Op(self.op)
        if not op.tolerant and (self.abs_tol or self.rel_tol):
            raise ValueError(
                f"`{self.op}` has no tolerance semantics (use `eq`/`ne` for abs_tol/rel_tol)"
            )
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

    def describe(self) -> str:
        """The row as prose: `measure of over op value`.

        The row label `Verdict.evaluate`'s boundary message reuses (the
        `Compare` twin keeps its own shape).
        """
        return f"{self.measure} of {_show(self.over)} {self.op} {_show(self.value)}"

    def operands(self) -> tuple[Any, ...]:
        """Every operand this row resolves; its `Ref`s live among them."""
        return (self.over, self.value)


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
    """Does a condition hold, and why not: the message is the row's own.

    `broken` marks a verdict whose row failed to *evaluate* (a deliberate
    fail-loud or an incommensurable pair) rather than judged False -- gate
    positions must refuse those loudly instead of reading them as "not
    applicable".
    """

    ok: bool
    why: str
    broken: bool = False

    @classmethod
    def evaluate(cls, row: Compare | Aggregate, scopes: Mapping[str, Any]) -> Verdict:
        """Evaluate one condition row against a node's scoped fields.

        An incommensurable pair from an untrusted artifact (a `TypeError`,
        `ValueError` or `OverflowError` escaping `op.holds`) becomes a failing
        verdict naming both sides and the exception class -- the fail-not-abort
        policy the semantics grid calls for and the same return-fail shape
        `Interval.__contains__` keeps. No artifact value ever aborts the run
        mid-verdict.
        """
        try:
            held, why = row._outcome(scopes)
        except TypeError as exc:
            return cls(
                False,
                f"incommensurable pair: {row.describe()}: {type(exc).__name__}: {exc}",
                broken=True,
            )
        except (ValueError, OverflowError) as exc:
            # A deliberate fail-loud (a wiring/shape error) is reported as what
            # it is -- never as an incommensurable pair that never happened.
            return cls(
                False,
                f"row failed to evaluate: {row.describe()}: {type(exc).__name__}: {exc}",
                broken=True,
            )
        return cls(held, why)

    @classmethod
    def evaluate_all(
        cls, rows: list[Compare | Aggregate], scopes: Mapping[str, Any]
    ) -> list[Verdict]:
        """Evaluate one condition list (a `when` or `assert` block) in order."""
        return [cls.evaluate(row, scopes) for row in rows]
