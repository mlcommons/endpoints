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
`operations.py` (`Op` subclasses). Which aggregate does what is the
`Aggregate` subclass family below. `Verdict.evaluate` wraps one row's own
`_outcome`, whose `why` is written by the row (the failing values, bounded), so
no downstream code has to reconstruct the message.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from inference_endpoint.validation.operations import Op, Predicate, vet_pattern
from inference_endpoint.validation.ranges import Interval
from inference_endpoint.validation.refs import Ref
from inference_endpoint.validation.sentinels import MISSING
from inference_endpoint.validation.typo_checks import near_match


def _rewrite_refs(row: Any, slots: tuple[str, ...], fn: Callable[[Ref], Ref]) -> None:
    """Replace each `Ref` operand of `row`'s `slots` with `fn`'s result.

    The slot names live here (one list per row shape) so no outside caller
    reaches into row fields to find the operands.

    Args:
        row (Any): A `Compare` or an `Aggregate`: the row whose `slots` fields
            are rewritten in place.
        slots (tuple[str, ...]): The row's operand-bearing field names, one
            tuple per row shape.
        fn (Callable[[Ref], Ref]): The rewrite applied to each operand that is
            a `Ref`.
    """
    for slot in slots:
        value = getattr(row, slot, None)
        if isinstance(value, Ref):
            setattr(row, slot, fn(value))


def _near_root_misspelling(head: str) -> str | None:
    """The scope root one edit from `head`, else None (`submisson` -> `submission`).

    A dotted operand that would silently decode as a literal raises instead
    when its head is a near-miss of a root. A direct hit is not a typo and
    returns None. The matching itself is `typo_checks.near_match`, shared with
    the ref table's fact names.

    Args:
        head (str): The head segment of a dotted operand, e.g. `submisson` in
            `submisson.tps`.
    Returns:
        str | None: The scope root one edit from `head` (`submission` above), or
            `None` when `head` misses every other root.
    """
    return near_match(head, (root for root in Ref.ROOTS if root != head))


def _decode_operand(value: Any) -> Any:
    """YAML decode hook on every `Operand`, run at model-validate time.

    Example:

        _decode_operand("point.tps")  # Ref(("point", "tps"))
        _decode_operand(["ruleset", "models", "llama3.1-8b-v1"])  # Ref, raw segments
        _decode_operand({"literal": "point.tps"})  # "point.tps"
        _decode_operand("tps")  # "tps"
        _decode_operand(7)  # 7

    `"point.tps"` is a root-prefixed dotted string decoded as `Ref(("point", "tps"))`,
    and `["ruleset", "models", "llama3.1-8b-v1"]` is a raw-segment list for keys
    that themselves contain dots (only a non-empty all-strings list headed by a
    scope root decodes this way: every other list is a literal collection).
    `{"literal": ...}` is that value verbatim, for a string that would
    otherwise read as a reference. `"tps"` or `7` is the literal itself. A
    dotted scalar whose head is one edit from a root raises rather than
    silently reading as a literal.

    Args:
        value (Any): The raw YAML value of one operand slot.
    Returns:
        Any: A `Ref`, the literal itself, or `{"literal": ...}`'s value.
    Raises:
        ValueError: When a dotted scalar's or list's head is one edit from a
            scope root (likely a misspelling), or is dotted with no scope root at all.
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
            raise ValueError(
                f"{value!r} is dotted but {head!r} is no scope root; "
                'write {"literal": ...} to mean the value'
            )
        return value
    return value


Operand = Annotated[Any, BeforeValidator(_decode_operand)]
AggregateName = Literal["count", "sum", "mean", "min", "max", "any", "all"]
AggregateOp = Literal["eq", "ne", "gt", "ge", "lt", "le"]


def _show(value: Any, limit: int = 80) -> str:
    # `Ref` renders its dotted form, not its dataclass repr: messages quote the
    # row's own operand syntax.
    text = str(value) if isinstance(value, Ref) else repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _operand(ref: Ref, scopes: Mapping[str, Any]) -> tuple[bool, Any]:
    """(found, value) of a comparison operand, materializing its declared default.

    A fact whose cohort declaration spells its absent-state value
    (`Ref.default`, today §8.3's `offline`: "absent or `none`") judges as that
    value when the upload omits it -- so `offline ne dedicated` holds for a
    non-Offline point instead of going dark. Presence rows never come here:
    `exists`/`absent`/`not_null` key on the key's raw absence.

    Args:
        ref (Ref): The operand's reference to resolve.
        scopes (Mapping[str, Any]): The node's scoped fields, one mapping per
            scope root.
    Returns:
        tuple[bool, Any]: `(found, value)`: a non-`MISSING` `Ref.default` is
            materialized as found when the upload omits the fact.
    """
    found, value = ref.resolve(scopes)
    if found or ref.default is MISSING:
        return found, value
    return True, ref.default


def _missing_outcome(missing: Sequence[Ref]) -> tuple[bool, str]:
    """A row's verdict when one of its operands is absent from the upload.

    A **required** fact (`Ref.required`, filled from the cohort's
    `valid_refs.yaml`) absent is a schema breach, and raises -- `Verdict.evaluate`
    maps it to a `broken` verdict, so a `when` gate refuses loudly instead of
    reading "not applicable". Any other absent fact is the declaration's own
    not-declared state: the row does not hold, and a `when` row over it gates
    its check off by declared semantics (a check meant to test presence says
    so with `exists`/`absent`/`not_null`).

    Args:
        missing (Sequence[Ref]): Every operand `Ref` absent from the upload, in
            row-slot order.
    Returns:
        tuple[bool, str]: The row's `(held, why)`: `False` with the missing
            refs named, once no absent operand is required.
    Raises:
        ValueError: When any absent operand is a required fact (`Ref.required`):
            the schema breach raises for `Verdict.evaluate` to map to a `broken`
            verdict.
    """
    for ref in missing:
        if ref.required:
            raise ValueError(f"required fact {ref} is absent from the upload")
    return False, f"missing {', '.join(str(ref) for ref in missing)}"


def _finite(value: Any, where: str) -> Any:
    """`value` if a comparison can carry it, else refuse (ValueError -> broken).

    `NaN`/`Infinity` decode from `json.loads` (`NaN`/`Infinity`) and YAML
    (`.nan`, `.inf`); a comparison over one answers quietly False and would
    switch a `when` gate off by omission. Refused here, at the one place
    operands resolve, instead of patching every op.

    Args:
        value (Any): One resolved comparison operand (or an aggregate's result).
        where (str): The operand's row syntax, for the message.
    Returns:
        Any: `value`, unchanged.
    Raises:
        ValueError: When `value` is a non-finite float.
    """
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where} resolved to a non-finite {value!r}")
    return value


class Compare(BaseModel):
    """One condition row: `op` applied to `left` and the operands the op takes.

    A YAML row this decodes from, inside a check's `when` or `assert` block:

        when:
          - left: point.result_summary.n_samples_completed
            op: eq
            right: point.result_summary.n_samples_issued

    Arity follows the op (`Op.slots` in `operations.py`): `exists`, `absent`
    and `not_null` take `left` alone, every other op takes `right`. An `in`
    row may spell `right` as an inclusive `{"lo": ..., "hi": ...}` range
    literal; the bounds are literals (a dynamic range names a checker root
    that yields an `Interval`, e.g. `regions.coverage`).

    A `right` given at all to a left-only op is a shape error.
    `abs_tol`/`rel_tol` reach `eq` and `ne` only.

    Args:
        op (str): The YAML op name, resolved to an `Op` subclass
            (`operations.py`).
        left (Any): The compared operand, or the `Ref` whose presence a
            presence op asks about (`exists`/`absent`/`not_null`).
        right (Any | None): The compare target of every op whose slots take it,
            or an `in` row's inclusive `{"lo": ..., "hi": ...}` range literal.
            (Default: None)
        abs_tol (float): Absolute tolerance, reaching `eq`/`ne` only.
            (Default: 0.0)
        rel_tol (float): Relative tolerance, reaching `eq`/`ne` only.
            (Default: 0.0)
    """

    model_config = ConfigDict(extra="forbid")

    op: str
    left: Operand
    right: Operand | None = None
    abs_tol: float = 0.0
    rel_tol: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _matches_right_is_explicit_literal(cls, data: Any) -> Any:
        """`matches` alone requires the `{"literal": ...}` form for its pattern.

        A pattern is never a reference, so its literal-ness is not guessed from
        a dotted-string heuristic here (as `_decode_operand` does for every
        other operand). It is stated. Bare strings, dotted or not, are refused
        before `_decode_operand` can misread a regex as a reference.

        Args:
            data (Any): The raw row mapping, before field validation.
        Returns:
            Any: `data` unchanged. The `{"literal": ...}` form is asserted here.
        Raises:
            ValueError: When a `matches` row's `right` is not the
                `{"literal": ...}` form over a string pattern.
        """
        if not isinstance(data, Mapping) or data.get("op") != "matches":
            return data
        raw = data.get("right")
        if not (
            isinstance(raw, Mapping)
            and set(raw) == {"literal"}
            and isinstance(raw["literal"], str)
        ):
            raise ValueError(
                '`matches` patterns must use the {"literal": ...} form: a '
                "pattern is always a literal, never a reference"
            )
        return data

    @model_validator(mode="after")
    def _check_shape(self) -> Compare:
        # Presence, not truthiness: `right: null` is a real operand (eq against
        # a literal null is legal), so arity is decided by field presence.
        op = self._operation()
        want = set(op.slots)
        used = {"right"} if "right" in self.model_fields_set else set()
        if used != want:
            raise ValueError(f"`{self.op}` takes `left` plus {sorted(want) or 'none'}")
        if op.needs_found and not isinstance(self.left, Ref):
            raise ValueError(
                f"`{self.op}` asks about presence: `left` must be a `Reference`, "
                "not a literal value"
            )
        if self.op == "matches":
            if not (isinstance(self.right, str) and not isinstance(self.right, Ref)):
                raise ValueError("`matches` needs `right` as a literal regex string")
            vet_pattern(self.right)
        if not op.tolerant and (self.abs_tol or self.rel_tol):
            raise ValueError(
                f"`{self.op}` has no tolerance semantics (use `eq`/`ne` for abs_tol/rel_tol)"
            )
        return self

    def _operation(self) -> Op:
        """The row's `Op` instance: the one dispatch site for the string name.

        The YAML stores an op by name; binding it here keeps `_check_shape`
        and `_outcome` from re-dispatching the string at each use.
        """
        return Op(self.op)

    def _range_bounds(self) -> tuple[Any, Any] | None:
        """The `{lo, hi}` bounds of an `in` range literal, or None elsewhere.

        Only `in` reads this form. Every other mapping (and every other op)
        keeps the ordinary key-set meaning, so `in` over `{"k": 1}` still asks
        whether `left` is one of `k`'s keys. The bounds are literals, never
        references: a dynamic range names a checker root that yields an `Interval`.
        """
        right = self.right
        if self.op != "in" or not isinstance(right, Mapping):
            return None
        if set(right) != {"lo", "hi"}:
            return None
        return right["lo"], right["hi"]

    def _outcome(self, scopes: Mapping[str, Any]) -> tuple[bool, str]:
        """(held, why): presence rows read `left` directly. The rest compare operands.

        A missing reference fails the row (its `why` names it) except where the
        op asks about presence itself; an `in` range literal resolves its
        bounds into an `Interval`. `_show` bounds each failing value.

        Args:
            scopes (Mapping[str, Any]): The node's scoped fields, one mapping per
                scope root.
        Returns:
            tuple[bool, str]: `(held, why)`: whether the row holds and, when not,
                why its operand values fail it.
        """
        op = self._operation()
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
        missing: list[Ref] = []
        bounds = self._range_bounds()
        sides = (self.left, self.right, *bounds) if bounds else (self.left, self.right)
        for side in sides:
            if isinstance(side, Ref):
                found, value = _operand(side, scopes)
                out.append(value)
                if not found:
                    missing.append(side)
            else:
                out.append(side)
        if missing:
            return _missing_outcome(missing)
        raw_left, raw_right, *raw_bounds = out
        left = _finite(raw_left, str(self.left))
        right = _finite(raw_right, str(self.right))
        if bounds:
            lo, hi = (
                _finite(value, str(side))
                for value, side in zip(raw_bounds, bounds, strict=False)
            )
            if lo is None or hi is None:
                raise ValueError(
                    f"`in` range bounds must both be values, got "
                    f"{bounds[0]}..{bounds[1]}"
                )
            right = Interval(lo, hi)
        predicate = Predicate(left, right)
        held = op.holds(predicate, self.abs_tol, self.rel_tol)
        if held:
            return True, ""
        if bounds:
            return False, (
                f"expected {self.left} {self.op} {bounds[0]}..{bounds[1]}, "
                f"got {_show(left)}"
            )
        return False, (
            f"expected {self.left} {self.op} {self.right}, "
            f"got {_show(left)} vs {_show(right)}"
        )

    def describe(self) -> str:
        """The row as prose: `left op right`, `left in lo..hi`, or `left op`.

        The row label every boundary message reuses: one shape per row kind, so
        a message never reaches for fields a row does not have.

        Returns:
            str: The row's operand form, per its op's slots and range literal.
        """
        op = self._operation()
        bounds = self._range_bounds()
        if bounds:
            lo, hi = (_show(bound) for bound in bounds)
            return f"{_show(self.left)} {self.op} {lo}..{hi}"
        if op.slots == ():
            return f"{_show(self.left)} {self.op}"
        return f"{_show(self.left)} {self.op} {_show(self.right)}"

    def operands(self) -> tuple[Any, ...]:
        """Every operand this row resolves. Its `Ref`s live among them."""
        return (self.left, self.right)

    def map_refs(self, fn: Callable[[Ref], Ref]) -> None:
        """Rewrite each `Ref` operand through `fn` (cohort-load name checking)."""
        _rewrite_refs(self, ("left", "right"), fn)


class Aggregate(BaseModel):
    """One collection row: an `aggregate` over `over`, compared to `value`.

    A YAML row this decodes from:

        when:
          - over: point.results.accuracy
            aggregate: count
            op: ge
            value: 1

    `over` reads a list, or the values of a mapping. The `aggregate` key picks
    the subclass registered for it (see `__new__`): `Count` over an empty
    collection is 0 and `AnyTrue` is False, while `Sum`, `Mean`, `Min`, `Max`
    and `AllTrue` fail with `no values` over empty rather than guessing (a
    vacuous `all` pass on an empty upload would be a silent false-pass). `op`
    is one of the scalar ops (`AggregateOp` above).

    Args:
        over (Any): The operand holding the collection (a list, or a mapping
            whose values are read).
        name (AggregateName): The computation to run (`count`, `sum`, `mean`,
            `min`, `max`, `any`, `all`); the YAML key is `aggregate`.
        op (AggregateOp): The scalar comparison applied to the computed value
            (`eq`, `ne`, `gt`, `ge`, `lt`, `le`).
        value (Any): The compare target.
        abs_tol (float): Absolute tolerance, reaching `eq`/`ne` only.
            (Default: 0.0)
        rel_tol (float): Relative tolerance, reaching `eq`/`ne` only.
            (Default: 0.0)
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # name -> subclass, filled by `__init_subclass__` on the family.
    registry: ClassVar[dict[str, type[Aggregate]]] = {}

    over: Operand
    name: AggregateName = Field(alias="aggregate")
    op: AggregateOp
    value: Operand
    abs_tol: float = 0.0
    rel_tol: float = 0.0

    defines_empty: ClassVar[bool] = False

    def __new__(cls, **data: Any) -> Aggregate:
        """`Aggregate(name="count", ...)` -> `Count(...)` over the registered family."""
        if cls is Aggregate:
            name = data.get("aggregate", data.get("name"))
            if name is None:
                raise ValueError("aggregate row needs an `aggregate` key") from None
            try:
                return super().__new__(cls.registry[name])
            except KeyError:
                raise ValueError(f"unknown aggregate {name!r}") from None
        return super().__new__(cls)

    def __init_subclass__(cls, *, name: str, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.registry[name] = cls

    @model_validator(mode="after")
    def _check_shape(self) -> Aggregate:
        # The row must run its declared aggregate: `Count(name="sum")` would
        # compute `len` while its messages claim `sum`.
        if Aggregate.registry.get(self.name) is not type(self):
            raise ValueError(
                f"row computes {type(self).__name__} but declares aggregate "
                f"{self.name!r}"
            )
        op = self._operation()
        if not op.tolerant and (self.abs_tol or self.rel_tol):
            raise ValueError(
                f"`{self.op}` has no tolerance semantics (use `eq`/`ne` for abs_tol/rel_tol)"
            )
        return self

    def _operation(self) -> Op:
        """The row's `Op` instance: the one dispatch site for the string name.

        The YAML stores an op by name; binding it here keeps `_check_shape`
        and `_outcome` from re-dispatching the string at each use.
        """
        return Op(self.op)

    def run(self, items: list[Any]) -> Any:
        """The aggregation itself over `items`. A subclass owns one.

        Example:

            Count().run([1, 2])  # 2

        Called on an empty `items` raises NotImplementedError: the
        empty-collection policy sits in `_outcome`.

        Args:
            items (list[Any]): The collection `_outcome` resolved from `over`.
        Returns:
            Any: The aggregate's value over `items`, e.g. a count or a sum.
        Raises:
            NotImplementedError: On `Aggregate` itself: every registered
                family defines its own aggregation.
        """
        raise NotImplementedError

    def _outcome(self, scopes: Mapping[str, Any]) -> tuple[bool, str]:
        """(held, why) of `aggregate(over) op value`. `why` names the computed result.

        Args:
            scopes (Mapping[str, Any]): The node's scoped fields, one mapping per
                scope root.
        Returns:
            tuple[bool, str]: `(held, why)`: whether `aggregate(over) op value`
                holds and, when not, why.
        """
        resolved = []
        missing: list[Ref] = []
        for side in (self.over, self.value):
            if isinstance(side, Ref):
                found, value = _operand(side, scopes)
                resolved.append(value)
                if not found:
                    missing.append(side)
            else:
                resolved.append(side)
        if missing:
            return _missing_outcome(missing)
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
            return False, f"expected {self.name} of {self.over}, got no values"
        measured = self.run(items)
        held = self._operation().holds(
            Predicate(
                _finite(measured, f"{self.name} of {self.over}"),
                _finite(value, str(self.value)),
            ),
            self.abs_tol,
            self.rel_tol,
        )
        if held:
            return True, ""
        return False, (
            f"expected {self.name} of {self.over} {self.op} {self.value}, "
            f"got {_show(measured)} vs {_show(value)}"
        )

    def describe(self) -> str:
        """The row as prose: `aggregate of over op value`.

        The row label `Verdict.evaluate`'s boundary message reuses (the
        `Compare` twin keeps its own shape).

        Returns:
            str: The row's operand form, `aggregate of over op value`.
        """
        return f"{self.name} of {_show(self.over)} {self.op} {_show(self.value)}"

    def operands(self) -> tuple[Any, ...]:
        """Every operand this row resolves. Its `Ref`s live among them."""
        return (self.over, self.value)

    def map_refs(self, fn: Callable[[Ref], Ref]) -> None:
        """Rewrite each `Ref` operand through `fn` (cohort-load name checking)."""
        _rewrite_refs(self, ("over", "value"), fn)


class Count(Aggregate, name="count"):
    """`len(items)`: `Count().run([1, 2])` -> 2, empty -> 0."""

    defines_empty = True

    def run(self, items: list[Any]) -> Any:
        return len(items)


class Sum(Aggregate, name="sum"):
    """`sum(items)`: `Sum().run([1, 2, 3])` -> 6."""

    def run(self, items: list[Any]) -> Any:
        return sum(items)


class Mean(Aggregate, name="mean"):
    """`sum(items) / len(items)`: `Mean().run([1, 3])` -> 2.0."""

    def run(self, items: list[Any]) -> Any:
        return sum(items) / len(items)


class Min(Aggregate, name="min"):
    """`min(items)`: `Min().run([2, 1, 3])` -> 1."""

    def run(self, items: list[Any]) -> Any:
        return min(items)


class Max(Aggregate, name="max"):
    """`max(items)`: `Max().run([2, 1, 3])` -> 3."""

    def run(self, items: list[Any]) -> Any:
        return max(items)


class AnyTrue(Aggregate, name="any"):
    """`any(items)` over truthiness: `AnyTrue().run([0, 5])` -> True, empty -> False."""

    defines_empty = True

    def run(self, items: list[Any]) -> Any:
        return any(items)


class AllTrue(Aggregate, name="all"):
    """`all(items)` over truthiness: `AllTrue().run([1, 0])` -> False (empty fails in `_outcome`)."""

    def run(self, items: list[Any]) -> Any:
        return all(items)


@dataclass(frozen=True)
class Verdict:
    """Does a condition hold, and why not: the message is the assertion's own.

    `broken` marks a verdict whose assertion failed to *evaluate* (a deliberate
    fail-loud or an incommensurable pair) rather than judged False -- gate
    positions must refuse those loudly instead of reading them as "not
    applicable".

    Args:
        ok (bool): Whether the condition held.
        why (str): The assertion's own message, empty when `ok`.
        broken (bool): Whether the assertion failed to *evaluate* (a deliberate
            fail-loud or an incommensurable pair) rather than judged False.
            (Default: False)
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

        Args:
            row (Compare | Aggregate): The condition row to evaluate.
            scopes (Mapping[str, Any]): The node's scoped fields, one mapping per
                scope root.
        Returns:
            Verdict: The row's verdict. `broken` marks evaluation failure rather
                than false judgment.
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
