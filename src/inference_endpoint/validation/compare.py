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

"""Condition rows: every check condition is a `Compare` row.

A row decodes from YAML and evaluates against the scoped facts of one node (a
mapping per scope root, e.g. `{"point": ...}`). Operands are literals or `Ref`
dotted references (decode hooks below). `Verdict.evaluate` wraps Compare's
`operands` and `run`, and writes its `why` from the failing row's own operands:
the failing values shown bounded, exact no matter the input.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, model_validator


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
        plain dicts here and typed nodes at run time. Either slot can be None;
        `found` is what says which is which.
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
CompareOp = Literal["eq", "ne", "gt", "ge", "lt", "le", "in", "matches", "between"]
UnaryOp = Literal["exists", "absent", "not_null"]


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

    `op` decides the arity: `exists`, `absent` and `not_null` take `left` alone
    (`right: null` is a case for those ops only: a `right` given at all is
    wrong); `between` takes `lo` and `hi`; the rest take `right`. `right: null`
    is a legal eq/uneq operand: null is distinct from absent, exactly like
    `Ref.resolve` reports.
    """

    model_config = ConfigDict(extra="forbid")

    op: CompareOp | UnaryOp
    left: Operand
    right: Operand | None = None
    lo: Operand | None = None
    hi: Operand | None = None
    abs_tol: float | None = None
    rel_tol: float | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> Compare:
        # Presence, not truthiness: `right: null` is a real operand (eq against
        # a literal null is legal), so arity is decided by field presence.
        given = {
            name for name in ("right", "lo", "hi") if name in self.model_fields_set
        }
        if self.op in ("exists", "absent", "not_null"):
            if given:
                raise ValueError(f"{self.op} takes only `left`")
        elif self.op == "between":
            if given != {"lo", "hi"}:
                raise ValueError("`between` takes `left`, `lo` and `hi`")
        elif given != {"right"}:
            raise ValueError(f"`{self.op}` takes `left` and `right`")
        return self

    def operands(self, scopes: Mapping[str, Any]) -> tuple[bool, str, list[Any]]:
        """(ok, why, [left, right, lo, hi]) resolving refs against `scopes`.

        `(True, "", [1, 2, None, None])` for `eq` with literal 1 vs `point.tps`
        where that fact is 2 (that pair fails the comparison, but reference
        resolution succeeded). `(False, "missing point.x", [...])` where a
        reference names an absent fact: a row cannot compare a missing
        reference. Presence ops and `*Self` aliases do not reach here (see
        `Verdict.evaluate`).
        """
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
            return False, f"missing {', '.join(missing)}", out
        return True, "", out

    def run(self, left: Any, right: Any, lo: Any, hi: Any) -> tuple[bool, str]:
        """(held, detail) of `op` over resolved operands; detail names an incommensurable pair.

        `op: "ge"`, left 2, right 1 -> (True, ""); left "a", right 1 ->
        (False, "incomparable"): the ordering is not total across types.
        """
        if self.op in ("eq", "ne"):
            held = self._eq(left, right)
            return (held, "") if self.op == "eq" else (not held, "")
        if self.op == "in":
            keys = right.keys() if isinstance(right, Mapping) else right
            return left in keys, ""
        if self.op == "matches":
            return (isinstance(left, str) and re.fullmatch(right, left) is not None), ""
        try:
            if self.op in ("gt", "ge", "lt", "le"):
                result = {
                    "gt": left > right,
                    "ge": left >= right,
                    "lt": left < right,
                    "le": left <= right,
                }[self.op]
            else:
                result = bool(lo <= left <= hi)
            return result, ""
        except TypeError:
            return False, "incomparable"

    def _eq(self, left: Any, right: Any) -> bool:
        """`math.isclose` for numbers under a tolerance; `==` for every other type."""
        if isinstance(left, float | int) and isinstance(right, float | int):
            return math.isclose(
                left, right, abs_tol=self.abs_tol or 0.0, rel_tol=self.rel_tol or 0.0
            )
        return left == right


@dataclass(frozen=True)
class Verdict:
    """Does a condition hold, and why not: the message is the row's own."""

    ok: bool
    why: str

    @classmethod
    def evaluate(cls, row: Compare, scopes: Mapping[str, Any]) -> Verdict:
        """Evaluate one condition row against a node's scoped facts (see `Ref.resolve`).

        Holds only where the row holds; a null operand, an incommensurable
        pair (a number vs a string), an unpresent `existence` op and a missing
        reference each fail with their own `why`.
        """
        if row.op in ("exists", "absent", "not_null"):
            found, value = (
                row.left.resolve(scopes)
                if isinstance(row.left, Ref)
                else (True, row.left)
            )
            held = {
                "exists": found,
                "absent": not found,
                "not_null": found and value is not None,
            }[row.op]
            if held:
                return cls(True, "")
            got = repr(value) if found else "absent"
            return cls(False, f"expected {row.left} to be {row.op}, got {got}")

        ok, why, values = row.operands(scopes)
        if not ok:
            return cls(False, why)
        held, detail = row.run(*values)
        if held:
            return cls(True, "")
        # Everything the message needs is built here: it is discarded on a pass.
        target = row.right if row.right is not None else f"{row.lo}..{row.hi}"
        return cls(
            False,
            (
                f"expected {row.left} {row.op} {target}, "
                f"got {_show(values[0])} vs {_show(values[1])}"
                + (f" ({detail})" if detail else "")
            ),
        )

    @classmethod
    def evaluate_all(
        cls, rows: list[Compare], scopes: Mapping[str, Any]
    ) -> list[Verdict]:
        """Evaluate one condition list (a `when` or `assert` block) in order."""
        return [cls.evaluate(row, scopes) for row in rows]
