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

"""Compare-row operations: one self-registering subclass per YAML `op` name.

Each op owns its operand `slots` and its `holds` test over a `Predicate`.
An `Interval` target gives `between` natural bounds semantics
(`5 in Interval(0, 10)` is True). Example under `Eq`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from inference_endpoint.validation.sentinels import MISSING


@dataclass(frozen=True)
class Predicate:
    """One comparison's resolved operands: the subject tested against its target.

    `subject` is the resolved `left`; `target` is the resolved `right`, or an
    `Interval` for `between`. A `MISSING` `subject` means no `left` fact exists
    at all: `Predicate(MISSING, 1)` is an absent fact where `Predicate(None, 1)`
    is a present fact whose value is null (the absent-vs-null grid lives in that
    distinction).
    """

    subject: Any
    target: Any


class Op:
    """One operation: its operand slots and its test, keyed on the YAML name.

        class Eq(Op, name="eq"): holds(Predicate(1.0, 1.0)) -> True

    Subclasses register themselves on `name`. `Op("eq")` -> `Eq()` and
    `Op("bogus")` -> ValueError. `slots` are the `Compare` fields beside `left`
    an op fills. `needs_found` marks presence ops: their `holds` keys on
    `predicate.subject is MISSING`.
    """

    name: ClassVar[str] = ""
    label: ClassVar[str] = ""
    slots: ClassVar[tuple[str, ...]] = ("right",)
    needs_found: ClassVar[bool] = False
    tolerant: ClassVar[bool] = False

    registry: ClassVar[dict[str, type[Op]]] = {}

    def __init_subclass__(
        cls,
        *,
        name: str,
        slots: tuple[str, ...] = ("right",),
        needs_found: bool = False,
        tolerant: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init_subclass__(**kwargs)
        cls.name = name
        cls.slots = slots
        cls.needs_found = needs_found
        cls.tolerant = tolerant
        cls.registry[name] = cls

    def __new__(cls, *args: Any, **kwargs: Any) -> Op:
        """`Op("eq")` -> `Eq()` (dispatch on the YAML name, like `Node`); unknown -> ValueError."""
        if cls is Op:
            name = args[0] if args else kwargs.get("name")
            try:
                return super().__new__(cls.registry[name])
            except KeyError:
                raise ValueError(f"unknown op {name!r}") from None
        return super().__new__(cls)

    def __init__(self, name: str | None = None) -> None:
        """Accepts the keying name so construction stays argument-symmetric; no instance state."""

    def holds(
        self, predicate: Predicate, abs_tol: float = 0.0, rel_tol: float = 0.0
    ) -> bool:
        """Whether this op holds over `predicate`; an incommensurable pair raises TypeError.

        `Eq().holds(Predicate(2, 2))` -> True.
        """
        raise NotImplementedError


def vet_pattern(pattern: str) -> None:
    """Compile-check a `matches` regex at load; an uncompilable one raises ValueError.

    Patterns are literal, cohort-authored strings (`Compare._check_shape`
    refuses resolved operands), so vetting happens once at load where
    `_load_check` names the offending file -- a malformed pattern must never
    reach `re.fullmatch` mid-run (its `re.error` is not even a `ValueError`,
    so it would escape the verdict boundary as a run-aborting surprise).
    """
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"`matches` pattern does not compile: {exc}") from exc


def _close_to(
    left: Any, right: Any, abs_tol: float = 0.0, rel_tol: float = 0.0
) -> bool:
    """`math.isclose` for a number pair under tolerance; `==` for every other pair.

    Integer pairs compare exactly, tolerances honored in integer arithmetic
    (`math.isclose`'s float coercion conflates distinct integers above 2^53).
    Ordering has no tolerance semantics (no standard meaning of an ordered
    comparison under `abs_tol`): those ops do not reach here.
    """
    if abs_tol == 0.0 and rel_tol == 0.0:
        return left == right
    if isinstance(left, int) and isinstance(right, int):
        return abs(left - right) <= max(abs_tol, rel_tol * max(abs(left), abs(right)))
    if isinstance(left, float | int) and isinstance(right, float | int):
        return math.isclose(left, right, abs_tol=abs_tol, rel_tol=rel_tol)
    return left == right


class Eq(Op, name="eq", tolerant=True):
    """`subject == target` numerically under `abs_tol`/`rel_tol` (see `_close_to`)."""

    def holds(
        self, predicate: Predicate, abs_tol: float = 0.0, rel_tol: float = 0.0
    ) -> bool:
        return _close_to(predicate.subject, predicate.target, abs_tol, rel_tol)


class Ne(Op, name="ne", tolerant=True):
    """`subject != target` under tolerance: `Eq` negated."""

    def holds(
        self, predicate: Predicate, abs_tol: float = 0.0, rel_tol: float = 0.0
    ) -> bool:
        return not _close_to(predicate.subject, predicate.target, abs_tol, rel_tol)


class Gt(Op, name="gt"):
    """`subject > target`; an incommensurable pair raises TypeError (fail loud)."""

    def holds(
        self, predicate: Predicate, abs_tol: float = 0.0, rel_tol: float = 0.0
    ) -> bool:
        return predicate.subject > predicate.target


class Ge(Op, name="ge"):
    """`subject >= target`; an incommensurable pair raises TypeError (fail loud)."""

    def holds(
        self, predicate: Predicate, abs_tol: float = 0.0, rel_tol: float = 0.0
    ) -> bool:
        return predicate.subject >= predicate.target


class Lt(Op, name="lt"):
    """`subject < target`; an incommensurable pair raises TypeError (fail loud)."""

    def holds(
        self, predicate: Predicate, abs_tol: float = 0.0, rel_tol: float = 0.0
    ) -> bool:
        return predicate.subject < predicate.target


class Le(Op, name="le"):
    """`subject <= target`; an incommensurable pair raises TypeError (fail loud)."""

    def holds(
        self, predicate: Predicate, abs_tol: float = 0.0, rel_tol: float = 0.0
    ) -> bool:
        return predicate.subject <= predicate.target


class In(Op, name="in"):
    """`subject` is among `target`'s keys (mapping) or values (list).

    `In().holds(Predicate("k", {"k": 1}))` -> True.
    """

    def holds(
        self, predicate: Predicate, abs_tol: float = 0.0, rel_tol: float = 0.0
    ) -> bool:
        keys = (
            predicate.target.keys()
            if isinstance(predicate.target, Mapping)
            else predicate.target
        )
        return predicate.subject in keys


class Matches(Op, name="matches"):
    """`subject` is a string fully matching the regex `target`.

    `Matches().holds(Predicate("r16", "r[0-9]+"))` -> True;
    the number 5 gives False: not a string, so no match.
    """

    def holds(
        self, predicate: Predicate, abs_tol: float = 0.0, rel_tol: float = 0.0
    ) -> bool:
        matched = isinstance(predicate.subject, str) and re.fullmatch(
            predicate.target, predicate.subject
        )
        return bool(matched)


class SameMultiset(Op, name="same_multiset"):
    """`[1, 2, 2]` matches `[2, 1, 2]`; `[1]` does not match `[1, 1]`.

    Unordered, multiplicity-respecting equality; hashable elements pair in
    linear time via counts, unhashable ones fall back to an `==` scanning pool
    so crafted large lists stay bounded work. A non-`Sequence` pair degrades to
    plain `==`.
    """

    def holds(
        self, predicate: Predicate, abs_tol: float = 0.0, rel_tol: float = 0.0
    ) -> bool:
        left, right = predicate.subject, predicate.target
        if not isinstance(left, Sequence) or not isinstance(right, Sequence):
            return left == right
        if len(left) != len(right):
            return False
        try:
            return Counter(left) == Counter(right)
        except TypeError:  # unhashable elements: pair by `==` scanning
            pool = list(right)
            for value in left:
                for i, other in enumerate(pool):
                    if value == other:
                        del pool[i]
                        break
                else:
                    return False
            return not pool


class Between(Op, name="between", slots=("lo", "hi")):
    """`subject in target`, where `target` is an `Interval`.

    `Between().holds(Predicate(5, Interval(0, 10)))` -> True.
    """

    def holds(
        self, predicate: Predicate, abs_tol: float = 0.0, rel_tol: float = 0.0
    ) -> bool:
        return predicate.subject in predicate.target


class Exists(Op, name="exists", slots=(), needs_found=True):
    """`left` names a present fact (a stored null still counts)."""

    label: ClassVar[str] = "to exist"

    def holds(
        self, predicate: Predicate, abs_tol: float = 0.0, rel_tol: float = 0.0
    ) -> bool:
        return predicate.subject is not MISSING


class Absent(Op, name="absent", slots=(), needs_found=True):
    """`left` names no such fact (a stored null is not absent)."""

    label: ClassVar[str] = "to be absent"

    def holds(
        self, predicate: Predicate, abs_tol: float = 0.0, rel_tol: float = 0.0
    ) -> bool:
        return predicate.subject is MISSING


class NotNull(Op, name="not_null", slots=(), needs_found=True):
    """`left` names a present fact whose value is not null."""

    label: ClassVar[str] = "to not be null"

    def holds(
        self, predicate: Predicate, abs_tol: float = 0.0, rel_tol: float = 0.0
    ) -> bool:
        return predicate.subject is not MISSING and predicate.subject is not None
