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

"""Dotted `Ref`s and the one cohort's `valid_refs.yaml` ref table.

A check row addresses facts as dotted `Ref`s (`point.config.concurrency`). What
a row may name is declared per cohort, in `valid_refs.yaml` beside the scope
files: nested like the fact tree (scope -> document -> fields, exactly the rows'
ref paths), each leaf naming its expected type (`bool`/`int`/`float`/`str`/
`list`, or `dict` for a subtree whose keys the rules describe without listing,
any deeper name addressable). A leaf spelled as a mapping adds the policy notes
`required` (a conforming upload carries it, so its absence breaks the row
loudly instead of gating it off) and `default` (the value an omitted fact judges
as, for rules that spell the absent-state out). The simple forms are all most
tables need:

    point:
      config:
        concurrency: { type: int, required: true }
        offline: { type: str, allowlist: [dedicated, elected, none], default: null }
        warmup: { duration_s: int }
        steady_state: { window: dict }
      accuracy_results: dict

Name checking runs in exactly one place (`RefTable.validate_names`, with
`typo_checks.near_match`'s one-edit hint), at cohort load: a row naming an
unknown or misspelled fact dies before any submission is touched. What a
present fact's value must carry is declared the same way (`allowlist` next to
its `type`) and checked at collect time (`RefTable.validate_values`).
A declared `int`/`float` leaf also reads a numeric string as the number it
spells (`decode_number`): writers stringify numbers, and refusing a real value
its quotation marks helps nobody. Every decoding is reported, never silent. A
numeric path component reads the key that spells the same number (`_key_for`),
so `point.result_summary.tpot.percentiles.90` finds a block written `"90.0"`.
`derived`/`regions`/`ruleset` namespaces are checker-owned (the derivation
registry, the region kinds, the catalog's own data) and are refused in the
YAML.

Endpoints Rules source for the field names (pinned):
https://github.com/mlcommons/endpoints_policies/blob/4ba8224dc746/endpoints_rules.md
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from math import isfinite
from pathlib import Path
from typing import Any, ClassVar

import yaml

from inference_endpoint.validation.sentinels import MISSING
from inference_endpoint.validation.tree import StrictLoader
from inference_endpoint.validation.typo_checks import near_match

# Numeric spellings: a string an artifact carries where a declaration expects a
# number, and the key spelling a percentile-shaped block was written under.
_NUMERIC_TYPES: frozenset[str] = frozenset({"int", "float"})


def _spelled_number(text: str) -> float | None:
    """The finite number `text` spells whole, or None when it spells none.

    Example:

        _spelled_number("90.0")  # 90.0
        _spelled_number("inf")  # None: not a metric
        _spelled_number("90 pct")  # None: the text spells no single number

    Args:
        text (str): The spelling to read.
    Returns:
        float | None: The number, or None when `float` cannot carry the whole
            text or the result is not finite.
    """
    try:
        number = float(text)
    except ValueError:
        return None
    return number if isfinite(number) else None


def decode_number(value: Any, type_name: str | None) -> Any:
    """`value` as a declared `type_name` reads it: a numeric string decodes to its number.

    One rule for every declared numeric fact: the ref table's leaves
    (`_Fact.coerce`) and the parser-enforced facts `submission.py` reads
    (`_int_fact`, `_finite_fact`) both decode through here, so a stringified
    number reads the same wherever it is declared.

    Artifact writers stringify numbers often enough -- a hand-edited YAML
    quoting one, a metrics dumper formatting floats -- that refusing them on
    type alone would refuse a fact the upload does carry. Only `int`/`float`
    declarations decode, and only exact spellings: an `int` takes `"512"` or
    the integral `"512.0"`, a `float` takes whatever `float` reads as finite
    (`"12.61"`, `"1e3"`). Units (`"512 kw"`), separators (`"1,000"`) and
    non-finite spellings (`"inf"`, `"nan"`) stay as they are, for
    `_Fact.matches` to refuse.

    Example:

        decode_number("512", "int")  # 512
        decode_number("512.0", "int")  # 512
        decode_number("512.5", "int")  # "512.5": an int leaf takes whole numbers
        decode_number("12.61", "float")  # 12.61
        decode_number("512", "str")  # "512": a str leaf means a str
        decode_number(512, "int")  # 512: already a number

    Args:
        value (Any): The artifact value.
        type_name (str | None): The declared type name, if any.
    Returns:
        Any: The decoded number, or `value` unchanged.
    """
    if type_name not in _NUMERIC_TYPES or not isinstance(value, str):
        return value
    if type_name == "float":
        number = _spelled_number(value)
        return value if number is None else number
    try:
        return int(value)
    except ValueError:
        pass
    number = _spelled_number(value)
    if number is None or not number.is_integer():
        return value
    return int(number)


def _key_for(obj: Mapping[str, Any], part: str) -> str | None:
    """The key of `obj` that `part` reads: its own spelling, or that number spelled otherwise.

    Percentile and histogram blocks are keyed by whatever spelling their writer
    produced (`"90.0"`, `"99.9"`), while a row names the percentile (`90`). A
    part that spells a finite number reads any key spelling the same number,
    the canonicalisation the submission checker applies to percentile keys.
    The exact key always wins, and equivalent spellings break by sorted key, so
    one artifact always reads the same way.

    Example:

        _key_for({"90.0": 12, "99.9": 40}, "90")  # "90.0"
        _key_for({"90": 12, "90.0": 40}, "90")  # "90": the exact key wins
        _key_for({"median": 12}, "90")  # None: no such key

    Args:
        obj (Mapping[str, Any]): The mapping to look the key up in.
        part (str): The path component naming the key.
    Returns:
        str | None: The key to read, or None when `obj` holds none that names it.
    """
    if part in obj:
        return part
    wanted = _spelled_number(part)
    if wanted is None:
        return None
    for key in sorted(name for name in obj if isinstance(name, str)):
        spelled = _spelled_number(key)
        if spelled is not None and spelled == wanted:
            return key
    return None


@dataclass(frozen=True)
class Ref:
    """A dotted reference into the scoped fields, e.g. `point.config.concurrency`.

    Args:
        parts (tuple[str, ...]): The dot-separated path components,
            e.g. `("point", "config", "concurrency")`.
        required (bool): Filled from the cohort's ref table at cohort load
            (`RefTable.annotate`): a required fact absent from the upload breaks the row
            loudly. (Default: False)
        default (Any): Filled from the cohort's ref table at cohort load
            (`RefTable.annotate`). `default` (not `MISSING`) is the value an omitted
            fact judges as. An un-annotated Ref (a hand-built row) has neither.
            (Default: MISSING)
        type (str | None): Filled from the cohort's ref table at cohort load
            (`RefTable.annotate`): the declared type of the leaf this ref names, which is
            what a numeric string on it decodes as. An un-annotated Ref (a hand-built
            row, or a checker-owned namespace) has none, and nothing decodes.
            (Default: None)
    """

    # The row language's scope namespaces: the node scopes a fact tree hangs under.
    SCOPE_ROOTS: ClassVar[tuple[str, ...]] = (
        "submission",
        "system",
        "pareto",
        "point",
    )
    # `derived` (the derivation registry), `regions` (the region kinds),
    # `ruleset` (the catalog's own data keys): the checker owns their names, so
    # `valid_refs.yaml` must not declare them (and no table checks them).
    CHECKER_ROOTS: ClassVar[tuple[str, ...]] = ("derived", "ruleset", "regions")
    ROOTS: ClassVar[tuple[str, ...]] = (*SCOPE_ROOTS, *CHECKER_ROOTS)

    parts: tuple[str, ...]
    required: bool = False
    default: Any = MISSING
    type: str | None = None

    @classmethod
    def from_string(cls, value: str) -> Ref | None:
        """A `Ref` from a dotted string headed by a namespace, else None for a literal.

        Example:

            Ref.from_string("point.config.concurrency")  # Ref(("point", "config", "concurrency"))
            Ref.from_string("config.concurrency")  # None: no namespace head (stays a literal)
            Ref.from_string("point.")  # None too: nothing behind the dot
            Ref.from_string("point")  # None too: no dot at all

        Args:
            value (str): The dotted string to read.
        Returns:
            Ref | None: The namespaced `Ref`, or None for a literal.
        """
        head, dot, rest = value.partition(".")
        if dot and head in cls.ROOTS and rest:
            return cls(tuple(value.split(".")))
        return None

    def resolve(self, scopes: Mapping[str, Any]) -> tuple[bool, Any]:
        """(found, value) of this reference: a missing key is absent, a stored null is a value.

        Example:

            scopes = {"point": {"tps": 1, "bad": None}}
            Ref(("point", "tps")).resolve(scopes)  # (True, 1)
            Ref(("point", "nope")).resolve(scopes)  # (False, None): no such field
            Ref(("point", "bad")).resolve(scopes)  # (True, None): the field exists and is null.

        Each step is a mapping key look-up -- a numeric component may read a key
        that spells the same number (`_key_for`) -- and the resolved value is a
        leaf, regardless of the value's type (a checker namespace's `Interval`
        is a value, never walked through). A leaf declared `int`/`float` reads
        a numeric string as the number it spells (`decode_number`), the same
        decoding the collect-time pass reports.

        Artifact documents are submitter-controlled, and attribute traversal would let
        `{"errors": []}` answer `errors.count` (the builtin method's) as a present fact.

        Args:
            scopes (Mapping[str, Any]): The scoped tree to resolve against.
        Returns:
            tuple[bool, Any]: `(found, value)`. `found` is False for a missing key.
        """
        obj: Any = scopes
        for part in self.parts:
            if not isinstance(obj, Mapping):
                return False, None
            key = _key_for(obj, part)
            if key is None:
                return False, None
            obj = obj[key]
        return True, decode_number(obj, self.type)

    def __str__(self) -> str:
        return ".".join(self.parts)


@dataclass(frozen=True)
class _Fact:
    """One declared leaf: its expected type and the table's policy notes.

    Args:
        type (str): The leaf's expected type. `type == "dict"` is a subtree the
            rules describe without listing its keys (any deeper name is addressable).
            Every other type is a leaf, with nothing deeper addressable under it.
        required (bool): A required fact absent from the upload breaks the row
            loudly. (Default: False)
        default (Any): The value an omitted fact judges as. `default is not MISSING`
            marks a fact whose absent-state value the rules spell out.
            (Default: MISSING)
        allowlist (tuple[Any, ...] | None): The valid values `allowlist` names,
            checked against the upload at collect time. (Default: None)
    """

    VALID_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"bool", "int", "float", "str", "list", "dict"}
    )

    type: str
    required: bool = False
    default: Any = MISSING
    allowlist: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        """Refuse an unknown `type` at construction, whoever builds the fact."""
        if self.type not in _Fact.VALID_TYPES:
            raise ValueError(f"{self.type!r} is no type name")

    @property
    def is_subtree(self) -> bool:
        """Whether deeper names stay addressable (`dict`).

        Its own value must still be a block (checked in `_check_values`); the
        values below it stay unchecked.
        """
        return self.type == "dict"

    def coerce(self, value: Any) -> Any:
        """`value` as this leaf's declared type reads it: a numeric string decodes.

        See `decode_number` for the spellings that count. Other declared types
        decode nothing: `"512"` on a `str` leaf is a string.

        Example:

            _Fact("int").coerce("512")  # 512
            _Fact("float").coerce(" 12.61 ")  # 12.61
            _Fact("int").coerce("512 kw")  # "512 kw": `matches` refuses it

        Args:
            value (Any): The artifact value.
        Returns:
            Any: The decoded number, or `value` unchanged.
        """
        return decode_number(value, self.type)

    def matches(self, value: Any) -> bool:
        """Whether an upload value may carry this leaf, by its declared type."""
        match self.type:
            case "bool":
                return isinstance(value, bool)
            case "int":
                # `bool` is an `int` in Python, so an int leaf must not accept `true`.
                return isinstance(value, int) and not isinstance(value, bool)
            case "float":
                # An `int` is accepted as a `float` (a count-valued metric stays valid).
                return isinstance(value, int | float) and not isinstance(value, bool)
            case "str":
                return isinstance(value, str)
            case "list":
                return isinstance(value, list)
            case "dict":
                # `dict` never asks here: `_check_values` requires the leaf's own
                # value to be a mapping, and the values below it stay unchecked.
                # The arm is spelled out so every declared type name has one.
                raise NotImplementedError(f"no value test for {self.type!r}")
            case _:
                # Any other name is a missing branch and must fail loud.
                raise NotImplementedError(f"no value test for {self.type!r}")


# The mapping form's reserved keys: `_Fact`'s own fields. A mapping whose
# keys stay inside this set and whose `type` is a valid type name is a leaf
# with policy notes, not a sub-block.
_ANNOTATION_KEYS: frozenset[str] = frozenset(field.name for field in fields(_Fact))


class RefTable:
    """One cohort's declared refs: what its rows may name, and how absences read.

    `RefTable.from_yaml(root / "valid_refs.yaml")` parses the declaration.
    `validate_names` refuses a path naming nothing (with `near_match`'s hint) or
    stepping below a leaf. `annotate` fills one row `Ref`'s `required`/
    `default` from the leaf it names.
    """

    FILENAME: ClassVar[str] = "valid_refs.yaml"

    __slots__ = ("_roots",)

    def __init__(self, roots: Mapping[str, Any]) -> None:
        self._roots = dict(roots)

    @classmethod
    def from_yaml(cls, path: Path) -> RefTable:
        """The ref table one cohort directory declares.

        The file MUST name only node scopes (`submission`/`system`/`pareto`/
        `point`): `derived`/`regions`/`ruleset` facts are the checker's own
        (the derivation registry, the region kinds, the catalog's data keys)
        and a YAML spelling of them is refused here. The strict loader applies
        (a duplicated key, a merge key or an alias is a parse error).

        Args:
            path (Path): The cohort's `valid_refs.yaml`, beside its scope files.
        Returns:
            RefTable: The ref table the YAML declares.
        Raises:
            ValueError: When the file is missing, when it is no ref table (a parse
                error, a non-mapping or an empty document), when it names a namespace
                outside the node scopes, or when an entry is neither a type name nor a
                block.
        """
        if not path.is_file():
            raise ValueError(
                f"cohort needs {cls.FILENAME}: the declared ref table sits "
                "beside the scope files (its nesting is the fact tree rows may name)"
            )
        try:
            data = yaml.load(path.read_text(encoding="utf-8"), Loader=StrictLoader)
        except yaml.YAMLError as exc:
            raise ValueError(f"{path.name} is not a ref table: {exc}") from exc
        if not isinstance(data, Mapping) or not data:
            raise ValueError(
                f"{path.name} is not a ref table: expected scope names to declare under"
            )
        roots: dict[str, Any] = {}
        for name, node in data.items():
            if name not in Ref.SCOPE_ROOTS:
                raise ValueError(
                    f"{path.name}: {name!r} is not a node scope "
                    "(derived/regions/ruleset facts are checker-owned)"
                )
            roots[str(name)] = _entry(str(name), node)
        return cls(roots)

    def validate_names(self, parts: tuple[str, ...]) -> None:
        """Refuse a `Ref` path that names nothing. ValueError says what went wrong.

        Example:

            table.validate_names(("point", "config", "ofline"))  # ValueError naming the near-miss `offline`
            table.validate_names(("point", "accuracy_results"))  # fine (an open document)
            table.validate_names(("point", "system_desc", "cooling", "detail"))  # ValueError: below a leaf fact

        Args:
            parts (tuple[str, ...]): The `Ref` path to check.
        Raises:
            ValueError: When a step names nothing declared (with `near_match`'s
                one-edit hint), when a step goes below a leaf fact, or when the path
                names no document.
        """
        level: Any = self._roots
        walked: list[str] = []
        for name in parts:
            if isinstance(level, _Fact):
                if level.is_subtree:
                    return
                raise ValueError(
                    f"{'.'.join(parts)} names no fact: "
                    f"{'.'.join(parts[: len(walked)])} is one leaf fact, nothing deeper"
                )
            if not isinstance(level, Mapping) or name not in level:
                near = near_match(name, level) if isinstance(level, Mapping) else None
                hint = f"; did you mean {near!r}?" if near is not None else ""
                raise ValueError(f"{'.'.join(parts)} names no fact{hint}")
            walked.append(name)
            level = level[name]
        if len(parts) < 2:
            raise ValueError(
                f"{'.'.join(parts)} names no fact: expected "
                f"{'.'.join(parts[:1])}.<document>[.<field>...]"
            )

    def annotate(self, ref: Ref) -> Ref:
        """`ref` with its leaf's `required`/`default`/`type` filled from the table.

        Call after `validate_names`: a path naming an open subtree or a block (or a
        table-free hand-built Ref) carries none of them.

        Args:
            ref (Ref): The row's `Ref`, after `validate_names`.
        Returns:
            Ref: `ref` with its leaf's `required`/`default`/`type` filled, or
                `ref` itself when it names no leaf.
        """
        level: Any = self._roots
        for name in ref.parts:
            if not isinstance(level, Mapping) or name not in level:
                return ref
            level = level[name]
        if isinstance(level, _Fact):
            return replace(
                ref,
                required=level.required,
                default=level.default,
                type=level.type,
            )
        return ref

    def validate_values(self, root: str, values: Mapping[str, Any]) -> tuple[str, ...]:
        """Refuse a present artifact value its declaration will not carry, reporting decodings.

        Every declared leaf below `root` that the upload does carry must match
        its `type` and be in its `allowlist` (where declared). A declared block
        must carry a block, and so must a `dict` leaf's own value: what stays
        unchecked is the values below it. An absent fact is checked here by
        nothing (its absence reads per its `required`/`default` notes). Runs per
        node at collect time (`runner._validate_declared_values`) because the
        table is cohort data.

        A declared `int`/`float` leaf whose value is a numeric string is no
        fault: it reads as the number it spells (`_Fact.coerce`), and the
        caller gets one note per decoding, so the report can say what was read.

        Args:
            root (str): The node scope to check.
            values (Mapping[str, Any]): The node's collected artifact document.
        Returns:
            tuple[str, ...]: One `"<path>: '<text>' read as <number>"` note per
                decoded value, in declaration order.
        Raises:
            ValueError: When a present value does not match its declared type or
                allowlist, or a declared block or `dict` leaf carries no block.
        """
        notes: list[str] = []
        tree = self._roots.get(root)
        if isinstance(tree, Mapping):
            _check_values(tree, values, [root], notes)
        return tuple(notes)


def _check_values(
    declared: Mapping[str, Any],
    values: Mapping[str, Any],
    path: list[str],
    notes: list[str],
) -> None:
    """Walk `declared` against one document's `values`. ValueError names the fault.

    A numeric string on a declared `int`/`float` leaf is decoded
    (`_Fact.coerce`) and one note appended to `notes`; the decoded value is what
    `matches` and the `allowlist` then judge.

    Args:
        declared (Mapping[str, Any]): A declared block of the cohort's ref table.
        values (Mapping[str, Any]): The document to check against it.
        path (list[str]): The scopes walked so far, for the fault's message.
        notes (list[str]): Appended with one decoding note per decoded value.
    Raises:
        ValueError: When a present value does not match its declared type or
            allowlist, or a declared block or `dict` leaf carries no block.
    """
    for name, entry in declared.items():
        if name not in values:
            continue
        value = values[name]
        here = [*path, name]
        if isinstance(entry, Mapping):
            if isinstance(value, Mapping):
                _check_values(entry, value, here, notes)
                continue
            raise ValueError(f"{'.'.join(here)}: {value!r} is no block")
        if entry.is_subtree:
            # The open subtree's own value must still be a block; the values
            # below it are what the `dict` declaration leaves unchecked.
            if not isinstance(value, Mapping):
                raise ValueError(f"{'.'.join(here)}: {value!r} is no block")
            continue
        decoded = entry.coerce(value)
        if decoded is not value:
            notes.append(f"{'.'.join(here)}: {value!r} read as {decoded!r}")
        if not entry.matches(decoded):
            raise ValueError(
                f"{'.'.join(here)}: {value!r} does not match its declared type "
                f"{entry.type!r}"
            )
        if entry.allowlist is not None and decoded not in entry.allowlist:
            raise ValueError(
                f"{'.'.join(here)}: {value!r} is not one of {entry.allowlist!r}"
            )


def _entry(name: str, node: Any) -> Any:
    """One declaration entry: `Fact` (leaf/open subtree) or a block of sub-entries.

    - `duration_s: int` -> a leaf
    - `window: dict` -> an open subtree
    - `offline: { type: str, allowlist: [dedicated, elected, none], default: null }`
      -> a leaf with policy notes (the allowlist is checked against the upload
      at collect time, and a non-`null` `default` must live in its allowlist,
      `null` being the absent-state's own value). The notes are cross-validated
      at load: `default` and every allowlist value must match the leaf's type,
      and `required` cannot carry a `default`.
    - `warmup: { ... }` -> a block

    A mapping is the annotated-leaf form when `type` names one of the valid
    types and every key is reserved (`type`/`required`/`default`/`allowlist`);
    anything else is a block of sub-entries, so a rule field literally named
    `type` stays a block when its value is not a type name:
    `field: { type: { ... } }` is a block.

    Args:
        name (str): The entry's name, for the fault messages.
        node (Any): The YAML spelling to read (a type name, a mapping, ...).
    Returns:
        Any: The `_Fact` for a leaf or open subtree, or a block of sub-entries.
    """
    if isinstance(node, str):
        if node not in _Fact.VALID_TYPES:
            raise ValueError(
                f"{name}: {node!r} is no type name "
                f"(use {'/'.join(sorted(_Fact.VALID_TYPES))})"
            )
        return _Fact(type=node)
    if isinstance(node, Mapping):
        if (
            "type" in node
            and (set(node) <= _ANNOTATION_KEYS and isinstance(node["type"], str))
            and node["type"] in _Fact.VALID_TYPES
        ):
            required = node.get("required", False)
            if not isinstance(required, bool):
                raise ValueError(f"{name}: `required` needs to be true or false")
            allowlist = node.get("allowlist")
            if allowlist is not None and node["type"] == "dict":
                raise ValueError(
                    f"{name}: `allowlist` needs a leaf type (`dict` subtrees stay open)"
                )
            if allowlist is not None and (
                not isinstance(allowlist, list)
                or not allowlist
                or not all(
                    isinstance(value, str | int | float | bool) for value in allowlist
                )
            ):
                raise ValueError(
                    f"{name}: `allowlist` needs a non-empty list of values"
                )
            default = node.get("default", MISSING)
            if required and default is not MISSING:
                raise ValueError(
                    f"{name}: `required` and `default` contradict: a declared "
                    "default answers the fact's absence"
                )
            if (
                allowlist is not None
                and default is not MISSING
                and default is not None
                and default not in allowlist
            ):
                raise ValueError(
                    f"{name}: the declared `default` {default!r} is not in its "
                    f"allowlist {allowlist!r}"
                )
            fact = _Fact(
                type=node["type"],
                required=required,
                default=default,
                allowlist=tuple(allowlist) if allowlist is not None else None,
            )
            if (
                default is not MISSING
                and default is not None
                and not fact.matches(default)
            ):
                raise ValueError(
                    f"{name}: the declared `default` {default!r} does not match "
                    f"its type {fact.type!r}"
                )
            mistyped = [value for value in allowlist or () if not fact.matches(value)]
            if mistyped:
                raise ValueError(
                    f"{name}: allowlist values {mistyped!r} do not match the "
                    f"declared type {fact.type!r}"
                )
            return fact
        return {str(key): _entry(str(key), value) for key, value in node.items()}
    raise ValueError(
        f"{name}: expected a type name ('int', 'dict', ...) or a block of field names"
    )
