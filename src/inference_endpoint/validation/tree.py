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

"""The submission tree: nodes for one uploaded submission, and what its scopes expose.

`SubmissionNode.from_directory("submission/acme")` walks
`results/<system>/<model>/r<N>/` into nodes whose class follows the node kind
(`PointNode`, `ParetoNode`, ...). Each subclass registers under its `RuleScopes`
kind and `Node(...)` construction picks it. Every artifact is parsed once at
construction, so nothing re-reads disk while checks run. `node.scopes()` is
what `Ref.resolve` reads.
"""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import yaml

from inference_endpoint.validation.regions import RegionKind, Regions
from inference_endpoint.validation.sentinels import MISSING

if TYPE_CHECKING:  # circular-avoidance: `StrictLoader` lives here, `catalog` borrows it
    from inference_endpoint.validation.catalog import Catalog

_MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
_POINT_DIR_PATTERN = re.compile(r"r(\d+)")

# The whole tree's artifact bytes: the per-file cap times a generous fan-out
# (32 points x 7 artifacts), so one upload cannot multiply the per-file cap by
# its node count.
_MAX_TREE_BYTES = 8 * _MAX_ARTIFACT_BYTES


class RuleScopes(str, Enum):  # noqa: UP042 - repository wire-enum convention
    """The scopes a check condition can address, as `Ref` roots and node kinds.

    `point` -> the run directory. `pareto`, `system`, `submission` likewise.
    """

    SUBMISSION = "submission"
    SYSTEM = "system"
    PARETO = "pareto"
    POINT = "point"


FILE_TO_KEY_MAP = {
    # YAML file name -> field name the scopes mapping exposes.
    "point.yaml": "config",
    "config.yaml": "client_config",
    "result_summary.json": "result_summary",
    "system_desc.json": "system_desc",
    "system_power.json": "system_power",
    "accuracy_results.json": "accuracy_results",
    "results.json": "results",
}


class StrictLoader(yaml.SafeLoader):
    """SafeLoader for untrusted submission artifacts.

    A key given twice, a `<<` merge key or any alias is a parse error: the
    consumers of these documents (compare rows, message formatting) read a
    value once per mention, so an alias fan-out costs 2**d re-walks of the
    shared subtree, and a silent last-wins key hides data from the review.
    """

    def compose_node(self, parent: Any, index: int) -> Any:
        if self.check_event(yaml.events.AliasEvent):
            event = self.get_event()
            raise yaml.constructor.ConstructorError(
                None, None, "aliases are not allowed", event.start_mark
            )
        return super().compose_node(parent, index)

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> Any:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                raise yaml.constructor.ConstructorError(
                    None, None, "merge keys are not allowed", key_node.start_mark
                )
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError:
                continue  # an unhashable key: the base constructor rejects it
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    None, None, f"duplicate key {key!r}", key_node.start_mark
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


@dataclass(eq=False)
class Node:
    """One directory of the tree: what it is, where it is, who contains it.

    Constructing `Node(kind, dir, parent)` yields the subclass registered for
    `kind`. Construction parses the node's artifacts once into `artifacts`.

    Nodes compare by identity (`eq=False`): a value-`eq` over `children`/
    `parent` recurses through that cycle, and the generated `__hash__ = None`
    would make any frozen holder (`Point` embeds a node) unhashable.

    Args:
        kind (RuleScopes): What the directory is. `Node(...)` construction resolves
            the subclass through this `registry` key.
        dir (Path): Where the directory is.
        parent (Node | None): Who contains it. `None` at the upload root.
            (Default: None)
        children (tuple[Node, ...]): The nodes below it, each with their own subtree.
            (Default: ())
        artifacts (dict[str, tuple[Any, str | None]]): Each `FILE_TO_KEY_MAP`
            artifact present: name -> its parsed `(value, error)` pair. (Default: {})
        ruleset (Catalog | None): The `ruleset` fold `scopes()` exposes. The root
            node holds it when the submission was built with one. (Default: None)
        regions (Regions | None): The `regions` fold `scopes()` exposes. Each pareto
            node latches its own. (Default: None)
    """

    registry: ClassVar[
        dict[RuleScopes, type[Node]]
    ] = {}  # kind -> subclass, self-filled

    kind: RuleScopes
    dir: Path
    parent: Node | None = None
    children: tuple[Node, ...] = field(default=(), repr=False)
    artifacts: dict[str, tuple[Any, str | None]] = field(
        default_factory=dict, repr=False
    )
    ruleset: Catalog | None = field(default=None, repr=False)
    regions: Regions | None = field(default=None, repr=False)

    def __init_subclass__(cls, kind: RuleScopes, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.registry[kind] = cls

    def __new__(cls, *args: Any, **kwargs: Any):
        if cls is Node:
            kind = args[0] if args else kwargs.get("kind")
            return super().__new__(cls.registry[kind])
        return super().__new__(cls)

    def __post_init__(self) -> None:
        budget = self._artifact_budget()
        for name in FILE_TO_KEY_MAP:
            value, error = _read_artifact(self.dir, name, budget)
            if value is not MISSING:
                self.artifacts[name] = (value, error)

    def root(self) -> Node:
        """The tree's root node. It carries the `ruleset` fold (see `regions`)."""
        node: Node = self
        while node.parent is not None:
            node = node.parent
        return node

    def _artifact_budget(self) -> _ArtifactBudget:
        """The submission-wide artifact budget, created once on the root node."""
        root = self.root()
        budget = getattr(root, "_tree_budget", None)
        if budget is None:
            budget = _ArtifactBudget(_MAX_TREE_BYTES)
            # Internal wiring, not a dataclass field: the root holds the pot
            # every node's reads spend from (`object.__setattr__` keeps the
            # type checker out of the undeclared attribute).
            object.__setattr__(root, "_tree_budget", budget)
        return budget

    def read(self, name: str) -> tuple[Any, str | None]:
        """(value, error) of the artifact `name` of this node. `MISSING` when absent.

        Args:
            name (str): The artifact's file name, one of the
                `FILE_TO_KEY_MAP` keys.
        Returns:
            tuple[Any, str | None]: `result_summary.json` -> ({"n": 1}, None).
                Absent -> (MISSING, None). Unparsable or oversized -> (None, "<name>
                parse error: ...").
        """
        return self.artifacts.get(name, (MISSING, None))

    def scopes(self) -> dict[str, dict[str, Any]]:
        """Per-node field mappings that `Ref.resolve` in `compare.py` reads.

        Returns:
            dict[str, dict[str, Any]]: For a `point` holding only `result_summary.json`
                ->
                `{"point": {"result_summary": {"n": 1}}, "pareto": {}, "system": {}, "submission": {}}`.
                Its ancestors contribute their own fields, an artifactless node just
                `{}`. The `ruleset` root holds `Catalog.scopes()` when the submission
                was built with one. The `regions` fold rides the nearest ancestor that
                carries one (each pareto node latches its own, so the fold is
                per-pareto. Nodes above every pareto resolve no `regions`).
        """
        scopes: dict[str, dict[str, Any]] = {}
        node: Node | None = self
        while node is not None:
            fields = {}
            for name, key in FILE_TO_KEY_MAP.items():
                # `read` is the only (value, error) reader: a corrupt artifact
                # is none of its keys' business (it reads as its key's absence;
                # `read` keeps the parse error for diagnostics).
                value, error = node.read(name)
                if value is not MISSING and error is None:
                    fields[key] = value
            scopes[node.kind.value] = fields
            node = node.parent
        root = self.root()
        if root.ruleset is not None:
            scopes["ruleset"] = root.ruleset.scopes()
        fold: Node | None = self
        while fold is not None and fold.regions is None:
            fold = fold.parent
        if fold is not None and fold.regions is not None:
            scopes["regions"] = {
                "coverage": fold.regions.coverage(),
                "by_kind": {
                    RegionKind.ULTRA_LOW_CONCURRENCY: fold.regions.low_latency,
                    RegionKind.LOW_CONCURRENCY: fold.regions.low_concurrency,
                    RegionKind.MED_CONCURRENCY: fold.regions.med_concurrency,
                    RegionKind.HIGH_CONCURRENCY: fold.regions.high_concurrency,
                    RegionKind.MARGIN: fold.regions.margin,
                },
            }
        return scopes


class SubmissionNode(Node, kind=RuleScopes.SUBMISSION):
    """The upload root. Entry point via `from_directory`."""

    @classmethod
    def from_directory(
        cls,
        root: Path,
        ruleset: Catalog | None = None,
        regions: Regions | None = None,
    ) -> Node:
        """The `submission` node for `root`, with `results/<system>/<model>/r<N>` below.

        Args:
            root (Path): The upload root. `submission/acme` -> that submission's
                tree.
            ruleset (Catalog | None): The `ruleset` root of `Ref.resolve`: a
                `Catalog`, or `None` for no `ruleset` fold. (Default: None)
            regions (Regions | None): The `regions` root of `Ref.resolve`: a
                `Regions`, or `None` for no `regions` fold. (Default: None)
        Returns:
            Node: The `submission` node, with its `results/<system>/<model>/r<N>`
                tree walked.
        Raises:
            NotADirectoryError: When `root` is not a directory (a load problem, not
                a report).
            ValueError: When `root` is a symlink (artifact reads are
                fd-anchored and refuse symlinked directories: one clear refusal
                here instead of per-artifact `ELOOP` parse errors), or when the
                artifact tree exceeds the submission-wide byte budget.
        """
        if not root.is_dir():
            raise NotADirectoryError(root)
        if root.is_symlink():
            raise ValueError(
                f"{root} is a symlink: pass the upload's real directory "
                "(artifact reads are fd-anchored and refuse symlinked directories)"
            )
        submission = cls(
            RuleScopes.SUBMISSION, root, None, ruleset=ruleset, regions=regions
        )
        submission.children = _walk(submission, RuleScopes.SYSTEM, root / "results")
        return submission


class SystemNode(Node, kind=RuleScopes.SYSTEM):
    """`results/<system>/`, where `system_power.json` and the curves live."""


class ParetoNode(Node, kind=RuleScopes.PARETO):
    """`results/<system>/<model>/`, holding the `r<N>` points of one pareto."""


class PointNode(Node, kind=RuleScopes.POINT):
    """`results/<system>/<model>/r<N>/`, one measured run."""


_BRANCH = {RuleScopes.SYSTEM: RuleScopes.PARETO, RuleScopes.PARETO: RuleScopes.POINT}


def _walk(parent: Node, kind: RuleScopes, where: Path) -> tuple[Node, ...]:
    """`parent`'s children of `kind` below `where`, each with their own subtree.

    Points are `r<N>` directories. Other kinds take every directory that
    resolves inside the submission (a symlinked directory out of it is skipped).

    Args:
        parent (Node): The node whose children are walked.
        kind (RuleScopes): Which child kind to take.
        where (Path): The directory the children are listed under.
    Returns:
        tuple[Node, ...]: The children of `kind`, sorted by name, each with their
            own subtree below it.
    """
    children = []
    for entry in sorted(
        where.iterdir() if where.is_dir() else [], key=lambda p: p.name
    ):
        name_in_kind = (
            kind != RuleScopes.POINT
            or _POINT_DIR_PATTERN.fullmatch(entry.name) is not None
        )
        if (
            not entry.is_dir()
            # A symlinked directory multiplies nodes that re-parse shared
            # artifacts and inflates point counts (the YAML alias ban's spirit).
            or entry.is_symlink()
            or not name_in_kind
            or not _inside(parent.dir, entry)
        ):
            continue
        node = Node(kind, entry, parent)
        below = _BRANCH.get(kind)
        if below:
            node.children = _walk(node, below, entry)
        children.append(node)
    return tuple(children)


class _ArtifactBudget:
    """The submission-wide artifact budget: every read spends one pot.

    `_MAX_ARTIFACT_BYTES` caps one artifact; without a tree budget a crafted
    upload multiplies that cap by its node count.
    """

    __slots__ = ("_left",)

    def __init__(self, total: int) -> None:
        self._left = total

    def spend(self, size: int) -> None:
        """Refuse once the reads exceed the pot.

        Args:
            size (int): The bytes this read cost.
        Raises:
            ValueError: When the tree's artifact bytes exceed the budget.
        """
        self._left -= size
        if self._left < 0:
            raise ValueError(
                f"submission artifacts exceed the "
                f"{_MAX_TREE_BYTES // (1024 * 1024)} MiB tree budget"
            )


def _read_artifact(
    directory: Path, name: str, budget: _ArtifactBudget | None = None
) -> tuple[Any, str | None]:
    """(value, error) of `name` in `directory`. Nothing here aborts the run.

    The channel is fd-anchored: the node's directory is opened `O_NOFOLLOW`
    and the artifact named relative to that fd, so a symlinked artifact cannot
    redirect a read. Containment of the directory itself is decided earlier by
    the path-based `_walk`, so a directory swapped between that check and this
    open is the residual race (a resolved-root, fd-anchored walk is the fixed
    shape).

    Args:
        directory (Path): The node's directory.
        name (str): The artifact's file name.
        budget (_ArtifactBudget | None): The submission-wide pot this read
            spends its bytes from. (Default: None)
    Returns:
        tuple[Any, str | None]: Present and null -> (None, None). Absent ->
            (MISSING, None). Unreadable, oversized or unparsable -> (None, "<name> parse
            error: ...").
    Raises:
        ValueError: When `budget` is given and this read exhausts it.
    """
    try:
        dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        return None, f"{name} parse error: {exc.strerror}"
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
    except FileNotFoundError:
        return MISSING, None
    except OSError as exc:
        return None, f"{name} parse error: {exc.strerror}"
    finally:
        os.close(dir_fd)
    # A FIFO/socket/device here would block `open(2)` or `read` forever.
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        return None, f"{name} parse error: not a regular file"
    try:
        with os.fdopen(fd, "rb") as handle:
            data = handle.read(_MAX_ARTIFACT_BYTES + 1)
    except OSError as exc:
        return None, f"{name} parse error: {exc.strerror}"
    if len(data) > _MAX_ARTIFACT_BYTES:
        return None, f"{name} parse error: too large"
    if budget is not None:
        budget.spend(len(data))
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        return None, f"{name} parse error: {exc}"
    if not text.strip():
        return None, f"{name} parse error: empty document"
    try:
        value = (
            json.loads(text, object_pairs_hook=_json_object_no_duplicates)
            if name.endswith(".json")
            else yaml.load(text, Loader=StrictLoader)
        )
    except (ValueError, RecursionError, yaml.YAMLError) as exc:
        return None, f"{name} parse error: {exc}"
    if value is not None and not isinstance(value, Mapping):
        return None, (
            f"{name} parse error: expected a mapping document, "
            f"got {type(value).__name__}"
        )
    return value, None


def _json_object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """`json.loads` object hook: a repeated key is a parse error, as in YAML.

    Without this, `{"n_errors": 12, "n_errors": 0}` last-wins to `0` on the
    artifacts carrying the compliance numbers.

    Args:
        pairs (list[tuple[str, Any]]): An object's key-value pairs, in document
            order, as `json.loads` hands them to the hook.
    Returns:
        dict[str, Any]: The object as a mapping, its keys in first-seen order.
    Raises:
        ValueError: When a key repeats, naming the duplicate: the report's
            compliance numbers must not last-win.
    """
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _inside(root: Path, entry: Path) -> bool:
    try:
        return root.resolve() in entry.resolve().parents
    except (OSError, RuntimeError):
        return False  # an unresolvable path or a symlink loop is not inside
