# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shard plan: the immutable binding between units and instance ids.

The plan is content-addressed. Every unit result carries the plan digest, and
the merge gate refuses to combine results whose digest differs from the plan
being merged. That is what makes it impossible to accidentally merge results
from a different run, a different instance list, or a different ordering into
one accuracy number.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec

PLAN_FILENAME = "units.json"


class PlanError(ValueError):
    """The requested shard plan cannot be built, or a plan file is invalid."""


@dataclass(frozen=True, slots=True)
class Unit:
    """One dispatchable shard of a run."""

    unit_id: str
    run_id: str
    shard: int
    instance_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "run_id": self.run_id,
            "shard": self.shard,
            "instance_ids": list(self.instance_ids),
        }


@dataclass(frozen=True, slots=True)
class UnitPlan:
    """The full, immutable set of units for one run id."""

    run_id: str
    shard_size: int
    digest: str
    units: tuple[Unit, ...]

    @property
    def instance_ids(self) -> tuple[str, ...]:
        return tuple(
            instance_id for unit in self.units for instance_id in unit.instance_ids
        )

    def unit(self, unit_id: str) -> Unit:
        for candidate in self.units:
            if candidate.unit_id == unit_id:
                return candidate
        raise KeyError(unit_id)

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return tuple(unit.unit_id for unit in self.units)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "shard_size": self.shard_size,
            "digest": self.digest,
            "units": [unit.to_dict() for unit in self.units],
        }

    def write(self, directory: Path) -> Path:
        """Write the plan once. Rewriting an existing, different plan is an error."""
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / PLAN_FILENAME
        if path.exists():
            existing = read_plan(path)
            if existing.digest != self.digest or existing.run_id != self.run_id:
                raise PlanError(
                    f"refusing to overwrite plan at {path}: existing run_id="
                    f"{existing.run_id!r} digest={existing.digest[:12]} differs from "
                    f"new run_id={self.run_id!r} digest={self.digest[:12]}"
                )
            return path
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_bytes(msgspec.json.encode(self.to_dict()))
        tmp.replace(path)
        return path


def plan_digest(run_id: str, instance_ids: list[str] | tuple[str, ...]) -> str:
    """Digest over the run id and the ordered instance list.

    Order is included deliberately: two plans over the same ids in a different
    order produce different shards, so they are different plans.
    """
    hasher = hashlib.sha256()
    hasher.update(run_id.encode())
    hasher.update(b"\0")
    for instance_id in instance_ids:
        hasher.update(instance_id.encode())
        hasher.update(b"\n")
    return hasher.hexdigest()


def plan_units(
    run_id: str,
    instance_ids: list[str] | tuple[str, ...],
    *,
    shard_size: int = 10,
) -> UnitPlan:
    """Split ``instance_ids`` into fixed-size shards, in order.

    The final shard is short when the count is not a multiple of ``shard_size``;
    it is never padded and never merged into its neighbour, because the merge
    gate compares id sets and a padded shard would claim ids it never ran.
    """
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise PlanError(f"invalid run_id: {run_id!r}")
    if shard_size < 1:
        raise PlanError(f"shard_size must be >= 1; got {shard_size}")
    ordered = [str(instance_id) for instance_id in instance_ids]
    if not ordered:
        raise PlanError("cannot plan a run with no instance ids")
    duplicates = sorted({x for x in ordered if ordered.count(x) > 1})
    if duplicates:
        raise PlanError(
            "instance ids must be unique; duplicated: " + ", ".join(duplicates[:10])
        )

    digest = plan_digest(run_id, ordered)
    units: list[Unit] = []
    for shard, start in enumerate(range(0, len(ordered), shard_size)):
        chunk = tuple(ordered[start : start + shard_size])
        units.append(
            Unit(
                unit_id=f"{run_id}.s{shard:02d}",
                run_id=run_id,
                shard=shard,
                instance_ids=chunk,
            )
        )
    return UnitPlan(
        run_id=run_id, shard_size=shard_size, digest=digest, units=tuple(units)
    )


def read_plan(path: Path) -> UnitPlan:
    try:
        raw = msgspec.json.decode(path.read_bytes(), type=dict)
    except (OSError, msgspec.DecodeError) as exc:
        raise PlanError(f"could not read unit plan at {path}") from exc
    try:
        units = tuple(
            Unit(
                unit_id=str(entry["unit_id"]),
                run_id=str(entry["run_id"]),
                shard=int(entry["shard"]),
                instance_ids=tuple(str(x) for x in entry["instance_ids"]),
            )
            for entry in raw["units"]
        )
        plan = UnitPlan(
            run_id=str(raw["run_id"]),
            shard_size=int(raw["shard_size"]),
            digest=str(raw["digest"]),
            units=units,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PlanError(f"malformed unit plan at {path}") from exc

    recomputed = plan_digest(plan.run_id, list(plan.instance_ids))
    if recomputed != plan.digest:
        raise PlanError(
            f"unit plan at {path} is inconsistent: recorded digest "
            f"{plan.digest[:12]} does not match its own instance list "
            f"({recomputed[:12]})"
        )
    return plan
