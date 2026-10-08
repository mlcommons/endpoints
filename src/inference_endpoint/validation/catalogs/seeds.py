# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Load published seed values and cohort adoption windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..cohorts import Cohort, adoption_window

_BUNDLED = Path(__file__).parent / "seed_sets.yaml"
_SEED_FIELDS = ("scheduler_rng_seed", "sample_index_rng_seed", "model_seed")


class SeedSetError(ValueError):
    """Raised when a seed-set file cannot be read or is malformed."""


@dataclass(frozen=True)
class SeedSet:
    """One published seed set."""

    id: str
    scheduler_rng_seed: int
    sample_index_rng_seed: int
    model_seed: int
    cohorts: tuple[str, ...] = field(default_factory=tuple)

    @property
    def seeds(self) -> dict[str, int]:
        """The set's seeds keyed by field name, for comparison against a point."""
        return {name: getattr(self, name) for name in _SEED_FIELDS}


def bundled_seed_sets_path() -> Path:
    """Path to the seed-set file shipped with this checker."""
    return _BUNDLED


def load_seed_sets() -> dict[str, SeedSet]:
    """Load the bundled published seed catalog."""
    return parse_seed_catalog(_BUNDLED)


def parse_seed_catalog(chosen: Path) -> dict[str, SeedSet]:
    """Parse a catalog and derive its publication adoption windows."""
    try:
        raw = yaml.safe_load(chosen.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        raise SeedSetError(f"Cannot read seed-set file {chosen}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SeedSetError(f"Invalid YAML in seed-set file {chosen}: {exc}") from exc
    entries, published = _unwrap(chosen, raw)
    cohorts: tuple[str, ...] = ()
    if published is not None:
        parsed = Cohort.parse(published)
        if parsed is None:
            raise SeedSetError(
                f"{chosen}: cohort-id {published!r} is not of the form YYYY-MM-C0/C1"
            )
        cohorts = tuple(str(c) for c in adoption_window(parsed))
    sets: dict[str, SeedSet] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SeedSetError(f"{chosen}: seed_sets[{index}] is not a mapping")
        set_id = entry.get("id")
        if not isinstance(set_id, str) or not set_id:
            raise SeedSetError(f"{chosen}: seed_sets[{index}] has no 'id'")
        if set_id in sets:
            raise SeedSetError(f"{chosen}: duplicate seed set {set_id!r}")
        try:
            seeds = {name: entry[name] for name in _SEED_FIELDS}
            if any(type(value) is not int for value in seeds.values()):
                raise ValueError("Seeds must be integers without coercion")
        except (KeyError, TypeError, ValueError) as exc:
            raise SeedSetError(
                f"{chosen}: seed set {set_id!r} must define integer {', '.join(_SEED_FIELDS)}: {exc}"
            ) from exc
        explicit = entry.get("cohorts")
        if explicit is not None and (not isinstance(explicit, list)):
            raise SeedSetError(
                f"{chosen}: seed set {set_id!r} has a non-list 'cohorts'"
            )
        window = tuple(str(c) for c in explicit) if explicit is not None else cohorts
        sets[set_id] = SeedSet(id=set_id, cohorts=window, **seeds)
    if not sets:
        raise SeedSetError(f"{chosen} defines no seed sets")
    return sets


def _unwrap(path: Path, raw: object) -> tuple[list[object], str | None]:
    """Return ``(seed-set entries, publication cohort)`` from either registry shape."""
    if not isinstance(raw, dict):
        raise SeedSetError(f"{path} must be a mapping")
    cohort = raw.get("cohort")
    if isinstance(cohort, dict):
        entries = cohort.get("seed_sets")
        published = cohort.get("cohort-id")
        if not isinstance(entries, list):
            raise SeedSetError(f"{path}: cohort.seed_sets must be a list")
        return (entries, str(published) if published is not None else None)
    entries = raw.get("seed_sets")
    if not isinstance(entries, list):
        raise SeedSetError(
            f"{path} must define cohort.seed_sets (or a top-level seed_sets list)"
        )
    return (entries, None)
