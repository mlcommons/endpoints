# SPDX-FileCopyrightText: Copyright (c) 2024 MLCommons
# SPDX-License-Identifier: Apache-2.0
"""Tests for the published seed-set registry (§4.6 Seed Rotation)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from inference_endpoint.validation.catalogs.seeds import (
    SeedSetError,
    bundled_seed_sets_path,
    load_seed_sets,
    parse_seed_catalog,
)

pytestmark = pytest.mark.unit

#: Literal published seed values guard against registry transcription errors.
_PR117_SET_A = {
    "scheduler_rng_seed": 10487924139932647040,
    "sample_index_rng_seed": 586478644936801402,
    "model_seed": 9315206023656308754,
}


def _write_registry(path: Path, entries: list[dict]) -> Path:
    path.write_text(yaml.safe_dump({"seed_sets": entries}))
    return path


class TestBundledRegistry:
    def test_mirrors_pr117(self) -> None:
        seed_set = load_seed_sets()["A"]
        assert seed_set.seeds == _PR117_SET_A

    def test_adoption_window_is_derived_from_the_publication_cohort(self) -> None:
        """§4.6: adoptable for the publication cohort and the three following it."""
        assert load_seed_sets()["A"].cohorts == (
            "2026-10-C1",
            "2026-11-C0",
            "2026-11-C1",
            "2026-12-C0",
        )

    def test_bundled_file_ships_with_the_package(self) -> None:
        assert bundled_seed_sets_path().is_file()


@pytest.mark.parametrize(
    "wrapper,cohorts,expected",
    [
        pytest.param(
            {"version": 1.0, "cohort-id": "2027-03-C0"},
            None,
            ("2027-03-C0", "2027-03-C1", "2027-04-C0", "2027-04-C1"),
            id="cohort-wrapper",
        ),
        pytest.param(None, None, (), id="flat-without-window"),
        pytest.param(
            {"cohort-id": "2026-10-C1"},
            ["2030-01-C0"],
            ("2030-01-C0",),
            id="explicit-window-overrides-derived",
        ),
    ],
)
def test_seed_registry_shapes(tmp_path, wrapper, cohorts, expected):
    entry = {
        "id": "Z",
        "scheduler_rng_seed": 1,
        "sample_index_rng_seed": 2,
        "model_seed": 3,
    }
    if cohorts is not None:
        entry["cohorts"] = cohorts
    document = {"seed_sets": [entry]}
    if wrapper is not None:
        document = {"cohort": {**wrapper, **document}}
    path = tmp_path / "seeds.yaml"
    path.write_text(yaml.safe_dump(document))
    assert parse_seed_catalog(path)["Z"].cohorts == expected


class TestCatalogParsings:
    def test_explicit_path_wins(self, tmp_path: Path) -> None:
        path = _write_registry(
            tmp_path / "seeds.yaml",
            [
                {
                    "id": "Z",
                    "scheduler_rng_seed": 1,
                    "sample_index_rng_seed": 2,
                    "model_seed": 3,
                }
            ],
        )
        sets = parse_seed_catalog(path)
        assert list(sets) == ["Z"]
        assert sets["Z"].seeds == {
            "scheduler_rng_seed": 1,
            "sample_index_rng_seed": 2,
            "model_seed": 3,
        }

    def test_environment_cannot_replace_published_seeds(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MLPERF_ENDPOINTS_SEED_SETS", str(tmp_path / "missing.yaml"))
        assert load_seed_sets() == parse_seed_catalog(bundled_seed_sets_path())

    def test_cohorts_are_read_when_present(self, tmp_path: Path) -> None:
        """Forward-compatibility: the loader already reads the key §4.6 needs."""
        path = _write_registry(
            tmp_path / "seeds.yaml",
            [
                {
                    "id": "A",
                    "scheduler_rng_seed": 1,
                    "sample_index_rng_seed": 2,
                    "model_seed": 3,
                    "cohorts": ["2026-09-C0", "2026-09-C1"],
                }
            ],
        )
        assert parse_seed_catalog(path)["A"].cohorts == ("2026-09-C0", "2026-09-C1")


class TestMalformedRegistries:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(SeedSetError, match="Cannot read"):
            parse_seed_catalog(tmp_path / "absent.yaml")

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "seeds.yaml"
        path.write_text("seed_sets: [\n")
        with pytest.raises(SeedSetError, match="Invalid YAML"):
            parse_seed_catalog(path)

    def test_not_a_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "seeds.yaml"
        path.write_text("- just\n- a\n- list\n")
        with pytest.raises(SeedSetError, match="must be a mapping"):
            parse_seed_catalog(path)

    def test_malformed_cohort_id(self, tmp_path: Path) -> None:
        path = tmp_path / "seeds.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "cohort": {
                        "cohort-id": "not-a-cohort",
                        "seed_sets": [
                            {
                                "id": "A",
                                "scheduler_rng_seed": 1,
                                "sample_index_rng_seed": 2,
                                "model_seed": 3,
                            }
                        ],
                    }
                }
            )
        )
        with pytest.raises(SeedSetError, match="YYYY-MM-C0/C1"):
            parse_seed_catalog(path)

    def test_cohort_without_seed_sets(self, tmp_path: Path) -> None:
        path = tmp_path / "seeds.yaml"
        path.write_text(yaml.safe_dump({"cohort": {"cohort-id": "2026-10-C1"}}))
        with pytest.raises(SeedSetError, match="cohort.seed_sets must be a list"):
            parse_seed_catalog(path)

    def test_empty_seed_set_list(self, tmp_path: Path) -> None:
        with pytest.raises(SeedSetError, match="defines no seed sets"):
            parse_seed_catalog(_write_registry(tmp_path / "seeds.yaml", []))

    def test_entry_without_id(self, tmp_path: Path) -> None:
        path = _write_registry(
            tmp_path / "seeds.yaml",
            [{"scheduler_rng_seed": 1, "sample_index_rng_seed": 2, "model_seed": 3}],
        )
        with pytest.raises(SeedSetError, match="has no 'id'"):
            parse_seed_catalog(path)

    def test_entry_missing_a_seed(self, tmp_path: Path) -> None:
        path = _write_registry(
            tmp_path / "seeds.yaml",
            [{"id": "A", "scheduler_rng_seed": 1, "model_seed": 3}],
        )
        with pytest.raises(SeedSetError, match="must define integer"):
            parse_seed_catalog(path)

    def test_entry_with_a_non_integer_seed(self, tmp_path: Path) -> None:
        path = _write_registry(
            tmp_path / "seeds.yaml",
            [
                {
                    "id": "A",
                    "scheduler_rng_seed": "not-a-number",
                    "sample_index_rng_seed": 2,
                    "model_seed": 3,
                }
            ],
        )
        with pytest.raises(SeedSetError, match="must define integer"):
            parse_seed_catalog(path)
