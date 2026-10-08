# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize readable submission cases as complete artifact trees."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import JsonValue

CASE_DIRECTORY = Path(__file__).resolve().parents[2] / "fixtures/validation/cases"
SUBMISSION_CASES = tuple(path.stem for path in sorted(CASE_DIRECTORY.glob("*.yaml")))


def _merge(
    base: dict[str, JsonValue], changes: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    merged = deepcopy(base)
    for key, value in changes.items():
        existing = merged.get(key)
        merged[key] = (
            _merge(existing, value)
            if isinstance(existing, dict) and isinstance(value, dict)
            else deepcopy(value)
        )
    return merged


def _write_artifact(path: Path, value: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        text = json.dumps(value, indent=2) + "\n"
    elif path.suffix == ".yaml":
        text = yaml.safe_dump(value, sort_keys=False)
    else:
        assert isinstance(value, str)
        text = value
    path.write_text(text, encoding="utf-8")


@pytest.fixture(scope="session")
def submission_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("validation-submissions")
    for name in SUBMISSION_CASES:
        case = yaml.safe_load(
            (CASE_DIRECTORY / f"{name}.yaml").read_text(encoding="utf-8")
        )
        target = root / name
        for filename, value in case["files"].items():
            _write_artifact(target / filename, value)
        for point, files in case["points"].items():
            for filename, overrides in files.items():
                value = _merge(case["defaults"][filename], overrides)
                _write_artifact(target / point / filename, value)
    return root


@pytest.fixture(scope="session")
def standardized_submission(submission_corpus: Path) -> Path:
    return submission_corpus / "valid_standardized"


@pytest.fixture(params=SUBMISSION_CASES, ids=SUBMISSION_CASES)
def submission_case(request: pytest.FixtureRequest, submission_corpus: Path) -> Path:
    return submission_corpus / request.param
