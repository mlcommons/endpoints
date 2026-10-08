# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomic policy loading, semantic fingerprints and bundle integrity."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from inference_endpoint.validation import bundled_policy_path, load_policy
from inference_endpoint.validation.schemas import BundleParser, parser_for
from inference_endpoint.validation.types import PolicyFile, Release, Scope
from inference_endpoint.validation.vocabulary import EvidenceReference

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FINGERPRINTS = ROOT / "tests/fixtures/validation/policy-fingerprints.json"


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.fixture
def policy_dir(tmp_path):
    target = tmp_path / "2026-10-C1"
    shutil.copytree(bundled_policy_path(), target)
    return target


def modify(path, transform):
    data = yaml.safe_load(path.read_text())
    transform(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def test_bundled_policy_matches_pinned_definitions():
    expected = json.loads(FINGERPRINTS.read_text())
    policy = load_policy(bundled_policy_path())
    monolith = policy.as_monolith()
    assert policy.release.version == expected["version"]
    assert policy.release.revision == expected["revision"]
    assert fingerprint(monolith["catalogs"]) == expected["catalog_digest"]
    assert fingerprint(monolith["enrollment"]) == expected["enrollment_digest"]
    actual = {
        f"{group}/{id_}": fingerprint(rule)
        for group, definition in monolith["check_groups"].items()
        for id_, rule in definition["checks"].items()
    }
    assert actual == expected["rules"]
    assert len(actual) == 89


def test_policy_is_immutable_and_preserves_source_hashes(policy_dir):
    policy = load_policy(policy_dir)
    with pytest.raises(TypeError):
        policy.catalogs["datasets"]["aime25"]["sample_count"] = 1
    with pytest.raises(TypeError):
        policy.checks[0].requirements["anything"] = 1
    for file, digest in policy.source_digests.items():
        assert (
            digest == hashlib.sha256((policy_dir / file.value).read_bytes()).hexdigest()
        )
    (policy_dir / PolicyFile.POINT.value).write_text(
        (policy_dir / PolicyFile.POINT.value).read_text() + "\n# Formatting-only edit\n"
    )
    edited = load_policy(policy_dir)
    assert edited.as_monolith() == policy.as_monolith()
    assert edited.digest != policy.digest


@pytest.mark.parametrize("file", list(PolicyFile))
def test_missing_any_file_rejects_whole_bundle(policy_dir, file):
    (policy_dir / file.value).unlink()
    with pytest.raises(ValueError, match="missing"):
        load_policy(policy_dir)


@pytest.mark.parametrize(
    "contents",
    [
        "version: '2026-10-C1'\nversion: '2026-10-C1'\nrevision: 1\n",
        "[]\n",
        "version: [broken\n",
        "version: '2026-10-C1'\nrevision: true\n",
    ],
)
def test_malformed_or_duplicate_yaml_rejects_bundle(policy_dir, contents):
    (policy_dir / PolicyFile.POINT.value).write_text(contents)
    with pytest.raises(ValueError):
        load_policy(policy_dir)


def test_extra_files_wrong_scope_and_mixed_revisions_are_rejected(policy_dir):
    extra = policy_dir / "point.yaml"
    extra.write_text("{}\n")
    with pytest.raises(ValueError, match="unexpected"):
        load_policy(policy_dir)
    extra.unlink()
    point = policy_dir / PolicyFile.POINT.value
    modify(point, lambda data: data.update(revision=2))
    with pytest.raises(ValueError, match="same version/revision"):
        load_policy(policy_dir)
    modify(point, lambda data: data.update(revision=1, scope="system"))
    with pytest.raises(ValueError, match="Wrong scope"):
        load_policy(policy_dir)


@pytest.mark.parametrize(
    "change",
    [
        {"kind": "magic_function"},
        {"applies_to": {"load_pattern": ["COP"]}},
        {"applies_to": {"curve_type": "agentic"}},
        {"requires": ["region_ready"]},
        {"source": "point.runtime_settings.typo"},
        {"selector": "agentic"},
        {"catalog": "catalogs.missing"},
    ],
)
def test_unknown_dispatch_and_reference_values_are_rejected(policy_dir, change):
    point = policy_dir / PolicyFile.POINT.value
    modify(
        point, lambda data: data["checks"]["agentic-trajectory-count"].update(change)
    )
    with pytest.raises(ValueError):
        load_policy(policy_dir)


def test_duplicate_ids_and_broken_catalog_bindings_are_rejected(policy_dir):
    point = policy_dir / PolicyFile.POINT.value
    modify(
        point,
        lambda data: data["checks"].update(
            {"path-exists": {"kind": "presence", "target": "point.directory"}}
        ),
    )
    with pytest.raises(ValueError, match="Duplicate rule ID"):
        load_policy(policy_dir)
    modify(point, lambda data: data["checks"].pop("path-exists"))
    modify(
        policy_dir / PolicyFile.CATALOG.value,
        lambda data: data["catalogs"]["datasets"].pop("cnn_dailymail"),
    )
    with pytest.raises(ValueError, match="Unknown accuracy dataset"):
        load_policy(policy_dir)


def test_evidence_addresses_are_enum_backed():
    policy = load_policy(bundled_policy_path())
    rule = next(rule for rule in policy.checks if rule.id == "agentic-trajectory-count")
    assert rule.scope is Scope.POINT
    assert (
        EvidenceReference.POINT_RUNTIME_SETTINGS_AGENTIC_INFERENCE_NUM_TRAJECTORIES_TO_ISSUE
        in rule.references
    )


def test_parser_dispatch_has_no_latest_revision_fallback():
    assert (
        type(parser_for(Release(version="2026-10-C1", revision=1))).__name__
        == "ParserV1"
    )
    for version, revision in (("2026-10-C1", 2), ("2027-01-C0", 1)):
        with pytest.raises(ValueError, match="Unsupported"):
            parser_for(Release(version=version, revision=revision))
    with pytest.raises(ValueError, match="Overlapping"):

        class Conflicting(
            BundleParser, version="2026-10-C1", first_revision=1, last_revision=2
        ):
            def __call__(self, documents, digests):
                raise AssertionError("Must not execute")


@pytest.mark.parametrize(
    "section", ["checks", "datasets", "models", "approved_checkpoints"]
)
def test_noncanonical_identifiers_cannot_replace_existing_entries(policy_dir, section):
    file = PolicyFile.POINT if section == "checks" else PolicyFile.CATALOG
    path = policy_dir / file.value

    def duplicate(data):
        entries = data["checks"] if section == "checks" else data["catalogs"][section]
        key = next(iter(entries))
        entries[f" {key} "] = entries[key]

    modify(path, duplicate)
    with pytest.raises(ValueError, match="whitespace"):
        load_policy(policy_dir)


def test_optional_offline_and_region_declarations_are_checked_only_when_present():
    policy = load_policy(bundled_policy_path())
    rules = {rule.id: rule for rule in policy.checks}
    assert rules["offline-declared"].requirements["required"] is False
    assert rules["region-declared"].requirements["required"] is False


def test_decode_head_can_record_its_publication_cohort(policy_dir):
    catalog = policy_dir / PolicyFile.CATALOG.value
    modify(
        catalog,
        lambda data: data["catalogs"]["approved_sped_decode_heads"][0].update(
            approved_cohort="2026-06-C0"
        ),
    )
    assert (
        load_policy(policy_dir).catalogs["approved_sped_decode_heads"][0][
            "approved_cohort"
        ]
        == "2026-06-C0"
    )
    modify(
        catalog,
        lambda data: data["catalogs"]["approved_sped_decode_heads"][0].update(
            approved_cohort="2026-06-C9"
        ),
    )
    with pytest.raises(ValueError):
        load_policy(policy_dir)


@pytest.mark.parametrize("revision", ["abc123", "A" * 40, "g" * 40])
def test_client_approval_list_rejects_malformed_sha(policy_dir, revision):
    modify(
        policy_dir / "catalog.yaml",
        lambda data: data["catalogs"].update(approved_client_revisions=[revision]),
    )
    with pytest.raises(ValueError):
        load_policy(policy_dir)


def test_client_approval_list_loads_full_shas(policy_dir):
    revisions = ["a" * 40, "b" * 40]
    modify(
        policy_dir / "catalog.yaml",
        lambda data: data["catalogs"].update(approved_client_revisions=revisions),
    )
    assert tuple(
        load_policy(policy_dir).catalogs["approved_client_revisions"]
    ) == tuple(revisions)
