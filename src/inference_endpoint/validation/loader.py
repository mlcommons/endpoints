# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load an atomic five-file policy bundle, retaining hashes for reproducibility."""

import hashlib
import json
from os import PathLike
from pathlib import Path

import yaml

from .models import Policy
from .schemas import parser_for
from .types import PolicyFile, Release


class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("Policy mapping keys must be strings")
            if key in result:
                raise ValueError(f"Duplicate YAML key {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def bundled_policy_path(version: str = "2026-10-C1") -> Path:
    Release(version=version, revision=1)
    return Path(__file__).parent / "policies" / version


def load_policy(directory: PathLike[str]) -> Policy:
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Policy directory does not exist: {directory}")
    expected = {file.value for file in PolicyFile}
    actual = {
        path.name for path in directory.iterdir() if path.suffix in {".yaml", ".yml"}
    }
    if actual != expected:
        raise ValueError(
            f"Invalid policy files; missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    documents = {}
    digests = {}
    release = None
    for file in PolicyFile:
        path = directory / file.value
        try:
            raw = path.read_bytes()
            document = yaml.load(raw, Loader=UniqueKeyLoader)
            if not isinstance(document, dict):
                raise ValueError("Expected a policy mapping")
            json.dumps(document, allow_nan=False)
            identity = Release.model_validate(
                {
                    key: document[key]
                    for key in ("version", "revision")
                    if key in document
                }
            )
            if release is not None and identity != release:
                raise ValueError("All policy files must have the same version/revision")
            release = identity
            documents[file] = document
            digests[file] = hashlib.sha256(raw).hexdigest()
        except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
            raise ValueError(f"Invalid {file.value}: {error}") from error
    assert release is not None
    if directory.name != release.version:
        raise ValueError("Policy directory name must match its cohort version")
    try:
        return parser_for(release)(documents, digests)
    except ValueError as error:
        raise ValueError(f"Invalid policy bundle: {error}") from error
