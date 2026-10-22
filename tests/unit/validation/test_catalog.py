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

"""The `Catalog` view and its `ruleset` root on `Node.scopes`.

Fixtures write tiny catalog YAMLs on demand. Cases assert the values each
subject computes: `Catalog.from_directory` its parsed mappings and
`Node.scopes` its `ruleset` fold.
"""

from pathlib import Path

import pytest
from inference_endpoint.validation.catalog import Catalog
from inference_endpoint.validation.refs import Ref
from inference_endpoint.validation.tree import SubmissionNode

MODELS = "gpt: {vocab_size: 50257, is_drafter: false}\n"
DATASETS = "ds: {num_prompt: 3}\n"


def _catalog(tmp_path: Path, models: str = MODELS, datasets: str = DATASETS) -> Catalog:
    """A catalog directory with `models.yaml` and `datasets.yaml` written on demand."""
    root = tmp_path / "ruleset"
    root.mkdir()
    (root / "models.yaml").write_text(models)
    (root / "datasets.yaml").write_text(datasets)
    return Catalog.from_directory(root)


class TestCatalogFromDirectory:
    """`Catalog.from_directory`: the two mappings, read as they are written."""

    @pytest.mark.unit
    def test_models_and_datasets_stay_plain_data(self, tmp_path):
        catalog = _catalog(tmp_path)
        assert catalog.models == {"gpt": {"vocab_size": 50257, "is_drafter": False}}
        assert catalog.datasets == {"ds": {"num_prompt": 3}}

    @pytest.mark.unit
    def test_non_mapping_document_is_a_load_error(self, tmp_path):
        with pytest.raises(ValueError, match="mapping"):
            _catalog(tmp_path, models="- a\n")

    @pytest.mark.unit
    def test_missing_directory_is_a_load_error(self, tmp_path):
        with pytest.raises(NotADirectoryError, match="nope"):
            Catalog.from_directory(tmp_path / "nope")


class TestRulesetRoot:
    """`Node.scopes` folds the catalog into the `ruleset` root of `Ref`."""

    def _scopes(self, tmp_path: Path):
        root = tmp_path / "acme"
        (root / "results").mkdir(parents=True)
        return SubmissionNode.from_directory(root, ruleset=_catalog(tmp_path)).scopes()

    @pytest.mark.unit
    def test_ruleset_root_answers_catalog_lookups(self, tmp_path):
        scopes = self._scopes(tmp_path)
        assert Ref(("ruleset", "models", "gpt", "vocab_size")).resolve(scopes) == (
            True,
            50257,
        )

    @pytest.mark.unit
    def test_without_a_catalog_the_ruleset_root_stays_away(self, tmp_path):
        root = tmp_path / "acme"
        (root / "results").mkdir(parents=True)
        scopes = SubmissionNode.from_directory(root).scopes()
        assert "ruleset" not in scopes
