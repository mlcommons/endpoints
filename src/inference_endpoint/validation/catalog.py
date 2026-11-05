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

"""The ruleset's catalog: model and dataset values as plain dictionaries, nothing more.

`Catalog.from_ruleset(name)` is the checker's loader: it derives the mappings
from the packaged ruleset the cohort names (`config/ruleset_registry.py`,
e.g. `mlperf-inference-v6.1`), so the checker and the benchmark read one
source. `Catalog.from_directory(root)` loads a development
`models.yaml` + `datasets.yaml` pair instead. Values stay raw data, not
pseudo-enum classes; a check addresses them through the `ruleset` namespace of
`Ref` (`ruleset.models.deepseek-r1.dataset.size`).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from inference_endpoint.config.ruleset_registry import get_ruleset
from inference_endpoint.config.rulesets.mlcommons.rules import RoundRuleset
from inference_endpoint.validation.tree import StrictLoader


class Catalog:
    """`models` and `datasets` mappings of a ruleset directory."""

    __slots__ = ("datasets", "models")

    def __init__(self, models: dict[str, Any], datasets: dict[str, Any]) -> None:
        self.models = models
        self.datasets = datasets

    @classmethod
    def from_ruleset(cls, name: str) -> Catalog:
        """The catalog the registered ruleset `name` carries.

        The checker reads packaged rulesets only: a cohort's manifest names
        one, and this view exposes its plain data -- one source with the
        benchmark side. Datasets key on their `description` (the Python
        `_Dataset` carries no short name); a round with no per-model rules
        (e.g. an Endpoints seed set) yields empty mappings.

        Args:
            name (str): A registered ruleset name (`config/ruleset_registry.py`).
        Returns:
            Catalog: The ruleset's `models` and `datasets` mappings, values as
                plain dictionaries.
        Raises:
            ValueError: When `name` is not registered, or the registered
                ruleset is not a round of per-model rules.
        """
        try:
            ruleset = get_ruleset(name)
        except KeyError as exc:
            raise ValueError(exc.args[0]) from None
        if not isinstance(ruleset, RoundRuleset):
            raise ValueError(
                f"ruleset {name!r} carries no model rounds: expected a RoundRuleset"
            )
        models: dict[str, Any] = {}
        datasets: dict[str, Any] = {}
        for model in ruleset.benchmark_rulesets:
            models[model.name] = asdict(model)
            datasets[model.dataset.description] = asdict(model.dataset)
        return cls(models, datasets)

    @classmethod
    def from_directory(cls, root: Path) -> Catalog:
        """Load `models.yaml` and `datasets.yaml` under `root`.

        Args:
            root (Path): The ruleset directory holding both catalog files.
        Returns:
            Catalog: The loaded `models` and `datasets` mappings.
        Raises:
            NotADirectoryError: When `root` is missing or is not a directory: the
                message names which and the path.
            FileNotFoundError: When `models.yaml` or `datasets.yaml` is missing:
                the ruleset is incomplete without its catalog.
            ValueError: When a document does not parse or is not a mapping: a
                load error, refused like every other bad cohort input.
        """
        if not root.exists():
            raise NotADirectoryError(f"the catalog directory {root} does not exist")
        if not root.is_dir():
            raise NotADirectoryError(f"the catalog path {root} is not a directory")
        return cls(
            _load_mapping(root / "models.yaml"),
            _load_mapping(root / "datasets.yaml"),
        )

    def scopes(self) -> dict[str, dict[str, Any]]:
        """What the `ruleset` namespace of `Ref.resolve` answers, for `Node.scopes`."""
        return {"models": self.models, "datasets": self.datasets}


def _load_mapping(path: Path) -> dict[str, Any]:
    """That file's top-level mapping. A malformed document is a load error.

    Raises:
        ValueError: When the document does not parse (a YAML error is a
            refusal, not a traceback escaping the CLI) or does not hold a
            mapping.
    """
    text = path.read_text(encoding="utf-8")
    try:
        value = yaml.load(text, Loader=StrictLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path.name} is not a catalog: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must hold a mapping, got {type(value).__name__}")
    return value
