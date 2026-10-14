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

`Catalog.from_directory("config/rulesets/mlcommons")` reads `models.yaml` and
`datasets.yaml` once; `catalog.models["gpt-oss-120b"]` is that model's entry as
written (sizes, flags and values stay raw data, not pseudo-enum classes). A
check addresses them through the `ruleset` root of `Ref`
(`ruleset.models.gpt-oss-120b.vocab_size`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from inference_endpoint.validation.tree import StrictLoader


class Catalog:
    """`models` and `datasets` mappings of a ruleset directory."""

    __slots__ = ("datasets", "models")

    def __init__(self, models: dict[str, Any], datasets: dict[str, Any]) -> None:
        self.models = models
        self.datasets = datasets

    @classmethod
    def from_directory(cls, root: Path) -> Catalog:
        """Load `models.yaml` and `datasets.yaml` under `root`.

        Without either file this raises FileNotFoundError: the ruleset is
        incomplete without its catalog. A document that is not a mapping is a
        load error too.
        """
        if not root.is_dir():
            raise NotADirectoryError(root)
        return cls(
            _load_mapping(root / "models.yaml"),
            _load_mapping(root / "datasets.yaml"),
        )

    def scopes(self) -> dict[str, dict[str, Any]]:
        """What the `ruleset` root of `Ref.resolve` answers, for `Node.scopes`."""
        return {"models": self.models, "datasets": self.datasets}


def _load_mapping(path: Path) -> dict[str, Any]:
    """That file's top-level mapping; a non-mapping document is a load error."""
    text = path.read_text(encoding="utf-8")
    value = yaml.load(text, Loader=StrictLoader)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must hold a mapping, got {type(value).__name__}")
    return value
