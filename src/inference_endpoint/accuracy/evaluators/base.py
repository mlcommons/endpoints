# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar


class Extractor:
    """An Extractor is used to extract phrases or substrings from the model's outputs using
    multiple regex patterns with a priority system. This is useful for extracting values from
    strings with the same general format but small variations, such as a model outputting a
    numeric value plain or inside a LaTeX block.
    """

    @classmethod
    def extract(cls, text: str) -> str | None:
        raise NotImplementedError


class Evaluator(ABC):
    """An Evaluator is used to calculate accuracy metrics for a given dataset given the
    outputs from the model. Evaluators are Regex based, and first extract phrases or substrings
    from the model's outputs to perform comparisons on. The format of the outputs are specified
    by the user prompt, so in effect, Evaluators also measure how good the model is at following
    the user's instructions.

    Evaluators follow Artificial Analysis's pass@k convention, see
    https://artificialanalysis.ai/methodology/intelligence-benchmarking
    for more details.
    """

    IMPLEMENTATIONS: ClassVar[dict[str, type["Evaluator"]]] = {}

    def __init_subclass__(cls, name: str | None = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name is None:
            name = cls.__name__

        if name in Evaluator.IMPLEMENTATIONS:
            raise ValueError(f"Evaluator with name {name} already exists")
        Evaluator.IMPLEMENTATIONS[name] = cls

    def __init__(self, extractor: Extractor):
        self.extractor = extractor

    @abstractmethod
    def extract_outputs(
        self,
        outputs_file: Path,
        uuid_map: dict[str, int],
        row_offset: int = 0,
    ):
        """Extracts the outputs from the model's outputs file and returns a list of strings"""
        # TODO: We need to figure out the following first:
        # 1. How we will run multiple datasets in succession
        # 2. How we store those outputs, whether it be in a single or multiple files
        pass
