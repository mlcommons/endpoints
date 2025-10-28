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

"""Dataset loader factory for creating appropriate loaders based on format.

TODO: Very simple factory for now. Will be expanded to support multiple formats and datasets.
"""

import logging
from collections.abc import Callable
from pathlib import Path

from .dataloader import (
    DataLoader,
    DeepSeekR1ChatCompletionDataLoader,
    HFDataLoader,
)

logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """Factory for creating dataset loaders based on format.

    Supports:
    - pkl: Pickle format (DeepSeekR1ChatCompletionDataLoader)
    - jsonl: JSON Lines format (TODO)
    - hf: HuggingFace datasets (HFDataLoader)
    """

    @staticmethod
    def create_loader(
        dataset_path: Path | str,
        format: str = "pkl",
        parser: Callable | None = None,
        **kwargs,
    ) -> DataLoader:
        """Create appropriate dataset loader based on format.

        Args:
            dataset_path: Path to dataset file or directory
            format: Dataset format ("pkl", "jsonl", "hf")
            parser: Optional parser function for data transformation
            **kwargs: Additional arguments for specific loaders

        Returns:
            DataLoader instance

        Raises:
            ValueError: If format is unsupported
        """
        format = format.lower()

        if format == "pkl" or format == "pickle":
            # Pickle format - use DeepSeekR1ChatCompletionDataLoader
            if parser is None:
                # Default parser for chat completion format
                def default_parser(x):
                    return {"prompt": x.text_input, "output": x.ref_output}

                parser = default_parser

            logger.info(f"Creating pickle dataset loader for {dataset_path}")
            return DeepSeekR1ChatCompletionDataLoader(dataset_path, parser=parser)

        elif format == "jsonl" or format == "json":
            # JSON Lines format
            # TODO: Implement JSONLDataLoader
            logger.error("JSONL format not yet implemented")
            raise NotImplementedError(
                "JSONL dataset format not yet supported. " "Supported formats: pkl, hf"
            )

        elif format == "hf" or format == "huggingface":
            # HuggingFace dataset
            split = kwargs.get("split", "train")
            logger.info(
                f"Creating HuggingFace dataset loader for {dataset_path}, split={split}"
            )
            return HFDataLoader(
                dataset_path,
                parser=parser,
                split=split,
                format="arrow",  # HF datasets are typically arrow format
            )

        else:
            logger.error(f"Unknown dataset format: {format}")
            raise ValueError(
                f"Unsupported dataset format: '{format}'. "
                f"Supported formats: pkl, hf (huggingface). "
                f"Coming soon: jsonl"
            )

    @staticmethod
    def infer_format(dataset_path: Path | str) -> str:
        """Infer dataset format from file extension.

        Args:
            dataset_path: Path to dataset

        Returns:
            Inferred format string ("pkl", "jsonl", or "hf")
        """
        path = Path(dataset_path)

        # Check if it's a directory (likely HuggingFace)
        if path.is_dir():
            return "hf"

        # Check file extension
        suffix = path.suffix.lower()
        if suffix == ".pkl" or suffix == ".pickle":
            return "pkl"
        elif suffix == ".jsonl" or suffix == ".json":
            return "jsonl"
        else:
            # Default to pkl
            logger.warning(
                f"Unknown file extension '{suffix}', defaulting to 'pkl' format"
            )
            return "pkl"
