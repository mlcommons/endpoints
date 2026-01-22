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

import random
from logging import getLogger
from pathlib import Path

import pandas as pd
from inference_endpoint.dataset_manager.transforms import Transform

from ...dataset import Dataset, load_from_huggingface
from . import presets

logger = getLogger(__name__)


class CNNDailyMail(
    Dataset,
    dataset_id="cnn_dailymail",
):
    """CNN/DailyMail Dataset
    Reference: https://huggingface.co/datasets/abisee/cnn_dailymail
    """

    COLUMN_NAMES = [
        "article",
        "highlights",
    ]

    PRESETS = presets

    @classmethod
    def generate(
        cls,
        datasets_dir: Path,
        seed: int = 0,
        max_samples: int | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Generates the CNN/DailyMail reference dataset for accuracy evaluation.

        The dataset variant is pulled from HuggingFace and is processed by extracting the correct answer and saving to a parquet file.

        Args:
            datasets_dir: The root datasets directory to save the dataset under. A
                subdirectory with the name and variant of the dataset will be created if
                it does not exist.
            seed: The random seed to use for sampling the dataset. Defaults to 0.
            max_samples: The maximum number of samples save to the file. If None, the
                entire dataset will be used as-is without shuffling. Otherwise, `max_samples`
                samples will be randomly sampled from the dataset.
            force: If True, the dataset will be regenerated even if it already exists.

        Returns:
            A pandas dataframe containing the dataset.
        """
        filename = f"{cls.DATASET_ID}.parquet"
        dst_path = datasets_dir / cls.DATASET_ID / filename
        if not dst_path.parent.exists():
            dst_path.parent.mkdir(parents=True)

        if dst_path.exists() and not force:
            logger.info(f"Dataset already exists at {dst_path}. Loading from file.")
            return pd.read_parquet(dst_path)

        try:
            df = load_from_huggingface(
                "abisee/cnn_dailymail",
                dataset_name="3.0.0",
                split="validation",
                cache_dir=datasets_dir / "hf_cache" / "cnndailymail",
            )
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            logger.error("Note: This dataset may require HuggingFace authentication.")
            logger.error("Run: huggingface-cli login")
            raise

        logger.info(f"Loaded {len(df)} samples from CNN/DailyMail")

        # If max_samples is specified, sample 'max_samples' rows from the dataset
        if max_samples is not None and max_samples < len(df):
            rng = random.Random(seed)
            sampled_indices = rng.sample(range(len(df)), max_samples)
            df = df.iloc[sampled_indices].reset_index(drop=True)
            logger.info(f"Sampled {max_samples} questions")

        # Save to parquet file
        df.to_parquet(dst_path)
        logger.info(f"Saved {len(df)} samples to {dst_path}")
        return df

    @classmethod
    def get_dataloader(
        cls,
        datasets_dir: Path = Path("datasets"),
        num_repeats: int = 1,
        transforms: list[Transform] | None = None,
        force_regenerate: bool = False,
    ) -> "Dataset":
        transforms = (transforms or []) + cls.PRESETS.llama3()
        df = cls.generate(force=force_regenerate, datasets_dir=datasets_dir)
        return cls(df, transforms=transforms, repeats=num_repeats)


__all__ = ["CNNDailyMail"]
