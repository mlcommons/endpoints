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
import re
from logging import getLogger
from pathlib import Path

import pandas as pd

from ...dataset import Dataset, load_from_huggingface
from . import presets

logger = getLogger(__name__)


def normalize_number(s):
    """Normalize a number string to an integer.
    Reference https://github.com/openai/gpt-oss/blob/48db88d8e29f48493fe75f084a8c9bd900a2b92f/gpt_oss/evals/aime_eval.py#L20
    """
    match = re.match(r"\d+", s)  # match digits from the start
    if not match:
        return None
    return int(match.group(0))


class AIME25(
    Dataset,
    dataset_id="aime25",
):
    """AIME25: AIME 2025 Dataset
    Reference: https://huggingface.co/datasets/opencompass/AIME2025/
    """

    COLUMN_NAMES = [
        "question",
        "answer",
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
        """Generates the AIME25 reference dataset for accuracy evaluation.

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
            df_i = load_from_huggingface(
                "opencompass/AIME2025",
                dataset_name="AIME2025-I",
                split="test",
                cache_dir=datasets_dir / "hf_cache" / "aime25",
            ).to_pandas()
            df_ii = load_from_huggingface(
                "opencompass/AIME2025",
                dataset_name="AIME2025-II",
                split="test",
                cache_dir=datasets_dir / "hf_cache" / "aime25",
            ).to_pandas()
            df = pd.concat([df_i, df_ii])
            logger.info(f"Loaded {len(df)} samples from AIME25-I and AIME25-II")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            logger.error("Note: This dataset may require HuggingFace authentication.")
            logger.error("Run: huggingface-cli login")
            raise

        logger.info(f"Loaded {len(df)} samples from AIME25")

        # If max_samples is specified, sample 'max_samples' rows from the dataset
        if max_samples is not None and max_samples < len(df):
            rng = random.Random(seed)
            sampled_indices = rng.sample(range(len(df)), max_samples)
            df = df.iloc[sampled_indices].reset_index(drop=True)
            logger.info(f"Sampled {max_samples} questions")

        processed_rows = []
        for _, row in df.iterrows():
            correct_answer = (
                normalize_number(row["answer"])
                if isinstance(row["answer"], str)
                else row["answer"]
            )
            # Create processed row
            processed_row = {
                "question": row["question"],  # Original question
                "answer": str(correct_answer),
            }

            processed_rows.append(processed_row)
        df = pd.DataFrame(processed_rows)

        # Save to parquet file
        df.to_parquet(dst_path)
        logger.info(f"Saved {len(df)} samples to {dst_path}")
        return df
