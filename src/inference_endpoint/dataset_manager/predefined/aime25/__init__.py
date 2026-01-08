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

import random
import re
from logging import getLogger
from pathlib import Path

import pandas as pd
from inference_endpoint.dataset_manager.transforms import (
    AddStaticColumns,
    DropColumns,
    Harmonize,
    UserPromptFormatter,
)

from ...dataset import Dataset, load_from_huggingface

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

    @classmethod
    def generate(
        cls,
        datasets_dir: Path,
        seed: int = 0,
        max_samples: int | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Generates the GPQA reference dataset for accuracy evaluation.

        The dataset variant is pulled from HuggingFace and is pre-processed by shuffling
        the choices are randomly for each question, and saved to a parquet file.

        Args:
            datasets_dir: The root datasets directory to save the dataset under. A
                subdirectory with the name and variant of the dataset will be created if
                it does not exist.
            seed: The random seed to use for shuffling the choices. Defaults to 0.
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
            )
            df_ii = load_from_huggingface(
                "opencompass/AIME2025",
                dataset_name="AIME2025-II",
                split="test",
                cache_dir=datasets_dir / "hf_cache" / "aime25",
            )
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


class AIME_MLPerf(AIME25):
    """AIME_MLPerf: AIME 2025 MLPerf Dataset
    Reference: https://huggingface.co/datasets/opencompass/AIME2025/
    """

    @classmethod
    def generate(
        cls,
        datasets_dir: Path,
        max_samples: int | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Generate the AIME25 MLPerf dataset to a file."""
        df = AIME25.generate(
            datasets_dir=Path(datasets_dir),
            max_samples=max_samples,
            force=force,
        )
        return df

    @classmethod
    def create_aime25_transforms(cls) -> list:
        """Create the list of transforms to apply to the AIME25 dataset."""
        prompt_format = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

        return [
            # Step 1: Format the prompt from question and choices
            UserPromptFormatter(
                user_prompt_format=prompt_format,
                output_column="user_prompt",
            ),
            # Step 2: Harmonize the prompt for SGLang/GPT-OSS
            Harmonize(
                prompt_column="user_prompt",
            ),
            # Step 3: Drop columns we don't need for inference
            DropColumns(
                columns=[
                    "question",
                    "user_prompt",
                ],
                errors="ignore",
            ),
            # Step 4: Add metadata columns since we don't want to do a dict update every iteration
            AddStaticColumns(
                {
                    "stream": True,
                    "max_new_tokens": 32768,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                }
            ),
        ]

    @classmethod
    def get_aime25_dataloader(cls, num_repeats: int = 5):
        df = AIME25.generate(datasets_dir=Path("datasets"))
        transforms = AIME_MLPerf.create_aime25_transforms()
        aime25_dataset = AIME25(df, transforms=transforms, repeats=num_repeats)
        return aime25_dataset
