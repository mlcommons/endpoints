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


class GPQA(
    Dataset,
    dataset_id="gpqa",
):
    """GPQA: A Graduate-Level Google-Proof Q&A Benchmark
    Reference: https://arxiv.org/abs/2311.12022
    OpenAI implementation: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/gpqa_eval.py
    """

    COLUMN_NAMES = [
        "question",
        "choice1",
        "choice2",
        "choice3",
        "choice4",
        "ground_truth",
        "domain",
        "subdomain",
    ]

    @classmethod
    def generate(
        cls,
        datasets_dir: Path,
        variant: str = "diamond",
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
            variant: The variant of the dataset to generate.
            seed: The random seed to use for shuffling the choices. Defaults to 0.
            max_samples: The maximum number of samples save to the file. If None, the
                entire dataset will be used as-is without shuffling. Otherwise, `max_samples`
                samples will be randomly sampled from the dataset.
            force: If True, the dataset will be regenerated even if it already exists.

        Returns:
            A pandas dataframe containing the dataset.
        """
        filename = f"{cls.DATASET_ID}_{variant}.parquet"
        dst_path = datasets_dir / cls.DATASET_ID / variant / filename
        if not dst_path.parent.exists():
            dst_path.parent.mkdir(parents=True)

        if dst_path.exists() and not force:
            logger.info(f"Dataset already exists at {dst_path}. Loading from file.")
            return pd.read_parquet(dst_path)

        try:
            df = load_from_huggingface(
                "Idavidrein/gpqa",
                dataset_name=f"gpqa_{variant}",
                split="train",
                cache_dir=datasets_dir / "hf_cache" / f"gpqa_{variant}",
            )
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            logger.error("Note: This dataset may require HuggingFace authentication.")
            logger.error("Run: huggingface-cli login")
            raise

        logger.info(f"Loaded {len(df)} samples from {variant} variant of GPQA")

        # If max_samples is specified, sample 'max_samples' rows from the dataset
        if max_samples is not None and max_samples < len(df):
            rng = random.Random(seed)
            sampled_indices = rng.sample(range(len(df)), max_samples)
            df = df.iloc[sampled_indices].reset_index(drop=True)
            logger.info(f"Sampled {max_samples} questions")

        rng = random.Random(seed)

        processed_rows = []
        for _, row in df.iterrows():
            # Create permutation for this example (following OpenAI's approach)
            # This shuffles the order of the 4 choices
            permutation = rng.sample(range(4), 4)

            # Extract original choices (following OpenAI's exact order)
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]

            # Apply permutation to shuffle choices
            choices = [choices[i] for i in permutation]

            # Find where the correct answer ended up after permutation
            correct_index = choices.index(row["Correct Answer"])
            correct_answer = f"choice{correct_index + 1}"

            # Create processed row
            processed_row = {
                "question": row["Question"],  # Original question
                "choice1": choices[0],
                "choice2": choices[1],
                "choice3": choices[2],
                "choice4": choices[3],
                "ground_truth": correct_answer,
                # Keep metadata for reference
                "domain": row.get("High-level domain", ""),
                "subdomain": row.get("Subdomain", ""),
            }

            processed_rows.append(processed_row)
        df = pd.DataFrame(processed_rows)

        # Save to parquet file
        df.to_parquet(dst_path)
        logger.info(f"Saved {len(df)} samples to {dst_path}")
        return df


class GPQA_MLPerf(GPQA):
    """GPQA_MLPerf: GPQA MLPerf Dataset
    Reference: https://huggingface.co/datasets/opencompass/GPQA/
    """

    @classmethod
    def create_gpqa_transforms(cls) -> list:
        """Create the list of transforms to apply to the GPQA dataset.

        Returns:
            List of transforms to apply
        """
        prompt_format = (
            "{question}\n\n"
            "(A) {choice1}\n"
            "(B) {choice2}\n"
            "(C) {choice3}\n"
            "(D) {choice4}\n\n"
            "Express your final answer as the corresponding option 'A', 'B', 'C', or 'D'."
        )

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
                    "choice1",
                    "choice2",
                    "choice3",
                    "choice4",
                    "domain",
                    "subdomain",
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
    def get_gpqa_dataloader(cls, num_repeats: int = 5):
        df = GPQA.generate(datasets_dir=Path("datasets"))
        transforms = GPQA_MLPerf.create_gpqa_transforms()
        gpqa_dataset = GPQA(df, transforms=transforms, repeats=num_repeats)
        return gpqa_dataset
