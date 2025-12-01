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

import logging
import random
from pathlib import Path

import datasets as hf_datasets
import pandas as pd

from ...dataset_manager.dataloader import DataLoader
from .base import AccuracyDataset, DatasetFormat

logger = logging.getLogger(__name__)


class GPQA(
    AccuracyDataset,
    columns=[
        "question",
        "choice1",
        "choice2",
        "choice3",
        "choice4",
        "ground_truth",
        "domain",
        "subdomain",
    ],
    format=DatasetFormat.PANDAS_DF,
):
    """GPQA: A Graduate-Level Google-Proof Q&A Benchmark
    Reference: https://arxiv.org/abs/2311.12022
    OpenAI implementation: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/gpqa_eval.py
    """

    def __init__(
        self,
        *args,
        variant: str = "diamond",
        seed: int = 0,  # GPT-OSS uses 0 as default seed
        max_samples: int | None = None,
        **kwargs,
    ):
        """
        Args:
            variant: The variant of the dataset to load. Must be one of "diamond", "extended", or "main".
            seed: The seed to use for the random number generator.
            max_samples: The maximum number of samples to load. If None, all samples will be loaded.
        """
        super().__init__(*args, variant=variant, **kwargs)
        if self.variant == "full":
            self.variant = "main"

        if self.variant not in ("diamond", "extended", "main"):
            raise ValueError(f"Invalid variant: {self.variant}")

        self.seed = seed
        self.max_samples = max_samples
        self._variant_name = self.variant
        if max_samples is not None and max_samples > 0:
            # Update the variant tag to also include the number of samples
            self.variant = f"{self._variant_name}_{max_samples}"

    def generate(self, datasets_dir: Path):
        # Load the variant from HuggingFace
        try:
            raw_ds = hf_datasets.load_dataset("Idavidrein/gpqa", f"gpqa_{self.variant}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Note: This dataset may require HuggingFace authentication.")
            print("Run: huggingface-cli login")
            raise

        df = raw_ds["train"].to_pandas()
        logger.info(f"Loaded {len(df)} samples from {self.variant} variant of GPQA")

        # If max_samples is specified, sample 'max_samples' rows from the dataset
        if self.max_samples is not None and self.max_samples < len(df):
            rng = random.Random(self.seed)
            sampled_indices = rng.sample(range(len(df)), self.max_samples)
            df = df.iloc[sampled_indices].reset_index(drop=True)
            logger.info(f"Sampled {self.max_samples} questions")

        rng = random.Random(self.seed)

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

        # Save to pickle file
        if not datasets_dir.exists():
            datasets_dir.mkdir(parents=True, exist_ok=True)
        df.to_pickle(datasets_dir / self.filename)
        logger.info(f"Saved {len(df)} samples to {datasets_dir / self.filename}")

    def load(
        self, datasets_dir: Path, create_if_not_exists: bool = False
    ) -> pd.DataFrame:
        """Load the dataset from the datasets_dir."""
        ds_path = datasets_dir / self.filename
        if not ds_path.exists():
            if create_if_not_exists:
                logger.info(
                    f"Attempted to load missing dataset file {ds_path}. Generating..."
                )
                self.generate(datasets_dir)
            else:
                raise FileNotFoundError(f"Dataset file {ds_path} does not exist")
        return pd.read_pickle(ds_path)


class GPQADataLoader(DataLoader):
    def __init__(
        self,
        datasets_dir: Path,
        user_prompt_format: str,
        *args,
        variant: str = "diamond",
        seed: int = 0,
        max_samples: int | None = None,
        **kwargs,
    ):
        """
        Args:
            datasets_dir: The directory containing the dataset.
            user_prompt_format: The format of the user prompt. The keys in the format string
                must match the column names in the dataset.
            variant: The variant of the dataset to load.
            seed: The seed to use for the random number generator.
            max_samples: The maximum number of samples to load. If None, all samples will be loaded.
        """
        super().__init__(*args, **kwargs)

        self.datasets_dir = datasets_dir

        # Load the dataset
        self.df = GPQA(variant=variant, seed=seed, max_samples=max_samples).load(
            datasets_dir
        )
        self.user_prompt_format = user_prompt_format

    def load_sample(self, index: int) -> str:
        d = self.df.iloc[index]
        return self.user_prompt_format.format(**d.to_dict())

    def num_samples(self) -> int:
        return len(self.df)
