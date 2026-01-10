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

from ...dataset import Dataset, load_from_huggingface
from ...transforms import (
    AddStaticColumns,
    Harmonize,
    UserPromptFormatter,
)

logger = getLogger(__name__)


class LiveCodeBench(
    Dataset,
    dataset_id="livecodebench",
):
    """LiveCodeBench

    Link: https://github.com/LiveCodeBench/LiveCodeBench
    Paper: https://arxiv.org/abs/2403.07974
    """

    COLUMN_NAMES = [
        "question",
        "starter_code",
    ]

    @classmethod
    def generate(
        cls,
        datasets_dir: Path,
        variant: str = "release_v6",
        seed: int = 0,
        max_samples: int | None = None,
        force: bool = False,
    ) -> pd.DataFrame:
        """Generates the LiveCodeBench reference dataset for accuracy evaluation."""
        filename = f"{cls.DATASET_ID}_{variant}.parquet"
        dst_path = datasets_dir / cls.DATASET_ID / variant / filename
        if not dst_path.parent.exists():
            dst_path.parent.mkdir(parents=True)

        if dst_path.exists() and not force:
            logger.info(f"Dataset already exists at {dst_path}. Loading from file.")
            return pd.read_parquet(dst_path)

        try:
            df = load_from_huggingface(
                "livecodebench/code_generation_lite",
                split="test",
                cache_dir=datasets_dir / "hf_cache" / f"{cls.DATASET_ID}_{variant}",
                load_options={
                    "version_tag": variant,
                },
            )
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            logger.error("Note: This dataset may require HuggingFace authentication.")
            logger.error("Run: huggingface-cli login")
            raise

        logger.info(f"Loaded {len(df)} samples from {variant} variant of LiveCodeBench")

        # Following pre-processing steps from GPT-OSS side branch for parity with MLPerf Inference v6.0 GPT-OSS:
        # https://github.com/v-shobhit/gpt-oss/blob/feat/mlperf_integration/gpt_oss/evals/livecodebench_eval.py#L75

        keep = ["question_id", "question_content", "starter_code"]
        df = df[keep]
        df.rename(columns={"question_content": "question"}, inplace=True)

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


class LiveCodeBench_GPTOSS_SGLang(LiveCodeBench):
    """LiveCodeBench_GPTOSS_SGLang: LiveCodeBench GTPOSS_SGLang Dataset
    Reference: https://huggingface.co/datasets/livecodebench/code_generation_lite/
    """

    @classmethod
    def create_transforms(cls) -> list:
        """Create the list of transforms to apply to the LiveCodeBench dataset."""

        instructions = (
            "You are a python coding expert that solves problems step-by-step.\n"
            "You must provide the reasoning to arriving at your solution and the code to solve the problem.\n"
            "Do not try simulating the code execution. The code must be enclosed within ```python delimiters.\n"
        )

        user_prompt_format = (
            f"{instructions}\n\n"
            "{question}\n"
            "### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
            "```python\n"
            "{starter_code}\n"
            "```\n"
        )

        return [
            # Step 1: Add instructions to the dataset
            UserPromptFormatter(
                user_prompt_format=user_prompt_format,
                output_column="user_prompt",
            ),
            # Step 2: Harmonize the prompt for SGLang/GPT-OSS
            Harmonize(
                prompt_column="user_prompt",
            ),
            # Step 3: Add metadata columns since we don't want to do a dict update every iteration
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
    def get_dataloader(cls, num_repeats: int = 5):
        df = LiveCodeBench.generate(datasets_dir=Path("datasets"))
        transforms = LiveCodeBench_GPTOSS_SGLang.create_transforms()
        livecodebench_dataset = LiveCodeBench(
            df, transforms=transforms, repeats=num_repeats
        )
        return livecodebench_dataset
