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

    PRESETS = presets

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

        keep = [
            "question_id",
            "question_content",
            "starter_code",
            "public_test_cases",
            "private_test_cases",
            "platform",
        ]
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


class LiveCodeBench_GPTOSS_SGLang(
    LiveCodeBench, dataset_id="livecodebench_gptoss_sglang"
):
    """LiveCodeBench for GPTOSS-SGLang"""

    @classmethod
    def get_dataloader(
        cls,
        datasets_dir: Path = Path("datasets"),
        num_repeats: int = 1,
        transforms: list[Transform] | None = None,
        force_regenerate: bool = False,
    ) -> "Dataset":
        transforms = (transforms or []) + LiveCodeBench.PRESETS.gptoss_sglang()

        df = LiveCodeBench.generate(force=force_regenerate, datasets_dir=datasets_dir)
        return cls(df, transforms=transforms, repeats=num_repeats)
