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

from logging import getLogger
from pathlib import Path

import pandas as pd

from ...dataset import Dataset
from ...transforms import Transform, apply_transforms
from ..aime25 import AIME25
from ..aime25 import presets as aime25_presets
from ..gpqa import GPQA
from ..gpqa import presets as gpqa_presets
from ..livecodebench import LiveCodeBench
from ..livecodebench import presets as livecodebench_presets

logger = getLogger(__name__)


class GptOss120bAccuracy(Dataset, dataset_id="gptoss_120b_accuracy"):
    """Composite MLPerf gpt-oss-120b accuracy dataset (single dataset -> single entry).

    Bundles the three gpt-oss-120b accuracy subsets (``aime25`` / ``gpqa`` /
    ``livecodebench``) into one dataset with a routing ``subset`` column, so a
    config can pick this single entry instead of wiring the three individual
    ``::gptoss`` datasets. It is graded by :class:`GptOss120bAccuracyScorer`,
    which routes per-subset in-process and reports one score + a per-subset
    breakdown — mirroring the DeepSeek-R1 composite model.

    Unified schema:

      - ``prompt``       : the per-subset-rendered user prompt. Each subset's own
        ``gptoss()`` preset is applied here at build time (gpqa needs its shuffled
        ``choice1-4``, livecodebench its ``starter_code``), so a single
        dataset-level preset can't reproduce it — the prompts are baked in.
      - ``subset``       : routing key ∈ {``aime25``, ``gpqa``, ``livecodebench``}.
      - ``ground_truth`` : expected answer as a string — aime25's numeric answer,
        gpqa's ``choiceN`` label, or livecodebench's question id.
      - ``question``     : human-readable question text (parity/debug).

    The build composes the three subsets from their live sources (HuggingFace for
    aime25/gpqa, the LiveCodeBench generator venv for livecodebench) and caches the
    result; there is no committed composite artifact. Per-subset ``num_repeats``
    (aime 8 / gpqa 5 / lcb 3) is NOT expressed here — the composite issues a single
    uniform repeat count; use the individual ``::gptoss`` datasets when per-subset
    repeats are required.
    """

    COLUMN_NAMES = ["prompt", "subset", "ground_truth", "question"]

    @classmethod
    def generate(
        cls,
        datasets_dir: Path,
        gpqa_variant: str = "diamond",
        lcb_variant: str = "release_v6",
        max_samples: int | None = None,
        seed: int = 0,
        force: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Build (or load the cached) composite gpt-oss-120b accuracy dataframe.

        Args:
            datasets_dir: Root cache dir; the parquet is written under
                ``<datasets_dir>/gptoss_120b_accuracy/gptoss_120b_accuracy.parquet``.
            gpqa_variant: GPQA variant to compose (passed to :meth:`GPQA.generate`).
            lcb_variant: LiveCodeBench variant (passed to
                :meth:`LiveCodeBench.generate`).
            max_samples: If set, return a stratified subset of this many rows
                (proportional per ``subset``) for a quick estimate.
            seed: Random seed for the stratified subset.
            force: Rebuild the cached parquet (and its subsets) even if present.
            **kwargs: Ignored (forwarded by ``get_dataloader``).
        """
        dst_path = datasets_dir / cls.DATASET_ID / f"{cls.DATASET_ID}.parquet"
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists() and not force:
            logger.info(
                "gpt-oss-120b composite dataset cached at %s; loading.", dst_path
            )
            df = pd.read_parquet(dst_path)
        else:
            parts = [
                cls._build_subset(
                    "aime25",
                    AIME25.generate(datasets_dir=datasets_dir, force=force),
                    aime25_presets.gptoss(),
                    ground_truth_column="answer",
                ),
                cls._build_subset(
                    "gpqa",
                    GPQA.generate(
                        datasets_dir=datasets_dir, variant=gpqa_variant, force=force
                    ),
                    gpqa_presets.gptoss(),
                    ground_truth_column="ground_truth",
                ),
                cls._build_subset(
                    "livecodebench",
                    LiveCodeBench.generate(
                        datasets_dir=datasets_dir, variant=lcb_variant, force=force
                    ),
                    livecodebench_presets.gptoss(),
                    ground_truth_column="question_id",
                ),
            ]
            df = pd.concat(parts, ignore_index=True)
            df.to_parquet(dst_path, index=False)
            logger.info("Wrote %d composite rows to %s", len(df), dst_path)

        if max_samples is not None and max_samples < len(df):
            full_n = len(df)
            df = cls._stratified_subset(df, max_samples, seed)
            logger.info("Stratified subset: %d of %d rows", len(df), full_n)
        return df

    @staticmethod
    def _build_subset(
        subset: str,
        raw: pd.DataFrame,
        transforms: list[Transform],
        ground_truth_column: str,
    ) -> pd.DataFrame:
        """Render one subset's ``gptoss()`` prompt and project to the unified schema.

        ``transforms`` is that subset's ``gptoss()`` preset; applying it adds the
        ``prompt`` column from the subset's native columns. ``ground_truth_column``
        selects the subset's answer column (aime ``answer`` / gpqa ``ground_truth``
        / lcb ``question_id``), cast to ``str`` for exact-match comparison.
        """
        rendered = apply_transforms(raw, transforms)
        return pd.DataFrame(
            {
                "prompt": rendered["prompt"].astype(str).to_numpy(),
                "subset": subset,
                "ground_truth": rendered[ground_truth_column].astype(str).to_numpy(),
                "question": rendered["question"].astype(str).to_numpy(),
            }
        )

    @staticmethod
    def _stratified_subset(
        df: pd.DataFrame, max_samples: int, seed: int
    ) -> pd.DataFrame:
        if df.empty:
            return df
        frac = max_samples / len(df)
        parts = [
            group.sample(
                n=min(max(1, round(len(group) * frac)), len(group)),
                random_state=seed,
            )
            for _, group in df.groupby("subset")
        ]
        # The per-subset >=1 floor + rounding can push the pool over max_samples;
        # shuffle and trim so the result stays stratified but never exceeds it.
        pool = pd.concat(parts).sample(frac=1, random_state=seed)
        return pool.head(max_samples).reset_index(drop=True)
