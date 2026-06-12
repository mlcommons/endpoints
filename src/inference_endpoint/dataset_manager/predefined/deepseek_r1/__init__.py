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

import os
from logging import getLogger
from pathlib import Path

import pandas as pd

from ...dataset import Dataset

logger = getLogger(__name__)

#: Env var pointing at the local MLPerf DeepSeek-R1 source dataset.
SOURCE_ENV = "DEEPSEEK_R1_DATASET_PKL"

#: Columns required in a raw MLPerf source (pre-prepared parquets carry the
#: output columns directly and skip this check).
_REQUIRED_SOURCE_COLUMNS = ("tok_input", "ground_truth", "dataset", "question")

#: Columns the benchmark + DeepSeekR1Scorer consume.
_OUTPUT_COLUMNS = ["input_tokens", "ground_truth", "dataset", "question"]


class DeepSeekR1(Dataset, dataset_id="deepseek_r1"):
    """MLPerf DeepSeek-R1 combined-subset dataset (local source).

    The official MLCommons DeepSeek-R1 accuracy set ships as a pandas pickle
    bundling five subsets (``math500``/``aime``/``gpqa``/``mmlu_pro``/
    ``livecodebench``) with a pre-tokenized MLPerf prompt. This loader converts
    that local source into the benchmark's columns and caches the result as a
    parquet under ``<datasets_dir>/deepseek_r1/``:

      - ``input_tokens`` : pre-tokenized MLPerf prompt (source ``tok_input``);
        named so the ``openai_completions`` adapter's ``Harmonize()`` is a no-op
        and the server chat template is bypassed - the model sees the exact
        MLPerf prompt.
      - ``ground_truth`` : expected answer (LCB rows carry the LiveCodeBench id).
      - ``dataset``      : subset id, used by ``DeepSeekR1Scorer`` to route
        per-subset grading.
      - ``question``     : human-readable question text.

    One loader serves both phases: the perf phase issues ``input_tokens`` and the
    accuracy phase hands the rows to ``DeepSeekR1Scorer`` (which grades by
    ``dataset``/``ground_truth``).

    The source is not bundled (it is the official MLCommons dataset). Point
    ``$DEEPSEEK_R1_DATASET_PKL`` at your local ``.pkl`` (or an already-prepared
    ``.parquet``), or pass ``source=`` to :meth:`generate`.
    """

    COLUMN_NAMES = _OUTPUT_COLUMNS

    @classmethod
    def generate(
        cls,
        datasets_dir: Path,
        source: str | os.PathLike | None = None,
        max_samples: int | None = None,
        seed: int = 42,
        force: bool = False,
    ) -> pd.DataFrame:
        """Build (or load the cached) DeepSeek-R1 benchmark dataframe.

        Args:
            datasets_dir: Root cache dir; the parquet is written under
                ``<datasets_dir>/deepseek_r1/deepseek_r1_eval.parquet``.
            source: Local MLPerf ``.pkl`` (or prepared ``.parquet``). Falls back
                to ``$DEEPSEEK_R1_DATASET_PKL`` when omitted.
            max_samples: If set, return a stratified subset of this many rows
                (proportional per ``dataset`` subset) for a quick estimate.
            seed: Random seed for the stratified subset.
            force: Rebuild the cached parquet even if it already exists.
        """
        dst_path = datasets_dir / cls.DATASET_ID / "deepseek_r1_eval.parquet"
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists() and not force:
            logger.info("DeepSeek-R1 dataset cached at %s; loading.", dst_path)
            df = pd.read_parquet(dst_path)
        else:
            df = cls._build_from_source(source)
            df.to_parquet(dst_path, index=False)
            logger.info("Wrote %d DeepSeek-R1 rows to %s", len(df), dst_path)

        if max_samples is not None and max_samples < len(df):
            full_n = len(df)
            df = cls._stratified_subset(df, max_samples, seed)
            logger.info("Stratified subset: %d of %d rows", len(df), full_n)
        return df

    @staticmethod
    def _build_from_source(source: str | os.PathLike | None) -> pd.DataFrame:
        resolved = source or os.environ.get(SOURCE_ENV)
        if not resolved:
            raise FileNotFoundError(
                "DeepSeek-R1 source dataset not found. Set "
                f"${SOURCE_ENV} to the local MLPerf DeepSeek-R1 .pkl (or pass "
                "source=...). The official dataset is not bundled with the repo."
            )
        path = Path(resolved)
        if not path.exists():
            raise FileNotFoundError(f"DeepSeek-R1 source not found at {path}")

        # read_pickle executes arbitrary code on load, so it is only safe on
        # trusted input: `path` is an operator-supplied local file (via
        # $DEEPSEEK_R1_DATASET_PKL or an explicit source=), namely the official
        # MLCommons DeepSeek-R1 dataset, which ships as a pandas .pkl. It is
        # never a network/remote source. A prepared .parquet (already carrying
        # input_tokens) is also accepted and passed through.
        raw = (
            pd.read_pickle(path)
            if path.suffix in (".pkl", ".pickle")
            else pd.read_parquet(path)
        )

        if "input_tokens" in raw.columns:
            missing = [c for c in _OUTPUT_COLUMNS if c not in raw.columns]
            if missing:
                raise ValueError(
                    f"Prepared DeepSeek-R1 source {path} missing columns "
                    f"{missing}; found {list(raw.columns)}"
                )
            return raw[_OUTPUT_COLUMNS].reset_index(drop=True)

        missing = [c for c in _REQUIRED_SOURCE_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(
                f"DeepSeek-R1 source {path} missing expected columns {missing}; "
                f"found {list(raw.columns)}"
            )
        return pd.DataFrame(
            {
                "input_tokens": raw["tok_input"].map(
                    lambda t: t.tolist() if hasattr(t, "tolist") else list(t)
                ),
                "ground_truth": raw["ground_truth"].astype(str),
                "dataset": raw["dataset"].astype(str),
                "question": raw["question"].astype(str),
            }
        )

    @staticmethod
    def _stratified_subset(
        df: pd.DataFrame, max_samples: int, seed: int
    ) -> pd.DataFrame:
        frac = max_samples / len(df)
        parts = [
            group.sample(
                n=min(max(1, round(len(group) * frac)), len(group)),
                random_state=seed,
            )
            for _, group in df.groupby("dataset")
        ]
        return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
