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

#: Env var pointing at the local prepared DeepSeek-R1 ``.parquet``.
SOURCE_ENV = "LEGACY_MLPERF_DEEPSEEK_R1_DATASET"

#: Prepared parquet bundled in the repo (git-LFS). Used as the default source
#: when neither an explicit ``source=`` nor ``$LEGACY_MLPERF_DEEPSEEK_R1_DATASET``
#: is set. Present only in a source checkout - it is not packaged in the wheel
#: (it lives under ``examples/``), so the lookup is guarded by ``.exists()``.
_BUNDLED_PARQUET = (
    Path(__file__).resolve().parents[5]
    / "examples/07_DeepSeekR1_Example/data/deepseek_r1_eval.parquet"
)


def _is_lfs_pointer(path: Path) -> bool:
    """True if ``path`` is an unresolved git-LFS pointer stub, not real content.

    A clone without git-LFS leaves a tiny text stub in place of the parquet;
    detecting it lets the loader give an actionable error instead of letting
    ``pd.read_parquet`` fail cryptically.
    """
    try:
        if path.stat().st_size > 1024:  # real parquet is MBs; pointers are ~130 B
            return False
        with path.open("rb") as f:
            return f.read(64).startswith(b"version https://git-lfs.github.com/spec")
    except OSError:
        return False


class LegacyMLPerfDeepSeekR1(Dataset, dataset_id="legacy_mlperf_deepseek_r1"):
    """MLPerf DeepSeek-R1 combined-subset dataset (local source).

    The official MLCommons DeepSeek-R1 accuracy set ships as a pandas pickle
    bundling five subsets (``math500``/``aime``/``gpqa``/``mmlu_pro``/
    ``livecodebench``) with a pre-tokenized MLPerf prompt. This loader consumes
    a prepared ``.parquet`` of that source (already carrying the benchmark's
    columns) and caches it under ``<datasets_dir>/legacy_mlperf_deepseek_r1/``:

      - ``input_tokens`` : pre-tokenized MLPerf prompt (source ``tok_input``);
        named so the ``openai_completions`` adapter's ``Harmonize()`` is a no-op
        and the server chat template is bypassed - the model sees the exact
        MLPerf prompt.
      - ``ground_truth`` : expected answer (LCB rows carry the LiveCodeBench id).
      - ``dataset``      : subset id, used by ``LegacyMLPerfDeepSeekR1Scorer`` to route
        per-subset grading.
      - ``question``     : human-readable question text.

    One loader serves both phases: the perf phase issues ``input_tokens`` and the
    accuracy phase hands the rows to ``LegacyMLPerfDeepSeekR1Scorer`` (which grades by
    ``dataset``/``ground_truth``).

    A prepared ``.parquet`` ships in the repo (git-LFS, under
    ``examples/07_DeepSeekR1_Example/data/``) and is the default source. To use a
    different or updated prepared ``.parquet`` (e.g. a new MLPerf revision or a
    variant), point ``$LEGACY_MLPERF_DEEPSEEK_R1_DATASET`` at it or pass
    ``source=`` to :meth:`generate`. Building from the raw MLPerf source is not
    supported.
    """

    COLUMN_NAMES = ["input_tokens", "ground_truth", "dataset", "question"]

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
                ``<datasets_dir>/legacy_mlperf_deepseek_r1/deepseek_r1_eval.parquet``.
            source: Prepared DeepSeek-R1 ``.parquet`` (already carrying
                ``input_tokens``). Falls back to ``$LEGACY_MLPERF_DEEPSEEK_R1_DATASET``
                when omitted.
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
            df = cls._load_prepared_parquet(source)
            df.to_parquet(dst_path, index=False)
            logger.info("Wrote %d DeepSeek-R1 rows to %s", len(df), dst_path)

        if max_samples is not None and max_samples < len(df):
            full_n = len(df)
            df = cls._stratified_subset(df, max_samples, seed)
            logger.info("Stratified subset: %d of %d rows", len(df), full_n)
        return df

    @classmethod
    def _load_prepared_parquet(cls, source: str | os.PathLike | None) -> pd.DataFrame:
        resolved = source or os.environ.get(SOURCE_ENV)
        if not resolved and _BUNDLED_PARQUET.exists():
            resolved = _BUNDLED_PARQUET
        if not resolved:
            raise FileNotFoundError(
                "DeepSeek-R1 source dataset not found. Set "
                f"${SOURCE_ENV} to a prepared DeepSeek-R1 .parquet (or pass "
                "source=...). A prepared copy ships at "
                "examples/07_DeepSeekR1_Example/data/deepseek_r1_eval.parquet "
                "and is used automatically from a source checkout."
            )
        path = Path(resolved)
        if not path.exists():
            raise FileNotFoundError(f"DeepSeek-R1 source not found at {path}")

        # A git-LFS pointer (clone without LFS) "exists" but is a tiny text stub,
        # not a parquet; surface that clearly instead of a cryptic read failure.
        if _is_lfs_pointer(path):
            raise FileNotFoundError(
                f"DeepSeek-R1 parquet {path} is an unresolved git-LFS pointer "
                "(cloned without LFS?). Run `git lfs pull`, or set "
                f"${SOURCE_ENV} to a real prepared .parquet."
            )

        # Only a prepared .parquet (already carrying the output columns,
        # including the pre-tokenized input_tokens) is supported. Converting a
        # raw MLPerf source is intentionally out of scope: prepare the parquet
        # offline (see examples/07_DeepSeekR1_Example/README.md).
        if path.suffix != ".parquet":
            raise NotImplementedError(
                f"DeepSeek-R1 source {path} is not a prepared .parquet. "
                "Building from a raw MLPerf source is not supported; pass a "
                "prepared .parquet carrying an 'input_tokens' column."
            )

        raw = pd.read_parquet(path)
        if "input_tokens" not in raw.columns:
            raise NotImplementedError(
                f"DeepSeek-R1 parquet {path} has no 'input_tokens' column. "
                "Building from a raw (un-tokenized) MLPerf source is not "
                "supported; pass a prepared .parquet."
            )
        missing = [c for c in cls.COLUMN_NAMES if c not in raw.columns]
        if missing:
            raise ValueError(
                f"Prepared DeepSeek-R1 source {path} missing columns "
                f"{missing}; found {list(raw.columns)}"
            )
        return raw[cls.COLUMN_NAMES].reset_index(drop=True)

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
            for _, group in df.groupby("dataset")
        ]
        # The per-subset >=1 floor + rounding can push the pool over max_samples
        # (e.g. many subsets, small max_samples); shuffle and trim so the result
        # stays stratified but never returns more than requested.
        pool = pd.concat(parts).sample(frac=1, random_state=seed)
        return pool.head(max_samples).reset_index(drop=True)
