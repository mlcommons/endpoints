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

from ...dataset import Dataset, load_from_huggingface

logger = getLogger(__name__)

_REPO_MAP = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "lite": "princeton-nlp/SWE-bench_Lite",
}


class SWEBench(
    Dataset,
    dataset_id="swe_bench",
):
    """SWE-bench: Software Engineering Benchmark for LLM agents.

    Loads instance IDs and problem statements from the SWE-bench Verified or
    Lite subset. Used as the accuracy dataset for the swe_bench_scorer, which
    runs mini-swe-agent against a live endpoint and grades patches with the
    SWE-bench evaluation harness.

    The ``instance_id`` column identifies which instances mini-swe-agent will
    evaluate. The endpoint phase is skipped entirely for this scorer
    (``SKIP_ENDPOINT_PHASE=True``); ``SWEBenchScorer`` drives the agent
    subprocess directly against the configured endpoint.

    Using this dataset as a performance dataset (type: performance) is not
    meaningful — problem statements sent directly to the model without an
    agent framework don't reflect real SWE-bench usage. Use a different
    dataset (e.g. ``random``) for the performance phase.
    """

    ACCURACY_ONLY = True
    COLUMN_NAMES = ["instance_id", "prompt"]

    @classmethod
    def generate(
        cls,
        datasets_dir: Path,
        subset: str = "verified",
        force: bool = False,
    ) -> pd.DataFrame:
        """Download and cache the SWE-bench dataset from HuggingFace.

        Args:
            datasets_dir: Root cache directory. Parquet is written under
                ``datasets_dir/swe_bench/{subset}/``.
            subset: ``"verified"`` (500 instances) or ``"lite"`` (300 instances).
            force: Re-download even if the local parquet cache exists.

        Returns:
            DataFrame with columns ``instance_id`` and ``prompt``.
        """
        hf_path = _REPO_MAP.get(subset)
        if hf_path is None:
            raise ValueError(
                f"Unknown SWE-bench subset {subset!r}; choose from: {list(_REPO_MAP)}"
            )

        dst_path = datasets_dir / "swe_bench" / subset / f"swe_bench_{subset}.parquet"
        if dst_path.exists() and not force:
            logger.info("Loading SWE-bench %s from cache: %s", subset, dst_path)
            try:
                return pd.read_parquet(dst_path)
            except Exception as e:
                raise RuntimeError(
                    f"Cached SWE-bench parquet at {dst_path} appears corrupt ({e}). "
                    "Delete it or pass force=True to re-download."
                ) from e

        try:
            df = load_from_huggingface(
                hf_path,
                split="test",
                cache_dir=datasets_dir / "hf_cache" / f"swe_bench_{subset}",
            )
        except Exception as e:
            logger.error("Error loading SWE-bench %s from HuggingFace: %s", subset, e)
            raise

        result = (
            df[["instance_id", "problem_statement"]]
            .rename(columns={"problem_statement": "prompt"})
            .reset_index(drop=True)
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(dst_path)
        logger.info(
            "Saved %d SWE-bench %s instances to %s", len(result), subset, dst_path
        )
        return result
