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

import gzip
import shutil
from logging import getLogger
from pathlib import Path

import pandas as pd

from ...dataset import Dataset
from ...download import download_r2_artifact, verify_sha256
from . import presets

logger = getLogger(__name__)


class OpenOrca(
    Dataset,
    dataset_id="open_orca",
):
    """OpenOrca GPT4 tokenized dataset for accuracy evaluation."""

    PRESETS = presets
    SOURCE_FILENAME = "open_orca_gpt4_tokenized_llama.sampled_24576.pkl"
    CACHE_FILENAME = "open_orca_gpt4_tokenized_llama.sampled_24576.jsonl"
    SOURCE_SHA256 = "b64e66e54b6267f79eb4f9ccec52d466bab3ac94747ed258c3b0f337ed166fab"
    DATASET_URI = (
        "https://inference.mlcommons-storage.org/metadata/"
        "llama-2-70b-open-orca-dataset.uri"
    )

    @classmethod
    def _extract_gz_files(cls, download_dir: Path, gzip_dir: Path) -> None:
        """Extract .pkl.gz files into download_dir, overwriting any existing .pkl files."""
        for gz_path in gzip_dir.glob("*.pkl.gz"):
            pkl_path = download_dir / gz_path.with_suffix("").name
            with gzip.open(gz_path, "rb") as f_in:
                with pkl_path.open(mode="wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
        logger.info("OpenOrca dataset extracted to: %s", download_dir.resolve())

    @classmethod
    def _convert_pickle_cache(cls, pickle_path: Path, jsonl_path: Path) -> pd.DataFrame:
        """Convert the upstream pickle artifact into the local JSONL cache."""
        verify_sha256(pickle_path, cls.SOURCE_SHA256)
        dataframe = pd.read_pickle(pickle_path)
        tmp_path = jsonl_path.with_suffix(".jsonl.tmp")
        try:
            dataframe.to_json(tmp_path, orient="records", lines=True)
            tmp_path.replace(jsonl_path)
        finally:
            tmp_path.unlink(missing_ok=True)
        pickle_path.unlink()
        return dataframe

    @classmethod
    def generate(
        cls,
        datasets_dir: Path,
        variant: str = "mlperf-inference",
        force: bool = False,
    ):
        """Download and extract the OpenOrca dataset files from MLCommons storage. This is
        a curated and preprocessed dataset from MLCommons based on OpenOrca for MLPerf Inference.

        The dataset contains 24576 samples and is filtered to be compatible with the ISL, OSL, and
        max tokens rules for Llama2-70b in MLPerf Inference.

        See https://github.com/mlcommons/inference/tree/master/language/llama2-70b for more details
        on the Llama2-70b benchmark. The script used to generate the dataset from the full OpenOrca
        dataset is here:
        https://github.com/mlcommons/inference/blob/master/language/llama2-70b/processorca.py

        Args:
            datasets_dir: The root datasets directory to save the dataset under. A
                subdirectory with the name and variant of the dataset will be created if
                it does not exist.
            variant: The variant of the dataset to generate. Defaults to "mlperf-inference".
                Currently only "mlperf-inference" is supported. The "full", default OpenOrca dataset
                should be added in the future.
            force: If True, the dataset will be regenerated even if it already exists.
                Defaults to False.

        Returns:
            A pandas dataframe containing the dataset.
        """
        if variant != "mlperf-inference":
            raise ValueError(f"Unsupported variant: {variant}")

        cache_dir = datasets_dir / cls.DATASET_ID / variant
        jsonl_path = cache_dir / cls.CACHE_FILENAME
        pickle_path = cache_dir / cls.SOURCE_FILENAME
        if not cache_dir.exists():
            cache_dir.mkdir(parents=True)

        if jsonl_path.exists() and not force:
            logger.info("Dataset already exists at %s. Loading from file.", jsonl_path)
            return pd.read_json(jsonl_path, lines=True)

        if pickle_path.exists() and not force:
            logger.info(
                "Converting existing upstream pickle cache at %s to JSONL.", pickle_path
            )
            return cls._convert_pickle_cache(pickle_path, jsonl_path)

        downloaded_gzip = download_r2_artifact(
            uri=cls.DATASET_URI,
            destination_dir=cache_dir,
            artifact_name=f"{cls.SOURCE_FILENAME}.gz",
        )
        try:
            cls._extract_gz_files(cache_dir, downloaded_gzip.parent)
            if not pickle_path.exists():
                raise FileNotFoundError(
                    f"OpenOrca was downloaded, but {pickle_path} does not exist"
                )
            return cls._convert_pickle_cache(pickle_path, jsonl_path)
        finally:
            for gz_path in downloaded_gzip.parent.glob("*.pkl.gz"):
                gz_path.unlink()


__all__ = ["OpenOrca"]
