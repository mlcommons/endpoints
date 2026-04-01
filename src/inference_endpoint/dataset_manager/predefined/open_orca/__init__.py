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
import subprocess
from logging import getLogger
from pathlib import Path

import pandas as pd
import requests

from ...dataset import Dataset
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

    @classmethod
    def _convert_pickle_cache(cls, pickle_path: Path, jsonl_path: Path) -> pd.DataFrame:
        """Convert the upstream pickle artifact into the local JSONL cache."""
        dataframe = pd.read_pickle(pickle_path)
        dataframe.to_json(jsonl_path, orient="records", lines=True)
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

        if pickle_path.exists():
            logger.info(
                "Converting existing upstream pickle cache at %s to JSONL.", pickle_path
            )
            return cls._convert_pickle_cache(pickle_path, jsonl_path)

        # Dataset URL from README
        dataset_url = "https://inference.mlcommons-storage.org/metadata/llama-2-70b-open-orca-dataset.uri"

        # Download the r2-downloader script into a temp file in the target dir
        COMMIT_HASH = "27da4421877f2831eeb615b43ee5098c4b70be7e"
        downloader_url = f"https://raw.githubusercontent.com/mlcommons/r2-downloader/{COMMIT_HASH}/mlc-r2-downloader.sh"
        download_dir = cache_dir
        script_path = cache_dir / "mlc-r2-downloader.sh"
        r = requests.get(downloader_url, timeout=30)
        r.raise_for_status()
        script_path.write_bytes(r.content)
        script_path.chmod(0o755)

        # Run the script with the dataset URL.
        try:
            # Use absolute path for the script to avoid path doubling when cwd is set
            script_abs = str(script_path.resolve())
            result = subprocess.run(
                ["bash", script_abs, "-d", str(download_dir), dataset_url],
                stdout=subprocess.DEVNULL,  # Suppress normal output
                stderr=subprocess.PIPE,  # Capture errors
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"R2 downloader failed with code {result.returncode}: {result.stderr}"
                )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"R2 downloader failed: {e}") from e

        # Script will generate a new 'open_orca' subdirectory with gzip'd pickle files
        gzip_dir = download_dir
        if (gzip_dir / "open_orca").exists():
            gzip_dir = gzip_dir / "open_orca"

        for gz_path in gzip_dir.glob("*.pkl.gz"):
            # Note: .with_suffix removes only one suffix, .pkl.gz -> .pkl
            pkl_filename = gz_path.with_suffix("").name
            pkl_path = download_dir / pkl_filename
            if pkl_path.exists():
                continue

            with gzip.open(gz_path, "rb") as f_in:
                with pkl_path.open(mode="wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        logger.info(
            "OpenOrca dataset downloaded and extracted to: %s",
            download_dir.resolve(),
        )

        if not pickle_path.exists():
            raise FileNotFoundError(
                f"OpenOrca was downloaded, but {pickle_path} does not exist"
            )

        # Clean up the intermediate gz files
        for gz_path in gzip_dir.glob("*.pkl.gz"):
            gz_path.unlink()

        script_path.unlink(missing_ok=True)
        logger.info(f"Cleaned up intermediate gz files in {gzip_dir}")

        return cls._convert_pickle_cache(pickle_path, jsonl_path)


__all__ = ["OpenOrca"]
