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

_UPSTREAM_PKL_FILENAME = "open_orca_gpt4_tokenized_llama.sampled_24576.pkl"
_PARQUET_FILENAME = "open_orca_gpt4_tokenized_llama.sampled_24576.parquet"


class OpenOrca(
    Dataset,
    dataset_id="open_orca",
):
    """OpenOrca GPT4 tokenized dataset for accuracy evaluation."""

    PRESETS = presets

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

        variant_dir = datasets_dir / cls.DATASET_ID / variant
        if not variant_dir.exists():
            variant_dir.mkdir(parents=True)

        parquet_path = variant_dir / _PARQUET_FILENAME
        pkl_path = variant_dir / _UPSTREAM_PKL_FILENAME

        # Return cached parquet if available
        if parquet_path.exists() and not force:
            logger.info(f"Dataset already exists at {parquet_path}. Loading from file.")
            return pd.read_parquet(parquet_path)

        # Legacy pickle cache — convert to parquet and remove
        if pkl_path.exists() and not force:
            logger.info("Converting legacy pickle cache to parquet: %s", pkl_path)
            df = pd.read_pickle(pkl_path)
            df.to_parquet(parquet_path)
            pkl_path.unlink()
            return df

        # Dataset URL from README
        dataset_url = "https://inference.mlcommons-storage.org/metadata/llama-2-70b-open-orca-dataset.uri"

        # Download the r2-downloader script into a temp file in the target dir
        COMMIT_HASH = "27da4421877f2831eeb615b43ee5098c4b70be7e"
        downloader_url = f"https://raw.githubusercontent.com/mlcommons/r2-downloader/{COMMIT_HASH}/mlc-r2-downloader.sh"
        download_dir = variant_dir
        script_path = variant_dir / "mlc-r2-downloader.sh"
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

        # Script generates gzip'd pickle files — decompress, convert to parquet, clean up
        gzip_dir = download_dir
        if (gzip_dir / "open_orca").exists():
            gzip_dir = gzip_dir / "open_orca"

        for gz_path in gzip_dir.glob("*.pkl.gz"):
            pkl_filename = gz_path.with_suffix("").name
            tmp_pkl_path = download_dir / pkl_filename
            if tmp_pkl_path.exists():
                continue

            with gzip.open(gz_path, "rb") as f_in:
                with tmp_pkl_path.open(mode="wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        logger.info(
            "OpenOrca dataset downloaded and extracted to: %s",
            download_dir.resolve(),
        )

        # Check the upstream pickle file exists, convert to parquet, and clean up
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"OpenOrca was downloaded, but {pkl_path} does not exist"
            )

        df = pd.read_pickle(pkl_path)
        df.to_parquet(parquet_path)
        logger.info("Converted to parquet: %s", parquet_path)

        # Clean up intermediate pickle and gz files
        pkl_path.unlink()
        for gz_path in gzip_dir.glob("*.pkl.gz"):
            gz_path.unlink()

        logger.info(f"Cleaned up intermediate files in {gzip_dir}")

        return df


__all__ = ["OpenOrca"]
