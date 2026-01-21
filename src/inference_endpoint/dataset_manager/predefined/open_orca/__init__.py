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
from inference_endpoint.dataset_manager.transforms import (
    AddStaticColumns,
    ColumnNameRemap,
)

from ...dataset import Dataset

logger = getLogger(__name__)


class OpenOrca(
    Dataset,
    dataset_id="open_orca",
):
    """OpenOrca GPT4 tokenized dataset for accuracy evaluation."""

    @classmethod
    def generate(cls):
        """Download and extract the OpenOrca dataset files into a local folder.

        Returns:
            Path: path to the extracted folder containing the dataset files.
        """
        target_dir = Path("open_orca")
        target_dir.mkdir(parents=True, exist_ok=True)

        # Dataset URL from README
        dataset_url = "https://inference.mlcommons-storage.org/metadata/llama-2-70b-open-orca-dataset.uri"

        # Download the r2-downloader script into a temp file in the target dir
        downloader_url = "https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh"
        script_path = target_dir / "mlc-r2-downloader.sh"
        r = requests.get(downloader_url, timeout=30)
        r.raise_for_status()
        script_path.write_bytes(r.content)
        script_path.chmod(0o755)

        # Run the script with the dataset URL.
        try:
            # Use absolute path for the script to avoid path doubling when cwd is set
            script_abs = str(script_path.resolve())
            result = subprocess.run(
                ["bash", script_abs, dataset_url],
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

        # After downloader ran, decompress any .pkl.gz files inside open_orca
        # Suppress decompression logs
        open_orca_dir = target_dir / "open_orca"
        if not open_orca_dir.exists():
            # Some versions of the script drop files directly under target_dir
            open_orca_dir = target_dir

        for gz_path in open_orca_dir.glob("*.pkl.gz"):
            pkl_path = gz_path.with_suffix("")
            # Note: .with_suffix removes only one suffix, .pkl.gz -> .pkl
            if pkl_path.exists():
                continue
            with gzip.open(gz_path, "rb") as f_in:
                with open(pkl_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        logger.info(
            "OpenOrca dataset downloaded and extracted to: %s", target_dir.resolve()
        )
        return target_dir

    @classmethod
    def get_dataloader(
        cls,
        metadata: dict[str, str] | None = None,
        num_repeats: int = 1,
    ):
        """Load the OpenOrca dataset from a file.

        Args:
            dataset_path: Path to the dataset file (typically .pkl format)
            metadata: Optional metadata to add to the dataset
            remap: Column name remapping dict. If None, defaults to {"prompt": "text_input"}

        Returns:
            OpenOrca dataset instance
        """
        # Generate dataset
        dataset_dir = cls.generate()

        # Determine dataset path
        dataset_path = dataset_dir / "open_orca_gpt4_tokenized_llama.sampled_24576.pkl"

        # Load the dataset from file
        df = pd.read_pickle(dataset_path)

        # Apply same parser/remap logic as factory.py
        remap = {"question": "prompt", "system_prompt": "system"}

        # Create transforms
        transforms = [ColumnNameRemap(remap)]

        if metadata is not None:
            transforms.append(AddStaticColumns(metadata))

        return cls(df, transforms=transforms)


__all__ = ["OpenOrca"]
