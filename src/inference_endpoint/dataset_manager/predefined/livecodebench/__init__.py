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

import json
import os
import urllib.request
from logging import getLogger
from pathlib import Path

import pandas as pd

from ...dataset import Dataset
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
        "question_id",
        "question",
        "starter_code",
        "difficulty",
        "public_test_cases",
        "private_test_cases",
        "func_name",
    ]

    PRESETS = presets

    SERVER_ADDRESS = os.getenv("LCB_SERVER_ADDRESS", "127.0.0.1:13835")

    @classmethod
    def generate(
        cls,
        datasets_dir: Path,
        variant: str = "release_v6",
        force: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Generates the LiveCodeBench reference dataset for accuracy evaluation. LiveCodeBench
        requires a container to be running in the background as a webservice. The base container
        contains a copy of the dataset inside at /opt/LiveCodeBench_Datasets/livecodebench_release_v6.parquet.

        You can either copy it out via `docker cp`, or use this method, which requires the
        container to be running, and the local <datasets_dir>/livecodebench/release_v6 directory
        to be mounted on the container at /mnt/datasets.

        ** NOTE ** It is the SUBDIRECTORY in the datasets_dir that must be mounted, not the entire datasets_dir

        If the copy fails, or this directory is not mounted correctly, this method will raise an exception,

        Args:
            datasets_dir: Path to the base datasets directory where all datasets are stored.
            variant: The variant of the dataset to generate. (Default: "release_v6")
            force: Whether to force the generation of the dataset. (Default: False)
            **kwargs: Additional keyword arguments to pass to the dataset generation method. These are ignored.

        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the dataset is not found at the destination path.
            RuntimeError: If the dataset copy fails.
        """
        filename = f"{cls.DATASET_ID}_{variant}.parquet"
        dst_path = datasets_dir / cls.DATASET_ID / variant / filename
        if not dst_path.parent.exists():
            dst_path.parent.mkdir(parents=True)

        if dst_path.exists() and not force:
            logger.info(f"Dataset already exists at {dst_path}. Loading from file.")
            return pd.read_parquet(dst_path)

        # Try to get dataset from LCB server endpoint first
        lcb_server_url = f"http://{cls.SERVER_ADDRESS}"
        logger.info(f"Attempting to copy dataset from LCB server at {lcb_server_url}")
        req = urllib.request.Request(f"{lcb_server_url}/copy_dataset", method="GET")

        # Any error that occurs here should be fatal, up to caller to handle gracefully.
        with urllib.request.urlopen(req, timeout=30) as response:
            response_data = response.read()
            result = json.loads(response_data)

            if result.get("success"):
                # Dataset was successfully copied to /mnt/datasets
                if dst_path.exists():
                    return pd.read_parquet(dst_path)
                else:
                    raise FileNotFoundError(
                        f"Server-side copy succeeded, but dataset not found at {dst_path}, is it mounted correctly?"
                    )
            else:
                raise RuntimeError(f"LCB server copy failed: {result.get('error')}")
