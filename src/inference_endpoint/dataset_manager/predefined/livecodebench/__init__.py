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

import logging
import subprocess
import sys
import venv
from pathlib import Path

import pandas as pd
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.evaluation.livecodebench.generate import SCRIPT_PATH

from . import presets

logger = logging.getLogger(__name__)


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
    ]

    PRESETS = presets

    @classmethod
    def _ensure_venv(cls, venv_path: Path) -> Path:
        """Ensure a virtual environment exists with datasets==3.6.0 installed.

        Args:
            venv_path: Path to the virtual environment directory.

        Returns:
            Path to the Python executable in the virtual environment.
        """
        if not venv_path.exists():
            logger.info(f"Creating virtual environment at {venv_path}")
            venv.create(venv_path, with_pip=True, clear=True, symlinks=False)

        # Determine Python executable path based on platform
        if sys.platform == "win32":
            python_executable = venv_path / "Scripts" / "python.exe"
        else:
            python_executable = venv_path / "bin" / "python"

        if not python_executable.exists():
            raise RuntimeError(f"Python executable not found at {python_executable}")

        # Check if datasets package is installed with correct version
        try:
            result = subprocess.run(
                [
                    str(python_executable),
                    "-c",
                    "import datasets; print(datasets.__version__)",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            installed_version = result.stdout.strip()
            if installed_version != "3.6.0":
                logger.info(
                    f"datasets version {installed_version} found, reinstalling 3.6.0"
                )
                raise ValueError("Wrong version")
        except (subprocess.CalledProcessError, ValueError):
            # Install datasets==3.6.0
            logger.info("Installing datasets==3.6.0 in virtual environment")
            subprocess.run(
                [str(python_executable), "-m", "pip", "install", "datasets==3.6.0"],
                check=True,
                capture_output=True,
            )
            logger.info(
                f"datasets==3.6.0 installed in virtual environment at {venv_path}"
            )

        return python_executable

    @classmethod
    def generate(
        cls,
        datasets_dir: Path,
        variant: str = "release_v6",
        force: bool = False,
        save_test_cases: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Generates the LiveCodeBench reference dataset. By default, since evaluation should be run via the
        lcb-service container, only the necessary model inputs are saved.

        This method creates an isolated Python virtual environment with datasets==3.6.0
        and invokes the dataset generation script as a subprocess. The venv is created
        at datasets_dir/livecodebench/venv.

        Args:
            datasets_dir: Path to the base datasets directory where all datasets are stored.
            variant: The variant of the dataset to generate. (Default: "release_v6")
            force: Whether to force the generation of the dataset. (Default: False)
            save_test_cases: Whether to save test cases as separate JSON files. (Default: False)
            **kwargs: Additional keyword arguments to pass to the dataset generation method. These are ignored.

        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the dataset is not found at the destination path.
            RuntimeError: If the dataset generation fails.
        """
        filename = f"{cls.DATASET_ID}_{variant}.parquet"
        dst_path = datasets_dir / cls.DATASET_ID / variant / filename
        if not dst_path.parent.exists():
            dst_path.parent.mkdir(parents=True)

        if dst_path.exists() and not force:
            logger.info(f"Dataset already exists at {dst_path}. Loading from file.")
            return pd.read_parquet(dst_path)

        # Ensure venv exists with correct dependencies
        venv_path = datasets_dir / cls.DATASET_ID / "venv"
        python_executable = cls._ensure_venv(venv_path)

        # Build subprocess command
        cmd = [
            str(python_executable),
            str(SCRIPT_PATH),
            "--datasets-dir",
            str(dst_path.parent),
            "--variant",
            variant,
        ]

        if force:
            cmd.append("--force")

        if not save_test_cases:
            cmd.append("--no-test-cases")

        # Execute subprocess
        logger.info(f"Generating dataset with command: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            logger.info("Dataset generation completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Dataset generation failed with exit code {e.returncode}")
            raise RuntimeError("Dataset generation failed") from e

        # Load and return the generated dataset
        if dst_path.exists():
            return pd.read_parquet(dst_path)
        else:
            raise FileNotFoundError(
                f"Dataset generation reported success, but dataset not found at {dst_path}"
            )
