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

"""LiveCodeBench (GitHub: https://github.com/LiveCodeBench/LiveCodeBench,
HuggingFace: https://huggingface.co/datasets/livecodebench/code_generation_lite)
is evaluated by running model-generated code against predefined test cases.

The LiveCodeBench authors note that running arbitrary code is dangerous, and
the existing lcb_runner scripts do not provide proper security and sandboxing.

As such, we provide a simple server that can be run inside containerized environments
such as Docker or enroot to run evaluation as a standalone service, which must be
started up manually before running Inference Endpoints.

This script is standalone, and does not require Inference Endpoints to be installed,
but can be invoked by running it as a module if it is.

It is assumed that:
1. LiveCodeBench is cloned in /opt/LiveCodeBench
2. LiveCodeBench has been installed to Python via pip or other means
3. A mechanism exists to transfer files from the Host to the Container LCB-Serve is
running in.
"""

import argparse
import os
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


class _LCBWorker:
    def __init__(
        self,
        lcb_root: Path = Path("/opt/LiveCodeBench"),
        n_lcb_workers: int = 1,
    ):
        if not lcb_root.exists():
            raise FileNotFoundError(
                f"LiveCodeBench root directory {lcb_root} does not exist"
            )
        self.lcb_root = lcb_root
        self.n_lcb_workers = n_lcb_workers

    def __call__(self, test_suites, extracted_code):
        # LiveCodeBench assumes that it is run from the root of its repository. As
        # such, we need to chdir() to it before any imports are done
        os.chdir(self.lcb_root)
        os.environ["TQDM_DISABLE"] = "1"

        from lcb_runner.evaluation import extract_instance_results
        from lcb_runner.runner.scenario_router import (
            get_metrics,
            sort_and_extract_save_results,
        )
        from lcb_runner.utils.scenarios import Scenario

        save_results = [
            suite.insert_output(output, output)
            for suite, output in zip(test_suites, extracted_code, strict=False)
        ]

        save_results, combined_results = sort_and_extract_save_results(
            Scenario.codegeneration, save_results
        )

        mock_args = argparse.Namespace(
            timeout=60,
            num_process_evaluate=self.n_lcb_workers,
        )
        _, instance_results, _ = get_metrics(
            Scenario.codegeneration,
            mock_args,
            sorted(test_suites, key=lambda x: x.question_id),
            combined_results,
        )
        graded = extract_instance_results(instance_results)

        # Currently, Endpoints scoring doesn't care about the reason for failed tests, just the
        # score itself. Also currently lcb_runner is hard-coded and only supports Pass@1 scoring.
        # In the future, if we want to log test failure reasons, it should be added here.
        return graded


@contextmanager
def chdir(path: Path):
    """Context manager to change the current working directory to the given path."""
    old_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_dir)


class LCBServe:
    def __init__(
        self,
        version_tag: str = "release_v6",
        use_lite: bool = True,
        n_workers: int = 8,
        output_file_store: Path = Path("/mnt/lcb_outputs"),
        lcb_root: Path = Path("/opt/LiveCodeBench"),
    ):
        self.version_tag = version_tag
        self.use_lite = use_lite
        self.n_workers = n_workers
        self.output_file_store = Path(output_file_store)
        self.lcb_root = Path(lcb_root)

        if not self.output_file_store.exists():
            self.output_file_store.mkdir(parents=True)

        if not self.lcb_root.exists():
            raise FileNotFoundError(
                f"LiveCodeBench root directory {self.lcb_root} does not exist"
            )

        self.test_suites = self.load_test_suites()

    def load_test_suites(self):
        with chdir(self.lcb_root):
            os.environ["TQDM_DISABLE"] = "1"

            from lcb_runner.runner.scenario_router import build_prompt_benchmark
            from lcb_runner.utils.scenarios import Scenario

            mock_args = argparse.Namespace(
                scenario=Scenario.codegeneration,
                release_version=self.version_tag,
                start_date=None,
                end_date=None,
                not_fast=not self.use_lite,
            )
            test_suites, _ = build_prompt_benchmark(mock_args)
        return {suite.question_id: suite for suite in test_suites}

    def eval_parquet(self, parquet_file: Path) -> tuple[float, int]:
        """Evaluates all LiveCodeBench problems in a parquet file.

        Args:
            parquet_file: Path to the parquet file containing the outputs to evaluate.

        Returns:
            tuple[float, int]: The pass@1 score and the number of samples that failed to extract code.

        Raises:
            FileNotFoundError: If the parquet file does not exist.
            ValueError: If the extracted code column or question ID column is not found in the parquet file.
        """
        file_path = self.output_file_store / parquet_file
        if not file_path.exists():
            raise FileNotFoundError(f"Output file {file_path} does not exist")

        df = pd.read_parquet(file_path)
        if "extracted_code" not in df.columns:
            raise ValueError(f"Extracted code column not found in {file_path}")
        if "question_id" not in df.columns:
            raise ValueError(f"Question ID column not found in {file_path}")

        total_samples = len(df)
        num_extract_fail = int(df["extracted_code"].isnull().sum())
        df = df.dropna().reset_index(drop=True)

        tests = []
        codes = []
        for _, row in df.iterrows():
            tests.append(self.test_suites[row["question_id"]])
            codes.append(row["extracted_code"])

        # In the eval code for GPT-OSS in MLPerf Inference v6.0, a ProcessPoolExecutor is used.
        # For now, we'll delegate the worker distribution to lcb_runner rather than handling it
        # ourselves.
        worker = _LCBWorker(n_lcb_workers=self.n_workers)
        graded = worker(tests, codes)
        pass_at_1 = sum([all(results) for results in graded]) / total_samples
        return pass_at_1, num_extract_fail
