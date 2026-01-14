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
# See the License for the specific permissions and
# limitations under the License.


import os
import subprocess
import sys
import tempfile
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import orjson
import pandas as pd

from ..dataset_manager.dataset import Dataset
from ..load_generator.events import SampleEvent
from .extractor import Extractor, PythonCodeExtractor


class Scorer(ABC):
    """Scorers will read in a dataset and outputs from a log and compute an accuracy score.
    An optional extractor can be provided to post-process the output to extract values that
    can be compared against the ground truth.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset: Dataset,
        report_dir: os.PathLike,
        extractor: type[Extractor] | None = None,
        ground_truth_column: str = "ground_truth",
    ):
        self.dataset = dataset
        self.report_dir = Path(report_dir)
        self.extractor = extractor
        self.dataset_name = dataset_name
        self.ground_truth_column = (
            ground_truth_column if ground_truth_column is not None else "ground_truth"
        )
        self.sample_index_map = self._load_sample_index_map()

    def _load_sample_index_map(self):
        sample_index_map_path = self.report_dir / "sample_idx_map.json"
        if not sample_index_map_path.exists():
            raise FileNotFoundError(
                f"Sample index map file not found at {sample_index_map_path}"
            )

        with sample_index_map_path.open("r") as f:
            d = orjson.loads(f.read())
            return d[self.dataset_name]  # Implicitly raises KeyError

    def get_outputs(self):
        # TODO: Currently, the outputs are only saved in the events.jsonl file, which is quite
        # large, and only saved optionally. Later, we should move to saving the outputs in a
        # separate file for easier compute.
        events_log_path = self.report_dir / "events.jsonl"
        if not events_log_path.exists():
            raise FileNotFoundError(f"Events log file not found at {events_log_path}")

        outputs = []
        with events_log_path.open("r") as f:
            for line in f:
                event = orjson.loads(line.strip())
                if event["event_type"] == SampleEvent.COMPLETE.value:
                    outputs.append(event)
        df = pd.DataFrame(outputs, columns=["sample_uuid", "value"])
        df.rename(columns={"value": "output"}, inplace=True)
        return df

    def match_sample_index(self, row: pd.Series) -> pd.Series:
        # Pandas Apply function to create a new 'sample_index' column
        row["sample_index"] = self.sample_index_map[row["sample_uuid"]]
        return row

    @abstractmethod
    def score_single_sample(self, value: str, ground_truth: str) -> float:
        raise NotImplementedError

    def score(self) -> tuple[float, int]:
        """Scores the dataset and returns the mean score and the number of repeats.

        Returns:
            tuple[float, int]: The mean score and the number of repeats.
        """
        df = self.get_outputs()

        # Outputs are for all samples, not just the target dataset
        valid_uuids = self.sample_index_map.keys()
        df = df[df["sample_uuid"].isin(valid_uuids)]

        # Match to sample index from dataset
        df = df.apply(self.match_sample_index, axis=1)

        empirical = df["output"]
        if self.extractor is not None:
            empirical = empirical.apply(self.extractor.extract)
        empirical = empirical.to_numpy()

        # Get ground truths
        order = df["sample_index"].to_numpy()
        assert (
            self.ground_truth_column in self.dataset.dataframe.columns
        ), f"Ground truth column {self.ground_truth_column} not found in dataset {self.dataset}"
        ground_truths = self.dataset.dataframe[self.ground_truth_column].to_numpy()[
            order
        ]

        scores = []
        for i in range(len(empirical)):
            scores.append(self.score_single_sample(empirical[i], ground_truths[i]))

        n_repeats = len(scores) // self.dataset.num_samples()
        return np.mean(scores), n_repeats


class PassAt1Scorer(Scorer):
    """Implements pass@1 scoring as defined by Artificial Analysis.
    pass@1 means the model gets exactly one attempt to produce the correct answer.
    The score is 1 if the output matches the ground truth exactly, 0 otherwise.
    This is the standard scoring method for multiple-choice questions and other
    tasks where there is a single correct answer.
    Reference: https://artificialanalysis.ai/methodology/intelligence-benchmarking

    This is equivalent to Exact Match Scoring.
    """

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        return 1.0 if value == ground_truth else 0.0


ExactMatchScorer = PassAt1Scorer


class LiveCodeBenchScorer(Scorer):
    """Scorer for LiveCodeBench code generation tasks.

    Uses the lcb_runner evaluation framework to execute generated code against test cases.
    Requires lcb_runner to be installed (pip install from LiveCodeBench).

    The scorer:
    1. Extracts Python code from model outputs (using PythonCodeExtractor)
    2. Runs code execution tests using lcb_runner
    3. Returns 1.0 if all tests pass, 0.0 otherwise

    Args:
        dataset_name: Name of the dataset
        dataset: Dataset object containing problems
        report_dir: Directory containing evaluation logs
        extractor: Extractor class (defaults to PythonCodeExtractor)
        lcb_version: LiveCodeBench version tag (e.g., "release_v5", "release_v6")
        num_workers: Number of parallel workers for code evaluation
        timeout: Timeout in seconds for each test execution
        question_id_column: Column name in dataset containing question IDs
    """

    def __init__(
        self,
        dataset_name: str,
        dataset: Dataset,
        report_dir: os.PathLike,
        extractor: type[Extractor] = PythonCodeExtractor,
        lcb_version: str = "release_v6",
        timeout: int = 60,
        question_id_column: str = "question_id",
        lcb_root: Path = Path("/opt/LiveCodeBench"),
    ):
        # Note: LiveCodeBench doesn't use ground_truth_column the same way
        # but we need to pass something to the parent
        super().__init__(
            dataset_name=dataset_name,
            dataset=dataset,
            report_dir=report_dir,
            extractor=extractor,
            ground_truth_column=question_id_column,
        )

        self.lcb_root = Path(lcb_root)
        if not self.lcb_root.exists():
            raise FileNotFoundError(
                f"LiveCodeBench root directory {lcb_root} does not exist"
            )

        self.lcb_version = lcb_version
        self.timeout = timeout
        self.question_id_column = question_id_column

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        raise RuntimeError(
            "This method should not be called. Use the score() method instead, which invokes lcb_runner."
        )

    def score(self) -> tuple[float, int]:
        """Score the dataset using parallel evaluation.

        This overrides the base class method to use parallel evaluation
        for better performance with code execution tests.

        Returns:
            tuple[float | None, int]: The mean score and the number of repeats. If an error occurs during scoring,
            returns None as the score.
        """
        df = self.get_outputs()

        # Outputs are for all samples, not just the target dataset
        valid_uuids = self.sample_index_map.keys()
        df = df[df["sample_uuid"].isin(valid_uuids)]

        # Match to sample index from dataset
        df = df.apply(self.match_sample_index, axis=1)

        # Get question IDs
        def get_question_id(sample_index: int) -> str:
            return self.dataset.dataframe.iloc[sample_index][self.question_id_column]

        df["question_id"] = df["sample_index"].apply(get_question_id)

        # Extract code from outputs
        df["extracted_code"] = df["output"].apply(self.extractor.extract)

        n_repeats = len(df) // self.dataset.num_samples()

        # TODO: Currently runs as a subprocess. In the future, we need to migrate
        # LCBServe to be running as a background service listening on a port or socket
        # so that it can be containerized for better security and isolation.
        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_name = f"{uuid.uuid4()}.parquet"
            parquet_path = Path(temp_dir) / parquet_name
            df.to_parquet(parquet_path)

            # Invoke lcb_serve.py as a subprocess to avoid importing LiveCodeBench dependencies
            # in the main inference endpoint environment, and also because LCB eval will
            # attempt to sandbox Python code execution by setting a bunch of core standard library
            # methods to None (i.e. most things in the os, sys, and other such modules), which would
            # impact the rest of the current Python process.
            cmd = [
                sys.executable,
                "-m",
                "inference_endpoint.dataset_manager.predefined.livecodebench.lcb_serve",
                parquet_name,
                "--version-tag",
                self.lcb_version,
                "--output-file-store",
                str(temp_dir),
                "--lcb-root",
                str(self.lcb_root),
                "--timeout",
                str(self.timeout),
            ]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                output = orjson.loads(result.stdout.encode("utf-8"))
                pass_at_1 = output["pass_at_1"]
            except (subprocess.CalledProcessError, orjson.JSONDecodeError, KeyError):
                # Return None if subprocess fails or JSON parsing fails
                pass_at_1 = None

        return pass_at_1, n_repeats
