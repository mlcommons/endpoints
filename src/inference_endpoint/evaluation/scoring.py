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


import inspect
import os
import subprocess
import sys
import tempfile
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import ClassVar

import numpy as np
import orjson
import pandas as pd
from tqdm import tqdm

try:
    import websocket
except ImportError:
    websocket = None

from ..dataset_manager.dataset import Dataset
from ..load_generator.events import SampleEvent
from .extractor import Extractor, PythonCodeExtractor


class Scorer(ABC):
    """Scorers will read in a dataset and outputs from a log and compute an accuracy score.
    An optional extractor can be provided to post-process the output to extract values that
    can be compared against the ground truth.
    """

    PREDEFINED: ClassVar[dict[str, type["Scorer"]]] = {}

    def __init_subclass__(
        cls,
        scorer_id: str | None = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)

        if not inspect.isabstract(cls):
            if scorer_id is None:
                scorer_id = cls.__name__
            cls.SCORER_ID = scorer_id
            Scorer.PREDEFINED[scorer_id] = cls

    @classmethod
    def get(cls, name: str) -> type["Scorer"]:
        """Look up an Scorer subclass by its registered name.

        Args:
            name: str, the registered scorer name

        Returns:
            Scorer subclass

        Raises:
            KeyError: If no scorer with the given name is found
        """
        try:
            return Scorer.PREDEFINED[name]
        except KeyError as e:
            raise KeyError(
                f"Scorer '{name}' is not registered - available scorers: {Scorer.available_scorers()}"
            ) from e

    @classmethod
    def available_scorers(cls) -> list[str]:
        """Return the list of registered scorer names."""
        return list(Scorer.PREDEFINED.keys())

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
        # If the dataset was transformed with a preset, we still treat it as the original
        # dataset name for the purposes of scoring
        if "::" in dataset_name:
            dataset_name = dataset_name.split("::")[0]
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


class PassAt1Scorer(Scorer, scorer_id="pass_at_1"):
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


class StringMatchScorer(Scorer, scorer_id="string_match"):
    """Implements exact string match scoring.
    The score is 1 if the output matches the ground truth exactly, 0 otherwise.
    This is useful for debugging and development.
    """

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        return 1.0 if value.strip() == ground_truth.strip() else 0.0


ExactMatchScorer = PassAt1Scorer


class RougeScorer(Scorer, scorer_id="rouge"):
    """Implements ROUGE scoring for text generation evaluation.
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures the overlap
    between generated text and reference text. Returns the ROUGE-L F1 score.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            import importlib.util as _importlib_util

            if (
                _importlib_util.find_spec("evaluate") is None
                or _importlib_util.find_spec("nltk") is None
                or _importlib_util.find_spec("rouge_score") is None
            ):
                raise ImportError

            import evaluate
            import nltk

            self.metric = evaluate.load("rouge")
            self.nltk = nltk

        except ImportError:
            raise ImportError(
                "nltk, evaluate, and rouge_score are required for ROUGE scoring. "
                "Install with: pip install nltk evaluate rouge_score"
            ) from None

    def postprocess_text(self, texts):
        texts = [text.strip() for text in texts]
        # rougeLSum expects newline after each sentence
        texts = ["\n".join(self.nltk.sent_tokenize(text)) for text in texts]
        return texts

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        # This method is not used
        raise RuntimeError(
            "ROUGE scoring requires batch processing for accurate aggregation. "
            "Call score() to compute metrics across the entire dataset instead of "
            "per-sample scoring."
        )

    def score(self) -> tuple[float, int]:
        df = self.get_outputs()

        # Outputs are for all samples, not just the target dataset
        valid_uuids = self.sample_index_map.keys()
        df = df[df["sample_uuid"].isin(valid_uuids)]

        # Match to sample index from dataset
        df = df.apply(self.match_sample_index, axis=1)

        empirical = df["output"].tolist()

        order = df["sample_index"].to_numpy().astype(int)
        assert (
            self.ground_truth_column in self.dataset.dataframe.columns
        ), f"Ground truth column {self.ground_truth_column} not found in dataset {self.dataset}"

        ground_truths = list(
            self.dataset.dataframe[self.ground_truth_column].to_numpy()[order]
        )

        empirical = self.postprocess_text(empirical)
        ground_truths = self.postprocess_text(ground_truths)

        result = self.metric.compute(
            predictions=empirical,
            references=ground_truths,
            use_stemmer=True,
            use_aggregator=False,
        )

        result = {k: f"{round(np.mean(v) * 100, 4)}" for k, v in result.items()}
        prediction_lens = [len(pred) for pred in empirical]
        gen_num = len(empirical)

        result = {
            **result,
            "gen_len": f"{np.sum(prediction_lens)}",
            "gen_num": gen_num,
        }

        # TODO: return only rouge1 for now to align with other scorers
        # Return the rest of the metrics later
        return result, 1


class LiveCodeBenchScorer(Scorer, scorer_id="code_bench_scorer"):
    """Scorer for LiveCodeBench code generation tasks.

    Uses the lcb_runner evaluation framework to execute generated code against test cases.
    Can connect to a containerized WebSocket evaluation service or fall back to subprocess.

    The scorer:
    1. Extracts Python code from model outputs (using PythonCodeExtractor)
    2. Attempts to use WebSocket service if lcb_websocket_port is provided
    3. Falls back to subprocess execution if WebSocket is unavailable
    4. Returns pass@1 score based on test results

    Args:
        dataset_name: Name of the dataset
        dataset: Dataset object containing problems
        report_dir: Directory containing evaluation logs
        extractor: Extractor class (defaults to PythonCodeExtractor)
        lcb_version: LiveCodeBench version tag (e.g., "release_v5", "release_v6")
        timeout: Timeout in seconds for each test execution
        question_id_column: Column name in dataset containing question IDs
        show_lcb_runner_output: Whether to show output during evaluation
        lcb_websocket_port: Port for WebSocket service on localhost (default: 13835)
                            Set to None to disable WebSocket and use subprocess only.
                            Why is the default port 13835? It's short for LCB WebSocket:
                            1=L, 3rd letter=C, 8=B, 3 rotated sideways=W, 5=S
    """

    def __init__(
        self,
        dataset_name: str,
        dataset: Dataset,
        report_dir: os.PathLike,
        extractor: type[Extractor] = PythonCodeExtractor,
        ground_truth_column: str | None = None,
        lcb_version: str = "release_v6",
        timeout: int = 60,
        question_id_column: str = "question_id",
        show_lcb_runner_output: bool = True,
        lcb_websocket_port: int | None = 13835,
    ):
        # Note: LiveCodeBench doesn't use ground_truth_column the same way
        # but we need to pass something to the parent
        assert (
            ground_truth_column is None
        ), "ground_truth_column should be None for LiveCodeBenchScorer"
        super().__init__(
            dataset_name=dataset_name,
            dataset=dataset,
            report_dir=report_dir,
            extractor=extractor,
            ground_truth_column=question_id_column,
        )

        self.lcb_version = lcb_version
        self.timeout = timeout
        self.question_id_column = question_id_column
        self.show_lcb_runner_output = show_lcb_runner_output

        # Construct WebSocket URL from port if provided
        self.lcb_websocket_url = (
            f"ws://localhost:{lcb_websocket_port}/evaluate"
            if lcb_websocket_port is not None
            else None
        )

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        raise RuntimeError(
            "This method should not be called. Use the score() method instead, which invokes lcb_runner."
        )

    def _evaluate_via_websocket(self, codes_dict: dict[str, list[str]]) -> dict | None:
        """Attempt to evaluate via WebSocket service (synchronous).

        Configured for long-running connections (minutes to hours) with:
        - Extended timeouts for send/receive operations
        - Automatic ping/pong for connection keep-alive
        - Proper error handling for network interruptions

        Returns:
            dict with evaluation results, or None if connection failed
        """
        if websocket is None:
            print(
                "Warning: websocket-client package not installed, falling back to subprocess"
            )
            print("Install with: pip install websocket-client")
            return None

        try:
            # Create WebSocket connection with settings for long-running operations
            # Timeout is set high for long evaluations (hours), but recv() will return
            # as soon as data is available (not blocking for the full timeout)
            ws = websocket.create_connection(
                self.lcb_websocket_url,
                timeout=7200,  # 2 hours connection timeout
                ping_interval=30,  # Send ping every 30 seconds to keep connection alive
                ping_timeout=10,  # Wait 10 seconds for pong response
            )

            # Setup progress tracking
            total_samples = sum(len(codes) for codes in codes_dict.values())
            pbar = None

            try:
                # Send evaluation request
                request = {
                    "codes_dict": codes_dict,
                    "timeout_sec": self.timeout,
                }
                ws.send(orjson.dumps(request).decode("utf-8"))

                print(f"Connected to WebSocket service: {self.lcb_websocket_url}")
                print(
                    f"Evaluating {len(codes_dict)} questions ({total_samples} samples)..."
                )
                pbar = tqdm(
                    total=total_samples,
                    desc="LCB Evaluation",
                    unit="sample",
                )

                # Process responses
                while True:
                    try:
                        message = ws.recv()
                        if not message:
                            # Connection closed cleanly
                            break

                        data = orjson.loads(message)
                        status = data.get("status")

                        if status == "started":
                            # Initial message, progress bar already initialized
                            pass

                        elif status == "progress":
                            completed = data.get("completed_samples", 0)
                            # Update progress bar to current position
                            pbar.n = completed
                            pbar.refresh()

                        elif status == "completed":
                            pbar.n = total_samples
                            pbar.refresh()
                            return data.get("result")

                        elif status == "error":
                            error_msg = data.get("error", "Unknown error")
                            print(f"WebSocket evaluation error: {error_msg}")
                            return None

                    except websocket.WebSocketTimeoutException:
                        # This shouldn't happen with ping/pong, but handle gracefully
                        print("WebSocket timeout - connection lost")
                        return None

                # If we exit the loop without returning, something went wrong
                return None

            finally:
                # Ensure progress bar is always closed
                if pbar:
                    pbar.close()

                # Close WebSocket connection
                try:
                    ws.close()
                except Exception:
                    pass  # Ignore errors on close

        except (ConnectionRefusedError, OSError, Exception) as e:
            print(f"WebSocket connection failed: {e}, falling back to subprocess")
            return None

    def _evaluate_via_subprocess(self, df: pd.DataFrame) -> float | None:
        """Evaluate via subprocess (fallback method).

        Returns:
            pass@1 score or None if evaluation failed
        """
        # Check if local evaluation is allowed via environment variable
        allow_local_eval = os.environ.get("ALLOW_LCB_LOCAL_EVAL", "").lower() in (
            "true",
            "1",
            "yes",
        )
        if not allow_local_eval:
            raise RuntimeError(
                "Local LiveCodeBench evaluation via subprocess is disabled by default for security reasons. "
                "To enable it, set the environment variable ALLOW_LCB_LOCAL_EVAL=true. "
                "This will allow execution of generated code on your local machine."
            )

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
                str(parquet_path),
                "--version-tag",
                self.lcb_version,
                "--datasets-dir",
                f"datasets/livecodebench/{self.lcb_version}",
                "--timeout",
                str(self.timeout),
            ]

            try:
                # Run subprocess with output both captured and displayed (tee-like behavior)
                # Note: We let stderr pass through directly for real-time progress bars/logs
                proc_stderr = (
                    None if self.show_lcb_runner_output else subprocess.DEVNULL
                )

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=proc_stderr,
                    text=True,
                    bufsize=1,  # Line buffered
                )

                # Collect stdout while displaying it character-by-character to support
                # progress bars that use carriage returns
                stdout_buffer = []
                while True:
                    char = process.stdout.read(1)
                    if not char:
                        break

                    if self.show_lcb_runner_output:
                        sys.stdout.write(char)
                        sys.stdout.flush()
                    stdout_buffer.append(char)

                # Wait for process to complete and check return code
                return_code = process.wait()
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, cmd)

                # Parse the JSON output from the captured stdout
                # Look for JSON at the end (after any progress bar output)
                stdout_text = "".join(stdout_buffer)
                # Try to find the last line that looks like JSON
                lines = stdout_text.strip().split("\n")
                for line in reversed(lines):
                    line = line.strip()
                    if line.startswith("{") and line.endswith("}"):
                        output = orjson.loads(line.encode("utf-8"))
                        return output["pass_at_1"]

                # No JSON found, try parsing the whole output
                output = orjson.loads(stdout_text.encode("utf-8"))
                return output["pass_at_1"]

            except (subprocess.CalledProcessError, orjson.JSONDecodeError, KeyError):
                # Return None if subprocess fails or JSON parsing fails
                return None

    def score(self) -> tuple[float | None, int]:
        """Score the dataset using parallel evaluation.

        Attempts WebSocket evaluation first if configured, falls back to subprocess.

        Returns:
            tuple[float | None, int]: The pass@1 score and the number of repeats.
            Returns None as the score if evaluation fails.
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

        # Extract code from outputs with default value for failed extractions
        # Use a comment that will fail all tests instead of None to maintain uniform list lengths
        df["extracted_code"] = df["output"].apply(
            lambda x: self.extractor.extract(x, default="# FAILED TO EXTRACT CODE")
        )

        n_repeats = len(df) // self.dataset.num_samples()

        # Try WebSocket evaluation first if URL is provided
        if self.lcb_websocket_url:
            # Group codes by question ID for WebSocket API
            codes_dict = defaultdict(list)
            for _, row in df.iterrows():
                codes_dict[row["question_id"]].append(row["extracted_code"])

            # Attempt WebSocket evaluation (synchronous)
            result = self._evaluate_via_websocket(codes_dict)

            if result is not None:
                # Successfully evaluated via WebSocket
                total_samples = result.get("total_samples", 0)
                per_problem_results = result.get("results", {})
                if not per_problem_results and total_samples:
                    print(
                        f"Server evaluated {total_samples} samples but returned an empty summary"
                    )
                    return None

                total_passed = sum(
                    sum(code_passed) for code_passed in per_problem_results.values()
                )
                pass_at_1 = total_passed / total_samples if total_samples > 0 else 0.0
                return pass_at_1, n_repeats

        # Fall back to subprocess evaluation
        if self.show_lcb_runner_output and self.lcb_websocket_url:
            print(
                "WebSocket evaluation unavailable, using subprocess evaluation method"
            )

        pass_at_1 = self._evaluate_via_subprocess(df)
        return pass_at_1, n_repeats
