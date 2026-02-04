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
import json
import logging
import multiprocessing as mp
import traceback
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .generate import generate_dataset

logger = logging.getLogger(__name__)


def execute_code_single(test_suite_json: str, code: str, timeout_sec: int = 60):
    # Run code with lcb_runner. Note that the lcb_runner has a very rudimentary sandbox
    # which is extremely easy to bypass, and as such it is recommended to run this both
    # in an unprivileged container and in a separate process.
    import numpy as np

    from .run_lcb_tests import run_test

    res, metadata = run_test(
        {"input_output": test_suite_json}, test=code, timeout=timeout_sec
    )

    # LCB results are expected to be plain booleans or error codes.
    # Reference: https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/evaluation/compute_code_generation_metrics.py#L72-L79
    fixed_types = []
    for elem in res:
        if isinstance(elem, np.ndarray):
            elem = elem.item(0)
        if isinstance(elem, np.bool_):
            elem = bool(elem)
        fixed_types.append(elem)
    return fixed_types, metadata


def execute_code_single_suppressed_errors(
    *args, resp_buffer: list | None = None, **kwargs
):
    """Wrapper around execute code so that all errors are resurfaced as failed tests"""
    try:
        res, metadata = execute_code_single(*args, **kwargs)
        if not isinstance(res, list):
            raise ValueError(f"Expected boolean result, got {type(res)}")

        if not isinstance(metadata, dict):
            raise ValueError(f"Expected metadata to be a dict, got {type(metadata)}")
    except Exception:
        # Magic number (see https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/evaluation/compute_code_generation_metrics.py#L65)
        res = [-2]  # LCB internal error code for test runner failed test cases
        metadata = {
            "error": traceback.format_exc(),
            "error_code": -5,
            "error_message": "TestRunnerError",
        }

    if resp_buffer is not None:
        resp_buffer.append((res, metadata))
    return res, metadata


def run_code_subprocess(
    test_suite_json: str,
    code: str,
    timeout_sec: int = 60,
):
    # Compute global timeout -
    # https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/evaluation/compute_code_generation_metrics.py#L43

    suite = json.loads(test_suite_json)
    per_test_case_padding = 1
    flat_timeout_extension = 5
    global_timeout = (timeout_sec + per_test_case_padding) * len(
        suite["inputs"]
    ) + flat_timeout_extension

    manager = mp.Manager()
    resp_buffer = manager.list()
    p = mp.Process(
        target=execute_code_single_suppressed_errors,
        args=(
            test_suite_json,
            code,
        ),
        kwargs={
            "resp_buffer": resp_buffer,
            "timeout_sec": timeout_sec,
        },
    )
    p.start()
    p.join(timeout=global_timeout)

    if p.is_alive():
        p.kill()

    if len(resp_buffer) == 0:
        # Assume timeout
        res = [-1] * len(suite["inputs"])
        metadata = {
            "error": "Test suite timeout",
            "error_code": -1,
            "error_message": f"Subprocess did not complete in time ({global_timeout}s)",
        }
        return res, metadata
    else:
        res, metadata = resp_buffer[0]
        return res, metadata


class LCBTestLoader:
    """Handles loading and preloading of test cases for LiveCodeBench.

    This class's operations are not thread-safe, and should only be used in a single-threaded context.
    """

    def __init__(
        self, datasets_dir: Path, cache_limit: int | None = None, strict: bool = True
    ):
        """
        Initializes the test case loader.

        Args:
            datasets_dir: The directory containing the datasets.
            cache_limit: The maximum number of test cases to cache. If None, no caching is done.
            strict: Whether to raise an error if a test case is not found. If False, an empty test case is returned.
        """
        self.datasets_dir = datasets_dir
        self.test_cases_dir = datasets_dir / "test_cases"
        self.strict = strict

        if not self.test_cases_dir.exists():
            raise FileNotFoundError(
                f"Test cases directory not found: {self.test_cases_dir}. "
                "Please regenerate the dataset with test cases enabled."
            )

        self.load_test_case = lru_cache(maxsize=cache_limit)(self._load_test_case)

        # Surface cache_info method for convenience
        self.cache_info = self.load_test_case.cache_info

    def _load_test_case(self, question_id: str) -> str:
        """Loads a test case for a given question ID.

        Args:
            question_id: The question ID to load the test case for.

        Returns:
            The test case as a JSON string to be consumed by the test runner.
        """
        test_case_path = self.test_cases_dir / f"{question_id}.json"
        if not test_case_path.exists():
            if self.strict:
                raise FileNotFoundError(
                    f"Test case file not found: {test_case_path}. "
                    "Please regenerate the dataset."
                )
            else:
                return json.dumps(
                    {
                        "inputs": [],
                        "outputs": [],
                        "fn_name": None,
                    }
                )

        with test_case_path.open(encoding="utf-8", mode="r") as f:
            test_case_data = json.load(f)

        test_inputs = []
        test_outputs = []
        for test_case in (
            test_case_data["public_test_cases"] + test_case_data["private_test_cases"]
        ):
            test_inputs.append(test_case["input"])
            test_outputs.append(test_case["output"])

        info = {
            "inputs": test_inputs,
            "outputs": test_outputs,
            "fn_name": test_case_data["func_name"],
        }
        return json.dumps(info)

    def __getitem__(self, question_id: str) -> str:
        """Gets a test case for a given question ID.

        Args:
            question_id: The question ID to get the test case for.

        Returns:
            The test case as a JSON string to be consumed by the test runner.
        """
        return self.load_test_case(question_id)


class _LCBWorker:
    def __init__(
        self,
        test_loader: LCBTestLoader,
        n_lcb_workers: int = 1,
        worker_timeout_sec: int = 60,
    ):
        self.test_loader = test_loader
        self.n_lcb_workers = n_lcb_workers
        self.worker_timeout_sec = worker_timeout_sec

    def __call__(
        self,
        question_ids: list[str],
        codes: list[list[str]],
        on_problem_complete: Callable[[list[str]], None] | None = None,
    ) -> dict[str, list[bool]]:
        """Evaluates LiveCodeBench problems given question IDs and their corresponding code samples.

        Args:
            question_ids: List of question IDs to evaluate.
            codes: List of lists of code samples to evaluate.
            on_problem_complete: Callback function to call when a single test case completes. The argument is the
                list of question IDs that completed.

        Returns:
            dict[str, list[bool]]: Dictionary mapping question IDs to lists of boolean results for each code sample.
        """
        # Create results dict with the expected size
        results: dict[str, list[bool]] = {}
        for qid, test_codes in zip(question_ids, codes, strict=False):
            results[qid] = [False] * len(test_codes)
        futures = {}

        with ProcessPoolExecutor(max_workers=self.n_lcb_workers) as executor:
            for qid, test_codes in zip(question_ids, codes, strict=False):
                test_suite_json = self.test_loader[qid]
                for i, code in enumerate(test_codes):
                    future = executor.submit(
                        run_code_subprocess,
                        test_suite_json,
                        code,
                        timeout_sec=self.worker_timeout_sec,
                    )

                    # Note that futures are hashable for bookkeeping purposes.
                    futures[future] = (qid, i)

            # Gather results as they complete
            for future in as_completed(futures):
                qid, code_idx = futures[future]
                res, metadata = future.result()
                if "error" in metadata:
                    logger.warning(
                        f"Test execution error for question {qid}: {metadata}"
                    )

                # LCB uses any result > 0 as a 'pass' since:
                # Negative numbers indicate error codes
                # 0 is False casted to an int
                # 1 is True casted to an int
                # For a generalized pass@k score, it should be grouped by question_id
                # `res` is a list of results for each test case in the problem, and for
                # a given code sample to be considered passing, all test cases must pass.
                results[qid][code_idx] = all(case_result > 0 for case_result in res)

                # For now we discard metadata since we only care about overall score.
                # In the future, for debugging purposes, we can log the metadata to see
                # reasons for failures if results are not as expected.
                if on_problem_complete is not None:
                    # Callback should not impede or interrupt execution
                    try:
                        on_problem_complete([qid])
                    except Exception as e:
                        logger.error(
                            "Error occurred during on_problem_complete callback: %r",
                            e,
                            exc_info=True,
                        )

        return results


class LCBServe:
    def __init__(
        self,
        version_tag: str = "release_v6",
        n_workers: int | None = None,
        datasets_dir: Path = Path("/opt/LiveCodeBench_Datasets"),
        auto_generate_dataset: bool = True,
        test_suite_cache_limit: int | None = None,
        preload_test_cases: bool = False,
    ):
        """
        Initializes an LCBServe instance to evaluate a set of LiveCodeBench problems.

        Args:
            version_tag: The version tag of the LiveCodeBench dataset to use.
            n_workers: The number of workers to use for evaluation. If None, the number of workers is set to the number of CPU cores divided by 2.
            datasets_dir: The directory containing the LiveCodeBench datasets.
            auto_generate_dataset: Whether to generate the dataset if it does not exist.
            test_suite_cache_limit: The maximum number of test suites to cache. If None, the cache will not have a maximum size.
            preload_test_cases: If the cache limit is not None, this does nothing. Otherwise, if True, all question IDs will be preloaded into
                the test case cache on initialization.
        """
        self.version_tag = version_tag
        self.datasets_dir = Path(datasets_dir)

        if n_workers is None:
            n_workers = mp.cpu_count() // 2
        logger.info("Using %d workers for LCB eval", n_workers)
        self.n_workers = n_workers

        self.path_to_dataset = (
            self.datasets_dir / f"livecodebench_{version_tag}.parquet"
        )

        self.df = None
        if not self.path_to_dataset.exists():
            if not auto_generate_dataset:
                raise FileNotFoundError(
                    f"Dataset file {self.path_to_dataset} does not exist"
                )
            else:
                logger.info("Generating dataset... This may take a while...")
                self.df = generate_dataset(
                    datasets_dir=self.datasets_dir,
                    variant=version_tag,
                )

        # Load the dataset - All test cases should already be extracted
        if self.df is None:
            logger.info("Reading parquet from %s", self.path_to_dataset)
            self.df = pd.read_parquet(self.path_to_dataset)
            logger.info("Loaded %d records", len(self.df))

        self.test_loader = LCBTestLoader(
            self.datasets_dir, cache_limit=test_suite_cache_limit
        )
        if preload_test_cases and test_suite_cache_limit is None:
            for qid in self.df["question_id"].values:
                self.test_loader.load_test_case(
                    qid
                )  # Accessing will populate the cache

    def cache_info(self):
        """Returns the cache information for the test loader."""
        return self.test_loader.cache_info()

    def evaluate(
        self,
        codes_dict: dict[str, list[str]],
        timeout_sec: int = 60,
        on_problem_complete: Callable[[list[str]], None] | None = None,
    ) -> dict[str, list[bool]]:
        """Evaluates LiveCodeBench problems given question IDs and their corresponding code samples.

        Args:
            codes_dict: Dictionary mapping question IDs to lists of code samples.
            timeout_sec: Timeout in seconds for each worker to use for each test case. If a test case does
                not complete within this timeout, it is treated as a test fail. (Default: 60)
            on_problem_complete: Callback function to call when a single test case completes. The argument is the
                list of question IDs that completed.

        Returns:
            dict[str, list[bool]]: Dictionary mapping question IDs to lists of boolean results for each code sample.

        Raises:
            KeyError: If any question_id is not found in the loaded test suites.
        """
        # Validate all question IDs exist in test suites
        assert self.df is not None, "Dataset not loaded"
        invalid_ids = [
            qid for qid in codes_dict.keys() if qid not in self.df["question_id"].values
        ]
        if invalid_ids:
            raise KeyError(
                f"Question IDs not found in test suites: {invalid_ids[:10]}"
                + (
                    f" and {len(invalid_ids) - 10} more"
                    if len(invalid_ids) > 10
                    else ""
                )
            )

        # lcb_runner does a sort() keyed by question_id. I'm not sure if this is 100% necessary,
        # but I'll do it anyway for consistency.
        qids = sorted(codes_dict.keys())
        codes = [codes_dict[qid] for qid in qids]

        worker = _LCBWorker(
            self.test_loader,
            n_lcb_workers=self.n_workers,
            worker_timeout_sec=timeout_sec,
        )
        return worker(qids, codes, on_problem_complete=on_problem_complete)

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        timeout_sec: int = 60,
        on_problem_complete: Callable[[list[str]], None] | None = None,
    ) -> dict[str, int | float]:
        """Evaluates all LiveCodeBench problems in a parquet file and returns a dictionary in the form:
        {
            "total_samples": int,
            "passed_samples": int,
            "pass_at_1": float,
        }

        Args:
            df: DataFrame containing the extracted codes and question IDs to evaluate.
            timeout_sec: Timeout in seconds for each worker to use for each test case. If a test case does
                not complete within this timeout, it is treated as a test fail. (Default: 60)
            on_problem_complete: Callback function to call when a single test case completes. The argument is the
                list of question IDs that completed.

        Returns:
            dict[str, int | float]: See description above.

        Raises:
            FileNotFoundError: If the parquet file does not exist.
            ValueError: If the extracted code column or question ID column is not found in the parquet file.
        """
        # Replace null extracted codes with empty strings (will fail evaluation naturally)
        df["extracted_code"] = df["extracted_code"].fillna("")

        # Group codes by question ID
        codes_dict = defaultdict(list)
        for _, row in df.iterrows():
            codes_dict[row["question_id"]].append(row["extracted_code"])

        # Evaluate and get results dictionary
        results_dict = self.evaluate(
            codes_dict=codes_dict,
            timeout_sec=timeout_sec,
            on_problem_complete=on_problem_complete,
        )

        # Count number of passed samples. Note values are lists of booleans, so we can sum them directly.
        num_passed = sum(sum(results) for results in results_dict.values())

        # Calculate pass@1
        total_samples = len(df)
        pass_at_1 = num_passed / total_samples if total_samples > 0 else 0.0

        return {
            "total_samples": total_samples,
            "passed_samples": num_passed,
            "pass_at_1": pass_at_1,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LiveCodeBench parquet file and output results as JSON"
    )
    parser.add_argument(
        "parquet_file",
        type=str,
        help="Path to the parquet file",
    )
    parser.add_argument(
        "--version-tag",
        type=str,
        default="release_v6",
        help="LiveCodeBench version tag (default: release_v6)",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("/opt/LiveCodeBench_Datasets"),
        help="Directory where datasets are stored (default: /opt/LiveCodeBench_Datasets)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for each test case (default: 60)",
    )

    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_file)
    if "extracted_code" not in df.columns:
        raise ValueError(f"Extracted code column not found in {args.parquet_file}")
    if "question_id" not in df.columns:
        raise ValueError(f"Question ID column not found in {args.parquet_file}")

    lcb_serve = LCBServe(
        version_tag=args.version_tag,
        datasets_dir=args.datasets_dir,
    )

    with tqdm(total=len(df)) as pbar:
        results = lcb_serve.evaluate_dataframe(
            df=df,
            timeout_sec=args.timeout,
            on_problem_complete=lambda x: pbar.update(len(x)),
        )

    print(json.dumps(results))
