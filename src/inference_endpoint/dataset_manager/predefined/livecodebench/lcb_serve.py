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
import multiprocessing as mp
import os
import traceback
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .generate import generate_dataset


@contextmanager
def chdir(path: Path):
    """Context manager to change the current working directory to the given path."""
    old_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_dir)


def execute_code_single(test_suite_json: str, code: str, timeout_sec: int = 60):
    # Run code with lcb_runner. Note that the lcb_runner has a very rudimentary sandbox
    # which is extremely easy to bypass, and as such it is recommended to run this both
    # in an unprivileged container and in a separate process.
    import numpy as np
    from lcb_runner.evaluation.testing_util import run_test

    res, metadata = run_test(
        {"input_output": test_suite_json}, test=code, debug=False, timeout=timeout_sec
    )

    # LCB results are expected to be plain booleans or error codes.
    # Reference: https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/evaluation/compute_code_generation_metrics.py#L72-L79
    fixed = []
    for elem in res:
        if isinstance(elem, np.ndarray):
            elem = elem.item(0)
        if isinstance(elem, np.bool_):
            elem = bool(elem)
        fixed.append(elem)
    res = fixed
    return res, metadata


def execute_code_single_suppressed_errors(*args, resp_buffer: list = None, **kwargs):
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
    else:
        return res, metadata


def run_code_subprocess(
    test_suite_json: str,
    code: str,
    timeout_sec: int = 60,
    lcb_root: Path = Path("/opt/LiveCodeBench"),
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
    with chdir(lcb_root):
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


class _LCBWorker:
    def __init__(
        self,
        test_suites: dict[int, str],
        lcb_root: Path = Path("/opt/LiveCodeBench"),
        n_lcb_workers: int = 1,
        worker_timeout_sec: int = 60,
    ):
        if not lcb_root.exists():
            raise FileNotFoundError(
                f"LiveCodeBench root directory {lcb_root} does not exist"
            )

        self.test_suites = test_suites
        self.lcb_root = lcb_root
        self.n_lcb_workers = n_lcb_workers
        self.worker_timeout_sec = worker_timeout_sec

    def __call__(
        self,
        question_ids: list[int],
        codes: list[list[str]],
        on_problem_complete: Callable[[list[int]], None] | None = None,
    ) -> dict[int, list[bool]]:
        """Evaluates LiveCodeBench problems given question IDs and their corresponding code samples.

        Args:
            question_ids: List of question IDs to evaluate.
            codes: List of lists of code samples to evaluate.
            on_problem_complete: Callback function to call when a single test case completes. The argument is the
                list of question IDs that completed.

        Returns:
            dict[int, list[bool]]: Dictionary mapping question IDs to lists of boolean results for each code sample.
        """
        # Create results dict with the expected size
        results = {}
        for qid, test_codes in zip(question_ids, codes, strict=False):
            results[qid] = [None] * len(test_codes)
        futures = {}

        with ProcessPoolExecutor(max_workers=self.n_lcb_workers) as executor:
            for qid, test_codes in zip(question_ids, codes, strict=False):
                test_suite = self.test_suites[qid]
                for i, code in enumerate(test_codes):
                    future = executor.submit(
                        run_code_subprocess,
                        test_suite,
                        code,
                        timeout_sec=self.worker_timeout_sec,
                        lcb_root=self.lcb_root,
                    )

                    # Note that futures are hashable for bookkeeping purposes.
                    futures[future] = (qid, i)

            # Gather results as they complete
            for future in as_completed(futures):
                qid, code_idx = futures[future]
                res, _ = future.result()

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
                        print(
                            f"Error occurred during on_problem_complete callback: {e!r}"
                        )
                        traceback.print_exc()

        return results


class LCBServe:
    def __init__(
        self,
        version_tag: str = "release_v6",
        use_lite: bool = True,
        n_workers: int | None = None,
        datasets_dir: Path = Path("/mnt/datasets"),
        lcb_root: Path = Path("/opt/LiveCodeBench"),
        auto_generate_dataset: bool = True,
    ):
        self.version_tag = version_tag
        self.use_lite = use_lite

        if n_workers is None:
            n_workers = mp.cpu_count() // 2
        print(f"Using {n_workers} workers for LCB eval...")
        self.n_workers = n_workers

        self.path_to_dataset = (
            Path(datasets_dir) / f"livecodebench_{version_tag}.parquet"
        )
        self.lcb_root = Path(lcb_root)

        self.df = None
        if not self.path_to_dataset.exists():
            if not auto_generate_dataset:
                raise FileNotFoundError(
                    f"Dataset file {self.path_to_dataset} does not exist"
                )

            self.df = generate_dataset(
                datasets_dir=datasets_dir,
                variant=version_tag,
            )

        if not self.lcb_root.exists():
            raise FileNotFoundError(
                f"LiveCodeBench root directory {self.lcb_root} does not exist"
            )

        # Load the dataset - All test cases should already be extracted
        if self.df is None:
            print("Reading parquet...")
            self.df = pd.read_parquet(self.path_to_dataset)
            print(f"Loaded {len(self.df)}")
        self.test_suites = self.consolidate_test_cases()

    def consolidate_test_cases(self) -> dict[int, str]:
        # Consolidate the inputs and outputs of the test cases into a JSON string.
        # Reference: https://github.com/LiveCodeBench/LiveCodeBench/blob/28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24/lcb_runner/benchmarks/code_generation.py#L106
        test_suites = {}
        for _, row in self.df.iterrows():
            q_id = row["question_id"]

            # Decode test cases
            public_test_cases = json.loads(row["public_test_cases"])
            private_test_cases = json.loads(row["private_test_cases"])

            info = {
                "inputs": [],
                "outputs": [],
                "fn_name": row["func_name"] if row["func_name"] else None,
            }
            for test_case in public_test_cases + private_test_cases:
                info["inputs"].append(test_case["input"])
                info["outputs"].append(test_case["output"])

            test_suites[q_id] = json.dumps(info)
        return test_suites

    def evaluate(
        self,
        codes_dict: dict[int, list[str]],
        timeout_sec: int = 60,
        on_problem_complete: Callable[[list[int]], None] | None = None,
    ) -> dict[int, list[bool]]:
        """Evaluates LiveCodeBench problems given question IDs and their corresponding code samples.

        Args:
            codes_dict: Dictionary mapping question IDs to lists of code samples.
            timeout_sec: Timeout in seconds for each worker to use for each test case. If a test case does
                not complete within this timeout, it is treated as a test fail. (Default: 60)
            on_problem_complete: Callback function to call when a single test case completes. The argument is the
                list of question IDs that completed.

        Returns:
            dict[int, list[bool]]: Dictionary mapping question IDs to lists of boolean results for each code sample.

        Raises:
            KeyError: If any question_id is not found in the loaded test suites.
        """
        # Validate all question IDs exist in test suites
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
            self.test_suites,
            lcb_root=self.lcb_root,
            n_lcb_workers=self.n_workers,
            worker_timeout_sec=timeout_sec,
        )
        return worker(qids, codes, on_problem_complete=on_problem_complete)

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        timeout_sec: int = 60,
        on_problem_complete: Callable[[list[int]], None] | None = None,
    ) -> dict[str, int | float]:
        """Evaluates all LiveCodeBench problems in a parquet file and returns a dictionary in the form:
        {
            "total_samples": int,
            "passed_samples": int,
            "pass_at_1": float,
            "num_extract_fail": int,
        }

        Args:
            df: DataFrame containing the extracted codes and question IDs to evaluate.
            timeout_sec: Timeout in seconds for each worker to use for each test case. If a test case does
                not complete within this timeout, it is treated as a test fail. (Default: 60)
            on_problem_complete: Callback function to call when a single test case completes. The argument is the
                list of question IDs that completed.

        Returns:
            dict[str, int | float]: Dictionary in the form:
            {
                "total_samples": int,
                "passed_samples": int,
                "pass_at_1": float,
                "num_extract_fail": int,
            }

        Raises:
            FileNotFoundError: If the parquet file does not exist.
            ValueError: If the extracted code column or question ID column is not found in the parquet file.
        """
        # Count extraction failures before dropping
        failed_extract = df[df["extracted_code"].isnull()]
        num_extract_fail = len(failed_extract)

        # Immediately signal failed extracts as complete
        if on_problem_complete is not None:
            qid_list = failed_extract["question_id"].tolist()
            on_problem_complete(qid_list)

        # Drop failed extractions from samples to be passed to evaluate()
        df = df.dropna().reset_index(drop=True)

        # Group codes by question ID
        codes_dict = defaultdict(list)
        for _, row in df.iterrows():
            codes_dict[row["question_id"]].append(row["extracted_code"])

        # Evaluate and get number of passed samples
        num_passed = self.evaluate(
            codes_dict=codes_dict,
            timeout_sec=timeout_sec,
            on_problem_complete=on_problem_complete,
        )

        # Calculate pass@1: total samples includes extraction failures
        total_samples = (
            sum(len(code_list) for code_list in codes_dict.values()) + num_extract_fail
        )
        pass_at_1 = num_passed / total_samples

        return {
            "total_samples": total_samples,
            "passed_samples": num_passed,
            "pass_at_1": pass_at_1,
            "num_extract_fail": num_extract_fail,
        }


if __name__ == "__main__":
    import json

    from tqdm import tqdm

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
        default=Path("/mnt/datasets"),
        help="Directory where datasets are stored (default: /mnt/datasets)",
    )
    parser.add_argument(
        "--lcb-root",
        type=Path,
        default=Path("/opt/LiveCodeBench"),
        help="Path to LiveCodeBench root directory (default: /opt/LiveCodeBench)",
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
        lcb_root=args.lcb_root,
    )

    with tqdm(total=len(df)) as pbar:
        results = lcb_serve.evaluate_dataframe(
            df=df,
            timeout_sec=args.timeout,
            on_problem_complete=lambda x: pbar.update(len(x)),
        )

    print(json.dumps(results))
