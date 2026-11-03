# SPDX-FileCopyrightText: Copyright (c) 2026 CoreWeave, Inc. All rights reserved.
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

"""Tests for the LCB local eval error handling in eval_accuracy.py.

evaluate_livecodebench_worker re-raises ImportError/FileNotFoundError/
RuntimeError (broken judge env) and keeps returning False for anything
else (bad generated code). Generated code runs inside the LCB grader
subprocesses, not in the worker, so a bad import in a sample shows up
as a normal grading failure, not as an ImportError here.

The executor is replaced with an in-process fake so monkeypatching
works regardless of the multiprocessing start method.
"""

import pandas as pd
import pytest
from inference_endpoint.evaluation.legacy_mlperf_deepseek_r1.mlperf_eval import (
    eval_accuracy,
)


class ImmediateFuture:
    def __init__(self, fn, *args):
        self._exc = None
        self._value = None
        try:
            self._value = fn(*args)
        except BaseException as exc:
            self._exc = exc

    def result(self, timeout=None):
        if self._exc is not None:
            raise self._exc
        return self._value


class ImmediateExecutor:
    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, fn, *args):
        return ImmediateFuture(fn, *args)


@pytest.fixture
def inline_pool(monkeypatch):
    monkeypatch.setattr(eval_accuracy, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(
        eval_accuracy, "as_completed", lambda futures, timeout=None: list(futures)
    )


def _lcb_df(n):
    return pd.DataFrame(
        {
            "dataset": ["livecodebench"] * n,
            "extracted_answer": [f"code_{i}" for i in range(n)],
            "ground_truth": [f"qid_{i}" for i in range(n)],
            "prompt_accuracy": [0.0] * n,
        }
    )


class TestWorker:
    def test_sample_level_failure_scores_incorrect(self, monkeypatch):
        # Pre-existing behavior, unchanged by the fix.
        def boom(code, question_id):
            raise ValueError("bad generated code")

        monkeypatch.setattr(eval_accuracy, "evaluate_livecodebench", boom)
        assert eval_accuracy.evaluate_livecodebench_worker(("code", "qid")) == (
            "qid",
            False,
        )

    @pytest.mark.parametrize("exc_type", [ImportError, FileNotFoundError, RuntimeError])
    def test_infra_failure_reraises(self, monkeypatch, exc_type):
        def boom(code, question_id):
            raise exc_type("judge is broken")

        monkeypatch.setattr(eval_accuracy, "evaluate_livecodebench", boom)
        with pytest.raises(exc_type):
            eval_accuracy.evaluate_livecodebench_worker(("code", "qid"))


class TestProcessLivecodebenchParallel:
    def test_one_bad_sample_does_not_abort_run(self, monkeypatch, inline_pool):
        monkeypatch.setattr(eval_accuracy, "load_lcb_benchmark", lambda: {})

        def grade(code, question_id):
            if question_id == "qid_1":
                raise ValueError("bad generated code")
            return True

        monkeypatch.setattr(eval_accuracy, "evaluate_livecodebench", grade)

        df = _lcb_df(3)
        correct, total = eval_accuracy.process_livecodebench_parallel(df, df.index)

        assert (correct, total) == (2, 3)
        assert df["prompt_accuracy"].tolist() == [100.0, 0.0, 100.0]

    def test_broken_judge_in_worker_raises(self, monkeypatch, inline_pool):
        # The 0/335 case: every worker hits the same env error.
        monkeypatch.setattr(eval_accuracy, "load_lcb_benchmark", lambda: {})

        def grade(code, question_id):
            raise ImportError("No module named 'anthropic'")

        monkeypatch.setattr(eval_accuracy, "evaluate_livecodebench", grade)

        with pytest.raises(ImportError):
            eval_accuracy.process_livecodebench_parallel(_lcb_df(3), _lcb_df(3).index)

    def test_preflight_raises_before_spawning_workers(self, monkeypatch):
        def broken_benchmark():
            raise RuntimeError("Dataset scripts are no longer supported")

        monkeypatch.setattr(eval_accuracy, "load_lcb_benchmark", broken_benchmark)

        def no_pool(*args, **kwargs):
            raise AssertionError("worker pool must not be created")

        monkeypatch.setattr(eval_accuracy, "ProcessPoolExecutor", no_pool)

        with pytest.raises(RuntimeError, match="no longer supported"):
            eval_accuracy.process_livecodebench_parallel(_lcb_df(2), _lcb_df(2).index)
