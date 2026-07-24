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

"""Unit tests for SWEBench predefined dataset."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.dataset_manager.predefined.swe_bench import SWEBench

pytestmark = pytest.mark.unit

_FAKE_INSTANCES = [
    {"instance_id": f"repo__repo-{i}", "problem_statement": f"Fix bug {i}"}
    for i in range(5)
]


def _make_hf_df() -> pd.DataFrame:
    return pd.DataFrame(_FAKE_INSTANCES)


class TestSWEBenchRegistration:
    def test_registered(self):
        assert "swe_bench" in Dataset.PREDEFINED
        assert Dataset.PREDEFINED["swe_bench"] is SWEBench

    def test_accuracy_only_flag(self):
        assert SWEBench.ACCURACY_ONLY is True

    @pytest.mark.parametrize(
        ("subset", "expected"),
        [
            ("verified", "princeton-nlp/SWE-bench_Verified"),
            ("lite", "princeton-nlp/SWE-bench_Lite"),
        ],
    )
    def test_hf_dataset_name(self, subset: str, expected: str):
        assert SWEBench.hf_dataset_name(subset) == expected

    def test_hf_dataset_name_invalid_subset_raises(self):
        with pytest.raises(ValueError, match="Unknown SWE-bench subset"):
            SWEBench.hf_dataset_name("invalid")


class TestSWEBenchGenerate:
    def test_downloads_and_caches(self, tmp_path: Path):
        with patch(
            "inference_endpoint.dataset_manager.predefined.swe_bench.load_from_huggingface",
            return_value=_make_hf_df(),
        ) as mock_hf:
            df1 = SWEBench.generate(datasets_dir=tmp_path)

        assert mock_hf.call_count == 1
        assert list(df1.columns) == ["instance_id", "prompt"]
        assert len(df1) == 5
        assert df1["prompt"].iloc[0] == "Fix bug 0"

        # Second call should hit parquet cache, not HF
        with patch(
            "inference_endpoint.dataset_manager.predefined.swe_bench.load_from_huggingface",
        ) as mock_hf2:
            df2 = SWEBench.generate(datasets_dir=tmp_path)

        mock_hf2.assert_not_called()
        assert list(df2.columns) == ["instance_id", "prompt"]
        assert len(df2) == 5

    def test_unknown_subset_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unknown SWE-bench subset"):
            SWEBench.generate(datasets_dir=tmp_path, subset="invalid")

    def test_force_regenerate(self, tmp_path: Path):
        with patch(
            "inference_endpoint.dataset_manager.predefined.swe_bench.load_from_huggingface",
            return_value=_make_hf_df(),
        ) as mock_hf:
            SWEBench.generate(datasets_dir=tmp_path)
            assert mock_hf.call_count == 1

            SWEBench.generate(datasets_dir=tmp_path, force=True)
            assert mock_hf.call_count == 2

    @pytest.mark.parametrize(
        ("subset", "split", "expected_repo"),
        [
            ("lite", "dev", "princeton-nlp/SWE-bench_Lite"),
            ("verified", "test", "princeton-nlp/SWE-bench_Verified"),
        ],
    )
    def test_subset_and_split_select_hf_repo_and_cache_directory(
        self, tmp_path: Path, subset: str, split: str, expected_repo: str
    ):
        with patch(
            "inference_endpoint.dataset_manager.predefined.swe_bench.load_from_huggingface",
            return_value=_make_hf_df(),
        ) as mock_hf:
            df = SWEBench.generate(datasets_dir=tmp_path, subset=subset, split=split)

        assert mock_hf.call_args.args == (expected_repo,)
        assert mock_hf.call_args.kwargs["split"] == split
        assert mock_hf.call_args.kwargs["cache_dir"] == (
            tmp_path / "hf_cache" / f"swe_bench_{subset}_{split}"
        )
        assert len(df) == 5
        cache_dir = tmp_path / "swe_bench" / subset / split
        assert len(list(cache_dir.glob("*.parquet"))) == 1

    def test_corrupt_parquet_cache_reports_cache_path(self, tmp_path: Path):
        cache_path = (
            tmp_path
            / "swe_bench"
            / "verified"
            / "test"
            / "swe_bench_verified_test.parquet"
        )
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text("not parquet")

        with (
            patch(
                "inference_endpoint.dataset_manager.predefined.swe_bench.pd.read_parquet",
                side_effect=ValueError("bad parquet"),
            ),
            pytest.raises(RuntimeError, match="appears corrupt"),
        ):
            SWEBench.generate(datasets_dir=tmp_path)

    def test_huggingface_load_failure_is_logged_and_reraised(
        self, tmp_path: Path, caplog
    ):
        error = RuntimeError("hf unavailable")
        with (
            patch(
                "inference_endpoint.dataset_manager.predefined.swe_bench.load_from_huggingface",
                side_effect=error,
            ),
            pytest.raises(RuntimeError, match="hf unavailable") as exc_info,
            caplog.at_level("ERROR"),
        ):
            SWEBench.generate(datasets_dir=tmp_path, subset="lite", split="dev")

        assert exc_info.value is error
        assert "Error loading SWE-bench lite/dev from HuggingFace" in caplog.text
