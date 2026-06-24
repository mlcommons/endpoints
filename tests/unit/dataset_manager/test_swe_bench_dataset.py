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

    def test_lite_subset(self, tmp_path: Path):
        with patch(
            "inference_endpoint.dataset_manager.predefined.swe_bench.load_from_huggingface",
            return_value=_make_hf_df(),
        ) as mock_hf:
            df = SWEBench.generate(datasets_dir=tmp_path, subset="lite")

        call_kwargs = mock_hf.call_args
        assert "princeton-nlp/SWE-bench_Lite" in call_kwargs[0]
        assert len(df) == 5
