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

"""Unit tests for the predefined DeepSeek-R1 dataset (local source)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.dataset_manager.predefined.deepseek_r1 import (
    SOURCE_ENV,
    DeepSeekR1,
)

pytestmark = pytest.mark.unit


def _raw_source_df() -> pd.DataFrame:
    """A minimal MLPerf-shaped source frame (pre-prepare columns)."""
    return pd.DataFrame(
        {
            "tok_input": [np.array([1, 2, 3]), np.array([4, 5]), np.array([6])],
            "ground_truth": ["42", "lcb_q7", "C"],
            "dataset": ["math500", "livecodebench", "gpqa"],
            "question": ["q-math", "q-code", "q-gpqa"],
            # An extra source column that must be dropped from the output.
            "tok_output": [[9], [9, 9], [9, 9, 9]],
        }
    )


@pytest.fixture
def pkl_source(tmp_path: Path) -> Path:
    path = tmp_path / "deepseek_r1.pkl"
    _raw_source_df().to_pickle(path)
    return path


def test_registered_in_predefined():
    assert Dataset.PREDEFINED["deepseek_r1"] is DeepSeekR1
    assert DeepSeekR1.DATASET_ID == "deepseek_r1"


def test_generate_transforms_and_caches(tmp_path: Path, pkl_source: Path):
    cache = tmp_path / "cache"
    df = DeepSeekR1.generate(datasets_dir=cache, source=pkl_source)

    assert list(df.columns) == ["input_tokens", "ground_truth", "dataset", "question"]
    # tok_input -> input_tokens, as plain Python lists (msgspec-friendly).
    assert df["input_tokens"].tolist() == [[1, 2, 3], [4, 5], [6]]
    assert all(isinstance(v, list) for v in df["input_tokens"])
    assert df["ground_truth"].tolist() == ["42", "lcb_q7", "C"]
    assert df["dataset"].tolist() == ["math500", "livecodebench", "gpqa"]

    # The cache parquet is written under <datasets_dir>/deepseek_r1/.
    cached = cache / "deepseek_r1" / "deepseek_r1_eval.parquet"
    assert cached.exists()


def test_generate_loads_from_cache_without_source(tmp_path: Path, pkl_source: Path):
    cache = tmp_path / "cache"
    DeepSeekR1.generate(datasets_dir=cache, source=pkl_source)
    # Second call needs no source: it reads the cached parquet.
    df = DeepSeekR1.generate(datasets_dir=cache)
    assert len(df) == 3
    assert "input_tokens" in df.columns


def test_force_rebuilds_cache(tmp_path: Path, pkl_source: Path):
    cache = tmp_path / "cache"
    DeepSeekR1.generate(datasets_dir=cache, source=pkl_source)
    # force=True must rebuild from source, so a missing source now raises.
    with pytest.raises(FileNotFoundError):
        DeepSeekR1.generate(datasets_dir=cache, force=True)


def test_generate_uses_env_var(tmp_path: Path, pkl_source: Path, monkeypatch):
    monkeypatch.setenv(SOURCE_ENV, str(pkl_source))
    df = DeepSeekR1.generate(datasets_dir=tmp_path / "cache")
    assert len(df) == 3


def test_prepared_parquet_passthrough(tmp_path: Path):
    prepared = tmp_path / "prepared.parquet"
    pd.DataFrame(
        {
            "input_tokens": [[1, 2], [3]],
            "ground_truth": ["a", "b"],
            "dataset": ["math500", "aime"],
            "question": ["q1", "q2"],
            "extra": [0, 1],
        }
    ).to_parquet(prepared, index=False)

    df = DeepSeekR1.generate(datasets_dir=tmp_path / "cache", source=prepared)
    assert list(df.columns) == ["input_tokens", "ground_truth", "dataset", "question"]
    assert len(df) == 2


def test_missing_source_raises(tmp_path: Path, monkeypatch):
    monkeypatch.delenv(SOURCE_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match=SOURCE_ENV):
        DeepSeekR1.generate(datasets_dir=tmp_path / "cache")


def test_source_missing_columns_raises(tmp_path: Path):
    bad = tmp_path / "bad.parquet"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_parquet(bad, index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        DeepSeekR1.generate(datasets_dir=tmp_path / "cache", source=bad)


def test_resolved_source_not_found_raises(tmp_path: Path):
    missing = tmp_path / "nope.pkl"
    with pytest.raises(FileNotFoundError, match="source not found"):
        DeepSeekR1.generate(datasets_dir=tmp_path / "cache", source=missing)


def test_prepared_parquet_missing_output_column_raises(tmp_path: Path):
    # input_tokens present but a required output column (question) absent.
    bad = tmp_path / "partial.parquet"
    pd.DataFrame(
        {"input_tokens": [[1]], "ground_truth": ["a"], "dataset": ["math500"]}
    ).to_parquet(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        DeepSeekR1.generate(datasets_dir=tmp_path / "cache", source=bad)


def test_stratified_subset(tmp_path: Path):
    # 100 rows across two subsets; ask for ~10 -> proportional, both present.
    src = tmp_path / "big.parquet"
    pd.DataFrame(
        {
            "input_tokens": [[1]] * 100,
            "ground_truth": ["g"] * 100,
            "dataset": ["math500"] * 60 + ["gpqa"] * 40,
            "question": ["q"] * 100,
        }
    ).to_parquet(src, index=False)

    df = DeepSeekR1.generate(
        datasets_dir=tmp_path / "cache", source=src, max_samples=10
    )
    assert 8 <= len(df) <= 12
    # Both subsets represented (proportional sampling, not all-from-one).
    assert set(df["dataset"]) == {"math500", "gpqa"}


def test_get_dataloader_threads_source(tmp_path: Path, pkl_source: Path):
    """create_loader -> get_dataloader -> generate(**kwargs) threads `source`."""
    ds = DeepSeekR1.get_dataloader(datasets_dir=tmp_path / "cache", source=pkl_source)
    ds.load()
    assert ds.num_samples() == 3


def test_factory_resolves_deepseek_r1_via_env(
    tmp_path: Path, pkl_source: Path, monkeypatch
):
    """The user-facing contract: `--dataset deepseek_r1` (name -> PREDEFINED)
    resolves through the factory and loads from the local env source."""
    from inference_endpoint.config.schema import Dataset as DatasetConfig
    from inference_endpoint.dataset_manager.factory import DataLoaderFactory

    monkeypatch.setenv(SOURCE_ENV, str(pkl_source))
    monkeypatch.chdir(tmp_path)  # default dataset_cache/ lands here

    cfg = DatasetConfig(name="deepseek_r1", type="accuracy")
    ds = DataLoaderFactory.create_loader(cfg, num_repeats=1)
    ds.load()

    assert isinstance(ds, DeepSeekR1)
    assert ds.num_samples() == 3
    row = ds.load_sample(0)
    assert set(row) >= {"input_tokens", "ground_truth", "dataset", "question"}
