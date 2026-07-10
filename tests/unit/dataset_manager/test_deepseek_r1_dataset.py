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

"""Unit tests for the predefined DeepSeek-R1 dataset (prepared parquet source)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.dataset_manager.predefined import (
    legacy_mlperf_deepseek_r1 as dsr1_mod,
)
from inference_endpoint.dataset_manager.predefined.legacy_mlperf_deepseek_r1 import (
    SOURCE_ENV,
    LegacyMLPerfDeepSeekR1,
)

pytestmark = pytest.mark.unit


def _prepared_df() -> pd.DataFrame:
    """A prepared DeepSeek-R1 frame: output columns already present.

    ``tok_output`` is an extra source column that must be dropped from output.
    """
    return pd.DataFrame(
        {
            "input_tokens": [[1, 2, 3], [4, 5], [6]],
            "ground_truth": ["42", "lcb_q7", "C"],
            "dataset": ["math500", "livecodebench", "gpqa"],
            "question": ["q-math", "q-code", "q-gpqa"],
            "tok_output": [[9], [9, 9], [9, 9, 9]],
        }
    )


@pytest.fixture
def prepared_parquet(tmp_path: Path) -> Path:
    path = tmp_path / "deepseek_r1_eval.parquet"
    _prepared_df().to_parquet(path, index=False)
    return path


@pytest.fixture
def raw_pkl(tmp_path: Path) -> Path:
    """A raw MLPerf-shaped ``.pkl`` (pre-tokenization columns)."""
    path = tmp_path / "deepseek_r1.pkl"
    pd.DataFrame(
        {
            "tok_input": [np.array([1, 2, 3]), np.array([4, 5])],
            "ground_truth": ["42", "C"],
            "dataset": ["math500", "gpqa"],
            "question": ["q-math", "q-gpqa"],
        }
    ).to_pickle(path)
    return path


def test_registered_in_predefined():
    assert Dataset.PREDEFINED["legacy_mlperf_deepseek_r1"] is LegacyMLPerfDeepSeekR1
    assert LegacyMLPerfDeepSeekR1.DATASET_ID == "legacy_mlperf_deepseek_r1"


def test_generate_prepared_parquet_caches(tmp_path: Path, prepared_parquet: Path):
    cache = tmp_path / "cache"
    df = LegacyMLPerfDeepSeekR1.generate(datasets_dir=cache, source=prepared_parquet)

    # Only the output columns survive; the extra source column is dropped.
    assert list(df.columns) == ["input_tokens", "ground_truth", "dataset", "question"]
    assert df["ground_truth"].tolist() == ["42", "lcb_q7", "C"]
    assert df["dataset"].tolist() == ["math500", "livecodebench", "gpqa"]

    # The cache parquet is written under <datasets_dir>/legacy_mlperf_deepseek_r1/.
    cached = cache / "legacy_mlperf_deepseek_r1" / "deepseek_r1_eval.parquet"
    assert cached.exists()


def test_generate_loads_from_cache_without_source(
    tmp_path: Path, prepared_parquet: Path
):
    cache = tmp_path / "cache"
    LegacyMLPerfDeepSeekR1.generate(datasets_dir=cache, source=prepared_parquet)
    # Second call needs no source: it reads the cached parquet.
    df = LegacyMLPerfDeepSeekR1.generate(datasets_dir=cache)
    assert len(df) == 3
    assert "input_tokens" in df.columns


def test_force_rebuilds_cache(tmp_path: Path, prepared_parquet: Path, monkeypatch):
    cache = tmp_path / "cache"
    LegacyMLPerfDeepSeekR1.generate(datasets_dir=cache, source=prepared_parquet)
    # force=True must rebuild from source: with no source/env and the bundled
    # parquet absent, it re-resolves and raises rather than reusing the cache.
    monkeypatch.delenv(SOURCE_ENV, raising=False)
    monkeypatch.setattr(dsr1_mod, "_BUNDLED_PARQUET", tmp_path / "absent.parquet")
    with pytest.raises(FileNotFoundError):
        LegacyMLPerfDeepSeekR1.generate(datasets_dir=cache, force=True)


def test_generate_uses_env_var(tmp_path: Path, prepared_parquet: Path, monkeypatch):
    monkeypatch.setenv(SOURCE_ENV, str(prepared_parquet))
    df = LegacyMLPerfDeepSeekR1.generate(datasets_dir=tmp_path / "cache")
    assert len(df) == 3


def test_raw_pkl_source_not_supported(tmp_path: Path, raw_pkl: Path):
    # A raw .pkl is rejected on suffix before any read.
    with pytest.raises(NotImplementedError, match="raw MLPerf source"):
        LegacyMLPerfDeepSeekR1.generate(datasets_dir=tmp_path / "cache", source=raw_pkl)


def test_prepared_pkl_rejected_by_suffix(tmp_path: Path):
    # Even a prepared .pkl is rejected: only .parquet is supported.
    p = tmp_path / "prepared.pkl"
    _prepared_df().to_pickle(p)
    with pytest.raises(NotImplementedError, match="prepared .parquet"):
        LegacyMLPerfDeepSeekR1.generate(datasets_dir=tmp_path / "cache", source=p)


def test_parquet_without_input_tokens_not_supported(tmp_path: Path):
    # A parquet of raw (un-tokenized) columns has no input_tokens -> unsupported.
    bad = tmp_path / "raw.parquet"
    pd.DataFrame(
        {
            "tok_input": [[1, 2]],
            "ground_truth": ["a"],
            "dataset": ["math500"],
            "question": ["q"],
        }
    ).to_parquet(bad, index=False)
    with pytest.raises(NotImplementedError, match="input_tokens"):
        LegacyMLPerfDeepSeekR1.generate(datasets_dir=tmp_path / "cache", source=bad)


def test_defaults_to_bundled_parquet_when_unset(
    tmp_path: Path, prepared_parquet: Path, monkeypatch
):
    # No source, no env -> falls back to the bundled example parquet.
    monkeypatch.delenv(SOURCE_ENV, raising=False)
    monkeypatch.setattr(dsr1_mod, "_BUNDLED_PARQUET", prepared_parquet)
    df = LegacyMLPerfDeepSeekR1.generate(datasets_dir=tmp_path / "cache")
    assert len(df) == 3
    assert "input_tokens" in df.columns


def test_missing_source_raises(tmp_path: Path, monkeypatch):
    # No source, no env, and the bundled parquet absent (wheel install) -> raise.
    monkeypatch.delenv(SOURCE_ENV, raising=False)
    monkeypatch.setattr(dsr1_mod, "_BUNDLED_PARQUET", tmp_path / "absent.parquet")
    with pytest.raises(FileNotFoundError, match=SOURCE_ENV):
        LegacyMLPerfDeepSeekR1.generate(datasets_dir=tmp_path / "cache")


def test_resolved_source_not_found_raises(tmp_path: Path):
    missing = tmp_path / "nope.parquet"
    with pytest.raises(FileNotFoundError, match="source not found"):
        LegacyMLPerfDeepSeekR1.generate(datasets_dir=tmp_path / "cache", source=missing)


def test_lfs_pointer_source_raises_actionable(tmp_path: Path):
    # A git-LFS pointer stub (clone without LFS) must give an actionable error,
    # not a cryptic pd.read_parquet failure.
    ptr = tmp_path / "deepseek_r1_eval.parquet"
    ptr.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc123\nsize 4700000\n"
    )
    with pytest.raises(FileNotFoundError, match="git-LFS pointer"):
        LegacyMLPerfDeepSeekR1.generate(datasets_dir=tmp_path / "cache", source=ptr)


def test_prepared_parquet_missing_output_column_raises(tmp_path: Path):
    # input_tokens present but a required output column (question) absent.
    bad = tmp_path / "partial.parquet"
    pd.DataFrame(
        {"input_tokens": [[1]], "ground_truth": ["a"], "dataset": ["math500"]}
    ).to_parquet(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        LegacyMLPerfDeepSeekR1.generate(datasets_dir=tmp_path / "cache", source=bad)


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

    df = LegacyMLPerfDeepSeekR1.generate(
        datasets_dir=tmp_path / "cache", source=src, max_samples=10
    )
    assert len(df) == 10  # never overshoots the requested size
    # Both subsets represented (proportional sampling, not all-from-one).
    assert set(df["dataset"]) == {"math500", "gpqa"}


def test_stratified_subset_never_overshoots(tmp_path: Path):
    # Many subsets + small max_samples: the per-subset >=1 floor summed to 5
    # (one per subset) and overshot a request of 3. Must be capped at max_samples.
    src = tmp_path / "many.parquet"
    pd.DataFrame(
        {
            "input_tokens": [[1]] * 50,
            "ground_truth": ["g"] * 50,
            "dataset": ["a"] * 10 + ["b"] * 10 + ["c"] * 10 + ["d"] * 10 + ["e"] * 10,
            "question": ["q"] * 50,
        }
    ).to_parquet(src, index=False)

    df = LegacyMLPerfDeepSeekR1.generate(
        datasets_dir=tmp_path / "cache", source=src, max_samples=3
    )
    assert len(df) == 3


def test_get_dataloader_loads_msgspec_safe_tokens(
    tmp_path: Path, prepared_parquet: Path
):
    """create_loader -> get_dataloader -> generate threads `source`, and
    Dataset.load() normalizes input_tokens to plain Python ints (msgspec-safe)."""
    ds = LegacyMLPerfDeepSeekR1.get_dataloader(
        datasets_dir=tmp_path / "cache", source=prepared_parquet
    )
    ds.load()
    assert ds.num_samples() == 3
    row = ds.load_sample(0)
    assert isinstance(row["input_tokens"], list)
    assert all(isinstance(t, int) for t in row["input_tokens"])


def test_factory_resolves_deepseek_r1_via_env(
    tmp_path: Path, prepared_parquet: Path, monkeypatch
):
    """The user-facing contract: `--dataset legacy_mlperf_deepseek_r1` (name -> PREDEFINED)
    resolves through the factory and loads from the local env source."""
    from inference_endpoint.config.schema import Dataset as DatasetConfig
    from inference_endpoint.dataset_manager.factory import DataLoaderFactory

    monkeypatch.setenv(SOURCE_ENV, str(prepared_parquet))
    monkeypatch.chdir(tmp_path)  # default dataset_cache/ lands here

    cfg = DatasetConfig(name="legacy_mlperf_deepseek_r1", type="accuracy")
    ds = DataLoaderFactory.create_loader(cfg, num_repeats=1)
    ds.load()

    assert isinstance(ds, LegacyMLPerfDeepSeekR1)
    assert ds.num_samples() == 3
    row = ds.load_sample(0)
    assert set(row) >= {"input_tokens", "ground_truth", "dataset", "question"}
