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

"""Unit tests for the composite gpt-oss-120b accuracy dataset.

The three subset ``generate()`` methods are monkeypatched to tiny in-memory
frames so the test never touches HuggingFace or the LiveCodeBench venv; it
isolates the compose-and-render step: unified schema, per-subset prompt
rendering, ground-truth mapping, and caching.
"""

from pathlib import Path

import pandas as pd
import pytest
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.dataset_manager.predefined.aime25 import AIME25
from inference_endpoint.dataset_manager.predefined.gpqa import GPQA
from inference_endpoint.dataset_manager.predefined.gptoss_120b_accuracy import (
    GptOss120bAccuracy,
)
from inference_endpoint.dataset_manager.predefined.livecodebench import LiveCodeBench

pytestmark = pytest.mark.unit


def _fake_aime(**kwargs) -> pd.DataFrame:
    return pd.DataFrame({"question": ["2+2?", "3*3?"], "answer": ["4", "9"]})


def _fake_gpqa(**kwargs) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question": ["Q1", "Q2"],
            "choice1": ["a1", "b1"],
            "choice2": ["a2", "b2"],
            "choice3": ["a3", "b3"],
            "choice4": ["a4", "b4"],
            "ground_truth": ["choice1", "choice3"],
            "domain": ["d", "d"],
            "subdomain": ["s", "s"],
        }
    )


def _fake_lcb(**kwargs) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question_id": ["q100", "q101", "q102"],
            "question": ["c1", "c2", "c3"],
            "starter_code": ["def f(): ...", "def g(): ...", "def h(): ..."],
        }
    )


@pytest.fixture
def patched_subsets(monkeypatch):
    """Monkeypatch each subset's generate() to a tiny frame; track call counts."""
    calls = {"aime25": 0, "gpqa": 0, "livecodebench": 0}

    def _wrap(name, fn):
        def inner(**kwargs):
            calls[name] += 1
            return fn(**kwargs)

        return inner

    monkeypatch.setattr(AIME25, "generate", _wrap("aime25", _fake_aime))
    monkeypatch.setattr(GPQA, "generate", _wrap("gpqa", _fake_gpqa))
    monkeypatch.setattr(LiveCodeBench, "generate", _wrap("livecodebench", _fake_lcb))
    return calls


def test_registered_in_predefined():
    assert Dataset.PREDEFINED["gptoss_120b_accuracy"] is GptOss120bAccuracy
    assert GptOss120bAccuracy.DATASET_ID == "gptoss_120b_accuracy"


def test_compose_unified_schema_and_mapping(tmp_path: Path, patched_subsets):
    df = GptOss120bAccuracy.generate(datasets_dir=tmp_path / "cache")

    assert list(df.columns) == ["prompt", "subset", "ground_truth", "question"]
    assert set(df["subset"]) == {"aime25", "gpqa", "livecodebench"}
    assert len(df) == 2 + 2 + 3  # sum of the three subsets

    # ground_truth carries each subset's native answer column, cast to str.
    by_subset = df.groupby("subset")["ground_truth"].apply(list)
    assert by_subset["aime25"] == ["4", "9"]
    assert by_subset["gpqa"] == ["choice1", "choice3"]
    assert by_subset["livecodebench"] == ["q100", "q101", "q102"]


def test_prompts_rendered_per_subset(tmp_path: Path, patched_subsets):
    df = GptOss120bAccuracy.generate(datasets_dir=tmp_path / "cache")
    prompt = dict(zip(df["subset"], df["prompt"], strict=False))

    # Each subset's own gptoss() preset is applied (different templates).
    assert "\\boxed{}" in prompt["aime25"]
    assert "(A)" in prompt["gpqa"] and "(D)" in prompt["gpqa"]
    assert "```python" in prompt["livecodebench"]


def test_caching_reads_parquet_without_regen(tmp_path: Path, patched_subsets):
    cache = tmp_path / "cache"
    GptOss120bAccuracy.generate(datasets_dir=cache)
    assert patched_subsets["aime25"] == 1

    cached = cache / "gptoss_120b_accuracy" / "gptoss_120b_accuracy.parquet"
    assert cached.exists()

    # Second call hits the cache: no subset generate() re-runs.
    df = GptOss120bAccuracy.generate(datasets_dir=cache)
    assert patched_subsets["aime25"] == 1
    assert len(df) == 7


def test_force_rebuilds(tmp_path: Path, patched_subsets):
    cache = tmp_path / "cache"
    GptOss120bAccuracy.generate(datasets_dir=cache)
    assert patched_subsets["aime25"] == 1

    GptOss120bAccuracy.generate(datasets_dir=cache, force=True)
    assert patched_subsets["aime25"] == 2


def test_stratified_subset(tmp_path: Path, patched_subsets):
    df = GptOss120bAccuracy.generate(datasets_dir=tmp_path / "cache", max_samples=3)
    assert len(df) == 3
    # Stratified across subsets, never overshoots the request.
    assert set(df["subset"]).issubset({"aime25", "gpqa", "livecodebench"})


def test_load_preserves_routing_columns(tmp_path: Path, patched_subsets):
    """get_dataloader -> load() keeps subset/ground_truth on dataframe for the scorer."""
    ds = GptOss120bAccuracy.get_dataloader(datasets_dir=tmp_path / "cache")
    ds.load()

    # Rows are replicated per-subset (8/5/3): 2*8 + 2*5 + 3*3 = 35.
    assert ds.num_samples() == 35
    # Dataset.load() transforms a local copy into self.data; dataframe is intact
    # (and replicated), so the scorer can still read the routing/answer columns.
    assert ds.dataframe is not None
    assert "subset" in ds.dataframe.columns
    assert "ground_truth" in ds.dataframe.columns
    row = ds.load_sample(0)
    assert set(row) >= {"prompt", "subset", "ground_truth", "question"}


def test_load_replicates_per_subset_repeats(tmp_path: Path, patched_subsets):
    ds = GptOss120bAccuracy.get_dataloader(datasets_dir=tmp_path / "cache")
    ds.load()

    # aime25 2x8=16, gpqa 2x5=10, livecodebench 3x3=9 -> 35 total.
    assert ds.num_samples() == 35
    assert ds.dataframe is not None
    counts = ds.dataframe["subset"].value_counts().to_dict()
    assert counts == {"aime25": 16, "gpqa": 10, "livecodebench": 9}


def test_config_num_repeats_multiplies_on_top(tmp_path: Path, patched_subsets):
    ds = GptOss120bAccuracy.get_dataloader(
        datasets_dir=tmp_path / "cache", num_repeats=2
    )
    ds.load()

    # Replication is fixed 8/5/3; the config num_repeats rides on top as the
    # issue-time multiplier (num_samples() x repeats), so effective issue = 70.
    assert ds.num_samples() == 35
    assert ds.repeats == 2
    assert ds.num_samples() * ds.repeats == 70


def test_load_force_does_not_double_replicate(tmp_path: Path, patched_subsets):
    ds = GptOss120bAccuracy.get_dataloader(datasets_dir=tmp_path / "cache")
    ds.load()
    assert ds.num_samples() == 35

    ds.load(force=True)  # rebuilds self.data from the already-replicated frame
    assert ds.num_samples() == 35
    assert ds.dataframe is not None
    assert len(ds.dataframe) == 35
