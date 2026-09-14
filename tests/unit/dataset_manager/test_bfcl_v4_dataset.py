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

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import inference_endpoint.dataset_manager.predefined.bfcl_v4 as bfcl_module
import pandas as pd
import pytest
from inference_endpoint.dataset_manager.predefined.bfcl_v4 import BFCLv4

pytestmark = pytest.mark.unit


def _dst(tmp_path: Path) -> Path:
    """Cache path the loader writes the full single-turn parquet to."""
    return tmp_path / "bfcl_v4" / "bfcl_v4_single_turn.parquet"


def _write_parquet_at(path: Path, subset_counts: dict[str, int]) -> None:
    """Write a synthetic single-turn parquet with the given per-subset row counts.

    messages/tools are stored as JSON strings (the on-disk shape generate()
    deserializes on load), so selection logic can run without bfcl-eval.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for subset, n in subset_counts.items():
        for i in range(n):
            rows.append(
                {
                    "messages": json.dumps(
                        [{"role": "user", "content": f"{subset}-{i}"}]
                    ),
                    "tools": json.dumps([]),
                    "ground_truth": "[]",
                    "func_description": "[]",
                    "subset": subset,
                    "sample_id": f"{subset}_{i}",
                }
            )
    pd.DataFrame(rows).to_parquet(path)


def _write_cached_parquet(tmp_path: Path, subset_counts: dict[str, int]) -> Path:
    dst = _dst(tmp_path)
    _write_parquet_at(dst, subset_counts)
    return dst


class TestR2DownloadFallback:
    def test_returns_false_when_no_uri(self, tmp_path, monkeypatch):
        """No constant and no env var -> caller falls back to bfcl-eval."""
        monkeypatch.setattr(BFCLv4, "R2_DATASET_URI", None)
        monkeypatch.delenv(BFCLv4.R2_DATASET_URI_ENV, raising=False)

        with patch.object(bfcl_module, "download_r2_artifact") as download:
            assert BFCLv4._download_full_parquet_from_r2(_dst(tmp_path)) is False
            download.assert_not_called()


class TestR2DownloadSuccess:
    def test_success_via_constant_uri(self, tmp_path, monkeypatch):
        dst = _dst(tmp_path)
        payload = b"PARQUET-BYTES"
        uri = "https://inference.mlcommons-storage.org/x.uri"
        monkeypatch.setattr(BFCLv4, "R2_DATASET_URI", uri)
        monkeypatch.delenv(BFCLv4.R2_DATASET_URI_ENV, raising=False)

        def fake_download_r2_artifact(**kwargs):
            downloaded_path = kwargs["destination_dir"] / kwargs["artifact_name"]
            downloaded_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_path.write_bytes(payload)
            return downloaded_path

        download = MagicMock(side_effect=fake_download_r2_artifact)
        with patch.object(bfcl_module, "download_r2_artifact", download):
            assert BFCLv4._download_full_parquet_from_r2(dst) is True
        assert dst.read_bytes() == payload
        download.assert_called_once_with(
            uri=uri,
            destination_dir=dst.parent,
            artifact_name=dst.name,
            expected_sha256=BFCLv4.SINGLE_TURN_SHA256,
        )

    def test_env_var_overrides_constant(self, tmp_path, monkeypatch):
        """BFCL_V4_DATASET_URI takes precedence over the class constant."""
        dst = _dst(tmp_path)
        constant_uri = "https://inference.mlcommons-storage.org/constant.uri"
        env_uri = "https://inference.mlcommons-storage.org/env.uri"
        monkeypatch.setattr(BFCLv4, "R2_DATASET_URI", constant_uri)
        monkeypatch.setenv(BFCLv4.R2_DATASET_URI_ENV, env_uri)

        download = MagicMock(return_value=dst)
        with patch.object(bfcl_module, "download_r2_artifact", download):
            assert BFCLv4._download_full_parquet_from_r2(dst) is True
        download.assert_called_once_with(
            uri=env_uri,
            destination_dir=dst.parent,
            artifact_name=dst.name,
            expected_sha256=BFCLv4.SINGLE_TURN_SHA256,
        )


class TestGenerateSelection:
    """generate()'s subset-selection / sampling logic.

    Runs against a synthetic cached parquet with no bfcl-eval and no network:
    with no R2 URI configured, generate() trusts the cache and only its own
    filter / sampling / floor / truncation code executes.
    """

    def _no_uri(self, monkeypatch):
        monkeypatch.setattr(BFCLv4, "R2_DATASET_URI", None)
        monkeypatch.delenv(BFCLv4.R2_DATASET_URI_ENV, raising=False)

    def test_unknown_category_raises(self, tmp_path, monkeypatch):
        self._no_uri(monkeypatch)
        with pytest.raises(ValueError, match="Unknown category"):
            BFCLv4.generate(tmp_path, categories=["does_not_exist"])

    def test_all_multi_turn_subsets_filtered_returns_empty(self, tmp_path, monkeypatch):
        self._no_uri(monkeypatch)
        df = BFCLv4.generate(tmp_path, subsets=["multi_turn_base"])
        assert df.empty
        assert list(df.columns) == list(BFCLv4.COLUMN_NAMES)

    @pytest.mark.parametrize("bad", [0, 150])
    def test_invalid_sample_pct_raises(self, tmp_path, monkeypatch, bad):
        self._no_uri(monkeypatch)
        _write_cached_parquet(tmp_path, {"simple_python": 5})
        with pytest.raises(ValueError, match="sampling percentage"):
            BFCLv4.generate(tmp_path, subsets=["simple_python"], sample_pct=bad)

    def test_subset_floor_full_take_vs_pct(self, tmp_path, monkeypatch):
        self._no_uri(monkeypatch)
        _write_cached_parquet(tmp_path, {"simple_python": 3, "live_simple": 10})
        df = BFCLv4.generate(
            tmp_path,
            subsets=["simple_python", "live_simple"],
            sample_pct=50,
            subset_floor=5,
        )
        counts = df["subset"].value_counts().to_dict()
        assert counts["simple_python"] == 3  # 3 <= floor 5 -> taken in full
        assert counts["live_simple"] == 5  # max(1, int(10 * 50 / 100))

    def test_max_samples_truncates(self, tmp_path, monkeypatch):
        self._no_uri(monkeypatch)
        _write_cached_parquet(tmp_path, {"simple_python": 10})
        df = BFCLv4.generate(tmp_path, subsets=["simple_python"], max_samples=4)
        assert len(df) == 4

    def test_resolve_subset_pct_category_precedence(self):
        # A listed category's rate wins over the uniform rate...
        assert BFCLv4._resolve_subset_pct("live_simple", 50, {"live": 10}) == 10
        # ...a subset whose category is not listed falls back to uniform...
        assert BFCLv4._resolve_subset_pct("simple_python", 50, {"live": 10}) == 50
        # ...and with no category map at all, uniform is used.
        assert BFCLv4._resolve_subset_pct("simple_python", 50, None) == 50

    def test_cached_parquet_reverified_and_rejected_on_mismatch(
        self, tmp_path, monkeypatch
    ):
        _write_cached_parquet(tmp_path, {"simple_python": 2})
        monkeypatch.setattr(
            BFCLv4, "R2_DATASET_URI", "https://inference.mlcommons-storage.org/x.uri"
        )
        monkeypatch.delenv(BFCLv4.R2_DATASET_URI_ENV, raising=False)

        def _download(dst_path):
            _write_parquet_at(dst_path, {"simple_python": 1})
            return True

        with (
            patch.object(
                bfcl_module, "verify_sha256", side_effect=ValueError("mismatch")
            ),
            patch.object(
                BFCLv4, "_download_full_parquet_from_r2", side_effect=_download
            ) as dl,
        ):
            df = BFCLv4.generate(tmp_path, subsets=["simple_python"])

        dl.assert_called_once()
        assert len(df) == 1

    def test_cached_parquet_trusted_when_sha_matches(self, tmp_path, monkeypatch):
        _write_cached_parquet(tmp_path, {"simple_python": 2})
        monkeypatch.setattr(
            BFCLv4, "R2_DATASET_URI", "https://inference.mlcommons-storage.org/x.uri"
        )
        monkeypatch.delenv(BFCLv4.R2_DATASET_URI_ENV, raising=False)

        with (
            patch.object(bfcl_module, "verify_sha256", return_value=None),
            patch.object(BFCLv4, "_download_full_parquet_from_r2") as dl,
        ):
            df = BFCLv4.generate(tmp_path, subsets=["simple_python"])

        dl.assert_not_called()  # verified cache is trusted, no re-download
        assert len(df) == 2
