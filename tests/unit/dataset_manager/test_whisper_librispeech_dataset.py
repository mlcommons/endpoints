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

"""Unit tests for the WhisperLibriSpeech predefined dataset.

The download tests run without network: requests.get (downloader script fetch)
and subprocess.run (the mlc-r2-downloader invocation) are mocked in
`dataset.download_from_r2`, so only the loader's own logic runs.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.dataset_manager.predefined.whisper_librispeech import (
    MANIFEST_FILENAME,
    SOURCE_ENV,
    WhisperLibriSpeech,
)

_MODULE = "inference_endpoint.dataset_manager.dataset"


def _entry(fname: str, transcript: str, duration: float = 29.5) -> dict:
    """One MLCommons dev-all-repack.json entry."""
    return {
        "transcript": transcript,
        "files": [{"fname": fname, "speed": 1, "duration": duration}],
        "original_duration": duration,
        "original_num_samples": int(duration * 16000),
    }


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    path = tmp_path / "dev-all-repack.json"
    path.write_text(
        json.dumps(
            [
                _entry("dev-all-repack/0.wav", "mister quilter is the apostle", 29.1),
                _entry("dev-all-repack/1.wav", "of the middle classes", 28.4),
            ]
        )
    )
    return path


def _mock_requests_get() -> MagicMock:
    """A requests.get returning a trivial downloader script body."""
    resp = MagicMock()
    resp.content = b"#!/usr/bin/env bash\nexit 0\n"
    resp.raise_for_status.return_value = None
    return resp


def _fake_run_unpacking(cmd, **kwargs) -> MagicMock:
    """subprocess.run side effect that unpacks a corpus under the -d directory."""
    download_dir = Path(cmd[cmd.index("-d") + 1])
    (download_dir / "dev-all-repack").mkdir(parents=True, exist_ok=True)
    (download_dir / "dev-all-repack" / "0.wav").write_bytes(b"RIFF")
    (download_dir / MANIFEST_FILENAME).write_text(
        json.dumps([_entry("dev-all-repack/0.wav", "downloaded transcript")])
    )
    return MagicMock(returncode=0, stderr="")


@pytest.fixture
def no_local_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the env overrides so generate() takes the download path."""
    monkeypatch.delenv(WhisperLibriSpeech.R2_DATASET_URI_ENV, raising=False)
    monkeypatch.delenv(SOURCE_ENV, raising=False)


@pytest.mark.unit
class TestRegistration:
    def test_registered_under_its_dataset_id(self):
        assert Dataset.PREDEFINED["whisper_librispeech"] is WhisperLibriSpeech

    def test_serves_both_phases(self):
        # ACCURACY_ONLY would make the perf phase reject this dataset.
        assert WhisperLibriSpeech.ACCURACY_ONLY is False

    def test_column_names(self):
        assert WhisperLibriSpeech.COLUMN_NAMES == [
            "audio_file",
            "transcript",
            "duration",
        ]


@pytest.mark.unit
class TestGenerate:
    def test_flattens_manifest_entries(self, tmp_path: Path, manifest: Path):
        df = WhisperLibriSpeech.generate(tmp_path / "cache", manifest=manifest)
        assert list(df.columns) == WhisperLibriSpeech.COLUMN_NAMES
        assert len(df) == 2
        assert df.iloc[0]["transcript"] == "mister quilter is the apostle"
        assert df.iloc[0]["duration"] == pytest.approx(29.1)

    def test_resolves_fname_against_the_manifest_directory(
        self, tmp_path: Path, manifest: Path
    ):
        df = WhisperLibriSpeech.generate(tmp_path / "cache", manifest=manifest)
        assert df.iloc[0]["audio_file"] == str(manifest.parent / "dev-all-repack/0.wav")

    def test_explicit_data_dir_overrides_the_manifest_directory(
        self, tmp_path: Path, manifest: Path
    ):
        df = WhisperLibriSpeech.generate(
            tmp_path / "cache", manifest=manifest, data_dir="/mnt/audio"
        )
        assert df.iloc[0]["audio_file"] == "/mnt/audio/dev-all-repack/0.wav"

    def test_drops_fname_prefix_absent_from_the_corpus_root(self, tmp_path: Path):
        # The hosted artifact unpacks dev-all-repack/ into the corpus root,
        # while the manifest's fnames keep the upstream pipeline's ./data/
        # prefix. The clips that exist on disk are what decide the layout.
        source = tmp_path / "source"
        (source / "dev-all-repack").mkdir(parents=True)
        for name in ("0.wav", "1.wav"):
            (source / "dev-all-repack" / name).write_bytes(b"RIFF")
        path = source / MANIFEST_FILENAME
        path.write_text(
            json.dumps(
                [
                    _entry("./data/dev-all-repack/0.wav", "first"),
                    _entry("./data/dev-all-repack/1.wav", "second"),
                ]
            )
        )

        df = WhisperLibriSpeech.generate(tmp_path / "cache", manifest=path)
        assert df.iloc[0]["audio_file"] == str(source / "dev-all-repack/0.wav")
        assert df.iloc[1]["audio_file"] == str(source / "dev-all-repack/1.wav")

    def test_keeps_fname_as_written_when_no_clip_resolves(self, tmp_path: Path):
        path = tmp_path / MANIFEST_FILENAME
        path.write_text(json.dumps([_entry("./data/dev-all-repack/0.wav", "first")]))

        df = WhisperLibriSpeech.generate(tmp_path / "cache", manifest=path)
        assert df.iloc[0]["audio_file"] == str(tmp_path / "data/dev-all-repack/0.wav")

    def test_writes_and_reuses_the_parquet_cache(self, tmp_path: Path, manifest: Path):
        cache = tmp_path / "cache"
        WhisperLibriSpeech.generate(cache, manifest=manifest)
        parquet = cache / "whisper_librispeech" / "whisper_librispeech.parquet"
        assert parquet.exists()

        # A second call must not need the manifest at all.
        manifest.unlink()
        df = WhisperLibriSpeech.generate(cache, manifest=manifest)
        assert len(df) == 2

    def test_force_rebuilds_from_the_manifest(self, tmp_path: Path, manifest: Path):
        cache = tmp_path / "cache"
        WhisperLibriSpeech.generate(cache, manifest=manifest)
        manifest.write_text(json.dumps([_entry("dev-all-repack/0.wav", "only one")]))
        df = WhisperLibriSpeech.generate(cache, manifest=manifest, force=True)
        assert len(df) == 1
        assert df.iloc[0]["transcript"] == "only one"

    def test_max_samples_truncates_in_manifest_order(
        self, tmp_path: Path, manifest: Path
    ):
        df = WhisperLibriSpeech.generate(
            tmp_path / "cache", manifest=manifest, max_samples=1
        )
        assert len(df) == 1
        assert df.iloc[0]["transcript"] == "mister quilter is the apostle"

    def test_reads_the_manifest_from_the_environment(
        self, tmp_path: Path, manifest: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(SOURCE_ENV, str(manifest))
        df = WhisperLibriSpeech.generate(tmp_path / "cache")
        assert len(df) == 2


@pytest.mark.unit
class TestR2Download:
    def test_downloads_when_no_manifest_is_supplied(
        self, tmp_path: Path, no_local_manifest: None
    ):
        with (
            patch(f"{_MODULE}.requests.get", return_value=_mock_requests_get()),
            patch(f"{_MODULE}.subprocess.run", side_effect=_fake_run_unpacking),
        ):
            df = WhisperLibriSpeech.generate(tmp_path / "cache")

        assert df.iloc[0]["transcript"] == "downloaded transcript"
        # fname resolves against the downloaded manifest's own directory.
        assert df.iloc[0]["audio_file"] == str(
            tmp_path / "cache/whisper_librispeech/source/dev-all-repack/0.wav"
        )

    def test_a_local_manifest_skips_the_download(self, tmp_path: Path, manifest: Path):
        with patch(f"{_MODULE}.requests.get") as get:
            WhisperLibriSpeech.generate(tmp_path / "cache", manifest=manifest)
            get.assert_not_called()

    def test_force_reuses_the_download_instead_of_refetching(
        self, tmp_path: Path, no_local_manifest: None
    ):
        cache = tmp_path / "cache"
        with (
            patch(f"{_MODULE}.requests.get", return_value=_mock_requests_get()),
            patch(f"{_MODULE}.subprocess.run", side_effect=_fake_run_unpacking),
        ):
            WhisperLibriSpeech.generate(cache)

        # force rebuilds the parquet; it must not re-fetch GBs of audio.
        with patch(f"{_MODULE}.requests.get") as get:
            assert len(WhisperLibriSpeech.generate(cache, force=True)) == 1
            get.assert_not_called()

    @pytest.mark.parametrize(
        "uri",
        [
            "https://evil.example.com/x.uri",
            "http://inference.mlcommons-storage.org/x.uri",
        ],
    )
    def test_untrusted_uri_is_refused_before_any_fetch(
        self,
        tmp_path: Path,
        uri: str,
        no_local_manifest: None,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv(WhisperLibriSpeech.R2_DATASET_URI_ENV, uri)
        with patch(f"{_MODULE}.requests.get") as get:
            with pytest.raises(ValueError, match="untrusted URI"):
                WhisperLibriSpeech.generate(tmp_path / "cache")
            get.assert_not_called()

    def test_downloader_failure_is_reported(
        self, tmp_path: Path, no_local_manifest: None
    ):
        with (
            patch(f"{_MODULE}.requests.get", return_value=_mock_requests_get()),
            patch(
                f"{_MODULE}.subprocess.run",
                return_value=MagicMock(returncode=2, stderr="r2 boom"),
            ),
        ):
            with pytest.raises(RuntimeError, match="r2 boom"):
                WhisperLibriSpeech.generate(tmp_path / "cache")


@pytest.mark.unit
class TestGenerateErrors:
    def test_absent_manifest_path(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="not found at"):
            WhisperLibriSpeech.generate(
                tmp_path / "cache", manifest=tmp_path / "missing.json"
            )

    def test_non_json_source_is_rejected(self, tmp_path: Path):
        raw = tmp_path / "dev-all.flac"
        raw.write_bytes(b"\x00")
        with pytest.raises(NotImplementedError, match="dev-all-repack.json"):
            WhisperLibriSpeech.generate(tmp_path / "cache", manifest=raw)

    def test_entry_without_files_is_rejected(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([{"transcript": "hi", "original_duration": 1.0}]))
        with pytest.raises(ValueError, match="no 'files' list"):
            WhisperLibriSpeech.generate(tmp_path / "cache", manifest=path)

    def test_speed_perturbed_only_entry_is_rejected(self, tmp_path: Path):
        path = tmp_path / "perturbed.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "transcript": "hi",
                        "files": [
                            {"fname": "slow.wav", "speed": 0.9, "duration": 1.0},
                            {"fname": "fast.wav", "speed": 1.1, "duration": 1.0},
                        ],
                        "original_duration": 1.0,
                    }
                ]
            )
        )
        with pytest.raises(ValueError, match="speed == 1"):
            WhisperLibriSpeech.generate(tmp_path / "cache", manifest=path)

    def test_unperturbed_clip_is_selected(self, tmp_path: Path):
        path = tmp_path / "mixed.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "transcript": "hi",
                        "files": [
                            {"fname": "slow.wav", "speed": 0.9, "duration": 1.1},
                            {"fname": "normal.wav", "speed": 1, "duration": 1.0},
                        ],
                        "original_duration": 1.0,
                    }
                ]
            )
        )
        df = WhisperLibriSpeech.generate(tmp_path / "cache", manifest=path)
        assert df.iloc[0]["audio_file"].endswith("normal.wav")

    def test_sidecar_transcript_manifest_is_rejected(self, tmp_path: Path):
        path = tmp_path / "sidecar.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "text_filepath": "0.txt",
                        "files": [{"fname": "0.wav", "speed": 1, "duration": 1.0}],
                        "original_duration": 1.0,
                    }
                ]
            )
        )
        with pytest.raises(ValueError, match="text_filepath"):
            WhisperLibriSpeech.generate(tmp_path / "cache", manifest=path)
