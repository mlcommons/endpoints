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

The download tests replace requests.Session in the loader's own module with a
fake that serves the dataset's .uri metadata, its .md5 file list and the corpus
files out of a dict, so they run without network and only the loader's logic
executes.
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.dataset_manager.predefined.whisper_librispeech import (
    MANIFEST_FILENAME,
    SOURCE_ENV,
    WhisperLibriSpeech,
)

_MODULE = "inference_endpoint.dataset_manager.predefined.whisper_librispeech"


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


_BASE_URL = "https://inference.mlcommons-storage.org/Whisper/dataset"
_CHECKSUMS_URL = WhisperLibriSpeech.R2_DATASET_URI.removesuffix(".uri") + ".md5"

_MANIFEST_BODY = json.dumps(
    [_entry("./data/dev-all-repack/0.wav", "downloaded transcript")]
).encode()
_CLIP = "dev-all-repack/0.wav"

#: The published list also carries the raw LibriSpeech FLACs and the unpacked
#: dev-all WAVs, which the benchmark never reads.
_UNUSED = (
    "LibriSpeech/dev-other/1630/73710/1630-73710-0020.flac",
    "dev-all/6841/88294/6841-88294-0029.wav",
)


def _corpus(**overrides: bytes) -> dict[str, bytes]:
    """The artifact's file bodies, keyed by their path in the published list."""
    files = {MANIFEST_FILENAME: _MANIFEST_BODY, _CLIP: b"RIFF0"}
    files.update(dict.fromkeys(_UNUSED, b"never read"))
    files.update(overrides)
    return files


class _FakeResponse:
    def __init__(self, body: bytes, error: Exception | None = None) -> None:
        self._body = body
        self._error = error
        self.text = body.decode()

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def iter_content(self, chunk_size: int = 1):
        yield self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _FakeSession:
    """Serves the .uri, the .md5 list and the corpus files out of a dict.

    ``served`` keys are paths relative to the dataset base URL. ``corrupt``
    bodies are handed out under a digest computed from the honest body, so the
    loader sees an MD5 mismatch.
    """

    def __init__(
        self,
        served: dict[str, bytes] | None = None,
        corrupt: dict[str, bytes] | None = None,
        fail_once: set[str] | None = None,
    ) -> None:
        files = _corpus() if served is None else served
        digests = {path: hashlib.md5(body).hexdigest() for path, body in files.items()}
        files = {**files, **(corrupt or {})}
        self.bodies = {
            WhisperLibriSpeech.R2_DATASET_URI: f"{_BASE_URL}\n".encode(),
            _CHECKSUMS_URL: "".join(
                f"{digest}  ./{path}\n" for path, digest in digests.items()
            ).encode(),
            **{f"{_BASE_URL}/{path}": body for path, body in files.items()},
        }
        self.fail_once = {f"{_BASE_URL}/{p}" for p in (fail_once or ())}
        self.requested: list[str] = []

    @property
    def fetched_paths(self) -> set[str]:
        """Corpus paths (not metadata URLs) that were actually requested."""
        return {
            url.removeprefix(f"{_BASE_URL}/")
            for url in self.requested
            if url.startswith(f"{_BASE_URL}/")
        }

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        self.requested.append(url)
        if url in self.fail_once:
            self.fail_once.discard(url)
            return _FakeResponse(b"", requests.HTTPError("503 transient"))
        if url not in self.bodies:
            return _FakeResponse(b"", requests.HTTPError(f"404 {url}"))
        return _FakeResponse(self.bodies[url])

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _patch_session(session: _FakeSession):
    """Hand every ``requests.Session()`` in the loader the same fake."""
    return patch(f"{_MODULE}.requests.Session", return_value=session)


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
        with _patch_session(_FakeSession()):
            df = WhisperLibriSpeech.generate(tmp_path / "cache")

        assert df.iloc[0]["transcript"] == "downloaded transcript"
        assert df.iloc[0]["audio_file"] == str(
            tmp_path / "cache/whisper_librispeech/source" / _CLIP
        )

    def test_only_the_repackaged_corpus_is_fetched(
        self, tmp_path: Path, no_local_manifest: None
    ):
        # The published list is an order of magnitude larger than the 1633
        # clips that are issued; mirroring it all is what outran the old
        # wall-clock ceiling on the first run.
        session = _FakeSession()
        with _patch_session(session):
            WhisperLibriSpeech.generate(tmp_path / "cache")

        assert session.fetched_paths == {MANIFEST_FILENAME, _CLIP}

    def test_a_local_manifest_skips_the_download(self, tmp_path: Path, manifest: Path):
        with patch(f"{_MODULE}.requests.Session") as session:
            WhisperLibriSpeech.generate(tmp_path / "cache", manifest=manifest)
            session.assert_not_called()

    def test_force_reuses_the_download_instead_of_refetching(
        self, tmp_path: Path, no_local_manifest: None
    ):
        cache = tmp_path / "cache"
        with _patch_session(_FakeSession()):
            WhisperLibriSpeech.generate(cache)

        with patch(f"{_MODULE}.requests.Session") as session:
            assert len(WhisperLibriSpeech.generate(cache, force=True)) == 1
            session.assert_not_called()

    def test_an_already_verified_file_is_not_refetched(
        self, tmp_path: Path, no_local_manifest: None
    ):
        source = tmp_path / "source"
        with _patch_session(_FakeSession()):
            WhisperLibriSpeech._download_from_r2(
                WhisperLibriSpeech.R2_DATASET_URI, source
            )

        resumed = _FakeSession()
        with _patch_session(resumed):
            WhisperLibriSpeech._download_from_r2(
                WhisperLibriSpeech.R2_DATASET_URI, source
            )
        assert resumed.fetched_paths == set()

    def test_an_interrupted_download_is_not_reused_as_a_corpus(
        self, tmp_path: Path, no_local_manifest: None
    ):
        # The manifest is one of the fetched files, so it can be on disk while
        # most of the clips are still missing.
        cache = tmp_path / "cache"
        source = cache / "whisper_librispeech" / "source"
        source.mkdir(parents=True)
        (source / MANIFEST_FILENAME).write_bytes(_MANIFEST_BODY)

        session = _FakeSession()
        with _patch_session(session):
            WhisperLibriSpeech.generate(cache)

        assert _CLIP in session.fetched_paths
        assert (source / WhisperLibriSpeech.DOWNLOAD_MARKER).exists()

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
        with patch(f"{_MODULE}.requests.Session") as session:
            with pytest.raises(ValueError, match="dataset URI"):
                WhisperLibriSpeech.generate(tmp_path / "cache")
            session.assert_not_called()

    def test_an_untrusted_base_url_is_refused(
        self, tmp_path: Path, no_local_manifest: None
    ):
        session = _FakeSession()
        session.bodies[WhisperLibriSpeech.R2_DATASET_URI] = (
            b"https://evil.example.com/d"
        )
        with _patch_session(session):
            with pytest.raises(ValueError, match="dataset base URL"):
                WhisperLibriSpeech.generate(tmp_path / "cache")
        assert session.fetched_paths == set()

    def test_a_corrupt_file_fails_the_download(
        self, tmp_path: Path, no_local_manifest: None
    ):
        session = _FakeSession(corrupt={_CLIP: b"truncated"})
        with _patch_session(session):
            with pytest.raises(RuntimeError, match="expected MD5"):
                WhisperLibriSpeech.generate(tmp_path / "cache")

    def test_a_corrupt_file_leaves_no_marker_behind(
        self, tmp_path: Path, no_local_manifest: None
    ):
        cache = tmp_path / "cache"
        with _patch_session(_FakeSession(corrupt={_CLIP: b"truncated"})):
            with pytest.raises(RuntimeError):
                WhisperLibriSpeech.generate(cache)
        assert not list(cache.rglob(WhisperLibriSpeech.DOWNLOAD_MARKER))
        assert not list(cache.rglob("*.part"))

    def test_a_transient_http_failure_is_retried(
        self, tmp_path: Path, no_local_manifest: None
    ):
        with _patch_session(_FakeSession(fail_once={_CLIP})):
            df = WhisperLibriSpeech.generate(tmp_path / "cache")
        assert df.iloc[0]["transcript"] == "downloaded transcript"

    def test_a_missing_file_is_reported_with_its_url(
        self, tmp_path: Path, no_local_manifest: None
    ):
        session = _FakeSession()
        del session.bodies[f"{_BASE_URL}/{_CLIP}"]
        with _patch_session(session):
            with pytest.raises(RuntimeError, match=f"Failed to download .*{_CLIP}"):
                WhisperLibriSpeech.generate(tmp_path / "cache")

    def test_a_file_list_without_the_corpus_is_reported(
        self, tmp_path: Path, no_local_manifest: None
    ):
        served = dict.fromkeys(_UNUSED, b"never read")
        with _patch_session(_FakeSession(served=served)):
            with pytest.raises(RuntimeError, match="no entries under"):
                WhisperLibriSpeech.generate(tmp_path / "cache")

    def test_a_corpus_without_the_manifest_is_reported(
        self, tmp_path: Path, no_local_manifest: None
    ):
        with _patch_session(_FakeSession(served={_CLIP: b"RIFF0"})):
            with pytest.raises(FileNotFoundError, match=MANIFEST_FILENAME):
                WhisperLibriSpeech.generate(tmp_path / "cache")

    def test_a_listed_path_escaping_the_download_dir_is_refused(self, tmp_path: Path):
        with pytest.raises(ValueError, match="outside"):
            WhisperLibriSpeech._safe_target(tmp_path, "dev-all-repack/../../etc/passwd")

    def test_the_file_list_keeps_only_the_corpus_paths(self):
        checksums = "\n".join(
            [
                f"{'0' * 32}  ./{MANIFEST_FILENAME}",
                f"{'1' * 32}  ./{_CLIP}",
                f"{'2' * 32}  {_UNUSED[0]}",
                f"{'3' * 32}  {_UNUSED[1]}",
                "",
                "   ",
            ]
        )
        assert WhisperLibriSpeech._wanted_files(checksums) == [
            (MANIFEST_FILENAME, "0" * 32),
            (_CLIP, "1" * 32),
        ]


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
