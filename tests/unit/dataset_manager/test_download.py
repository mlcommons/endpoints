# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the guarded MLCommons R2 download utility."""

import hashlib
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import inference_endpoint.dataset_manager.download as download_module
import pytest
from inference_endpoint.dataset_manager.download import (
    DEFAULT_R2_DOWNLOADER_COMMIT,
    download_r2_artifact,
    verify_sha256,
)

pytestmark = pytest.mark.unit
_MODULE = "inference_endpoint.dataset_manager.download"
_URI = "https://inference.mlcommons-storage.org/data.uri"
_COMMIT = "a" * 40


def _response() -> MagicMock:
    response = MagicMock()
    response.content = b"#!/bin/sh\nexit 0\n"
    return response


def _run_result(returncode: int = 0, stderr: str = "") -> MagicMock:
    result = MagicMock()
    result.returncode = returncode
    result.stderr = stderr
    return result


def test_verify_sha256_incrementally(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    payload = b"0123456789"
    path.write_bytes(payload)
    verify_sha256(path, hashlib.sha256(payload).hexdigest(), chunk_size=3)


def test_verify_sha256_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_sha256(path, "0" * 64)


@pytest.mark.parametrize(
    "uri",
    [
        "http://inference.mlcommons-storage.org/data.uri",
        "https://evil.example/data.uri",
        "file:///tmp/data.uri",
        "https://mlcommons-storage.org.evil.example/data.uri",
    ],
)
def test_rejects_untrusted_uri_before_network(tmp_path: Path, monkeypatch, uri: str):
    get = MagicMock()
    monkeypatch.setattr(f"{_MODULE}.requests.get", get)
    with pytest.raises(ValueError, match="untrusted URI"):
        download_r2_artifact(uri, tmp_path, "artifact.bin", _COMMIT)
    get.assert_not_called()


def test_rejects_untrusted_downloader_url_before_network(
    tmp_path: Path, monkeypatch
) -> None:
    get = MagicMock()
    monkeypatch.setattr(
        download_module, "_DOWNLOADER_URL_TEMPLATE", "http://evil/{commit}"
    )
    monkeypatch.setattr(download_module.requests, "get", get)

    with pytest.raises(ValueError, match="untrusted URI"):
        download_r2_artifact(_URI, tmp_path, "artifact.bin", _COMMIT)

    get.assert_not_called()


def test_timeout_cleans_temporary_script(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(f"{_MODULE}.requests.get", lambda *args, **kwargs: _response())

    def timeout(*args, **kwargs):
        assert any(
            path.name.startswith(".mlc-r2-downloader-") for path in tmp_path.iterdir()
        )
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(f"{_MODULE}.subprocess.run", timeout)
    with pytest.raises(RuntimeError, match="timed out"):
        download_r2_artifact(_URI, tmp_path, "artifact.bin", _COMMIT, timeout_s=2)
    assert not list(tmp_path.glob(".mlc-r2-downloader-*.sh"))


def test_nonzero_exit_cleans_script(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(f"{_MODULE}.requests.get", lambda *args, **kwargs: _response())
    monkeypatch.setattr(
        f"{_MODULE}.subprocess.run",
        lambda *args, **kwargs: _run_result(3, "boom"),
    )
    with pytest.raises(RuntimeError, match="code 3"):
        download_r2_artifact(_URI, tmp_path, "artifact.bin", _COMMIT)
    assert not list(tmp_path.glob(".mlc-r2-downloader-*.sh"))


def test_missing_artifact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(f"{_MODULE}.requests.get", lambda *args, **kwargs: _response())
    monkeypatch.setattr(
        f"{_MODULE}.subprocess.run", lambda *args, **kwargs: _run_result()
    )
    with pytest.raises(FileNotFoundError, match="artifact.bin"):
        download_r2_artifact(_URI, tmp_path, "artifact.bin", _COMMIT)


def test_relocates_exact_nested_artifact(tmp_path: Path, monkeypatch) -> None:
    payload = b"nested artifact"
    monkeypatch.setattr(f"{_MODULE}.requests.get", lambda *args, **kwargs: _response())

    def run(*args, **kwargs):
        nested = tmp_path / "manifest-directory" / "artifact.bin"
        nested.parent.mkdir()
        nested.write_bytes(payload)
        return _run_result()

    monkeypatch.setattr(f"{_MODULE}.subprocess.run", run)
    result = download_r2_artifact(
        _URI,
        tmp_path,
        "artifact.bin",
        _COMMIT,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert result == tmp_path / "artifact.bin"
    assert result.read_bytes() == payload


def test_ambiguous_artifact_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(f"{_MODULE}.requests.get", lambda *args, **kwargs: _response())

    def run(*args, **kwargs):
        for directory in ("one", "two"):
            path = tmp_path / directory / "artifact.bin"
            path.parent.mkdir()
            path.write_bytes(directory.encode())
        return _run_result()

    monkeypatch.setattr(f"{_MODULE}.subprocess.run", run)
    with pytest.raises(RuntimeError, match="ambiguous"):
        download_r2_artifact(_URI, tmp_path, "artifact.bin", _COMMIT)


def test_checksum_mismatch_cleans_script(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(f"{_MODULE}.requests.get", lambda *args, **kwargs: _response())

    def run(*args, **kwargs):
        (tmp_path / "artifact.bin").write_bytes(b"wrong")
        return _run_result()

    monkeypatch.setattr(f"{_MODULE}.subprocess.run", run)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_r2_artifact(
            _URI, tmp_path, "artifact.bin", _COMMIT, expected_sha256="0" * 64
        )
    assert not list(tmp_path.glob(".mlc-r2-downloader-*.sh"))


def test_success_uses_pinned_script_and_timeout(tmp_path: Path, monkeypatch) -> None:
    payload = b"success"
    response = _response()
    get = MagicMock(return_value=response)
    monkeypatch.setattr(f"{_MODULE}.requests.get", get)
    run = MagicMock(
        side_effect=lambda *args, **kwargs: (
            (tmp_path / "artifact.bin").write_bytes(payload),
            _run_result(),
        )[1]
    )
    monkeypatch.setattr(f"{_MODULE}.subprocess.run", run)

    result = download_r2_artifact(
        _URI,
        tmp_path,
        "artifact.bin",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        timeout_s=17,
    )
    assert result == tmp_path / "artifact.bin"
    get.assert_called_once_with(
        "https://raw.githubusercontent.com/mlcommons/r2-downloader/"
        f"{DEFAULT_R2_DOWNLOADER_COMMIT}/"
        "mlc-r2-downloader.sh",
        timeout=30,
    )
    assert run.call_args.kwargs["timeout"] == 17
    assert not list(tmp_path.glob(".mlc-r2-downloader-*.sh"))
