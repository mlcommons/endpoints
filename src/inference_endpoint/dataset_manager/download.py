# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, guarded helpers for downloading MLCommons R2 artifacts."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests

_DOWNLOADER_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/mlcommons/r2-downloader/"
    "{commit}/mlc-r2-downloader.sh"
)
_DOWNLOADER_HOST = "raw.githubusercontent.com"
_REQUEST_TIMEOUT_S = 30
_DEFAULT_CHUNK_SIZE = 1 << 20
DEFAULT_R2_DOWNLOADER_COMMIT = "27da4421877f2831eeb615b43ee5098c4b70be7e"
DEFAULT_R2_ALLOWED_HOST = "mlcommons-storage.org"


def _validate_https_host(uri: str, allowed_host: str) -> None:
    parsed = urlparse(uri)
    host = (parsed.hostname or "").lower()
    allowed = allowed_host.lower().rstrip(".")
    if (
        parsed.scheme != "https"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or not (host == allowed or host.endswith(f".{allowed}"))
    ):
        raise ValueError(
            f"Refusing to download dataset from untrusted URI {uri!r}: "
            f"expected https on {allowed_host}"
        )


def verify_sha256(
    path: Path, expected: str, chunk_size: int = _DEFAULT_CHUNK_SIZE
) -> None:
    """Verify a file's SHA-256 digest without loading it all into memory."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected.lower():
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}"
        )


def _artifact_target(destination_dir: Path, artifact_name: str) -> Path:
    name = Path(artifact_name)
    if not artifact_name or name.name != artifact_name or artifact_name in {".", ".."}:
        raise ValueError(f"artifact_name must be a plain filename: {artifact_name!r}")
    return destination_dir / artifact_name


def download_r2_artifact(
    uri: str,
    destination_dir: Path,
    artifact_name: str,
    downloader_commit: str = DEFAULT_R2_DOWNLOADER_COMMIT,
    expected_sha256: str | None = None,
    timeout_s: float = 1800,
    allowed_host: str = DEFAULT_R2_ALLOWED_HOST,
) -> Path:
    """Download one exact artifact through a pinned MLCommons R2 script."""
    _validate_https_host(uri, allowed_host)
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = _artifact_target(destination_dir, artifact_name)
    target.unlink(missing_ok=True)
    downloader_url = _DOWNLOADER_URL_TEMPLATE.format(commit=downloader_commit)
    _validate_https_host(downloader_url, _DOWNLOADER_HOST)

    script_fd, script_name = tempfile.mkstemp(
        prefix=".mlc-r2-downloader-", suffix=".sh", dir=destination_dir
    )
    os.close(script_fd)
    script_path = Path(script_name)
    try:
        response = requests.get(downloader_url, timeout=_REQUEST_TIMEOUT_S)
        response.raise_for_status()
        script_path.write_bytes(response.content)
        script_path.chmod(0o755)

        try:
            result = subprocess.run(
                [
                    "bash",
                    str(script_path.resolve()),
                    "-d",
                    str(destination_dir.resolve()),
                    uri,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"R2 downloader timed out after {timeout_s} seconds"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"R2 downloader failed with code {result.returncode}: "
                f"{result.stderr}"
            )

        candidates = [
            path
            for path in destination_dir.rglob("*")
            if path.is_file() and path.name == artifact_name and path != script_path
        ]
        if len(candidates) != 1:
            if not candidates:
                raise FileNotFoundError(
                    f"R2 download completed but exact artifact {artifact_name!r} "
                    f"is missing under {destination_dir}"
                )
            raise RuntimeError(
                f"R2 download produced ambiguous artifact {artifact_name!r}: "
                f"{len(candidates)} files found under {destination_dir}"
            )

        found = candidates[0]
        if found != target:
            target.unlink(missing_ok=True)
            found.replace(target)
        if expected_sha256 is not None:
            verify_sha256(target, expected_sha256)
        return target
    finally:
        script_path.unlink(missing_ok=True)
