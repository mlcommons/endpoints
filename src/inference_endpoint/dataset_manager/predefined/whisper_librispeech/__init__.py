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

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import getLogger
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import pandas as pd

# Dataset setup only, never a measured request: the benchmark's own HTTP path
# uses the httptools-based ConnectionPool in endpoint_client/.
import requests

from ...dataset import Dataset

logger = getLogger(__name__)

SOURCE_ENV = "WHISPER_LIBRISPEECH_MANIFEST"

MANIFEST_FILENAME = "dev-all-repack.json"

#: Official repackaged dev-all corpus size; a mismatch is warned, not enforced.
MLPERF_SAMPLE_COUNT = 1633

#: Read size for both hashing and streaming; large enough that a multi-GB
#: corpus is not paced by syscall count.
_BLOCK_SIZE = 1 << 20


def _md5_file(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_BLOCK_SIZE), b""):
            md5.update(block)
    return md5.hexdigest()


class _Progress:
    """Thread-safe completed-file counter, logged at roughly every decile."""

    __slots__ = ("_done", "_lock", "_next", "_step", "_total")

    def __init__(self, total: int) -> None:
        self._lock = threading.Lock()
        self._done = 0
        self._total = total
        self._step = max(1, total // 10)
        self._next = self._step

    def tick(self) -> None:
        with self._lock:
            self._done += 1
            if self._done < self._next and self._done != self._total:
                return
            self._next = self._done + self._step
            done = self._done
        logger.info("Corpus download: %d/%d files", done, self._total)


class WhisperLibriSpeech(Dataset, dataset_id="whisper_librispeech"):
    """MLPerf speech2text LibriSpeech corpus (MLCommons-prepared manifest).

    Consumes the ``dev-all-repack.json`` manifest and WAV directory produced by
    the official MLCommons pipeline, which merges LibriSpeech dev-clean and
    dev-other, converts FLAC to 16 kHz WAV, and repackages utterances into
    clips approaching 30 s. ``generate()`` downloads it on first use;
    ``manifest`` / ``$WHISPER_LIBRISPEECH_MANIFEST`` point at a local copy
    instead.

    Building the corpus here is intentionally out of scope. The ~30 s
    repackaging is part of the benchmark definition — a client-side
    reimplementation would change the clip boundaries and therefore the
    reported WER, making the number incomparable to the MLPerf reference.

    One loader serves both phases: the performance phase issues ``audio_file``
    and the accuracy phase scores ``transcript``. ``duration`` is kept in the
    parquet for inspection only — the adapter's ``ColumnFilter`` drops it
    before issuance.
    """

    COLUMN_NAMES = ["audio_file", "transcript", "duration"]

    #: Published ``.uri`` metadata for the preprocessed speech2text corpus. The
    #: sibling ``.md5`` beside it is the artifact's file list.
    R2_DATASET_URI = (
        "https://inference.mlcommons-storage.org/metadata/whisper-dataset.uri"
    )
    R2_DATASET_URI_ENV = "WHISPER_LIBRISPEECH_DATASET_URI"

    #: Every URL built from the metadata — the ``.uri`` itself, its ``.md5``,
    #: and the dataset base URL the ``.uri`` resolves to — is held to this host,
    #: so a mis-set ``$WHISPER_LIBRISPEECH_DATASET_URI`` or a redirected base
    #: URL cannot make the fetch reach somewhere else.
    R2_ALLOWED_HOST = "mlcommons-storage.org"

    #: The only part of the published artifact the benchmark reads. The artifact
    #: also carries the raw LibriSpeech FLACs and the unpacked dev-all WAVs —
    #: an order of magnitude more files than the repackaged clips scored here —
    #: so the published file list is filtered rather than mirrored wholesale.
    CORPUS_PATHS = (MANIFEST_FILENAME, "dev-all-repack/")

    #: Written once every file in the filtered list is present and verified. An
    #: interrupted download can already carry the manifest, so the marker — not
    #: the manifest — is what says the corpus is complete.
    DOWNLOAD_MARKER = ".download-complete"

    R2_DOWNLOAD_WORKERS = 16

    #: ``requests`` (connect, read) seconds. A read timeout bounds a *stalled*
    #: transfer rather than a slow one, so no wall-clock ceiling has to be
    #: guessed for a multi-GB fetch over an unknown link.
    R2_HTTP_TIMEOUT_S = (10, 60)

    R2_FILE_ATTEMPTS = 3

    @classmethod
    def _require_trusted_url(cls, url: str, what: str) -> None:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if parsed.scheme != "https" or not (
            host == cls.R2_ALLOWED_HOST or host.endswith("." + cls.R2_ALLOWED_HOST)
        ):
            raise ValueError(
                f"Refusing to use {what} '{url}': expected https on "
                f"{cls.R2_ALLOWED_HOST}"
            )

    @classmethod
    def _download_from_r2(cls, uri: str, download_dir: Path) -> None:
        """Download the MLCommons-hosted speech2text corpus into ``download_dir``.

        Resolves the dataset's ``.uri`` metadata to a base URL, reads the
        published ``.md5`` file list, and fetches the entries under
        ``CORPUS_PATHS`` concurrently, MD5-verifying each one against that list.
        An already-verified file is left alone, so an interrupted download
        resumes instead of starting over.
        """
        cls._require_trusted_url(uri, "dataset URI")
        checksums_url = uri.removesuffix(".uri") + ".md5"

        with requests.Session() as session:
            base_url = cls._fetch_text(session, uri).strip().rstrip("/")
            cls._require_trusted_url(base_url, "dataset base URL")
            wanted = cls._wanted_files(cls._fetch_text(session, checksums_url))

        if not wanted:
            raise RuntimeError(
                f"The published file list at {checksums_url} has no entries "
                f"under {cls.CORPUS_PATHS}; the hosted artifact must carry the "
                "repackaged manifest and its clips."
            )

        download_dir.mkdir(parents=True, exist_ok=True)
        marker = download_dir / cls.DOWNLOAD_MARKER
        marker.unlink(missing_ok=True)

        logger.info(
            "Fetching %d files from %s into %s", len(wanted), base_url, download_dir
        )
        cls._fetch_all(base_url, download_dir, wanted)
        marker.write_text(f"{base_url}\n{len(wanted)}\n")

    @classmethod
    def _wanted_files(cls, checksums: str) -> list[tuple[str, str]]:
        """Parse the published ``.md5`` list down to ``(path, digest)`` pairs.

        Each line is ``<md5>  <path>`` with the path relative to the dataset
        base URL, optionally ``./``-prefixed.
        """
        wanted = []
        for line in checksums.splitlines():
            digest, _, rest = line.strip().partition(" ")
            path = rest.strip().removeprefix("./")
            if digest and path and path.startswith(cls.CORPUS_PATHS):
                wanted.append((path, digest))
        return wanted

    @classmethod
    def _fetch_all(
        cls,
        base_url: str,
        download_dir: Path,
        wanted: list[tuple[str, str]],
    ) -> None:
        """Fetch the file list concurrently, one session per worker.

        The list is dealt out round-robin so every worker gets a mix of clip
        sizes, and each worker keeps one ``Session`` for the whole slice —
        ``Session`` is not documented as thread-safe, and the connection reuse
        is what makes a thousand small files worth parallelising.
        """
        workers = min(cls.R2_DOWNLOAD_WORKERS, len(wanted))
        progress = _Progress(len(wanted))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    cls._fetch_slice,
                    base_url,
                    download_dir,
                    wanted[i::workers],
                    progress,
                )
                for i in range(workers)
            ]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    @classmethod
    def _fetch_slice(
        cls,
        base_url: str,
        download_dir: Path,
        slice_: list[tuple[str, str]],
        progress: _Progress,
    ) -> None:
        with requests.Session() as session:
            for path, digest in slice_:
                cls._fetch_one(session, base_url, download_dir, path, digest)
                progress.tick()

    @classmethod
    def _fetch_one(
        cls,
        session: requests.Session,
        base_url: str,
        download_dir: Path,
        path: str,
        digest: str,
    ) -> None:
        """Fetch one corpus file unless a verified copy is already on disk."""
        target = cls._safe_target(download_dir, path)
        if target.exists() and _md5_file(target) == digest:
            return

        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"{base_url}/{path}"
        failure: Exception | None = None
        for attempt in range(1, cls.R2_FILE_ATTEMPTS + 1):
            try:
                got = cls._stream_to(session, url, target)
            except requests.RequestException as e:
                failure = e
            else:
                if got == digest:
                    return
                failure = ValueError(f"expected MD5 {digest}, got {got}")
            logger.warning(
                "Corpus file %s failed on attempt %d/%d: %s",
                path,
                attempt,
                cls.R2_FILE_ATTEMPTS,
                failure,
            )
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {failure}") from failure

    @classmethod
    def _stream_to(cls, session: requests.Session, url: str, target: Path) -> str:
        """Stream ``url`` into ``target``, returning the body's MD5 hexdigest.

        Written to a sibling ``.part`` and renamed, so an interrupted transfer
        never leaves a short file that the resume path would have to tell apart
        from a complete one.
        """
        md5 = hashlib.md5()
        part = target.with_name(target.name + ".part")
        with session.get(url, stream=True, timeout=cls.R2_HTTP_TIMEOUT_S) as resp:
            resp.raise_for_status()
            with part.open("wb") as fh:
                for block in resp.iter_content(chunk_size=_BLOCK_SIZE):
                    md5.update(block)
                    fh.write(block)
        part.replace(target)
        return md5.hexdigest()

    @classmethod
    def _fetch_text(cls, session: requests.Session, url: str) -> str:
        resp = session.get(url, timeout=cls.R2_HTTP_TIMEOUT_S)
        resp.raise_for_status()
        return resp.text

    @staticmethod
    def _safe_target(download_dir: Path, path: str) -> Path:
        """Resolve a listed path under the download dir, refusing escapes.

        The paths come from a remote file list, and the prefix filter alone
        would still admit ``dev-all-repack/../..``.
        """
        root = download_dir.resolve()
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Refusing to write corpus entry '{path}' outside {root}")
        return target

    @classmethod
    def generate(
        cls,
        datasets_dir: Path,
        manifest: str | os.PathLike | None = None,
        data_dir: str | os.PathLike | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        force: bool = False,
    ) -> pd.DataFrame:
        """Build (or load the cached) LibriSpeech benchmark dataframe.

        Args:
            datasets_dir: Root cache dir; the parquet is written under
                ``<datasets_dir>/whisper_librispeech/`` and the corpus
                downloads into that directory's ``source/``.
            manifest: An already-prepared ``dev-all-repack.json``, which skips
                the download. Falls back to ``$WHISPER_LIBRISPEECH_MANIFEST``,
                then to the MLCommons R2 download.
            data_dir: Root the manifest's ``fname`` entries resolve against.
                Defaults to the manifest's own directory.
            max_samples: Keep the first N rows. Manifest order is the MLPerf
                sample order, so truncating rather than sampling keeps a smoke
                run's indices aligned with the full corpus.
            seed: Unused; present for the ``get_dataloader`` contract.
            force: Rebuild the cached parquet, reusing an already-downloaded
                corpus. The cache path is keyed on the dataset ID alone, so
                changing ``manifest`` or ``data_dir`` reuses the previous rows
                until this is set.
        """
        dst_path = datasets_dir / cls.DATASET_ID / "whisper_librispeech.parquet"
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists() and not force:
            logger.info("LibriSpeech dataset cached at %s; loading.", dst_path)
            df = pd.read_parquet(dst_path)
        else:
            manifest_path = cls._resolve_manifest(manifest, dst_path.parent / "source")
            df = cls._load_manifest(manifest_path, data_dir)
            df.to_parquet(dst_path, index=False)
            logger.info("Wrote %d LibriSpeech rows to %s", len(df), dst_path)

        if len(df) != MLPERF_SAMPLE_COUNT:
            logger.warning(
                "Manifest has %d samples; the official MLPerf speech2text "
                "corpus has %d. A submission run must use the full corpus.",
                len(df),
                MLPERF_SAMPLE_COUNT,
            )

        if max_samples is not None and max_samples < len(df):
            logger.info("Truncating to the first %d of %d rows", max_samples, len(df))
            df = df.head(max_samples).reset_index(drop=True)
        return df

    @classmethod
    def _resolve_manifest(
        cls,
        manifest: str | os.PathLike | None,
        download_dir: Path,
    ) -> Path:
        """Locate the manifest, downloading the corpus if nothing local is given.

        Precedence: explicit ``manifest`` / ``$WHISPER_LIBRISPEECH_MANIFEST``,
        then a *completed* earlier download under ``download_dir``, then an R2
        fetch. A download that was interrupted after writing the manifest is
        not reused as a corpus — the fetch resumes it instead, so a truncated
        set of clips can never be scored as a full run.
        """
        resolved = manifest or os.environ.get(SOURCE_ENV)
        if resolved:
            manifest_path = Path(resolved)
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"LibriSpeech manifest not found at {manifest_path}"
                )
            if manifest_path.suffix != ".json":
                raise NotImplementedError(
                    f"LibriSpeech source {manifest_path} is not a .json manifest. "
                    f"Only the MLCommons-prepared {MANIFEST_FILENAME} is supported; "
                    "building the corpus from raw LibriSpeech is out of scope."
                )
            return manifest_path

        if (download_dir / cls.DOWNLOAD_MARKER).exists():
            cached = cls._find_manifest(download_dir)
            if cached is not None:
                logger.info("Using the downloaded LibriSpeech corpus at %s", cached)
                return cached

        logger.info(
            "Downloading the MLCommons speech2text corpus (~1.4 GB of audio) to "
            "%s; pass generate_params.manifest to use a local copy instead.",
            download_dir,
        )
        cls._download_from_r2(
            os.environ.get(cls.R2_DATASET_URI_ENV) or cls.R2_DATASET_URI, download_dir
        )
        found = cls._find_manifest(download_dir)
        if found is None:
            raise FileNotFoundError(
                f"R2 download completed but no {MANIFEST_FILENAME} was found "
                f"under {download_dir}; the hosted artifact must carry the "
                "repackaged manifest."
            )
        return found

    @staticmethod
    def _find_manifest(download_dir: Path) -> Path | None:
        """Find the manifest anywhere under a downloaded corpus, or None."""
        if not download_dir.exists():
            return None
        return next(download_dir.rglob(MANIFEST_FILENAME), None)

    @classmethod
    def _load_manifest(
        cls,
        manifest_path: Path,
        data_dir: str | os.PathLike | None,
    ) -> pd.DataFrame:
        root = Path(data_dir) if data_dir is not None else manifest_path.parent
        entries = json.loads(manifest_path.read_text())
        clips = [cls._clip(entry, i) for i, entry in enumerate(entries)]
        strip = cls._fname_strip(root, clips[0]["fname"]) if clips else 0
        rows = [
            cls._row(entry, clip, root, strip)
            for entry, clip in zip(entries, clips, strict=True)
        ]
        return pd.DataFrame(rows, columns=cls.COLUMN_NAMES)

    @staticmethod
    def _fname_strip(root: Path, fname: str) -> int:
        """Leading ``fname`` components to drop so the clip resolves under root.

        The manifest's ``fname`` entries are relative to the upstream pipeline's
        working directory (``./data/dev-all-repack/<clip>.wav``), but the hosted
        artifact unpacks the WAV directory straight into the corpus root, so a
        literal join misses by one component. One probe settles the layout for
        the whole manifest, which is internally consistent. Falls back to 0 so a
        manifest whose audio is not mounted here still produces a dataframe.
        """
        parts = PurePosixPath(fname).parts
        for strip in range(len(parts)):
            if root.joinpath(*parts[strip:]).exists():
                if strip:
                    logger.info(
                        "Manifest fname prefix %s is not present under %s; "
                        "resolving clips as %s",
                        "/".join(parts[:strip]),
                        root,
                        root.joinpath(*parts[strip:]),
                    )
                return strip
        logger.warning(
            "No audio found for the first manifest clip %s under %s. Clip paths "
            "are recorded as-is and will fail at request time; pass "
            "generate_params.data_dir if the audio lives elsewhere.",
            fname,
            root,
        )
        return 0

    @staticmethod
    def _clip(entry: dict, index: int) -> dict:
        """Select one manifest entry's audio clip.

        ``files`` is a list because the upstream schema carries speed-perturbed
        variants. Like the MLPerf reference's manifest reader, pick the
        unperturbed ``speed == 1`` clip — the repackaged dev-all manifest has
        exactly one per sample.
        """
        files = entry.get("files")
        if not files:
            raise ValueError(f"Manifest entry {index} has no 'files' list: {entry}")
        if "transcript" not in entry:
            raise ValueError(
                f"Manifest entry {index} has no inline 'transcript'. The "
                "repackaged dev-all manifest carries transcripts inline; a "
                "'text_filepath' sidecar manifest is not supported."
            )
        clip = next((f for f in files if f.get("speed", 1) == 1), None)
        if clip is None:
            raise ValueError(
                f"Manifest entry {index} has no unperturbed (speed == 1) clip; "
                "speed-perturbed-only manifests are not supported."
            )
        return clip

    @staticmethod
    def _row(entry: dict, clip: dict, root: Path, strip: int) -> dict:
        """Flatten one manifest entry into a benchmark row."""
        parts = PurePosixPath(clip["fname"]).parts
        return {
            "audio_file": str(root.joinpath(*parts[strip:])),
            "transcript": entry["transcript"],
            "duration": float(
                entry.get("original_duration", clip.get("duration", 0.0))
            ),
        }


__all__ = ["WhisperLibriSpeech"]
