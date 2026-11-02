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
import os
from logging import getLogger
from pathlib import Path, PurePosixPath

import pandas as pd

from ...dataset import Dataset, download_from_r2

logger = getLogger(__name__)

#: Env var pointing at an already-prepared ``dev-all-repack.json`` manifest,
#: which skips the download.
SOURCE_ENV = "WHISPER_LIBRISPEECH_MANIFEST"

#: Manifest filename inside the MLCommons-prepared corpus.
MANIFEST_FILENAME = "dev-all-repack.json"

#: Sample count of the official repackaged dev-all corpus. Logged as a warning
#: on mismatch, not enforced: a truncated manifest is a valid smoke-test input.
MLPERF_SAMPLE_COUNT = 1633


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

    Columns:
      - ``audio_file`` : absolute path to the clip's WAV, issued by
        ``OpenAITranscriptionsAdapter`` (which reads the bytes in the worker).
      - ``transcript`` : reference text, read by ``WERScorer`` off the
        dataframe as the ground-truth column.
      - ``duration``   : clip length in seconds. Retained in the cached parquet
        for inspection only — nothing in the benchmark reads it, and the
        adapter's ``ColumnFilter`` drops it before issuance.

    One loader serves both phases: the performance phase issues ``audio_file``
    and the accuracy phase scores against ``transcript``.
    """

    COLUMN_NAMES = ["audio_file", "transcript", "duration"]

    #: Published ``.uri`` metadata for the preprocessed speech2text corpus.
    #: Override for staging via ``$WHISPER_LIBRISPEECH_DATASET_URI``.
    R2_DATASET_URI = (
        "https://inference.mlcommons-storage.org/metadata/whisper-dataset.uri"
    )
    R2_DATASET_URI_ENV = "WHISPER_LIBRISPEECH_DATASET_URI"

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
                ``<datasets_dir>/whisper_librispeech/`` and the downloaded
                corpus lands in that directory's ``source/`` subdirectory.
            manifest: An already-prepared ``dev-all-repack.json``, which skips
                the download. Falls back to ``$WHISPER_LIBRISPEECH_MANIFEST``,
                then to the MLCommons R2 download.
            data_dir: Root the manifest's ``fname`` entries are relative to.
                Defaults to the manifest's parent directory. Leading ``fname``
                components that are absent under the root are dropped, so the
                hosted artifact's flattened layout resolves without an override.
            max_samples: If set, keep the first N rows. Manifest order is the
                MLPerf sample order, so truncating (rather than sampling) keeps
                a smoke run's indices aligned with the full corpus.
            seed: Unused; present for the ``get_dataloader`` contract.
            force: Rebuild the cached parquet even if it already exists. An
                already-downloaded corpus is reused — this does not re-fetch
                several GB of audio. The cache path is keyed on the dataset ID
                alone, so changing ``manifest`` or ``data_dir`` reuses the rows
                built from the previous one until this is set.
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
        then an earlier download under ``download_dir``, then a fresh R2 fetch.
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

        cached = cls._find_manifest(download_dir)
        if cached is not None:
            logger.info("Using the downloaded LibriSpeech corpus at %s", cached)
            return cached

        logger.info(
            "Downloading the MLCommons speech2text corpus (several GB of audio) "
            "to %s; pass generate_params.manifest to use a local copy instead.",
            download_dir,
        )
        download_from_r2(
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
        """Find the manifest anywhere under a downloaded corpus, or None.

        The WAV directory sits beside it, so the caller's default ``data_dir``
        (the manifest's parent) is the right root; ``_fname_strip`` absorbs the
        relative prefix the artifact's layout does not reproduce.
        """
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

        The manifest's ``fname`` entries are written relative to the upstream
        pipeline's working directory (``./data/dev-all-repack/<clip>.wav``), but
        the hosted artifact unpacks the WAV directory straight into the corpus
        root, so a literal join misses by one component. Probe the first clip
        against each suffix of its relative path and reuse the offset for the
        whole manifest: the manifest is internally consistent, so one lookup
        settles the layout.

        Falls back to 0 (a literal join) when nothing resolves, so a manifest
        describing audio that is not mounted here still produces a dataframe.
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

        The upstream schema nests the clip under ``files`` (a list, to carry
        speed-perturbed variants). Like the MLPerf reference's manifest reader,
        pick the unperturbed ``speed == 1`` clip; the repackaged dev-all
        manifest carries exactly one such entry per sample.
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
