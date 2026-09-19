# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-run steady-state detection.

Runs the detector over a finished run's ``events.jsonl`` and leaves
``steady_state.json`` / ``steady_state.txt`` beside the report. The detector runs
out-of-process so that nothing it does -- an unbounded tokenization pass, an
unhandled input shape -- can touch the run's primary artifacts or fail finalize.

Every input the detector would otherwise guess is pinned on its command line: the
tokenizer is the one the run itself resolved, and the super-pass size comes from
the loaded dataset. Pinning the tokenizer also keeps the child off its internal
model registry, whose entries can request ``trust_remote_code``.

Which workloads the detector may judge is its own decision, read from
``Profile.supported``, so enabling a new one is a change in the detector rather
than here.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from inference_endpoint.config.schema import BenchmarkConfig, LoadPatternType
from inference_endpoint.metrics.steady_state_diagnostics import (
    profile_for_load_pattern,
)

logger = logging.getLogger(__name__)

DETECTOR_MODULE = "inference_endpoint.metrics.steady_state_diagnostics"

_STDERR_LOG_CHARS = 500

# The detector's stderr also carries bookkeeping ("wrote <path>") and, because the
# child imports transformers, third-party warnings. Only lines in the detector's
# own caveat shape are its reliability notes; see format_profile_caveat.
_CAVEAT_PREFIX = "[profile: "

# Artifacts this step owns. report_dir is user-settable and reusable, so a run
# that produces no fresh verdict must leave none behind: a stale file describes a
# run whose sibling artifacts have already been overwritten. run_meta.json is
# separate because a run that spawns the detector keeps its own fresh copy even
# when the detector then fails -- a stale one would silently feed the wrong
# super-pass size to a later hand re-run.
_VERDICT_ARTIFACTS = ("steady_state.json", "steady_state.txt")
_ALL_ARTIFACTS = (*_VERDICT_ARTIFACTS, "run_meta.json")

# Matched as substrings against model_params.name, which typically carries a repo
# id ("deepseek-ai/DeepSeek-R1") or a cluster path ("/models/gpt-oss-120b").
# This is a policy list, not a capability list: the detector has been validated
# against these workloads. It is deliberately narrower than the detector's own
# tokenizer registry -- "deepseek-r1" rather than "deepseek" so that other
# DeepSeek generations do not silently inherit a verdict nobody checked.
_SUPPORTED_MODEL_SUBSTRINGS = ("gpt-oss", "deepseek-r1", "dsr1")


def is_eligible(*, model_name: str, load_pattern: LoadPatternType) -> bool:
    """Whether this workload is one the detector has been validated against.

    Workload support is the detector's own call (``Profile.supported``): its
    agentic profile is unsupported today, and an unrecognised load pattern
    resolves to an unsupported profile rather than a validated one. Reading that
    flag here means the parent's gate and the child's profile selection cannot
    drift apart.
    """
    if not profile_for_load_pattern(load_pattern.value).supported:
        return False
    lowered = (model_name or "").lower()
    return any(sub in lowered for sub in _SUPPORTED_MODEL_SUBSTRINGS)


def build_command(
    report_dir: Path, *, tokenizer_name: str, dataset_size: int
) -> list[str]:
    return [
        sys.executable,
        "-m",
        DETECTOR_MODULE,
        str(report_dir),
        "--json",
        str(report_dir / "steady_state.json"),
        "--tokenizer",
        tokenizer_name,
        "--dataset-size",
        str(dataset_size),
    ]


def _write_best_effort(path: Path, text: str) -> None:
    try:
        path.write_text(text)
    except OSError as e:
        logger.warning("Steady-state detection could not write %s: %s", path.name, e)


def _discard(report_dir: Path, names: tuple[str, ...]) -> bool:
    """Remove artifacts, reporting whether the directory is now genuinely clear."""
    cleared = True
    for name in names:
        try:
            (report_dir / name).unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Steady-state detection could not remove %s: %s", name, e)
            cleared = False
    return cleared


def write_run_meta(report_dir: Path, dataset_size: int) -> None:
    """Record the super-pass size so the detector can be re-run by hand later.

    The integration pins this on the command line; the sidecar exists so that
    ``python -m inference_endpoint.metrics.steady_state_diagnostics <report_dir>``
    works against the finished directory with no arguments.
    """
    # Cleared first: a failed overwrite would otherwise leave the previous run's
    # size in place, and a wrong sidecar is worse than a missing one.
    _discard(report_dir, ("run_meta.json",))
    _write_best_effort(
        report_dir / "run_meta.json",
        json.dumps({"dataset_size": dataset_size}, indent=2),
    )


def _detector_caveats(stderr: str | None) -> str:
    """The detector's own reliability notes, and nothing else.

    Matches the detector's caveat shape rather than excluding known noise: the
    child also prints a 'wrote <path>' receipt, and anything transformers emits
    lands on the same channel. Neither is something the detector said.
    """
    lines = [
        line.strip()
        for line in (stderr or "").splitlines()
        if line.strip().startswith(_CAVEAT_PREFIX)
    ]
    return "\n".join(lines)


def detect_steady_state(
    report_dir: Path,
    config: BenchmarkConfig,
    *,
    tokenizer_name: str | None,
    dataset_size: int | None,
) -> Path | None:
    """Run the detector for a finished run, best-effort.

    ``tokenizer_name`` and ``dataset_size`` are resolved by the run rather than
    read from the config: the former honours the ``model_params.tokenizer_name``
    override, and the latter only exists once the dataset is loaded.

    Returns the verdict path on success, ``None`` when the run was skipped or the
    detector failed. Every failure path is absorbed: this is additive reporting,
    and a run that produced valid performance artifacts must not be failed by a
    diagnostic.
    """
    load_pattern = config.settings.load_pattern.type

    def skip(reason: str) -> None:
        logger.info("Steady-state detection skipped: %s", reason)
        _discard(report_dir, _ALL_ARTIFACTS)

    if not config.settings.steady_state.enabled:
        skip("disabled by configuration")
        return None
    if tokenizer_name is None:
        skip("the run resolved no tokenizer")
        return None
    if dataset_size is None:
        skip("dataset size unknown")
        return None
    if not is_eligible(model_name=config.model_params.name, load_pattern=load_pattern):
        logger.info(
            "Steady-state detection skipped: not a validated workload "
            "(model=%s, load_pattern=%s); re-run by hand against %s for a "
            "diagnostic pass",
            config.model_params.name,
            load_pattern.value,
            report_dir,
        )
        # Unlike the other skips, this run's metadata is known good -- it is the
        # workload that is unvalidated, not the inputs. Leave the sidecar so the
        # hand re-run suggested above works with no arguments.
        _discard(report_dir, _VERDICT_ARTIFACTS)
        write_run_meta(report_dir, dataset_size)
        return None
    if not (report_dir / "events.jsonl").is_file():
        skip(f"no events.jsonl in {report_dir}")
        return None

    # Clear the previous run's verdict before spawning: the success check below is
    # an existence test, so a child that exits 0 without writing would otherwise
    # republish a stale verdict as this run's.
    if not _discard(report_dir, _VERDICT_ARTIFACTS):
        # The success check below is an existence test, so an uncleared stale
        # verdict would be republished as this run's result.
        logger.warning(
            "Steady-state detection skipped: could not clear stale artifacts in %s",
            report_dir,
        )
        return None
    write_run_meta(report_dir, dataset_size)
    verdict = report_dir / "steady_state.json"

    try:
        proc = subprocess.run(
            build_command(
                report_dir, tokenizer_name=tokenizer_name, dataset_size=dataset_size
            ),
            capture_output=True,
            text=True,
            timeout=config.settings.timeouts.steady_state_timeout_s,
            check=False,
        )
    except KeyboardInterrupt:
        # The run's artifacts are already on disk; a ^C aimed at a slow
        # diagnostic must not downgrade the run to interrupted.
        logger.warning("Steady-state detection cancelled by interrupt")
        _discard(report_dir, _VERDICT_ARTIFACTS)
        return None
    except Exception:  # noqa: BLE001 - diagnostic; never fail a finished run
        # exc_info so a programming error here stays distinguishable from an
        # environment failure instead of silently disabling the feature.
        logger.warning("Steady-state detection skipped", exc_info=True)
        _discard(report_dir, _VERDICT_ARTIFACTS)
        return None

    if proc.returncode != 0:
        logger.warning(
            "Steady-state detection failed (exit %s): %s",
            proc.returncode,
            # The useful part of a crash -- the exception and message -- is last.
            (proc.stderr or "").strip()[-_STDERR_LOG_CHARS:],
        )
        # The child writes its JSON non-atomically, so a kill can truncate it.
        _discard(report_dir, _VERDICT_ARTIFACTS)
        return None

    caveats = _detector_caveats(proc.stderr)
    if caveats:
        # e.g. "offline: system TPS is unreliable" -- without this the numbers in
        # steady_state.json look more trustworthy than the detector claims.
        logger.info("Steady-state detector caveats: %s", caveats[:_STDERR_LOG_CHARS])

    stdout_text = (proc.stdout or "").rstrip("\n")
    if stdout_text or caveats:
        body = f"{stdout_text}\n{caveats}\n" if caveats else f"{stdout_text}\n"
        _write_best_effort(report_dir / "steady_state.txt", body)

    if not verdict.is_file():
        logger.warning("Steady-state detection produced no %s", verdict.name)
        _discard(report_dir, _VERDICT_ARTIFACTS)
        return None

    logger.info("Steady-state detection complete: %s", verdict)
    return verdict
