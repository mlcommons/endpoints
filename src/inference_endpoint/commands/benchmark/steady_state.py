# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-run steady-state detection.

Runs the detector over a finished run's ``events.jsonl`` and leaves
``steady_state.json`` / ``steady_state.txt`` beside the report. The detector runs
out-of-process. Nothing it does can fail finalize.

``finalize_benchmark`` calls this between the metrics drain and the report, so
``verdict_headline`` can put the steady window on the ``Report`` itself -- and
therefore into ``result_summary.json``, ``report.txt``, and the console summary.
The cost of that ordering is that the report waits on the detector, bounded by
``settings.timeouts.steady_state_timeout_s``.

Every input the detector would otherwise guess is pinned on its command line:
tokenizer, super-pass size, and profile. Pinning the tokenizer also keeps the
child off its own model registry, whose entries can request
``trust_remote_code``.

Which workloads the detector may judge is read from ``Profile.supported``.
Enabling a new one is a change in the detector, not here.

There is no model allowlist. Detection is opt-in per run
(``settings.steady_state.enabled``) and the detector has been validated against
a handful of workloads, so a verdict for anything else is the submitter's to
interpret. See ``docs/steady_state_diagnostics.md``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from inference_endpoint.config.schema import BenchmarkConfig, LoadPatternType
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.metrics.steady_state_diagnostics import (
    profile_for_load_pattern,
)

logger = logging.getLogger(__name__)

DETECTOR_MODULE = "inference_endpoint.metrics.steady_state_diagnostics"

_STDERR_LOG_CHARS = 500

# stderr also carries the child's 'wrote <path>' receipt and transformers
# warnings. Only this shape is a reliability note. See format_profile_caveat.
_CAVEAT_PREFIX = "[profile: "

# Artifacts this step owns. report_dir is reusable. A run that produces no
# verdict must leave none behind. run_meta.json is listed separately: a run that
# spawned the detector keeps its own fresh copy even on failure. A stale one
# would feed the wrong super-pass size to a hand re-run.
_VERDICT_ARTIFACTS = ("steady_state.json", "steady_state.txt")
_ALL_ARTIFACTS = (*_VERDICT_ARTIFACTS, "run_meta.json")


def is_eligible(*, load_pattern: LoadPatternType) -> bool:
    """Whether the detector can analyse this load pattern at all.

    Support is the detector's own call (``Profile.supported``). Its agentic and
    offline profiles are unsupported today. An unrecognised load pattern
    resolves to an unsupported profile, not a validated one. Reading that flag
    here keeps the parent's gate and the child's profile selection from
    drifting apart.

    The model is not consulted. Detection is opt-in per run.
    """
    return profile_for_load_pattern(load_pattern.value).supported


def build_command(
    report_dir: Path,
    *,
    tokenizer_name: str,
    dataset_size: int,
    load_pattern: LoadPatternType,
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
        # Pinned, not left to the child's read of config.yaml. An unreadable
        # config.yaml falls back to a profile with no caveat, silently dropping
        # the note this step publishes.
        "--profile",
        profile_for_load_pattern(load_pattern.value).name,
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


def discard_artifacts(report_dir: Path) -> None:
    """Clear this step's artifacts before a run decides whether to produce any.

    ``finalize_benchmark`` calls this up front. Its later paths can exit
    through a ``KeyboardInterrupt`` that never reaches the detection call, so
    cleanup cannot live only at the end.
    """
    _discard(report_dir, _ALL_ARTIFACTS)


def dataset_size_of(dataset: Dataset | None) -> int | None:
    """The loaded dataset's sample count, or None if it cannot be determined.

    ``num_samples`` is implemented by Dataset subclasses. A raise here would
    fail a run whose artifacts are already written. This step promises never to
    do that.
    """
    if dataset is None:
        return None
    try:
        return dataset.num_samples()
    except Exception:  # noqa: BLE001 - diagnostic input; never fail a finished run
        logger.warning(
            "Steady-state detection: dataset size unavailable", exc_info=True
        )
        return None


def write_run_meta(report_dir: Path, dataset_size: int) -> None:
    """Record the super-pass size for a later hand re-run.

    The integration pins this on the command line. run_meta.json lets
    ``python -m inference_endpoint.metrics.steady_state_diagnostics <report_dir>``
    work with no arguments.
    """
    # Cleared first. A failed overwrite would leave the previous run's size in
    # place. A wrong run_meta.json is worse than a missing one.
    _discard(report_dir, ("run_meta.json",))
    _write_best_effort(
        report_dir / "run_meta.json",
        json.dumps({"dataset_size": dataset_size}, indent=2),
    )


# Top-level detector keys worth carrying next to the headline: they state what
# the window was measured over. The rest of the blob (per-super-pass
# trajectories, CoV and drift tables) stays in steady_state.json.
_HEADLINE_CONTEXT = ("superpass_size", "n_super_passes", "n_post_warmup")


def verdict_headline(verdict_path: Path) -> dict[str, Any] | None:
    """The compact steady-window summary from a verdict file, for the Report.

    Best-effort like everything else here: an unreadable or malformed verdict
    means the Report simply carries no steady-state block.
    """
    try:
        blob = json.loads(verdict_path.read_text())
    except (OSError, ValueError):
        logger.warning(
            "Steady-state detection: could not read %s", verdict_path, exc_info=True
        )
        return None
    headline = blob.get("steady_state")
    if not isinstance(headline, dict):
        logger.warning("Steady-state detection: %s has no verdict", verdict_path)
        return None
    context = {k: blob[k] for k in _HEADLINE_CONTEXT if k in blob}
    return {**context, **headline}


def _detector_caveats(stderr: str | None) -> str:
    """The detector's own reliability notes, and nothing else.

    Matches the caveat shape rather than excluding known noise. The child also
    prints a 'wrote <path>' receipt, and transformers writes to the same
    channel. Neither is something the detector said.
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

    ``tokenizer_name`` and ``dataset_size`` come from the run, not the config.
    The first honours the ``model_params.tokenizer_name`` override. The second
    only exists once the dataset is loaded.

    Returns the verdict path on success, ``None`` on skip or failure. Every
    failure path is absorbed. A run that produced valid performance artifacts
    must not be failed by a diagnostic.
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
    if not (report_dir / "events.jsonl").is_file():
        skip(f"no events.jsonl in {report_dir}")
        return None
    if not is_eligible(load_pattern=load_pattern):
        logger.info(
            "Steady-state detection skipped: the detector has no profile for "
            "load_pattern=%s; re-run by hand against %s for a diagnostic pass",
            load_pattern.value,
            report_dir,
        )
        # Every other skip clears run_meta.json. This one writes a fresh copy.
        # The message above points at a hand re-run, which reads run_meta.json
        # for the super-pass size.
        if not _discard(report_dir, _VERDICT_ARTIFACTS):
            # A fresh run_meta.json beside a steady_state.json we could not
            # clear would read as this run's verdict.
            return None
        write_run_meta(report_dir, dataset_size)
        return None
    # Clear the previous run's verdict before spawning. The success check below
    # is an existence test. A child that exits 0 without writing would otherwise
    # republish a stale verdict as this run's.
    if not _discard(report_dir, _VERDICT_ARTIFACTS):
        skip(f"could not clear stale artifacts in {report_dir}")
        return None
    write_run_meta(report_dir, dataset_size)
    verdict = report_dir / "steady_state.json"

    logger.info(
        "Running steady-state detection over %s; this tokenizes every response "
        "(--no-steady-state opts out)",
        report_dir,
    )
    try:
        proc = subprocess.run(
            build_command(
                report_dir,
                tokenizer_name=tokenizer_name,
                dataset_size=dataset_size,
                load_pattern=load_pattern,
            ),
            capture_output=True,
            text=True,
            timeout=config.settings.timeouts.steady_state_timeout_s,
            check=False,
        )
    except KeyboardInterrupt:
        # Absorbed to remove the half-written verdict and keep a traceback out
        # of finalize. The run still exits 130. SigintGovernor recorded the
        # interrupt before this fired.
        logger.warning("Steady-state detection cancelled by interrupt")
        _discard(report_dir, _VERDICT_ARTIFACTS)
        return None
    except Exception:  # noqa: BLE001 - diagnostic; never fail a finished run
        # exc_info keeps a programming error distinguishable from an environment
        # failure. Without it the feature could be silently disabled.
        logger.warning("Steady-state detection skipped", exc_info=True)
        _discard(report_dir, _VERDICT_ARTIFACTS)
        return None

    if proc.returncode != 0:
        logger.warning(
            "Steady-state detection failed (exit %s): %s",
            proc.returncode,
            # A crash puts the exception and message last.
            (proc.stderr or "").strip()[-_STDERR_LOG_CHARS:],
        )
        # The child writes its JSON non-atomically, so a kill can truncate it.
        _discard(report_dir, _VERDICT_ARTIFACTS)
        return None

    caveats = _detector_caveats(proc.stderr)
    if caveats:
        # e.g. poisson's "usually under-saturated" note. Without it,
        # steady_state.json looks more trustworthy than the detector claims.
        logger.info("Steady-state detector caveats: %s", caveats[:_STDERR_LOG_CHARS])

    stdout_text = (proc.stdout or "").rstrip("\n")
    if stdout_text:
        body = f"{stdout_text}\n{caveats}\n" if caveats else f"{stdout_text}\n"
        _write_best_effort(report_dir / "steady_state.txt", body)

    if not verdict.is_file():
        logger.warning("Steady-state detection produced no %s", verdict.name)
        _discard(report_dir, _VERDICT_ARTIFACTS)
        return None

    logger.info("Steady-state detection complete: %s", verdict)
    return verdict
