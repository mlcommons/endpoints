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

from inference_endpoint.config.schema import BenchmarkConfig
from inference_endpoint.metrics.steady_state_diagnostics import profile_for_load_pattern

logger = logging.getLogger(__name__)

DETECTOR_MODULE = "inference_endpoint.metrics.steady_state_diagnostics"

_STDERR_LOG_CHARS = 500

# The child prints this receipt to stderr on every --json run. It is bookkeeping,
# not a caveat, and must not be reported as one.
_RECEIPT_PREFIX = "wrote "

# Matched as substrings against model_params.name, which typically carries a repo
# id ("deepseek-ai/DeepSeek-R1") or a cluster path ("/models/gpt-oss-120b").
# This is a policy list, not a capability list: the detector has been validated
# against these workloads. It is deliberately narrower than the detector's own
# tokenizer registry -- "deepseek-r1" rather than "deepseek" so that other
# DeepSeek generations do not silently inherit a verdict nobody checked.
_SUPPORTED_MODEL_SUBSTRINGS = ("gpt-oss", "deepseek-r1", "dsr1")


def should_run(*, model_name: str, enabled: bool, load_pattern: str) -> bool:
    """Whether this workload is one the detector may judge.

    Workload support is the detector's own call (``Profile.supported``): its
    agentic profile is unsupported today and would otherwise emit a verdict that
    must not be trusted. Reading that flag here means the parent's gate and the
    child's profile selection cannot drift apart.
    """
    if not enabled or not profile_for_load_pattern(load_pattern).supported:
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


def _discard_best_effort(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as e:
        logger.warning("Steady-state detection could not remove %s: %s", path.name, e)


def write_run_meta(report_dir: Path, dataset_size: int) -> None:
    """Record the super-pass size so the detector can be re-run by hand later.

    The integration pins this on the command line; the sidecar exists so that
    ``python -m inference_endpoint.metrics.steady_state_diagnostics <report_dir>``
    works against the finished directory with no arguments.
    """
    _write_best_effort(
        report_dir / "run_meta.json",
        json.dumps({"dataset_size": dataset_size}, indent=2),
    )


def _detector_caveats(stderr: str | None) -> str:
    """The detector's reliability notes, minus its own 'wrote <path>' receipt."""
    lines = [
        line
        for line in (stderr or "").splitlines()
        if line.strip() and not line.strip().startswith(_RECEIPT_PREFIX)
    ]
    return "\n".join(lines).strip()


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
    if tokenizer_name is None:
        logger.info("Steady-state detection skipped: the run resolved no tokenizer")
        return None

    if dataset_size is None:
        logger.info("Steady-state detection skipped: dataset size unknown")
        return None

    load_pattern = config.settings.load_pattern.type.value
    if not should_run(
        model_name=config.model_params.name,
        enabled=config.settings.steady_state.enabled,
        load_pattern=load_pattern,
    ):
        logger.info(
            "Steady-state detection not applicable (model=%s, load_pattern=%s)",
            config.model_params.name,
            load_pattern,
        )
        return None

    if not (report_dir / "events.jsonl").is_file():
        logger.info("Steady-state detection skipped: no events.jsonl in %s", report_dir)
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
        _discard_best_effort(verdict)
        return None
    except Exception as e:  # noqa: BLE001 - diagnostic; never fail a finished run
        logger.warning("Steady-state detection skipped: %s", e)
        _discard_best_effort(verdict)
        return None

    caveats = _detector_caveats(proc.stderr)
    if proc.returncode != 0:
        logger.warning(
            "Steady-state detection failed (exit %s): %s",
            proc.returncode,
            # The useful part of a crash -- the exception and message -- is last.
            (proc.stderr or "").strip()[-_STDERR_LOG_CHARS:],
        )
        # The child writes its JSON non-atomically, so a kill can truncate it.
        _discard_best_effort(verdict)
        return None

    if caveats:
        # e.g. "offline: system TPS is unreliable" -- without this the numbers in
        # steady_state.json look more trustworthy than the detector claims.
        logger.info("Steady-state detector caveats: %s", caveats[:_STDERR_LOG_CHARS])

    stdout_text = (proc.stdout or "").rstrip("\n")
    if stdout_text or caveats:
        body = f"{stdout_text}\n{caveats}\n" if caveats else f"{stdout_text}\n"
        _write_best_effort(report_dir / "steady_state.txt", body)

    logger.info("Steady-state detection complete: %s", verdict)
    return verdict
