# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-run steady-state detection.

Runs the detector over a finished run's ``events.jsonl`` and leaves
``steady_state.json`` / ``steady_state.txt`` beside the report. The detector runs
out-of-process so that nothing it does -- an unbounded tokenization pass, an
unhandled input shape -- can touch the run's primary artifacts or fail finalize.

Every input the detector cannot safely guess is pinned here rather than left to
its own auto-detection: the tokenizer is the one the run itself resolved, and the
super-pass size is written to ``run_meta.json``. Pinning the tokenizer also keeps
the child off its internal model registry, whose entries can request
``trust_remote_code``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from inference_endpoint.config.schema import BenchmarkConfig, DatasetType

logger = logging.getLogger(__name__)

DETECTOR_MODULE = "inference_endpoint.metrics.steady_state_diagnostics"

_STDERR_LOG_CHARS = 500

# Matched as substrings against model_params.name, which typically carries a repo
# id ("deepseek-ai/DeepSeek-R1") or a cluster path ("/models/gpt-oss-120b").
# This is a policy list, not a capability list: the detector has been validated
# against these workloads. It is deliberately narrower than the detector's own
# tokenizer registry -- "deepseek-r1" rather than "deepseek" so that other
# DeepSeek generations do not silently inherit a verdict nobody checked.
_SUPPORTED_MODEL_SUBSTRINGS = ("gpt-oss", "deepseek-r1", "dsr1")


def should_run(
    *,
    model_name: str,
    enabled: bool,
    is_agentic: bool,
    tokenizer_name: str | None,
) -> bool:
    """Whether this run earns a steady-state pass.

    Agentic runs are excluded because the detector's agentic profile is marked
    unsupported -- it would emit a verdict that must not be trusted. A run whose
    tokenizer could not be resolved is excluded because the detector would then
    fall back to guessing one from the model name.
    """
    if not enabled or is_agentic or tokenizer_name is None:
        return False
    lowered = (model_name or "").lower()
    return any(sub in lowered for sub in _SUPPORTED_MODEL_SUBSTRINGS)


def build_command(report_dir: Path, *, tokenizer_name: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        DETECTOR_MODULE,
        str(report_dir),
        "--json",
        str(report_dir / "steady_state.json"),
        "--tokenizer",
        tokenizer_name,
    ]


def _write_text(path: Path, text: str) -> None:
    """Write an optional artifact without ever failing the run."""
    try:
        path.write_text(text)
    except OSError as e:
        logger.warning("Steady-state detection could not write %s: %s", path.name, e)


def write_run_meta(report_dir: Path, dataset_size: int) -> None:
    """Write the detector's ``run_meta.json`` sidecar.

    Super-pass size is the one input the detector needs that the run's
    ``config.yaml`` does not already carry, and it refuses to run without it.
    Recording it in the run directory also lets the detector be re-run by hand
    against that directory later with no arguments.
    """
    _write_text(
        report_dir / "run_meta.json",
        json.dumps({"dataset_size": dataset_size}, indent=2),
    )


def _perf_dataset_is_agentic(config: BenchmarkConfig) -> bool:
    return any(
        d.agentic_inference is not None
        for d in config.datasets
        if d.type == DatasetType.PERFORMANCE
    )


def run_for_context(
    report_dir: Path,
    config: BenchmarkConfig,
    *,
    tokenizer_name: str | None,
    dataset_size: int | None,
) -> None:
    """Run the detector for a finished run, best-effort.

    ``tokenizer_name`` and ``dataset_size`` are resolved by the run rather than
    read from the config: the former honours the ``model_params.tokenizer_name``
    override, and the latter only exists once the dataset is loaded.

    Every failure path is absorbed: this is additive reporting, and a run that
    produced valid performance artifacts must not be failed by a diagnostic.
    """
    if not should_run(
        model_name=config.model_params.name,
        enabled=config.settings.steady_state.enabled,
        is_agentic=_perf_dataset_is_agentic(config),
        tokenizer_name=tokenizer_name,
    ):
        logger.debug(
            "Steady-state detection not applicable (model=%s)",
            config.model_params.name,
        )
        return

    if dataset_size is None:
        logger.info("Steady-state detection skipped: dataset size unknown")
        return

    if not (report_dir / "events.jsonl").is_file():
        logger.info("Steady-state detection skipped: no events.jsonl in %s", report_dir)
        return

    write_run_meta(report_dir, dataset_size)

    # tokenizer_name is not None here -- should_run rejected that case.
    assert tokenizer_name is not None
    try:
        proc = subprocess.run(
            build_command(report_dir, tokenizer_name=tokenizer_name),
            capture_output=True,
            text=True,
            timeout=config.settings.timeouts.steady_state_timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Steady-state detection skipped: %s", e)
        return
    except KeyboardInterrupt:
        # The run's artifacts are already on disk; a ^C aimed at a slow
        # diagnostic must not downgrade the run to interrupted.
        logger.warning("Steady-state detection cancelled by interrupt")
        return

    notes = (proc.stderr or "").strip()
    if proc.returncode != 0:
        logger.warning(
            "Steady-state detection failed (exit %s): %s",
            proc.returncode,
            notes[:_STDERR_LOG_CHARS],
        )
        return

    if notes:
        # The detector reports profile caveats (e.g. "system TPS is unreliable"
        # for offline runs) on stderr. Dropping them would leave the numbers in
        # steady_state.json looking more trustworthy than the detector claims.
        logger.info("Steady-state detector notes: %s", notes[:_STDERR_LOG_CHARS])

    report = (proc.stdout or "").rstrip("\n")
    if report or notes:
        body = f"{report}\n{notes}\n" if notes else f"{report}\n"
        _write_text(report_dir / "steady_state.txt", body)
    logger.info("Steady-state detection wrote %s", report_dir / "steady_state.json")
