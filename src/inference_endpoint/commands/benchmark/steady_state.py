# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-run steady-state detection.

Runs the detector over a finished run's ``events.jsonl`` and leaves
``steady_state.json`` / ``steady_state.txt`` beside the report. The detector runs
out-of-process so that nothing it does -- a missing tokenizer, an unbounded
tokenization pass, an unhandled input shape -- can touch the run's primary
artifacts or fail finalize.

The detector resolves its own tokenizer and workload profile from the run
directory's ``config.yaml`` and ``run_meta.json`` sidecars, so the only inputs it
needs from here are which directory to read and whether this run qualifies.
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

# Matched as substrings against model_params.name, which typically carries a repo
# id ("deepseek-ai/DeepSeek-R1") or a cluster path ("/models/gpt-oss-120b").
# "deepseek-r1" rather than "deepseek" so that other DeepSeek generations do not
# silently inherit a detection pass that was only validated for R1.
_SUPPORTED_MODEL_SUBSTRINGS = ("gpt-oss", "deepseek-r1", "dsr1")


def should_run(*, model_name: str, enabled: bool, is_agentic: bool) -> bool:
    """Whether this run earns a steady-state pass.

    Agentic runs are excluded because the detector's agentic profile is marked
    unsupported -- it would emit a verdict that must not be trusted.
    """
    if not enabled or is_agentic:
        return False
    lowered = (model_name or "").lower()
    return any(sub in lowered for sub in _SUPPORTED_MODEL_SUBSTRINGS)


def write_run_meta(report_dir: Path, dataset_size: int | None) -> None:
    """Write the detector's ``run_meta.json`` sidecar.

    Super-pass size is the one input the detector needs that the run's
    ``config.yaml`` does not already carry, and it refuses to run without it.
    Recording it in the run directory also lets the detector be re-run by hand
    against that directory later with no arguments.
    """
    if dataset_size is None:
        return
    path = report_dir / "run_meta.json"
    path.write_text(json.dumps({"dataset_size": dataset_size}, indent=2))
    logger.debug("Wrote %s", path)


def build_command(report_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        DETECTOR_MODULE,
        str(report_dir),
        "--json",
        str(report_dir / "steady_state.json"),
    ]


def run_for_report(
    report_dir: Path,
    *,
    model_name: str,
    enabled: bool,
    is_agentic: bool,
    timeout_s: float | None,
) -> None:
    """Run the detector for a finished run, best-effort.

    Every failure path is absorbed: this is additive reporting, and a run that
    produced valid performance artifacts must not be failed by a diagnostic.
    """
    if not should_run(model_name=model_name, enabled=enabled, is_agentic=is_agentic):
        logger.debug("Steady-state detection not applicable (model=%s)", model_name)
        return

    if not (report_dir / "events.jsonl").is_file():
        logger.info("Steady-state detection skipped: no events.jsonl in %s", report_dir)
        return

    try:
        proc = subprocess.run(
            build_command(report_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Steady-state detection skipped: %s", e)
        return

    if proc.stdout:
        (report_dir / "steady_state.txt").write_text(proc.stdout)
    if proc.returncode != 0:
        logger.warning(
            "Steady-state detection failed (exit %s): %s",
            proc.returncode,
            (proc.stderr or "").strip()[:500],
        )
    else:
        logger.info("Steady-state detection wrote %s", report_dir / "steady_state.json")


def _perf_dataset_is_agentic(config: BenchmarkConfig) -> bool:
    perf = next(
        (d for d in config.datasets if d.type == DatasetType.PERFORMANCE),
        None,
    )
    return perf is not None and perf.agentic_inference is not None


def run_for_context(
    report_dir: Path,
    config: BenchmarkConfig,
    dataset_size: int | None,
) -> None:
    """Write the detector's sidecar and run it for a finished run.

    ``dataset_size`` comes from the loaded dataset rather than the config, which
    does not carry a resolved sample count.
    """
    write_run_meta(report_dir, dataset_size)
    run_for_report(
        report_dir,
        model_name=config.model_params.name,
        enabled=config.settings.steady_state.enabled,
        is_agentic=_perf_dataset_is_agentic(config),
        timeout_s=config.settings.timeouts.steady_state_timeout_s,
    )
