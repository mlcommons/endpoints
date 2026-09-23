# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Steady-state detection: the run's side of it.

The metrics aggregator does the detecting. It already timestamps every sample
event and already tokenizes each output for its TPOT trigger, so it rolls
super-passes up as the run happens and writes ``steady_state.json`` /
``steady_state.txt`` beside the report before its final snapshot lands.

This module decides whether that should happen (``collection_plan``, consulted
before the run starts, since collection is not something finalize can ask for
after the fact) and turns the verdict into the compact headline the Report
carries (``verdict_headline``).

The standalone detector remains the way to re-derive a verdict from an archived
run, re-run it with different windows, or check a submission after the fact::

    python -m inference_endpoint.metrics.steady_state_diagnostics <report_dir>

``run_meta.json`` exists for that: it records the super-pass size so a hand
re-run needs no arguments.

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
import math
from pathlib import Path
from typing import Any, NamedTuple

from inference_endpoint.config.schema import BenchmarkConfig, LoadPatternType
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.metrics.report import (
    LatencyBlock,
    LevelShift,
    ShortWindow,
    SteadyStateHeadline,
    TpsBlock,
    WindowBlock,
)
from inference_endpoint.metrics.steady_state_diagnostics import (
    profile_for_load_pattern,
)

logger = logging.getLogger(__name__)


class SteadyStatePlan(NamedTuple):
    """What the aggregator needs to collect and judge a steady-state series."""

    superpass_size: int
    verdict_path: Path
    profile: str


# Artifacts this step owns. report_dir is reusable. A run that produces no
# verdict must leave none behind. run_meta.json is listed separately: a run
# that opted into collection keeps its own fresh copy, so a hand re-run never
# reads a previous run's super-pass size.
_VERDICT_JSON = "steady_state.json"
_VERDICT_ARTIFACTS = (_VERDICT_JSON, "steady_state.txt")
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


def discard_artifacts(report_dir: Path) -> bool:
    """Clear this step's artifacts, reporting whether the directory is now clear.

    Called before the run starts, so the aggregator never writes its verdict
    next to a previous run's. The return value matters: a verdict that survives
    this is one the run cannot overwrite, and publishing it would report a
    previous run's steady window as this one's.
    """
    return _discard(report_dir, _ALL_ARTIFACTS)


def discard_verdict(report_dir: Path) -> None:
    """Withdraw a verdict the run turned out not to deserve.

    run_meta.json survives: a run that aborted or drained incompletely is
    exactly the one worth re-running the standalone detector against by hand.
    """
    _discard(report_dir, _VERDICT_ARTIFACTS)


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

    The run pins the size on the aggregator's command line. run_meta.json lets
    ``python -m inference_endpoint.metrics.steady_state_diagnostics <report_dir>``
    resolve the same size with no arguments.
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


# Kept from each latency block. The histograms are what made the block bulky --
# on a real run they were two thirds of it -- not these scalars, and p99 is what
# a tail-latency reader reaches for first.
_LATENCY_KEYS = ("p50", "p90", "p99", "mean", "count")

# The detector names these without a unit; the report names them with one.
_LATENCY_FIELD = {k: f"{k}_ns" for k in ("p50", "p90", "p99", "mean")}

# A throughput number without its interval is harder to judge, so the batch-means
# CIs ride along with the point estimates.
_TPS_KEYS = ("per_user", "system")
_TPS_INTERVAL_KEYS = ("per_user_ci", "system_ci")


def _interval(value: object) -> tuple[float, float] | None:
    """A two-ended confidence interval, or None if it is not one."""
    if not isinstance(value, list) or len(value) != 2:
        return None
    lo, hi = finite_number(value[0]), finite_number(value[1])
    return None if lo is None or hi is None else (lo, hi)


def finite_number(value: object) -> float | None:
    """``value`` as a float, or None if it cannot be rendered as a number.

    The detector's JSON is arbitrary input, so every number read out of it goes
    through here. bools are rejected because ``True`` would reach a report as
    1.0; NaN and infinity because JSON admits the bare ``NaN`` and ``Infinity``
    tokens and they would render as "nan"/"inf tok/s".
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


# Kept from the detector's window block: all small whole numbers, and between
# them the grounds for trusting the verdict -- which plateau was reported, how
# many there were, and how many earlier ones were rejected as too brief.
_WINDOW_KEYS = (
    "sp_lo",
    "sp_hi",
    "n_samples",
    "n_super_passes",
    "plateau_index",
    "n_plateaus",
    "skipped_short",
)


def _numeric_block(value: object, keys: tuple[str, ...]) -> dict[str, float]:
    """``keys`` from a mapping, keeping only the usable numbers."""
    if not isinstance(value, dict):
        return {}
    return {k: n for k in keys if (n := finite_number(value.get(k))) is not None}


def _latency(value: object) -> LatencyBlock | None:
    """A latency block: percentiles and mean in ns, plus the sample tally."""
    block = _numeric_block(value, _LATENCY_KEYS)
    if not block:
        return None
    count = block.pop("count", None)
    return LatencyBlock(
        **{_LATENCY_FIELD[k]: v for k, v in block.items()},
        count=int(count) if count is not None else None,
    )


def _tps(value: object) -> TpsBlock | None:
    """Throughput point estimates with their intervals."""
    block: dict[str, Any] = dict(_numeric_block(value, _TPS_KEYS))
    if isinstance(value, dict):
        for key in _TPS_INTERVAL_KEYS:
            if (interval := _interval(value.get(key))) is not None:
                block[key] = interval
    return TpsBlock(**block) if block else None


def _window(value: object) -> WindowBlock | None:
    """The window's extent and how it was selected, all whole numbers."""
    block = _numeric_block(value, _WINDOW_KEYS)
    return WindowBlock(**{k: int(v) for k, v in block.items()}) if block else None


def _short_window(value: object) -> ShortWindow | None:
    """Whether the window cleared the min-duration gate, and by how much.

    ``is_short`` alone says only pass or fail; the durations say how close it
    was and which term bound it. The tuning scalars (kstar, cov_b, tau_sp_s,
    l_p90_s) stay in steady_state.json.
    """
    if not isinstance(value, dict):
        return None
    out: dict[str, Any] = {"is_short": bool(value.get("is_short"))}
    for key in ("window_duration_s", "min_duration_s"):
        if (number := finite_number(value.get(key))) is not None:
            out[key] = number
    if isinstance(dominant := value.get("dominant"), str):
        out["dominant"] = dominant
    return ShortWindow(**out)


def _anomaly(value: object) -> LevelShift | None:
    """The detector's level-shift warning, or None if there isn't one.

    Only the fields the report renders are kept; ``plateaus`` is a full
    segmentation table and belongs in steady_state.json.
    """
    if not isinstance(value, dict) or value.get("detected") is not True:
        return None
    return LevelShift(
        detected=True,
        # Checked, not copied: this is interpolated into a line of report.txt,
        # so an unusable value would be printed rather than just stored.
        #
        # Truthiness rather than `is not None` is deliberate here, unlike the
        # other guards in this module. pettitt() searches range(1, n), so a
        # significant change point is always >= 1; it reports 0 only from its
        # too-short-series guard, which also reports significant=False. A zero
        # that reaches here is therefore degenerate and has nothing to say.
        change_point_sp=(
            int(sp) if (sp := finite_number(value.get("change_point_sp"))) else None
        ),
        delta_pct=finite_number(value.get("delta_pct")),
    )


def verdict_headline(
    verdict_path: Path, *, load_pattern: LoadPatternType | None = None
) -> SteadyStateHeadline | None:
    """The compact steady-window summary from a verdict file, for the Report.

    This is the only place the detector's JSON is read, so it is the only place
    the values can be checked. They are arbitrary input -- a file truncated by a
    kill, or simply a different shape after a detector change -- and everything
    downstream formats them into the run's primary artifacts. Every value is
    checked here and anything unusable is dropped, so a consumer does not have
    to ask whether a number is really a number.

    ``load_pattern`` supplies the workload profile's reliability note. That note
    is a property of the profile, not of the verdict file, and the detector only
    emits it on stderr -- where it reaches steady_state.txt but not the report a
    submitter actually reads. Computing it here from the same profile table the
    child uses avoids both the omission and a stderr text contract.

    Best-effort like the rest of this module: an unreadable or malformed
    verdict means the Report carries no steady-state block.
    """
    try:
        blob = json.loads(verdict_path.read_text())
    except (OSError, ValueError):
        logger.warning(
            "Steady-state detection: could not read %s", verdict_path, exc_info=True
        )
        return None
    headline = blob.get("steady_state") if isinstance(blob, dict) else None
    if not isinstance(headline, dict):
        logger.warning("Steady-state detection: %s has no verdict", verdict_path)
        return None

    trend = headline.get("global_trend")
    reason = headline.get("reason")
    profile_name: str | None = None
    caveat: str | None = None
    if load_pattern is not None:
        profile = profile_for_load_pattern(load_pattern.value)
        profile_name = profile.name
        caveat = " ".join(profile.note.split()) or None
    raw_drifting = headline.get("drifting_up")
    context = {
        k: int(n)
        for k in _HEADLINE_CONTEXT
        if (n := finite_number(blob.get(k))) is not None
    }
    return SteadyStateHeadline(
        found=bool(headline.get("found")),
        reason=reason if isinstance(reason, str) else None,
        window=_window(headline.get("window")),
        tps=_tps(headline.get("tps")),
        ttft=_latency(headline.get("ttft")),
        tpot=_latency(headline.get("tpot")),
        short_window=_short_window(headline.get("short_window")),
        # Metric names only. A bare string here would otherwise be iterated
        # character by character downstream and printed as "t, p, o, t".
        drifting_up=tuple(m for m in raw_drifting if isinstance(m, str))
        if isinstance(raw_drifting, list)
        else (),
        anomaly=_anomaly(headline.get("anomaly")),
        global_trend=(
            {k: v for k, v in trend.items() if isinstance(v, str)}
            if isinstance(trend, dict)
            else None
        ),
        profile=profile_name,
        profile_caveat=caveat,
        **context,
    )


def collection_plan(
    config: BenchmarkConfig,
    report_dir: Path,
    *,
    tokenizer_name: str | None,
    dataset_size: int | None,
    accuracy_only: bool,
) -> SteadyStatePlan | None:
    """What the aggregator should collect, or None to skip.

    Decided before the run starts, because the aggregator rolls super-passes up
    as the run happens rather than re-reading the event log afterwards. The
    size is the dataset's sample count: one super-pass is one pass over it.

    ``tokenizer_name`` and ``dataset_size`` come from the run, not the config --
    the first honours the ``model_params.tokenizer_name`` override, the second
    only exists once the dataset is loaded.

    Clearing stale artifacts happens here and happens whatever this returns:
    ``report_dir`` is user-settable and reusable, and a previous run's verdict
    left beside this run's results would read as this run's.
    """
    load_pattern = config.settings.load_pattern.type

    def skip(reason: str) -> None:
        logger.info("Steady-state detection skipped: %s", reason)

    cleared = discard_artifacts(report_dir)

    if not config.settings.steady_state.enabled:
        skip("disabled by configuration")
        return None
    if not cleared:
        # The verdict is published by existence, so one the aggregator cannot
        # overwrite would be read back at finalize as this run's.
        skip(f"a previous run's artifacts could not be cleared from {report_dir}")
        return None
    if accuracy_only:
        skip("accuracy-only runs have no performance phase")
        return None
    if tokenizer_name is None:
        skip("the run resolved no tokenizer")
        return None
    if dataset_size is None or dataset_size <= 0:
        skip("dataset size unknown")
        return None
    if not is_eligible(load_pattern=load_pattern):
        logger.info(
            "Steady-state detection skipped: the detector has no profile for "
            "load_pattern=%s; re-run by hand against %s for a diagnostic pass",
            load_pattern.value,
            report_dir,
        )
        # The message points at a hand re-run, which reads run_meta.json for
        # the super-pass size.
        write_run_meta(report_dir, dataset_size)
        return None

    write_run_meta(report_dir, dataset_size)
    logger.info(
        "Steady-state detection enabled: the metrics aggregator will roll up "
        "super-passes of %d samples (--no-steady-state opts out)",
        dataset_size,
    )
    return SteadyStatePlan(
        superpass_size=dataset_size,
        verdict_path=report_dir / _VERDICT_JSON,
        # Resolved here, where the load pattern is known. The standalone
        # detector resolves the same profile from the run's config, so both
        # judge the verdict against the same CoV bounds and warmup driver.
        profile=profile_for_load_pattern(load_pattern.value).name,
    )


def collected_verdict(report_dir: Path) -> Path | None:
    """The verdict the aggregator wrote for this run, if it wrote one."""
    verdict = report_dir / _VERDICT_JSON
    if verdict.is_file():
        return verdict
    logger.info("Steady-state detection produced no %s", verdict.name)
    return None
