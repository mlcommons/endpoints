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

"""Standardized result plots for Edge-Agentic benchmark runs.

Consumes a run's report directory (the artifacts written by a benchmark run:
``results.json`` for accuracy, ``scores.json`` for the agentic performance run,
``result_summary.json`` for latency/throughput distributions) and renders a fixed
set of PNGs for leaderboard / report use:

* accuracy — overall + normalized vs the ruleset gate, per-category and per-subset
  bars (BFCL v4 single-turn);
* performance — inline-IoU per-turn score distribution, turn completion, and
  TTFT / latency distributions (percentile curve + histogram).

Data extraction is pure and unit-tested; rendering is matplotlib-guarded (optional
test-extra dependency) following the same ``Agg`` pattern as
``utils/benchmark_httpclient.py``. Distributions that the run did not record
(e.g. TPOT/OSL when no tokenizer was attached) are skipped, not faked.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config.ruleset_registry import get_ruleset

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None  # type: ignore[assignment]
    plt = None  # type: ignore[assignment]

# Accuracy: ruleset golden-metric name -> key in the scorer's breakdown block.
_ACCURACY_METRIC_KEYS = {
    "bfcl_overall_accuracy": "overall_accuracy",
    "bfcl_normalized_accuracy": "normalized_single_turn_score",
}

# Latency-style summary blocks are nanoseconds; report in seconds.
_NS_TO_S = 1e-9


@dataclass
class AccuracyBreakdown:
    overall: float
    normalized: float
    total_samples: int
    category_scores: dict[str, float]
    subset_scores: dict[str, float]


@dataclass
class Distribution:
    name: str
    unit: str
    minimum: float
    maximum: float
    median: float | None
    avg: float | None
    percentiles: dict[float, float]
    hist_buckets: list[tuple[float, float]]
    hist_counts: list[int]


@dataclass
class RunArtifacts:
    report_dir: Path
    accuracy: AccuracyBreakdown | None = None
    turn_scores: list[float] = field(default_factory=list)
    turn_summary: dict[str, Any] = field(default_factory=dict)
    inline_score: float | None = None
    distributions: dict[str, Distribution] = field(default_factory=dict)


def _to_float(value: Any) -> float | None:
    """Coerce a metric to float (older artifacts stored strings like "86.23")."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_accuracy_score(results: dict[str, Any]) -> dict[str, Any] | None:
    accuracy_scores = results.get("accuracy_scores")
    if not isinstance(accuracy_scores, dict):
        return None
    for entry in accuracy_scores.values():
        if not isinstance(entry, dict):
            continue
        # Prefer the structured breakdown; fall back to score-as-dict (older runs).
        for block in (entry.get("breakdown"), entry.get("score")):
            if isinstance(block, dict) and "overall_accuracy" in block:
                return block
    return None


def extract_accuracy(results: dict[str, Any]) -> AccuracyBreakdown | None:
    """Pull the accuracy breakdown from a BFCL ``results.json`` dict."""
    score = _find_accuracy_score(results)
    if score is None:
        return None
    overall = _to_float(score.get("overall_accuracy"))
    normalized = _to_float(score.get("normalized_single_turn_score"))
    if overall is None or normalized is None:
        return None

    def _floats(block: Any) -> dict[str, float]:
        if not isinstance(block, dict):
            return {}
        out = {}
        for k, v in block.items():
            f = _to_float(v)
            if f is not None:
                out[k] = f
        return out

    return AccuracyBreakdown(
        overall=overall,
        normalized=normalized,
        total_samples=int(score.get("total_samples", 0) or 0),
        category_scores=_floats(score.get("category_scores")),
        subset_scores=_floats(score.get("subset_scores")),
    )


def extract_turn_scores(scores: dict[str, Any]) -> list[float]:
    """Per-turn inline-IoU scores from a perf ``scores.json`` dict."""
    per_turn = scores.get("per_turn")
    if not isinstance(per_turn, list):
        return []
    out = []
    for row in per_turn:
        f = _to_float(row.get("score")) if isinstance(row, dict) else None
        if f is not None:
            out.append(f)
    return out


def extract_distribution(
    summary: dict[str, Any], key: str, scale: float = _NS_TO_S, unit: str = "s"
) -> Distribution | None:
    """Build a Distribution from a ``result_summary.json`` metric block.

    Returns None when the block is absent or empty (e.g. the run recorded no
    samples for this metric), so callers can skip it cleanly.
    """
    block = summary.get(key)
    if not isinstance(block, dict) or not block.get("percentiles"):
        return None

    percentiles: dict[float, float] = {}
    for k, v in block["percentiles"].items():
        fv = _to_float(v)
        if fv is not None:
            percentiles[float(k)] = fv * scale
    hist = block.get("histogram") or {}
    raw_buckets = hist.get("buckets") or []
    counts = [int(c) for c in (hist.get("counts") or [])]
    buckets = [(float(lo) * scale, float(hi) * scale) for lo, hi in raw_buckets]

    return Distribution(
        name=key,
        unit=unit,
        minimum=(_to_float(block.get("min")) or 0.0) * scale,
        maximum=(_to_float(block.get("max")) or 0.0) * scale,
        median=(lambda m: m * scale if m is not None else None)(
            _to_float(block.get("median"))
        ),
        avg=(lambda a: a * scale if a is not None else None)(
            _to_float(block.get("avg"))
        ),
        percentiles=percentiles,
        hist_buckets=buckets,
        hist_counts=counts,
    )


def load_run(report_dir: str | Path) -> RunArtifacts:
    """Load and normalize a run's plottable artifacts from its report directory."""
    report_dir = Path(report_dir)
    artifacts = RunArtifacts(report_dir=report_dir)

    results_path = report_dir / "results.json"
    if results_path.exists():
        artifacts.accuracy = extract_accuracy(json.loads(results_path.read_text()))

    scores_path = report_dir / "scores.json"
    if scores_path.exists():
        scores = json.loads(scores_path.read_text())
        artifacts.turn_scores = extract_turn_scores(scores)
        if isinstance(scores.get("turns"), dict):
            artifacts.turn_summary = scores["turns"]
        artifacts.inline_score = _to_float(scores.get("score"))

    summary_path = report_dir / "result_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for key, unit in (("ttft", "s"), ("latency", "s"), ("tpot", "s")):
            dist = extract_distribution(summary, key, unit=unit)
            if dist is not None:
                artifacts.distributions[key] = dist
        osl = extract_distribution(
            summary, "output_sequence_lengths", scale=1.0, unit="tokens"
        )
        if osl is not None:
            artifacts.distributions["osl"] = osl

    return artifacts


def _accuracy_gate(ruleset_name: str, model_name: str) -> dict[str, float]:
    """Map result-score keys -> pass threshold (golden x factor) from the ruleset."""
    ruleset = get_ruleset(ruleset_name)
    benchmark_rulesets = getattr(ruleset, "benchmark_rulesets", {})
    model = next((m for m in benchmark_rulesets if m.name == model_name), None)
    if model is None:
        return {}
    _, golden = model.golden_accuracy
    factors = model.accuracy_target_settings[0]
    thresholds = {}
    for golden_key, result_key in _ACCURACY_METRIC_KEYS.items():
        if golden_key in golden and golden_key in factors:
            thresholds[result_key] = golden[golden_key] * factors[golden_key][0]
    return thresholds


def _require_matplotlib() -> bool:
    if plt is None:
        logger.warning(
            "matplotlib not installed; skipping plot generation. "
            "Install with the test extra: pip install '.[test]'"
        )
        return False
    return True


def plot_accuracy(
    breakdown: AccuracyBreakdown, gate: dict[str, float], out_path: Path
) -> Path | None:
    if not _require_matplotlib():
        return None
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    headline = {
        "overall": breakdown.overall,
        "normalized": breakdown.normalized,
    }
    bars = axes[0].bar(list(headline.keys()), list(headline.values()), color="#4c72b0")
    axes[0].set_title(f"Headline accuracy (n={breakdown.total_samples})")
    axes[0].set_ylabel("score (%)")
    axes[0].set_ylim(0, 100)
    gate_labels = {
        "overall": gate.get("overall_accuracy"),
        "normalized": gate.get("normalized_single_turn_score"),
    }
    for bar, thr in zip(bars, gate_labels.values(), strict=False):
        if thr is not None:
            axes[0].hlines(thr, bar.get_x(), bar.get_x() + bar.get_width(), color="red")
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                thr + 1,
                f"gate {thr:.1f}",
                ha="center",
                color="red",
                fontsize=8,
            )

    cats = breakdown.category_scores
    axes[1].bar(list(cats.keys()), list(cats.values()), color="#55a868")
    axes[1].set_title("Per-category accuracy")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=30)

    subs = dict(sorted(breakdown.subset_scores.items(), key=lambda kv: kv[1]))
    axes[2].barh(list(subs.keys()), list(subs.values()), color="#c44e52")
    axes[2].set_title("Per-subset accuracy")
    axes[2].set_xlim(0, 100)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_turn_scores(
    turn_scores: list[float],
    turn_summary: dict[str, Any],
    inline_score: float | None,
    out_path: Path,
) -> Path | None:
    if not _require_matplotlib():
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(turn_scores, bins=20, range=(0, 1), color="#4c72b0", edgecolor="white")
    title = "Per-turn inline-IoU score"
    if inline_score is not None:
        title += f" (mean {inline_score:.3f})"
    axes[0].set_title(title)
    axes[0].set_xlabel("IoU")
    axes[0].set_ylabel("turns")

    keys = ["issued", "observed", "missing"]
    vals = [int(turn_summary.get(k, 0) or 0) for k in keys]
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    axes[1].bar(keys, vals, color=colors)
    axes[1].set_title("Turn completion")
    for i, v in enumerate(vals):
        axes[1].text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_distribution(dist: Distribution, out_path: Path) -> Path | None:
    if not _require_matplotlib():
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pcts = sorted(dist.percentiles.items())
    if pcts:
        axes[0].plot([p for p, _ in pcts], [v for _, v in pcts], marker="o")
    axes[0].set_title(f"{dist.name} percentiles")
    axes[0].set_xlabel("percentile")
    axes[0].set_ylabel(dist.unit)

    if dist.hist_buckets and dist.hist_counts:
        # buckets and counts are extracted independently; a length mismatch would
        # make matplotlib's bar() raise. Slice both to the common length.
        n = min(len(dist.hist_buckets), len(dist.hist_counts))
        buckets = dist.hist_buckets[:n]
        centers = [(lo + hi) / 2 for lo, hi in buckets]
        widths = [(hi - lo) for lo, hi in buckets]
        axes[1].bar(
            centers,
            dist.hist_counts[:n],
            width=widths,
            color="#8172b3",
            align="center",
            edgecolor="white",
        )
    axes[1].set_title(f"{dist.name} histogram")
    axes[1].set_xlabel(dist.unit)
    axes[1].set_ylabel("count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_plots(
    report_dir: str | Path,
    out_dir: str | Path | None = None,
    ruleset_name: str = "mlperf-edge-current",
    model_name: str = "qwen3.6-27b",
) -> list[Path]:
    """Render all applicable plots for a run; returns the written PNG paths."""
    if not _require_matplotlib():
        return []

    artifacts = load_run(report_dir)
    out = Path(out_dir) if out_dir else Path(report_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    if artifacts.accuracy is not None:
        gate = _accuracy_gate(ruleset_name, model_name)
        p = plot_accuracy(artifacts.accuracy, gate, out / "accuracy.png")
        if p:
            written.append(p)

    if artifacts.turn_scores or artifacts.turn_summary:
        p = plot_turn_scores(
            artifacts.turn_scores,
            artifacts.turn_summary,
            artifacts.inline_score,
            out / "perf_turns.png",
        )
        if p:
            written.append(p)

    for key, dist in artifacts.distributions.items():
        p = plot_distribution(dist, out / f"perf_{key}.png")
        if p:
            written.append(p)

    return written
