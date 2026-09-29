# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Steady-state and drift diagnostics for benchmark event logs.

This module rebuilds per-sample TTFT and TPOT from ``events.jsonl``. The event
shapes come from ``core/record.py`` and ``core/types.py``. The parser is covered
by tests/unit/metrics/test_steady_state_diagnostics.py.

For each performance-tracked sample it computes:
  - ttft_ns = recv_first.ts - issued.ts
  - tpot_ns = (complete.ts - recv_first.ts) / tokens(text_after_first_chunk)

Token counts use plain tokenization of ``text_after_first_chunk``. The live
metrics aggregator uses the chat-template path for reasoning and tool-call
outputs. For reasoning models, absolute TPOT can differ from the run report.
CoV and trend tests are scale-invariant, so the steady/drift verdict is the
same.

Samples are grouped by issue order into super-passes. ``--superpass-size`` sets
the samples per bucket. By default it is the dataset size. Each metric and
percentile gets one trajectory across those buckets.

The headline output is the ``steady_state`` block. It reports the first steady
plateau found by grow-from-left segmentation. A plateau is admissible when the
gated metrics are trend-steady and within the CoV bound. The block includes
TTFT/TPOT histograms, percentiles, and per-user and system TPS with batch-means
confidence intervals.

A late staircase level shift is reported as ``anomaly``. The detector requires
a multi-plateau difference and a Pettitt change point. See
docs/steady-state-detection.md.

The CLI also prints whole-run trend verdicts for each metric and trend
algorithm. The headline carries the CoV for the reported window. If no window
is admissible but the whole run is trend-steady, it carries the whole-span CoV.

Only TPOT p50 and p90 gate admissibility. TTFT is diagnostic. It appears in the
headline percentiles and whole-run trend table, and it can raise the Drifting-Up
warning. It does not reject a window. At high concurrency, TTFT tail variance
comes from prefill, dataset ISL skew, and queueing, not from decode instability.
p99 and end-to-end latency are also diagnostic. Latency variation tracks the
OSL mix.

usage (auto-detects the tokenizer and workload profile from config.yaml, and
the super-pass size from the phase_start event; see MODEL_REGISTRY + PROFILES):
  python -m inference_endpoint.metrics.steady_state_diagnostics <run_dir>       # 0 flags
  python -m inference_endpoint.metrics.steady_state_diagnostics <events.jsonl> \
      --model kimi-k3                                                          # 1 flag
Every derived setting has an explicit override (--tokenizer, --dataset-size,
--superpass-size, --profile, --cov-bounds, --warmup, --json, ...).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist, median, pstdev
from typing import Final, Literal, NamedTuple, TypedDict

import msgspec
import yaml
from transformers import AutoTokenizer

from inference_endpoint.core.types import PhaseData, PhaseType

# --------------------------------------------------------------------------- #
# Event wire constants (mirror core/record.py category.value topics)
# --------------------------------------------------------------------------- #
EV_START_TRACKING = "session.start_performance_tracking"
EV_STOP_TRACKING = "session.stop_performance_tracking"
EV_PHASE_START = "session.phase_start"
EV_ISSUED = "sample.issued"
EV_RECV_FIRST = "sample.recv_first"
EV_COMPLETE = "sample.complete"


# Below this many super-passes a trend test is statistically meaningless.
MIN_TREND_N = 4

# Slope-vs-scatter drift thresholds.
# A metric drifts when the fitted run-length change is large relative to both
# its level and the residual scatter around the fitted line.
REL_DRIFT_THRESHOLD = 0.15
SNR_THRESHOLD = 2.0

# Effect-size floor for trend breaks during plateau segmentation (§5.5).
# The rank-based gate tests significance only. Over a long window, it can flag
# a negligible monotonic change of a few percent as a trend.
#
# A trend breaks a window only when it is significant and at least this large
# end-to-end. Smaller drift is treated as noise. CoV still guards real scatter.
#
# Corpus calibration: over-fragmenting breaks had |rel_drift| <= 0.03. Real
# drift and level shifts were well above 0.05. The threshold is cumulative, so
# persistent slow drift still breaks the growing window once total change
# crosses the floor.
TREND_REL_DRIFT_MIN = 0.05

# z for a two-sided 95% confidence interval (Hamed-Rao autocorrelation significance).
CI_Z_95 = 1.96
# Floor substituted for a zero median so relative-drift ratios stay finite.
ZERO_MEDIAN_FLOOR = 1e-9
# Texts buffered before a tokenizer flush during the parse.
TOKENIZE_BATCH_SIZE = 4096

# One default per knob, shared by the library and the CLI: a verdict re-derived by
# hand from a run's event log must be judged on the same grid as the live one.
DEFAULT_COV_BOUNDS: tuple[float, ...] = (0.03, 0.05, 0.08)

# Minimum steady-window wall time (docs/steady-state-detection.md §5.5).
# MIN_TREND_N super-passes are not enough by themselves. At high throughput, a
# few super-passes can cover only seconds of offered load.
#
# The required duration is max(precision, relaxation, floor):
#   precision  = k* . tau_sp
#   k*         = max(ceil((1.96 . CoV_b / eps)^2), MIN_TREND_N)
#                This is the batch-means target for +-eps precision.
#   relaxation = mult . p90(sample latency)
#                This covers queue and KV-eviction transients. p90 is used
#                instead of p99 so a few extreme request lifetimes do not set
#                the floor.
#   floor      = MLPerf min-duration floor.
MIN_DUR_EPS = 0.05
MIN_DUR_RELAX_MULT = 5.0
MIN_DUR_FLOOR_S = 600.0
# k* is floored at MIN_TREND_N batches. The batch-means CI grows with CoV, so
# k* rises for noisy metrics. A higher fixed floor would only penalize clean runs.
MIN_DUR_KSTAR_FLOOR = MIN_TREND_N

# A per-super-pass metric trajectory is classified into one of these states.
Verdict = Literal["up", "down", "steady", "insufficient"]


# The verdict rides ``MetricsSnapshot`` and lands in ``result_summary.json``.
# Use msgspec Structs so both boundaries validate it and mypy checks consumers.
class Anomaly(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    detected: bool
    change_point_sp: int | None
    delta_pct: float  # signed % change of the later TPOT level vs the first plateau
    pettitt: dict | None
    plateaus: list[list[int]]


class SteadyWindow(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    sp_lo: int  # post-warmup super-pass index (inclusive)
    sp_hi: int  # exclusive
    n_super_passes: int
    n_samples: int
    start_ns: int  # first issue of series[sp_lo] (duration is derivable; no field)
    end_ns: int  # last issue of series[sp_hi - 1]
    plateau_index: int  # 0-based index of this plateau among all admissible plateaus
    n_plateaus: int  # total admissible plateaus found
    skipped_short: int  # earlier plateaus skipped for failing the min-duration gate


class TpsBlock(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    per_user: float  # 1e9 / mean(TPOT ns) = output tok/s/user
    per_user_ci: list[float]  # [lo, hi]
    system: float  # total output tokens / window wall-clock
    system_ci: list[float]


class ShortWindow(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    is_short: bool  # window wall-time < required min-duration
    window_duration_s: float  # offered-load (issue) span of the reported window
    min_duration_s: float  # max(precision, relaxation, floor)
    dominant: str  # "precision" | "relaxation" | "floor" — which term set min_duration
    t_precision_s: float
    t_relaxation_s: float
    t_floor_s: float
    kstar: int  # batch-count target for +-MIN_DUR_EPS precision
    cov_b: float  # CoV of per-super-pass tpot_p50 over the window
    tau_sp_s: float  # median per-super-pass offered span
    l_p90_s: float  # p90 sample e2e latency over the window


class CovCell(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    gated: bool
    n: int  # super-passes the CoV was computed over
    cov: float | None  # None when there are < 2 points
    passes: dict[str, bool | None]  # cov-bound (as str) -> pass / fail / inconclusive


class CovBasis(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    """The span ``cov`` was measured over, when it is not the reported window."""

    sp_lo: int  # post-warmup super-pass index (inclusive)
    sp_hi: int  # exclusive
    n_super_passes: int


class SteadyState(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    found: bool
    reason: str | None
    superpass_size: (
        int  # samples per bucket; not recoverable from a partial last bucket
    )
    n_super_passes: int  # buckets in the whole series, warmup included
    warmup: int  # leading buckets cropped before selection
    window: SteadyWindow | None
    ttft: dict | None  # summarize() output
    tpot: dict | None
    osl: dict | None  # per-sample post-first-chunk output token counts
    latency: dict | None  # per-sample end-to-end latency
    tps: TpsBlock | None
    # CoV per tracked metric. When a window is reported, this is measured over
    # that window and justifies the verdict.
    #
    # If no window is found but some span is trend-steady, this is measured over
    # that span instead. ``cov_basis`` then identifies the span. In that case,
    # scatter, not drift, explains why no window passed.
    cov: dict[str, CovCell]
    # Set only when ``cov`` describes a span other than ``window``. ``None`` when
    # the reported window is the basis.
    cov_basis: CovBasis | None
    anomaly: Anomaly
    short_window: ShortWindow | None  # min-duration gate detail (None if no plateau)
    global_trend: dict[
        str, Verdict
    ]  # watched metric (TPOT + TTFT) -> trend from the plateau to run end
    drifting_up: list[str]  # watched metrics Drifting Up over the rest of the run


class TrackedMetric(NamedTuple):
    key: str  # display key, e.g. "ttft_p90"
    source_attr: str  # SuperPassRollup attribute holding the raw samples
    percentile: float
    gated: bool  # participates in the convergence gate (vs. diagnostic-only)


# Metric-percentile trajectories to track.
# Gated metrics participate in convergence. The rest are diagnostic.
# TTFT is intentionally diagnostic only; see GATED_METRICS.
TRACKED_METRICS: tuple[TrackedMetric, ...] = (
    TrackedMetric("ttft_p50", "ttft_ns", 0.50, False),
    TrackedMetric("ttft_p90", "ttft_ns", 0.90, False),
    TrackedMetric("tpot_p50", "tpot_ns", 0.50, True),
    TrackedMetric("tpot_p90", "tpot_ns", 0.90, True),
    TrackedMetric("ttft_p99", "ttft_ns", 0.99, False),
    TrackedMetric("tpot_p99", "tpot_ns", 0.99, False),
    # End-to-end sample latency is diagnostic. Its variance tracks the OSL mix
    # (§5.1). It is also useful for agentic runs with turbulent per-turn TTFT.
    TrackedMetric("latency_p50", "latency_ns", 0.50, False),
    TrackedMetric("latency_p90", "latency_ns", 0.90, False),
    # Warm-turn TTFT for agentic turn >= 2. The cold first turn is excluded.
    TrackedMetric("ttft_warm_p50", "ttft_warm_ns", 0.50, False),
    TrackedMetric("ttft_warm_p90", "ttft_warm_ns", 0.90, False),
)

# The admissibility gate uses only TPOT p50 and p90. These measure decode-rate
# steadiness.
#
# TTFT is not a hard gate. At high concurrency, TTFT tail variance comes from
# prefill, dataset ISL skew, and queue wait. It is structural, not decode
# instability, and it fragments steady runs (docs/steady-state-detection.md §5.5).
#
# Calibration evidence: two ~800s runs with steady TPOT, c7k and c22k, were
# rejected purely by TTFT-tail fragmentation. TPOT-only gating recovers them.
GATED_METRICS: tuple[TrackedMetric, ...] = tuple(m for m in TRACKED_METRICS if m.gated)
# Drifting-Up watches the gated TPOT pair plus TTFT p50 and p90. TTFT remains a
# warning-only signal: genuine TTFT saturation drift is surfaced, not rejected.
DRIFT_WATCH_METRICS: tuple[TrackedMetric, ...] = tuple(
    m
    for m in TRACKED_METRICS
    if m.key in ("tpot_p50", "tpot_p90", "ttft_p50", "ttft_p90")
)
_METRIC_BY_KEY: dict[str, TrackedMetric] = {m.key: m for m in TRACKED_METRICS}


# --------------------------------------------------------------------------- #
# Model registry + workload profiles + run-dir auto-detection
# --------------------------------------------------------------------------- #
# Maintained source of truth: model-name substring -> (HF tokenizer id, trust_remote_code).
# A run's config often carries a cluster path (e.g. /models/Kimi-K3) that will not load
# off the cluster, so the model name is mapped to a portable HF tokenizer id here.
MODEL_REGISTRY: tuple[tuple[str, str, bool], ...] = (
    ("kimi-k3", "moonshotai/Kimi-K3", True),
    ("kimi-k2", "moonshotai/Kimi-K2-Instruct", True),
    ("gpt-oss", "openai/gpt-oss-120b", False),
    ("deepseek-r1", "deepseek-ai/DeepSeek-R1", False),
    ("dsr1", "deepseek-ai/DeepSeek-R1", False),
    ("deepseek", "deepseek-ai/DeepSeek-R1", False),
)


def resolve_tokenizer(model: str) -> tuple[str, bool] | None:
    """Map a model name to (HF tokenizer id, trust_remote_code); None if unknown."""
    low = model.lower()
    for sub, tok, trust in MODEL_REGISTRY:
        if sub in low:
            return (tok, trust)
    return None


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    metric: str  # "window" | "natl"
    superpass_unit: str  # "samples" | "trajectories"
    superpass_size: int | None  # None -> use dataset-size
    cov_bounds: tuple[float, ...]
    warmup_driver: str
    tokenize_batch_size: int
    supported: bool
    note: str = ""


PROFILES: dict[str, Profile] = {
    "concurrency": Profile(
        "concurrency",
        "window",
        "samples",
        None,
        DEFAULT_COV_BOUNDS,
        "tpot_p50",
        4096,
        True,
    ),
    "poisson": Profile(
        "poisson",
        "window",
        "samples",
        None,
        DEFAULT_COV_BOUNDS,
        "tpot_p50",
        4096,
        True,
        "poisson: same detection as concurrency; usually under-saturated with the "
        "weakest ramp/drain.",
    ),
    "offline": Profile(
        "offline",
        "window",
        "samples",
        None,
        DEFAULT_COV_BOUNDS,
        "tpot_p50",
        4096,
        False,
        "offline: issue-time throughput degenerate (all issued at t=0), so the "
        "min-duration gate -- which measures the issue span -- collapses to ~0 and "
        "system TPS is unreliable. Runnable by hand; not collected during a run.",
    ),
    "agentic": Profile(
        "agentic",
        "natl",
        "trajectories",
        32,
        (0.10, 0.15),
        "tpot_p50",
        512,
        False,
        "agentic steady-state detection is NOT yet supported and requires further study.",
    ),
}
_LOAD_PATTERN_PROFILE: dict[str, str] = {
    "agentic_inference": "agentic",
    "poisson": "poisson",
    "max_throughput": "offline",
    "offline": "offline",
    "concurrency": "concurrency",
}


def profile_for_load_pattern(lp: str) -> Profile | None:
    """Return the validated profile for a load pattern.

    Return ``None`` for unclassified load patterns. The detector must fail
    closed so an unvalidated workload does not inherit another profile's verdict.
    """
    name = _LOAD_PATTERN_PROFILE.get(lp)
    return PROFILES[name] if name is not None else None


def find_run_files(target: str) -> tuple[str, str | None]:
    """Resolve ``events.jsonl`` and an optional ``config.yaml``.

    ``target`` may be a run directory or an event-log path. Directories are
    searched in ``./`` and ``./client/``.
    """
    if os.path.isdir(target):
        cands = [
            os.path.join(target, "events.jsonl"),
            os.path.join(target, "client", "events.jsonl"),
        ]
        events = next((c for c in cands if os.path.isfile(c)), None)
        if events is None:
            raise FileNotFoundError(
                f"no events.jsonl under {target} (looked in ./ and ./client/)"
            )
    else:
        events = target
        if not os.path.isfile(events):
            raise FileNotFoundError(f"no such events file: {events}")
    cfg = os.path.join(os.path.dirname(events), "config.yaml")
    return events, (cfg if os.path.isfile(cfg) else None)


def read_run_config(config_yaml_path: str | None) -> dict:
    """Best-effort model / load-pattern / trajectory count from a run's config.yaml."""
    out: dict = {"model": None, "load_pattern": None, "num_trajectories": None}
    if config_yaml_path and os.path.isfile(config_yaml_path):
        try:
            with open(config_yaml_path) as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception:  # malformed YAML -> best-effort empty
            cfg = {}
        if isinstance(cfg, dict):
            mp = cfg.get("model_params") or {}
            out["model"] = mp.get("name") or mp.get("tokenizer_name")
            lp = (
                (cfg.get("settings") or {}).get("load_pattern")
                or cfg.get("load_pattern")
                or {}
            )
            if isinstance(lp, dict):
                out["load_pattern"] = lp.get("type")
            for ds in cfg.get("datasets") or []:
                nt = ((ds or {}).get("agentic_inference") or {}).get(
                    "num_trajectories_to_issue"
                )
                if nt:
                    out["num_trajectories"] = int(nt)
                    break
    return out


def superpass_size_from_events(events_path: str) -> int | None:
    """Return the super-pass size from the first performance phase.

    This lets a re-run against a report directory take no arguments. It also
    keeps the replay on the same bucket size as the original run.

    ``PhaseData`` is decoded rather than hand-indexed. It is logged as a
    positional array because ``EventRecord.data`` members are ``array_like``.
    ``msgspec.convert`` checks both the tag and arity, so a shape mismatch fails
    instead of reading the wrong field.
    """
    with open(events_path) as f:
        for line in f:
            if EV_PHASE_START not in line:
                continue
            try:
                phase = msgspec.convert(json.loads(line).get("data"), type=PhaseData)
            except (json.JSONDecodeError, msgspec.ValidationError):
                continue
            if phase.phase_type is PhaseType.PERFORMANCE:
                return phase.num_turns
    return None


# --------------------------------------------------------------------------- #
# NATL (agentic per-trajectory throughput) -- EXPERIMENTAL, see the CLI warning
# --------------------------------------------------------------------------- #
def full_output_text(data: object) -> str:
    """Return all generated text for a turn, for OSL counting."""
    if not isinstance(data, list):
        return ""
    parts: list[str] = []
    reasoning = data[2] if len(data) > 2 else None
    output = data[1] if len(data) > 1 else ""
    if reasoning:
        parts.extend(reasoning if isinstance(reasoning, list) else [reasoning])
    if output:
        parts.extend(output if isinstance(output, list) else [output])
    tool_calls = data[3] if len(data) > 3 else None
    if tool_calls:
        parts.append(json.dumps(tool_calls))
    return "".join(str(p) for p in parts)


def build_trajectory_natl(
    events_path: str,
    count_tokens: Callable[[list[str]], list[int]],
    flush_size: int = 512,
) -> list[tuple[int, float]]:
    """Compute per-trajectory NATL values sorted by completion time.

    NATL is sum(output tokens) / sum(e2e latency seconds), grouped by
    ``conversation_id``. Each returned pair is ``(last_complete_ns, natl)``.
    """
    issue: dict[str, tuple[str, int]] = {}
    conv_lat_s: dict[str, float] = {}
    conv_tokens: dict[str, int] = {}
    conv_end_ns: dict[str, int] = {}
    b_conv: list[str] = []
    b_text: list[str] = []
    tracking = False

    def flush() -> None:
        if b_text:
            for conv, cnt in zip(b_conv, count_tokens(b_text), strict=True):
                conv_tokens[conv] = conv_tokens.get(conv, 0) + cnt
        b_conv.clear()
        b_text.clear()

    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            et = rec.get("event_type")
            ts = rec.get("timestamp_ns")
            if et == EV_START_TRACKING:
                tracking = True
            elif et == EV_STOP_TRACKING:
                tracking = False
            elif et == EV_ISSUED:
                uuid = rec.get("sample_uuid")
                if tracking and uuid and ts is not None:
                    issue[uuid] = (rec.get("conversation_id") or "", ts)
            elif et == EV_COMPLETE:
                iv = issue.pop(rec.get("sample_uuid"), None)
                if iv is None or ts is None:
                    continue
                conv, its = iv
                conv_lat_s[conv] = conv_lat_s.get(conv, 0.0) + (ts - its) / 1e9
                conv_end_ns[conv] = max(conv_end_ns.get(conv, 0), ts)
                text = full_output_text(rec.get("data"))
                if text:
                    b_conv.append(conv)
                    b_text.append(text)
                    if len(b_text) >= flush_size:
                        flush()
    flush()
    pairs = [
        (conv_end_ns[c], conv_tokens[c] / conv_lat_s[c])
        for c in conv_lat_s
        if conv_lat_s[c] > 0 and conv_tokens.get(c, 0) > 0
    ]
    pairs.sort(key=lambda p: p[0])
    return pairs


def build_natl_result(
    pairs: Sequence[tuple[int, float]],
    superpass_trajectories: int = 32,
    cov_bounds: Sequence[float] = (0.10, 0.15),
) -> dict:
    """Distribution + steadiness (batch-means over trajectory super-passes) for NATL."""
    natl = [n for _, n in pairs]  # completion-ordered
    n = len(natl)
    sp_median: list[float] = []
    for i in range(0, n, superpass_trajectories):
        chunk = natl[i : i + superpass_trajectories]
        if len(chunk) >= superpass_trajectories // 2:
            sp_median.append(median(chunk))
    n_sp = len(sp_median)
    across = cov(sp_median) if n_sp >= 2 else 0.0
    se_pct = 100.0 * across / math.sqrt(n_sp) if n_sp else 0.0
    s = sorted(natl)
    dist = {
        "p10": percentile_lower(s, 0.10) if s else 0.0,
        "p50": percentile_lower(s, 0.50) if s else 0.0,
        "p90": percentile_lower(s, 0.90) if s else 0.0,
        "p99": percentile_lower(s, 0.99) if s else 0.0,
        "mean": (sum(s) / n) if n else 0.0,
        "cov": cov(natl),
    }
    return {
        "n_trajectories": n,
        "superpass_trajectories": superpass_trajectories,
        "n_super_passes": n_sp,
        "distribution": dist,
        "sp_median": sp_median,
        "across_cov": across,
        "batch_means_se_pct": se_pct,
        "found": n_sp >= 2 and across <= max(cov_bounds),
    }


# --------------------------------------------------------------------------- #
# TextModelOutput.text_after_first_chunk for the parsed JSON array
# --------------------------------------------------------------------------- #
def text_after_first_chunk(data: object) -> str:
    """Return output text after the first streamed chunk.

    This is the text used as the TPOT numerator. ``data`` is the COMPLETE event
    payload: ``[tag, output, reasoning?, tool_calls?]`` with trailing defaults
    omitted by msgspec ``array_like`` and ``omit_defaults``.

    ``output`` and ``reasoning`` are each either a string for non-streaming
    output or a list of streamed chunks. The logic mirrors
    ``TextModelOutput.text_after_first_chunk`` in core/types.py.
    """
    if not isinstance(data, list) or not data:
        return ""
    output = data[1] if len(data) > 1 else ""
    reasoning = data[2] if len(data) > 2 else None
    parts: list[str] = []
    if reasoning:
        if isinstance(reasoning, list) and len(reasoning) > 1:
            parts.extend(reasoning[1:])
        # str reasoning is a single (first) chunk -> skip entirely
    if output:
        if isinstance(output, str):
            # Non-streaming output: keep it only if a first chunk already lived in a
            # (streaming) reasoning trace; otherwise the str output IS the first chunk.
            if parts or (reasoning and isinstance(reasoning, list)):
                parts.append(output)
        elif isinstance(output, list):
            if parts or reasoning:
                parts.extend(output)
            elif len(output) > 1:
                parts.extend(output[1:])
    # Tool-call reconstruction is intentionally omitted: tool-call samples use a
    # chat-template tokenization path this diagnostic does not replicate.
    return "".join(parts)


# --------------------------------------------------------------------------- #
# Super-pass series
# --------------------------------------------------------------------------- #
def _samples() -> array[float]:
    """An empty per-sample series. Shared default factory for SuperPassRollup."""
    return array("d")


@dataclass(slots=True)
class SuperPassRollup:
    index: int
    n_issued: int = 0  # per-super-pass sample count (coverage / bucketing invariant)
    first_issue_ns: int = -1  # earliest issue ts (offered-load span start)
    last_issue_ns: int = -1  # latest issue ts (offered-load span end; throughput denom)
    last_event_ns: int = -1  # latest event ts incl. completions (drain-inclusive end)
    # Keep sample series in array("d"). It stores 8 packed bytes per sample
    # instead of a boxed float plus a pointer: 108 -> 34 bytes/sample across
    # the five series, or ~220 MB -> ~70 MB retained for a 2M-sample run.
    # Callers only need Sequence-style operations, so list semantics are not
    # required.
    ttft_ns: array[float] = field(default_factory=_samples)
    ttft_warm_ns: array[float] = field(default_factory=_samples)  # turn >= 2 (warm KV)
    tpot_ns: array[float] = field(default_factory=_samples)
    latency_ns: array[float] = field(
        default_factory=_samples
    )  # issue -> complete (e2e)
    osl: array[float] = field(default_factory=_samples)  # post-first-chunk tokens
    out_tokens: int = 0


@dataclass(slots=True)
class _PendingRow:
    """In-flight sample state during the parse, keyed by uuid until COMPLETE."""

    superpass_index: int
    issue_ns: int
    recv_first_ns: int | None = None


class SuperPassCollector:
    """Bucket performance-tracked samples into super-passes by issue order.

    This class owns the bucketing arithmetic used by both producers: the live
    metrics aggregator and :func:`build_super_pass_series`. Both paths therefore
    produce the same ``list[SuperPassRollup]`` and the same verdict.

    The collector keeps no per-sample state. The caller already has an in-flight
    row, either ``SampleRow`` in the aggregator or :class:`_PendingRow` in the
    replay parser. The caller stores the super-pass index there and passes it
    back with timestamps. A uuid map here would duplicate that state for the
    whole run.

    Token counts can arrive during drain, after the in-flight row is gone.
    Callers pass the stored super-pass index back to attach TPOT later.

    Callers must report only tracked samples. ``superpass_index < 0`` means the sample
    was issued before the bucket size was known. Every later hook for that
    sample must be skipped.
    """

    __slots__ = ("_issue_counter", "_series", "superpass_size")

    def __init__(self, superpass_size: int = 0) -> None:
        # 0 = not yet known; assign refuses samples until announce_phase lands.
        self.superpass_size = superpass_size
        self._series: list[SuperPassRollup] = []
        self._issue_counter = 0

    def announce_phase(self, num_turns: int) -> None:
        """Latch the bucket size from a phase announcement.

        After bucketing starts, later phase announcements are ignored. Accuracy
        phases and additional performance phases cannot resize an existing
        series.
        """
        if num_turns > 0 and not self._series:
            self.superpass_size = num_turns

    def series(self) -> list[SuperPassRollup]:
        """Return the collected super-pass rollups."""
        return self._series

    def assign(self, ts_ns: int) -> int:
        """Assign a newly issued sample to a super-pass.

        Returns ``-1`` until the bucket size is known. The caller stores the
        result on its in-flight row and passes it to later hooks.
        """
        if not self.superpass_size:
            return -1
        superpass_index = self._issue_counter // self.superpass_size
        self._issue_counter += 1
        while len(self._series) <= superpass_index:
            self._series.append(SuperPassRollup(index=len(self._series)))
        sp = self._series[superpass_index]
        sp.n_issued += 1
        if sp.first_issue_ns < 0:
            sp.first_issue_ns = ts_ns
        sp.last_issue_ns = max(sp.last_issue_ns, ts_ns)
        sp.last_event_ns = max(sp.last_event_ns, ts_ns)
        return superpass_index

    def reissue(self, superpass_index: int, ts_ns: int) -> None:
        """Update issue timing for a retry of an already-bucketed sample."""
        sp = self._series[superpass_index]
        sp.last_issue_ns = max(sp.last_issue_ns, ts_ns)
        sp.last_event_ns = max(sp.last_event_ns, ts_ns)

    def on_recv_first(
        self,
        superpass_index: int,
        issue_ns: int,
        ts_ns: int,
        turn: int | None,
        *,
        first: bool,
    ) -> None:
        """Record TTFT for the sample's first streamed chunk."""
        sp = self._series[superpass_index]
        sp.last_event_ns = max(sp.last_event_ns, ts_ns)
        # Retries re-emit recv_first. Count TTFT only once per sample.
        if not first:
            return
        ttft = float(ts_ns - issue_ns)
        sp.ttft_ns.append(ttft)
        # Warm-turn TTFT excludes turn 1 of each agentic trajectory because it
        # has no KV-cache hit. Single-turn workloads have no turn number, so
        # they stay in the warm series.
        if turn is None or turn > 1:
            sp.ttft_warm_ns.append(ttft)

    def on_complete(
        self, superpass_index: int, issue_ns: int, recv_first_ns: int | None, ts_ns: int
    ) -> float | None:
        """Record e2e latency and return the streamed TPOT numerator.

        The returned value is nanoseconds from first chunk to completion. The
        caller later divides it by token count via :meth:`add_tpot`. Return
        ``None`` when the sample never streamed a first chunk.
        """
        sp = self._series[superpass_index]
        sp.last_event_ns = max(sp.last_event_ns, ts_ns)
        sp.latency_ns.append(float(ts_ns - issue_ns))  # e2e, no recv_first needed
        if recv_first_ns is None:
            return None
        return float(ts_ns - recv_first_ns)

    def add_tpot(self, superpass_index: int, tpot_ns: float, token_count: int) -> None:
        """Attach a resolved TPOT and its token count to an earlier bucket."""
        sp = self._series[superpass_index]
        sp.tpot_ns.append(tpot_ns)
        sp.osl.append(float(token_count))
        sp.out_tokens += token_count


def build_super_pass_series(
    events_path: str,
    superpass_size: int,
    count_tokens: Callable[[list[str]], list[int]],
    flush_size: int = TOKENIZE_BATCH_SIZE,
) -> list[SuperPassRollup]:
    """Replay an ``events.jsonl`` through a :class:`SuperPassCollector`.

    ``count_tokens`` maps a text batch to token counts. It is injected so tests
    can avoid a real tokenizer and callers can swap tokenizers.

    ``flush_size`` caps buffered output texts before tokenizer flush. Lower
    values bound peak memory for long reasoning outputs at the cost of smaller
    tokenizer batches.
    """
    if superpass_size <= 0:
        raise ValueError("superpass_size must be positive")
    collector = SuperPassCollector(superpass_size)
    # The parse keeps its own in-flight rows: it has no MetricsTable to hang the
    # super-pass index off, which is where the live producer stores it.
    rows: dict[str, _PendingRow] = {}
    tracking = False
    # Buffered by batch position, not by uuid: a retried sample completes twice and
    # would collide on a uuid key, losing the first completion or raising.
    pending: list[tuple[int, float]] = []
    batch_texts: list[str] = []

    def flush_tpot() -> None:
        if batch_texts:
            counts = count_tokens(batch_texts)
            for (superpass_index, delta), cnt in zip(pending, counts, strict=True):
                if cnt > 0:
                    collector.add_tpot(superpass_index, delta / cnt, cnt)
        pending.clear()
        batch_texts.clear()

    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip a truncated/partial line (e.g. last line of a killed run)
            et = rec.get("event_type")
            if et == EV_START_TRACKING:
                tracking = True
                continue
            if et == EV_STOP_TRACKING:
                tracking = False
                continue
            ts = rec.get("timestamp_ns")  # may be absent on a truncated/partial event
            uuid = rec.get("sample_uuid")
            if ts is None or not uuid:
                continue
            if et == EV_ISSUED:
                if not tracking:
                    continue
                existing = rows.get(uuid)
                if existing is not None:
                    existing.issue_ns = ts  # retry: refresh issue ts only
                    collector.reissue(existing.superpass_index, ts)
                    continue
                superpass_index = collector.assign(ts)
                if superpass_index >= 0:
                    rows[uuid] = _PendingRow(
                        superpass_index=superpass_index, issue_ns=ts
                    )
            elif et == EV_RECV_FIRST:
                row = rows.get(uuid)
                if row is not None:
                    collector.on_recv_first(
                        row.superpass_index,
                        row.issue_ns,
                        ts,
                        rec.get("turn"),
                        first=row.recv_first_ns is None,
                    )
                    if row.recv_first_ns is None:
                        row.recv_first_ns = ts
            elif et == EV_COMPLETE:
                row = rows.pop(uuid, None)
                if row is None:
                    continue
                delta = collector.on_complete(
                    row.superpass_index, row.issue_ns, row.recv_first_ns, ts
                )
                if delta is None:
                    continue
                text = text_after_first_chunk(rec.get("data"))
                if text:
                    pending.append((row.superpass_index, delta))
                    batch_texts.append(text)
                    if len(batch_texts) >= flush_size:
                        flush_tpot()
    flush_tpot()
    return collector.series()


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #
def percentile_lower(sorted_values: Sequence[float], p: float) -> float:
    n = len(sorted_values)
    if n == 0:
        raise ValueError("percentile of empty series")
    return sorted_values[int(p * (n - 1))]


def cov(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    if m == 0:
        return 0.0
    return pstdev(values) / abs(m)


def _phi(z: float) -> float:
    """Standard-normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _two_sided_p(stat: float) -> float:
    return 2.0 * (1.0 - _phi(abs(stat)))


def super_pass_percentile_series(
    series: Sequence[SuperPassRollup], source_attr: str, percentile: float
) -> list[float]:
    """Per-super-pass percentile trajectory; super-passes with no samples are skipped."""
    out: list[float] = []
    for sp in series:
        vals = getattr(sp, source_attr)
        if vals:
            out.append(percentile_lower(sorted(vals), percentile))
    return out


def pooled(
    series: Sequence[SuperPassRollup], lo: int, hi: int, source_attr: str
) -> list[float]:
    """All raw samples of an attribute pooled across super-passes ``[lo, hi)``."""
    out: list[float] = []
    for sp in series[lo:hi]:
        out.extend(getattr(sp, source_attr))
    return out


def pooled_out_tokens(series: Sequence[SuperPassRollup], lo: int, hi: int) -> int:
    return sum(sp.out_tokens for sp in series[lo:hi])


def window_elapsed_ns(series: Sequence[SuperPassRollup], lo: int, hi: int) -> int:
    """Completion span of ``[lo, hi)``: earliest issue to latest event (drain-inclusive)."""
    window = series[lo:hi]
    firsts = [sp.first_issue_ns for sp in window if sp.first_issue_ns >= 0]
    lasts = [sp.last_event_ns for sp in window if sp.last_event_ns >= 0]
    if not firsts or not lasts:
        return 0
    return max(lasts) - min(firsts)


def window_issue_span_ns(series: Sequence[SuperPassRollup], lo: int, hi: int) -> int:
    """Offered-load span of ``[lo, hi)``: earliest to latest *issue*.

    This is the throughput denominator (§5.1). Drain time starts after the last
    issue, so counting to the last completion would inflate the denominator and
    deflate TPS, especially for high-tail workloads with long TTFT and decode.
    """
    window = series[lo:hi]
    firsts = [sp.first_issue_ns for sp in window if sp.first_issue_ns >= 0]
    lasts = [sp.last_issue_ns for sp in window if sp.last_issue_ns >= 0]
    if not firsts or not lasts:
        return 0
    return max(lasts) - min(firsts)


def histogram(values: Sequence[float], nbins: int = 20) -> list[dict]:
    """Bin counts over ``[min, max]``; log-spaced edges when strictly positive."""
    lo, hi = min(values), max(values)
    if lo == hi:
        return [{"lo": lo, "hi": hi, "count": len(values)}]
    if lo > 0:
        ratio = hi / lo
        edges = [lo * ratio ** (i / nbins) for i in range(nbins + 1)]
    else:
        edges = [lo + (hi - lo) * i / nbins for i in range(nbins + 1)]
    counts = [0] * nbins
    for v in values:
        if v >= hi:
            counts[-1] += 1
            continue
        for b in range(nbins):
            if v < edges[b + 1]:
                counts[b] += 1
                break
    return [
        {"lo": edges[b], "hi": edges[b + 1], "count": counts[b]} for b in range(nbins)
    ]


def summarize(values: Sequence[float]) -> dict:
    """Count, mean, min/max, p50/p90/p99 (nearest-rank-lower), and a histogram."""
    s = sorted(values)
    n = len(s)
    return {
        "count": n,
        "mean": sum(s) / n,
        "min": s[0],
        "max": s[-1],
        "p50": percentile_lower(s, 0.50),
        "p90": percentile_lower(s, 0.90),
        "p99": percentile_lower(s, 0.99),
        "histogram": histogram(s),
    }


# --------------------------------------------------------------------------- #
# Estimation: batch-means CI, Pettitt change-point, TPS
# --------------------------------------------------------------------------- #
# Two-sided 95% Student-t critical values by degrees of freedom (df>30 -> ~1.96).
_T_CRIT_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _t_crit_95(df: int) -> float:
    return _T_CRIT_95.get(df, 1.96)


def batch_means_ci(
    batch_means: Sequence[float], confidence: float = 0.95
) -> tuple[float, float]:
    """Confidence interval for the grand mean from non-overlapping batch means.

    Batches (here: super-passes) are treated as approximately independent, so the
    interval accounts for per-super-pass autocorrelation that a raw-sample CI would
    ignore. Uses a Student-t critical value (small-sample correct) for 95%.
    """
    k = len(batch_means)
    if k == 0:
        return (0.0, 0.0)
    m = sum(batch_means) / k
    if k < 2:
        return (m, m)
    var = sum((b - m) ** 2 for b in batch_means) / (k - 1)
    se = math.sqrt(var) / math.sqrt(k)
    if confidence == 0.95:
        crit = _t_crit_95(k - 1)
    else:
        crit = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    return (m - crit * se, m + crit * se)


def _average_ranks(values: Sequence[float]) -> list[float]:
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0  # average of the tied 1-based ranks
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pettitt(values: Sequence[float], alpha: float = 0.05) -> dict:
    """Pettitt nonparametric single-change-point test.

    Returns the split index (size of the first segment), the ``K`` statistic, an
    approximate p-value, and whether a change point is significant at ``alpha``.
    Rank-based, so it pairs with the Mann-Kendall trend gate.
    """
    n = len(values)
    if n < MIN_TREND_N:
        return {"change_point": 0, "k_stat": 0.0, "pvalue": 1.0, "significant": False}
    ranks = _average_ranks(values)
    cum = 0.0
    k_stat = 0.0
    cp = 0
    for t in range(1, n):  # t = size of the first segment
        cum += ranks[t - 1]
        u = 2.0 * cum - t * (n + 1)
        if abs(u) > k_stat:
            k_stat = abs(u)
            cp = t
    pvalue = min(1.0, 2.0 * math.exp(-6.0 * k_stat * k_stat / (n**3 + n**2)))
    return {
        "change_point": cp,
        "k_stat": k_stat,
        "pvalue": pvalue,
        "significant": pvalue < alpha,
    }


def per_user_tps(mean_tpot_ns: float) -> float:
    """Output tokens/s/user from mean time-per-output-token (ns)."""
    return 1e9 / mean_tpot_ns if mean_tpot_ns > 0 else 0.0


def system_tps(out_tokens: int, elapsed_ns: int) -> float:
    """Aggregate output tokens/s over a window's wall-clock span."""
    return out_tokens / (elapsed_ns / 1e9) if elapsed_ns > 0 else 0.0


# --------------------------------------------------------------------------- #
# Trend algorithms -- each returns a TrendResult with verdict in
# {"up", "steady", "down", "insufficient"}.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TrendResult:
    verdict: Verdict
    slope: float = 0.0
    statistic: float = 0.0  # primary test statistic (Mann-Kendall S, Newey-West t)
    pvalue: float | None = (
        None  # None for effect-size tests (theil_sen, slope_vs_scatter)
    )
    variance: float = 0.0
    rel_drift: float = 0.0  # signed total change / median
    snr: float = 0.0  # |total change| / residual scatter


def _insufficient() -> TrendResult:
    return TrendResult("insufficient")


def _direction(x: float) -> Verdict:
    return "up" if x > 0 else "down" if x < 0 else "steady"


def _significant_verdict(effect: float, pvalue: float, alpha: float) -> Verdict:
    """up/down when the effect is significant (pvalue < alpha) in that direction."""
    if pvalue < alpha:
        if effect > 0:
            return "up"
        if effect < 0:
            return "down"
    return "steady"


def _median_or_floor(values: Sequence[float]) -> float:
    """Median, floored away from zero so relative-drift ratios stay finite."""
    m = median(values)
    return m if m else ZERO_MEDIAN_FLOOR


def _mk_S(values: Sequence[float]) -> int:
    n = len(values)
    s = 0
    for i in range(n - 1):
        vi = values[i]
        for j in range(i + 1, n):
            d = values[j] - vi
            s += (d > 0) - (d < 0)
    return s


def _mk_variance(values: Sequence[float]) -> float:
    n = len(values)
    counts: dict[float, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in counts.values())
    return (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0


def _mk_verdict(s: int, variance: float, alpha: float) -> TrendResult:
    if variance <= 0:
        return TrendResult(
            _direction(s), statistic=float(s), pvalue=0.0, variance=variance
        )
    if s > 0:
        z = (s - 1) / math.sqrt(variance)
    elif s < 0:
        z = (s + 1) / math.sqrt(variance)
    else:
        z = 0.0
    p = _two_sided_p(z)
    return TrendResult(
        _significant_verdict(s, p, alpha),
        statistic=float(s),
        pvalue=p,
        variance=variance,
    )


def mann_kendall(values: Sequence[float], alpha: float = 0.05) -> TrendResult:
    if len(values) < MIN_TREND_N:
        return _insufficient()
    return _mk_verdict(_mk_S(values), _mk_variance(values), alpha)


def _autocorr_of_ranks(values: Sequence[float]) -> list[float]:
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = float(rank)
    mean = sum(ranks) / n
    dev = [r - mean for r in ranks]
    denom = sum(d * d for d in dev)
    acf: list[float] = []
    if denom == 0:
        return [0.0] * (n - 1)
    for k in range(1, n):
        num = sum(dev[t] * dev[t - k] for t in range(k, n))
        acf.append(num / denom)
    return acf


def mann_kendall_hamed_rao(values: Sequence[float], alpha: float = 0.05) -> TrendResult:
    """Mann-Kendall with Hamed-Rao autocorrelation correction.

    The correction scales MK variance by an effective-sample-size factor from
    significant autocorrelations in the data ranks. This keeps serial
    correlation from creating false significance.
    """
    n = len(values)
    if n < MIN_TREND_N:
        return _insufficient()
    s = _mk_S(values)
    var0 = _mk_variance(values)
    acf = _autocorr_of_ranks(values)
    ci = CI_Z_95 / math.sqrt(n)
    factor_sum = 0.0
    for k in range(1, n):
        r = acf[k - 1]
        if abs(r) <= ci:  # only statistically significant lags contribute
            continue
        factor_sum += (n - k) * (n - k - 1) * (n - k - 2) * r
    correction = 1.0 + (2.0 / (n * (n - 1) * (n - 2))) * factor_sum
    # A non-positive correction is degenerate, usually from over-correction under
    # strong negative autocorrelation. Use the uncorrected MK variance. Clamping
    # near zero would collapse variance and manufacture a significant trend.
    if correction <= 0:
        correction = 1.0
    return _mk_verdict(s, var0 * correction, alpha)


def theil_sen(
    values: Sequence[float], rel_threshold: float = REL_DRIFT_THRESHOLD
) -> TrendResult:
    n = len(values)
    if n < MIN_TREND_N:
        return _insufficient()
    slopes = [
        (values[j] - values[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)
    ]
    slope = median(slopes)
    rel = slope * (n - 1) / _median_or_floor(values)
    verdict: Verdict = "steady" if abs(rel) < rel_threshold else _direction(rel)
    return TrendResult(verdict, slope=slope, rel_drift=rel)


def _ols(values: Sequence[float]) -> tuple[float, float, list[float]]:
    n = len(values)
    xbar = (n - 1) / 2.0
    ybar = sum(values) / n
    sxx = sum((x - xbar) ** 2 for x in range(n))
    sxy = sum((x - xbar) * (v - ybar) for x, v in enumerate(values))
    slope = sxy / sxx if sxx else 0.0
    intercept = ybar - slope * xbar
    resid = [v - (intercept + slope * x) for x, v in enumerate(values)]
    return slope, sxx, resid


def _rel_drift(values: Sequence[float]) -> float:
    """Return signed end-to-end fractional change of the OLS line.

    This matches ``slope_vs_scatter``'s ``rel_drift``. It is the effect-size
    floor that keeps negligible but significant trends from fragmenting a steady
    plateau (§5.5).
    """
    if len(values) < 2:
        return 0.0
    slope, _sxx, _resid = _ols(values)
    return slope * (len(values) - 1) / _median_or_floor(values)


def newey_west(
    values: Sequence[float], lag: int | None = None, alpha: float = 0.05
) -> TrendResult:
    """OLS slope significance with a Newey-West (HAC) standard error."""
    n = len(values)
    if n < MIN_TREND_N:
        return _insufficient()
    slope, sxx, resid = _ols(values)
    if sxx == 0:
        return TrendResult("steady", slope=0.0)
    xbar = (n - 1) / 2.0
    u = [(x - xbar) * resid[x] for x in range(n)]
    if lag is None:
        lag = max(1, int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    s = sum(ui * ui for ui in u)
    for lg in range(1, min(lag, n - 1) + 1):
        w = 1.0 - lg / (lag + 1.0)
        s += 2.0 * w * sum(u[t] * u[t - lg] for t in range(lg, n))
    var_b = s / (sxx * sxx)
    se = math.sqrt(var_b) if var_b > 0 else 0.0
    if se == 0:
        return TrendResult(_direction(slope), slope=slope, pvalue=0.0)
    t = slope / se
    p = _two_sided_p(t)
    return TrendResult(
        _significant_verdict(slope, p, alpha), slope=slope, statistic=t, pvalue=p
    )


def slope_vs_scatter(
    values: Sequence[float],
    rel_threshold: float = REL_DRIFT_THRESHOLD,
    snr_threshold: float = SNR_THRESHOLD,
) -> TrendResult:
    n = len(values)
    if n < MIN_TREND_N:
        return _insufficient()
    slope, _sxx, resid = _ols(values)
    resid_std = pstdev(resid) if n > 1 else 0.0
    total_change = slope * (n - 1)
    rel_drift = total_change / _median_or_floor(values)
    snr = abs(total_change) / (resid_std + ZERO_MEDIAN_FLOOR)
    drifting = abs(rel_drift) >= rel_threshold and snr >= snr_threshold
    verdict: Verdict = _direction(rel_drift) if drifting else "steady"
    return TrendResult(verdict, slope=slope, snr=snr, rel_drift=rel_drift)


ALGORITHMS: dict[str, Callable[[Sequence[float]], TrendResult]] = {
    "mk_hamed_rao": mann_kendall_hamed_rao,
    "mann_kendall": mann_kendall,
    "newey_west": newey_west,
    "theil_sen": theil_sen,
    "slope_vs_scatter": slope_vs_scatter,
}


# --------------------------------------------------------------------------- #
# Rolling scan + CoV table
# --------------------------------------------------------------------------- #
def cov_pass_row(
    values: Sequence[float], bounds: Sequence[float]
) -> dict[float, bool | None]:
    # CoV is undefined with fewer than two points. Report inconclusive, not
    # PASS, so a short or empty window cannot masquerade as steady.
    if len(values) < 2:
        return {b: None for b in bounds}
    c = cov(values)
    return {b: c <= b for b in bounds}


def cov_table(
    series: Sequence[SuperPassRollup], bounds: Sequence[float]
) -> dict[str, CovCell]:
    """CoV of every tracked metric across ``series``, scored against each bound."""
    out: dict[str, CovCell] = {}
    for m in TRACKED_METRICS:
        traj = super_pass_percentile_series(series, m.source_attr, m.percentile)
        out[m.key] = CovCell(
            gated=m.gated,
            n=len(traj),
            cov=cov(traj) if len(traj) >= 2 else None,
            passes={str(b): v for b, v in cov_pass_row(traj, bounds).items()},
        )
    return out


# --------------------------------------------------------------------------- #
# Steady-window selection: admissibility, plateau segmentation, level shift
# --------------------------------------------------------------------------- #
def _window_percentile_series(
    series: Sequence[SuperPassRollup], lo: int, hi: int, source_attr: str, pct: float
) -> list[float] | None:
    """Per-super-pass percentile over ``[lo, hi)``; None if any super-pass is empty."""
    out: list[float] = []
    for sp in series[lo:hi]:
        vals = getattr(sp, source_attr)
        if not vals:
            return None
        out.append(percentile_lower(sorted(vals), pct))
    return out


# Gate outcomes for candidate windows.
# Trend and scatter failures stay separate because they need different advice.
_GATE_OK: Final[str] = "ok"
_GATE_SHORT: Final[str] = "short"
_GATE_TREND: Final[str] = "trend"
_GATE_COV: Final[str] = "cov"


def _window_gate(
    series: Sequence[SuperPassRollup],
    lo: int,
    hi: int,
    gate_algo: str,
    cov_bounds: Sequence[float],
    gated_metrics: Sequence[TrackedMetric],
) -> str:
    """Return the first gate a window fails, or ``ok``.

    Trend failure disqualifies the window immediately. CoV failure is remembered
    but does not stop the scan. That lets a trend-steady window report ``cov``
    even if later metrics are checked.
    """
    gate = ALGORITHMS[gate_algo]
    loosest = max(cov_bounds)
    worst = _GATE_OK
    for m in gated_metrics:
        traj = _window_percentile_series(series, lo, hi, m.source_attr, m.percentile)
        if traj is None or len(traj) < MIN_TREND_N:
            return _GATE_SHORT
        if (
            gate(traj).verdict != "steady"
            and abs(_rel_drift(traj)) >= TREND_REL_DRIFT_MIN
        ):
            # Break only on trends that are significant and practically large.
            # Smaller significant drift is treated as noise. CoV still guards
            # scatter below.
            return _GATE_TREND
        if cov(traj) > loosest:
            worst = _GATE_COV
    return worst


def window_admissible(
    series: Sequence[SuperPassRollup],
    lo: int,
    hi: int,
    gate_algo: str,
    cov_bounds: Sequence[float],
    gated_metrics: Sequence[TrackedMetric] = GATED_METRICS,
) -> bool:
    """True iff every gated metric is trend-steady and within the loosest CoV bound."""
    return (
        _window_gate(series, lo, hi, gate_algo, cov_bounds, gated_metrics) == _GATE_OK
    )


def gated_trend_drifters(
    series: Sequence[SuperPassRollup],
    gate_algo: str,
    gated_metrics: Sequence[TrackedMetric] = GATED_METRICS,
) -> list[str] | None:
    """Return gated metrics that trend across the whole series.

    Returns ``[]`` when no gated metric trends. Returns ``None`` when the test
    cannot run because there are too few super-passes or an empty super-pass.

    This applies the same trend rule as :func:`_window_gate`, including the
    effect-size floor. A whole-span test separates drifting runs from flat but
    noisy runs when no plateau is admissible.
    """
    gate = ALGORITHMS[gate_algo]
    drifting: list[str] = []
    for m in gated_metrics:
        traj = _window_percentile_series(
            series, 0, len(series), m.source_attr, m.percentile
        )
        if traj is None or len(traj) < MIN_TREND_N:
            return None
        if (
            gate(traj).verdict != "steady"
            and abs(_rel_drift(traj)) >= TREND_REL_DRIFT_MIN
        ):
            drifting.append(m.key)
    return drifting


def segment_plateaus(
    series: Sequence[SuperPassRollup],
    gate_algo: str,
    cov_bounds: Sequence[float],
    gated_metrics: Sequence[TrackedMetric] = GATED_METRICS,
    min_len: int = MIN_TREND_N,
) -> list[tuple[int, int]]:
    """Segment the series into maximal admissible plateaus.

    From each start, extend the window until admissibility fails. The longest
    passing span is one plateau. Then scanning resumes after that plateau.
    Plateaus shorter than ``min_len`` are impossible by construction.
    """
    n = len(series)
    plateaus: list[tuple[int, int]] = []
    start = 0
    while start <= n - min_len:
        hi: int | None = None
        for end in range(start + min_len, n + 1):
            if window_admissible(
                series, start, end, gate_algo, cov_bounds, gated_metrics
            ):
                hi = end
            else:
                break
        if hi is not None:
            plateaus.append((start, hi))
            start = hi
        else:
            start += 1
    return plateaus


def detect_level_shift(
    series: Sequence[SuperPassRollup],
    plateaus: Sequence[tuple[int, int]],
    cov_band: float = 0.05,
    baseline_idx: int = 0,
) -> Anomaly:
    """Flag a late TPOT level shift.

    A shift requires a later plateau to differ from ``baseline_idx`` by more
    than ``cov_band`` and a Pettitt change point on per-super-pass TPOT means.
    ``delta_pct`` > 0 means the later level is worse because TPOT rose.

    ``baseline_idx`` is the reported plateau. Degradation is measured relative
    to it. Earlier plateaus do not count as degradations.
    """
    found = [list(p) for p in plateaus]

    def _none(pet: dict | None = None) -> Anomaly:
        return Anomaly(
            detected=False,
            change_point_sp=None,
            delta_pct=0.0,
            pettitt=pet,
            plateaus=found,
        )

    if len(plateaus) <= baseline_idx + 1:
        return _none()

    def _tpot_mean(lo: int, hi: int) -> float:
        vals = pooled(series, lo, hi, "tpot_ns")
        return sum(vals) / len(vals) if vals else 0.0

    first_mean = _tpot_mean(*plateaus[baseline_idx])
    if first_mean <= 0:
        return _none()
    sp_means = [
        (sum(sp.tpot_ns) / len(sp.tpot_ns)) if sp.tpot_ns else 0.0 for sp in series
    ]
    pet = pettitt(sp_means)
    for lo, hi in plateaus[baseline_idx + 1 :]:
        rel = (_tpot_mean(lo, hi) - first_mean) / first_mean
        if abs(rel) > cov_band and pet["significant"]:
            return Anomaly(
                detected=True,
                change_point_sp=pet["change_point"],
                delta_pct=rel * 100.0,
                pettitt=pet,
                plateaus=found,
            )
    return _none(pet)


def global_trend(
    series: Sequence[SuperPassRollup],
    from_idx: int,
    gate_algo: str,
    metrics: Sequence[TrackedMetric] = DRIFT_WATCH_METRICS,
) -> dict[str, Verdict]:
    """Return trend verdicts from plateau onset to the run end.

    A local plateau can be flat while a metric climbs later. This whole-tail
    test catches slow drift missed by short per-window gates. It watches TPOT
    and TTFT, so TTFT saturation is surfaced without gating admissibility.
    """
    gate = ALGORITHMS[gate_algo]
    out: dict[str, Verdict] = {}
    for m in metrics:
        traj = super_pass_percentile_series(
            series[from_idx:], m.source_attr, m.percentile
        )
        out[m.key] = gate(traj).verdict if len(traj) >= MIN_TREND_N else "insufficient"
    return out


def adaptive_warmup(
    series: Sequence[SuperPassRollup],
    driver: str = "tpot_p50",
    band: float = 0.05,
    min_warmup: int = 1,
    max_frac: float = 0.5,
) -> int:
    """Drop leading super-passes that are outside the steady band.

    The steady level is the driver's median over the back half of the series.
    Leading super-passes are cropped while their driver value is more than
    ``band`` away from that level in either direction.

    The check is symmetric because TPOT ramps up to steady while TTFT decays
    down. The crop is capped at ``max_frac`` of the run.
    """
    m = _METRIC_BY_KEY[driver]
    vals = super_pass_percentile_series(series, m.source_attr, m.percentile)
    n = len(vals)
    if n < MIN_TREND_N:
        return min_warmup
    steady = median(vals[n // 2 :]) or ZERO_MEDIAN_FLOOR
    cap = max(min_warmup, int(n * max_frac))
    w = 0
    while w < cap and abs(vals[w] - steady) / steady > band:
        w += 1
    return max(min_warmup, w)


def min_steady_duration(
    series: Sequence[SuperPassRollup], lo: int, hi: int
) -> ShortWindow:
    """Return the required steady-window wall time for ``[lo, hi)`` (§5.5).

    The requirement is ``max(precision, relaxation, floor)``. ``is_short``
    compares that requirement to the window's offered-load span. This matches
    the throughput denominator used for the reported TPS.

    All inputs come from the window itself. A high-throughput window with a
    short offered span can therefore require many more super-passes than the
    trend floor.
    """
    window = series[lo:hi]
    tpot_p50 = [
        percentile_lower(sorted(sp.tpot_ns), 0.50) for sp in window if sp.tpot_ns
    ]
    cov_b = cov(tpot_p50) if len(tpot_p50) >= 2 else 0.0
    spans = [
        (sp.last_issue_ns - sp.first_issue_ns) / 1e9
        for sp in window
        if sp.last_issue_ns > sp.first_issue_ns >= 0
    ]
    tau_sp = median(spans) if spans else 0.0
    lat = pooled(series, lo, hi, "latency_ns")
    l_p90 = percentile_lower(sorted(lat), 0.90) / 1e9 if lat else 0.0

    kstar = max(math.ceil((CI_Z_95 * cov_b / MIN_DUR_EPS) ** 2), MIN_DUR_KSTAR_FLOOR)
    # The precision term binds only when noise demands more batches than the
    # trend floor. At k* == MIN_TREND_N, the trend requirement is already met.
    # Then relaxation and floor control duration. This keeps a clean minimal
    # plateau valid when it meets the wall-time floor.
    t_prec = kstar * tau_sp if kstar > MIN_DUR_KSTAR_FLOOR else 0.0
    t_relax = MIN_DUR_RELAX_MULT * l_p90
    min_s = max(t_prec, t_relax, MIN_DUR_FLOOR_S)
    dominant = (
        "precision"
        if min_s == t_prec
        else "relaxation"
        if min_s == t_relax
        else "floor"
    )
    window_dur = window_issue_span_ns(series, lo, hi) / 1e9
    return ShortWindow(
        is_short=window_dur < min_s,
        window_duration_s=window_dur,
        min_duration_s=min_s,
        dominant=dominant,
        t_precision_s=t_prec,
        t_relaxation_s=t_relax,
        t_floor_s=MIN_DUR_FLOOR_S,
        kstar=kstar,
        cov_b=cov_b,
        tau_sp_s=tau_sp,
        l_p90_s=l_p90,
    )


def compute_steady_state_metrics(
    full_series: Sequence[SuperPassRollup],
    *,
    superpass_size: int,
    warmup: int | str = "auto",
    cov_bounds: Sequence[float] = DEFAULT_COV_BOUNDS,
    warmup_driver: str = "tpot_p50",
    warmup_band: float = 0.05,
    gate_algo: str = "mk_hamed_rao",
    gated_metrics: Sequence[TrackedMetric] = GATED_METRICS,
    enforce_min_duration: bool = True,
) -> SteadyState:
    """Select and summarize the first steady plateau after warmup.

    This is pure over a ``SuperPassRollup`` series. The live collector and the
    event-log replay therefore reach the same verdict. Window indices are
    relative to the post-warmup series.

    ``warmup`` is ``"auto"`` for a data-driven crop on ``warmup_driver``, or a
    fixed super-pass count. ``superpass_size`` is passed in because a partial
    last bucket cannot reveal it.

    With ``enforce_min_duration=True``, a plateau below the §5.5 wall-time
    minimum is rejected with ``found=False``. When disabled, the plateau is
    reported and ``short_window`` carries the advisory detail.
    """
    if isinstance(warmup, int) and warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    resolved_warmup = (
        adaptive_warmup(full_series, warmup_driver, warmup_band)
        if warmup == "auto"
        else int(warmup)
    )
    series = full_series[resolved_warmup:] if resolved_warmup < len(full_series) else []
    plateaus = segment_plateaus(series, gate_algo, cov_bounds, gated_metrics)
    if not plateaus:
        gt = global_trend(series, 0, gate_algo)  # drift-watch set (TPOT + TTFT)
        # No plateau was admissible, so report whether trend or scatter failed.
        # If the whole run is trend-steady, report CoV over that same span. If
        # it drifts, there is no steady span for CoV to describe.
        drifters = gated_trend_drifters(series, gate_algo, gated_metrics)
        steady_throughout = drifters == []
        reason = "no admissible steady plateau"
        if steady_throughout:
            reason = (
                "no admissible steady plateau: the run is trend-steady across all "
                f"{len(series)} super-passes, but its CoV exceeds {max(cov_bounds)}"
            )
        elif drifters:
            reason = (
                "no admissible steady plateau: "
                f"{', '.join(drifters)} trends across the run"
            )
        return SteadyState(
            superpass_size=superpass_size,
            n_super_passes=len(full_series),
            warmup=resolved_warmup,
            found=False,
            reason=reason,
            window=None,
            ttft=None,
            tpot=None,
            osl=None,
            latency=None,
            tps=None,
            cov=cov_table(series, cov_bounds) if steady_throughout else {},
            cov_basis=(
                CovBasis(sp_lo=0, sp_hi=len(series), n_super_passes=len(series))
                if steady_throughout
                else None
            ),
            anomaly=detect_level_shift(series, plateaus),
            short_window=None,
            global_trend=gt,
            drifting_up=[k for k, v in gt.items() if v == "up"],
        )
    # Min-duration selection. When enforced, report the first plateau that
    # clears the gate. Earlier plateaus are too brief to certify. If none
    # qualify, reject and report the longest candidate for context. When the
    # gate is disabled, report the first plateau with an advisory.
    shorts = [min_steady_duration(series, lo, hi) for lo, hi in plateaus]
    if enforce_min_duration:
        sel = next((i for i, sw in enumerate(shorts) if not sw.is_short), None)
    else:
        sel = 0
    reject_all_short = sel is None
    report_idx = (
        sel
        if sel is not None
        else max(range(len(plateaus)), key=lambda i: shorts[i].window_duration_s)
    )
    short = shorts[report_idx]
    lo, hi = plateaus[report_idx]  # the reported steady state
    anomaly = detect_level_shift(series, plateaus, baseline_idx=report_idx)
    gt = global_trend(series, lo, gate_algo)  # drift-watch set (TPOT + TTFT)
    ttft = pooled(series, lo, hi, "ttft_ns")
    tpot = pooled(series, lo, hi, "tpot_ns")
    osl = pooled(series, lo, hi, "osl")
    latency = pooled(series, lo, hi, "latency_ns")
    mean_tpot = sum(tpot) / len(tpot) if tpot else 0.0
    sp_tpot_means = [
        sum(sp.tpot_ns) / len(sp.tpot_ns) for sp in series[lo:hi] if sp.tpot_ns
    ]
    tpot_ci = (
        batch_means_ci(sp_tpot_means)
        if len(sp_tpot_means) >= 2
        else (mean_tpot, mean_tpot)
    )
    # per-user TPS = 1e9/TPOT is monotone-decreasing, so invert the CI bounds.
    per_user_ci = [per_user_tps(tpot_ci[1]), per_user_tps(tpot_ci[0])]
    # Aggregate TPS uses tokens divided by offered-load issue span (§5.1), not
    # completion span. Including drain would inflate the denominator and deflate
    # TPS on high-tail workloads.
    #
    # The CI uses a batch-means half-width from per-super-pass throughput,
    # centered on the aggregate point. Per-super-pass issue spans exclude
    # inter-super-pass gaps, so their mean need not equal the aggregate.
    system = system_tps(
        pooled_out_tokens(series, lo, hi), window_issue_span_ns(series, lo, hi)
    )
    sp_system = [
        system_tps(sp.out_tokens, sp.last_issue_ns - sp.first_issue_ns)
        for sp in series[lo:hi]
        if sp.last_issue_ns > sp.first_issue_ns >= 0
    ]
    if len(sp_system) >= 2:
        clo, chi = batch_means_ci(sp_system)
        half = (chi - clo) / 2.0
        system_ci = [system - half, system + half]
    else:
        system_ci = [system, system]
    skipped_short = sum(1 for i in range(report_idx) if shorts[i].is_short)
    # All admissible plateaus can be too brief to certify. Keep the longest
    # candidate in the result for context, but report no steady state.
    short_reason: str | None = (
        (
            f"all {len(plateaus)} admissible plateau(s) too short: longest "
            f"{short.window_duration_s:.0f}s < {short.min_duration_s:.0f}s required "
            f"({short.dominant}-dominated)"
        )
        if reject_all_short
        else None
    )
    return SteadyState(
        superpass_size=superpass_size,
        n_super_passes=len(full_series),
        warmup=resolved_warmup,
        found=not reject_all_short,
        reason=short_reason,
        window=SteadyWindow(
            sp_lo=lo,
            sp_hi=hi,
            n_super_passes=hi - lo,
            n_samples=len(ttft),
            start_ns=series[lo].first_issue_ns,
            end_ns=series[hi - 1].last_issue_ns,
            plateau_index=report_idx,
            n_plateaus=len(plateaus),
            skipped_short=skipped_short,
        ),
        ttft=summarize(ttft) if ttft else None,
        tpot=summarize(tpot) if tpot else None,
        osl=summarize(osl) if osl else None,
        latency=summarize(latency) if latency else None,
        cov=cov_table(series[lo:hi], cov_bounds),
        cov_basis=None,  # the reported window is the basis
        tps=TpsBlock(
            per_user=per_user_tps(mean_tpot),
            per_user_ci=per_user_ci,
            system=system,
            system_ci=system_ci,
        ),
        anomaly=anomaly,
        short_window=short,
        global_trend=gt,
        drifting_up=[k for k, v in gt.items() if v == "up"],
    )


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #
class DiagnosticsResult(TypedDict):
    """Standalone CLI result: the verdict plus debug tables.

    Benchmark runs produce only ``steady_state``. The CLI also includes
    ``drift``, a whole-run trend scan used to explain ``found: false`` verdicts.
    """

    steady_state: SteadyState
    drift: dict[str, dict[str, Verdict]]  # metric key -> algorithm -> verdict


def _drift_verdicts(trajectory: Sequence[float]) -> dict[str, Verdict]:
    return {name: fn(trajectory).verdict for name, fn in ALGORITHMS.items()}


def run(
    events_path: str,
    superpass_size: int,
    count_tokens: Callable[[list[str]], list[int]],
    warmup: int | str = "auto",
    cov_bounds: Sequence[float] = DEFAULT_COV_BOUNDS,
    trend_gate: str = "mk_hamed_rao",
    tokenize_batch_size: int = TOKENIZE_BATCH_SIZE,
    warmup_band: float = 0.05,
    warmup_driver: str = "tpot_p50",
    enforce_min_duration: bool = True,
) -> DiagnosticsResult:
    """Reconstruct the series from an event log, then analyse and diagnose it.

    ``superpass_size`` is a count of samples.
    """
    series = build_super_pass_series(
        events_path, superpass_size, count_tokens, tokenize_batch_size
    )
    ss = compute_steady_state_metrics(
        series,
        superpass_size=superpass_size,
        warmup=warmup,
        cov_bounds=cov_bounds,
        warmup_driver=warmup_driver,
        warmup_band=warmup_band,
        gate_algo=trend_gate,
        enforce_min_duration=enforce_min_duration,
    )
    post = series[ss.warmup :]
    return {
        "steady_state": ss,
        "drift": {
            m.key: _drift_verdicts(
                super_pass_percentile_series(post, m.source_attr, m.percentile)
            )
            for m in TRACKED_METRICS
        },
    }


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
_VERDICT_GLYPH = {
    "up": "^ up",
    "down": "v down",
    "steady": "= steady",
    "insufficient": ". n/a",
}


def _pass_glyph(ok: bool | None) -> str:
    if ok is None:
        return "n/a"
    return "PASS" if ok else "fail"


def _fmt_ms(ns: float) -> str:
    return f"{ns / 1e6:.2f}ms"


def _render_steady_state(ss: SteadyState) -> list[str]:
    out = ["=== STEADY STATE (headline) ==="]
    if not ss.found:
        out.append(f"  not found: {ss.reason}")
    else:
        w = ss.window
        tps = ss.tps
        assert w is not None and tps is not None
        out.append(
            f"  window: super-passes {w.sp_lo}..{w.sp_hi - 1} (post-warmup), "
            f"{w.n_samples} samples"
        )
        if w.skipped_short > 0:
            out.append(
                f"  note: skipped {w.skipped_short} earlier plateau(s) below "
                f"min-duration; reporting plateau {w.plateau_index + 1} of "
                f"{w.n_plateaus}"
            )
        out.append(
            f"  TPS per-user: {tps.per_user:8.1f} tok/s/user  "
            f"CI [{tps.per_user_ci[0]:.1f}, {tps.per_user_ci[1]:.1f}]"
        )
        out.append(
            f"  TPS system:   {tps.system:8.1f} tok/s        "
            f"CI [{tps.system_ci[0]:.1f}, {tps.system_ci[1]:.1f}]"
        )
        for name, stat in (("ttft", ss.ttft), ("tpot", ss.tpot)):
            if stat:
                out.append(
                    f"  {name.upper():4} p50 {_fmt_ms(stat['p50'])}"
                    f"  p90 {_fmt_ms(stat['p90'])}"
                    f"  p99 {_fmt_ms(stat['p99'])}"
                    f"  mean {_fmt_ms(stat['mean'])}"
                )
    sw = ss.short_window
    if ss.found and sw is not None and sw.is_short:
        # Reached only with the min-duration gate disabled (--no-min-duration); the
        # enforced path reports found=False with the same numbers in `reason`.
        out.append(
            f"  WARNING: Window too short -- {sw.window_duration_s:.0f}s steady vs "
            f"{sw.min_duration_s:.0f}s desired ({sw.dominant}-dominated); "
            f"the steady number is a best-effort estimate over too little wall-time"
        )
    if ss.drifting_up:
        out.append(
            f"  WARNING: {', '.join(ss.drifting_up)} drifting UP over the rest of the "
            f"run -- the window is a local plateau; global steady state is questionable"
        )
    an = ss.anomaly
    if an.detected:
        out.append(
            f"  ANOMALY: level shift at super-pass {an.change_point_sp}, "
            f"TPOT {an.delta_pct:+.1f}% toward end of run (likely degradation)"
        )
    return out


def _render_cov_table(
    label: str, table: dict[str, CovCell], cov_bounds: Sequence[float]
) -> list[str]:
    bound_hdr = "  ".join(f"cov<={b}" for b in cov_bounds)
    lines = [label, f"  {'metric':<12} {'gate':<5} {'CoV':>8}   {bound_hdr}"]
    for m in TRACKED_METRICS:
        cell = table[m.key]
        covs = f"{cell.cov:.4f}" if cell.cov is not None else "   n/a"
        passes = "  ".join(f"{_pass_glyph(cell.passes[str(b)]):>7}" for b in cov_bounds)
        lines.append(
            f"  {m.key:<12} {'gate' if m.gated else 'diag':<5} {covs:>8}   {passes}"
        )
    return lines


def render_text(result: DiagnosticsResult, cov_bounds: Sequence[float]) -> str:
    ss = result["steady_state"]
    lines = [
        f"super-passes: {ss.n_super_passes} "
        f"(size {ss.superpass_size}, warmup {ss.warmup})",
        "",
    ]
    lines.extend(_render_steady_state(ss))
    if ss.cov:
        basis = ss.cov_basis
        title = (
            "CoV inside the reported window"
            if basis is None
            else (
                "CoV over the longest trend-steady span "
                f"({basis.n_super_passes} super-passes; no window was admissible)"
            )
        )
        lines.append("")
        lines.extend(_render_cov_table(title, ss.cov, cov_bounds))
    lines.append("")
    lines.append("--- diagnostics ---")
    lines.append("")
    lines.append("drift (whole-run trend per metric)")
    algos = list(ALGORITHMS)
    lines.append(f"  {'metric':<12} " + "  ".join(f"{a:>16}" for a in algos))
    for m in TRACKED_METRICS:
        whole = result["drift"][m.key]
        cells = "  ".join(f"{_VERDICT_GLYPH[whole[a]]:>16}" for a in algos)
        lines.append(f"  {m.key:<12} {cells}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _make_token_counter(
    tokenizer_id: str, trust_remote_code: bool = False
) -> Callable[[list[str]], list[int]]:
    tok = AutoTokenizer.from_pretrained(
        tokenizer_id, trust_remote_code=trust_remote_code
    )

    def count(texts: list[str]) -> list[int]:
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in enc]

    return count


def _parse_float_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _warmup_arg(s: str) -> int | str:
    return "auto" if s == "auto" else int(s)


def _agentic_warning() -> str:
    msg = [
        "AGENTIC STEADY-STATE DETECTION IS NOT YET SUPPORTED.",
        "",
        "The NATL metric above is EXPERIMENTAL and REQUIRES FURTHER STUDY.",
        "Per-turn TTFT/TPOT do NOT converge on agentic multi-turn runs (structural",
        "server-side queue/eviction variance), so the standard steady-state window",
        "does not apply. NATL (per-trajectory throughput) is a promising candidate but",
        "is NOT validated for official reporting.",
        "",
        "DO NOT use these numbers for submissions or gating decisions.",
    ]
    w = max(len(m) for m in msg) + 6
    bar = "#" * w
    out = ["", bar, bar]
    out += ["##  " + m.ljust(w - 6) + "##" for m in msg]
    out += [bar, bar, ""]
    return "\n".join(out)


def render_natl(r: dict) -> str:
    d = r["distribution"]
    return "\n".join(
        [
            "=== NATL (agentic per-trajectory throughput) ===",
            f"  trajectories: {r['n_trajectories']}  (super-pass = "
            f"{r['superpass_trajectories']} trajectories -> {r['n_super_passes']} "
            "super-passes)",
            f"  NATL tok/s: p10 {d['p10']:.1f}  p50 {d['p50']:.1f}  p90 {d['p90']:.1f}  "
            f"p99 {d['p99']:.1f}  mean {d['mean']:.1f}  (distribution CoV {d['cov']:.2f})",
            f"  steadiness: across-super-pass CoV {r['across_cov']:.3f}  (batch-means SE "
            f"{r['batch_means_se_pct']:.1f}%)  -> "
            f"{'STEADY' if r['found'] else 'NOT steady'}",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("target", help="run directory OR path to events.jsonl")
    ap.add_argument(
        "--model",
        default=None,
        help="model name (auto-detected from config if omitted)",
    )
    ap.add_argument(
        "--tokenizer", default=None, help="HF tokenizer id/dir (overrides the registry)"
    )
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow the tokenizer's custom code (needed for Kimi-K3 and similar)",
    )
    ap.add_argument(
        "--dataset-size",
        type=int,
        default=None,
        help="samples per dataset pass (overrides the phase_start announcement)",
    )
    ap.add_argument(
        "--superpass-size",
        type=int,
        default=None,
        help="samples (or, agentic: trajectories) per super-pass",
    )
    ap.add_argument(
        "--profile",
        default=None,
        choices=list(PROFILES),
        help="workload profile (auto-selected from the run's load pattern)",
    )
    ap.add_argument(
        "--warmup",
        type=_warmup_arg,
        default="auto",
        help="'auto' (data-driven crop) or a fixed super-pass count",
    )
    ap.add_argument("--warmup-band", type=float, default=0.05)
    ap.add_argument("--warmup-driver", default=None, choices=list(_METRIC_BY_KEY))
    ap.add_argument("--cov-bounds", type=_parse_float_list, default=None)
    ap.add_argument("--trend-gate", default="mk_hamed_rao", choices=list(ALGORITHMS))
    ap.add_argument("--tokenize-batch-size", type=int, default=None)
    ap.add_argument(
        "--no-min-duration",
        dest="enforce_min_duration",
        action="store_false",
        help="do not hard-reject a steady window shorter than the required min-duration "
        "(§5.5); instead report it with a 'Window too short' warning and the desired "
        "duration",
    )
    ap.add_argument(
        "--json", dest="json_out", default=None, help="write JSON blob here"
    )
    args = ap.parse_args(argv)

    events, cfg_path = find_run_files(args.target)
    cfg = read_run_config(cfg_path)

    # tokenizer: explicit flag > model registry > config tokenizer path
    model = args.model or cfg["model"]
    reg = resolve_tokenizer(model) if model else None
    if args.tokenizer:
        tokenizer, trust = args.tokenizer, args.trust_remote_code
    elif reg is not None:
        tokenizer, trust = reg
    elif model and "/" in model:
        tokenizer, trust = model, args.trust_remote_code
    else:
        ap.error(
            "could not resolve a tokenizer; pass --model (a known model) or --tokenizer"
        )

    profile = (
        PROFILES[args.profile]
        if args.profile
        else profile_for_load_pattern(cfg["load_pattern"] or "concurrency")
    )
    if profile is None:
        ap.error(
            f"unclassified load pattern {cfg['load_pattern']!r}; pass --profile "
            f"({', '.join(PROFILES)})"
        )
    if profile.note:
        print(f"[profile: {profile.name}] {profile.note}\n", file=sys.stderr)

    cov_bounds = args.cov_bounds or list(profile.cov_bounds)
    warmup_driver = args.warmup_driver or profile.warmup_driver
    flush = args.tokenize_batch_size or profile.tokenize_batch_size
    count_tokens = _make_token_counter(tokenizer, trust)

    if profile.metric == "natl":
        sp_traj = args.superpass_size or profile.superpass_size or 32
        pairs = build_trajectory_natl(events, count_tokens, flush)
        natl = build_natl_result(pairs, sp_traj, tuple(cov_bounds))
        print(render_natl(natl))
        if args.json_out:
            with open(args.json_out, "w") as fh:
                json.dump(natl, fh, indent=2)
            print(f"wrote {args.json_out}", file=sys.stderr)
        print(_agentic_warning())
        return 0

    # window profiles (concurrency / poisson / offline): super-pass = samples
    size = (
        args.superpass_size or args.dataset_size or superpass_size_from_events(events)
    )
    if not size or size <= 0:
        ap.error(
            "could not resolve dataset/super-pass size; the log carries no "
            "performance session.phase_start, so pass --superpass-size"
        )
    result = run(
        events,
        superpass_size=int(size),
        count_tokens=count_tokens,
        warmup=args.warmup,
        cov_bounds=cov_bounds,
        trend_gate=args.trend_gate,
        tokenize_batch_size=flush,
        warmup_band=args.warmup_band,
        warmup_driver=warmup_driver,
        enforce_min_duration=args.enforce_min_duration,
    )
    print(render_text(result, cov_bounds))
    if args.json_out:
        # msgspec encodes the Struct tree and validates its shape.
        with open(args.json_out, "wb") as fh:
            fh.write(msgspec.json.format(msgspec.json.encode(result)))
        print(f"\nwrote {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
