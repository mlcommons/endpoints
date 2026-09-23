# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Steady-state / drift diagnostics from a benchmark run's ``events.jsonl``.

Parses the event log itself: the wire shapes are defined by the product's
``core/record.py`` (event names, ``EventRecord`` fields) and ``core/types.py``
(``TextModelOutput`` array layout); the parse here mirrors them and is pinned by
tests/unit/metrics/test_steady_state_diagnostics.py.

What it reconstructs (per performance-tracked sample):
  - ttft_ns = recv_first.ts - issued.ts
  - tpot_ns = (complete.ts - recv_first.ts) / tokens(text_after_first_chunk)
Token counts use plain tokenization of ``text_after_first_chunk``. The live metrics
aggregator instead tokenizes reasoning/tool-call outputs via the chat-template path
(``apply_chat_template``), so absolute TPOT ms here can differ from a run's report for
reasoning models. CoV and the trend tests are scale-invariant, so the steady/drift
diagnosis is unaffected -- only the absolute TPOT magnitude shifts.

Samples are bucketed into super-passes by issue order (``--superpass-size`` samples per
super-pass, default = ``--dataset-size``), giving a per-super-pass trajectory for each
metric*percentile.

The **headline** output is the ``steady_state`` block: the first steady plateau (grow-
from-left segmentation, admissible = trend-steady + within a CoV bound on the gated
metrics), summarized with TTFT/TPOT histograms + percentiles and per-user / system TPS
with batch-means confidence intervals. A staircase level-shift toward the end of the run
(multi-plateau difference corroborated by a Pettitt change-point) is flagged as an
``anomaly`` rather than hidden. See docs/steady-state-detection.md.

Below the headline the tool prints diagnostics: a CoV pass/fail table per requested
window size and a whole-run trend summary. The headline itself carries the CoV of the
reported window, which is the number that justifies the verdict.

Admissibility gates on TPOT at p50 + p90 only (decode-rate steadiness). TTFT is a
diagnostic: shown in the headline percentiles and the whole-run trend, and it raises the
Drifting-Up warning, but it does not gate a window — at high concurrency its tail variance
is structural (prefill/dataset-ISL skew + queue), not decode un-steadiness. p99 and
end-to-end latency are diagnostic too (latency's variation tracks the OSL mix).

usage (auto-detects tokenizer, dataset size, and workload profile from the run's
config.yaml / run_meta.json sidecars; see the model registry + PROFILES below):
  python -m inference_endpoint.metrics.steady_state_diagnostics <run_dir>       # 0 flags
  python -m inference_endpoint.metrics.steady_state_diagnostics <events.jsonl> \
      --model kimi-k3                                                          # 1 flag
Every derived setting has an explicit override (--tokenizer, --dataset-size,
--superpass-size, --profile, --cov-bounds, --window-sizes, --warmup, --json, ...).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist, median, pstdev
from typing import Literal, NamedTuple, TypedDict

import yaml
from transformers import AutoTokenizer

# --------------------------------------------------------------------------- #
# Event wire constants (mirror core/record.py category.value topics)
# --------------------------------------------------------------------------- #
EV_START_TRACKING = "session.start_performance_tracking"
EV_STOP_TRACKING = "session.stop_performance_tracking"
EV_ISSUED = "sample.issued"
EV_RECV_FIRST = "sample.recv_first"
EV_COMPLETE = "sample.complete"

# Below this many super-passes a trend test is statistically meaningless.
MIN_TREND_N = 4

# slope-vs-scatter thresholds (mirror the reference drift detector): a metric drifts
# when the run-length change is a large fraction of its level AND large vs the residual
# scatter around the fitted line.
REL_DRIFT_THRESHOLD = 0.15
SNR_THRESHOLD = 2.0

# Effect-size floor for a trend to disqualify a steady window during plateau segmentation
# (§5.5). The rank-based trend gate is significance-only: over a long window it flags a
# practically negligible monotonic drift (a couple of percent end-to-end) as a "trend" and
# fragments a genuinely steady run into many sub-window plateaus. A window is broken on
# trend only when the drift is BOTH significant AND at least this fraction end-to-end;
# below it, the drift is within noise and the window holds. CoV still guards genuine
# variance/choppiness, so this relaxes over-sensitive trend fragmentation only — not
# scatter. Calibrated on the corpus: over-fragmenting breaks had |rel_drift| <= 0.03,
# while real drift and level shifts far exceed 0.05. Cumulative, so a persistent slow
# drift still breaks once the growing window's total change crosses the floor.
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
DEFAULT_WINDOW_SIZES: tuple[int, ...] = (4, 6, 8)

# Minimum steady-window TIME duration (docs/steady-state-detection.md §5.5). Passing the
# >= MIN_TREND_N super-pass floor is not enough: at high throughput a window of a few
# super-passes is only seconds of wall-time, far too brief to certify steadiness. The
# required duration is max(precision, relaxation, floor):
#   precision  = k*.tau_sp,  k* = max(ceil((1.96.CoV_b / eps)^2), MIN_TREND_N)  (batch-means +-eps)
#   relaxation = mult . p90(sample latency)   (queue/KV-eviction transient safety; p90 not
#                p99, so a few extreme-outlier request lifetimes don't dominate the floor)
#   floor      = MLPerf min-duration floor
MIN_DUR_EPS = 0.05
MIN_DUR_RELAX_MULT = 5.0
MIN_DUR_FLOOR_S = 600.0
# k* floor: >= MIN_TREND_N batches. The batch-means CI already widens when CoV is high,
# so k* self-raises for noisy metrics; a >4 floor would only over-penalize clean runs.
MIN_DUR_KSTAR_FLOOR = MIN_TREND_N

# A per-super-pass metric trajectory is classified into one of these states.
Verdict = Literal["up", "down", "steady", "insufficient"]


class Anomaly(TypedDict):
    detected: bool
    change_point_sp: int | None
    delta_pct: float  # signed % change of the later TPOT level vs the first plateau
    pettitt: dict | None
    plateaus: list[list[int]]


class SteadyWindow(TypedDict):
    sp_lo: int  # post-warmup super-pass index (inclusive)
    sp_hi: int  # exclusive
    n_super_passes: int
    n_samples: int
    start_ns: int  # first issue of series[sp_lo] (duration is derivable; no field)
    end_ns: int  # last issue of series[sp_hi - 1]
    plateau_index: int  # 0-based index of this plateau among all admissible plateaus
    n_plateaus: int  # total admissible plateaus found
    skipped_short: int  # earlier plateaus skipped for failing the min-duration gate


class TpsBlock(TypedDict):
    per_user: float  # 1e9 / mean(TPOT ns) = output tok/s/user
    per_user_ci: list[float]  # [lo, hi]
    system: float  # total output tokens / window wall-clock
    system_ci: list[float]


class ShortWindow(TypedDict):
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


class CovCell(TypedDict):
    gated: bool
    n: int  # super-passes the CoV was computed over
    cov: float | None  # None when there are < 2 points
    passes: dict[str, bool | None]  # cov-bound (as str) -> pass / fail / inconclusive


class SteadyState(TypedDict):
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
    # CoV per tracked metric across the super-passes INSIDE the reported window --
    # the number that justifies the verdict. Empty when no window was reported.
    cov: dict[str, CovCell]
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


# Metric*percentile trajectories tracked. ``gated`` ones participate in convergence;
# everything else is diagnostic. TTFT is intentionally NOT gated (see GATED_METRICS).
TRACKED_METRICS: tuple[TrackedMetric, ...] = (
    TrackedMetric("ttft_p50", "ttft_ns", 0.50, False),
    TrackedMetric("ttft_p90", "ttft_ns", 0.90, False),
    TrackedMetric("tpot_p50", "tpot_ns", 0.50, True),
    TrackedMetric("tpot_p90", "tpot_ns", 0.90, True),
    TrackedMetric("ttft_p99", "ttft_ns", 0.99, False),
    TrackedMetric("tpot_p99", "tpot_ns", 0.99, False),
    # End-to-end sample latency (issue->complete). Diagnostic by default (its variance
    # tracks the OSL mix, §5.1); useful for agentic where per-turn TTFT is turbulent.
    TrackedMetric("latency_p50", "latency_ns", 0.50, False),
    TrackedMetric("latency_p90", "latency_ns", 0.90, False),
    # Warm-turn TTFT (agentic turn >= 2): cold first-turn prefill discarded.
    TrackedMetric("ttft_warm_p50", "ttft_warm_ns", 0.50, False),
    TrackedMetric("ttft_warm_p90", "ttft_warm_ns", 0.90, False),
)

# Admissibility gate: TPOT p50/p90 only (decode-rate steadiness). TTFT is deliberately
# NOT a hard gate — at high concurrency its tail variance (prefill time tracking dataset
# ISL skew + queue wait) is structural, not decode un-steadiness, and fragments genuinely
# steady runs (docs/steady-state-detection.md §5.5). Measured: two ~800s steady-TPOT runs
# (c7k, c22k) were rejected purely by TTFT-tail fragmentation; TPOT-only recovers them.
GATED_METRICS: tuple[TrackedMetric, ...] = tuple(m for m in TRACKED_METRICS if m.gated)
# Metrics watched for the whole-run Drifting-Up *warning*: the gated TPOT pair plus TTFT
# p50/p90. TTFT is soft here — a genuine TTFT saturation drift is still surfaced (warning),
# just never a hard reject.
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
        True,
        "offline: issue-time throughput degenerate (all issued at t=0); the window and "
        "drain are completion-based (partial support -- system TPS is unreliable).",
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


def profile_for_load_pattern(lp: str) -> Profile:
    return PROFILES[_LOAD_PATTERN_PROFILE.get(lp, "concurrency")]


def find_run_files(target: str) -> tuple[str, str | None, str | None]:
    """Resolve (events.jsonl, config.yaml|None, run_meta.json|None) from a run dir or an
    events.jsonl path. A directory is searched in ``./`` and ``./client/``."""
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
    d = os.path.dirname(events)
    cfg = os.path.join(d, "config.yaml")
    meta = os.path.join(d, "run_meta.json")
    return (
        events,
        cfg if os.path.isfile(cfg) else None,
        meta if os.path.isfile(meta) else None,
    )


def read_run_config(
    config_yaml_path: str | None, run_meta_json_path: str | None
) -> dict:
    """Best-effort model / load-pattern / dataset-size from a run's sidecar files."""
    out: dict = {
        "model": None,
        "load_pattern": None,
        "dataset_size": None,
        "num_trajectories": None,
    }
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
    if run_meta_json_path and os.path.isfile(run_meta_json_path):
        try:
            with open(run_meta_json_path) as fh:
                meta = json.load(fh)
            if isinstance(meta, dict) and meta.get("dataset_size"):
                out["dataset_size"] = int(meta["dataset_size"])
        except Exception:  # malformed JSON -> leave dataset_size None
            pass
    return out


# --------------------------------------------------------------------------- #
# NATL (agentic per-trajectory throughput) -- EXPERIMENTAL, see the CLI warning
# --------------------------------------------------------------------------- #
def full_output_text(data: object) -> str:
    """All generated text for a turn (reasoning + output + tool_calls), for OSL counting."""
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
    """Per-trajectory NATL = sum(output tokens) / sum(e2e latency s), keyed by
    conversation_id. Returns (last_complete_ns, natl) sorted by completion."""
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
# TextModelOutput.text_after_first_chunk, ported to the parsed JSON array
# --------------------------------------------------------------------------- #
def text_after_first_chunk(data: object) -> str:
    """Return output text excluding the first streamed chunk (the TPOT numerator).

    ``data`` is the COMPLETE event payload: ``[tag, output, reasoning?, tool_calls?]``
    with trailing defaults omitted (msgspec ``array_like`` + ``omit_defaults``). ``output``
    and ``reasoning`` are each either a string (non-streaming) or a list of chunks
    (streaming). Mirrors ``TextModelOutput.text_after_first_chunk`` in core/types.py.
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
@dataclass(slots=True)
class SuperPassRollup:
    index: int
    n_issued: int = 0  # per-super-pass sample count (coverage / bucketing invariant)
    first_issue_ns: int = -1  # earliest issue ts (offered-load span start)
    last_issue_ns: int = -1  # latest issue ts (offered-load span end; throughput denom)
    last_event_ns: int = -1  # latest event ts incl. completions (drain-inclusive end)
    ttft_ns: list[float] = field(default_factory=list)
    ttft_warm_ns: list[float] = field(default_factory=list)  # turn >= 2 (KV-cache warm)
    tpot_ns: list[float] = field(default_factory=list)
    latency_ns: list[float] = field(default_factory=list)  # issue -> complete (e2e)
    osl: list[float] = field(default_factory=list)  # per-sample post-first-chunk tokens
    out_tokens: int = 0


@dataclass(slots=True)
class _PendingRow:
    """In-flight sample state during the parse, keyed by uuid until COMPLETE."""

    sp_index: int
    issue_ns: int
    recv_first_ns: int | None = None


def build_super_pass_series(
    events_path: str,
    superpass_size: int,
    count_tokens: Callable[[list[str]], list[int]],
    flush_size: int = TOKENIZE_BATCH_SIZE,
) -> list[SuperPassRollup]:
    """Bucket performance-tracked samples into super-passes by issue order.

    ``count_tokens`` maps a batch of texts to token counts; injected so the parse is
    testable without a real tokenizer and the tokenizer is swappable. ``flush_size``
    caps how many output texts are buffered before a tokenizer flush — lower it to bound
    peak memory on long reasoning outputs (fewer texts held, smaller tokenizer calls).
    """
    if superpass_size <= 0:
        raise ValueError("superpass_size must be positive")
    series: list[SuperPassRollup] = []
    rows: dict[str, _PendingRow] = {}
    tracking = False
    issue_counter = 0
    batch_uuids: list[str] = []
    batch_texts: list[str] = []
    pending_tpot: dict[str, tuple[int, float]] = {}

    def _ensure(idx: int) -> SuperPassRollup:
        while len(series) <= idx:
            series.append(SuperPassRollup(index=len(series)))
        return series[idx]

    def flush_tpot() -> None:
        if batch_texts:
            counts = count_tokens(batch_texts)
            for uuid, cnt in zip(batch_uuids, counts, strict=True):
                sp_idx, delta = pending_tpot.pop(uuid)
                if cnt > 0:
                    series[sp_idx].tpot_ns.append(delta / cnt)
                    series[sp_idx].osl.append(float(cnt))
                    series[sp_idx].out_tokens += cnt
        batch_uuids.clear()
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
            ts = rec.get("timestamp_ns")  # may be absent on a truncated/partial event
            if et == EV_START_TRACKING:
                tracking = True
            elif et == EV_STOP_TRACKING:
                tracking = False
            elif et == EV_ISSUED:
                uuid = rec.get("sample_uuid")
                if not tracking or not uuid or ts is None:
                    continue
                existing = rows.get(uuid)
                if existing is not None:
                    existing.issue_ns = ts  # retry: refresh issue ts only
                    sp = series[existing.sp_index]
                    sp.last_issue_ns = max(sp.last_issue_ns, ts)
                    sp.last_event_ns = max(sp.last_event_ns, ts)
                    continue
                sp_idx = issue_counter // superpass_size
                issue_counter += 1
                rows[uuid] = _PendingRow(sp_index=sp_idx, issue_ns=ts)
                sp = _ensure(sp_idx)
                sp.n_issued += 1
                if sp.first_issue_ns < 0:
                    sp.first_issue_ns = ts
                sp.last_issue_ns = max(sp.last_issue_ns, ts)
                sp.last_event_ns = max(sp.last_event_ns, ts)
            elif et == EV_RECV_FIRST:
                row = rows.get(rec.get("sample_uuid"))
                if row is not None and ts is not None:
                    series[row.sp_index].last_event_ns = max(
                        series[row.sp_index].last_event_ns, ts
                    )
                    # First recv_first only: a retried sample re-emits recv_first and
                    # must not contribute a second TTFT to the super-pass.
                    if row.recv_first_ns is None:
                        row.recv_first_ns = ts
                        ttft = float(ts - row.issue_ns)
                        sp = series[row.sp_index]
                        sp.ttft_ns.append(ttft)
                        # Warm-turn TTFT excludes the cold first turn of each agentic
                        # trajectory (turn 1 = no KV-cache hit). turn is None for
                        # single-turn workloads -> treated as warm (kept).
                        turn = rec.get("turn")
                        if turn is None or turn > 1:
                            sp.ttft_warm_ns.append(ttft)
            elif et == EV_COMPLETE:
                uuid = rec.get("sample_uuid")
                row = rows.pop(uuid, None)
                if row is None or ts is None:
                    continue
                sp = series[row.sp_index]
                sp.last_event_ns = max(sp.last_event_ns, ts)
                sp.latency_ns.append(
                    float(ts - row.issue_ns)
                )  # e2e, no recv_first needed
                if row.recv_first_ns is None:
                    continue
                text = text_after_first_chunk(rec.get("data"))
                if text:
                    pending_tpot[uuid] = (row.sp_index, float(ts - row.recv_first_ns))
                    batch_uuids.append(uuid)
                    batch_texts.append(text)
                    if len(batch_texts) >= flush_size:
                        flush_tpot()
    flush_tpot()
    return series


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

    This is the throughput denominator (§5.1): the drain lives after the last issue, so
    counting to the last completion would inflate the denominator and deflate TPS —
    badly so for high-tail workloads (long TTFT + decode).
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
    """Mann-Kendall with the Hamed-Rao autocorrelation variance correction.

    Inflates (or, for negatively autocorrelated data, deflates) the MK variance by an
    effective-sample-size factor computed from the significant autocorrelations of the
    data ranks, so serial correlation does not fake significance.
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
    # A non-positive effective-sample correction is degenerate (over-correction under
    # strong negative autocorrelation). Fall back to the uncorrected MK variance rather
    # than clamping to a sliver, which would collapse the variance and manufacture a
    # significant trend from essentially no evidence.
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
    """Signed end-to-end fractional change of the OLS trend line over the window.

    Matches ``slope_vs_scatter``'s ``rel_drift``; used as the effect-size floor that keeps
    a significant-but-negligible trend from fragmenting a steady plateau (§5.5).
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
    # Fewer than 2 points -> CoV is undefined; report inconclusive (None), never PASS,
    # so a short/empty window can't masquerade as steady.
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
        out[m.key] = {
            "gated": m.gated,
            "n": len(traj),
            "cov": cov(traj) if len(traj) >= 2 else None,
            "passes": {str(b): v for b, v in cov_pass_row(traj, bounds).items()},
        }
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


def window_admissible(
    series: Sequence[SuperPassRollup],
    lo: int,
    hi: int,
    gate_algo: str,
    cov_bounds: Sequence[float],
    gated_metrics: Sequence[TrackedMetric] = GATED_METRICS,
) -> bool:
    """True iff every gated metric is trend-steady and within the loosest CoV bound."""
    gate = ALGORITHMS[gate_algo]
    loosest = max(cov_bounds)
    for m in gated_metrics:
        traj = _window_percentile_series(series, lo, hi, m.source_attr, m.percentile)
        if traj is None or len(traj) < MIN_TREND_N:
            return False
        if (
            gate(traj).verdict != "steady"
            and abs(_rel_drift(traj)) >= TREND_REL_DRIFT_MIN
        ):
            # Significant trend AND practically large: a genuine drift/level-shift breaks
            # the window. A significant-but-negligible drift (< the effect-size floor) is
            # within noise and does not fragment the plateau; CoV below still guards scatter.
            return False
        if cov(traj) > loosest:
            return False
    return True


def segment_plateaus(
    series: Sequence[SuperPassRollup],
    gate_algo: str,
    cov_bounds: Sequence[float],
    gated_metrics: Sequence[TrackedMetric] = GATED_METRICS,
    min_len: int = MIN_TREND_N,
) -> list[tuple[int, int]]:
    """Grow-from-left segmentation into maximal admissible plateaus.

    From each start, extend the window until admissibility breaks (a staircase jump
    fails the CoV/trend gate); the maximal admissible span is one plateau, then resume
    past it. Plateaus shorter than ``min_len`` are impossible by construction.
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
    """Flag a staircase: a plateau *after* ``baseline_idx`` whose TPOT level differs from
    the reported plateau by more than ``cov_band``, corroborated by a Pettitt change-point
    on the per-super-pass TPOT means. ``delta_pct`` > 0 means the later level is worse
    (TPOT rose). ``baseline_idx`` is the index of the reported plateau: degradation is
    measured relative to it, and only plateaus after it count (earlier ones were skipped,
    not degradations)."""
    result: Anomaly = {
        "detected": False,
        "change_point_sp": None,
        "delta_pct": 0.0,
        "pettitt": None,
        "plateaus": [list(p) for p in plateaus],
    }
    if len(plateaus) <= baseline_idx + 1:
        return result

    def _tpot_mean(lo: int, hi: int) -> float:
        vals = pooled(series, lo, hi, "tpot_ns")
        return sum(vals) / len(vals) if vals else 0.0

    first_mean = _tpot_mean(*plateaus[baseline_idx])
    if first_mean <= 0:
        return result
    sp_means = [
        (sum(sp.tpot_ns) / len(sp.tpot_ns)) if sp.tpot_ns else 0.0 for sp in series
    ]
    pet = pettitt(sp_means)
    result["pettitt"] = pet
    for lo, hi in plateaus[baseline_idx + 1 :]:
        rel = (_tpot_mean(lo, hi) - first_mean) / first_mean
        if abs(rel) > cov_band and pet["significant"]:
            result["detected"] = True
            result["change_point_sp"] = pet["change_point"]
            result["delta_pct"] = rel * 100.0
            break
    return result


def global_trend(
    series: Sequence[SuperPassRollup],
    from_idx: int,
    gate_algo: str,
    metrics: Sequence[TrackedMetric] = DRIFT_WATCH_METRICS,
) -> dict[str, Verdict]:
    """Trend verdict per watched metric over ``series[from_idx:]`` (plateau onset to end).

    A window can be locally flat while a metric climbs across the rest of the run (a slow
    drift the short per-window gate misses); this whole-tail test catches it. Watches the
    drift set (TPOT + TTFT), so a soft TTFT saturation drift is surfaced even though TTFT
    does not gate admissibility.
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
    """Data-driven warmup crop: drop leading super-passes still off the steady level.

    The driver's steady level is estimated from the median of the series' back half;
    leading super-passes whose driver value is more than ``band`` (fractional) away from
    it — in *either* direction — are cropped. Symmetric because the natural driver, TPOT,
    ramps *up* to steady (unlike TTFT, which decays down). Capped at ``max_frac`` of the
    run so it can never crop everything.
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
    """Required steady-window wall-time for the window ``[lo, hi)`` (§5.5).

    ``max(precision, relaxation, floor)``. ``is_short`` compares it to the window's
    offered-load span (the throughput denominator, so it matches the TPS reported for the
    same window). All inputs come from the window itself, so a high-throughput window with
    a short offered span is correctly asked for far more super-passes than the trend floor.
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
    # The precision term only binds when the metric is noisy enough to demand MORE batches
    # than the trend floor. At the floor (k* == MIN_TREND_N), the >= MIN_TREND_N super-passes
    # already satisfy the trend requirement, so duration is governed by relaxation/floor
    # alone — a clean minimal plateau that meets the wall-time floor is valid, not rejected
    # by a precision term that degenerates to ~the window's own duration at 4 super-passes.
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
    return {
        "is_short": window_dur < min_s,
        "window_duration_s": window_dur,
        "min_duration_s": min_s,
        "dominant": dominant,
        "t_precision_s": t_prec,
        "t_relaxation_s": t_relax,
        "t_floor_s": MIN_DUR_FLOOR_S,
        "kstar": kstar,
        "cov_b": cov_b,
        "tau_sp_s": tau_sp,
        "l_p90_s": l_p90,
    }


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
    """Crop warmup, select the first steady plateau, and summarize it.

    Pure over a ``SuperPassRollup`` series, so the two producers -- the live
    collector in the metrics aggregator and ``build_super_pass_series`` over an
    archived ``events.jsonl`` -- reach the same verdict. Window indices are
    relative to the post-warmup series.

    ``warmup`` is ``"auto"`` (data-driven crop on ``warmup_driver``) or a fixed
    super-pass count. ``superpass_size`` is passed in rather than derived: it is
    carried in the result and a partial last bucket cannot reveal it.
    ``enforce_min_duration`` (default) hard-rejects a plateau whose wall-time is
    below the §5.5 minimum (``found=False``); when disabled the plateau is still
    reported and the ``short_window`` detail carries an advisory instead.
    """
    if isinstance(warmup, int) and warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    resolved_warmup = (
        adaptive_warmup(full_series, warmup_driver, warmup_band)
        if warmup == "auto"
        else int(warmup)
    )
    series = full_series[resolved_warmup:] if resolved_warmup < len(full_series) else []
    shape: dict = {
        "superpass_size": superpass_size,
        "n_super_passes": len(full_series),
        "warmup": resolved_warmup,
    }
    plateaus = segment_plateaus(series, gate_algo, cov_bounds, gated_metrics)
    if not plateaus:
        gt = global_trend(series, 0, gate_algo)  # drift-watch set (TPOT + TTFT)
        return {
            **shape,  # type: ignore[typeddict-item]
            "found": False,
            "reason": "no admissible steady plateau",
            "window": None,
            "ttft": None,
            "tpot": None,
            "osl": None,
            "latency": None,
            "tps": None,
            "cov": {},
            "anomaly": detect_level_shift(series, plateaus),
            "short_window": None,
            "global_trend": gt,
            "drifting_up": [k for k, v in gt.items() if v == "up"],
        }
    # Min-duration selection. When enforced, walk plateaus in order and report the FIRST
    # that clears the min-duration gate — skipping earlier plateaus too brief to certify.
    # If none qualify, reject, reporting the longest candidate for context. When the gate
    # is disabled, always report the first plateau (advisory only).
    shorts = [min_steady_duration(series, lo, hi) for lo, hi in plateaus]
    if enforce_min_duration:
        sel = next((i for i, sw in enumerate(shorts) if not sw["is_short"]), None)
    else:
        sel = 0
    reject_all_short = sel is None
    report_idx = (
        sel
        if sel is not None
        else max(range(len(plateaus)), key=lambda i: shorts[i]["window_duration_s"])
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
    # Aggregate tokens / offered-load (issue) span (§5.1) — NOT the completion span, which
    # would inflate the denominator with the drain and deflate TPS on high-tail workloads.
    # The CI is a batch-means half-width from per-super-pass throughput, centered on the
    # point (per-super-pass issue spans exclude inter-super-pass gaps, so their mean would
    # not equal the aggregate).
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
    skipped_short = sum(1 for i in range(report_idx) if shorts[i]["is_short"])
    ss: SteadyState = {
        **shape,  # type: ignore[typeddict-item]
        "found": True,
        "reason": None,
        "window": {
            "sp_lo": lo,
            "sp_hi": hi,
            "n_super_passes": hi - lo,
            "n_samples": len(ttft),
            "start_ns": series[lo].first_issue_ns,
            "end_ns": series[hi - 1].last_issue_ns,
            "plateau_index": report_idx,
            "n_plateaus": len(plateaus),
            "skipped_short": skipped_short,
        },
        "ttft": summarize(ttft) if ttft else None,
        "tpot": summarize(tpot) if tpot else None,
        "osl": summarize(osl) if osl else None,
        "latency": summarize(latency) if latency else None,
        "cov": cov_table(series[lo:hi], cov_bounds),
        "tps": {
            "per_user": per_user_tps(mean_tpot),
            "per_user_ci": per_user_ci,
            "system": system,
            "system_ci": system_ci,
        },
        "anomaly": anomaly,
        "short_window": short,
        "global_trend": gt,
        "drifting_up": [k for k, v in gt.items() if v == "up"],
    }
    if reject_all_short:
        # Every admissible plateau is genuine but too brief to certify. Keep the longest
        # candidate's window in the blob (informative), but report no steady state.
        ss["found"] = False
        ss["reason"] = (
            f"all {len(plateaus)} admissible plateau(s) too short: longest "
            f"{short['window_duration_s']:.0f}s < {short['min_duration_s']:.0f}s required "
            f"({short['dominant']}-dominated); pass --no-min-duration to override"
        )
    return ss


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #
class DiagnosticsResult(TypedDict):
    """The standalone CLI's blob: the verdict plus its debug tables.

    Only ``steady_state`` is produced in a benchmark run; ``cov`` and ``drift``
    are per-window-size scans that earn their keep when a verdict comes back
    ``found: false`` and someone has to work out why.
    """

    steady_state: SteadyState
    cov: dict[str, dict[str, CovCell]]  # window size (str) -> metric key -> cell
    drift: dict[str, dict[str, Verdict]]  # metric key -> algorithm -> verdict


def _drift_verdicts(trajectory: Sequence[float]) -> dict[str, Verdict]:
    return {name: fn(trajectory).verdict for name, fn in ALGORITHMS.items()}


def run(
    events_path: str,
    superpass_size: int,
    count_tokens: Callable[[list[str]], list[int]],
    window_sizes: Sequence[int] = DEFAULT_WINDOW_SIZES,
    warmup: int | str = "auto",
    cov_bounds: Sequence[float] = DEFAULT_COV_BOUNDS,
    trend_gate: str = "mk_hamed_rao",
    tokenize_batch_size: int = TOKENIZE_BATCH_SIZE,
    warmup_band: float = 0.05,
    warmup_driver: str = "tpot_p50",
    enforce_min_duration: bool = True,
) -> DiagnosticsResult:
    """Reconstruct the series from an event log, then analyse and diagnose it.

    ``window_sizes`` are counts of super-passes and drive the debug tables only;
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
    post = series[ss["warmup"] :]
    return {
        "steady_state": ss,
        "cov": {str(w): cov_table(post[-w:], cov_bounds) for w in window_sizes},
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
    if not ss["found"]:
        out.append(f"  not found: {ss['reason']}")
    else:
        w = ss["window"]
        tps = ss["tps"]
        assert w is not None and tps is not None
        out.append(
            f"  window: super-passes {w['sp_lo']}..{w['sp_hi'] - 1} (post-warmup), "
            f"{w['n_samples']} samples"
        )
        if w["skipped_short"] > 0:
            out.append(
                f"  note: skipped {w['skipped_short']} earlier plateau(s) below "
                f"min-duration; reporting plateau {w['plateau_index'] + 1} of "
                f"{w['n_plateaus']}"
            )
        out.append(
            f"  TPS per-user: {tps['per_user']:8.1f} tok/s/user  "
            f"CI [{tps['per_user_ci'][0]:.1f}, {tps['per_user_ci'][1]:.1f}]"
        )
        out.append(
            f"  TPS system:   {tps['system']:8.1f} tok/s        "
            f"CI [{tps['system_ci'][0]:.1f}, {tps['system_ci'][1]:.1f}]"
        )
        for name in ("ttft", "tpot"):
            s = ss[name]  # type: ignore[literal-required]
            if s:
                out.append(
                    f"  {name.upper():4} p50 {_fmt_ms(s['p50'])}  p90 {_fmt_ms(s['p90'])}"
                    f"  p99 {_fmt_ms(s['p99'])}"
                    f"  mean {_fmt_ms(s['mean'])}"
                )
    sw = ss["short_window"]
    if ss["found"] and sw is not None and sw["is_short"]:
        # Reached only with the min-duration gate disabled (--no-min-duration); the
        # enforced path reports found=False with the same numbers in `reason`.
        out.append(
            f"  WARNING: Window too short -- {sw['window_duration_s']:.0f}s steady vs "
            f"{sw['min_duration_s']:.0f}s desired ({sw['dominant']}-dominated); "
            f"the steady number is a best-effort estimate over too little wall-time"
        )
    if ss["drifting_up"]:
        out.append(
            f"  WARNING: {', '.join(ss['drifting_up'])} drifting UP over the rest of the "
            f"run -- the window is a local plateau; global steady state is questionable"
        )
    an = ss["anomaly"]
    if an["detected"]:
        out.append(
            f"  ANOMALY: level shift at super-pass {an['change_point_sp']}, "
            f"TPOT {an['delta_pct']:+.1f}% toward end of run (likely degradation)"
        )
    return out


def _render_cov_table(
    label: str, table: dict[str, CovCell], cov_bounds: Sequence[float]
) -> list[str]:
    bound_hdr = "  ".join(f"cov<={b}" for b in cov_bounds)
    lines = [label, f"  {'metric':<12} {'gate':<5} {'CoV':>8}   {bound_hdr}"]
    for m in TRACKED_METRICS:
        cell = table[m.key]
        covv = cell["cov"]
        covs = f"{covv:.4f}" if covv is not None else "   n/a"
        passes = "  ".join(
            f"{_pass_glyph(cell['passes'][str(b)]):>7}" for b in cov_bounds
        )
        lines.append(
            f"  {m.key:<12} {'gate' if m.gated else 'diag':<5} {covs:>8}   {passes}"
        )
    return lines


def render_text(result: DiagnosticsResult, cov_bounds: Sequence[float]) -> str:
    ss = result["steady_state"]
    lines = [
        f"super-passes: {ss['n_super_passes']} "
        f"(size {ss['superpass_size']}, warmup {ss['warmup']})",
        "",
    ]
    lines.extend(_render_steady_state(ss))
    if ss["cov"]:
        lines.append("")
        lines.extend(
            _render_cov_table("CoV inside the reported window", ss["cov"], cov_bounds)
        )
    lines.append("")
    lines.append("--- diagnostics ---")
    for w in sorted(result["cov"], key=int):
        lines.append("")
        lines.extend(
            _render_cov_table(
                f"CoV steadiness (trailing {w} super-passes)",
                result["cov"][w],
                cov_bounds,
            )
        )
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


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


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
        help="samples per dataset pass (overrides run_meta/config)",
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
        "--window-sizes",
        type=_parse_int_list,
        default=None,
        help="comma-separated window sizes, in super-passes",
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

    events, cfg_path, meta_path = find_run_files(args.target)
    cfg = read_run_config(cfg_path, meta_path)

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
    if profile.note:
        print(f"[profile: {profile.name}] {profile.note}\n", file=sys.stderr)

    cov_bounds = args.cov_bounds or list(profile.cov_bounds)
    warmup_driver = args.warmup_driver or profile.warmup_driver
    flush = args.tokenize_batch_size or profile.tokenize_batch_size
    window_sizes = args.window_sizes or list(DEFAULT_WINDOW_SIZES)
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
    size = args.superpass_size or args.dataset_size or cfg["dataset_size"]
    if not size or size <= 0:
        ap.error(
            "could not resolve dataset/super-pass size; pass --dataset-size or "
            "--superpass-size"
        )
    result = run(
        events,
        superpass_size=int(size),
        count_tokens=count_tokens,
        window_sizes=window_sizes,
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
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwrote {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
