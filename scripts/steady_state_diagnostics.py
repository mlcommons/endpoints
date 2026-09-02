#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["transformers>=4.40", "pyyaml>=6.0"]
# ///
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Steady-state / drift diagnostics from a benchmark run's ``events.jsonl``.

Self-contained: no ``inference_endpoint`` import, so it runs anywhere with just a
tokenizer available (``uv run scripts/steady_state_diagnostics.py ...``). The event
wire shapes it parses are defined by the product's ``core/record.py`` (event names,
``EventRecord`` fields) and ``core/types.py`` (``TextModelOutput`` array layout); the
parse here mirrors them and is pinned by tests/unit/scripts/test_steady_state_diagnostics.py.

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

Below the headline, per requested window size, the tool also prints diagnostics: a CoV
pass/fail table and a whole-run trend summary (the full rolling drift scan is in
``--json``).

Convergence metrics are TTFT/TPOT at p50 + p95; p99 is carried as an optional diagnostic
(shown, not gated). End-to-end latency is intentionally excluded (its variation tracks the
OSL mix, not system steadiness).

usage (auto-detects tokenizer, dataset size, and workload profile from the run's
config.yaml / run_meta.json sidecars; see the model registry + PROFILES below):
  uv run scripts/steady_state_diagnostics.py <run_dir>                 # 0 flags
  uv run scripts/steady_state_diagnostics.py <events.jsonl> --model kimi-k3   # 1 flag
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

# z for a two-sided 95% confidence interval (Hamed-Rao autocorrelation significance).
CI_Z_95 = 1.96
# Floor substituted for a zero median so relative-drift ratios stay finite.
ZERO_MEDIAN_FLOOR = 1e-9
# Texts buffered before a tokenizer flush during the parse.
TOKENIZE_BATCH_SIZE = 4096

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


class TpsBlock(TypedDict):
    per_user: float  # 1e9 / mean(TPOT ns) = output tok/s/user
    per_user_ci: list[float]  # [lo, hi]
    system: float  # total output tokens / window wall-clock
    system_ci: list[float]


class SteadyState(TypedDict):
    found: bool
    reason: str | None
    window: SteadyWindow | None
    ttft: dict | None  # summarize() output
    tpot: dict | None
    tps: TpsBlock | None
    anomaly: Anomaly
    global_trend: dict[
        str, Verdict
    ]  # gated metric -> trend from the plateau to run end
    drifting_up: list[str]  # gated metrics Drifting Up over the rest of the run


class TrackedMetric(NamedTuple):
    key: str  # display key, e.g. "ttft_p95"
    source_attr: str  # SuperPassRollup attribute holding the raw samples
    percentile: float
    gated: bool  # participates in the convergence gate (vs. diagnostic-only)


# Metric*percentile trajectories tracked. ``gated`` ones participate in convergence;
# p99 is diagnostic only.
TRACKED_METRICS: tuple[TrackedMetric, ...] = (
    TrackedMetric("ttft_p50", "ttft_ns", 0.50, True),
    TrackedMetric("ttft_p95", "ttft_ns", 0.95, True),
    TrackedMetric("tpot_p50", "tpot_ns", 0.50, True),
    TrackedMetric("tpot_p95", "tpot_ns", 0.95, True),
    TrackedMetric("ttft_p99", "ttft_ns", 0.99, False),
    TrackedMetric("tpot_p99", "tpot_ns", 0.99, False),
    # End-to-end sample latency (issue->complete). Diagnostic by default (its variance
    # tracks the OSL mix, §5.1); useful for agentic where per-turn TTFT is turbulent.
    TrackedMetric("latency_p50", "latency_ns", 0.50, False),
    TrackedMetric("latency_p90", "latency_ns", 0.90, False),
    TrackedMetric("latency_p95", "latency_ns", 0.95, False),
    # Warm-turn TTFT (agentic turn >= 2): cold first-turn prefill discarded.
    TrackedMetric("ttft_warm_p50", "ttft_warm_ns", 0.50, False),
    TrackedMetric("ttft_warm_p95", "ttft_warm_ns", 0.95, False),
)

# Metrics that gate admissibility (p50/p95); p99 is diagnostic-only.
GATED_METRICS: tuple[TrackedMetric, ...] = tuple(m for m in TRACKED_METRICS if m.gated)
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
        (0.03, 0.05, 0.08),
        "tpot_p50",
        4096,
        True,
    ),
    "poisson": Profile(
        "poisson",
        "window",
        "samples",
        None,
        (0.03, 0.05, 0.08),
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
        (0.03, 0.05, 0.08),
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
    """Count, mean, min/max, p50/p90/p95/p99 (nearest-rank-lower), and a histogram."""
    s = sorted(values)
    n = len(s)
    return {
        "count": n,
        "mean": sum(s) / n,
        "min": s[0],
        "max": s[-1],
        "p50": percentile_lower(s, 0.50),
        "p90": percentile_lower(s, 0.90),
        "p95": percentile_lower(s, 0.95),
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
def rolling_windows(n: int, window: int) -> list[tuple[int, int]]:
    if window <= 0 or window > n:
        return []
    return [(s, s + window) for s in range(0, n - window + 1)]


def cov_pass_row(
    values: Sequence[float], bounds: Sequence[float]
) -> dict[float, bool | None]:
    # Fewer than 2 points -> CoV is undefined; report inconclusive (None), never PASS,
    # so a short/empty window can't masquerade as steady.
    if len(values) < 2:
        return {b: None for b in bounds}
    c = cov(values)
    return {b: c <= b for b in bounds}


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
        if gate(traj).verdict != "steady":
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
) -> Anomaly:
    """Flag a staircase: a later plateau whose TPOT level differs from the first by
    more than ``cov_band``, corroborated by a Pettitt change-point on the per-super-pass
    TPOT means. ``delta_pct`` > 0 means the later level is worse (TPOT rose)."""
    result: Anomaly = {
        "detected": False,
        "change_point_sp": None,
        "delta_pct": 0.0,
        "pettitt": None,
        "plateaus": [list(p) for p in plateaus],
    }
    if len(plateaus) < 2:
        return result

    def _tpot_mean(lo: int, hi: int) -> float:
        vals = pooled(series, lo, hi, "tpot_ns")
        return sum(vals) / len(vals) if vals else 0.0

    first_mean = _tpot_mean(*plateaus[0])
    if first_mean <= 0:
        return result
    sp_means = [
        (sum(sp.tpot_ns) / len(sp.tpot_ns)) if sp.tpot_ns else 0.0 for sp in series
    ]
    pet = pettitt(sp_means)
    result["pettitt"] = pet
    for lo, hi in plateaus[1:]:
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
    gated_metrics: Sequence[TrackedMetric] = GATED_METRICS,
) -> dict[str, Verdict]:
    """Trend verdict per gated metric over ``series[from_idx:]`` (plateau onset to end).

    A window can be locally flat while the metric climbs across the rest of the run
    (a slow drift the short per-window gate misses); this whole-tail test catches it.
    """
    gate = ALGORITHMS[gate_algo]
    out: dict[str, Verdict] = {}
    for m in gated_metrics:
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


def build_steady_state(
    series: Sequence[SuperPassRollup],
    gate_algo: str = "mk_hamed_rao",
    cov_bounds: Sequence[float] = (0.03, 0.05, 0.08),
    gated_metrics: Sequence[TrackedMetric] = GATED_METRICS,
) -> SteadyState:
    """Select the first steady plateau and summarize it (window, TTFT/TPOT, TPS).

    ``series`` is the post-warmup super-pass series; window indices are relative to it.
    """
    plateaus = segment_plateaus(series, gate_algo, cov_bounds, gated_metrics)
    anomaly = detect_level_shift(series, plateaus)
    if not plateaus:
        gt = global_trend(series, 0, gate_algo, gated_metrics)
        return {
            "found": False,
            "reason": "no admissible steady plateau",
            "window": None,
            "ttft": None,
            "tpot": None,
            "tps": None,
            "anomaly": anomaly,
            "global_trend": gt,
            "drifting_up": [k for k, v in gt.items() if v == "up"],
        }
    lo, hi = plateaus[0]  # first plateau is the reported steady state
    gt = global_trend(series, lo, gate_algo, gated_metrics)
    ttft = pooled(series, lo, hi, "ttft_ns")
    tpot = pooled(series, lo, hi, "tpot_ns")
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
    return {
        "found": True,
        "reason": None,
        "window": {
            "sp_lo": lo,
            "sp_hi": hi,
            "n_super_passes": hi - lo,
            "n_samples": len(ttft),
        },
        "ttft": summarize(ttft) if ttft else None,
        "tpot": summarize(tpot) if tpot else None,
        "tps": {
            "per_user": per_user_tps(mean_tpot),
            "per_user_ci": per_user_ci,
            "system": system,
            "system_ci": system_ci,
        },
        "anomaly": anomaly,
        "global_trend": gt,
        "drifting_up": [k for k, v in gt.items() if v == "up"],
    }


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #
class CovCell(TypedDict):
    gated: bool
    n: int  # super-passes in the trailing window
    cov: float | None  # None when the window has < 2 points
    passes: dict[str, bool | None]  # cov-bound (as str) -> pass / fail / inconclusive


class RollingCell(TypedDict):
    window: list[int]  # [lo, hi) super-pass indices
    verdicts: dict[str, Verdict]  # algorithm name -> verdict


class DriftEntry(TypedDict):
    whole_run: dict[str, Verdict]  # algorithm name -> verdict over the full trajectory
    rolling: list[RollingCell]


class DiagnosticsResult(TypedDict):
    n_super_passes: int
    superpass_size: int
    warmup: int  # resolved super-pass crop count
    warmup_mode: str  # "auto" (adaptive) or "fixed"
    n_post_warmup: int
    metrics: list[str]
    gated_metrics: list[str]
    trajectories: dict[
        str, list[float]
    ]  # metric key -> per-super-pass percentile series
    cov: dict[str, dict[str, CovCell]]  # window size (str) -> metric key -> cell
    drift: dict[str, dict[str, DriftEntry]]  # window size (str) -> metric key -> entry
    steady_state: SteadyState  # the headline: first steady plateau + TPS + anomaly
    per_super_pass: list[dict]  # raw post-warmup per-super-pass rollups (for plotting)
    alpha: float


def _drift_verdicts(trajectory: Sequence[float]) -> dict[str, Verdict]:
    return {name: fn(trajectory).verdict for name, fn in ALGORITHMS.items()}


def per_super_pass_diagnostics(series: Sequence[SuperPassRollup]) -> list[dict]:
    """Raw per-super-pass rollups (timestamps, tokens, percentiles) for plotting."""
    out: list[dict] = []
    for i, sp in enumerate(series):
        tt = sorted(sp.ttft_ns)
        tp = sorted(sp.tpot_ns)
        out.append(
            {
                "i": i,
                "n_issued": sp.n_issued,
                "out_tokens": sp.out_tokens,
                "first_issue_ns": sp.first_issue_ns,
                "last_issue_ns": sp.last_issue_ns,
                "last_event_ns": sp.last_event_ns,
                "ttft_p50": percentile_lower(tt, 0.50) if tt else None,
                "ttft_p95": percentile_lower(tt, 0.95) if tt else None,
                "tpot_p50": percentile_lower(tp, 0.50) if tp else None,
                "tpot_p95": percentile_lower(tp, 0.95) if tp else None,
            }
        )
    return out


def run(
    events_path: str,
    superpass_size: int,
    count_tokens: Callable[[list[str]], list[int]],
    window_sizes: Sequence[int] = (4, 5),
    warmup: int | str = "auto",
    cov_bounds: Sequence[float] = (0.03, 0.05, 0.08),
    alpha: float = 0.05,
    trend_gate: str = "mk_hamed_rao",
    tokenize_batch_size: int = TOKENIZE_BATCH_SIZE,
    warmup_band: float = 0.05,
    warmup_driver: str = "tpot_p50",
) -> DiagnosticsResult:
    """Build the full diagnostics result (the ``--json`` blob).

    ``window_sizes`` are counts of super-passes; ``superpass_size`` is a count of samples.
    ``warmup`` is either ``"auto"`` (data-driven crop via ``adaptive_warmup`` on the
    ``warmup_driver`` metric) or a fixed super-pass count. ``trend_gate`` names the trend
    algorithm gating admissibility.
    """
    if isinstance(warmup, int) and warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")
    series = build_super_pass_series(
        events_path, superpass_size, count_tokens, tokenize_batch_size
    )
    if warmup == "auto":
        resolved_warmup = adaptive_warmup(series, warmup_driver, warmup_band)
        warmup_mode = "auto"
    else:
        resolved_warmup = int(warmup)
        warmup_mode = "fixed"
    post = series[resolved_warmup:] if resolved_warmup < len(series) else []
    trajectories = {
        m.key: super_pass_percentile_series(post, m.source_attr, m.percentile)
        for m in TRACKED_METRICS
    }

    result: DiagnosticsResult = {
        "n_super_passes": len(series),
        "superpass_size": superpass_size,
        "warmup": resolved_warmup,
        "warmup_mode": warmup_mode,
        "n_post_warmup": len(post),
        "metrics": [m.key for m in TRACKED_METRICS],
        "gated_metrics": [m.key for m in TRACKED_METRICS if m.gated],
        "trajectories": trajectories,
        "cov": {},
        "drift": {},
        "steady_state": build_steady_state(post, trend_gate, cov_bounds),
        "per_super_pass": per_super_pass_diagnostics(post),
        "alpha": alpha,
    }

    for w in window_sizes:
        cov_tbl: dict[str, CovCell] = {}
        drift_tbl: dict[str, DriftEntry] = {}
        for m in TRACKED_METRICS:
            traj = trajectories[m.key]
            trailing = traj[-w:] if len(traj) >= w else traj
            cov_tbl[m.key] = {
                "gated": m.gated,
                "n": len(trailing),
                "cov": cov(trailing) if len(trailing) >= 2 else None,
                "passes": {
                    str(b): v for b, v in cov_pass_row(trailing, cov_bounds).items()
                },
            }
            rolling: list[RollingCell] = [
                {"window": [lo, hi], "verdicts": _drift_verdicts(traj[lo:hi])}
                for lo, hi in rolling_windows(len(traj), w)
            ]
            drift_tbl[m.key] = {"whole_run": _drift_verdicts(traj), "rolling": rolling}
        result["cov"][str(w)] = cov_tbl
        result["drift"][str(w)] = drift_tbl

    return result


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
                    f"  p95 {_fmt_ms(s['p95'])}  p99 {_fmt_ms(s['p99'])}"
                    f"  mean {_fmt_ms(s['mean'])}"
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


def render_text(result: DiagnosticsResult, cov_bounds: Sequence[float]) -> str:
    lines: list[str] = []
    lines.append(
        f"super-passes: {result['n_super_passes']} "
        f"(size {result['superpass_size']}, warmup {result['warmup']} "
        f"[{result['warmup_mode']}], post-warmup {result['n_post_warmup']})"
    )
    lines.append("")
    lines.extend(_render_steady_state(result["steady_state"]))
    lines.append("")
    lines.append("--- diagnostics (per window size) ---")
    gated = set(result["gated_metrics"])
    metrics = result["metrics"]
    bound_hdr = "  ".join(f"cov<={b}" for b in cov_bounds)
    for w in sorted(result["cov"], key=int):
        lines.append("")
        lines.append(f"=== window size {w} ===")
        lines.append("")
        lines.append(f"CoV steadiness (trailing {w} super-passes)")
        lines.append(f"  {'metric':<12} {'gate':<5} {'CoV':>8}   {bound_hdr}")
        for key in metrics:
            cell = result["cov"][w][key]
            covv = cell["cov"]
            covs = f"{covv:.4f}" if covv is not None else "   n/a"
            passes = "  ".join(
                f"{_pass_glyph(cell['passes'][str(b)]):>7}" for b in cov_bounds
            )
            tag = "gate" if key in gated else "diag"
            lines.append(f"  {key:<12} {tag:<5} {covs:>8}   {passes}")

        lines.append("")
        lines.append("drift (whole-run trend per metric; rolling scan is in --json)")
        algos = list(ALGORITHMS)
        lines.append(f"  {'metric':<12} " + "  ".join(f"{a:>16}" for a in algos))
        for key in metrics:
            whole = result["drift"][w][key]["whole_run"]
            cells = "  ".join(f"{_VERDICT_GLYPH[whole[a]]:>16}" for a in algos)
            lines.append(f"  {key:<12} {cells}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _make_token_counter(
    tokenizer_id: str, trust_remote_code: bool = False
) -> Callable[[list[str]], list[int]]:
    from transformers import AutoTokenizer

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
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--trend-gate", default="mk_hamed_rao", choices=list(ALGORITHMS))
    ap.add_argument("--tokenize-batch-size", type=int, default=None)
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
    window_sizes = args.window_sizes or [4, 6, 8]
    count_tokens = _make_token_counter(tokenizer, trust)

    if profile.metric == "natl":
        sp_traj = args.superpass_size or profile.superpass_size or 32
        pairs = build_trajectory_natl(events, count_tokens, flush)
        result = build_natl_result(pairs, sp_traj, tuple(cov_bounds))
        print(render_natl(result))
        if args.json_out:
            with open(args.json_out, "w") as fh:
                json.dump(result, fh, indent=2)
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
        alpha=args.alpha,
        trend_gate=args.trend_gate,
        tokenize_batch_size=flush,
        warmup_band=args.warmup_band,
        warmup_driver=warmup_driver,
    )
    print(render_text(result, cov_bounds))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwrote {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
