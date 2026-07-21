# Steady-State Windowing (Milestone 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build strategy-agnostic offline tooling that re-ingests a run's `events.jsonl`, buckets samples into super-passes, computes issue-time steady-state windows, and runs an A/B sweep of fixed-budget vs adaptive-CoV stopping rules against the full-series asymptote.

**Architecture:** Cold-path only, mirroring `scripts/early_stopping_estimate_from_events.py`. A new pure-function library `src/inference_endpoint/metrics/steady_state/` (series → window → stopping → harness) plus a CLI `scripts/steady_state_from_events.py`. No hot-path change; no report.py wiring (deferred to Milestone 2).

**Tech Stack:** Python 3.12, `msgspec` (event decode), `dataclass(slots=True, frozen=True)` for value types, `pytest` (unit), reuses `metrics/early_stopping.py`.

## Global Constraints

- License header (Apache-2.0 SPDX) required on every new `.py` file. The pre-commit hook `scripts/add_license_header.py` adds it; run `uv run pre-commit run --all-files` and re-stage.
- `uv run pre-commit run --all-files` MUST pass before every commit (ruff, ruff-format, mypy, prettier, license). If a hook modifies files, re-stage and re-commit.
- Run all commands with `uv run` (never bare `python3`).
- Every test function needs a marker: `@pytest.mark.unit`.
- Imports at top of file only (no lazy imports).
- Scope: `concurrency` load pattern only. This milestone does NOT modify `report.py` or any hot-path/aggregator code.
- Event decode MUST use the product's own `EventRecord` typed decoder, exactly as the ES script does:
  `msgspec.json.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)`.

## Established interfaces (read-only, from the existing codebase)

- `inference_endpoint.core.record`: `EventRecord` (fields `event_type`, `timestamp_ns: int`, `sample_uuid: str`, `data`), `EventType`, `SampleEventType.{ISSUED,RECV_FIRST,COMPLETE}`, `SessionEventType.{START_PERFORMANCE_TRACKING,STOP_PERFORMANCE_TRACKING}`.
- `inference_endpoint.core.types.TextModelOutput` with `.tool_calls` and `.text_after_first_chunk() -> str`.
- `inference_endpoint.metrics.early_stopping`: `es_percentile_estimate(sorted_latencies: Sequence[float], percentile: float, confidence: float = CONFIDENCE) -> EarlyStoppingResult` where `EarlyStoppingResult.estimate` is `float | None` (None ⇒ too few samples). Also `CONFIDENCE: float`.
- `inference_endpoint.async_utils.services.metrics_aggregator.token_metrics`: `load_reference_backend(path)` and `encode_lengths(backend, texts) -> list[int]` (TPOT/OSL token counting).

## Super-pass definition (used throughout)

```
S = ceil(concurrency / dataset_size)          # passes per super-pass
super_pass_samples = dataset_size * S         # issued samples per super-pass
```

Samples are assigned to a super-pass by **issue order** among performance-tracked
`ISSUED` events: the k-th first-time-issued sample (0-based counter `k`) belongs to
`super_pass = k // super_pass_samples`. Duplicate `ISSUED` for an already-seen
`sample_uuid` (a retry) does NOT advance the counter.

---

### Task 1: Super-pass series builder

**Files:**

- Create: `src/inference_endpoint/metrics/steady_state/__init__.py`
- Create: `src/inference_endpoint/metrics/steady_state/series.py`
- Test: `tests/unit/metrics/steady_state/test_series.py`
- Create (empty package marker): `tests/unit/metrics/steady_state/__init__.py` (only if the test dir needs it — match sibling test dirs; most use none, so create only if `tests/unit/metrics/` has `__init__.py`).

**Interfaces:**

- Produces:

  - `SuperPassRollup` (frozen slotted dataclass): `index: int`, `n_issued: int`, `first_issue_ns: int`, `last_issue_ns: int`, `ttft_ns: list[float]`, `latency_ns: list[float]`, `tpot_ns: list[float]`, `out_tokens: int`.
  - `super_pass_size(dataset_size: int, concurrency: int) -> int` returns `dataset_size * ceil(concurrency / dataset_size)`.
  - `build_super_pass_series(events_path, dataset_size, concurrency, count_tokens=None) -> list[SuperPassRollup]` where `count_tokens: Callable[[list[str]], list[int]] | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/metrics/steady_state/test_series.py
# SPDX header added by pre-commit.
import msgspec.json
import pytest
from inference_endpoint.core.record import (
    EventRecord,
    EventType,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.metrics.steady_state.series import (
    build_super_pass_series,
    super_pass_size,
)

_ENC = msgspec.json.Encoder(enc_hook=EventType.encode_hook)


def _write_events(path, records):
    with open(path, "wb") as f:
        for r in records:
            f.write(_ENC.encode(r))
            f.write(b"\n")


def _sample(uuid, issue_ns, first_ns, complete_ns):
    return [
        EventRecord(event_type=SampleEventType.ISSUED, sample_uuid=uuid, timestamp_ns=issue_ns),
        EventRecord(event_type=SampleEventType.RECV_FIRST, sample_uuid=uuid, timestamp_ns=first_ns),
        EventRecord(event_type=SampleEventType.COMPLETE, sample_uuid=uuid, timestamp_ns=complete_ns),
    ]


@pytest.mark.unit
def test_super_pass_size_rounds_up():
    # dataset_size 4, concurrency 10 -> S=ceil(10/4)=3 -> 12 samples/super-pass
    assert super_pass_size(4, 10) == 12
    # exact multiple
    assert super_pass_size(5, 10) == 10


@pytest.mark.unit
def test_buckets_by_issue_order(tmp_path):
    # dataset_size=2, concurrency=2 -> S=1 -> 2 samples per super-pass.
    p = tmp_path / "events.jsonl"
    recs = [EventRecord(event_type=SessionEventType.START_PERFORMANCE_TRACKING, timestamp_ns=0)]
    # 4 issued samples -> super-pass 0 = {s0,s1}, super-pass 1 = {s2,s3}
    for i in range(4):
        recs += _sample(f"s{i}", issue_ns=100 + i, first_ns=200 + i, complete_ns=500 + i)
    recs.append(EventRecord(event_type=SessionEventType.STOP_PERFORMANCE_TRACKING, timestamp_ns=999))
    _write_events(p, recs)

    series = build_super_pass_series(str(p), dataset_size=2, concurrency=2)

    assert [sp.index for sp in series] == [0, 1]
    assert series[0].n_issued == 2
    assert series[0].first_issue_ns == 100 and series[0].last_issue_ns == 101
    assert series[0].ttft_ns == [100.0, 100.0]  # first_ns - issue_ns = 100 each
    assert series[1].n_issued == 2
    assert series[1].first_issue_ns == 102


@pytest.mark.unit
def test_untracked_issued_excluded(tmp_path):
    p = tmp_path / "events.jsonl"
    # one sample issued BEFORE tracking starts (warmup) must be dropped
    recs = _sample("warm", 1, 2, 3)
    recs.append(EventRecord(event_type=SessionEventType.START_PERFORMANCE_TRACKING, timestamp_ns=10))
    recs += _sample("s0", 100, 200, 500)
    recs += _sample("s1", 101, 201, 501)
    recs.append(EventRecord(event_type=SessionEventType.STOP_PERFORMANCE_TRACKING, timestamp_ns=999))
    _write_events(p, recs)

    series = build_super_pass_series(str(p), dataset_size=2, concurrency=2)
    assert len(series) == 1
    assert series[0].n_issued == 2  # warmup sample excluded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/metrics/steady_state/test_series.py -v`
Expected: FAIL with `ModuleNotFoundError: inference_endpoint.metrics.steady_state.series`

- [ ] **Step 3: Write minimal implementation**

```python
# src/inference_endpoint/metrics/steady_state/__init__.py
# (empty; SPDX header added by pre-commit)
```

```python
# src/inference_endpoint/metrics/steady_state/series.py
# SPDX header added by pre-commit.
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import msgspec
import msgspec.json
from inference_endpoint.core.record import (
    EventRecord,
    EventType,
    SampleEventType,
    SessionEventType,
)
from inference_endpoint.core.types import TextModelOutput

_DECODER = msgspec.json.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)


@dataclass(slots=True)
class SuperPassRollup:
    index: int
    n_issued: int = 0
    first_issue_ns: int = -1
    last_issue_ns: int = -1
    ttft_ns: list[float] = field(default_factory=list)
    latency_ns: list[float] = field(default_factory=list)
    tpot_ns: list[float] = field(default_factory=list)
    out_tokens: int = 0


def super_pass_size(dataset_size: int, concurrency: int) -> int:
    if dataset_size <= 0 or concurrency <= 0:
        raise ValueError("dataset_size and concurrency must be positive")
    s = -(-concurrency // dataset_size)  # ceil division
    return dataset_size * s


def build_super_pass_series(
    events_path: str,
    dataset_size: int,
    concurrency: int,
    count_tokens: Callable[[list[str]], list[int]] | None = None,
) -> list[SuperPassRollup]:
    """Bucket performance-tracked samples into super-passes by issue order.

    Each sample's TTFT/latency is attributed to the super-pass its ISSUED event
    fell into; completions after STOP_PERFORMANCE_TRACKING still count for rows
    that exist. TPOT/out_tokens are populated only when ``count_tokens`` is given.
    """
    sp_samples = super_pass_size(dataset_size, concurrency)
    series: list[SuperPassRollup] = []
    # uuid -> (super_pass_index, issue_ns, recv_first_ns|None, tpot_text|None)
    rows: dict[str, list] = {}
    tracking = False
    issue_counter = 0
    # buffered TPOT token counting (uuid-ordered)
    batch_uuids: list[str] = []
    batch_texts: list[str] = []
    # TPOT pairs a token count (deferred, batched) with the complete-recv delta;
    # the row is popped at COMPLETE, so stash (sp_idx, delta_ns) until flush.
    pending_tpot: dict[str, tuple[int, float]] = {}

    def _ensure(idx: int) -> SuperPassRollup:
        while len(series) <= idx:
            series.append(SuperPassRollup(index=len(series)))
        return series[idx]

    def flush_tpot() -> None:
        if not batch_texts or count_tokens is None:
            batch_uuids.clear()
            batch_texts.clear()
            return
        for uuid, cnt in zip(batch_uuids, count_tokens(batch_texts), strict=True):
            sp_idx, delta = pending_tpot.pop(uuid)
            if cnt > 0:
                series[sp_idx].tpot_ns.append(delta / cnt)
                series[sp_idx].out_tokens += cnt
        batch_uuids.clear()
        batch_texts.clear()

    with open(events_path, "rb") as f:
        for line in f:
            try:
                rec = _DECODER.decode(line)
            except (msgspec.DecodeError, NotImplementedError):
                continue
            et = rec.event_type
            if et is SessionEventType.START_PERFORMANCE_TRACKING:
                tracking = True
            elif et is SessionEventType.STOP_PERFORMANCE_TRACKING:
                tracking = False
            elif et is SampleEventType.ISSUED:
                if not tracking or not rec.sample_uuid:
                    continue
                existing = rows.get(rec.sample_uuid)
                if existing is not None:
                    existing[1] = rec.timestamp_ns  # retry: refresh issue ts only
                    continue
                sp_idx = issue_counter // sp_samples
                issue_counter += 1
                rows[rec.sample_uuid] = [sp_idx, rec.timestamp_ns, None, None]
                sp = _ensure(sp_idx)
                sp.n_issued += 1
                if sp.first_issue_ns < 0:
                    sp.first_issue_ns = rec.timestamp_ns
                sp.last_issue_ns = rec.timestamp_ns
            elif et is SampleEventType.RECV_FIRST:
                row = rows.get(rec.sample_uuid)
                if row is not None:
                    row[2] = rec.timestamp_ns
                    series[row[0]].ttft_ns.append(float(rec.timestamp_ns - row[1]))
            elif et is SampleEventType.COMPLETE:
                row = rows.pop(rec.sample_uuid, None)
                if row is None:
                    continue
                sp_idx, issue_ns, recv_ns, _ = row
                series[sp_idx].latency_ns.append(float(rec.timestamp_ns - issue_ns))
                if count_tokens is None or recv_ns is None:
                    continue
                data = rec.data
                if not isinstance(data, TextModelOutput) or data.tool_calls:
                    continue
                text = data.text_after_first_chunk()
                if text:
                    pending_tpot[rec.sample_uuid] = (
                        sp_idx,
                        float(rec.timestamp_ns - recv_ns),
                    )
                    batch_uuids.append(rec.sample_uuid)
                    batch_texts.append(text)
                    if len(batch_texts) >= 4096:
                        flush_tpot()
    flush_tpot()
    return series
```

Note: delete the unused `flush()` stub — keep only `flush_tpot()`. (Written above for clarity; final code has one flush function.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/metrics/steady_state/test_series.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Pre-commit + commit**

```bash
uv run pre-commit run --all-files
git add src/inference_endpoint/metrics/steady_state/ tests/unit/metrics/steady_state/
git commit -m "feat(metrics): super-pass series builder for steady-state windowing"
```

---

### Task 2: Windowing core

**Files:**

- Create: `src/inference_endpoint/metrics/steady_state/window.py`
- Test: `tests/unit/metrics/steady_state/test_window.py`

**Interfaces:**

- Consumes: `SuperPassRollup` list from Task 1; `es_percentile_estimate`, `CONFIDENCE` from `metrics.early_stopping`.
- Produces:
  - `WindowMetrics` (frozen slotted dataclass): `sp_start: int`, `sp_end: int`, `n_samples: int`, `issue_span_ns: int`, `qps: float`, `token_tps: float | None`, `ttft: dict[float, float]`, `tpot: dict[float, float] | None`, `latency: dict[float, float]`, `valid: dict[str, bool]`.
  - `windowed_metrics(series, sp_start, sp_end, percentiles=(0.5, 0.9, 0.99), es_percentile=0.99, confidence=CONFIDENCE) -> WindowMetrics`.
  - `percentile_lower(sorted_values: list[float], p: float) -> float` (numpy-free `method="lower"`: index `int(p*(n-1))`).

Windowing rules: measured set = all samples issued in `[sp_start, sp_end)`.
`issue_span_ns = series[sp_end-1].last_issue_ns - series[sp_start].first_issue_ns`.
`qps = n_samples / (issue_span_ns / 1e9)`. `token_tps = sum(out_tokens)/(span_s)` or None if no tokens. Latency/ttft/tpot percentiles over the concatenated raw lists. `valid[series_name] = es_percentile_estimate(sorted_vals, es_percentile, confidence).estimate is not None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/metrics/steady_state/test_window.py
import pytest
from inference_endpoint.metrics.steady_state.series import SuperPassRollup
from inference_endpoint.metrics.steady_state.window import (
    percentile_lower,
    windowed_metrics,
)


@pytest.mark.unit
def test_percentile_lower():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile_lower(vals, 0.5) == 3.0  # index int(0.5*4)=2
    assert percentile_lower(vals, 0.99) == 5.0  # index int(0.99*4)=3 -> 4.0? verify
    # int(0.99*4)=int(3.96)=3 -> vals[3]=4.0
    assert percentile_lower(vals, 0.99) == 4.0


@pytest.mark.unit
def test_window_issue_span_excludes_drain():
    # Two super-passes. Latencies vary but issue span is issue-time only.
    sp0 = SuperPassRollup(
        index=0, n_issued=2, first_issue_ns=0, last_issue_ns=1_000_000_000,
        ttft_ns=[10.0, 20.0], latency_ns=[100.0, 200.0], tpot_ns=[], out_tokens=0,
    )
    sp1 = SuperPassRollup(
        index=1, n_issued=2, first_issue_ns=1_000_000_001, last_issue_ns=2_000_000_000,
        ttft_ns=[30.0, 40.0], latency_ns=[300.0, 400.0], tpot_ns=[], out_tokens=0,
    )
    m = windowed_metrics([sp0, sp1], 0, 2)
    # span = 2e9 - 0 = 2s; n=4 -> qps = 2.0
    assert m.issue_span_ns == 2_000_000_000
    assert m.qps == pytest.approx(2.0)
    assert m.n_samples == 4
    # ttft over all 4 samples
    assert m.ttft[0.5] == percentile_lower([10.0, 20.0, 30.0, 40.0], 0.5)


@pytest.mark.unit
def test_window_subrange():
    sp0 = SuperPassRollup(index=0, n_issued=1, first_issue_ns=0, last_issue_ns=0,
                          ttft_ns=[10.0], latency_ns=[100.0])
    sp1 = SuperPassRollup(index=1, n_issued=1, first_issue_ns=1_000_000_000,
                          last_issue_ns=1_000_000_000, ttft_ns=[20.0], latency_ns=[200.0])
    m = windowed_metrics([sp0, sp1], 1, 2)  # drop warmup sp0
    assert m.n_samples == 1
    assert m.ttft[0.5] == 20.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/metrics/steady_state/test_window.py -v`
Expected: FAIL with `ModuleNotFoundError: ...window`

- [ ] **Step 3: Write minimal implementation**

```python
# src/inference_endpoint/metrics/steady_state/window.py
# SPDX header added by pre-commit.
from __future__ import annotations

from dataclasses import dataclass

from inference_endpoint.metrics.early_stopping import CONFIDENCE, es_percentile_estimate
from inference_endpoint.metrics.steady_state.series import SuperPassRollup


@dataclass(frozen=True, slots=True)
class WindowMetrics:
    sp_start: int
    sp_end: int
    n_samples: int
    issue_span_ns: int
    qps: float
    token_tps: float | None
    ttft: dict[float, float]
    tpot: dict[float, float] | None
    latency: dict[float, float]
    valid: dict[str, bool]


def percentile_lower(sorted_values: list[float], p: float) -> float:
    n = len(sorted_values)
    if n == 0:
        raise ValueError("percentile of empty series")
    return sorted_values[int(p * (n - 1))]


def _grid(values: list[float], percentiles) -> dict[float, float]:
    s = sorted(values)
    return {p: percentile_lower(s, p) for p in percentiles}


def windowed_metrics(
    series: list[SuperPassRollup],
    sp_start: int,
    sp_end: int,
    percentiles=(0.5, 0.9, 0.99),
    es_percentile: float = 0.99,
    confidence: float = CONFIDENCE,
) -> WindowMetrics:
    if not 0 <= sp_start < sp_end <= len(series):
        raise ValueError(f"bad window [{sp_start},{sp_end}) over {len(series)} super-passes")
    window = series[sp_start:sp_end]
    ttft: list[float] = []
    tpot: list[float] = []
    latency: list[float] = []
    out_tokens = 0
    n_samples = 0
    for sp in window:
        ttft.extend(sp.ttft_ns)
        tpot.extend(sp.tpot_ns)
        latency.extend(sp.latency_ns)
        out_tokens += sp.out_tokens
        n_samples += sp.n_issued
    issue_span_ns = window[-1].last_issue_ns - window[0].first_issue_ns
    span_s = issue_span_ns / 1e9
    qps = n_samples / span_s if span_s > 0 else 0.0
    token_tps = (out_tokens / span_s) if (out_tokens and span_s > 0) else None

    def _valid(vals: list[float]) -> bool:
        if not vals:
            return False
        return es_percentile_estimate(sorted(vals), es_percentile, confidence).estimate is not None

    return WindowMetrics(
        sp_start=sp_start,
        sp_end=sp_end,
        n_samples=n_samples,
        issue_span_ns=issue_span_ns,
        qps=qps,
        token_tps=token_tps,
        ttft=_grid(ttft, percentiles) if ttft else {},
        tpot=_grid(tpot, percentiles) if tpot else None,
        latency=_grid(latency, percentiles) if latency else {},
        valid={
            "ttft": _valid(ttft),
            "latency": _valid(latency),
            "tpot": _valid(tpot),
        },
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/metrics/steady_state/test_window.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Pre-commit + commit**

```bash
uv run pre-commit run --all-files
git add src/inference_endpoint/metrics/steady_state/window.py tests/unit/metrics/steady_state/test_window.py
git commit -m "feat(metrics): issue-time windowing core for steady-state metrics"
```

---

### Task 3: Stopping rules (A fixed-budget, B adaptive-CoV)

**Files:**

- Create: `src/inference_endpoint/metrics/steady_state/stopping.py`
- Test: `tests/unit/metrics/steady_state/test_stopping.py`

**Interfaces:**

- Consumes: `SuperPassRollup` list; `percentile_lower` from Task 2.
- Produces:

  - `rule_fixed_budget(series, k, warmup=1) -> tuple[int, int]` returns `(warmup, warmup + k)`, clamped to `len(series)`; raises `ValueError` if `warmup >= len(series)`.
  - `cov(values: list[float]) -> float` = `stdev/mean` (population stdev; `0.0` if mean is 0 or <2 values).
  - `rule_cov_converged(series, window=3, cov_bound=0.05, warmup=1, percentiles=(0.5, 0.99)) -> tuple[int, int] | None` — scan `sp_end` from `warmup+window` upward; for the trailing `window` super-passes compute each super-pass's own p50/p99 of TTFT and latency (`percentile_lower` over that super-pass's list); if every metric's CoV across the window `< cov_bound`, return `(warmup, sp_end)`. Return `None` if never converges.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/metrics/steady_state/test_stopping.py
import pytest
from inference_endpoint.metrics.steady_state.series import SuperPassRollup
from inference_endpoint.metrics.steady_state.stopping import (
    cov,
    rule_cov_converged,
    rule_fixed_budget,
)


def _sp(index, ttft_vals, lat_vals):
    return SuperPassRollup(
        index=index, n_issued=len(ttft_vals),
        first_issue_ns=index, last_issue_ns=index + 1,
        ttft_ns=list(ttft_vals), latency_ns=list(lat_vals),
    )


@pytest.mark.unit
def test_cov_zero_for_constant():
    assert cov([5.0, 5.0, 5.0]) == 0.0


@pytest.mark.unit
def test_fixed_budget_region():
    series = [_sp(i, [1.0, 2.0], [1.0, 2.0]) for i in range(6)]
    assert rule_fixed_budget(series, k=3, warmup=1) == (1, 4)
    # clamps to end
    assert rule_fixed_budget(series, k=100, warmup=1) == (1, 6)


@pytest.mark.unit
def test_cov_converges_when_stable():
    # sp0 warmup (noisy), sp1..sp4 stable -> converges at window of 3 stable ones
    series = [_sp(0, [100.0], [100.0])]
    for i in range(1, 5):
        series.append(_sp(i, [10.0, 10.0], [50.0, 50.0]))
    region = rule_cov_converged(series, window=3, cov_bound=0.01, warmup=1)
    assert region == (1, 4)  # first sp_end where trailing 3 (sp1,sp2,sp3) are stable


@pytest.mark.unit
def test_cov_never_converges_returns_none():
    series = [_sp(0, [1.0], [1.0])]
    for i in range(1, 6):
        series.append(_sp(i, [float(10**i)], [float(10**i)]))  # wildly varying
    assert rule_cov_converged(series, window=3, cov_bound=0.01, warmup=1) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/metrics/steady_state/test_stopping.py -v`
Expected: FAIL with `ModuleNotFoundError: ...stopping`

- [ ] **Step 3: Write minimal implementation**

```python
# src/inference_endpoint/metrics/steady_state/stopping.py
# SPDX header added by pre-commit.
from __future__ import annotations

from statistics import pstdev

from inference_endpoint.metrics.steady_state.series import SuperPassRollup
from inference_endpoint.metrics.steady_state.window import percentile_lower


def rule_fixed_budget(
    series: list[SuperPassRollup], k: int, warmup: int = 1
) -> tuple[int, int]:
    n = len(series)
    if warmup >= n:
        raise ValueError(f"warmup {warmup} >= {n} super-passes")
    if k < 1:
        raise ValueError("k must be >= 1")
    return (warmup, min(warmup + k, n))


def cov(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = sum(values) / len(values)
    if m == 0:
        return 0.0
    return pstdev(values) / abs(m)


def rule_cov_converged(
    series: list[SuperPassRollup],
    window: int = 3,
    cov_bound: float = 0.05,
    warmup: int = 1,
    percentiles=(0.5, 0.99),
) -> tuple[int, int] | None:
    n = len(series)

    def _pct(sp: SuperPassRollup, source: str, p: float) -> float:
        vals = sorted(getattr(sp, source))
        return percentile_lower(vals, p) if vals else 0.0

    for sp_end in range(warmup + window, n + 1):
        trailing = series[sp_end - window : sp_end]
        converged = True
        for source in ("ttft_ns", "latency_ns"):
            for p in percentiles:
                across = [_pct(sp, source, p) for sp in trailing]
                if cov(across) >= cov_bound:
                    converged = False
                    break
            if not converged:
                break
        if converged:
            return (warmup, sp_end)
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/metrics/steady_state/test_stopping.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Pre-commit + commit**

```bash
uv run pre-commit run --all-files
git add src/inference_endpoint/metrics/steady_state/stopping.py tests/unit/metrics/steady_state/test_stopping.py
git commit -m "feat(metrics): fixed-budget and adaptive-CoV steady-state stopping rules"
```

---

### Task 4: A/B sweep + scoring harness

**Files:**

- Create: `src/inference_endpoint/metrics/steady_state/harness.py`
- Test: `tests/unit/metrics/steady_state/test_harness.py`

**Interfaces:**

- Consumes: `windowed_metrics`, `WindowMetrics` (Task 2); `rule_fixed_budget`, `rule_cov_converged` (Task 3).
- Produces:

  - `asymptote(series, warmup=1, **kw) -> WindowMetrics` — window `(warmup, len(series))`.
  - `RuleScore` (frozen slotted dataclass): `name: str`, `region: tuple[int, int] | None`, `super_passes: int`, `metrics: WindowMetrics | None`, `qps_rel_err: float | None`, `ttft_p99_rel_err: float | None`.
  - `score_rule(name, region, series, ref: WindowMetrics, **kw) -> RuleScore` — relative error vs `ref` for qps and ttft p99; `region=None` ⇒ zeroed/None score flagged unconverged.
  - `sweep(series, k=3, cov_window=3, cov_bound=0.05, warmup=1, **kw) -> tuple[WindowMetrics, list[RuleScore]]` — returns `(asymptote, [score_A, score_B])`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/metrics/steady_state/test_harness.py
import pytest
from inference_endpoint.metrics.steady_state.series import SuperPassRollup
from inference_endpoint.metrics.steady_state.harness import asymptote, sweep


def _sp(index, ttft, lat, n=2, span_ns=1_000_000_000):
    return SuperPassRollup(
        index=index, n_issued=n,
        first_issue_ns=index * span_ns, last_issue_ns=index * span_ns + span_ns,
        ttft_ns=list(ttft), latency_ns=list(lat),
    )


@pytest.mark.unit
def test_asymptote_covers_all_after_warmup():
    series = [_sp(i, [10.0, 10.0], [50.0, 50.0]) for i in range(5)]
    a = asymptote(series, warmup=1)
    assert a.sp_start == 1 and a.sp_end == 5


@pytest.mark.unit
def test_sweep_returns_two_rule_scores():
    series = [_sp(0, [100.0, 100.0], [500.0, 500.0])]  # noisy warmup
    for i in range(1, 6):
        series.append(_sp(i, [10.0, 10.0], [50.0, 50.0]))  # stable
    ref, scores = sweep(series, k=3, cov_window=3, cov_bound=0.01, warmup=1)
    names = {s.name for s in scores}
    assert names == {"fixed_budget", "cov_converged"}
    for s in scores:
        if s.region is not None:
            assert s.qps_rel_err is not None and s.qps_rel_err >= 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/metrics/steady_state/test_harness.py -v`
Expected: FAIL with `ModuleNotFoundError: ...harness`

- [ ] **Step 3: Write minimal implementation**

```python
# src/inference_endpoint/metrics/steady_state/harness.py
# SPDX header added by pre-commit.
from __future__ import annotations

from dataclasses import dataclass

from inference_endpoint.metrics.steady_state.series import SuperPassRollup
from inference_endpoint.metrics.steady_state.stopping import (
    rule_cov_converged,
    rule_fixed_budget,
)
from inference_endpoint.metrics.steady_state.window import WindowMetrics, windowed_metrics


@dataclass(frozen=True, slots=True)
class RuleScore:
    name: str
    region: tuple[int, int] | None
    super_passes: int
    metrics: WindowMetrics | None
    qps_rel_err: float | None
    ttft_p99_rel_err: float | None


def asymptote(series: list[SuperPassRollup], warmup: int = 1, **kw) -> WindowMetrics:
    return windowed_metrics(series, warmup, len(series), **kw)


def _rel_err(est: float, ref: float) -> float:
    if ref == 0:
        return 0.0
    return abs(est - ref) / abs(ref)


def score_rule(
    name: str,
    region: tuple[int, int] | None,
    series: list[SuperPassRollup],
    ref: WindowMetrics,
    **kw,
) -> RuleScore:
    if region is None:
        return RuleScore(name, None, 0, None, None, None)
    m = windowed_metrics(series, region[0], region[1], **kw)
    ref_p99 = ref.ttft.get(0.99)
    est_p99 = m.ttft.get(0.99)
    p99_err = (
        _rel_err(est_p99, ref_p99)
        if (ref_p99 is not None and est_p99 is not None)
        else None
    )
    return RuleScore(
        name=name,
        region=region,
        super_passes=region[1] - region[0],
        metrics=m,
        qps_rel_err=_rel_err(m.qps, ref.qps),
        ttft_p99_rel_err=p99_err,
    )


def sweep(
    series: list[SuperPassRollup],
    k: int = 3,
    cov_window: int = 3,
    cov_bound: float = 0.05,
    warmup: int = 1,
    **kw,
) -> tuple[WindowMetrics, list[RuleScore]]:
    ref = asymptote(series, warmup=warmup, **kw)
    region_a = rule_fixed_budget(series, k=k, warmup=warmup)
    region_b = rule_cov_converged(
        series, window=cov_window, cov_bound=cov_bound, warmup=warmup
    )
    return ref, [
        score_rule("fixed_budget", region_a, series, ref, **kw),
        score_rule("cov_converged", region_b, series, ref, **kw),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/metrics/steady_state/test_harness.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Pre-commit + commit**

```bash
uv run pre-commit run --all-files
git add src/inference_endpoint/metrics/steady_state/harness.py tests/unit/metrics/steady_state/test_harness.py
git commit -m "feat(metrics): A/B sweep + scoring harness for steady-state rules"
```

---

### Task 5: CLI — `steady_state_from_events.py`

**Files:**

- Create: `scripts/steady_state_from_events.py`
- Test: `tests/unit/metrics/steady_state/test_cli.py`

**Interfaces:**

- Consumes: `build_super_pass_series` (Task 1), `sweep`/`asymptote` (Task 4), `load_reference_backend`/`encode_lengths` (existing token_metrics).
- Produces: `main(argv=None) -> dict` — parses args, builds series, runs sweep, prints a `total` vs `steady_state` comparison and an A/B markdown table; with `--json OUT` writes `{total, steady_state, rules}` to OUT. CLI args: `events` (positional), `--dataset-size` (int, required), `--concurrency` (int, required), `--warmup` (int, default 1), `--k` (int, default 3), `--cov-window` (int, default 3), `--cov-bound` (float, default 0.05), `--tokenizer` (optional, enables TPOT/token_tps), `--json` (optional out path).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/metrics/steady_state/test_cli.py
import json

import msgspec.json
import pytest
from inference_endpoint.core.record import (
    EventRecord,
    EventType,
    SampleEventType,
    SessionEventType,
)

import importlib.util
from pathlib import Path

_CLI = Path(__file__).resolve().parents[4] / "scripts" / "steady_state_from_events.py"
_spec = importlib.util.spec_from_file_location("steady_state_cli", _CLI)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)

_ENC = msgspec.json.Encoder(enc_hook=EventType.encode_hook)


def _write(path, records):
    with open(path, "wb") as f:
        for r in records:
            f.write(_ENC.encode(r))
            f.write(b"\n")


@pytest.mark.unit
def test_cli_writes_json(tmp_path):
    p = tmp_path / "events.jsonl"
    recs = [EventRecord(event_type=SessionEventType.START_PERFORMANCE_TRACKING, timestamp_ns=0)]
    t = 100
    for i in range(8):  # dataset_size=2, concurrency=2 -> 2/super-pass -> 4 super-passes
        u = f"s{i}"
        recs += [
            EventRecord(event_type=SampleEventType.ISSUED, sample_uuid=u, timestamp_ns=t),
            EventRecord(event_type=SampleEventType.RECV_FIRST, sample_uuid=u, timestamp_ns=t + 10),
            EventRecord(event_type=SampleEventType.COMPLETE, sample_uuid=u, timestamp_ns=t + 50),
        ]
        t += 1_000_000_000
    recs.append(EventRecord(event_type=SessionEventType.STOP_PERFORMANCE_TRACKING, timestamp_ns=t))
    _write(p, recs)

    out = tmp_path / "steady.json"
    cli.main([
        str(p), "--dataset-size", "2", "--concurrency", "2",
        "--warmup", "1", "--k", "2", "--cov-window", "2", "--cov-bound", "0.5",
        "--json", str(out),
    ])
    doc = json.loads(out.read_text())
    assert "total" in doc and "steady_state" in doc and "rules" in doc
    assert doc["total"]["sp_start"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/metrics/steady_state/test_cli.py -v`
Expected: FAIL with `FileNotFoundError` / module load error (script absent)

- [ ] **Step 3: Write minimal implementation**

```python
#!/usr/bin/env python3
# scripts/steady_state_from_events.py
# SPDX header added by pre-commit.
"""Offline steady-state windowing + A/B stopping-rule sweep from a run's events.jsonl.

Cold-path companion to ``scripts/early_stopping_estimate_from_events.py``. Buckets
performance-tracked samples into super-passes by issue order, computes the
issue-time steady-state window, and scores fixed-budget (A) vs adaptive-CoV (B)
stopping rules against the full-series asymptote. ``steady_state`` is the official
number; ``total`` is reported alongside so their divergence is visible.

usage:
  uv run python scripts/steady_state_from_events.py <events.jsonl> \
      --dataset-size N --concurrency N [--warmup 1] [--k 3] \
      [--cov-window 3] [--cov-bound 0.05] [--tokenizer DIR] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from inference_endpoint.async_utils.services.metrics_aggregator.token_metrics import (
    encode_lengths,
    load_reference_backend,
)
from inference_endpoint.metrics.steady_state.harness import asymptote, sweep
from inference_endpoint.metrics.steady_state.series import build_super_pass_series
from inference_endpoint.metrics.steady_state.window import windowed_metrics


def _make_counter(path):
    backend = load_reference_backend(path)
    if backend is None:
        raise SystemExit(f"FATAL: could not load tokenizer backend from {path}")
    return lambda texts: encode_lengths(backend, texts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("events")
    ap.add_argument("--dataset-size", type=int, required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--cov-window", type=int, default=3)
    ap.add_argument("--cov-bound", type=float, default=0.05)
    ap.add_argument("--tokenizer")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv)

    counter = _make_counter(args.tokenizer) if args.tokenizer else None
    series = build_super_pass_series(
        args.events, args.dataset_size, args.concurrency, count_tokens=counter
    )
    if len(series) <= args.warmup:
        raise SystemExit(
            f"FATAL: only {len(series)} super-passes; need > warmup ({args.warmup})"
        )

    total = windowed_metrics(series, 0, len(series))
    ref, scores = sweep(
        series,
        k=args.k,
        cov_window=args.cov_window,
        cov_bound=args.cov_bound,
        warmup=args.warmup,
    )

    print(f"super-passes: {len(series)}  (warmup dropped: {args.warmup})")
    print(f"total     : qps={total.qps:,.2f}  ttft_p99={total.ttft.get(0.99)}")
    print(f"steady(ref): qps={ref.qps:,.2f}  ttft_p99={ref.ttft.get(0.99)}")
    print("\n| rule | super-passes | region | qps | qps_rel_err | ttft_p99_rel_err |")
    print("|---|---|---|---|---|---|")
    for s in scores:
        if s.region is None:
            print(f"| {s.name} | - | UNCONVERGED | - | - | - |")
            continue
        print(
            f"| {s.name} | {s.super_passes} | {s.region} | "
            f"{s.metrics.qps:,.2f} | {s.qps_rel_err:.4f} | {s.ttft_p99_rel_err} |"
        )

    doc = {
        "total": asdict(total),
        "steady_state": asdict(ref),
        "rules": [asdict(s) for s in scores],
    }
    if args.json_out:
        # WindowMetrics dicts use float percentile keys; JSON needs str keys.
        with open(args.json_out, "w") as f:
            json.dump(doc, f, indent=2, default=str)
        print(f"\nwritten to {args.json_out}")
    return doc


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/metrics/steady_state/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Full suite + pre-commit + commit**

```bash
uv run pytest tests/unit/metrics/steady_state/ -v
uv run pre-commit run --all-files
git add scripts/steady_state_from_events.py tests/unit/metrics/steady_state/test_cli.py
git commit -m "feat(scripts): steady-state windowing + A/B sweep CLI from events.jsonl"
```

---

## Self-Review

**Spec coverage:**

- Windowing core (issue-time, unified set, no end-crop) → Task 2 ✓
- Super-pass unit `S = ceil(N/dataset_size)` + warmup drop → Task 1 (`super_pass_size`) + `warmup` param in Tasks 3–5 ✓
- Per-super-pass series from `events.jsonl` → Task 1 ✓
- Rule A (fixed budget) + Rule B (CoV, not KL) with ES validity gate → Task 3 (rules) + Task 2 (`valid`) ✓
- A/B harness, asymptote ground truth, score = |est−asymptote| + super-passes → Task 4 ✓
- `total` + `steady_state` output, divergence visible → Task 5 CLI ✓
- Cold-path, mirrors ES script, no hot-path change → whole plan ✓
- Rollout steps 1–3 (strategy-agnostic tooling + sweep) → this milestone; step 5 report wiring → deferred (Milestone 2, noted below) ✓

**Deferred to Milestone 2 (out of this plan):** wiring `steady_state` into `metrics/report.py` as the official result block. Milestone 1 emits a standalone JSON/markdown, sufficient to run the sweep and choose a rule from data.

**Placeholder scan:** none — every code step is complete and self-contained.

**Type consistency:** `SuperPassRollup`, `WindowMetrics`, `RuleScore` field names are consistent across Tasks 1–5; `windowed_metrics(series, start, end, ...)` signature matches all call sites; percentile keys are floats (`0.5/0.9/0.99`) everywhere, stringified only at JSON write.

## Open parameters (tuned empirically after Milestone 1, per spec)

`k` (Rule A), `cov_window` / `cov_bound` (Rule B), and the ES target percentile
are CLI-exposed defaults, to be chosen from the sweep over a real long run.
