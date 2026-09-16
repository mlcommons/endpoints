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

"""End-to-end performance tests for the benchmark CLI.

Two families of tests, both driving the cyclopts ``inference-endpoint``
app in-process and parameterized on stream/non-stream:

* **Roofline** against :class:`MaxThroughputServer` (instant pre-compiled
  responses). Measures peak QPS for each load pattern
  (``max_throughput``, ``concurrency``, ``poisson``). Prints numbers
  rather than asserting on them.

* **Low-QPS correctness** against :class:`VariableResponseServer`
  (realistic TTFT + per-token TPOT). Asserts zero ``failed`` requests at
  5 QPS for 20 s — guards keep-alive / idle-pool / slow-response
  regressions.

Both families are marked ``performance`` and are therefore CI-skipped;
run them explicitly when investigating throughput regressions or
benchmarking a new machine. Results from every parametrized case are
written via the ``record_result`` fixture and rendered as a single
summary table by ``conftest.py`` after the session completes.

Run::

    pytest -vs -m performance --no-cov tests/performance/commands/test_e2e_perf.py

Knobs are pytest options (see the "roofline" group in ``pytest --help``)::

    pytest -vs -m performance --no-cov tests/performance/commands/test_e2e_perf.py \
        --roofline-server-workers=8 --roofline-stream-interval=20
"""

from __future__ import annotations

import pytest
from inference_endpoint.testing.max_throughput_server import MaxThroughputServer
from inference_endpoint.testing.variable_throughput_server import VariableResponseServer

from .utils import run_cli

# =============================================================================
# Roofline tests — MaxThroughputServer, every load pattern, stream + non-stream
# =============================================================================


@pytest.fixture(scope="module")
def client_opts(request):
    """Client-side ``run_cli`` overrides from the ``--roofline-*`` options."""
    return {
        "workers": request.config.getoption("--roofline-client-workers"),
        "init_timeout": request.config.getoption("--roofline-init-timeout"),
    }


@pytest.fixture(scope="module", params=[False, True], ids=["nonstream", "stream"])
def max_tput_server(request):
    """Stub server returning fixed pre-compiled responses (roofline target).

    ``stream_interval`` is the number of characters packed into each SSE
    event (the stub's ``output_length`` is 160 chars, so 10 means 16 events
    per response). It shapes what the streaming roofline measures: at 1 the
    client's per-chunk SSE parsing dominates and the number reflects
    chunk-parse cost rather than the request pipeline; at ``output_length``
    a response is one big event and streaming accumulation is barely
    exercised. 16 events approximates a realistic completion while keeping
    the stub cheap.

    Both knobs are pytest options: ``--roofline-server-workers`` (default 4)
    and ``--roofline-stream-interval`` (default 10).
    """
    with MaxThroughputServer(
        port=0,
        num_workers=request.config.getoption("--roofline-server-workers"),
        stream=request.param,
        stream_interval=request.config.getoption("--roofline-stream-interval"),
        quiet=True,
    ) as srv:
        yield srv


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
def test_max_throughput_roofline(max_tput_server, tmp_path, record_result, client_opts):
    """Offline burst — issue 2,000,000 queries at t=0."""
    results = run_cli(
        [
            "offline",
            "--load-pattern",
            "max_throughput",
            "--num-samples",
            "2000000",
        ],
        tmp_path,
        max_tput_server,
        **client_opts,
    )
    r = results
    failed = r["n_samples_failed"]
    total = r["n_samples_completed"]
    elapsed = r["duration_ns"] / 1e9
    assert r["complete"], "run did not complete cleanly"
    assert failed == 0, f"failed={failed} (expected 0)"
    record_result(
        "max_throughput (2M burst)",
        stream=max_tput_server.stream,
        qps=r["qps"],
        total=total,
        elapsed=elapsed,
        failed=failed,
    )
    print(
        f"\n  max_throughput  stream={max_tput_server.stream}: "
        f"QPS={r['qps']:>10,.0f}  total={total:>9,}  "
        f"elapsed={elapsed:6.2f}s"
    )


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
@pytest.mark.parametrize("concurrency", [1000, 4000, 16000])
def test_concurrency_roofline(
    max_tput_server, concurrency, tmp_path, record_result, client_opts
):
    """Online concurrency — N in-flight requests for fixed duration."""
    results = run_cli(
        [
            "online",
            "--load-pattern",
            "concurrency",
            "--concurrency",
            str(concurrency),
            "--runtime.max-issue-duration-ms",
            "12000",
            # Headroom so wall time, not sample count, is the limit.
            "--num-samples",
            "10000000",
        ],
        tmp_path,
        max_tput_server,
        **client_opts,
    )
    r = results
    failed = r["n_samples_failed"]
    total = r["n_samples_completed"]
    elapsed = r["duration_ns"] / 1e9
    assert r["complete"], "run did not complete cleanly"
    assert failed == 0, f"failed={failed} (expected 0)"
    record_result(
        f"concurrency c={concurrency:,}",
        stream=max_tput_server.stream,
        qps=r["qps"],
        total=total,
        elapsed=elapsed,
        failed=failed,
    )
    print(
        f"\n  concurrency  c={concurrency:>5}  stream={max_tput_server.stream}: "
        f"QPS={r['qps']:>10,.0f}  total={total:>9,}  "
        f"elapsed={elapsed:6.2f}s"
    )


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
def test_poisson_binary_search_max_qps(
    max_tput_server, tmp_path, record_result, client_opts
):
    """Binary search for the largest 10k-multiple target_qps the server sustains."""
    STEP = 10_000
    LO, HI = 10_000, 250_000  # search space (inclusive)
    PASS_RATIO = 0.95  # achieved/target threshold for "sustained"

    # Standard binary search over candidate targets so the LO boundary is
    # actually exercised: with ``while lo < hi`` we could converge to
    # ``lo == hi == LO/STEP`` without ever issuing a run at LO, leaving
    # ``max_sustained`` reported as 0 even if LO is sustainable.
    history: list[tuple[int, float, bool]] = []
    best_sustained = 0
    lo, hi = LO // STEP, HI // STEP  # integer bounds in units of STEP
    while lo <= hi:
        mid = (lo + hi) // 2
        target = mid * STEP
        results = run_cli(
            [
                "online",
                "--load-pattern",
                "poisson",
                "--target-qps",
                str(target),
                "--runtime.max-issue-duration-ms",
                "12000",
                # Headroom so wall time, not sample count, is the limit.
                "--num-samples",
                str(max(100_000, target * 15)),
            ],
            tmp_path / f"qps_{target}",
            max_tput_server,
            **client_opts,
        )
        achieved = results["qps"]
        sustained = achieved >= target * PASS_RATIO
        history.append((target, achieved, sustained))
        if sustained:
            best_sustained = target
            lo = mid + 1
        else:
            hi = mid - 1

    max_sustained = best_sustained
    record_result(
        "poisson max_sustained",
        stream=max_tput_server.stream,
        qps=max_sustained,
        failed=0,
    )
    print(
        f"\n  poisson binary search  stream={max_tput_server.stream}: "
        f"max_sustained={max_sustained:>7,} QPS  (PASS_RATIO={PASS_RATIO})"
    )
    for t, a, s in history:
        print(f"    target={t:>7,}  achieved={a:>10,.0f}  sustained={s}")


# =============================================================================
# Low-QPS correctness — VariableResponseServer, 5 QPS, no network errors
# =============================================================================


@pytest.fixture(scope="module", params=[False, True], ids=["nonstream", "stream"])
def variable_server(request):
    """Realistic LLM stub: ~100-char responses, 50ms TTFT, 10ms/token TPOT."""
    with VariableResponseServer(
        host="127.0.0.1",
        port=0,
        output_len_mean=100,
        output_len_spread=0.2,
        inter_token_latency=10.0,
        inter_token_spread=0.1,
        first_chunk_latency=0.05,
        first_chunk_spread=0.1,
        stream=request.param,
        stream_interval=20,
        num_workers=2,
        quiet=True,
    ) as srv:
        yield srv


@pytest.mark.performance
@pytest.mark.xdist_group(name="serial_performance")
def test_low_qps_no_network_errors(
    variable_server, tmp_path, record_result, client_opts
):
    """Sustain 5 QPS Poisson for 20 s — must complete with zero failed requests.

    Low QPS spaces requests far enough apart that idle connections may
    sit past ``TCP_KEEPIDLE`` (1 s in :class:`_SocketConfig`). A regression
    in keep-alive probing, idle pool eviction, or slow-response handling
    surfaces here as non-zero ``failed`` count.
    """
    TARGET_QPS = 5
    DURATION_S = 20

    results = run_cli(
        [
            "online",
            "--load-pattern",
            "poisson",
            "--target-qps",
            str(TARGET_QPS),
            "--runtime.max-issue-duration-ms",
            f"{DURATION_S}s",
            # 2x the expected count leaves headroom so the issue-duration cap,
            # not sample-count variance, ends the full idle-connection window.
            "--num-samples",
            str(TARGET_QPS * DURATION_S * 2),
            # Low QPS needs neither many workers nor pre-warmed connections;
            # using auto defaults makes startup slow and flaky against a stub
            # that has TTFT + per-token delays.
            "--workers",
            "4",
            "--client.warmup-connections",
            "0",
        ],
        tmp_path,
        variable_server,
        # --workers is pinned to 4 above (deliberate for the low-QPS stub);
        # only the init-timeout knob applies here.
        init_timeout=client_opts["init_timeout"],
    )
    r = results
    failed = r["n_samples_failed"]
    total = r["n_samples_completed"]
    elapsed = r["duration_ns"] / 1e9
    assert r["complete"], "run did not complete cleanly"
    assert failed == 0, f"failed={failed} of {total}"
    record_result(
        f"low_qps target={TARGET_QPS}",
        stream=variable_server.stream,
        qps=r["qps"],
        total=total,
        elapsed=elapsed,
        failed=failed,
    )
    print(
        f"\n  low_qps  target={TARGET_QPS}  stream={variable_server.stream}: "
        f"achieved={r['qps']:.2f} QPS  total={total}  "
        f"failed={failed}  elapsed={elapsed:.2f}s"
    )
