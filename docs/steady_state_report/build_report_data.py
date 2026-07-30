#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate report_data.js for the steady-state HTML report.

For concurrency / poisson runs it parses a local ``events.jsonl`` into a
per-super-pass scalar series (p50/p99 TTFT + latency, per-super-pass QPS, issue
span). For offline (max-throughput) runs it embeds the completion-rate-per-minute
series (issue time is degenerate at t=0). The front-end re-runs all detection
(adaptive warmup, drift, CoV ensemble, plateau-edge) live on these series, so the
data file is raw series only — no pre-computed windows.

Concurrency/poisson parsing needs the inference_endpoint package, so run this in
the dev container:

  docker run --rm -v "$PWD":/mnt/inference-endpoint -w /mnt/inference-endpoint \
    -v <artifacts>:/data inference-endpoint-dev bash -lc \
    "uv run python docs/steady_state_report/build_report_data.py /data > docs/steady_state_report/report_data.js"
"""

from __future__ import annotations

import json
import sys

from inference_endpoint.metrics.steady_state.drift import adaptive_warmup
from inference_endpoint.metrics.steady_state.series import (
    build_super_pass_series,
    super_pass_size,
)
from inference_endpoint.metrics.steady_state.window import (
    percentile_lower,
    windowed_metrics,
)

# (label, mode, events path under the mounted data root, dataset_size, concurrency)
CONC_RUNS = [
    ("gpt-oss c1024", "concurrency", "C1024/events.jsonl", 6396, 1024),
    ("gpt-oss c7168", "concurrency", "C7168/events.jsonl", 6396, 7168),
    ("gpt-oss c22528", "concurrency", "C22528/events.jsonl", 6396, 22528),
    ("DSR1 poisson 37qps", "poisson", "poisson-232386/events.jsonl", 4388, 1),
]

# Offline completion-rate per 1-min bin (from the hecate-side regex parse of the
# 31 GB / 5.8 GB events.jsonl — issue time is degenerate for max-throughput).
OFFLINE_RUNS = [
    {
        "label": "DSR1 offline-IFB 293038",
        "mode": "offline",
        "dataset_size": 4388,
        "bin_seconds": 60,
        "complete_per_bin": [
            12511,
            25424,
            25206,
            22858,
            21892,
            21257,
            20170,
            19656,
            19368,
            18648,
            18564,
            17423,
            18003,
            17779,
            17311,
            16638,
            16717,
            16653,
            17135,
            16568,
            16757,
            17153,
            16646,
            16701,
            16821,
            17057,
            16818,
            17153,
            16923,
            16989,
            16528,
            17135,
            17200,
            16669,
            16974,
            16738,
            16771,
            17180,
            16815,
            16446,
            16651,
            16883,
            17108,
            16581,
            16500,
            16600,
            17042,
            16477,
            16676,
            16419,
            16492,
            16843,
            16962,
            16747,
            16686,
            16223,
            16742,
            16451,
            16800,
            16480,
            16764,
            16929,
            16753,
            16676,
            16516,
            16546,
            16683,
            16801,
            16882,
            16442,
            16696,
            16492,
            16698,
            16402,
            16435,
            16522,
            16750,
            16297,
            16453,
            16985,
            16589,
            16915,
            16973,
            17045,
            16840,
            16531,
            16654,
            17148,
            16737,
            17091,
            16906,
            16481,
            16187,
            16927,
            16127,
            16063,
            14719,
            12370,
            10281,
            8480,
            7721,
            6655,
            5366,
            4857,
            4570,
            4133,
            3567,
            2855,
            3041,
            2935,
            2859,
            2805,
            2850,
            2686,
            2711,
            1837,
            2467,
            1950,
            1823,
            1878,
            1554,
            1742,
            1627,
            1374,
            1113,
            935,
            1302,
            1478,
            1361,
            1251,
            1517,
            1583,
            1250,
            1536,
            959,
            1122,
            1217,
            1385,
            879,
            1364,
            1577,
            1010,
            898,
            707,
            785,
            569,
            452,
            395,
            344,
            272,
            2974,
            1036,
            481,
            176,
            114,
            71,
            8,
        ],
    },
    {
        "label": "DSR1 offline max-tput 272936",
        "mode": "offline",
        "dataset_size": 4388,
        "bin_seconds": 60,
        "complete_per_bin": [
            2485,
            4531,
            5111,
            5324,
            5480,
            5629,
            5443,
            5125,
            5033,
            4942,
            4739,
            4762,
            4663,
            4781,
            4703,
            4613,
            4687,
            4613,
            4640,
            4596,
            4627,
            4669,
            4645,
            4744,
            4584,
            4814,
            4661,
            4731,
            4795,
            4715,
            4799,
            4722,
            4675,
            4667,
            4741,
            4751,
            4715,
            4779,
            4727,
            4706,
            4738,
            4704,
            4647,
            4785,
            4599,
            4707,
            4758,
            4641,
            4747,
            4701,
            4753,
            4709,
            4757,
            4777,
            4661,
            4751,
            4774,
            4660,
            4675,
            4694,
            4697,
            4706,
            4657,
            4721,
            4626,
            4926,
            4594,
            4688,
            4714,
            4630,
            4664,
            4689,
            4636,
            3954,
            3049,
            2325,
            1713,
            1358,
            938,
            706,
            473,
            643,
            2467,
            764,
            288,
            143,
            88,
            126,
        ],
    },
]

_PCTS = (0.5, 0.9, 0.99)


def _grid(vals):
    if not vals:
        return {}
    s = sorted(vals)
    return {str(p): percentile_lower(s, p) for p in _PCTS}


def build_conc_run(label, mode, path, ds, conc):
    series = build_super_pass_series(path, ds, conc)
    n_issued = sum(sp.n_issued for sp in series)
    out = []
    for sp in series:
        span_s = (sp.last_issue_ns - sp.first_issue_ns) / 1e9
        out.append(
            {
                "i": sp.index,
                "n": sp.n_issued,
                "qps": (sp.n_issued / span_s) if span_s > 0 else 0.0,
                "ttft": _grid(sp.ttft_ns),
                "lat": _grid(sp.latency_ns),
                "span_s": span_s,
            }
        )
    # Exact windowed percentiles (need raw values, unavailable to the front-end):
    # full-run "total" vs steady window at the adaptive warmup. The live sliders
    # move the window edges + verdicts; these anchor the honest metric recovery.
    aw = adaptive_warmup(series)
    total = windowed_metrics(series, 0, len(series))
    steady = windowed_metrics(series, aw, len(series)) if aw < len(series) else total

    def _m(w):
        return {
            "qps": w.qps,
            "ttft_p99": w.ttft.get(0.99),
            "ttft_p50": w.ttft.get(0.5),
            "lat_p99": w.latency.get(0.99),
        }

    return {
        "label": label,
        "mode": mode,
        "dataset_size": ds,
        "concurrency": conc,
        "n_issued": n_issued,
        "super_pass_samples": super_pass_size(ds, conc),
        "series": out,
        "exact": {"warmup": aw, "total": _m(total), "steady": _m(steady)},
    }


def main():
    data_root = sys.argv[1] if len(sys.argv) > 1 else "/data"
    runs = []
    for label, mode, rel, ds, conc in CONC_RUNS:
        runs.append(build_conc_run(label, mode, f"{data_root}/{rel}", ds, conc))
    runs.extend(OFFLINE_RUNS)
    doc = {"version": 1, "runs": runs}
    print("// Generated by build_report_data.py. Do not edit by hand.")
    print("window.STEADY_REPORT_DATA = " + json.dumps(doc) + ";")


if __name__ == "__main__":
    main()
