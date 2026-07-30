/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Steady-state experiments report. Two layers:
 *   1. Static study charts (§3, §4) drawn from the digested experiment results
 *      below (Experiments 1-3 over the events.jsonl corpora).
 *   2. The live detector (§7), re-implemented from
 *      src/inference_endpoint/metrics/steady_state/{drift,stopping,window}.py +
 *      scripts/steady_state_drift.py, running on report_data.js series. */
"use strict";

/* ---- palette (mirrors styles.css :root) ---- */
const C = {
  surface: "#fcfcfb",
  ink: "#0b0b0b",
  secondary: "#52514e",
  muted: "#898781",
  grid: "#e1e0d9",
  baseline: "#c3c2b7",
  s1: "#2a78d6", // blue
  s2: "#eb6834", // orange
  good: "#0ca30c",
  critical: "#d03b3b",
  steadyShade: "rgba(12,163,12,0.12)",
  dropShade: "rgba(208,59,59,0.08)",
};
const FONT = "11px system-ui, -apple-system, 'Segoe UI', sans-serif";
const FONT_SM = "10px system-ui, -apple-system, 'Segoe UI', sans-serif";

/* ---- digested experiment results (Experiments 1-3) ---- */
const STUDY = {
  // §3 staging sim (N=2048, prefill_rate=276 req/s), factors vs burst
  staging: {
    steps: [1, 2, 4, 8, 16],
    peak: [1.0, 0.52, 0.28, 0.16, 0.11], // peak ramp-up TTFT / burst
    dur: [1.0, 0.98, 0.93, 0.89, 0.86], // ramp duration / burst
  },
  // §4.2 tail vs steady TPOT p50 (ms) + delta
  tpot: {
    pts: ["C1024", "C7168", "C22528", "dsr1*"],
    steady: [4.972, 9.58, 22.529, 47.358],
    tail: [4.903, 9.286, 21.327, 45.589],
    dpct: [-1.4, -3.0, -5.3, -3.7],
  },
  // §4.2 throughput overstatement (drain fraction), by concurrency
  overstate: {
    pts: ["C8", "C140", "C1024", "C2048", "C7168", "C22528", "dsr1*", "pois*"],
    pctv: [0.44, 0.69, 0.84, 1.34, 2.15, 3.52, 13.12, 3.79],
    tentative: [0, 0, 0, 0, 0, 0, 1, 1],
  },
  // §4.3 CoV converged-window p99-TTFT error vs asymptote (fraction)
  cov: {
    pts: ["C1024", "C2048", "pois*", "C22528", "dsr1*", "C7168"],
    err: [0.015, 0.017, 0.035, 0.195, 0.389, 0.377],
    tentative: [0, 0, 1, 0, 1, 0],
  },
  // §2 cross-mode hairball: cumulative % complete vs % of wall-clock (Hecate DSR1)
  hairballCurves: {
    xs: [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100],
    poisson: [0, 3.1, 7.1, 11.2, 15.4, 19.6, 23.8, 27.9, 32.1, 36.2, 40.4, 44.5, 48.6, 52.7, 56.8, 60.9, 65.1, 69.3, 73.5, 77.6, 81.8, 86.0, 90.1, 94.3, 98.5, 100],
    off272: [0, 3.8, 8.7, 13.3, 17.6, 21.9, 26.1, 30.3, 34.6, 38.9, 43.2, 47.4, 51.7, 56.0, 60.3, 64.5, 68.8, 73.1, 77.4, 81.6, 85.9, 90.2, 94.4, 97.9, 99.5, 100],
    off293: [0, 7.3, 13.7, 19.5, 25.3, 31.0, 36.8, 42.5, 48.1, 53.8, 59.4, 65.1, 70.8, 76.4, 82.1, 87.8, 93.1, 95.6, 96.8, 97.7, 98.3, 98.8, 99.2, 99.6, 99.9, 100],
  },
  // §4.3 CoV worked example: C1024 real per-sample TTFT (s), issue-order
  // subsample per super-pass 18..22 (super_pass_size 6396, S=1). Each band's
  // p99 becomes one point in the trailing window; CoV = sigma/mu of the p99s.
  covSteps: {
    sps: [18, 19, 20, 21, 22],
    p50: [0.336, 0.335, 0.333, 0.332, 0.327],
    p99: [1.143, 1.148, 1.066, 1.126, 1.117],
    counts: [6396, 6396, 6396, 6396, 457],
    sub: [
      [0.362, 0.619, 0.3, 0.865, 0.2, 0.476, 0.159, 0.45, 0.073, 0.215, 0.633, 0.375, 0.482, 0.213, 0.489, 0.139, 0.304, 0.516, 0.187, 0.536, 0.264, 0.516, 0.217, 0.497, 0.101, 0.322, 0.479, 0.279, 0.491, 0.243, 0.3, 0.189, 0.424, 1.043, 0.261, 0.78, 0.252, 0.324, 0.111, 0.274, 0.091, 0.312, 0.458, 0.269, 0.548, 0.187, 0.438, 0.147, 0.491, 0.903, 0.264, 0.812, 0.175, 0.51, 0.104, 0.631, 0.081, 0.339, 0.753, 0.197, 0.554, 0.196, 0.42, 0.181, 0.402, 0.564, 0.309, 0.609, 0.157, 0.433, 0.126, 0.272, 0.061, 0.384, 0.805, 0.207, 0.765, 0.17, 0.387, 0.157, 0.341, 0.796, 0.204, 0.569, 0.228, 0.717, 0.112, 0.349, 0.064, 0.313, 0.643],
      [0.199, 0.407, 0.176, 0.396, 0.096, 0.31, 0.734, 0.255, 0.931, 0.237, 0.329, 0.131, 0.27, 0.554, 0.315, 0.538, 0.243, 0.429, 0.131, 0.443, 0.055, 0.253, 0.602, 0.325, 0.596, 0.18, 0.445, 0.093, 0.4, 0.45, 0.606, 0.528, 0.194, 0.385, 0.109, 0.273, 0.097, 0.332, 0.997, 0.34, 0.478, 0.116, 0.43, 0.093, 0.334, 0.545, 0.254, 0.569, 0.231, 0.608, 0.204, 0.406, 0.06, 0.435, 0.794, 0.217, 0.345, 0.147, 0.395, 0.168, 0.309, 0.599, 0.332, 0.522, 0.29, 0.559, 0.135, 0.34, 0.082, 0.369, 0.763, 0.253, 0.566, 0.129, 0.37, 0.14, 0.336, 0.515, 0.242, 0.537, 0.221, 0.372, 0.153, 0.745, 0.117, 0.35, 0.595, 0.176, 0.567, 0.203, 0.331],
      [0.059, 0.308, 0.395, 0.288, 0.782, 0.156, 0.385, 0.146, 0.377, 0.664, 0.31, 0.584, 0.224, 0.398, 0.13, 0.416, 0.104, 0.26, 0.503, 0.235, 0.411, 0.2, 0.702, 0.147, 0.436, 0.528, 0.287, 0.708, 0.192, 0.379, 0.263, 0.543, 0.055, 0.23, 0.54, 0.26, 0.444, 0.22, 0.47, 0.095, 0.305, 0.991, 0.295, 0.515, 0.277, 0.435, 0.137, 0.586, 0.073, 0.326, 0.497, 0.28, 0.387, 0.243, 0.512, 0.137, 0.375, 0.512, 0.308, 0.969, 0.2, 0.488, 0.155, 0.586, 0.09, 0.31, 0.486, 0.247, 0.594, 0.196, 0.371, 0.124, 0.255, 0.671, 0.453, 0.447, 0.259, 0.422, 0.143, 0.304, 0.055, 0.357, 0.506, 0.219, 0.374, 0.138, 0.353, 0.111, 0.219, 0.534, 0.378],
      [1.16, 0.245, 0.468, 0.048, 0.301, 1.102, 0.266, 0.386, 0.134, 0.368, 0.141, 0.509, 0.098, 0.252, 0.512, 0.344, 0.654, 0.246, 0.395, 0.186, 0.279, 0.807, 0.378, 0.983, 0.268, 0.725, 0.085, 0.376, 0.06, 0.337, 0.413, 0.344, 0.358, 0.281, 0.391, 0.08, 0.321, 0.527, 0.257, 0.635, 0.232, 0.626, 0.185, 0.379, 0.222, 0.307, 0.466, 0.256, 0.616, 0.089, 0.441, 0.157, 0.386, 0.848, 0.309, 0.919, 0.252, 0.443, 0.143, 0.507, 0.057, 0.329, 0.957, 0.26, 0.353, 0.182, 0.512, 0.186, 0.278, 0.661, 0.233, 0.465, 0.164, 0.641, 0.094, 0.432, 0.082, 0.408, 0.654, 0.232, 0.736, 0.212, 0.435, 0.141, 0.418, 0.489, 0.221, 0.839, 0.271, 0.477, 0.1],
      [0.334, 0.719, 0.218, 0.645, 0.224, 0.601, 0.151],
    ],
  },
};

/* ================= shared canvas helpers ================= */
function fitCanvas(cv, h) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(320, Math.floor(cv.clientWidth));
  cv.width = Math.floor(w * dpr);
  cv.height = Math.floor(h * dpr);
  cv.style.height = h + "px";
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = C.surface;
  ctx.fillRect(0, 0, w, h);
  return { ctx, w, h };
}
function plotBox(ctx, w, h, pad) {
  const b = { L: pad.L, R: w - pad.R, T: pad.T, B: h - pad.B };
  b.pw = b.R - b.L;
  b.ph = b.B - b.T;
  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 1;
  ctx.strokeRect(b.L, b.T, b.pw, b.ph);
  return b;
}
function yGrid(ctx, b, ticks, fmt) {
  ctx.font = FONT;
  ctx.textBaseline = "middle";
  for (const t of ticks) {
    const py = b.B - t.f * b.ph;
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(b.L, py);
    ctx.lineTo(b.R, py);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = C.muted;
    ctx.textAlign = "right";
    ctx.fillText(fmt(t.v), b.L - 6, py);
  }
  ctx.textBaseline = "alphabetic";
}
function hatch(ctx, x, y, w, h, color) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  ctx.strokeStyle = "rgba(255,255,255,0.55)";
  ctx.lineWidth = 1;
  for (let i = -h; i < w; i += 5) {
    ctx.beginPath();
    ctx.moveTo(x + i, y + h);
    ctx.lineTo(x + i + h, y);
    ctx.stroke();
  }
  ctx.restore();
}

/* grouped bars: charts A (staging) + B (tpot) */
function groupedBars(cvId, groups, series, opts) {
  const cv = document.getElementById(cvId);
  if (!cv) return;
  const { ctx, w, h } = fitCanvas(cv, opts.height || 260);
  const b = plotBox(ctx, w, h, { L: 46, R: 16, T: 16, B: 34 });
  const yMax = opts.yMax;
  yGrid(ctx, b, opts.ticks, opts.yfmt);

  const nG = groups.length;
  const gW = b.pw / nG;
  const nS = series.length;
  const barW = Math.min(34, (gW * 0.62) / nS);
  const gap = 3;
  const clusterW = nS * barW + (nS - 1) * gap;
  ctx.textAlign = "center";
  groups.forEach((g, gi) => {
    const cx = b.L + gi * gW + gW / 2;
    const x0 = cx - clusterW / 2;
    series.forEach((s, si) => {
      const v = s.values[gi];
      const bx = x0 + si * (barW + gap);
      const bh = (v / yMax) * b.ph;
      const by = b.B - bh;
      ctx.fillStyle = s.color;
      ctx.fillRect(bx, by, barW, bh);
      if (s.hatch) hatch(ctx, bx, by, barW, bh, s.color);
      ctx.fillStyle = C.secondary;
      ctx.font = FONT_SM;
      ctx.fillText(s.label2 ? s.label2(v, gi) : "", bx + barW / 2, by - 3);
    });
    ctx.fillStyle = C.secondary;
    ctx.font = FONT;
    ctx.fillText(g, cx, b.B + 14);
    if (opts.subLabel) {
      ctx.fillStyle = C.muted;
      ctx.font = FONT_SM;
      ctx.fillText(opts.subLabel(gi), cx, b.B + 26);
    }
  });
  if (opts.xlabel) {
    ctx.fillStyle = C.muted;
    ctx.font = FONT;
    ctx.textAlign = "center";
    ctx.fillText(opts.xlabel, b.L + b.pw / 2, h - 4);
  }
}

/* single series, per-bar color: charts C (overstate) + D (cov) */
function coloredBars(cvId, labels, values, colors, opts) {
  const cv = document.getElementById(cvId);
  if (!cv) return;
  const { ctx, w, h } = fitCanvas(cv, opts.height || 260);
  const b = plotBox(ctx, w, h, { L: 46, R: 16, T: 16, B: 40 });
  yGrid(ctx, b, opts.ticks, opts.yfmt);
  if (opts.refLine != null) {
    const py = b.B - (opts.refLine / opts.yMax) * b.ph;
    ctx.strokeStyle = C.muted;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(b.L, py);
    ctx.lineTo(b.R, py);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = C.muted;
    ctx.font = FONT_SM;
    ctx.textAlign = "left";
    ctx.fillText(opts.refLabel || "", b.L + 4, py - 3);
  }
  const n = labels.length;
  const slot = b.pw / n;
  const barW = Math.min(38, slot * 0.6);
  ctx.textAlign = "center";
  labels.forEach((lab, i) => {
    const cx = b.L + i * slot + slot / 2;
    const v = values[i];
    const bh = (v / opts.yMax) * b.ph;
    const by = b.B - bh;
    ctx.fillStyle = colors[i];
    ctx.fillRect(cx - barW / 2, by, barW, bh);
    if (opts.hatch && opts.hatch[i]) hatch(ctx, cx - barW / 2, by, barW, bh);
    ctx.fillStyle = C.secondary;
    ctx.font = FONT_SM;
    ctx.fillText(opts.vfmt(v), cx, by - 3);
    ctx.fillStyle = C.secondary;
    ctx.font = FONT;
    ctx.save();
    ctx.translate(cx, b.B + 6);
    if (opts.rotate) {
      ctx.rotate(-Math.PI / 5);
      ctx.textAlign = "right";
      ctx.fillText(lab, 0, 6);
    } else {
      ctx.fillText(lab, 0, 8);
    }
    ctx.restore();
  });
  if (opts.xlabel) {
    ctx.fillStyle = C.muted;
    ctx.font = FONT;
    ctx.textAlign = "center";
    ctx.fillText(opts.xlabel, b.L + b.pw / 2, h - 3);
  }
}

/* CoV worked example: raw per-sample TTFT → super-pass bands → per-band p99 */
function drawCovSteps(cvId) {
  const cv = document.getElementById(cvId);
  if (!cv) return;
  const s = STUDY.covSteps;
  const { ctx, w, h } = fitCanvas(cv, 300);
  const b = plotBox(ctx, w, h, { L: 46, R: 16, T: 18, B: 40 });
  const yMax = 1.3;
  yGrid(ctx, b, [
    { v: 0, f: 0 },
    { v: 0.5, f: 0.5 / yMax },
    { v: 1.0, f: 1.0 / yMax },
  ], (v) => v.toFixed(1) + "s");
  const yTo = (v) => b.B - (Math.min(v, yMax) / yMax) * b.ph;

  const totalN = s.sub.reduce((a, arr) => a + arr.length, 0);
  let x = b.L;
  s.sps.forEach((sp, k) => {
    const arr = s.sub[k];
    const bandW = (arr.length / totalN) * b.pw;
    // alternating band tint + boundary
    ctx.fillStyle = k % 2 ? "rgba(42,120,214,0.06)" : "rgba(42,120,214,0.13)";
    ctx.fillRect(x, b.T, bandW, b.ph);
    if (k > 0) {
      ctx.strokeStyle = C.s1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(x, b.T);
      ctx.lineTo(x, b.B);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // per-sample TTFT stems (issue order within band)
    ctx.strokeStyle = "rgba(82,81,78,0.5)";
    ctx.lineWidth = 1;
    arr.forEach((v, i) => {
      const px = x + ((i + 0.5) / arr.length) * bandW;
      ctx.beginPath();
      ctx.moveTo(px, b.B);
      ctx.lineTo(px, yTo(v));
      ctx.stroke();
    });
    // p50 (muted dashed) + p99 (orange solid) aggregate segments
    const p50y = yTo(s.p50[k]);
    ctx.strokeStyle = C.muted;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 2]);
    ctx.beginPath();
    ctx.moveTo(x + 2, p50y);
    ctx.lineTo(x + bandW - 2, p50y);
    ctx.stroke();
    ctx.setLineDash([]);
    const p99y = yTo(s.p99[k]);
    ctx.strokeStyle = C.s2;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(x + 2, p99y);
    ctx.lineTo(x + bandW - 2, p99y);
    ctx.stroke();
    // labels
    ctx.fillStyle = C.s2;
    ctx.font = FONT_SM;
    ctx.textAlign = "center";
    if (arr.length > 12) ctx.fillText(s.p99[k].toFixed(3), x + bandW / 2, p99y - 4);
    ctx.fillStyle = C.secondary;
    ctx.font = FONT;
    ctx.fillText("sp" + sp, x + bandW / 2, b.B + 14);
    x += bandW;
  });
  ctx.fillStyle = C.muted;
  ctx.font = FONT_SM;
  ctx.textAlign = "center";
  ctx.fillText("issue order → (each band = one super-pass = 6396 issued samples; sp22 partial)", b.L + b.pw / 2, b.B + 30);
}

/* cross-mode hairball: cumulative completion vs wall-clock (line chart) */
function drawHairballCurves(cvId) {
  const cv = document.getElementById(cvId);
  if (!cv) return;
  const s = STUDY.hairballCurves;
  const { ctx, w, h } = fitCanvas(cv, 280);
  const b = plotBox(ctx, w, h, { L: 46, R: 16, T: 16, B: 40 });
  yGrid(ctx, b, [
    { v: 0, f: 0 },
    { v: 50, f: 0.5 },
    { v: 100, f: 1 },
  ], (v) => v + "%");
  const xTo = (x) => b.L + (x / 100) * b.pw;
  const yTo = (y) => b.B - (y / 100) * b.ph;
  // diagonal reference (completion evenly paced with wall-clock)
  ctx.strokeStyle = C.muted;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(xTo(0), yTo(0));
  ctx.lineTo(xTo(100), yTo(100));
  ctx.stroke();
  ctx.setLineDash([]);
  const line = (ys, color, dash) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.setLineDash(dash || []);
    ctx.beginPath();
    s.xs.forEach((x, i) => {
      const px = xTo(x), py = yTo(ys[i]);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  };
  line(s.poisson, C.good);
  line(s.off272, C.s2, [5, 3]);
  line(s.off293, C.critical);
  ctx.fillStyle = C.muted;
  ctx.font = FONT;
  ctx.textAlign = "center";
  ctx.fillText("% of wall-clock elapsed  (dashed = paced reference)", b.L + b.pw / 2, b.B + 28);
}

/* ---- the four static study charts ---- */
function renderStudyCharts() {
  drawHairballCurves("hairballCanvas");
  drawCovSteps("covStepsCanvas");
  // A — staging: peak-TTFT factor (blue) vs ramp-duration factor (orange)
  const s = STUDY.staging;
  groupedBars(
    "stagingCanvas",
    s.steps.map((k) => "S=" + k),
    [
      {
        color: C.s1,
        values: s.peak,
        label2: (v) => v.toFixed(2) + "×",
      },
      {
        color: C.s2,
        hatch: true,
        values: s.dur,
        label2: (v) => v.toFixed(2) + "×",
      },
    ],
    {
      yMax: 1.1,
      ticks: [
        { v: 0, f: 0 },
        { v: 0.5, f: 0.5 / 1.1 },
        { v: 1.0, f: 1.0 / 1.1 },
      ],
      yfmt: (v) => v.toFixed(1) + "×",
      xlabel: "ramp stages S  (× relative to burst, lower = flatter)",
    }
  );

  // B — tail vs steady TPOT p50 (ms)
  const t = STUDY.tpot;
  const tMax = Math.max(...t.steady, ...t.tail) * 1.18;
  groupedBars("tpotCanvas", t.pts, [
    { color: C.s1, values: t.steady },
    {
      color: C.s2,
      hatch: true,
      values: t.tail,
      label2: (v, gi) => t.dpct[gi].toFixed(1) + "%",
    },
  ], {
    yMax: tMax,
    ticks: [
      { v: 0, f: 0 },
      { v: 20, f: 20 / tMax },
      { v: 40, f: 40 / tMax },
    ],
    yfmt: (v) => v + "ms",
    xlabel: "point  (orange label = tail Δp50 vs steady)",
  });

  // C — throughput overstatement by concurrency (sequential blue)
  const o = STUDY.overstate;
  coloredBars(
    "overstateCanvas",
    o.pts,
    o.pctv,
    o.pts.map((_, i) => (o.tentative[i] ? "#9ec5f4" : C.s1)),
    {
      yMax: 14,
      ticks: [
        { v: 0, f: 0 },
        { v: 5, f: 5 / 14 },
        { v: 10, f: 10 / 14 },
      ],
      yfmt: (v) => v + "%",
      vfmt: (v) => v.toFixed(1),
      hatch: o.tentative,
      rotate: true,
      xlabel: "concurrency → (lighter/hatched = tentative, config-less)",
    }
  );

  // D — CoV converged-window p99-TTFT error (status good/critical)
  const cv = STUDY.cov;
  coloredBars(
    "covCanvas",
    cv.pts,
    cv.err.map((e) => e * 100),
    cv.err.map((e) => (e < 0.05 ? C.good : e >= 0.15 ? C.critical : C.muted)),
    {
      yMax: 45,
      ticks: [
        { v: 0, f: 0 },
        { v: 15, f: 15 / 45 },
        { v: 30, f: 30 / 45 },
        { v: 45, f: 1 },
      ],
      yfmt: (v) => v + "%",
      vfmt: (v) => v.toFixed(1) + "%",
      hatch: cv.err.map((e) => (e >= 0.15 ? 1 : 0)),
      refLine: 5,
      refLabel: "5% reliable",
      rotate: true,
      xlabel: "point  (green = reliable, red/hatched = misleading despite converging)",
    }
  );
}

/* ================= live detector (ported) ================= */
const DATA = typeof window !== "undefined" ? window.STEADY_REPORT_DATA : null;
const ENSEMBLE = [[3, 0.03], [3, 0.05], [4, 0.05], [5, 0.08], [6, 0.1], [6, 0.15]];
const REL_DRIFT = 0.15;
const SNR = 2.0;

function median(a) {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function pstdev(a) {
  if (a.length < 2) return 0;
  const m = a.reduce((s, x) => s + x, 0) / a.length;
  return Math.sqrt(a.reduce((s, x) => s + (x - m) * (x - m), 0) / a.length);
}
function cov(a) {
  if (a.length < 2) return 0;
  const m = a.reduce((s, x) => s + x, 0) / a.length;
  return m === 0 ? 0 : pstdev(a) / Math.abs(m);
}

function analyzeTrend(vals, relThr, snrThr) {
  const n = vals.length;
  if (n < 4) return { verdict: "insufficient", rel_drift: 0, snr: 0 };
  const xbar = (n - 1) / 2;
  const ybar = vals.reduce((s, x) => s + x, 0) / n;
  let sxx = 0, sxy = 0;
  for (let x = 0; x < n; x++) {
    sxx += (x - xbar) ** 2;
    sxy += (x - xbar) * (vals[x] - ybar);
  }
  const slope = sxx ? sxy / sxx : 0;
  const intercept = ybar - slope * xbar;
  const resid = vals.map((v, x) => v - (intercept + slope * x));
  const totalChange = slope * (n - 1);
  const med = median(vals) || 1e-9;
  const relDrift = totalChange / med;
  const snr = Math.abs(totalChange) / (pstdev(resid) + 1e-12);
  let verdict = "steady";
  if (Math.abs(relDrift) >= relThr && snr >= snrThr)
    verdict = relDrift > 0 ? "drifting_up" : "drifting_down";
  return { verdict, rel_drift: relDrift, snr };
}
function adaptiveWarmup(driver, band, minW = 1, maxFrac = 0.5) {
  const n = driver.length;
  if (n < 4) return minW;
  const back = driver.slice(n >> 1);
  const steady = median(back) || 1e-9;
  const thr = steady * (1 + band);
  const cap = Math.max(minW, Math.floor(n * maxFrac));
  let w = 0;
  while (w < cap && driver[w] > thr) w++;
  return Math.max(minW, w);
}
function covConverged(metricArrays, window, bound, warmup) {
  const n = metricArrays[0].length;
  for (let spEnd = warmup + window; spEnd <= n; spEnd++) {
    let ok = true;
    for (const arr of metricArrays) {
      if (cov(arr.slice(spEnd - window, spEnd)) >= bound) {
        ok = false;
        break;
      }
    }
    if (ok) return spEnd;
  }
  return null;
}
function ensembleVote(metricArrays, warmup) {
  const ends = [];
  for (const [w, b] of ENSEMBLE) {
    const r = covConverged(metricArrays, w, b, warmup);
    if (r !== null) ends.push(r);
  }
  const n = metricArrays[0].length;
  let conc = 0;
  if (ends.length >= 2)
    conc = Math.max(0, 1 - (Math.max(...ends) - Math.min(...ends)) / Math.max(1, n - warmup));
  return { nConv: ends.length, ends, conc };
}
function plateauEdges(rate, band) {
  const steady = median(rate);
  const lo = steady * (1 - band), hi = steady * (1 + band);
  const inband = rate.map((r) => r >= lo && r <= hi);
  let best = [0, 0], i = 0;
  while (i < rate.length) {
    if (inband[i]) {
      let j = i;
      while (j < rate.length && inband[j]) j++;
      if (j - i > best[1] - best[0]) best = [i, j];
      i = j;
    } else i++;
  }
  return { start: best[0], end: best[1], steady, lo, hi, rate: median(rate.slice(best[0], best[1])) };
}

function isOffline(run) {
  return run.mode === "offline";
}
function metricArrayCC(run, key) {
  if (key === "qps") return run.series.map((sp) => sp.qps);
  const [g, p] = key.split(":");
  return run.series.map((sp) => (g === "ttft" ? sp.ttft : sp.lat)[p]);
}
function driftMetricArrays(run) {
  return [
    run.series.map((sp) => sp.ttft["0.5"]),
    run.series.map((sp) => sp.ttft["0.99"]),
    run.series.map((sp) => sp.lat["0.5"]),
    run.series.map((sp) => sp.lat["0.99"]),
  ];
}

function drawDetector(run, params) {
  const cv = document.getElementById("seriesCanvas");
  const { ctx, w, h } = fitCanvas(cv, 380);
  const L = 62, R = 20, T = 24, B = 40;
  const pw = w - L - R, ph = h - T - B;
  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 1;
  ctx.strokeRect(L, T, pw, ph);

  const off = isOffline(run);
  const ys = off
    ? run.complete_per_bin.map((c) => c / (run.bin_seconds / 60))
    : metricArrayCC(run, params.metric);
  const n = ys.length;
  const maxY = Math.max(1, ...ys), minY = 0;
  const xToPix = (i) => L + (n <= 1 ? 0 : (i / (n - 1)) * pw);
  const yToPix = (v) => T + ph - ((v - minY) / (maxY - minY)) * ph;

  let win = null, warmup = null, covStop = null, plateau = null, verdict = null, ens = null;
  if (off) {
    plateau = plateauEdges(ys, params.band);
    win = [plateau.start, plateau.end];
    for (const g of [plateau.lo, plateau.hi]) {
      ctx.strokeStyle = C.baseline;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(L, yToPix(g));
      ctx.lineTo(L + pw, yToPix(g));
      ctx.stroke();
      ctx.setLineDash([]);
    }
  } else {
    warmup = adaptiveWarmup(run.series.map((sp) => sp.ttft["0.99"]), params.band);
    win = [warmup, n];
    covStop = covConverged(driftMetricArrays(run), params.covWindow, params.covBound, warmup);
    ens = ensembleVote(driftMetricArrays(run), warmup);
    verdict = analyzeTrend(ys.slice(warmup), REL_DRIFT, SNR);
  }

  if (win && win[1] > win[0]) {
    ctx.fillStyle = C.steadyShade;
    ctx.fillRect(xToPix(win[0]), T, xToPix(win[1] - 1) - xToPix(win[0]), ph);
  }
  ctx.fillStyle = C.dropShade;
  if (win && win[0] > 0) ctx.fillRect(L, T, xToPix(win[0]) - L, ph);
  if (win && win[1] < n) ctx.fillRect(xToPix(win[1] - 1), T, L + pw - xToPix(win[1] - 1), ph);

  ctx.strokeStyle = C.s1;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ys.forEach((v, i) => {
    const px = xToPix(i), py = yToPix(v);
    i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
  });
  ctx.stroke();

  function vline(i, color, label, dash) {
    const px = xToPix(i);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(px, T);
    ctx.lineTo(px, T + ph);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = color;
    ctx.font = FONT;
    ctx.textAlign = "left";
    ctx.fillText(label, Math.min(px + 4, L + pw - 90), T + 12);
  }
  if (off) {
    vline(plateau.start, C.good, "plateau start", [6, 3]);
    if (plateau.end < n) vline(plateau.end, C.critical, "drain-onset", [2, 3]);
  } else {
    if (warmup > 0) vline(warmup, C.good, "warmup crop", [6, 3]);
    if (covStop !== null) vline(covStop, C.critical, "CoV stop", [2, 3]);
  }

  ctx.fillStyle = C.muted;
  ctx.font = FONT;
  ctx.textAlign = "left";
  ctx.fillText(fmt(maxY), 6, T + 8);
  ctx.fillText("0", 6, T + ph);
  ctx.textAlign = "center";
  ctx.fillText(off ? "minutes" : "super-pass", L + pw / 2, h - 12);

  updatePanels(run, params, { win, warmup, covStop, plateau, verdict, ens });
}

function fmt(v, d = 1) {
  if (v == null || !isFinite(v)) return "–";
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return v.toLocaleString(undefined, { maximumFractionDigits: d });
}
function nsToS(v) {
  return v == null ? null : v / 1e9;
}
function pct(a, b) {
  return a == null || b == null || b === 0 ? "–" : `${((a / b - 1) * 100).toFixed(0)}%`;
}

function updatePanels(run, params, det) {
  const diff = document.querySelector("#diffTable tbody");
  const verd = document.querySelector("#verdictTable tbody");
  const cap = document.getElementById("seriesCaption");

  if (isOffline(run)) {
    const ys = run.complete_per_bin.map((c) => c / (run.bin_seconds / 60));
    const fullRate = ys.reduce((s, x) => s + x, 0) / ys.length;
    const p = det.plateau;
    diff.innerHTML =
      row3("metric", "full-run", "steady window", true) +
      row3("throughput (req/s)", fmt(fullRate), fmt(p.rate), false, pct(p.rate, fullRate));
    verd.innerHTML =
      row2("plateau bins", `${p.start}..${p.end - 1} (${p.end - p.start} of ${ys.length})`) +
      row2("dropped", `settle 0..${p.start - 1}, drain ${p.end}..${ys.length - 1}`) +
      row2("recovery", `full-run undercounts steady by ${fmt((1 - fullRate / p.rate) * 100, 0)}%`);
    cap.textContent =
      `${run.label} — completion-rate/min; plateau band ±${(params.band * 100).toFixed(0)}%. ` +
      `Green = steady plateau; red = ramp/drain (dropped from the sustained rate).`;
    return;
  }

  const e = run.exact;
  diff.innerHTML =
    row3("metric", "full-run (reported)", `steady (warmup ${e.warmup})`, true) +
    row3("p99 TTFT (s)", fmt(nsToS(e.total.ttft_p99), 2), fmt(nsToS(e.steady.ttft_p99), 2), false, pct(e.steady.ttft_p99, e.total.ttft_p99)) +
    row3("p50 TTFT (s)", fmt(nsToS(e.total.ttft_p50), 2), fmt(nsToS(e.steady.ttft_p50), 2), false, pct(e.steady.ttft_p50, e.total.ttft_p50)) +
    row3("p99 latency (s)", fmt(nsToS(e.total.lat_p99), 1), fmt(nsToS(e.steady.lat_p99), 1), false, pct(e.steady.lat_p99, e.total.lat_p99)) +
    row3("QPS", fmt(e.total.qps), fmt(e.steady.qps), false, pct(e.steady.qps, e.total.qps));

  const names = [["QPS", "qps"], ["p50 TTFT", "ttft:0.5"], ["p99 TTFT", "ttft:0.99"], ["p50 lat", "lat:0.5"], ["p99 lat", "lat:0.99"]];
  let rows = "";
  for (const [nm, key] of names) {
    const t = analyzeTrend(metricArrayCC(run, key).slice(det.warmup), REL_DRIFT, SNR);
    rows += `<tr><td>${nm}</td><td>${t.verdict.replace("_", " ")}</td><td class="num">${t.rel_drift.toFixed(2)}</td><td class="num">${t.snr.toFixed(1)}</td></tr>`;
  }
  verd.innerHTML =
    `<tr><th>metric</th><th>verdict</th><th class="num">rel_drift</th><th class="num">snr</th></tr>` +
    rows +
    `<tr><td colspan="4" class="ens">CoV ensemble: ${det.ens.nConv}/6 converged, concordance ${det.ens.conc.toFixed(2)}; live (cov_window ${params.covWindow}, cov_bound ${params.covBound}) stop = ${det.covStop == null ? "UNCONVERGED" : "super-pass " + det.covStop}</td></tr>`;

  cap.textContent =
    `${run.label} — ${labelFor(params.metric)} per super-pass. ` +
    `Green = steady window (adaptive warmup ${det.warmup}); red = dropped ramp. ` +
    `Recovery table is exact at the tool's warmup; sliders drive the live warmup edge, CoV stop, and drift verdicts.`;
}
function row3(a, b, c, head, extra) {
  const tag = head ? "th" : "td";
  const ex = extra != null ? `<${tag} class="num">${extra}</${tag}>` : head ? `<${tag} class="num">Δ</${tag}>` : "<td></td>";
  return `<tr><${tag}>${a}</${tag}><${tag}>${b}</${tag}><${tag}>${c}</${tag}>${ex}</tr>`;
}
function row2(a, b) {
  return `<tr><td>${a}</td><td colspan="3">${b}</td></tr>`;
}
function labelFor(k) {
  return { qps: "QPS", "ttft:0.99": "p99 TTFT", "ttft:0.5": "p50 TTFT", "lat:0.99": "p99 latency", "lat:0.5": "p50 latency" }[k];
}

/* ---- UI ---- */
function params() {
  return {
    metric: document.getElementById("metricSelect").value,
    band: parseFloat(document.getElementById("band").value),
    covWindow: parseInt(document.getElementById("covWindow").value, 10),
    covBound: parseFloat(document.getElementById("covBound").value),
  };
}
function currentRun() {
  return DATA.runs[document.getElementById("runSelect").value];
}
function renderDetector() {
  const run = currentRun();
  const off = isOffline(run);
  document.getElementById("metricWrap").style.display = off ? "none" : "";
  document.getElementById("cwWrap").style.display = off ? "none" : "";
  document.getElementById("cbWrap").style.display = off ? "none" : "";
  document.getElementById("bandOut").textContent = parseFloat(document.getElementById("band").value).toFixed(2);
  document.getElementById("cwOut").textContent = document.getElementById("covWindow").value;
  document.getElementById("cbOut").textContent = parseFloat(document.getElementById("covBound").value).toFixed(2);
  drawDetector(run, params());
}
function renderAll() {
  renderStudyCharts();
  if (DATA) renderDetector();
}
function init() {
  renderStudyCharts();
  if (!DATA) {
    const host = document.querySelector("#seriesCanvas");
    if (host) host.replaceWith(Object.assign(document.createElement("p"), { textContent: "report_data.js not loaded — live detector unavailable." }));
    return;
  }
  const sel = document.getElementById("runSelect");
  DATA.runs.forEach((r, i) => {
    const o = document.createElement("option");
    o.value = i;
    o.textContent = `${r.label} [${r.mode}]`;
    sel.appendChild(o);
  });
  ["runSelect", "metricSelect", "band", "covWindow", "covBound"].forEach((id) =>
    document.getElementById(id).addEventListener("input", renderDetector)
  );
  window.addEventListener("resize", renderAll);
  renderDetector();
}

if (typeof module !== "undefined") {
  module.exports = {
    median, cov, analyzeTrend, adaptiveWarmup, plateauEdges,
    covConverged, ensembleVote, driftMetricArrays, metricArrayCC,
  };
}
if (typeof document !== "undefined") init();
