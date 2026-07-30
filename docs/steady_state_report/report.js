/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Detection re-implemented from src/inference_endpoint/metrics/steady_state/
 * {drift.py, stopping.py, window.py} + scripts/steady_state_drift.py. Runs live. */
"use strict";

const DATA = typeof window !== "undefined" ? window.STEADY_REPORT_DATA : null;
const ENSEMBLE = [[3, 0.03], [3, 0.05], [4, 0.05], [5, 0.08], [6, 0.1], [6, 0.15]];
const REL_DRIFT = 0.15;
const SNR = 2.0;

/* ---- stats ---- */
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
function percentileLower(sorted, p) {
  return sorted[Math.floor(p * (sorted.length - 1))];
}

/* ---- detection (mirrors drift.py / stopping.py) ---- */
function analyzeTrend(vals, relThr, snrThr) {
  const n = vals.length;
  if (n < 4) return { verdict: "insufficient", rel_drift: 0, snr: 0 };
  const xbar = (n - 1) / 2;
  const ybar = vals.reduce((s, x) => s + x, 0) / n;
  let sxx = 0, sxy = 0;
  for (let x = 0; x < n; x++) { sxx += (x - xbar) ** 2; sxy += (x - xbar) * (vals[x] - ybar); }
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
      if (cov(arr.slice(spEnd - window, spEnd)) >= bound) { ok = false; break; }
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

/* offline plateau-edge (mirrors the completion-rate detector) */
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

/* ---- series extraction ---- */
function isOffline(run) { return run.mode === "offline"; }
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

/* ---- canvas helpers ---- */
function fitCanvas(cv, h) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(360, Math.floor(cv.clientWidth));
  cv.width = Math.floor(w * dpr); cv.height = Math.floor(h * dpr);
  cv.style.height = h + "px";
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function draw(run, params) {
  const cv = document.getElementById("seriesCanvas");
  const { ctx, w, h } = fitCanvas(cv, 380);
  const L = 62, R = 20, T = 24, B = 40;
  const pw = w - L - R, ph = h - T - B;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fffdf8"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#d8d1c5"; ctx.lineWidth = 1; ctx.strokeRect(L, T, pw, ph);

  const off = isOffline(run);
  const ys = off ? run.complete_per_bin.map((c) => c / (run.bin_seconds / 60)) : metricArrayCC(run, params.metric);
  const n = ys.length;
  const maxY = Math.max(1, ...ys), minY = 0;
  const xToPix = (i) => L + (n <= 1 ? 0 : (i / (n - 1)) * pw);
  const yToPix = (v) => T + ph - ((v - minY) / (maxY - minY)) * ph;

  let win = null, warmup = null, covStop = null, plateau = null, verdict = null, ens = null;
  if (off) {
    plateau = plateauEdges(ys, params.band);
    win = [plateau.start, plateau.end];
    // band guides
    for (const g of [plateau.lo, plateau.hi]) {
      ctx.strokeStyle = "#d9c9a6"; ctx.setLineDash([3, 3]); ctx.beginPath();
      ctx.moveTo(L, yToPix(g)); ctx.lineTo(L + pw, yToPix(g)); ctx.stroke(); ctx.setLineDash([]);
    }
  } else {
    warmup = adaptiveWarmup(run.series.map((sp) => sp.ttft["0.99"]), params.band);
    win = [warmup, n];
    covStop = covConverged(driftMetricArrays(run), params.covWindow, params.covBound, warmup);
    ens = ensembleVote(driftMetricArrays(run), warmup);
    verdict = analyzeTrend(ys.slice(warmup), REL_DRIFT, SNR);
  }

  // shade steady window
  if (win && win[1] > win[0]) {
    ctx.fillStyle = "rgba(71,111,54,0.12)";
    ctx.fillRect(xToPix(win[0]), T, xToPix(win[1] - 1) - xToPix(win[0]), ph);
  }
  // shade dropped regions (light red)
  ctx.fillStyle = "rgba(155,52,40,0.08)";
  if (win && win[0] > 0) ctx.fillRect(L, T, xToPix(win[0]) - L, ph);
  if (win && win[1] < n) ctx.fillRect(xToPix(win[1] - 1), T, L + pw - xToPix(win[1] - 1), ph);

  // the metric line
  ctx.strokeStyle = "#087f8c"; ctx.lineWidth = 2; ctx.beginPath();
  ys.forEach((v, i) => { const px = xToPix(i), py = yToPix(v); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); });
  ctx.stroke();

  // markers
  function vline(i, color, label, dash) {
    const px = xToPix(i);
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash);
    ctx.beginPath(); ctx.moveTo(px, T); ctx.lineTo(px, T + ph); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = color; ctx.font = "11px Inter, system-ui, sans-serif";
    ctx.fillText(label, Math.min(px + 4, L + pw - 90), T + 12);
  }
  if (off) {
    vline(plateau.start, "#476f36", "plateau start", [6, 3]);
    if (plateau.end < n) vline(plateau.end, "#9b3428", "drain-onset", [2, 3]);
  } else {
    if (warmup > 0) vline(warmup, "#476f36", "warmup crop", [6, 3]);
    if (covStop !== null) vline(covStop, "#9b3428", "CoV stop", [2, 3]);
  }

  // axes labels
  ctx.fillStyle = "#657075"; ctx.font = "11px Inter, system-ui, sans-serif";
  ctx.fillText(fmt(maxY), 6, T + 8);
  ctx.fillText("0", 6, T + ph);
  ctx.fillText(off ? "minutes" : "super-pass", L + pw / 2 - 26, h - 12);

  updatePanels(run, params, { win, warmup, covStop, plateau, verdict, ens });
}

/* ---- panels ---- */
function fmt(v, d = 1) {
  if (v == null || !isFinite(v)) return "–";
  if (Math.abs(v) >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return v.toLocaleString(undefined, { maximumFractionDigits: d });
}
function nsToS(v) { return v == null ? null : v / 1e9; }
function pct(a, b) { return a == null || b == null || b === 0 ? "–" : `${((a / b - 1) * 100).toFixed(0)}%`; }

function updatePanels(run, params, det) {
  const diff = document.querySelector("#diffTable tbody");
  const verd = document.querySelector("#verdictTable tbody");
  const cap = document.getElementById("seriesCaption");

  if (isOffline(run)) {
    const ys = run.complete_per_bin.map((c) => c / (run.bin_seconds / 60));
    const full = median(ys) * 0; // placeholder to avoid lint
    const fullRate = ys.reduce((s, x) => s + x, 0) / ys.length;
    const p = det.plateau;
    diff.innerHTML = row3("metric", "full-run", "steady window", true) +
      row3("throughput (req/s)", fmt(fullRate), fmt(p.rate), false, pct(p.rate, fullRate));
    verd.innerHTML = row2("plateau bins", `${p.start}..${p.end - 1} (${p.end - p.start} of ${ys.length})`) +
      row2("dropped", `settle 0..${p.start - 1}, drain ${p.end}..${ys.length - 1}`) +
      row2("recovery", `full-run undercounts steady by ${fmt((1 - fullRate / p.rate) * 100, 0)}%`);
    cap.textContent = `${run.label} — completion-rate/min; plateau band ±${(params.band * 100).toFixed(0)}%. `
      + `Green = steady plateau; red = ramp/drain (dropped from the sustained rate).`;
    void full;
    return;
  }

  const e = run.exact;
  diff.innerHTML =
    row3("metric", "full-run (reported)", `steady (warmup ${e.warmup})`, true) +
    row3("p99 TTFT (s)", fmt(nsToS(e.total.ttft_p99), 2), fmt(nsToS(e.steady.ttft_p99), 2), false, pct(e.steady.ttft_p99, e.total.ttft_p99)) +
    row3("p50 TTFT (s)", fmt(nsToS(e.total.ttft_p50), 2), fmt(nsToS(e.steady.ttft_p50), 2), false, pct(e.steady.ttft_p50, e.total.ttft_p50)) +
    row3("p99 latency (s)", fmt(nsToS(e.total.lat_p99), 1), fmt(nsToS(e.steady.lat_p99), 1), false, pct(e.steady.lat_p99, e.total.lat_p99)) +
    row3("QPS", fmt(e.total.qps), fmt(e.steady.qps), false, pct(e.steady.qps, e.total.qps));

  const names = [["QPS", "qps"], ["p50 TTFT", "ttft:0.5"], ["p99 TTFT", "ttft:0.99"],
    ["p50 lat", "lat:0.5"], ["p99 lat", "lat:0.99"]];
  let rows = "";
  for (const [nm, key] of names) {
    const t = analyzeTrend(metricArrayCC(run, key).slice(det.warmup), REL_DRIFT, SNR);
    rows += `<tr><td>${nm}</td><td>${t.verdict.replace("_", " ")}</td>` +
      `<td>${t.rel_drift.toFixed(2)}</td><td>${t.snr.toFixed(1)}</td></tr>`;
  }
  verd.innerHTML = `<tr><th>metric</th><th>verdict</th><th>rel_drift</th><th>snr</th></tr>` + rows +
    `<tr><td colspan="4" class="ens">CoV ensemble: ${det.ens.nConv}/6 converged, ` +
    `concordance ${det.ens.conc.toFixed(2)}; live (cov_window ${params.covWindow}, cov_bound ${params.covBound}) ` +
    `stop = ${det.covStop == null ? "UNCONVERGED" : "super-pass " + det.covStop}</td></tr>`;

  cap.textContent = `${run.label} — ${labelFor(params.metric)} per super-pass. `
    + `Green = steady window (adaptive warmup ${det.warmup}); red = dropped ramp. `
    + `Recovery table is exact at the tool's warmup; sliders drive the live warmup edge, CoV stop, and drift verdicts.`;
}
function row3(a, b, c, head, extra) {
  const tag = head ? "th" : "td";
  const ex = extra != null ? `<${tag}>${extra}</${tag}>` : (head ? `<${tag}>Δ</${tag}>` : "<td></td>");
  return `<tr><${tag}>${a}</${tag}><${tag}>${b}</${tag}><${tag}>${c}</${tag}>${ex}</tr>`;
}
function row2(a, b) { return `<tr><td>${a}</td><td colspan="3">${b}</td></tr>`; }
function labelFor(k) {
  return { "qps": "QPS", "ttft:0.99": "p99 TTFT", "ttft:0.5": "p50 TTFT", "lat:0.99": "p99 latency", "lat:0.5": "p50 latency" }[k];
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
function currentRun() { return DATA.runs[document.getElementById("runSelect").value]; }
function render() {
  const run = currentRun();
  const off = isOffline(run);
  document.getElementById("metricWrap").style.display = off ? "none" : "";
  document.getElementById("cwWrap").style.display = off ? "none" : "";
  document.getElementById("cbWrap").style.display = off ? "none" : "";
  document.getElementById("bandOut").textContent = parseFloat(document.getElementById("band").value).toFixed(2);
  document.getElementById("cwOut").textContent = document.getElementById("covWindow").value;
  document.getElementById("cbOut").textContent = parseFloat(document.getElementById("covBound").value).toFixed(2);
  draw(run, params());
}
function init() {
  if (!DATA) { document.body.innerHTML = "<p>report_data.js not loaded.</p>"; return; }
  const sel = document.getElementById("runSelect");
  DATA.runs.forEach((r, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = `${r.label} [${r.mode}]`; sel.appendChild(o);
  });
  ["runSelect", "metricSelect", "band", "covWindow", "covBound"].forEach((id) =>
    document.getElementById(id).addEventListener("input", render));
  window.addEventListener("resize", render);
  render();
}

if (typeof module !== "undefined") {
  module.exports = {
    median, cov, analyzeTrend, adaptiveWarmup, plateauEdges,
    covConverged, ensembleVote, driftMetricArrays, metricArrayCC,
  };
}
if (typeof document !== "undefined") init();
