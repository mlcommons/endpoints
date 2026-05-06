#!/usr/bin/env python3
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

"""Generate a single dataset analysis summary plot from a flat multi-turn JSONL file.

A *flat* JSONL has one message per line with keys:
    conversation_id, turn, role, content, system, tools,
    reasoning_content, tool_calls, tool_results

System and tools appear only on the first row of each conversation.

For an input file ``<dir>/<stem>.jsonl`` the script writes a single composite plot:
    <dir>/analysis/<stem>_summary.png

The summary contains:
    Row 1: Turns per conversation, Final-turn ISL, Total OSL  (per-conversation histograms)
    Row 2: New ISL per turn, OSL per assistant message, Median ISL growth per turn
    Row 3: Token Distribution per Assistant Turn, OSL Token Classifications  (violins)

Usage:
    python examples/09_MultiTurn/scripts/analyze_flat_jsonl.py path/to/file.jsonl [--title "..."]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("analyze_flat_jsonl")

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("o200k_base")
except Exception:
    _ENC = None
    log.warning("tiktoken unavailable - falling back to len(text)//4")


def count_tokens(text: str | None) -> int:
    if not text:
        return 0
    if _ENC is None:
        return len(text) // 4
    return len(_ENC.encode(text))


def _content_tokens(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return count_tokens(content)
    if isinstance(content, list):
        n = 0
        for blk in content:
            if isinstance(blk, dict):
                t = blk.get("type")
                if t == "text":
                    n += count_tokens(blk.get("text", ""))
                elif t == "thinking":
                    n += count_tokens(blk.get("thinking", ""))
                elif t == "tool_use":
                    inp = blk.get("input", {})
                    n += count_tokens(
                        inp
                        if isinstance(inp, str)
                        else json.dumps(inp, ensure_ascii=False)
                    )
                elif t == "tool_result":
                    c = blk.get("content")
                    if isinstance(c, str):
                        n += count_tokens(c)
                    elif isinstance(c, list):
                        n += _content_tokens(c)
                    else:
                        n += count_tokens(json.dumps(c, ensure_ascii=False))
                else:
                    n += count_tokens(json.dumps(blk, ensure_ascii=False))
            else:
                n += count_tokens(str(blk))
        return n
    return count_tokens(json.dumps(content, ensure_ascii=False))


def _assistant_osl_split(row: dict[str, Any]) -> tuple[int, int, int]:
    """Return (thinking, tool_call, prose) token counts for an assistant message."""
    thinking = 0
    tool_call = 0
    prose = 0
    rc = row.get("reasoning_content")
    if isinstance(rc, str):
        thinking += count_tokens(rc)
    content = row.get("content")
    if isinstance(content, str):
        prose += count_tokens(content)
    elif isinstance(content, list):
        for blk in content:
            if isinstance(blk, dict):
                t = blk.get("type")
                if t == "thinking":
                    thinking += count_tokens(blk.get("thinking", ""))
                elif t == "text":
                    prose += count_tokens(blk.get("text", ""))
                elif t == "tool_use":
                    inp = blk.get("input", {})
                    tool_call += count_tokens(
                        inp
                        if isinstance(inp, str)
                        else json.dumps(inp, ensure_ascii=False)
                    )
                else:
                    prose += count_tokens(json.dumps(blk, ensure_ascii=False))
            else:
                prose += count_tokens(str(blk))
    elif content is not None:
        prose += count_tokens(json.dumps(content, ensure_ascii=False))
    for tc in row.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments", "")
        if isinstance(args, str):
            tool_call += count_tokens(args)
        elif args is not None:
            tool_call += count_tokens(json.dumps(args, ensure_ascii=False))
        tool_call += count_tokens(fn.get("name", ""))
    return thinking, tool_call, prose


def _row_tokens(row: dict[str, Any]) -> int:
    if row.get("role") == "assistant":
        return sum(_assistant_osl_split(row))
    n = _content_tokens(row.get("content"))
    rc = row.get("reasoning_content")
    if isinstance(rc, str):
        n += count_tokens(rc)
    for tc in row.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments", "")
        if isinstance(args, str):
            n += count_tokens(args)
        elif args is not None:
            n += count_tokens(json.dumps(args, ensure_ascii=False))
        n += count_tokens(fn.get("name", ""))
    for tr in row.get("tool_results") or []:
        if isinstance(tr, dict):
            for k in ("content", "output", "result"):
                v = tr.get(k)
                if v is not None:
                    n += _content_tokens(v)
                    break
            else:
                n += count_tokens(json.dumps(tr, ensure_ascii=False))
        else:
            n += count_tokens(str(tr))
    return n


def _system_tools_tokens(first_row: dict[str, Any]) -> int:
    n = count_tokens(first_row.get("system") or "")
    tools = first_row.get("tools")
    if tools:
        n += count_tokens(json.dumps(tools, ensure_ascii=False))
    return n


# ---------------------------------------------------------------------------
# Per-conversation walk
# ---------------------------------------------------------------------------


def _conversation_stats(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    sys_tok = _system_tools_tokens(rows[0])
    msg_tokens = [_row_tokens(r) for r in rows]
    assistant_idxs = [i for i, r in enumerate(rows) if r.get("role") == "assistant"]
    if not assistant_idxs:
        return None

    new_isl: list[int] = []
    osl: list[int] = []
    osl_thinking: list[int] = []
    osl_tool_call: list[int] = []
    osl_prose: list[int] = []
    precomputed: list[int] = []
    cumulative_isl: list[int] = []

    prev_a = -1
    running_before = sys_tok
    for a_idx in assistant_idxs:
        between = sum(msg_tokens[prev_a + 1 : a_idx])
        precomputed.append(running_before)
        new_isl.append(between)
        osl.append(msg_tokens[a_idx])
        th, tc, pr = _assistant_osl_split(rows[a_idx])
        osl_thinking.append(th)
        osl_tool_call.append(tc)
        osl_prose.append(pr)
        cumulative_isl.append(running_before + between)
        running_before += between + msg_tokens[a_idx]
        prev_a = a_idx

    return {
        "turns": len(assistant_idxs),
        "final_turn_isl": cumulative_isl[-1],
        "total_osl": sum(osl),
        "new_isl_per_turn": new_isl,
        "osl_per_message": osl,
        "precomputed_per_turn": precomputed,
        "cumulative_isl_per_turn": cumulative_isl,
        "osl_thinking_per_turn": osl_thinking,
        "osl_tool_call_per_turn": osl_tool_call,
        "osl_prose_per_turn": osl_prose,
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_flat(path: Path) -> dict[str, list[dict[str, Any]]]:
    convs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            # Skip optional dataset_metadata sentinel record (license/source
            # attribution carried on line 1 of some upstream snapshots).
            if r.get("_type") == "dataset_metadata" or "conversation_id" not in r:
                continue
            convs[r["conversation_id"]].append(r)
    for rows in convs.values():
        rows.sort(key=lambda r: r.get("turn", 0))
    return convs


# ---------------------------------------------------------------------------
# Plotting helpers (all draw onto a provided axes)
# ---------------------------------------------------------------------------

_PANELS = [
    ("turns_per_conversation", "Turns per conversation", "assistant turns", "#3b78c2"),
    ("final_turn_isl", "Final-turn ISL (per conv.)", "input tokens", "#3b78c2"),
    ("total_osl", "Total OSL (per conv.)", "output tokens", "#f5b400"),
    ("new_isl_per_turn", "New ISL per turn", "tokens added between turns", "#3b78c2"),
    ("osl_per_message", "OSL per assistant message", "output tokens", "#f5b400"),
]


def _percentiles(vals: list[int]) -> dict[str, float]:
    if not vals:
        return {"p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0}
    a = np.asarray(vals)
    return {
        "p25": float(np.percentile(a, 25)),
        "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)),
        "p95": float(np.percentile(a, 95)),
    }


def _draw_hist(ax, vals: list[int], title: str, xlabel: str, color: str) -> None:
    if not vals:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    ax.hist(vals, bins=40, color=color, edgecolor="white")
    p = _percentiles(vals)
    ax.axvline(
        p["p50"], color="k", linestyle="--", linewidth=1, label=f"p50={p['p50']:,.0f}"
    )
    ax.axvline(
        p["p95"], color="r", linestyle=":", linewidth=1, label=f"p95={p['p95']:,.0f}"
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("count", fontsize=10)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, axis="y", alpha=0.25)


def _draw_isl_growth(ax, per_conv_cum: list[list[int]], min_n: int = 10) -> None:
    if not per_conv_cum:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    max_t = max(len(c) for c in per_conv_cum)
    by_turn: list[list[int]] = [[] for _ in range(max_t)]
    for c in per_conv_cum:
        for t, v in enumerate(c):
            by_turn[t].append(v)

    turns, p25s, p50s, p75s = [], [], [], []
    for t, vals in enumerate(by_turn, start=1):
        if len(vals) < min_n:
            continue
        turns.append(t)
        p25s.append(np.percentile(vals, 25))
        p50s.append(np.percentile(vals, 50))
        p75s.append(np.percentile(vals, 75))

    if turns:
        ax.fill_between(turns, p25s, p75s, alpha=0.25, color="#dd8452", label="P25-P75")
        ax.plot(turns, p50s, color="#dd8452", linewidth=2.0, label="P50")
    ax.set_xlabel("Turn index (Nth assistant generation)", fontsize=10)
    ax.set_ylabel("ISL (input tokens in prompt)", fontsize=10)
    ax.set_title(
        f"Median ISL growth per turn  (n>={min_n})", fontsize=12, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.tick_params(axis="both", labelsize=9)


def _log_decade_ticks(ax, vmax: float) -> None:
    decades = [
        d
        for d in (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000)
        if d <= max(vmax, 10) * 1.5
    ]
    ax.set_yticks([np.log10(d) for d in decades])
    ax.set_yticklabels(
        [
            f"{d//1_000_000}M"
            if d >= 1_000_000
            else (f"{d//1000}k" if d >= 1000 else str(d))
            for d in decades
        ]
    )


def _draw_violin_token_dist(
    ax, precomputed: list[int], new_isl: list[int], osl: list[int]
) -> None:
    series = [
        ("Precomputed ISL", precomputed, "#1f3fa6"),
        ("New ISL", new_isl, "#e7522e"),
        ("OSL", osl, "#2ca02c"),
    ]
    raw = [np.array(vals, dtype=float) for _, vals, _ in series]
    log_data = [np.log10(np.clip(v, 1, None)) for v in raw]

    parts = ax.violinplot(
        log_data, showmeans=False, showmedians=False, showextrema=False, widths=0.85
    )
    for pc, (_, _, color) in zip(parts["bodies"], series, strict=False):
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.85)

    for i, vals in enumerate(raw, start=1):
        if vals.size == 0:
            continue
        clipped = np.clip(vals, 1, None)
        q1, med, q3 = np.percentile(clipped, [25, 50, 75])
        mean = clipped.mean()
        ax.add_patch(
            plt.Rectangle(
                (i - 0.08, np.log10(q1)),
                0.16,
                np.log10(q3) - np.log10(q1),
                fill=True,
                facecolor="white",
                edgecolor="black",
                linewidth=1,
                zorder=5,
            )
        )
        ax.hlines(
            np.log10(med), i - 0.08, i + 0.08, colors="black", linewidth=2.0, zorder=6
        )
        ax.hlines(
            np.log10(mean),
            i - 0.08,
            i + 0.08,
            colors="black",
            linestyles="dashed",
            linewidth=1.2,
            zorder=6,
        )

    vmax = max((v.max() if v.size else 1) for v in raw)
    _log_decade_ticks(ax, vmax)
    ax.set_ylabel("Number of Tokens (log scale)", fontsize=10)
    ax.set_xticks(range(1, len(series) + 1))
    ax.set_xticklabels([s[0] for s in series], fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    legend_elems = [
        plt.Line2D([0], [0], color="black", linewidth=2.0, label="Median"),
        plt.Line2D(
            [0], [0], color="black", linestyle="dashed", linewidth=1.2, label="Mean"
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            fill=True,
            facecolor="white",
            edgecolor="black",
            label="IQR box (Q1-Q3)",
        ),
    ]
    ax.legend(
        handles=legend_elems,
        title="Statistics",
        loc="upper right",
        fontsize=9,
        framealpha=0.95,
    )
    ax.set_title(
        "Token Distribution per Assistant Turn", fontsize=12, fontweight="bold"
    )


def _draw_violin_osl_categories(
    ax, thinking: list[int], tool_call: list[int], prose: list[int]
) -> None:
    series = [
        ("Thinking Tokens", thinking, "#7e57c2"),
        ("Tool-Call Output", tool_call, "#ffb74d"),
        ("Prose Output", prose, "#e1bee7"),
    ]
    raw = [np.array(v, dtype=float) for _, v, _ in series]
    log_data = [np.log10(np.clip(v, 1, None)) for v in raw]

    parts = ax.violinplot(
        log_data, showmeans=False, showmedians=False, showextrema=False, widths=0.85
    )
    for pc, (_, _, color) in zip(parts["bodies"], series, strict=False):
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(0.55)

    for i, vals in enumerate(raw, start=1):
        if vals.size == 0:
            continue
        clipped = np.clip(vals, 1, None)
        q1, med, q3 = np.percentile(clipped, [25, 50, 75])
        lo = max(1, np.percentile(clipped, 2.5))
        hi = np.percentile(clipped, 97.5)
        color = series[i - 1][2]
        ax.add_patch(
            plt.Rectangle(
                (i - 0.10, np.log10(q1)),
                0.20,
                np.log10(q3) - np.log10(q1),
                fill=True,
                facecolor=color,
                edgecolor="black",
                linewidth=1,
                alpha=0.9,
                zorder=5,
            )
        )
        ax.hlines(
            np.log10(med), i - 0.10, i + 0.10, colors="black", linewidth=2.0, zorder=6
        )
        ax.vlines(
            i, np.log10(lo), np.log10(q1), colors="black", linewidth=1.0, zorder=4
        )
        ax.vlines(
            i, np.log10(q3), np.log10(hi), colors="black", linewidth=1.0, zorder=4
        )

    vmax = max((v.max() if v.size else 1) for v in raw)
    _log_decade_ticks(ax, vmax)
    ax.set_ylabel("Tokens (log scale)", fontsize=10)
    ax.set_xticks(range(1, len(series) + 1))
    ax.set_xticklabels([s[0] for s in series], fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(
        "OSL Token Classifications  (per assistant turn)",
        fontsize=12,
        fontweight="bold",
    )


# ---------------------------------------------------------------------------
# Composite summary plot
# ---------------------------------------------------------------------------


def plot_summary(
    scalars: dict[str, list[int]],
    cum_per_conv: list[list[int]],
    precomputed: list[int],
    new_isl: list[int],
    osl: list[int],
    osl_thinking: list[int],
    osl_tool_call: list[int],
    osl_prose: list[int],
    out: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(
        3,
        6,
        figure=fig,
        height_ratios=[1.0, 1.0, 1.15],
        hspace=0.45,
        wspace=0.55,
        left=0.05,
        right=0.985,
        top=0.93,
        bottom=0.055,
    )

    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.975)

    # Row 1: Turns / Final-turn ISL / Total OSL  (each spans 2 columns)
    row1_keys = _PANELS[:3]
    for col, (key, t, xl, c) in enumerate(row1_keys):
        ax = fig.add_subplot(gs[0, col * 2 : (col + 1) * 2])
        _draw_hist(ax, scalars.get(key, []), t, xl, c)

    # Row 2: New ISL / OSL per assistant message / ISL growth
    for col, (key, t, xl, c) in enumerate(_PANELS[3:5]):
        ax = fig.add_subplot(gs[1, col * 2 : (col + 1) * 2])
        _draw_hist(ax, scalars.get(key, []), t, xl, c)
    ax_growth = fig.add_subplot(gs[1, 4:6])
    _draw_isl_growth(ax_growth, cum_per_conv)

    # Row 3: two violins (each spans 3 columns)
    ax_v1 = fig.add_subplot(gs[2, 0:3])
    _draw_violin_token_dist(ax_v1, precomputed, new_isl, osl)
    ax_v2 = fig.add_subplot(gs[2, 3:6])
    _draw_violin_osl_categories(ax_v2, osl_thinking, osl_tool_call, osl_prose)

    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def analyze(path: Path, title: str | None = None, out_dir: Path | None = None) -> None:
    out_dir = out_dir or (path.parent / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s", path)
    convs = load_flat(path)
    n_conv = len(convs)
    n_msg = sum(len(v) for v in convs.values())
    log.info("  %d conversations / %d messages", n_conv, n_msg)

    scalars: dict[str, list[int]] = defaultdict(list)
    cum_per_conv: list[list[int]] = []
    precomputed_all: list[int] = []
    new_all: list[int] = []
    osl_all: list[int] = []
    osl_thinking_all: list[int] = []
    osl_tool_call_all: list[int] = []
    osl_prose_all: list[int] = []

    for rows in convs.values():
        s = _conversation_stats(rows)
        if s is None:
            continue
        scalars["turns_per_conversation"].append(s["turns"])
        scalars["final_turn_isl"].append(s["final_turn_isl"])
        scalars["total_osl"].append(s["total_osl"])
        scalars["new_isl_per_turn"].extend(s["new_isl_per_turn"])
        scalars["osl_per_message"].extend(s["osl_per_message"])
        cum_per_conv.append(s["cumulative_isl_per_turn"])
        precomputed_all.extend(s["precomputed_per_turn"])
        new_all.extend(s["new_isl_per_turn"])
        osl_all.extend(s["osl_per_message"])
        osl_thinking_all.extend(s["osl_thinking_per_turn"])
        osl_tool_call_all.extend(s["osl_tool_call_per_turn"])
        osl_prose_all.extend(s["osl_prose_per_turn"])

    n_turns = len(osl_all)
    full_title = title or (
        f"{path.stem} ({len(scalars['turns_per_conversation']):,} "
        f"conversations, {n_turns:,} assistant turns)"
    )
    if title and "conversation" not in title.lower():
        full_title = (
            f"{title} ({len(scalars['turns_per_conversation']):,} "
            f"conversations, {n_turns:,} assistant turns)"
        )

    out_path = out_dir / f"{path.stem}_summary.png"
    plot_summary(
        scalars,
        cum_per_conv,
        precomputed_all,
        new_all,
        osl_all,
        osl_thinking_all,
        osl_tool_call_all,
        osl_prose_all,
        out_path,
        full_title,
    )

    print(f"\n=== {full_title} ===")
    for key, label, _xl, _c in _PANELS:
        p = _percentiles(scalars[key])
        n = len(scalars[key])
        print(
            f"  {label:<32} n={n:>7}  "
            f"p25={p['p25']:>10,.0f}  p50={p['p50']:>10,.0f}  "
            f"p75={p['p75']:>10,.0f}  p95={p['p95']:>10,.0f}"
        )
    for label, vals in [
        ("Precomputed ISL (per turn)", precomputed_all),
        ("OSL (per turn)", osl_all),
        ("  - Thinking", osl_thinking_all),
        ("  - Tool-Call Output", osl_tool_call_all),
        ("  - Prose Output", osl_prose_all),
    ]:
        p = _percentiles(vals)
        print(
            f"  {label:<32} n={len(vals):>7}  "
            f"p25={p['p25']:>10,.0f}  p50={p['p50']:>10,.0f}  "
            f"p75={p['p75']:>10,.0f}  p95={p['p95']:>10,.0f}"
        )
    print(f"\nWrote: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input", type=Path, nargs="+", help="Flat JSONL file(s) to analyze")
    ap.add_argument(
        "--title",
        default=None,
        help="Plot title prefix (conversation/turn counts auto-appended)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dir (default: <input-dir>/analysis)",
    )
    args = ap.parse_args()

    for inp in args.input:
        if not inp.exists():
            log.error("not found: %s", inp)
            sys.exit(1)
        analyze(inp, title=args.title, out_dir=args.output)


if __name__ == "__main__":
    main()
