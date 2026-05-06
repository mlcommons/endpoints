# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inline accuracy scorer for multi-turn benchmark runs.

Per docs/EVALUATION.md:
    Coding turn:   IoU of the multisets of canonical bash exes used in the turn
    Workflow turn: 1 if `intent: IXXX` code matches gt, else 0
    pass_rate    = mean score across scorable turns

Usage:
    score.py --gt <gt.jsonl> --domain {coding,workflow} \\
             [--report-dir <dir> | --model <jsonl>] \\
             [--dataset-name <key>]                  \\
             [--out <scores.json>]

In ``--report-dir`` mode the script reads ``events.jsonl`` and
``sample_idx_map.json`` from the benchmark run, derives the model's
assistant turns (mirroring ``MultiTurnDataset._build_metadata``'s
client-turn ordering), and scores them against the gt jsonl.

In ``--model`` mode the model assistant turns are read from a pre-built
flat JSONL — same shape as the gt assistant rows.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("inline_accuracy")

# ---------------------------------------------------------------------------
# Coding scorer — extract whitelisted exes per turn, score = multiset IoU
# ---------------------------------------------------------------------------

# Canonical exe whitelist. Synonyms collapse to one canonical name.
# Anything not in this map is dropped. cd / pwd / echo are intentionally
# absent — they're navigation/output verbs, not action classes.
EXE_MAP: dict[str, str] = {
    # python ecosystem
    "python": "python", "python2": "python", "python3": "python", "py": "python",
    "pip": "pip", "pip3": "pip",
    "pytest": "pytest", "pylint": "pylint",
    "sphinx-build": "sphinx", "sphinx-quickstart": "sphinx",
    "cython": "cython", "make": "make", "conda": "conda",
    # text view / search / transform
    "cat": "cat", "head": "head", "tail": "tail", "less": "cat", "more": "cat",
    "wc": "wc", "diff": "diff",
    "grep": "grep", "egrep": "grep", "fgrep": "grep", "rg": "grep", "ag": "grep",
    "sed": "sed", "awk": "awk", "gawk": "awk", "tr": "tr",
    "sort": "sort", "uniq": "uniq", "cut": "cut",
    # find / list
    "find": "find", "ls": "ls", "locate": "find", "xargs": "xargs",
    # filesystem ops
    "cp": "cp", "mv": "mv", "rm": "rm", "mkdir": "mkdir", "touch": "touch", "tee": "tee",
    # shell
    "source": "source", ".": "source",
    "which": "which", "alias": "alias", "unset": "unset", "export": "export",
    # vcs / fetch
    "git": "git", "curl": "curl", "wget": "curl",
    # misc
    "true": "true", "false": "false", "timeout": "timeout", "date": "date",
    "apt-get": "apt", "apt": "apt", "yum": "yum",
}
_WRAPPERS = {"env", "time", "nice", "sudo", "exec", "command"}

_HEREDOC_RE = re.compile(
    r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?[\s\S]*?\n\1\s*$",
    re.MULTILINE,
)
_QUOTED_RE = re.compile(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"|`[^`]*`")
_STAGE_SEP_RE = re.compile(r"\|\||\||&&|;|\n")
_ENVKV_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_PATH_LEAF = re.compile(r"[^/]+$")
_PYVER_RE = re.compile(r"\.\d+(\.\d+)?$")


def _canonicalize_stage(stage: str) -> str | None:
    """Return the whitelisted canonical exe of a single pipeline stage, or None."""
    tokens = stage.split()
    i = 0
    while i < len(tokens) and (_ENVKV_RE.match(tokens[i]) or tokens[i] in _WRAPPERS):
        i += 1
    if i >= len(tokens):
        return None
    leaf = _PATH_LEAF.search(tokens[i]).group(0).lower()
    leaf = _PYVER_RE.sub("", leaf)
    return EXE_MAP.get(leaf)


def extract_exes_from_turn(turn: dict) -> list[str]:
    """Flat list of canonical exes used across all bash tool_calls in the turn.

    Per-tool-call boundaries are erased (the result is a single multiset).
    Heredoc bodies and quoted strings are stripped before stage-splitting so
    embedded code/text can't leak in as fake stages.
    """
    exes: list[str] = []
    for tc in turn.get("tool_calls") or []:
        fn = tc.get("function") or {}
        if fn.get("name") != "bash":
            continue
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if not isinstance(args, dict):
            continue
        cmd = args.get("command") or args.get("cmd") or ""
        if not cmd:
            continue
        cmd = _HEREDOC_RE.sub(" ", cmd)
        cmd = _QUOTED_RE.sub(" ", cmd)
        for stage in _STAGE_SEP_RE.split(cmd):
            exe = _canonicalize_stage(stage)
            if exe:
                exes.append(exe)
    return exes


def _multiset_iou(a: list[str], b: list[str]) -> float | None:
    """Multiset IoU: |intersection| / |union| using min/max counts.

    Returns None iff both multisets are empty.
    """
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    if union == 0:
        return None
    return inter / union


def score_coding(gt: dict, model: dict) -> float | None:
    """IoU of the gt and model exe multisets in [0, 1]; None if gt has no exes."""
    g = extract_exes_from_turn(gt)
    if not g:
        return None
    return _multiset_iou(g, extract_exes_from_turn(model))


# ---------------------------------------------------------------------------
# Workflow scorer — extract `intent: IXXX` code, exact match
# ---------------------------------------------------------------------------

_INTENT_RE = re.compile(r"\bintent:\s*(I\d{3})\b", re.IGNORECASE)
_BARE_INTENT_RE = re.compile(r"\bI(\d{3})\b")


def _extract_intent_code(turn: dict) -> str | None:
    """Two-stage extractor:

    1. First match of `intent: IXXX` (case-insensitive) in `reasoning_content`
       then `content`. Unambiguous when present.
    2. Fallback: last match of bare `\\bI(\\d{3})\\b` in `reasoning_content`
       then `content`. Catches inline forms like `I057 (Pickup Delay)` or
       `is now I058 (Marked Picked Up But Not Collected)`. "Last" because
       reasoning often references prior intents earlier and concludes with
       the final classification.
    """
    for field in ("reasoning_content", "content"):
        text = turn.get(field) or ""
        m = _INTENT_RE.search(text)
        if m:
            return m.group(1).upper()
    for field in ("reasoning_content", "content"):
        text = turn.get(field) or ""
        ms = list(_BARE_INTENT_RE.finditer(text))
        if ms:
            return f"I{ms[-1].group(1)}"
    return None


def score_workflow(gt: dict, model: dict) -> float | None:
    """1.0 if intent codes match, 0.0 if not, None if gt has no intent code."""
    gt_code = _extract_intent_code(gt)
    if gt_code is None:
        return None
    return float(gt_code == _extract_intent_code(model))


# ---------------------------------------------------------------------------
# Run scoring — pair assistant turns by (conv_id, turn) and aggregate
# ---------------------------------------------------------------------------


def _iter_assistant_turns(jsonl_path: Path) -> Iterable[dict]:
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("role") == "assistant":
                yield d


def _key(d: dict) -> tuple[str, int]:
    return (d.get("conversation_id") or d.get("conv") or "", int(d.get("turn") or 0))


def score_run(gt_jsonl: Path, model_jsonl: Path, domain: str) -> dict:
    """Score a captured run: pair gt and model assistant turns by (conv_id, turn).

    domain: "coding" | "workflow". Returns:
        pass_rate         — mean score across scorable turns (∈ [0, 1])
        n_scorable        — turns where gt was scorable
        n_perfect         — turns scoring exactly 1.0
        n_zero            — turns scoring exactly 0.0
        skipped_unscorable_gt — gt had no intent (workflow) or no exes (coding)
        missing_in_model  — gt scorable but model has no matching turn
        per_turn          — list of {conv, turn, score}
    """
    if domain not in ("coding", "workflow"):
        raise ValueError("domain must be 'coding' or 'workflow'")
    scorer = score_coding if domain == "coding" else score_workflow

    gt_by_key = {_key(t): t for t in _iter_assistant_turns(gt_jsonl)}
    model_by_key = {_key(t): t for t in _iter_assistant_turns(model_jsonl)}

    results: list[dict] = []
    n_perfect = n_zero = unscorable = missing = 0
    total = 0.0
    for k, gt in gt_by_key.items():
        md = model_by_key.get(k)
        if md is None:
            if scorer(gt, {"role": "assistant"}) is None:
                unscorable += 1
            else:
                missing += 1
            continue
        s = scorer(gt, md)
        if s is None:
            unscorable += 1
            continue
        results.append({"conv": k[0], "turn": k[1], "score": round(s, 4)})
        total += s
        if s == 1.0:
            n_perfect += 1
        elif s == 0.0:
            n_zero += 1
    n = len(results)
    return {
        "domain": domain,
        "n_scorable": n,
        "n_perfect": n_perfect,
        "n_zero": n_zero,
        "pass_rate": round(total / n, 4) if n else float("nan"),
        "skipped_unscorable_gt": unscorable,
        "missing_in_model": missing,
        "per_turn": results,
    }


# ---------------------------------------------------------------------------
# Events → model assistants (folded in from the old events_to_jsonl.py)
# ---------------------------------------------------------------------------


def _build_index_to_key(gt_jsonl: Path) -> list[tuple[str, int]]:
    keys: list[tuple[str, int]] = []
    by_conv: dict[str, list[dict]] = {}
    for line in gt_jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "conversation_id" not in d:
            continue
        by_conv.setdefault(str(d["conversation_id"]), []).append(d)
    for conv_id, rows in by_conv.items():
        rows = sorted(rows, key=lambda r: int(r["turn"]))
        for r in rows:
            if r.get("role") in ("user", "tool"):
                keys.append((conv_id, int(r["turn"])))
    return keys


def _parse_model_output(data: object) -> dict | None:
    if not isinstance(data, dict):
        return None
    if data.get("type") not in (None, "TextModelOutput"):
        return None
    out: dict[str, object] = {}
    for src, dst in (("output", "content"), ("reasoning", "reasoning_content")):
        v = data.get(src)
        if v is None:
            out[dst] = None
        elif isinstance(v, str):
            out[dst] = v
        elif isinstance(v, list):
            out[dst] = "".join(p for p in v if isinstance(p, str))
        else:
            out[dst] = str(v)
    tcs = data.get("tool_calls")
    out["tool_calls"] = list(tcs) if isinstance(tcs, list) else None
    return out


def derive_model_assistants(
    report_dir: Path,
    gt_jsonl: Path,
    out_path: Path,
    dataset_name: str | None = None,
) -> int:
    events_path = report_dir / "events.jsonl"
    map_path = report_dir / "sample_idx_map.json"
    if not events_path.exists():
        raise SystemExit(f"missing {events_path}")
    if not map_path.exists():
        raise SystemExit(f"missing {map_path}")

    full_map = json.loads(map_path.read_text())
    if dataset_name is None:
        if len(full_map) != 1:
            raise SystemExit(
                f"sample_idx_map.json has {len(full_map)} dataset keys "
                f"({sorted(full_map.keys())}); pass --dataset-name to disambiguate"
            )
        dataset_name = next(iter(full_map))
    if dataset_name not in full_map:
        raise SystemExit(
            f"dataset '{dataset_name}' not in sample_idx_map.json "
            f"(have {sorted(full_map.keys())})"
        )
    uuid_to_index = {k: int(v) for k, v in full_map[dataset_name].items()}

    index_to_key = _build_index_to_key(gt_jsonl)
    logger.info("loaded %d client-turn keys from gt", len(index_to_key))

    uuid_to_key: dict[str, tuple[str, int]] = {}
    for uuid, idx in uuid_to_index.items():
        if 0 <= idx < len(index_to_key):
            uuid_to_key[uuid] = index_to_key[idx]
        else:
            logger.warning(
                "uuid %s -> index %d out of range (max %d)",
                uuid, idx, len(index_to_key) - 1,
            )

    rows_written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open() as f, out_path.open("w") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = ev.get("event_type") or ev.get("event") or ""
            if "complete" not in str(etype).lower():
                continue
            uuid = ev.get("sample_uuid") or ""
            if not uuid:
                continue
            key = uuid_to_key.get(uuid)
            if key is None:
                continue
            data = _parse_model_output(ev.get("data"))
            if data is None:
                continue
            conv_id, client_turn = key
            row = {
                "conversation_id": conv_id,
                "turn": client_turn + 1,
                "role": "assistant",
                "content": data.get("content"),
                "reasoning_content": data.get("reasoning_content"),
                "tool_calls": data.get("tool_calls"),
            }
            out.write(json.dumps(row) + "\n")
            rows_written += 1

    logger.info("wrote %d assistant rows to %s", rows_written, out_path)
    return rows_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Score a multi-turn benchmark run.")
    p.add_argument("--gt", required=True, type=Path)
    p.add_argument("--domain", required=True, choices=("coding", "workflow"))
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--report-dir", type=Path,
                     help="Benchmark report dir — derives model assistants from events.jsonl")
    src.add_argument("--model", type=Path, help="Pre-built model assistants JSONL")
    p.add_argument("--dataset-name", default=None,
                   help="Key in sample_idx_map.json (only needed if multiple datasets in one run)")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    if args.report_dir is not None:
        model_path = args.report_dir / "model_assistants.jsonl"
        derive_model_assistants(args.report_dir, args.gt, model_path, args.dataset_name)
    else:
        model_path = args.model

    out_path = args.out or (
        (args.report_dir / "scores.json") if args.report_dir else (model_path.parent / "scores.json")
    )

    result = score_run(args.gt, model_path, args.domain)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("wrote %s", out_path)
    print(json.dumps({k: v for k, v in result.items() if k != "per_turn"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
