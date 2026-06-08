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

"""Inline accuracy scorer for multi-turn performance replay logs."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import msgspec
import msgspec.json

from inference_endpoint.core.record import EventRecord, EventType, SampleEventType
from inference_endpoint.core.types import TextModelOutput

logger = logging.getLogger(__name__)

EXE_MAP: dict[str, str] = {
    "python": "python",
    "python2": "python",
    "python3": "python",
    "py": "python",
    "pip": "pip",
    "pip3": "pip",
    "pytest": "pytest",
    "pylint": "pylint",
    "sphinx-build": "sphinx",
    "sphinx-quickstart": "sphinx",
    "cython": "cython",
    "make": "make",
    "conda": "conda",
    "cat": "cat",
    "head": "head",
    "tail": "tail",
    "less": "cat",
    "more": "cat",
    "wc": "wc",
    "diff": "diff",
    "grep": "grep",
    "egrep": "grep",
    "fgrep": "grep",
    "rg": "grep",
    "ag": "grep",
    "sed": "sed",
    "awk": "awk",
    "gawk": "awk",
    "tr": "tr",
    "sort": "sort",
    "uniq": "uniq",
    "cut": "cut",
    "find": "find",
    "ls": "ls",
    "locate": "find",
    "xargs": "xargs",
    "cp": "cp",
    "mv": "mv",
    "rm": "rm",
    "mkdir": "mkdir",
    "touch": "touch",
    "tee": "tee",
    "source": "source",
    ".": "source",
    "which": "which",
    "alias": "alias",
    "unset": "unset",
    "export": "export",
    "git": "git",
    "curl": "curl",
    "wget": "curl",
    "true": "true",
    "false": "false",
    "timeout": "timeout",
    "date": "date",
    "apt-get": "apt",
    "apt": "apt",
    "yum": "yum",
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
_INTENT_RE = re.compile(r"\bintent:\s*(I\d{3})\b", re.IGNORECASE)
_BARE_INTENT_RE = re.compile(r"\bI(\d{3})\b")
_WORKFLOW_CONVERSATION_RE = re.compile(r"^sim_\d+$")


def _canonicalize_stage(stage: str) -> str | None:
    """Return the normalized executable for one shell command stage."""
    tokens = stage.split()
    i = 0
    while i < len(tokens) and (_ENVKV_RE.match(tokens[i]) or tokens[i] in _WRAPPERS):
        i += 1
    if i >= len(tokens):
        return None
    match = _PATH_LEAF.search(tokens[i])
    if match is None:
        return None
    leaf = _PYVER_RE.sub("", match.group(0).lower())
    return EXE_MAP.get(leaf)


def _normalized_tool_calls(raw_tool_calls: object) -> list[dict[str, Any]]:
    """Normalize complete or streamed tool-call payloads into one list."""
    if not isinstance(raw_tool_calls, list | tuple) or not raw_tool_calls:
        return []
    if all(isinstance(tc, dict) for tc in raw_tool_calls):
        return [tc for tc in raw_tool_calls if isinstance(tc, dict)]

    merged: dict[int, dict[str, Any]] = {}
    for chunk in raw_tool_calls:
        if not isinstance(chunk, list | tuple):
            continue
        for partial in chunk:
            if not isinstance(partial, dict):
                continue
            idx = int(partial.get("index") or 0)
            tool_call = merged.setdefault(
                idx, {"type": "function", "function": {"arguments": ""}}
            )
            if partial.get("id"):
                tool_call["id"] = partial["id"]
            if partial.get("type"):
                tool_call["type"] = partial["type"]
            fn = partial.get("function") or {}
            if not isinstance(fn, dict):
                continue
            tool_fn = tool_call.setdefault("function", {"arguments": ""})
            if fn.get("name"):
                tool_fn["name"] = fn["name"]
            if fn.get("arguments"):
                tool_fn["arguments"] = tool_fn.get("arguments", "") + fn["arguments"]
    return [merged[i] for i in sorted(merged)]


def extract_exes_from_turn(turn: dict[str, Any]) -> list[str]:
    """Extract normalized bash executables from an assistant turn.

    Example: ``python -m pytest && git status`` becomes ``["python", "git"]``.
    """
    exes: list[str] = []
    for tc in _normalized_tool_calls(turn.get("tool_calls")):
        fn = tc.get("function") or {}
        if not isinstance(fn, dict) or fn.get("name") != "bash":
            continue
        args: object = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if not isinstance(args, dict):
            continue
        cmd = args.get("command") or args.get("cmd") or ""
        if not isinstance(cmd, str) or not cmd:
            continue
        cmd = _HEREDOC_RE.sub(" ", cmd)
        cmd = _QUOTED_RE.sub(" ", cmd)
        for stage in _STAGE_SEP_RE.split(cmd):
            exe = _canonicalize_stage(stage)
            if exe:
                exes.append(exe)
    return exes


def _multiset_iou(a: list[str], b: list[str]) -> float | None:
    """Compute multiset intersection-over-union for executable sequences."""
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    if union == 0:
        return None
    return inter / union


def _extract_intent_code(turn: dict[str, Any]) -> str | None:
    """Extract the final workflow intent code from assistant text fields."""
    for field in ("reasoning_content", "content"):
        text = turn.get(field) or ""
        if not isinstance(text, str):
            continue
        match = _INTENT_RE.search(text)
        if match:
            return match.group(1).upper()
    for field in ("reasoning_content", "content"):
        text = turn.get(field) or ""
        if not isinstance(text, str):
            continue
        matches = list(_BARE_INTENT_RE.finditer(text))
        if matches:
            return f"I{matches[-1].group(1)}"
    return None


def _ground_truth_intent_set(turn: dict[str, Any]) -> set[str] | None:
    """Return workflow ground-truth intent codes from a dataset assistant turn."""
    codes = turn.get("intent_codes")
    if isinstance(codes, list | tuple) and codes:
        out = {code.upper() for code in codes if isinstance(code, str) and code}
        return out or None
    return None


def _domain_for_conversation_id(conversation_id: str) -> str:
    """Classify a turn using only the normalized conversation id.

    Example: ``sim_000001`` is workflow; any other id in this combined dataset
    is treated as coding.
    """
    if _WORKFLOW_CONVERSATION_RE.match(conversation_id):
        return "workflow"
    return "coding"


def _score_turn(gt: dict[str, Any], model: dict[str, Any], domain: str) -> float | None:
    """Score one assistant turn for its already-classified domain."""
    if domain == "workflow":
        gt_intents = _ground_truth_intent_set(gt)
        if gt_intents is None:
            return None
        model_code = _extract_intent_code(model)
        return 1.0 if model_code in gt_intents else 0.0

    if domain != "coding":
        return None
    gt_exes = extract_exes_from_turn(gt)
    if gt_exes:
        return _multiset_iou(gt_exes, extract_exes_from_turn(model))
    return None


def _build_expected_assistants(
    gt_jsonl: Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Map each client turn to the assistant turn that should answer it.

    Example: if user/tool turn 3 is followed by assistant turn 4, the expected
    key is ``(conversation_id, 3)`` and the stored row keeps ``_assistant_turn=4``.
    """
    by_conv: dict[str, list[dict[str, Any]]] = {}
    with gt_jsonl.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            conv_id = row.get("conversation_id")
            if conv_id is not None:
                by_conv.setdefault(str(conv_id), []).append(row)

    expected: dict[tuple[str, int], dict[str, Any]] = {}
    for conv_id, rows in by_conv.items():
        rows = sorted(rows, key=lambda row: int(row.get("turn") or 0))
        for idx, row in enumerate(rows[:-1]):
            if row.get("role") not in ("user", "tool"):
                continue
            next_row = rows[idx + 1]
            if next_row.get("role") == "assistant":
                try:
                    client_turn = int(row.get("turn") or 0)
                    assistant_turn = int(next_row.get("turn") or 0)
                except (TypeError, ValueError):
                    continue
                expected[(conv_id, client_turn)] = {
                    **next_row,
                    "_client_turn": client_turn,
                    "_assistant_turn": assistant_turn,
                }
    return expected


def _split_conversation_instance_id(conversation_id: str) -> tuple[str, int]:
    """Return ``(source_id, repeat_id)`` for a logged conversation id.

    Example: ``abc__repeat_3`` is treated as repeat 3 of source ``abc``;
    an unsuffixed id is repeat 1.
    """
    match = re.search(r"__repeat_(\d+)$", conversation_id)
    if match is None:
        return conversation_id, 1
    return conversation_id[: match.start()], int(match.group(1))


def _text_part(value: object) -> str | None:
    """Convert response message content parts into a single string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, tuple | list):
        return "".join(part for part in value if isinstance(part, str))
    return str(value)


def _model_assistant_from_output(data: object) -> dict[str, Any] | None:
    """Convert a completed TextModelOutput into an assistant-style row."""
    if not isinstance(data, TextModelOutput):
        return None
    content, reasoning, tool_calls = data.as_message_parts()
    return {
        "role": "assistant",
        "content": _text_part(content),
        "reasoning_content": _text_part(reasoning),
        "tool_calls": list(tool_calls) if tool_calls else None,
    }


def _iter_complete_records(events_path: Path):
    """Yield COMPLETE sample events from an event log."""
    decoder = msgspec.json.Decoder(type=EventRecord, dec_hook=EventType.decode_hook)
    with events_path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = decoder.decode(line)
            except msgspec.DecodeError as exc:
                logger.warning(
                    "Skipping malformed event log line %d in %s: %s",
                    line_no,
                    events_path,
                    exc,
                )
                continue
            if record.event_type == SampleEventType.COMPLETE:
                yield record


def _avg(total: float, n: int) -> float | None:
    """Return a rounded average, or None when there are no scored turns."""
    return round(total / n, 4) if n else None


def _per_turn_result(
    key: tuple[str, int, int],
    gt: dict[str, Any],
    model: dict[str, Any] | None,
    domain: str,
    score: float | None,
) -> dict[str, Any]:
    """Build the compact per-turn evidence row for scores.json."""
    row: dict[str, Any] = {
        "conversation_id": key[0],
        "repeat": key[1],
        "turn": gt["_assistant_turn"],
        "domain": domain,
        "score": round(score, 4) if score is not None else None,
    }
    if model is None:
        row["missing"] = True
        model = {"role": "assistant"}

    if domain == "coding":
        row["gt_actions"] = extract_exes_from_turn(gt)
        row["model_actions"] = extract_exes_from_turn(model)
    else:
        gt_intents = _ground_truth_intent_set(gt) or set()
        row["gt_intents"] = sorted(gt_intents)
        row["model_intent"] = _extract_intent_code(model)
    return row


def score_report(
    gt_jsonl: Path,
    report_dir: Path,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Score completed multi-turn replay events and optionally write scores.json."""
    events_path = report_dir / "events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"Events log file not found at {events_path}")

    expected = _build_expected_assistants(gt_jsonl)
    models: dict[tuple[str, int, int], dict[str, Any] | None] = {}
    instances_by_source: dict[str, set[int]] = {}
    for record in _iter_complete_records(events_path):
        if record.turn is None or not record.conversation_id:
            continue
        source_id, instance_id = _split_conversation_instance_id(record.conversation_id)
        expected_key = (source_id, int(record.turn))
        if expected_key not in expected:
            continue
        key = (source_id, instance_id, int(record.turn))
        model = _model_assistant_from_output(record.data)
        if model is not None or key not in models:
            models[key] = model
        instances_by_source.setdefault(source_id, set()).add(instance_id)

    expected_sources = {conv_id for conv_id, _turn in expected}
    observed_instances: set[int] = set()
    for source_id in expected_sources:
        observed_instances.update(instances_by_source.get(source_id, set()))
    instances_for_scoring = sorted(observed_instances or {1})

    total_score = 0.0
    n_scored = 0
    domain_totals = {"coding": 0.0, "workflow": 0.0}
    domain_counts = {"coding": 0, "workflow": 0}
    per_turn: list[dict[str, Any]] = []
    for instance_id in instances_for_scoring:
        for expected_key, gt in sorted(expected.items()):
            source_id, turn = expected_key
            key = (source_id, instance_id, turn)
            domain = _domain_for_conversation_id(source_id)
            model = models.get(key)
            score = _score_turn(gt, model, domain) if model is not None else None
            per_turn.append(_per_turn_result(key, gt, model, domain, score))

            if score is None:
                continue
            n_scored += 1
            total_score += score
            domain_counts[domain] += 1
            domain_totals[domain] += score

    observed_outputs = {key for key, model in models.items() if model is not None}
    expected_total = len(expected) * len(instances_for_scoring)
    expected_output_keys = {
        (source_id, instance_id, turn)
        for source_id, turn in expected
        for instance_id in instances_for_scoring
    }
    missing_outputs = len(expected_output_keys - observed_outputs)
    complete_repeat_sets = bool(observed_instances) and all(
        instances_by_source.get(source_id, set()) == observed_instances
        for source_id in expected_sources
    )
    valid = (
        len(expected) > 0
        and complete_repeat_sets
        and missing_outputs == 0
        and bool(observed_instances)
    )
    result: dict[str, Any] = {
        "score": _avg(total_score, n_scored),
        "valid": valid,
        "turns": {
            "expected_per_repeat": len(expected),
            "repeats": len(observed_instances),
            "expected": expected_total,
            "observed": len(observed_outputs),
            "missing": missing_outputs,
            "scored": n_scored,
        },
        "domains": {
            domain: {
                "score": _avg(domain_totals[domain], domain_counts[domain]),
                "scored": domain_counts[domain],
            }
            for domain in ("coding", "workflow")
            if domain_counts[domain]
        },
        "per_turn": per_turn,
    }
    if not valid:
        reasons: list[str] = []
        if len(expected) == 0:
            reasons.append("no expected assistant turns found")
        elif not observed_instances:
            reasons.append("no matching completed turns found")
        elif not complete_repeat_sets:
            reasons.append("observed turns do not cover whole dataset repeat(s)")
        if missing_outputs:
            reasons.append(
                f"{missing_outputs}/{expected_total} expected turn(s) missing output"
            )
        result["invalid_reason"] = "; ".join(reasons)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
    return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse standalone scorer arguments."""
    parser = argparse.ArgumentParser(description="Score a multi-turn benchmark run.")
    parser.add_argument(
        "--gt", required=True, type=Path, help="Ground-truth JSONL file"
    )
    parser.add_argument(
        "--report-dir",
        required=True,
        type=Path,
        help="Benchmark report directory containing events.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output scores.json path; defaults to <report-dir>/scores.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the scorer as a standalone script."""
    args = _parse_args(argv)
    result = score_report(
        gt_jsonl=args.gt,
        report_dir=args.report_dir,
        out_path=args.out or args.report_dir / "scores.json",
    )
    print(json.dumps({k: v for k, v in result.items() if k != "per_turn"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
