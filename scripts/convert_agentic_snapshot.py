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

"""Convert agentic snapshot datasets to the flat-row JSONL format expected by MultiTurnDataset.

Each snapshot record contains the full conversation history up to a checkpoint:
    {"conversation_id": "sim_000001", "conversation_idx": 0,
     "messages": [{"role": "system", ...}, ...], "tools": [...], "metadata": {}}

For each conversation only the final snapshot (highest conversation_idx) is used.
Its messages array is expanded into individual flat rows, one per message.

Usage:
    python scripts/convert_agentic_snapshot.py INPUT.jsonl OUTPUT.jsonl
    python scripts/convert_agentic_snapshot.py INPUT.jsonl OUTPUT.jsonl --verify
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers shared between convert() and verify()
# ---------------------------------------------------------------------------


def _load_final_snapshots(input_path: Path) -> dict[str, dict]:
    """Return {conv_id: record} keeping only the highest conversation_idx per conv."""
    final: dict[str, dict] = {}
    with input_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            conv_id = record["conversation_id"]
            if (
                conv_id not in final
                or record["conversation_idx"] > final[conv_id]["conversation_idx"]
            ):
                final[conv_id] = record
    return final


def _apply_collapses(non_system: list[dict]) -> list[tuple[dict, int]]:
    """Apply user-collapse and tool-merge passes, tracking the last source index each
    output row covers.

    Returns list of (output_msg, last_source_idx) pairs where last_source_idx is the
    0-based index within non_system of the final source message folded into this row.
    """
    # Pass 1: collapse consecutive user messages
    collapsed: list[tuple[dict, int]] = []  # (msg, last_source_idx)
    for src_idx, msg in enumerate(non_system):
        if collapsed and collapsed[-1][0]["role"] == "user" and msg["role"] == "user":
            prev_msg, _ = collapsed[-1]
            prev_text = prev_msg.get("content") or ""
            cur_text = msg.get("content") or ""
            collapsed[-1] = (
                {**prev_msg, "content": f"{prev_text}\n\n{cur_text}".strip()},
                src_idx,
            )
        else:
            collapsed.append((msg, src_idx))

    # Pass 2: merge consecutive tool messages
    # Input messages are raw snapshot wire-format (tool_call_id + content on each msg).
    # On merge, upgrade the first message to a tool_results list so the output always
    # uses the tool_results array form regardless of how many results there are.
    merged: list[tuple[dict, int]] = []
    for msg, last_src in collapsed:
        if merged and merged[-1][0]["role"] == "tool" and msg["role"] == "tool":
            prev_msg, _ = merged[-1]
            tool_results = prev_msg.get("tool_results")
            if tool_results is None:
                tool_results = [
                    {
                        "tool_call_id": prev_msg.get("tool_call_id"),
                        "content": prev_msg.get("content"),
                    }
                ]
                prev_msg = {"role": "tool", "tool_results": tool_results}
            tool_results.append(
                {
                    "tool_call_id": msg.get("tool_call_id"),
                    "content": msg.get("content"),
                }
            )
            merged[-1] = (prev_msg, last_src)
        else:
            merged.append((msg, last_src))

    return merged


def _normalize_msg(msg: dict) -> dict:
    """Drop None values for comparison."""
    return {k: v for k, v in msg.items() if v is not None}


def _expand_row_to_wire_msgs(row: dict) -> list[dict]:
    """Expand a single flat row into one or more OpenAI wire-format messages.

    Handles two tool row forms:
    - Output flat rows: tool_results array (always used after conversion)
    - Raw snapshot messages passed through verify(): tool_call_id + content directly
    """
    if isinstance(row.get("tool_results"), list):
        return [
            {
                "role": "tool",
                "tool_call_id": r.get("tool_call_id"),
                "content": r.get("content"),
            }
            for r in row["tool_results"]
        ]
    msg: dict = {"role": row["role"], "content": row.get("content")}
    if row.get("tool_calls"):
        msg["tool_calls"] = row["tool_calls"]
    if row.get("tool_call_id"):
        msg["tool_call_id"] = row["tool_call_id"]
    return [msg]


def verify(input_path: Path, output_path: Path) -> bool:
    """Cross-check every client-turn's pre_built_messages against the source snapshot.

    For each output client turn, reconstruct the pre_built_messages that
    MultiTurnDataset would build from the flat rows and compare it against the
    ground-truth messages built directly from the source snapshot up to the same
    point (accounting for user-collapse and tool-merge).

    Returns:
        True if all checks pass, False if any mismatch found.
    """
    final = _load_final_snapshots(input_path)

    # Load converted rows grouped by conversation_id
    conv_rows: dict[str, list[dict]] = {}
    with output_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cid = row["conversation_id"]
            conv_rows.setdefault(cid, []).append(row)
    for cid in conv_rows:
        conv_rows[cid].sort(key=lambda r: r["turn"])

    errors: list[str] = []
    total_checked = 0

    for conv_id in sorted(final):
        record = final[conv_id]
        system_content: str | None = None
        non_system: list[dict] = []
        for msg in record["messages"]:
            if msg["role"] == "system":
                system_content = msg.get("content")
            else:
                non_system.append(msg)

        # Re-apply the same collapses the converter applies, tracking source coverage
        processed = _apply_collapses(non_system)  # [(output_msg, last_source_idx), ...]
        flat_rows = conv_rows.get(conv_id, [])

        if len(processed) != len(flat_rows):
            errors.append(
                f"{conv_id}: expected {len(processed)} flat rows after collapses, "
                f"got {len(flat_rows)} in output"
            )
            continue

        client_turn_pairs = [
            (out_pos, flat_row)
            for out_pos, (flat_row, _) in enumerate(
                zip(flat_rows, processed, strict=True)
            )
            if flat_row["role"] in ("user", "tool")
        ]

        for ct_idx, (out_pos, flat_row) in enumerate(client_turn_pairs):
            # Ground truth: apply the same collapses the converter applies, then
            # build the message list from the processed (collapsed/merged) rows up to
            # and including this client turn.  This correctly reflects what the
            # converter produces — consecutive user/tool merges mean history is
            # shorter than the raw source but content-equivalent.
            expected: list[dict] = []
            if system_content:
                expected.append({"role": "system", "content": system_content})
            for proc_msg, _ in processed[: out_pos + 1]:
                expected.extend(_expand_row_to_wire_msgs(proc_msg))

            # Reconstructed output: system + expand all flat rows up to this turn
            got: list[dict] = []
            if system_content:
                got.append({"role": "system", "content": system_content})
            for row in flat_rows[: out_pos + 1]:
                got.extend(_expand_row_to_wire_msgs(row))

            exp_norm = [_normalize_msg(m) for m in expected]
            got_norm = [_normalize_msg(m) for m in got]

            if exp_norm != got_norm:
                errors.append(
                    f"{conv_id} client-turn {ct_idx + 1} (flat turn {flat_row['turn']}):\n"
                    f"  expected {len(exp_norm)} msgs, got {len(got_norm)}\n"
                    f"  EXPECTED: {json.dumps(exp_norm, ensure_ascii=False)[:400]}\n"
                    f"  GOT:      {json.dumps(got_norm, ensure_ascii=False)[:400]}"
                )
            total_checked += 1

    if errors:
        print(
            f"FAIL: {len(errors)} mismatches out of {total_checked} client turns checked.",
            file=sys.stderr,
        )
        for err in errors[:20]:
            print(err, file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)
        return False

    print(
        f"OK: all {total_checked} client turns verified against source.",
        file=sys.stderr,
    )
    return True


def convert(input_path: Path, output_path: Path) -> None:
    # Group records by conversation_id, keep only the final snapshot per conversation.
    final: dict[str, dict] = {}
    with input_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            conv_id = record["conversation_id"]
            if (
                conv_id not in final
                or record["conversation_idx"] > final[conv_id]["conversation_idx"]
            ):
                final[conv_id] = record

    print(f"Found {len(final)} conversations in {input_path.name}", file=sys.stderr)

    rows_written = 0
    with output_path.open("w") as out:
        for conv_id, record in sorted(final.items()):
            messages = record["messages"]
            tools = record.get("tools") or []

            # Extract system message (always first if present).
            system_content: str | None = None
            non_system: list[dict] = []
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg.get("content")
                else:
                    non_system.append(msg)

            # Apply the same user-collapse and tool-merge passes used by verify().
            # _apply_collapses returns [(msg, last_source_idx), ...]; strip the indices.
            non_system = [msg for msg, _ in _apply_collapses(non_system)]

            first_user_seen = False
            for position, msg in enumerate(non_system):
                role = msg["role"]
                turn = position + 1  # 1-indexed

                row: dict = {"conversation_id": conv_id, "turn": turn, "role": role}

                # System prompt on the first user row only.
                if role == "user" and not first_user_seen:
                    if system_content is not None:
                        row["system"] = system_content
                    first_user_seen = True

                # tool_calls for assistant messages that dispatch tools.
                if msg.get("tool_calls"):
                    row["tool_calls"] = msg["tool_calls"]

                if role == "tool":
                    # All tool rows use tool_results array (single results have one entry).
                    if msg.get("tool_results"):
                        row["tool_results"] = msg["tool_results"]
                    else:
                        row["tool_results"] = [
                            {
                                "tool_call_id": msg.get("tool_call_id"),
                                "content": msg.get("content"),
                            }
                        ]
                else:
                    # content field (may be None for tool-dispatching assistant messages)
                    row["content"] = msg.get("content")

                # Attach tool definitions to client-turn rows only (user + tool).
                # This avoids duplicating the large tools array on every assistant row
                # while still making them available via load_sample().
                if role in ("user", "tool") and tools:
                    row["tools"] = tools

                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows_written += 1

    print(f"Wrote {rows_written} rows to {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert agentic snapshot JSONL to MultiTurnDataset flat-row JSONL."
    )
    parser.add_argument("input", type=Path, help="Input snapshot JSONL file")
    parser.add_argument("output", type=Path, help="Output flat-row JSONL file")
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "After converting, cross-check every client-turn's pre_built_messages "
            "against the source snapshot. Exits with code 1 if any mismatch found."
        ),
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    convert(args.input, args.output)

    if args.verify:
        ok = verify(args.input, args.output)
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
