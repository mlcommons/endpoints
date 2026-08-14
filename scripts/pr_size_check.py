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

"""Classify a pull request by non-test churn per the Endpoints PR Review Policy.

Measures additions + deletions and the changed-file count of the current branch
against its merge-base with the base branch, excluding tests, lockfiles,
vendored sources, and generated files, then reports the size class:

    Normal      <= 500 non-test lines AND <= 20 non-test files
    Large       501-1500 lines OR 21-50 files
    Very large  > 1500 lines OR > 50 files

The class is the higher of the two dimensions, evaluated independently.

The check is advisory and always exits 0. By default it prints a one-line
verdict; ``-v/--verbose`` adds the threshold grid and per-class review
requirements. With ``--github-output`` it also writes
``size_class``/``churn``/``files`` to ``$GITHUB_OUTPUT`` for the PR-labeling
workflow.
"""

from __future__ import annotations

import argparse
import os
import subprocess

# Paths excluded from the size gate. Tests stay part of review but do not count
# toward size so thorough testing is not discouraged; lockfiles, vendored
# sources, and generated files are churn a reviewer does not read line by line.
EXCLUDE_PATHSPECS = (
    ":(exclude)tests/**",
    ":(exclude)*.lock",
    ":(exclude)vendor/**",
    ":(exclude)src/inference_endpoint/openai/openai_types_gen.py",
    ":(exclude)src/inference_endpoint/openai/openapi.yaml",
    ":(exclude)src/inference_endpoint/config/templates/**",
    ":(exclude)src/inference_endpoint/evaluation/legacy_mlperf_deepseek_r1/mlperf_eval/**",
)

# Base refs tried in order when neither PR_BASE_REF nor GITHUB_BASE_REF is set.
# Covers the common local-checkout and fork-remote layouts.
_BASE_CANDIDATES = ("origin/main", "main", "upstream/main")

NORMAL, LARGE, VERY_LARGE = "normal", "large", "very-large"

_REQUIREMENTS = {
    NORMAL: "2 approvals (>=1 from @mlcommons/endpoints-developers).",
    LARGE: 'add a "How to Review" section and split the PR when practical (2 approvals).',
    VERY_LARGE: 'add "Why This Cannot Be Split" and "How to Review" sections (3 approvals).',
}

_GRID = "Normal <=500 & <=20 files | Large 501-1500 or 21-50 | Very large >1500 or >50"


def classify(churn: int, files: int) -> str:
    """Return the size class from non-test churn and file count.

    Thresholds are evaluated independently and the higher class wins.
    """
    if churn > 1500 or files > 50:
        return VERY_LARGE
    if churn > 500 or files > 20:
        return LARGE
    return NORMAL


def parse_numstat(text: str) -> tuple[int, int]:
    """Sum added+deleted lines and count files from ``git diff --numstat`` output.

    Binary files render as ``-\t-\tpath``; they count as a changed file but add
    zero lines.
    """
    churn = 0
    files = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        added, deleted, *_ = line.split("\t")
        files += 1
        if added != "-" and deleted != "-":
            churn += int(added) + int(deleted)
    return churn, files


def _git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False)


def _rev_parse_ok(ref: str) -> bool:
    return (
        _git(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"]).returncode == 0
    )


def resolve_base_ref() -> str | None:
    """Pick the base ref to diff against, or None if none can be resolved.

    Precedence: PR_BASE_REF -> GITHUB_BASE_REF -> origin/main -> main ->
    upstream/main -> any ``<remote>/main``.
    """
    env = os.environ.get("PR_BASE_REF")
    if env:
        return env if _rev_parse_ok(env) else None

    candidates: list[str] = []
    gh_base = os.environ.get("GITHUB_BASE_REF")
    if gh_base:
        candidates += [f"origin/{gh_base}", gh_base]
    candidates += list(_BASE_CANDIDATES)
    for ref in candidates:
        if _rev_parse_ok(ref):
            return ref

    for remote in _git(["remote"]).stdout.split():
        ref = f"{remote}/main"
        if _rev_parse_ok(ref):
            return ref
    return None


def measure(base_ref: str, head_ref: str | None) -> tuple[int, int, str] | None:
    """Return (churn, files, merge_base_short), or None if the diff cannot run.

    If ``head_ref`` is None, this diffs the merge-base against the working tree
    (the pre-commit case: the working tree holds the staged changes). With
    ``head_ref`` set it diffs merge-base..head_ref (the CI case, where the
    checkout is the base repo and the PR head is fetched separately).
    """
    head = head_ref or "HEAD"
    merge_base = _git(["merge-base", base_ref, head])
    if merge_base.returncode != 0:
        return None
    base_sha = merge_base.stdout.strip()

    diff_args = ["diff", "--numstat", base_sha]
    if head_ref:
        diff_args.append(head_ref)
    diff_args += ["--", ".", *EXCLUDE_PATHSPECS]
    diff = _git(diff_args)
    if diff.returncode != 0:
        return None

    churn, files = parse_numstat(diff.stdout)
    return churn, files, base_sha[:9]


def build_summary(
    cls: str, churn: int, files: int, base_ref: str, base_short: str
) -> str:
    """One-line size verdict (default output)."""
    return (
        f"PR size: {cls.replace('-', ' ').upper()} "
        f"(non-test churn: {churn} lines, {files} files; "
        f"base: {base_ref} @ {base_short})"
    )


def build_details(cls: str) -> str:
    """Threshold grid and per-class review requirements (verbose output)."""
    return "\n".join(
        [
            f"  {_GRID}",
            f"  -> {_REQUIREMENTS[cls]}",
            "  High-risk PRs need 3 approvals regardless of size (author-declared).",
            "  Excluded from size: tests/**, *.lock, vendor/**, generated files. "
            "Advisory only.",
        ]
    )


def _write_github_output(cls: str, churn: int, files: int) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"size_class={cls}\nchurn={churn}\nfiles={files}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PR non-test-churn size class.")
    parser.add_argument(
        "--base", default=None, help="Base ref (overrides PR_BASE_REF/auto-detect)."
    )
    parser.add_argument(
        "--head",
        default=None,
        help="Head ref to measure (default: working tree / HEAD).",
    )
    parser.add_argument(
        "--github-output",
        action="store_true",
        help="Also write size_class/churn/files to $GITHUB_OUTPUT.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print the threshold grid and per-class review requirements.",
    )
    args = parser.parse_args(argv)

    if not _rev_parse_ok("HEAD"):
        print("pr-size-check: not a git repository with commits; skipping.")
        return 0

    base_ref = args.base or resolve_base_ref()
    if base_ref is None:
        print(
            "pr-size-check: could not resolve a base branch "
            "(tried PR_BASE_REF, GITHUB_BASE_REF, origin/main, main); skipping."
        )
        return 0

    result = measure(base_ref, args.head)
    if result is None:
        print(
            f"pr-size-check: could not diff against {base_ref} "
            "(shallow clone or unrelated history?); skipping."
        )
        return 0

    churn, files, base_short = result
    cls = classify(churn, files)
    print(build_summary(cls, churn, files, base_ref, base_short))
    if args.verbose:
        print(build_details(cls))
    if args.github_output:
        _write_github_output(cls, churn, files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
