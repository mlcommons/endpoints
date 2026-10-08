# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate classified submission artifacts against a selected cohort policy."""

from dataclasses import dataclass
from os import PathLike

from .engine import execute_submission
from .loader import bundled_policy_path, load_policy
from .models import Policy
from .results import CheckResult, Report

__all__ = ["CheckResult", "Report", "SubmissionChecker", "validate_submission"]


@dataclass(frozen=True)
class SubmissionChecker:
    """Reusable checker configuration; each run parses and evaluates fresh evidence."""

    submission_path: PathLike[str]
    policy: Policy | PathLike[str] | None = None

    def run(self) -> Report:
        policy = (
            self.policy
            if isinstance(self.policy, Policy)
            else load_policy(self.policy or bundled_policy_path())
        )
        return execute_submission(
            self.submission_path,
            policy,
        )


def validate_submission(
    path: PathLike[str],
    *,
    policy: Policy | PathLike[str] | None = None,
) -> Report:
    """Load a cohort policy, classify artifacts, and evaluate its applicable checks."""
    return SubmissionChecker(
        submission_path=path,
        policy=policy,
    ).run()
