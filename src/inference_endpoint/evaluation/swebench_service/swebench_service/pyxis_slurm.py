# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared retry policy for Slurm steps that have not started their command."""

import logging
import random
import time

logger = logging.getLogger(__name__)
SRUN_MAX_ATTEMPTS = 5
_RETRYABLE_PRELAUNCH_ERRORS = (
    "spank_sybil: rpc request error",
    "required plugin spank_sybil.so",
    "failed to connect to any sack sockets",
    "failed to create token",
    "curl: (56) connect tunnel failed",
    "unable to confirm allocation for job",
)


def is_retryable_prelaunch_failure(status: str, output: str) -> bool:
    """Return whether Slurm rejected the step before its command started."""
    if status != "pending":
        return False
    lowered = output.lower()
    return any(marker in lowered for marker in _RETRYABLE_PRELAUNCH_ERRORS)


def wait_for_prelaunch_retry(attempt: int) -> None:
    backoff_s = min(2**attempt, 16)
    delay_s = backoff_s + random.uniform(0.0, backoff_s)
    logger.warning(
        "Retrying Pyxis pre-launch failure in %.1fs (attempt %d/%d)",
        delay_s,
        attempt,
        SRUN_MAX_ATTEMPTS,
    )
    time.sleep(delay_s)
