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

"""Cache-bursting salt for multi-turn replay.

A short hash derived from ``conversation_id`` is appended to the end of
each trajectory's system message so the engine's prefix cache cannot
extend past the system boundary across trajectories. See
``examples/09_MultiTurn/docs/CACHE_BUSTING.md`` for the methodology and
Phase A measurements.
"""

from __future__ import annotations

import hashlib

DIGEST_BYTES = 8  # 16 hex chars / 64 bits — collision-resistant on the
# benchmark's trajectory count (low thousands), not crypto.

SALT_MARKER_PREFIX = "[cache_salt: "
SALT_MARKER_SUFFIX = "]"


def compute_salt(conversation_id: str) -> str:
    """Return the hex salt for one trajectory.

    Same ``conversation_id`` always yields the same salt — the salt
    varies across trajectories but is stable across re-runs.
    """
    return hashlib.blake2b(
        str(conversation_id).encode("utf-8"), digest_size=DIGEST_BYTES
    ).hexdigest()


def apply_salt(system_text: str, salt_hex: str) -> str:
    """Append the salt marker at the end of the system message content.

    Format::

        <system_text>\\n\\n[cache_salt: <hex>]

    The leading paragraph break gives the chat template a clean
    tokenization boundary so the salt tokens don't merge with the
    preceding system content.
    """
    return f"{system_text}\n\n{SALT_MARKER_PREFIX}{salt_hex}{SALT_MARKER_SUFFIX}"
