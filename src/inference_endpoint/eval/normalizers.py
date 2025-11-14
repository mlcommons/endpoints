# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Answer normalization utilities for consistent comparison."""

import re


def normalize_number(text: str) -> str | None:
    """Normalize numeric answer for comparison.

    Based on OpenAI's normalize_number from AIME eval.
    Reference: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/aime_eval.py

    Extracts leading digits from the string. Returns None if no digits found.

    Args:
        text: Numeric string to normalize

    Returns:
        Normalized numeric string (leading digits only), or None if no digits found

    Examples:
        >>> normalize_number("42")
        '42'
        >>> normalize_number("123abc")
        '123'
        >>> normalize_number("abc")
        None
    """
    if not text:
        return None

    # Match digits from the start (following OpenAI's pattern)
    match = re.match(r"\d+", str(text))
    if not match:
        return None

    return match.group(0)


def normalize_multiple_choice(text: str) -> str:
    """Normalize ABCD multiple choice answer to uppercase single letter.

    Simple normalization: strips whitespace/punctuation and uppercases.

    Args:
        text: Multiple choice answer to normalize

    Returns:
        Uppercase single letter (A/B/C/D), or original text if invalid
    """
    if not text:
        return text

    # Strip whitespace and common punctuation
    text = text.strip().strip("()[]").strip(".")

    # Convert to uppercase
    text = text.upper()

    return text
