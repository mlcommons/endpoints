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

"""The one-edit name matcher behind every "did you mean ...?" hint.

`near_match` serves two callers: `RefTable.validate_names`'s declared fact
names (`refs.py`) and `compare._near_namespace_misspelling`'s operand
namespaces (`compare.py`). It is dependency-free so both can import it.
"""

from __future__ import annotations

from collections.abc import Iterable


def near_match(name: str, candidates: Iterable[str]) -> str | None:
    """The candidate one insert/delete/substitute from `name`, else None.

    Example:

        near_match("ofline", ["offline"])  # "offline": one insert
        near_match("oflin", ["offline"])  # None: two edits, matches nothing

    The one matcher behind every name check (ref namespaces and declared facts).

    Args:
        name (str): The name to find the near match for.
        candidates (Iterable[str]): The declared names to search.
    Returns:
        str | None: The candidate one edit from `name`, or None when no candidate is.
    """
    for key in candidates:
        if key == name or abs(len(key) - len(name)) > 1:
            continue
        if len(key) == len(name):
            if sum(a != b for a, b in zip(key, name, strict=True)) == 1:
                return key
            continue
        short, long = sorted((key, name), key=len)
        i = j = edits = 0
        while i < len(short) and j < len(long):
            if short[i] == long[j]:
                i += 1
            else:
                edits += 1
            j += 1
        if edits + (len(long) - j) == 1:
            return key
    return None
