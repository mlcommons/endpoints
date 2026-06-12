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

"""Truncate a benchmark ``results.json``.

Perf+accuracy runs store every query's full response text under
``responses``, which can reach gigabytes. ``truncate-results`` keeps the
first ``keep_n`` responses verbatim and replaces the rest with a per-sample
content hash, so the file stays small while still proving which outputs were
produced (proof of work).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Annotated, Any

import cyclopts
from pydantic import BaseModel, ConfigDict, Field

from inference_endpoint.exceptions import InputValidationError

logger = logging.getLogger(__name__)

_HASH_ALGORITHM = "sha256"


def truncate_results_dict(results: dict[str, Any], keep_n: int = 5) -> dict[str, Any]:
    """Return a truncated copy of a ``results.json`` dict.

    Keeps ``config``/``results``/``accuracy_scores``/``errors`` verbatim,
    keeps the first ``keep_n`` ``responses`` full, and adds a ``truncation``
    block holding a ``sha256`` hash of every response plus counts. A dict
    without a non-empty ``responses`` section (e.g. a perf-only run) is
    returned unchanged.
    """
    responses = results.get("responses")
    if not responses:
        return dict(results)

    uuids = list(responses.keys())
    kept = uuids[:keep_n]

    out = dict(results)
    out["responses"] = {uuid: responses[uuid] for uuid in kept}
    out["truncation"] = {
        "responses_truncated": True,
        "hash_algorithm": _HASH_ALGORITHM,
        "n_responses_total": len(uuids),
        "n_responses_kept": len(kept),
        "response_hashes": {
            uuid: hashlib.sha256(str(text).encode("utf-8")).hexdigest()
            for uuid, text in responses.items()
        },
    }
    return out


@cyclopts.Parameter(name="*")
class TruncateConfig(BaseModel):
    """truncate-results command config."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    results: Path
    keep_n: Annotated[
        int,
        cyclopts.Parameter(
            alias="--keep-n", help="Number of full responses to keep verbatim"
        ),
    ] = Field(5, ge=0)
    output: Annotated[
        Path | None,
        cyclopts.Parameter(
            alias="--output", help="Output path (default: *.truncated.json)"
        ),
    ] = None
    in_place: Annotated[
        bool,
        cyclopts.Parameter(alias="--in-place", help="Overwrite the input file"),
    ] = False


def execute_truncate(config: TruncateConfig) -> None:
    """Read ``config.results``, truncate it, and write the result."""
    if not config.results.exists():
        raise InputValidationError(f"Results file not found: {config.results}")

    data = json.loads(config.results.read_text())
    truncated = truncate_results_dict(data, keep_n=config.keep_n)

    if config.in_place:
        out_path = config.results
    elif config.output is not None:
        out_path = config.output
    else:
        out_path = config.results.with_name(config.results.stem + ".truncated.json")

    out_path.write_text(json.dumps(truncated, indent=2))

    meta = truncated.get("truncation")
    if meta is None:
        logger.info("No responses to truncate; wrote passthrough copy to %s", out_path)
    else:
        logger.info(
            "Truncated %d responses to %d full + %d hashes; wrote %s",
            meta["n_responses_total"],
            meta["n_responses_kept"],
            meta["n_responses_total"],
            out_path,
        )
