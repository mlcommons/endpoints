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

"""Smoke-check that the ``.[bfcl]`` extra installs and imports.

``bfcl-eval`` hard-pins an old dependency set and is resolved in its own uv fork
(see ``[tool.uv].conflicts`` in ``pyproject.toml``), so the regular CI ``test``
job installs *without* it. That leaves the repo's most brittle dependency with no
CI coverage: a yanked/repinned ``bfcl-eval``, a broken ``soundfile`` import
chain, or a fork that stops resolving would all pass CI silently while
``pip install -e ".[bfcl]"`` is broken for users.

This script is the cheap guard for that gap. It does NOT run a benchmark — it
only verifies the import/resolution surface our BFCL scorer and dataset depend
on. Run it in an env with the ``bfcl`` extra installed:

    uv sync --frozen --extra bfcl
    uv run --frozen --extra bfcl python scripts/bfcl_import_smoke.py

Exits 0 if everything imports, 1 otherwise.
"""

import sys
from pathlib import Path

import bfcl_eval
from inference_endpoint.evaluation import bfcl_v4_execution, bfcl_v4_scorer
from inference_endpoint.evaluation.scoring import Scorer


def main() -> int:
    print(f"bfcl-eval {getattr(bfcl_eval, '__version__', '<unknown>')} imported")

    # The scorer guards `bfcl_eval` behind try/except → None on ImportError.
    # With the extra installed these must resolve to the real objects, otherwise
    # scoring silently degrades.
    if bfcl_v4_scorer.Language is None or bfcl_v4_scorer.ast_checker is None:
        print(
            "ERROR: bfcl_v4_scorer.Language/ast_checker are None — bfcl-eval "
            "import failed inside the scorer despite the extra being installed.",
            file=sys.stderr,
        )
        return 1

    # The execution bridge imports a different bfcl-eval submodule (also guarded
    # behind try/except → None), so check it resolved too — catches a partial
    # bfcl-eval breakage that leaves the ast_checker path working.
    if bfcl_v4_execution.execute_multi_turn_func_call is None:
        print(
            "ERROR: bfcl_v4_execution bfcl-eval imports are None — the "
            "multi_turn_eval/model_handler import chain is broken.",
            file=sys.stderr,
        )
        return 1

    # The dataset resolves its data from the installed bfcl_eval package dir.
    bfcl_data_dir = Path(bfcl_eval.__file__).parent / "data"
    if not bfcl_data_dir.is_dir():
        print(
            f"ERROR: bfcl-eval data directory not found at {bfcl_data_dir}",
            file=sys.stderr,
        )
        return 1

    # The scorer/dataset are registered via `scoring` import side effects.
    Scorer.get("bfcl_v4")

    print("bfcl import smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
