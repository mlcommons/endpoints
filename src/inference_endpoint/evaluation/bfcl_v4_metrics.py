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

"""BFCL v4 compliance-gate metric mapping.

Maps each ruleset golden-metric name (defined on the model in
``config/rulesets/mlcommons/models.py``) to the key the BFCL scorer writes into
its breakdown block (``bfcl_v4_scorer.py``). The accuracy gate
(``compliance/checker.py``) and the accuracy plots (``metrics/results_plots.py``)
read a scorer's breakdown by this mapping.

Kept here — beside the other ``bfcl_v4_*`` modules but dependency-light (no
``bfcl-eval`` import) — so both consumers import it without pulling in the scorer
or the optional ``bfcl`` extra. Both halves are BFCL-specific: the ``bfcl_*``
golden names and the ``normalized_single_turn_score`` breakdown key have no
meaning for the other scorers.
"""

# Ruleset golden-metric name -> key in the BFCL scorer's breakdown block.
ACCURACY_METRIC_KEYS = {
    "bfcl_overall_accuracy": "overall_accuracy",
    "bfcl_normalized_accuracy": "normalized_single_turn_score",
}
