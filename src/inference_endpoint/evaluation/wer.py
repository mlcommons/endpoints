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

"""Word Error Rate as defined by the MLPerf speech2text benchmark.

Ported from the MLCommons reference implementation
(https://github.com/mlcommons/inference/tree/master/speech2text), which is also
Apache-2.0: ``legacy_helpers.__levenshtein``,
``helpers.compute_wer_with_concatenation``, and the character-label projection
``accuracy_eval.py`` applies to both sides before scoring.

The concatenation-aware alignment is why a generic WER library (jiwer,
torchmetrics) will not reproduce the reference number: the reference credits a
hypothesis word that equals up to three joined reference words, and vice versa,
before measuring edit distance. See ``docs/evaluation/DESIGN.md``.
"""

from collections.abc import Sequence

#: Character inventory the reference projects both hypotheses and references
#: through before normalization. Anything outside it is dropped.
LABELS = frozenset(" abcdefghijklmnopqrstuvwxyz'0123456789$%+-")

#: Longest run of words the alignment will try to join on either side.
MAX_CONCATENATION_SPAN = 3


def project_to_labels(text: str) -> str:
    """Lowercase `text` and drop every character outside :data:`LABELS`.

    The reference encodes both the SUT output and the manifest transcript as
    indices into its label list and decodes them back before scoring, which is
    lossy in exactly this way. Reproducing it keeps punctuation handling
    identical to the reference.

    The lowercasing is applied to both sides. The reference lowercases only the
    SUT output (in its SUT) and not the manifest transcript (in
    ``manifest.parse_transcript``), so an uppercase transcript — which is how
    LibriSpeech ships — would have every letter dropped as out-of-inventory,
    leaving an empty reference. Lowercasing both is a no-op whenever the
    manifest is already lowercase, and the only behaviour that yields a
    meaningful number when it is not.
    """
    lowered = text.lower()
    return "".join(c for c in lowered if c in LABELS)


def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Edit distance between two word sequences, in O(min(len(a), len(b))) space."""
    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current[j] = min(add, delete, change)
    return current[n]


def wer_counts(hypothesis: Sequence[str], reference: Sequence[str]) -> tuple[int, int]:
    """Return ``(edit_distance, reference_word_count)`` for one sample.

    Walks both word lists in lockstep, consuming a joined run of up to
    :data:`MAX_CONCATENATION_SPAN` words on either side when it matches a single
    word on the other, then takes the edit distance over the reconciled
    sequences. Corpus WER is ``sum(distances) / sum(word_counts)`` — never the
    mean of per-sample rates.
    """
    alignment: list[tuple[str | None, str | None]] = []
    i = j = 0

    while i < len(reference) and j < len(hypothesis):
        if reference[i] == hypothesis[j]:
            alignment.append((reference[i], hypothesis[j]))
            i += 1
            j += 1
            continue

        ref_match_len = hyp_match_len = 1
        match_found = False
        for k in range(1, MAX_CONCATENATION_SPAN + 1):
            if (
                i + k <= len(reference)
                and "".join(reference[i : i + k]) == hypothesis[j]
            ):
                ref_match_len, hyp_match_len = k, 1
                match_found = True
                break
            if (
                j + k <= len(hypothesis)
                and "".join(hypothesis[j : j + k]) == reference[i]
            ):
                ref_match_len, hyp_match_len = 1, k
                match_found = True
                break

        if match_found:
            alignment.append(
                (
                    " ".join(reference[i : i + ref_match_len]),
                    " ".join(hypothesis[j : j + hyp_match_len]),
                )
            )
            i += ref_match_len
            j += hyp_match_len
        else:
            # No reconciliation possible: count it as a substitution and advance
            # both sides, matching the reference's alignment walk.
            alignment.append((reference[i], hypothesis[j]))
            i += 1
            j += 1

    while i < len(reference):
        alignment.append((reference[i], None))
        i += 1
    while j < len(hypothesis):
        alignment.append((None, hypothesis[j]))
        j += 1

    # Spaces are stripped so a reconciled pair ("can not", "cannot") compares
    # equal; the distance is measured over these joined tokens.
    ref_aligned = [r.replace(" ", "") for r, _ in alignment if r is not None]
    hyp_aligned = [h.replace(" ", "") for _, h in alignment if h is not None]
    return levenshtein(ref_aligned, hyp_aligned), len(reference)
