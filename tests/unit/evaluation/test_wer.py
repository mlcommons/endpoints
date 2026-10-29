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

"""Unit tests for the MLPerf speech2text WER metric."""

import pytest
from inference_endpoint.evaluation.wer import (
    LABELS,
    levenshtein,
    project_to_labels,
    wer_counts,
)


@pytest.mark.unit
class TestLevenshtein:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ([], [], 0),
            (["a"], [], 1),
            ([], ["a"], 1),
            (["a", "b", "c"], ["a", "b", "c"], 0),
            (["a", "b", "c"], ["a", "b", "d"], 1),
            (list("kitten"), list("sitting"), 3),
            (["a", "b"], ["b", "a"], 2),
        ],
    )
    def test_distance(self, a, b, expected):
        assert levenshtein(a, b) == expected

    def test_is_symmetric(self):
        a, b = list("saturday"), list("sunday")
        assert levenshtein(a, b) == levenshtein(b, a)


@pytest.mark.unit
class TestProjectToLabels:
    def test_lowercases(self):
        assert project_to_labels("MISTER QUILTER") == "mister quilter"

    def test_drops_punctuation_outside_the_inventory(self):
        assert project_to_labels("Hello, world! (yes)") == "hello world yes"

    def test_keeps_apostrophes_digits_and_symbols(self):
        assert project_to_labels("IT'S 42% +3 -1 $5") == "it's 42% +3 -1 $5"

    def test_inventory_matches_the_reference_label_set(self):
        assert LABELS == frozenset(" abcdefghijklmnopqrstuvwxyz'0123456789$%+-")


@pytest.mark.unit
class TestWerCounts:
    def test_exact_match_scores_zero_errors(self):
        words = ["the", "cat", "sat"]
        assert wer_counts(words, words) == (0, 3)

    def test_substitution_counts_one_error(self):
        assert wer_counts(["the", "dog", "sat"], ["the", "cat", "sat"]) == (1, 3)

    def test_deletion_counts_one_error(self):
        assert wer_counts(["the", "sat"], ["the", "cat", "sat"]) == (1, 3)

    def test_insertion_counts_one_error(self):
        assert wer_counts(["the", "big", "cat", "sat"], ["the", "cat", "sat"]) == (1, 3)

    def test_empty_hypothesis_counts_every_reference_word(self):
        assert wer_counts([], ["the", "cat", "sat"]) == (3, 3)

    def test_empty_reference_reports_zero_words(self):
        # Denominator of 0 - the caller must sum counts across the corpus before
        # dividing, which is why this returns counts rather than a rate.
        assert wer_counts(["the"], []) == (1, 0)

    def test_both_empty(self):
        assert wer_counts([], []) == (0, 0)

    def test_joined_hypothesis_word_is_credited(self):
        # "can not" -> "cannot" is the concatenation case a generic WER library
        # would score as 2 errors.
        assert wer_counts(["cannot"], ["can", "not"]) == (0, 2)

    def test_joined_reference_word_is_credited(self):
        assert wer_counts(["can", "not"], ["cannot"]) == (0, 1)

    def test_three_word_join_is_credited(self):
        assert wer_counts(["anyoneelse"], ["any", "one", "else"]) == (0, 3)

    def test_four_word_join_exceeds_the_span_and_counts_errors(self):
        distance, words = wer_counts(["abcd"], ["a", "b", "c", "d"])
        assert words == 4
        assert distance > 0

    def test_join_credited_mid_sentence(self):
        assert wer_counts(["i", "cannot", "go"], ["i", "can", "not", "go"]) == (0, 4)

    def test_corpus_wer_is_summed_not_averaged(self):
        samples = [
            (["the", "dog"], ["the", "cat"]),
            (["a"] * 10, ["a"] * 10),
        ]
        distance = words = 0
        for hyp, ref in samples:
            d, w = wer_counts(hyp, ref)
            distance += d
            words += w
        # 1 error over 12 reference words, not the mean of 50% and 0%.
        assert (distance, words) == (1, 12)


@pytest.mark.unit
class TestEndToEndOnText:
    def test_reference_pipeline_on_a_librispeech_style_pair(self):
        reference = "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES"
        hypothesis = "Mister Quilter is the apostle of the middle classes."
        hyp = project_to_labels(hypothesis).split()
        ref = project_to_labels(reference).split()
        assert wer_counts(hyp, ref) == (0, 9)
