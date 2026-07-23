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

"""Unit tests for Dataset.with_salt() and Dataset._apply_salt()."""

import random
import re

import pandas as pd
import pytest
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.exceptions import DatasetValidationError


def _make_loaded_dataset(rows: list[dict]) -> Dataset:
    """Return a Dataset with .data already populated (no file I/O)."""
    ds = Dataset.__new__(Dataset)
    ds.dataframe = None
    ds.transforms = None
    ds.repeats = 1
    ds.data = list(rows)
    ds._salt_rng = None
    return ds


@pytest.mark.unit
class TestDatasetWithSalt:
    """Dataset.with_salt() returns a shallow-copy view with salt applied on load_sample."""

    def test_returns_same_type(self):
        inner = _make_loaded_dataset([{"prompt": "x"}])
        salted = inner.with_salt(random.Random())
        assert type(salted) is type(inner)

    def test_salt_rng_set_on_clone(self):
        inner = _make_loaded_dataset([{"prompt": "x"}])
        rng = random.Random(42)
        salted = inner.with_salt(rng)
        assert salted._salt_rng is rng

    def test_original_has_no_salt_rng(self):
        inner = _make_loaded_dataset([{"prompt": "x"}])
        inner.with_salt(random.Random())
        assert inner._salt_rng is None

    def test_data_shared_by_reference(self):
        inner = _make_loaded_dataset([{"prompt": "x"}])
        salted = inner.with_salt(random.Random())
        assert salted.data is inner.data

    def test_num_samples_matches_inner(self):
        inner = _make_loaded_dataset(
            [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
        )
        salted = inner.with_salt(random.Random())
        assert salted.num_samples() == 3

    def test_repeats_matches_inner(self):
        inner = _make_loaded_dataset([{"prompt": "x"}])
        inner.repeats = 5
        salted = inner.with_salt(random.Random())
        assert salted.repeats == 5

    def test_unsalted_load_sample_unchanged(self):
        inner = _make_loaded_dataset([{"prompt": "hello"}])
        assert inner.load_sample(0) == {"prompt": "hello"}


@pytest.mark.unit
class TestSaltBehavior:
    """Salt is injected correctly into prompts via load_sample on a salted dataset."""

    _SALT_RE = re.compile(r"^\[([0-9a-f]{16})\] (.+)$")

    def test_prompt_prefixed_with_salt(self):
        inner = _make_loaded_dataset([{"prompt": "hello world"}])
        sd = inner.with_salt(random.Random())
        result = sd.load_sample(0)
        assert self._SALT_RE.match(
            result["prompt"]
        ), f"Expected '[<16-hex>] hello world', got: {result['prompt']!r}"

    def test_salt_is_exactly_16_hex_chars(self):
        inner = _make_loaded_dataset([{"prompt": "test"}])
        sd = inner.with_salt(random.Random())
        m = self._SALT_RE.match(sd.load_sample(0)["prompt"])
        assert m is not None
        assert len(m.group(1)) == 16

    def test_original_prompt_preserved_after_salt(self):
        inner = _make_loaded_dataset([{"prompt": "my question"}])
        sd = inner.with_salt(random.Random())
        m = self._SALT_RE.match(sd.load_sample(0)["prompt"])
        assert m is not None
        assert m.group(2) == "my question"

    def test_salt_unique_across_calls_same_index(self):
        inner = _make_loaded_dataset([{"prompt": "repeated"}])
        sd = inner.with_salt(random.Random())
        salts = {sd.load_sample(0)["prompt"][:18] for _ in range(20)}
        assert len(salts) == 20

    def test_salt_unique_across_same_prompt_at_different_indices(self):
        # Both rows have the same prompt — salt alone must make them distinct.
        inner = _make_loaded_dataset([{"prompt": "same"}, {"prompt": "same"}])
        sd = inner.with_salt(random.Random())
        assert sd.load_sample(0)["prompt"] != sd.load_sample(1)["prompt"]

    def test_other_fields_unchanged(self):
        inner = _make_loaded_dataset(
            [{"prompt": "hi", "system": "you are helpful", "extra": 42}]
        )
        sd = inner.with_salt(random.Random())
        result = sd.load_sample(0)
        assert result["system"] == "you are helpful"
        assert result["extra"] == 42

    def test_original_dict_not_mutated(self):
        row = {"prompt": "original"}
        inner = _make_loaded_dataset([row])
        sd = inner.with_salt(random.Random())
        sd.load_sample(0)
        assert inner.data[0]["prompt"] == "original"

    def test_seeded_rng_is_reproducible(self):
        inner = _make_loaded_dataset([{"prompt": "x"}])
        sd1 = inner.with_salt(random.Random(99))
        sd2 = inner.with_salt(random.Random(99))
        assert sd1.load_sample(0)["prompt"] == sd2.load_sample(0)["prompt"]


@pytest.mark.unit
class TestSaltValidation:
    """with_salt() hard-errors up front unless every sample has a text 'prompt'.

    salt=True guarantees a KV-cache-busting prefix; a sample it cannot salt is a
    configuration error, not something to skip silently. Validation runs in
    with_salt() (before any load is issued), so the error names the offending
    sample and no partial warmup runs against an unsalted dataset.
    """

    def test_dict_without_prompt_key_raises(self):
        inner = _make_loaded_dataset([{"question": "what is 2+2?", "answer": "4"}])
        with pytest.raises(DatasetValidationError, match="prompt"):
            inner.with_salt(random.Random())

    def test_empty_dict_raises(self):
        inner = _make_loaded_dataset([{}])
        with pytest.raises(DatasetValidationError, match="prompt"):
            inner.with_salt(random.Random())

    def test_non_dict_sample_raises(self):
        inner = _make_loaded_dataset([{"prompt": "x"}])
        inner.data = ["raw string sample"]
        with pytest.raises(DatasetValidationError, match="dict"):
            inner.with_salt(random.Random())

    def test_multimodal_list_prompt_raises(self):
        content_parts = [
            {"type": "text", "text": "describe this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        inner = _make_loaded_dataset([{"prompt": content_parts}])
        with pytest.raises(DatasetValidationError, match="str"):
            inner.with_salt(random.Random())

    def test_non_str_prompt_raises(self):
        inner = _make_loaded_dataset([{"prompt": 42}])
        with pytest.raises(DatasetValidationError, match="str"):
            inner.with_salt(random.Random())

    def test_input_tokens_only_raises(self):
        inner = _make_loaded_dataset([{"input_tokens": [1, 2, 3]}])
        with pytest.raises(DatasetValidationError, match="input_tokens"):
            inner.with_salt(random.Random())

    def test_input_tokens_and_prompt_raises(self):
        inner = _make_loaded_dataset([{"input_tokens": [1, 2, 3], "prompt": "hello"}])
        with pytest.raises(DatasetValidationError, match="input_tokens"):
            inner.with_salt(random.Random())

    def test_error_names_offending_sample_index(self):
        inner = _make_loaded_dataset([{"prompt": "ok"}, {"prompt": 42}])
        with pytest.raises(DatasetValidationError, match=r"\b1\b"):
            inner.with_salt(random.Random())

    def test_valid_str_prompt_dataset_does_not_raise(self):
        inner = _make_loaded_dataset([{"prompt": "a"}, {"prompt": "b"}])
        sd = inner.with_salt(random.Random())
        assert sd.load_sample(0)["prompt"].startswith("[")

    def test_data_none_does_not_raise(self):
        inner = _make_loaded_dataset([{"prompt": "x"}])
        inner.data = None
        # No samples to salt (e.g. EmptyDataset) — nothing to validate.
        assert inner.with_salt(random.Random())._salt_rng is not None

    def test_data_empty_list_does_not_raise(self):
        inner = _make_loaded_dataset([])
        # Zero samples — no violation, so no error.
        assert inner.with_salt(random.Random())._salt_rng is not None

    def test_validate_saltable_noop_on_valid(self):
        inner = _make_loaded_dataset([{"prompt": "a"}, {"prompt": "b"}])
        assert inner.validate_saltable() is None

    def test_validate_saltable_raises_on_bad_sample(self):
        inner = _make_loaded_dataset([{"prompt": "ok"}, {"input_tokens": [1, 2]}])
        with pytest.raises(DatasetValidationError, match="input_tokens"):
            inner.validate_saltable()


@pytest.mark.unit
class TestSaltWithRealDataset:
    """Integration-style: with_salt() on a real Dataset loaded from a DataFrame."""

    @pytest.fixture
    def loaded_ds(self):
        df = pd.DataFrame(
            {
                "prompt": ["What is AI?", "Explain gradient descent"],
                "category": ["general", "ml"],
            }
        )
        ds = Dataset(df)
        ds.load()
        return ds

    def test_num_samples_unchanged(self, loaded_ds):
        assert loaded_ds.with_salt(random.Random()).num_samples() == 2

    def test_prompts_are_salted(self, loaded_ds):
        sd = loaded_ds.with_salt(random.Random())
        for i in range(sd.num_samples()):
            assert sd.load_sample(i)["prompt"].startswith("[")

    def test_category_field_preserved(self, loaded_ds):
        sd = loaded_ds.with_salt(random.Random())
        assert sd.load_sample(0)["category"] == "general"
        assert sd.load_sample(1)["category"] == "ml"

    def test_all_salts_unique(self, loaded_ds):
        sd = loaded_ds.with_salt(random.Random())
        prompts = [sd.load_sample(i)["prompt"] for i in range(sd.num_samples())]
        assert len(set(prompts)) == len(prompts)
