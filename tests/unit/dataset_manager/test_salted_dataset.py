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
from unittest.mock import MagicMock

import pandas as pd
import pytest
from inference_endpoint.dataset_manager.dataset import Dataset


def _make_loaded_dataset(rows: list[dict]) -> Dataset:
    """Return a Dataset with .data already populated (no file I/O)."""
    ds = Dataset.__new__(Dataset)
    ds.dataframe = None
    ds.transforms = None
    ds.repeats = 1
    ds.data = list(rows)
    ds.logger = MagicMock()
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
class TestSaltPassthrough:
    """Samples without a 'prompt' key, or non-dict samples, are passed through unchanged."""

    def test_dict_without_prompt_key_is_unchanged(self):
        inner = _make_loaded_dataset([{"question": "what is 2+2?", "answer": "4"}])
        sd = inner.with_salt(random.Random())
        assert sd.load_sample(0) == {"question": "what is 2+2?", "answer": "4"}

    def test_empty_dict_is_unchanged(self):
        inner = _make_loaded_dataset([{}])
        sd = inner.with_salt(random.Random())
        assert sd.load_sample(0) == {}

    def test_non_dict_sample_is_returned_as_is(self):
        inner = _make_loaded_dataset([{"prompt": "x"}])
        inner.data = ["raw string sample"]
        sd = inner.with_salt(random.Random())
        assert sd.load_sample(0) == "raw string sample"

    def test_multimodal_list_prompt_first_text_part_is_salted(self):
        content_parts = [
            {"type": "text", "text": "describe this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        inner = _make_loaded_dataset([{"prompt": content_parts}])
        sd = inner.with_salt(random.Random())
        parts = sd.load_sample(0)["prompt"]
        assert isinstance(parts, list)
        assert len(parts) == 2
        assert re.match(r"^\[([0-9a-f]{16})\] describe this image$", parts[0]["text"])
        assert parts[1] == content_parts[1]

    def test_multimodal_image_first_text_at_index_1_is_salted(self):
        content_parts = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "text", "text": "what do you see?"},
        ]
        inner = _make_loaded_dataset([{"prompt": content_parts}])
        sd = inner.with_salt(random.Random())
        parts = sd.load_sample(0)["prompt"]
        assert parts[0] == content_parts[0]
        assert re.match(r"^\[([0-9a-f]{16})\] what do you see\?$", parts[1]["text"])

    def test_multimodal_list_prompt_original_not_mutated(self):
        content_parts = [{"type": "text", "text": "original text"}]
        inner = _make_loaded_dataset([{"prompt": content_parts}])
        sd = inner.with_salt(random.Random())
        sd.load_sample(0)
        assert inner.data[0]["prompt"][0]["text"] == "original text"

    def test_unknown_prompt_type_is_not_salted(self):
        inner = _make_loaded_dataset([{"prompt": 42}])
        sd = inner.with_salt(random.Random())
        assert sd.load_sample(0) == {"prompt": 42}

    def test_input_tokens_only_warns_and_passes_through(self):
        inner = _make_loaded_dataset([{"input_tokens": [1, 2, 3]}])
        sd = inner.with_salt(random.Random())
        result = sd.load_sample(0)
        assert result == {"input_tokens": [1, 2, 3]}
        sd.logger.warning.assert_called_once()
        assert "input_tokens" in sd.logger.warning.call_args[0][0]

    def test_input_tokens_and_prompt_warns_and_salts_prompt(self):
        inner = _make_loaded_dataset([{"input_tokens": [1, 2, 3], "prompt": "hello"}])
        sd = inner.with_salt(random.Random())
        result = sd.load_sample(0)
        assert result["input_tokens"] == [1, 2, 3]
        assert result["prompt"].startswith("[")
        sd.logger.warning.assert_called_once()
        assert "input_tokens" in sd.logger.warning.call_args[0][0]


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
