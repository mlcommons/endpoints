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

"""Unit tests for SaltedDataset."""

import re
from unittest.mock import MagicMock

import pandas as pd
import pytest
from inference_endpoint.dataset_manager.dataset import Dataset, SaltedDataset


def _make_loaded_dataset(rows: list[dict]) -> Dataset:
    """Return a Dataset with .data already populated (no file I/O)."""
    ds = Dataset.__new__(Dataset)
    ds.dataframe = None
    ds.transforms = None
    ds.repeats = 1
    ds.data = list(rows)
    ds.logger = MagicMock()
    return ds


@pytest.mark.unit
class TestSaltedDatasetDelegation:
    """SaltedDataset correctly mirrors inner-dataset properties."""

    def test_num_samples_delegates_to_inner(self):
        inner = _make_loaded_dataset(
            [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]
        )
        sd = SaltedDataset(inner)
        assert sd.num_samples() == 3

    def test_data_attribute_is_inner_data(self):
        inner = _make_loaded_dataset([{"prompt": "x"}])
        sd = SaltedDataset(inner)
        assert sd.data is inner.data

    def test_repeats_matches_inner(self):
        inner = _make_loaded_dataset([{"prompt": "x"}])
        inner.repeats = 5
        sd = SaltedDataset(inner)
        assert sd.repeats == 5

    def test_load_is_noop(self):
        inner = _make_loaded_dataset([{"prompt": "x"}])
        sd = SaltedDataset(inner)
        sd.load()  # must not raise
        assert sd.data is inner.data  # data unchanged after load()


@pytest.mark.unit
class TestSaltedDatasetSaltBehavior:
    """Salt is injected correctly into prompts."""

    _SALT_RE = re.compile(r"^\[([0-9a-f]{16})\] (.+)$")

    def test_prompt_prefixed_with_salt(self):
        inner = _make_loaded_dataset([{"prompt": "hello world"}])
        sd = SaltedDataset(inner)
        result = sd.load_sample(0)
        assert self._SALT_RE.match(
            result["prompt"]
        ), f"Expected '[<16-hex>] hello world', got: {result['prompt']!r}"

    def test_salt_is_exactly_16_hex_chars(self):
        inner = _make_loaded_dataset([{"prompt": "test"}])
        sd = SaltedDataset(inner)
        result = sd.load_sample(0)
        m = self._SALT_RE.match(result["prompt"])
        assert m is not None
        assert len(m.group(1)) == 16

    def test_original_prompt_preserved_after_salt(self):
        inner = _make_loaded_dataset([{"prompt": "my question"}])
        sd = SaltedDataset(inner)
        result = sd.load_sample(0)
        m = self._SALT_RE.match(result["prompt"])
        assert m is not None
        assert m.group(2) == "my question"

    def test_salt_unique_across_calls_same_index(self):
        inner = _make_loaded_dataset([{"prompt": "repeated"}])
        sd = SaltedDataset(inner)
        salts = {sd.load_sample(0)["prompt"][:18] for _ in range(20)}
        # With 8 bytes of randomness we expect all 20 samples to be distinct
        assert len(salts) == 20

    def test_salt_unique_across_different_indices(self):
        inner = _make_loaded_dataset([{"prompt": "a"}, {"prompt": "b"}])
        sd = SaltedDataset(inner)
        prompt0 = sd.load_sample(0)["prompt"]
        prompt1 = sd.load_sample(1)["prompt"]
        # Salts should differ (original prompts differ, salts are random)
        assert prompt0 != prompt1

    def test_other_fields_unchanged(self):
        inner = _make_loaded_dataset(
            [{"prompt": "hi", "system": "you are helpful", "extra": 42}]
        )
        sd = SaltedDataset(inner)
        result = sd.load_sample(0)
        assert result["system"] == "you are helpful"
        assert result["extra"] == 42

    def test_original_dict_not_mutated(self):
        row = {"prompt": "original"}
        inner = _make_loaded_dataset([row])
        sd = SaltedDataset(inner)
        sd.load_sample(0)
        assert inner.data[0]["prompt"] == "original"


@pytest.mark.unit
class TestSaltedDatasetPassthrough:
    """Samples without a 'prompt' key, or non-dict samples, are passed through unchanged."""

    def test_dict_without_prompt_key_is_unchanged(self):
        inner = _make_loaded_dataset([{"question": "what is 2+2?", "answer": "4"}])
        sd = SaltedDataset(inner)
        result = sd.load_sample(0)
        assert result == {"question": "what is 2+2?", "answer": "4"}

    def test_empty_dict_is_unchanged(self):
        inner = _make_loaded_dataset([{}])
        sd = SaltedDataset(inner)
        assert sd.load_sample(0) == {}

    def test_non_dict_sample_is_returned_as_is(self):
        inner = _make_loaded_dataset([{"prompt": "x"}])
        inner.data = ["raw string sample"]  # override with non-dict
        sd = SaltedDataset(inner)
        sd.data = inner.data
        assert sd.load_sample(0) == "raw string sample"

    def test_multimodal_list_prompt_first_text_part_is_salted(self):
        """Salt is injected into the 'text' field of the first content part."""
        content_parts = [
            {"type": "text", "text": "describe this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        inner = _make_loaded_dataset([{"prompt": content_parts}])
        sd = SaltedDataset(inner)
        result = sd.load_sample(0)
        parts = result["prompt"]
        assert isinstance(parts, list)
        assert len(parts) == 2
        # First text part must carry a salt prefix
        assert re.match(r"^\[([0-9a-f]{16})\] describe this image$", parts[0]["text"])
        # Image part must be unchanged
        assert parts[1] == content_parts[1]

    def test_multimodal_list_prompt_original_not_mutated(self):
        """Salting a multimodal prompt must not modify the original list or dicts."""
        content_parts = [{"type": "text", "text": "original text"}]
        inner = _make_loaded_dataset([{"prompt": content_parts}])
        sd = SaltedDataset(inner)
        sd.load_sample(0)
        assert inner.data[0]["prompt"][0]["text"] == "original text"

    def test_unknown_prompt_type_is_not_salted(self):
        """A prompt that is neither str nor a recognised list-of-parts is returned unchanged."""
        inner = _make_loaded_dataset([{"prompt": 42}])
        sd = SaltedDataset(inner)
        result = sd.load_sample(0)
        assert result == {"prompt": 42}


@pytest.mark.unit
class TestSaltedDatasetWithRealDataset:
    """Integration-style: SaltedDataset wrapping a real Dataset loaded from a DataFrame."""

    @pytest.fixture
    def loaded_inner(self):
        df = pd.DataFrame(
            {
                "prompt": ["What is AI?", "Explain gradient descent"],
                "category": ["general", "ml"],
            }
        )
        ds = Dataset(df)
        ds.load()
        return ds

    def test_wraps_real_dataset_correctly(self, loaded_inner):
        sd = SaltedDataset(loaded_inner)
        assert sd.num_samples() == 2

    def test_real_dataset_prompts_are_salted(self, loaded_inner):
        sd = SaltedDataset(loaded_inner)
        for i in range(sd.num_samples()):
            result = sd.load_sample(i)
            assert result["prompt"].startswith("[")
            assert "] " in result["prompt"]

    def test_category_field_preserved(self, loaded_inner):
        sd = SaltedDataset(loaded_inner)
        assert sd.load_sample(0)["category"] == "general"
        assert sd.load_sample(1)["category"] == "ml"

    def test_all_salts_unique_across_full_dataset(self, loaded_inner):
        sd = SaltedDataset(loaded_inner)
        prompts = [sd.load_sample(i)["prompt"] for i in range(sd.num_samples())]
        assert len(set(prompts)) == len(prompts)
