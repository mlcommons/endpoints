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

"""Unit tests for _precompute_isl_for_multi_turn."""

from unittest.mock import MagicMock, patch

import pytest
from inference_endpoint.commands.benchmark.execute import _precompute_isl_for_multi_turn


def _make_dataloader(samples: list[dict]) -> MagicMock:
    dl = MagicMock()
    dl.data = samples
    return dl


class TestPrecomputeIslForMultiTurn:
    @pytest.mark.unit
    def test_sets_input_tokens_for_samples_with_messages(self):
        samples = [
            {"messages": [{"role": "user", "content": "hello"}]},
            {"messages": [{"role": "user", "content": "world"}]},
        ]
        dataloader = _make_dataloader(samples)
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = lambda msgs, **_: list(
            range(len(msgs) * 3)
        )

        with patch("transformers.AutoTokenizer") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_tokenizer
            _precompute_isl_for_multi_turn(dataloader, "test-model")

        for sample in samples:
            assert "input_tokens" in sample
            assert isinstance(sample["input_tokens"], list)

    @pytest.mark.unit
    def test_leaves_samples_without_messages_untouched(self):
        samples = [
            {"prompt": "no messages here"},
            {"input_tokens": [1, 2, 3]},
        ]
        dataloader = _make_dataloader(samples)
        mock_tokenizer = MagicMock()

        with patch("transformers.AutoTokenizer") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_tokenizer
            _precompute_isl_for_multi_turn(dataloader, "test-model")

        mock_tokenizer.apply_chat_template.assert_not_called()
        assert "input_tokens" not in samples[0]
        assert samples[1]["input_tokens"] == [1, 2, 3]

    @pytest.mark.unit
    def test_skips_failed_template_calls_with_warning(self, caplog):
        samples = [
            {"messages": [{"role": "user", "content": "good"}]},
            {"messages": [{"role": "user", "content": "bad"}]},
        ]
        dataloader = _make_dataloader(samples)

        def side_effect(msgs, **_):
            if msgs[0]["content"] == "bad":
                raise ValueError("template error")
            return [10, 20, 30]

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.side_effect = side_effect

        with patch("transformers.AutoTokenizer") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_tokenizer
            with caplog.at_level("WARNING"):
                _precompute_isl_for_multi_turn(dataloader, "test-model")

        assert "input_tokens" in samples[0]
        assert "input_tokens" not in samples[1]
        assert "1 turn(s) skipped" in caplog.text

    @pytest.mark.unit
    def test_add_generation_prompt_true(self):
        samples = [{"messages": [{"role": "user", "content": "hi"}]}]
        dataloader = _make_dataloader(samples)
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = [1, 2, 3]

        with patch("transformers.AutoTokenizer") as mock_cls:
            mock_cls.from_pretrained.return_value = mock_tokenizer
            _precompute_isl_for_multi_turn(dataloader, "test-model")

        _, kwargs = mock_tokenizer.apply_chat_template.call_args
        assert kwargs.get("add_generation_prompt") is True
        assert kwargs.get("tokenize") is True
