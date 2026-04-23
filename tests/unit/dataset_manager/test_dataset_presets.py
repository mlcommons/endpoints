# SPDX-FileCopyrightText: 2026 Intel Corporation
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

"""
Unit tests for preset dataset transforms.

Tests verify that each preset configuration:
1. Can be instantiated without errors
2. Applies transforms correctly to sample data
3. Produces expected output columns

These tests do NOT require end-to-end benchmarking or external compute resources.
Instead, they use minimal dummy datasets with the required columns.
"""

import pandas as pd
import pytest
from inference_endpoint.dataset_manager.predefined.aime25 import AIME25
from inference_endpoint.dataset_manager.predefined.cnndailymail import CNNDailyMail
from inference_endpoint.dataset_manager.predefined.gpqa import GPQA
from inference_endpoint.dataset_manager.predefined.livecodebench import LiveCodeBench
from inference_endpoint.dataset_manager.predefined.open_orca import OpenOrca
from inference_endpoint.dataset_manager.transforms import apply_transforms


class TestCNNDailyMailPresets:
    """Test CNN/DailyMail dataset presets."""

    @pytest.fixture
    def sample_cnn_data(self):
        """Create minimal sample data matching CNN/DailyMail schema."""
        return pd.DataFrame(
            {
                "article": [
                    "CNN reported today that markets are up. Stocks rose 2%.",
                    "Breaking news: New policy announced. Impact expected next quarter.",
                ],
                "highlights": [
                    "Markets up 2%",
                    "Policy announced",
                ],
            }
        )

    @pytest.fixture
    def llama3_8b_transformed(self, sample_cnn_data):
        """Apply llama3_8b preset transforms to sample data."""
        transforms = CNNDailyMail.PRESETS.llama3_8b()
        return apply_transforms(sample_cnn_data, transforms)

    @pytest.fixture
    def llama3_8b_sglang_transformed(self, sample_cnn_data):
        """Apply llama3_8b_sglang preset transforms to sample data."""
        transforms = CNNDailyMail.PRESETS.llama3_8b_sglang()
        return apply_transforms(sample_cnn_data, transforms)

    def test_llama3_8b_preset_instantiation(self):
        """Test that llama3_8b preset can be instantiated."""
        transforms = CNNDailyMail.PRESETS.llama3_8b()
        assert transforms is not None
        assert len(transforms) > 0

    def test_llama3_8b_transforms_apply(self, llama3_8b_transformed):
        """Test that llama3_8b transforms apply without errors."""
        assert llama3_8b_transformed is not None
        assert "prompt" in llama3_8b_transformed.columns
        assert len(llama3_8b_transformed["prompt"][0]) > 0

    def test_llama3_8b_prompt_format(self, llama3_8b_transformed, sample_cnn_data):
        """Test that llama3_8b produces properly formatted prompts."""
        prompt = llama3_8b_transformed["prompt"][0]
        assert "Summarize" in prompt
        assert "news article" in prompt
        assert "article" in sample_cnn_data.columns
        # The original article should be embedded in the prompt
        assert sample_cnn_data["article"][0] in prompt

    @pytest.mark.slow
    def test_llama3_8b_sglang_preset_instantiation(self):
        """Test that llama3_8b_sglang preset can be instantiated."""
        transforms = CNNDailyMail.PRESETS.llama3_8b_sglang()
        assert transforms is not None
        assert len(transforms) > 0

    @pytest.mark.slow
    def test_llama3_8b_sglang_transforms_apply(self, llama3_8b_sglang_transformed):
        """Test that llama3_8b_sglang transforms apply without errors."""
        assert llama3_8b_sglang_transformed is not None
        # SGLang preset should still provide a prompt column
        assert "prompt" in llama3_8b_sglang_transformed.columns
        # Key output for SGLang preset is tokenized input
        assert "input_tokens" in llama3_8b_sglang_transformed.columns
        input_tokens = llama3_8b_sglang_transformed["input_tokens"].iloc[0]
        assert isinstance(input_tokens, list)
        assert len(input_tokens) > 0
        assert all(isinstance(token, int) for token in input_tokens)
        # harmonized_column is expected to be None for this preset
        assert "harmonized_prompt" not in llama3_8b_sglang_transformed.columns


class TestAIME25Presets:
    """Test AIME25 dataset presets."""

    @pytest.fixture
    def sample_aime_data(self):
        """Create minimal sample data matching AIME25 schema."""
        return pd.DataFrame(
            {
                "question": [
                    "If x + 1 = 5, then x = ?",
                    "What is 2 + 2 * 3?",
                ],
                "answer": [4, 8],
            }
        )

    @pytest.fixture
    def gptoss_transformed(self, sample_aime_data):
        """Apply gptoss preset transforms to sample data."""
        transforms = AIME25.PRESETS.gptoss()
        return apply_transforms(sample_aime_data, transforms)

    def test_gptoss_preset_instantiation(self):
        """Test that gptoss preset can be instantiated."""
        transforms = AIME25.PRESETS.gptoss()
        assert transforms is not None
        assert len(transforms) > 0

    def test_gptoss_transforms_apply(self, gptoss_transformed):
        """Test that gptoss transforms apply without errors."""
        assert gptoss_transformed is not None
        assert "prompt" in gptoss_transformed.columns

    def test_gptoss_includes_boxed_answer_format(self, gptoss_transformed):
        """Test that gptoss format includes boxed answer format."""
        prompt = gptoss_transformed["prompt"][0]
        # AIME preset should instruct to put answer in \boxed{}
        assert "boxed" in prompt.lower() or "box" in prompt


class TestGPQAPresets:
    """Test GPQA dataset presets."""

    @pytest.fixture
    def sample_gpqa_data(self):
        """Create minimal sample data matching GPQA schema."""
        return pd.DataFrame(
            {
                "question": [
                    "What is the capital of France?",
                    "Who discovered penicillin?",
                ],
                "choice1": ["Paris", "Alexander Fleming"],
                "choice2": ["London", "Marie Curie"],
                "choice3": ["Berlin", "Louis Pasteur"],
                "choice4": ["Madrid", "Joseph Lister"],
                "correct_choice": ["A", "A"],
            }
        )

    @pytest.fixture
    def gptoss_transformed(self, sample_gpqa_data):
        """Apply gptoss preset transforms to sample data."""
        transforms = GPQA.PRESETS.gptoss()
        return apply_transforms(sample_gpqa_data, transforms)

    def test_gptoss_preset_instantiation(self):
        """Test that gptoss preset can be instantiated."""
        transforms = GPQA.PRESETS.gptoss()
        assert transforms is not None
        assert len(transforms) > 0

    def test_gptoss_transforms_apply(self, gptoss_transformed):
        """Test that gptoss transforms apply without errors."""
        assert gptoss_transformed is not None
        assert "prompt" in gptoss_transformed.columns

    def test_gptoss_format_includes_choices(self, gptoss_transformed):
        """Test that gptoss format includes all multiple choice options."""
        prompt = gptoss_transformed["prompt"][0]
        # Should include all four choices formatted as (A), (B), (C), (D)
        assert "(A)" in prompt
        assert "(B)" in prompt
        assert "(C)" in prompt
        assert "(D)" in prompt
        # Should instruct to express answer as option letter
        assert "A" in prompt or "option" in prompt.lower()


class TestLiveCodeBenchPresets:
    """Test LiveCodeBench dataset presets."""

    @pytest.fixture
    def sample_lcb_data(self):
        """Create minimal sample data matching LiveCodeBench schema."""
        return pd.DataFrame(
            {
                "question": [
                    "Write a function that returns the sum of two numbers.",
                    "Write a function that reverses a string.",
                ],
                "starter_code": [
                    "def add(a, b):\n    pass",
                    "def reverse(s):\n    pass",
                ],
            }
        )

    @pytest.fixture
    def gptoss_transformed(self, sample_lcb_data):
        """Apply gptoss preset transforms to sample data."""
        transforms = LiveCodeBench.PRESETS.gptoss()
        return apply_transforms(sample_lcb_data, transforms)

    def test_gptoss_preset_instantiation(self):
        """Test that gptoss preset can be instantiated."""
        transforms = LiveCodeBench.PRESETS.gptoss()
        assert transforms is not None
        assert len(transforms) > 0

    def test_gptoss_transforms_apply(self, gptoss_transformed):
        """Test that gptoss transforms apply without errors."""
        assert gptoss_transformed is not None
        assert "prompt" in gptoss_transformed.columns

    def test_gptoss_format_includes_code_delimiters(
        self, gptoss_transformed, sample_lcb_data
    ):
        """Test that gptoss format includes code delimiters."""
        prompt = gptoss_transformed["prompt"][0]
        # Should include ```python delimiters for code
        assert "```python" in prompt
        assert "starter_code" in sample_lcb_data.columns
        # Starter code should be included in prompt
        assert sample_lcb_data["starter_code"][0] in prompt


class TestOpenOrcaPresets:
    """Test OpenOrca dataset presets."""

    @pytest.fixture
    def sample_openorca_data(self):
        """Create minimal sample data matching OpenOrca schema."""
        return pd.DataFrame(
            {
                "question": [
                    "What is machine learning?",
                    "Explain neural networks.",
                ],
                "system_prompt": [
                    "You are an AI expert.",
                    "You are a technical educator.",
                ],
                "response": [
                    "Machine learning is...",
                    "Neural networks are...",
                ],
            }
        )

    @pytest.fixture
    def llama2_70b_transformed(self, sample_openorca_data):
        """Apply llama2_70b preset transforms to sample data."""
        transforms = OpenOrca.PRESETS.llama2_70b()
        return apply_transforms(sample_openorca_data, transforms)

    def test_llama2_70b_preset_instantiation(self):
        """Test that llama2_70b preset can be instantiated."""
        transforms = OpenOrca.PRESETS.llama2_70b()
        assert transforms is not None
        assert len(transforms) > 0

    def test_llama2_70b_transforms_apply(self, llama2_70b_transformed):
        """Test that llama2_70b transforms apply without errors."""
        assert llama2_70b_transformed is not None
        assert "prompt" in llama2_70b_transformed.columns
        assert "system" in llama2_70b_transformed.columns

    def test_llama2_70b_remaps_columns(
        self, llama2_70b_transformed, sample_openorca_data
    ):
        """Test that llama2_70b correctly remaps question->prompt and system_prompt->system."""
        # After transformation, original columns should be renamed
        assert "prompt" in llama2_70b_transformed.columns
        assert "system" in llama2_70b_transformed.columns
        # Data should be preserved in renamed columns
        assert (
            llama2_70b_transformed["prompt"][0] == sample_openorca_data["question"][0]
        )
        assert (
            llama2_70b_transformed["system"][0]
            == sample_openorca_data["system_prompt"][0]
        )
