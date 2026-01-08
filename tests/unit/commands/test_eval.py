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

"""Tests for eval command.

These tests verify the evaluation command's input validation and error handling.
Since the full eval framework is not yet implemented, tests focus on:
- Dataset path validation (built-in vs custom paths)
- Proper exception raising (NotImplementedError for unimplemented features)
- Error messages and user guidance

The eval command is a stub that will be fully implemented later, but proper
error handling ensures users get clear feedback when trying to use it.
"""

from unittest.mock import MagicMock

import pytest
from inference_endpoint.commands.eval import run_eval_command
from inference_endpoint.exceptions import InputValidationError


class TestRunEvalCommand:
    """Test eval command error handling.

    Validates that dataset paths are checked and proper exceptions are raised.
    Critical for providing clear error messages before full eval implementation.
    """

    @pytest.mark.asyncio
    async def test_eval_not_implemented(self):
        """Test that eval raises NotImplementedError."""
        args = MagicMock()
        args.dataset = "gpqa"
        args.endpoint = "http://test.com"

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            await run_eval_command(args)

    @pytest.mark.asyncio
    async def test_eval_nonexistent_dataset_file(self):
        """Test eval with non-existent dataset file."""
        args = MagicMock()
        args.dataset = "/nonexistent/dataset.pkl"
        args.endpoint = "http://test.com"

        with pytest.raises(InputValidationError, match="Dataset not found"):
            await run_eval_command(args)

    @pytest.mark.asyncio
    async def test_eval_existing_dataset_file(self, tmp_path):
        """Test eval with existing dataset file (custom path)."""
        # Create a dummy dataset file
        dataset_file = tmp_path / "test_dataset.pkl"
        dataset_file.write_text("dummy data")

        args = MagicMock()
        args.dataset = str(dataset_file)
        args.endpoint = "http://test.com"

        # Should still raise NotImplementedError, but not InputValidationError
        with pytest.raises(NotImplementedError):
            await run_eval_command(args)

    @pytest.mark.asyncio
    async def test_eval_builtin_datasets(self, caplog):
        """Test eval recognizes built-in datasets."""
        args = MagicMock()
        args.dataset = "gpqa,math500,aime"
        args.endpoint = "http://test.com"

        with caplog.at_level("INFO"):
            with pytest.raises(NotImplementedError):
                await run_eval_command(args)

        # Should log that datasets are supported
        log_text = caplog.text
        assert "gpqa" in log_text
        assert "math500" in log_text
        assert "aime" in log_text
        assert "built-in" in log_text

    @pytest.mark.asyncio
    async def test_eval_mixed_datasets(self, tmp_path, caplog):
        """Test eval with mix of built-in and custom datasets."""
        # Create custom dataset
        custom_ds = tmp_path / "custom.pkl"
        custom_ds.write_text("data")

        args = MagicMock()
        args.dataset = f"gpqa,{custom_ds},math500"
        args.endpoint = "http://test.com"

        with caplog.at_level("INFO"):
            with pytest.raises(NotImplementedError):
                await run_eval_command(args)

        # Should recognize both types
        log_text = caplog.text
        assert "gpqa" in log_text and "built-in" in log_text
        assert "custom.pkl" in log_text and "custom path" in log_text
