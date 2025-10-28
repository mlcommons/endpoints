# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for custom exceptions.

These tests verify the CLI exception hierarchy and ensure proper error
handling behavior throughout the CLI commands. The exception-based error
handling allows for:
- Testable error conditions (can assert exceptions instead of process exits)
- Composable commands (can be called programmatically)
- Centralized error handling in main()
- Clear error categorization (validation vs setup vs execution)
"""

from inference_endpoint.exceptions import (
    CLIError,
    ExecutionError,
    InputValidationError,
    SetupError,
)


class TestExceptionHierarchy:
    """Test exception class hierarchy.

    Ensures all CLI exceptions inherit properly and can be caught at
    different levels (specific exception or base CLIError).
    """

    def test_cli_error_base(self):
        """Test CLIError is base exception."""
        err = CLIError("test")
        assert isinstance(err, Exception)
        assert str(err) == "test"

    def test_input_validation_error_inherits_cli_error(self):
        """Test InputValidationError inherits from CLIError."""
        err = InputValidationError("validation failed")
        assert isinstance(err, CLIError)
        assert isinstance(err, Exception)

    def test_setup_error_inherits_cli_error(self):
        """Test SetupError inherits from CLIError."""
        err = SetupError("setup failed")
        assert isinstance(err, CLIError)
        assert isinstance(err, Exception)

    def test_execution_error_inherits_cli_error(self):
        """Test ExecutionError inherits from CLIError."""
        err = ExecutionError("execution failed")
        assert isinstance(err, CLIError)
        assert isinstance(err, Exception)

    def test_exception_messages(self):
        """Test exception messages are preserved."""
        msg = "Custom error message"
        assert str(InputValidationError(msg)) == msg
        assert str(SetupError(msg)) == msg
        assert str(ExecutionError(msg)) == msg

    def test_exception_chaining(self):
        """Test exception chaining with 'from'."""
        original = ValueError("original error")
        chained = InputValidationError("wrapped error")
        chained.__cause__ = original

        assert chained.__cause__ is original
        assert isinstance(chained, InputValidationError)
