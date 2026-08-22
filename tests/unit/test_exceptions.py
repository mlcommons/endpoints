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

"""Tests for custom exceptions.

These tests verify the CLI exception hierarchy and ensure proper error
handling behavior throughout the CLI commands. The exception-based error
handling allows for:
- Testable error conditions (can assert exceptions instead of process exits)
- Composable commands (can be called programmatically)
- Centralized error handling in main()
- Clear error categorization (validation vs setup vs execution)
"""

import copy
import pickle

import pytest
from inference_endpoint.exceptions import (
    CLIError,
    DatasetValidationError,
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


class TestDatasetValidationErrorRoundTrip:
    """DatasetValidationError carries structured fields (reason + detail). Its
    args mirror the constructor signature, so copy/pickle round-trip via
    CPython's default exception machinery without a custom reducer."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "exc",
        [
            DatasetValidationError(
                DatasetValidationError.Reason.INPUT_TOKENS_SHADOWING
            ),
            DatasetValidationError(
                DatasetValidationError.Reason.PROMPT_LIST_UNSUPPORTED, "sample 3"
            ),
        ],
        ids=["reason-only", "reason-and-detail"],
    )
    @pytest.mark.parametrize(
        # pickle round-trips a self-constructed exception (trusted, not external
        # input) — this asserts it reconstructs from reason + detail.
        "clone_fn",
        [copy.copy, copy.deepcopy, lambda e: pickle.loads(pickle.dumps(e))],
        ids=["copy", "deepcopy", "pickle"],
    )
    def test_round_trip_preserves_reason_detail_message(self, exc, clone_fn):
        clone = clone_fn(exc)
        assert clone.reason is exc.reason
        assert clone.detail == exc.detail
        assert str(clone) == str(exc)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "clone_fn",
        [copy.copy, copy.deepcopy, lambda e: pickle.loads(pickle.dumps(e))],
        ids=["copy", "deepcopy", "pickle"],
    )
    def test_round_trip_preserves_notes_and_extra_state(self, clone_fn):
        # CPython's default reducer restores __dict__ as state, so __notes__
        # (PEP 678) and any caller-added attribute survive the round-trip.
        exc = DatasetValidationError(
            DatasetValidationError.Reason.MESSAGES_SHADOWING, "sample 7"
        )
        exc.add_note("diagnostic note")
        exc.extra_attr = "kept"
        clone = clone_fn(exc)
        assert getattr(clone, "__notes__", None) == ["diagnostic note"]
        assert clone.extra_attr == "kept"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "exc, expected",
        [
            (
                DatasetValidationError(
                    DatasetValidationError.Reason.INPUT_TOKENS_SHADOWING
                ),
                DatasetValidationError.Reason.INPUT_TOKENS_SHADOWING.value,
            ),
            (
                DatasetValidationError(
                    DatasetValidationError.Reason.PROMPT_LIST_UNSUPPORTED, "sample 3"
                ),
                f"{DatasetValidationError.Reason.PROMPT_LIST_UNSUPPORTED.value}: "
                f"sample 3",
            ),
        ],
        ids=["reason-only", "reason-and-detail"],
    )
    def test_str_pins_the_user_facing_message(self, exc, expected):
        # __str__ is the sole producer of the message main.py logs via str(e);
        # pin the absolute output so a regression in formatting can't pass on the
        # round-trip test's str(clone) == str(exc) tautology alone.
        assert str(exc) == expected
