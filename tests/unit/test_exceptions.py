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
