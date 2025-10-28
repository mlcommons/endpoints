"""Custom exceptions for CLI error handling."""


class CLIError(Exception):
    """Base exception for CLI errors.

    All CLI commands should raise CLIError subclasses instead of calling sys.exit().
    The main() function catches these and exits with appropriate codes.
    """

    pass


class InputValidationError(CLIError):
    """Input validation error.

    Raised when user input is invalid (missing required args, invalid config, etc.).
    These are user errors that should be caught before execution starts.
    """

    pass


class SetupError(CLIError):
    """Error during initialization/setup.

    Raised when setup fails (dataset loading, connection failed, etc.).
    These occur during initialization phase before main execution.
    """

    pass


class ExecutionError(CLIError):
    """Error during benchmark/command execution.

    Raised when execution fails after setup completed successfully.
    """

    pass
