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


class DatasetValidationError(InputValidationError):
    """Invalid --dataset string or dataset configuration."""

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
