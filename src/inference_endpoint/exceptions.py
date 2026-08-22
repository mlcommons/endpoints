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

from enum import Enum


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
    """A loaded dataset sample fails salt validation.

    The failure category is a ``Reason``; ``detail`` carries the specifics
    (offending sample index, remediation hint).
    """

    class Reason(Enum):
        """Why a sample cannot be salted."""

        TYPE_MISMATCH = "sample is not a dict"
        INPUT_TOKENS_SHADOWING = (
            "sample has 'input_tokens'; salt cannot bust a pre-tokenized cache"
        )
        MESSAGES_SHADOWING = (
            "sample has 'messages'; adapters send that verbatim and prefer it "
            "over 'prompt', so a salted 'prompt' would never reach the server"
        )
        PROMPT_MISSING = "sample has no 'prompt' field"
        PROMPT_TYPE_MISMATCH = "sample 'prompt' is not a str"
        PROMPT_LIST_UNSUPPORTED = (
            "sample 'prompt' is a list (OpenAI batch / token-IDs, or this "
            "project's multimodal content parts); salt supports only a single "
            "text 'prompt'"
        )

    def __init__(
        self,
        reason: "DatasetValidationError.Reason",
        detail: str | None = None,
    ) -> None:
        self.reason = reason
        self.detail = detail
        # args mirrors the constructor signature, so CPython's default exception
        # reducer round-trips copy/pickle (and __notes__) with no override.
        super().__init__(reason, detail)

    def __str__(self) -> str:
        if self.detail is None:
            return self.reason.value
        return f"{self.reason.value}: {self.detail}"


class DatasetParseError(InputValidationError):
    """A --dataset CLI string could not be parsed into a dataset config.

    Distinct from DatasetValidationError: the failure is in the user's input
    string (bad format / key=value), before any dataset is loaded — so there is
    no sample index or Reason, just the underlying parse message.
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
