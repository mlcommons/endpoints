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

"""Preset transforms for the BFCL v4 dataset.

BFCL v4 uses pre-formatted messages and tools, so the transform pipeline
is minimal -- it passes through the messages/tools columns directly rather
than formatting a single prompt string.
"""

from inference_endpoint.dataset_manager.transforms import (
    ColumnFilter,
    Transform,
)


def function_calling() -> list[Transform]:
    """Default transform for function-calling evaluation.

    Passes through pre-formatted messages, tools, and tool_choice columns directly.
    The OpenAI adapter will use these to construct the request with tool definitions.

    tool_choice is set to "auto" on each row (see BFCLv4._process_sample). Sending
    it explicitly avoids server-default ambiguity: some servers stall when tools are
    present but tool_choice is omitted.
    """
    return [
        ColumnFilter(
            required_columns=["messages", "tools", "tool_choice"],
        ),
    ]
