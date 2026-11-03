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

"""mini-swe-agent model extension for the Qwen SWE-bench tool contract."""

import litellm
from minisweagent.models.litellm_model import LitellmModel

from .qwen_tools import (
    TOOL_SCHEMAS,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)


class QwenToolsModel(LitellmModel):
    """Expose the Qwen tool behavior through mini-swe-agent's model hook."""

    def _query(self, messages: list[dict[str, str]], **kwargs):
        return litellm.completion(
            model=self.config.model_name,
            messages=messages,
            tools=TOOL_SCHEMAS,
            **(self.config.model_kwargs | kwargs),
        )

    def _parse_actions(self, response) -> list[dict]:
        tool_calls = response.choices[0].message.tool_calls or []
        return parse_toolcall_actions(
            tool_calls,
            format_error_template=self.config.format_error_template,
        )

    def format_observation_messages(
        self,
        message: dict,
        outputs: list[dict],
        template_vars: dict | None = None,
    ) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )
