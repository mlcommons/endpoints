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

# Portions of this file are derived from mini-swe-agent v2.3.0.
# See MINI_SWE_AGENT_LICENSE.md in this directory for the upstream MIT notice.

"""Parse actions & format observations with toolcalls"""

import base64
import json
import shlex
import time

from jinja2 import StrictUndefined, Template
from minisweagent.exceptions import FormatError
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}

FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": (
            "Submit your solution when complete. The patch is automatically extracted "
            "from your git changes — do NOT create patch.txt manually."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "files_modified": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Source file paths you modified, relative to /testbed "
                        "(e.g. ['django/db/models/query.py'])"
                    ),
                }
            },
            "required": ["files_modified"],
        },
    },
}

STR_REPLACE_EDITOR_TOOL = {
    "type": "function",
    "function": {
        "name": "str_replace_editor",
        "description": (
            "View or edit files precisely. Use 'view' to read a file with line numbers, "
            "'str_replace' to replace an exact string (must match exactly once), "
            "'create' to write a new file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "str_replace", "create"],
                },
                "path": {
                    "type": "string",
                    "description": "Absolute file path",
                },
                "old_str": {
                    "type": "string",
                    "description": "Exact string to replace (str_replace only)",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement string (str_replace) or file content (create)",
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Optional [start_line, end_line] for view (1-indexed)",
                },
            },
            "required": ["command", "path"],
        },
    },
}

TOOL_SCHEMAS = [BASH_TOOL, FINISH_TOOL, STR_REPLACE_EDITOR_TOOL]


def parse_toolcall_actions(
    tool_calls: list, *, format_error_template: str
) -> list[dict]:
    """Parse tool calls from the response. Raises FormatError if unknown tool or invalid args."""
    if not tool_calls:
        raise FormatError(
            {
                "role": "user",
                "content": Template(
                    format_error_template, undefined=StrictUndefined
                ).render(
                    error="No tool calls found in the response. Every response MUST include at least one tool call.",
                    actions=[],
                ),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    actions = []
    errors = []
    for tool_call in tool_calls:
        args = {}
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception as e:
            errors.append(f"Error parsing tool call arguments: {e}.")
            continue
        name = tool_call.function.name
        tool_call_id = tool_call.id

        if name == "bash":
            if not isinstance(args, dict) or "command" not in args:
                errors.append("Missing 'command' argument in bash tool call.")
                continue
            actions.append({"command": args["command"], "tool_call_id": tool_call_id})

        elif name == "finish":
            files = args.get("files_modified", [])
            if not isinstance(files, list) or not files:
                errors.append(
                    "finish: files_modified must be a non-empty list of file paths."
                )
                continue

            def _to_rel(p: str) -> str:
                p = p.lstrip("/")
                if p.startswith("testbed/"):
                    p = p[len("testbed/") :]
                return p

            files_str = " ".join(shlex.quote(_to_rel(f)) for f in files)
            actions.append(
                {
                    "command": (
                        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && "
                        f"git -C /testbed add -N -- {files_str} && "
                        f"git -C /testbed diff HEAD -- {files_str}"
                    ),
                    "tool_call_id": tool_call_id,
                }
            )

        elif name == "str_replace_editor":
            command = args.get("command")
            path = args.get("path", "")
            if command == "view":
                view_range = args.get("view_range")
                if view_range is not None:
                    if not isinstance(view_range, list | tuple) or len(view_range) != 2:
                        errors.append(
                            "str_replace_editor view_range must be [start, end]."
                        )
                        continue
                    try:
                        start, end = int(view_range[0]), int(view_range[1])
                    except (TypeError, ValueError):
                        errors.append(
                            "str_replace_editor view_range values must be integers."
                        )
                        continue
                    bash_cmd = (
                        f"awk 'NR>={start} && NR<={end} "
                        f'{{printf "%6d\\t%s\\n", NR, $0}}\' {shlex.quote(path)}'
                    )
                else:
                    bash_cmd = f"cat -n {shlex.quote(path)}"
                actions.append({"command": bash_cmd, "tool_call_id": tool_call_id})
            elif command == "str_replace":
                old_str = args.get("old_str", "")
                new_str = args.get("new_str", "")
                path_b64 = base64.b64encode(path.encode()).decode()
                old_b64 = base64.b64encode(old_str.encode()).decode()
                new_b64 = base64.b64encode(new_str.encode()).decode()
                bash_cmd = (
                    f"python3 -c 'import base64,sys;"
                    f'path=base64.b64decode("{path_b64}").decode();'
                    f'old=base64.b64decode("{old_b64}").decode();'
                    f'new=base64.b64decode("{new_b64}").decode();'
                    f"c=open(path).read();n=c.count(old);"
                    f'sys.exit("Expected exactly 1 match, found "+str(n)) if n!=1 else open(path,"w").write(c.replace(old,new,1))\''
                )
                actions.append({"command": bash_cmd, "tool_call_id": tool_call_id})
            elif command == "create":
                content = args.get("new_str", "")
                path_b64 = base64.b64encode(path.encode()).decode()
                content_b64 = base64.b64encode(content.encode()).decode()
                bash_cmd = (
                    f"python3 -c 'import base64,os;"
                    f'path=base64.b64decode("{path_b64}").decode();'
                    f'os.makedirs(os.path.dirname(path) or ".",exist_ok=True);'
                    f'open(path,"w").write(base64.b64decode("{content_b64}").decode())\''
                )
                actions.append({"command": bash_cmd, "tool_call_id": tool_call_id})
            else:
                errors.append(f"str_replace_editor: unknown command {command!r}.")
                continue

        else:
            errors.append(f"Unknown tool '{name}'.")
            continue

    if errors:
        raise FormatError(
            {
                "role": "user",
                "content": Template(
                    format_error_template, undefined=StrictUndefined
                ).render(actions=actions, error=" ".join(errors).strip()),
                "extra": {"interrupt_type": "FormatError"},
            }
        )
    return actions


def format_toolcall_observation_messages(
    *,
    actions: list[dict],
    outputs: list[dict],
    observation_template: str,
    template_vars: dict | None = None,
    multimodal_regex: str = "",
) -> list[dict]:
    """Format execution outputs into tool result messages."""
    not_executed = {
        "output": "",
        "returncode": -1,
        "exception_info": "action was not executed",
    }
    padded_outputs = outputs + [not_executed] * (len(actions) - len(outputs))
    results = []
    for action, output in zip(actions, padded_outputs, strict=False):
        content = Template(observation_template, undefined=StrictUndefined).render(
            output=output, **(template_vars or {})
        )
        msg = {
            "content": content,
            "extra": {
                "raw_output": output.get("output", ""),
                "returncode": output.get("returncode"),
                "timestamp": time.time(),
                "exception_info": output.get("exception_info"),
                **output.get("extra", {}),
            },
        }
        if "tool_call_id" in action:
            msg["tool_call_id"] = action["tool_call_id"]
            msg["role"] = "tool"
        else:
            msg["role"] = "user"  # human issued commands
        if multimodal_regex:
            msg = expand_multimodal_content(msg, pattern=multimodal_regex)
        results.append(msg)
    return results
