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

"""BFCL v4 multi-turn runner.

Orchestrates multi-turn agentic conversations for BFCL accuracy evaluation,
using Endpoints' HTTP client for request dispatch. Each test entry is executed
as a sequential conversation with local function execution between steps.

This runner integrates with the existing accuracy pipeline by producing
per-entry results that the BFCLv4MultiTurnScorer can evaluate.
"""

import logging
import time
from typing import Any

import httpx

from ..dataset_manager.predefined.bfcl_v4.multi_turn import (
    BFCLv4MultiTurnEntry,
)
from .bfcl_v4_execution import BFCLExecutionBridge

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 300.0
DEFAULT_MAX_STEPS_PER_TURN = 25


class BFCLMultiTurnRunner:
    """Runs BFCL multi-turn conversations against an OpenAI-compatible endpoint.

    Uses httpx for synchronous HTTP calls (matching the serial execution model
    required for deterministic accuracy evaluation). Each conversation is
    processed sequentially: send request → parse response → execute functions →
    construct tool messages → send next request → ... until all turns complete.

    This mimics how evalscope's handler processes multi-turn entries but uses
    Endpoints' request format for wire-level consistency.
    """

    def __init__(
        self,
        endpoint_url: str,
        model_name: str,
        api_key: str = "not-needed",
        temperature: float = 0.0,
        seed: int | None = None,
        max_steps_per_turn: int = DEFAULT_MAX_STEPS_PER_TURN,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self._endpoint_url = endpoint_url.rstrip("/")
        if not self._endpoint_url.endswith("/v1/chat/completions"):
            self._endpoint_url += "/v1/chat/completions"
        self._model_name = model_name
        self._api_key = api_key
        self._temperature = temperature
        self._seed = seed
        self._timeout_s = timeout_s
        self._bridge = BFCLExecutionBridge(max_steps_per_turn=max_steps_per_turn)
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self._timeout_s, connect=30.0),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
            )
        return self._client

    def run_entry(self, entry: BFCLv4MultiTurnEntry) -> dict[str, Any]:
        """Execute a single multi-turn conversation and return results.

        Args:
            entry: The BFCL test entry to execute.

        Returns:
            Dict with entry results ready for scoring via multi_turn_checker.
        """
        state = self._bridge.start_conversation(entry)

        # Get initial request for turn 0
        messages, tools = self._bridge.get_initial_request(state)

        total_requests = 0
        t0 = time.monotonic()

        while not state.completed:
            # Send request via HTTP
            response_data = self._send_request(messages, tools)
            total_requests += 1

            if response_data is None:
                # Request failed — mark as force terminated
                state.force_terminated = True
                state.completed = True
                state.model_results_per_turn.append(state.current_turn_results)
                break

            # Parse model response
            tool_calls, tool_call_ids, content = self._parse_response(response_data)

            # Process through execution bridge
            action = self._bridge.process_response(
                state, tool_calls, tool_call_ids, content
            )

            if action.action_type == "continue":
                messages = action.messages
                tools = action.tools
            elif action.action_type == "next_turn":
                messages = action.messages
                tools = action.tools
            elif action.action_type in ("complete", "force_terminated"):
                break

        elapsed = time.monotonic() - t0
        logger.debug(
            "Completed %s: %d requests, %.1fs, %s",
            entry.entry_id,
            total_requests,
            elapsed,
            "force_terminated" if state.force_terminated else "success",
        )

        results = self._bridge.get_results(state)
        results["total_requests"] = total_requests
        results["elapsed_s"] = elapsed
        self._bridge.cleanup(entry.entry_id)
        return results

    def run_all(
        self,
        entries: list[BFCLv4MultiTurnEntry],
        progress_callback: Any = None,
    ) -> list[dict[str, Any]]:
        """Execute all entries sequentially for deterministic evaluation.

        Args:
            entries: List of BFCL multi-turn entries to process.
            progress_callback: Optional callable(entry_idx, total, entry_id) for progress.

        Returns:
            List of per-entry result dicts.
        """
        results = []
        total = len(entries)

        for idx, entry in enumerate(entries):
            if progress_callback:
                progress_callback(idx, total, entry.entry_id)

            result = self.run_entry(entry)
            results.append(result)

        return results

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _send_request(
        self, messages: list[dict], tools: list[dict]
    ) -> dict[str, Any] | None:
        """Send a chat completion request to the endpoint.

        Returns the parsed JSON response or None on failure.
        """
        payload: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": int(self._temperature)
            if self._temperature == 0
            else self._temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self._seed is not None:
            payload["seed"] = self._seed

        try:
            client = self._get_client()
            resp = client.post(self._endpoint_url, json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            logger.error("Request timed out after %.0fs", self._timeout_s)
            return None
        except httpx.HTTPStatusError as exc:
            logger.error(
                "HTTP error %d: %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            return None
        except Exception as exc:
            logger.error("Request failed: %s", exc)
            return None

    def _parse_response(
        self, response_data: dict[str, Any]
    ) -> tuple[list[dict] | None, list[str] | None, str | None]:
        """Parse the OpenAI chat completion response.

        Returns:
            Tuple of (tool_calls, tool_call_ids, content).
            tool_calls is None if no function calls were made.
        """
        try:
            choice = response_data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError) as exc:
            logger.warning("Malformed response: %s", exc)
            return None, None, None

        content = message.get("content")
        raw_tool_calls = message.get("tool_calls")

        if not raw_tool_calls:
            return None, None, content

        tool_calls = []
        tool_call_ids = []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            tool_calls.append(
                {
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", "{}"),
                }
            )
            tool_call_ids.append(tc.get("id", f"call_{len(tool_call_ids)}"))

        return tool_calls, tool_call_ids, content
