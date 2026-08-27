# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-dispatch gates on the inference endpoints.

A gate proves, before a single instance is dispatched, that the endpoints under
test can actually do the thing the benchmark requires. Gates fail closed: an
endpoint that cannot be identified or reached is a failure, never a pass.

THE SCALE RULE. Every gate must first prove it is testing at the scale it
claims, via :meth:`Gate.assert_scale`, and a scale failure is a *gate failure*,
not a skip. This is not defensive programming; it is the most expensive lesson
in this codebase's history. A tool-call gate that exercised exactly the right
operation with a 278-token prompt passed cleanly while every prompt above 2000
tokens silently returned an empty completion -- and SWE-bench prompts are all
far larger than 2000 tokens. The gate was green and the run scored zero. A gate
that cannot prove its scale is not a gate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib import error as urllib_error
from urllib import request as urllib_request

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 60.0
#: SWE-bench prompts are far larger than this; the threshold is a floor, not a
#: target.
DEFAULT_MIN_PROMPT_TOKENS = 2000


class GateFailure(RuntimeError):
    """A gate refused to let the run start."""


class GateScaleError(GateFailure):
    """A gate could not prove it was testing at the scale it claims."""


@dataclass(slots=True)
class GateReport:
    name: str
    passed: bool
    checked: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        head = (
            f"{self.name}: {'pass' if self.passed else 'FAIL'} ({self.checked} checked)"
        )
        detail = "".join(
            f"\n    {target} -> {reason}" for target, reason in self.failures[:8]
        )
        notes = "".join(f"\n    note: {note}" for note in self.notes)
        return head + detail + notes


class Gate(Protocol):
    name: str

    def assert_scale(self, targets: list[str]) -> None:
        """Prove this gate tests what it claims. Raise :class:`GateScaleError`."""
        pass

    def check(self, targets: list[str]) -> GateReport:
        pass


def _http_json(
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    api_key: str | None = None,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib_request.Request(url, data=data, headers=headers)
    with urllib_request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read())


def run_gates(gates: list[Gate], targets: list[str]) -> list[GateReport]:
    """Run every gate; raise :class:`GateFailure` if any refused.

    Every gate runs even after one fails, so one preflight reports every problem
    rather than sending the operator round the loop once per endpoint.
    """
    reports: list[GateReport] = []
    for gate in gates:
        try:
            gate.assert_scale(targets)
        except GateScaleError as exc:
            reports.append(
                GateReport(
                    name=gate.name,
                    passed=False,
                    failures=[("<scale>", str(exc))],
                    notes=[
                        "a gate that cannot prove its scale is a failing gate, "
                        "not a skipped one"
                    ],
                )
            )
            continue
        reports.append(gate.check(targets))

    failed = [report for report in reports if not report.passed]
    if failed:
        raise GateFailure(
            "pre-dispatch gate(s) refused:\n"
            + "\n".join(report.summary() for report in failed)
        )
    return reports


class CheckpointIdentityGate:
    """Every endpoint must serve exactly the expected checkpoint.

    Two traps, both of which produced silently contaminated results:

    1. ``/v1/models`` echoes ``--served-model-name``, which operators routinely
       set identically for two different checkpoints (e.g. an FP8 and a BF16
       build of the same model). ``/get_model_info`` reports the real model
       path, so it is tried first.
    2. Checkpoint names nest: ``Org/Model`` is a strict prefix of
       ``Org/Model-FP8``. Any ``startswith``/``in`` test therefore accepts an
       FP8 endpoint as BF16. Comparison is ``==`` and nothing else.
    """

    name = "checkpoint_identity"

    def __init__(
        self,
        expected_model: str,
        *,
        timeout_s: float = 10.0,
        api_key: str | None = None,
    ) -> None:
        if not expected_model:
            raise ValueError("expected_model is required")
        self.expected_model = expected_model
        self.timeout_s = timeout_s
        self.api_key = api_key

    def assert_scale(self, targets: list[str]) -> None:
        if not targets:
            raise GateScaleError("no endpoints to identify")

    def probe(self, url: str) -> tuple[str | None, str]:
        base = url.rstrip("/")
        try:
            info = _http_json(
                f"{base}/get_model_info",
                timeout_s=self.timeout_s,
                api_key=self.api_key,
            )
            model_path = info.get("model_path")
            if model_path:
                return str(model_path), "get_model_info"
        except (urllib_error.URLError, OSError, ValueError, TimeoutError):
            pass  # not an SGLang endpoint; fall through to the OpenAI route
        try:
            listing = _http_json(
                f"{base}/v1/models", timeout_s=self.timeout_s, api_key=self.api_key
            )
        except (urllib_error.URLError, OSError, ValueError, TimeoutError) as exc:
            return None, f"unreachable: {type(exc).__name__}"
        ids = [
            entry.get("id") for entry in listing.get("data") or [] if entry.get("id")
        ]
        if len(ids) == 1:
            return str(ids[0]), "v1/models"
        if len(ids) > 1:
            return None, f"ambiguous /v1/models: {ids!r}"
        return None, "no model id from either endpoint"

    def check(self, targets: list[str]) -> GateReport:
        report = GateReport(name=self.name, passed=True, checked=len(targets))
        sources: set[str] = set()
        for url in targets:
            identity, source = self.probe(url)
            if identity is None:
                report.failures.append((url, source))
            elif identity != self.expected_model:  # EXACT; never startswith/in
                report.failures.append(
                    (url, f"serves {identity!r}, expected {self.expected_model!r}")
                )
            else:
                sources.add(source)
        if "v1/models" in sources:
            report.notes.append(
                "identity came from /v1/models, which echoes --served-model-name; "
                "that string can be identical across checkpoints, so it cannot "
                "separate two builds that share a served name"
            )
        report.passed = not report.failures
        report.data["expected_model"] = self.expected_model
        return report


class ToolCallGate:
    """Every endpoint must return a well-formed tool call at SWE-bench scale.

    The prompt is measured with the *server's own* tokenizer (``/tokenize``),
    never estimated from character count, and a prompt that measures below
    ``min_prompt_tokens`` fails the scale assertion rather than passing the
    gate.
    """

    name = "tool_call"

    def __init__(
        self,
        model: str,
        *,
        min_prompt_tokens: int = DEFAULT_MIN_PROMPT_TOKENS,
        prompt: str | None = None,
        timeout_s: float = 180.0,
        api_key: str | None = None,
        tool_name: str = "bash",
    ) -> None:
        self.model = model
        self.min_prompt_tokens = min_prompt_tokens
        self.prompt = prompt if prompt is not None else build_scale_prompt()
        self.timeout_s = timeout_s
        self.api_key = api_key
        self.tool_name = tool_name
        self._measured: dict[str, int] = {}

    @property
    def tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": self.tool_name,
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]

    def count_tokens(self, url: str) -> int | None:
        try:
            response = _http_json(
                f"{url.rstrip('/')}/tokenize",
                {"model": self.model, "prompt": self.prompt},
                timeout_s=self.timeout_s,
                api_key=self.api_key,
            )
        except (urllib_error.URLError, OSError, ValueError, TimeoutError):
            return None
        count = response.get("count")
        if count is None:
            tokens = response.get("tokens")
            count = len(tokens) if isinstance(tokens, list) else None
        try:
            return int(count) if count is not None else None
        except (TypeError, ValueError):
            return None

    def assert_scale(self, targets: list[str]) -> None:
        if not targets:
            raise GateScaleError("no endpoints to gate")
        measured = False
        for url in targets:
            count = self.count_tokens(url)
            if count is None:
                continue
            self._measured[url] = count
            measured = True
            if count < self.min_prompt_tokens:
                raise GateScaleError(
                    f"{url}: gate prompt measures {count} tokens, below the "
                    f"{self.min_prompt_tokens}-token floor this gate claims to "
                    "test. A tool-call gate that passes at a small prompt says "
                    "nothing about SWE-bench-sized prompts."
                )
        if not measured and self.min_prompt_tokens > 0:
            raise GateScaleError(
                "no endpoint exposed /tokenize, so the gate cannot prove the "
                f"prompt reaches {self.min_prompt_tokens} tokens. Serve a "
                "tokenizer endpoint or set min_prompt_tokens=0 to accept an "
                "unverified prompt size."
            )

    def check(self, targets: list[str]) -> GateReport:
        report = GateReport(name=self.name, passed=True, checked=len(targets))
        for url in targets:
            tokens = self._measured.get(url)
            try:
                response = _http_json(
                    f"{url.rstrip('/')}/v1/chat/completions",
                    {
                        "model": self.model,
                        "messages": [{"role": "user", "content": self.prompt}],
                        "tools": self.tools,
                        "tool_choice": "auto",
                        "max_tokens": 256,
                        "temperature": 0.0,
                    },
                    timeout_s=self.timeout_s,
                    api_key=self.api_key,
                )
            except (urllib_error.URLError, OSError, ValueError, TimeoutError) as exc:
                report.failures.append((url, f"{type(exc).__name__}: {exc}"))
                continue
            failure = self._validate(response)
            if failure is not None:
                report.failures.append((url, f"tokens={tokens}: {failure}"))
        report.passed = not report.failures
        report.data["measured_tokens"] = dict(self._measured)
        return report

    def _validate(self, response: dict[str, Any]) -> str | None:
        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return "malformed chat completion response"
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            content = (message.get("content") or "")[:120]
            return f"no tool_calls; content={content!r}"
        function = tool_calls[0].get("function") or {}
        if function.get("name") != self.tool_name:
            return f"wrong tool {function.get('name')!r}"
        try:
            arguments = json.loads(function.get("arguments") or "")
        except (TypeError, ValueError):
            return f"arguments are not valid JSON: {function.get('arguments')!r}"
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return f"malformed arguments {function.get('arguments')!r}"
        return None


def build_scale_prompt(repetitions: int = 120) -> str:
    """A prompt long enough to exercise the large-context path."""
    filler = "\n".join(
        f"def helper_{index}(path, flags=None):\n"
        f"    # legacy shim retained for compatibility with the v{index} api\n"
        "    result = compute_checksum(path, flags or DEFAULT_FLAGS)\n"
        "    return normalise(result), path, flags\n"
        for index in range(repetitions)
    )
    return (
        "You are working in a Python repository checked out at /testbed.\n"
        "Below is the current content of /testbed/legacy/helpers.py.\n\n"
        "<file>\n" + filler + "</file>\n\n"
        "Before proposing any change you must inspect the repository.\n"
        "List the files in /testbed using the shell tool. Call the tool; do not "
        "answer in prose."
    )


#: Response fields that change on every request and carry no checkpoint
#: identity. vLLM's ``/v1/models`` stamps ``created`` with the request time and
#: mints a fresh ``permission[].id`` per call, so hashing the raw payload makes
#: the fingerprint differ between any two reads of a perfectly healthy engine.
#: The dispatcher compares the claim-time and publish-time fingerprints and
#: treats a difference as ``endpoint_changed`` -- an infrastructure fault -- so
#: an unstable fingerprint retries and then abandons every unit, and the merge
#: gate can never produce a number.
_VOLATILE_IDENTITY_KEYS = frozenset({"created", "created_at", "permission"})


def _strip_volatile(value: Any) -> Any:
    """Drop per-request fields so a fingerprint reflects identity, not time."""
    if isinstance(value, dict):
        return {
            key: _strip_volatile(item)
            for key, item in value.items()
            if key not in _VOLATILE_IDENTITY_KEYS
        }
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


class EndpointFingerprintGate:
    """Record a per-endpoint fingerprint for later comparison.

    An engine restarted under a live client yields a run that scores near zero
    and still exits successfully -- nothing in the result distinguishes it from
    a genuinely bad model. The dispatcher therefore records each endpoint's
    fingerprint when a unit is claimed and re-reads it when the unit is
    published; a change means the unit was scored against something other than
    what it was dispatched to, and the unit is requeued rather than counted.
    """

    name = "endpoint_fingerprint"

    def __init__(self, *, timeout_s: float = 10.0, api_key: str | None = None) -> None:
        self.timeout_s = timeout_s
        self.api_key = api_key
        self.fingerprints: dict[str, str] = {}

    def assert_scale(self, targets: list[str]) -> None:
        if not targets:
            raise GateScaleError("no endpoints to fingerprint")

    def fingerprint(self, url: str) -> str | None:
        base = url.rstrip("/")
        parts: list[str] = []
        for path in ("/get_model_info", "/v1/models"):
            try:
                payload = _http_json(
                    base + path, timeout_s=self.timeout_s, api_key=self.api_key
                )
            except (urllib_error.URLError, OSError, ValueError, TimeoutError):
                continue
            parts.append(
                json.dumps(_strip_volatile(payload), sort_keys=True, default=str)
            )
        if not parts:
            return None
        import hashlib

        return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]

    def check(self, targets: list[str]) -> GateReport:
        report = GateReport(name=self.name, passed=True, checked=len(targets))
        for url in targets:
            value = self.fingerprint(url)
            if value is None:
                report.failures.append(
                    (url, "could not read an identity to fingerprint")
                )
                continue
            self.fingerprints[url] = value
        report.passed = not report.failures
        report.data["fingerprints"] = dict(self.fingerprints)
        return report
