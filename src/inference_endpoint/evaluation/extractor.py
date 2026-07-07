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
# See the License for the specific permissions and
# limitations under the License.


import ast
import inspect
import json
import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar


class Extractor(ABC):
    """An Extractor is used to extract phrases or substrings from the model's outputs using
    multiple regex patterns with a priority system. This is useful for extracting values from
    strings with the same general format but small variations, such as a model outputting a
    numeric value plain or inside a LaTeX block.
    """

    # Provide a registration and lookup system for derived Extractor classes by name.
    # This allows registering new extractors that can be instantiated via config/lookup.
    PREDEFINED: ClassVar[dict[str, type["Extractor"]]] = {}

    EXTRACTOR_ID: ClassVar[str]
    """The unique identifier for the extractor. Automatically set by __init_subclass__."""

    def __init_subclass__(
        cls,
        extractor_id: str | None = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)

        if not inspect.isabstract(cls):
            if extractor_id is None:
                extractor_id = cls.__name__
            cls.EXTRACTOR_ID = extractor_id
            Extractor.PREDEFINED[extractor_id] = cls

    @classmethod
    def get(cls, name: str) -> type["Extractor"]:
        """Look up an Extractor subclass by its registered name.

        Args:
            name: str, the registered extractor name

        Returns:
            Extractor subclass

        Raises:
            KeyError: If no extractor with the given name is found
        """
        try:
            return Extractor.PREDEFINED[name]
        except KeyError as e:
            raise KeyError(
                f"Extractor '{name}' is not registered - available extractors: {Extractor.available_extractors()}"
            ) from e

    @classmethod
    def available_extractors(cls) -> list[str]:
        """Return the list of registered extractor names."""
        return list(Extractor.PREDEFINED.keys())

    @classmethod
    @abstractmethod
    def extract(cls, text: str, default: str | None = None) -> str | None:
        """Extract value from text.

        Args:
            text: The text to extract from
            default: Default value to return if extraction fails (instead of None)

        Returns:
            Extracted value, or default if extraction fails, or None if no default provided
        """
        raise NotImplementedError


class ABCDExtractor(Extractor, extractor_id="abcd_extractor"):
    """Extract ABCD multiple choice answer from response text.
    Based on OpenAI's extract_abcd function from GPT-OSS.
    Reference: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/abcd_grader.py
    Scans text (with Markdown/LaTeX wrappers intact) for ABCD MCQ answers, and returns
    the corresponding 'choice' key (i.e. "choice1" for 'A' and "choice4" for 'D').
    If no answer is found, returns an empty string.

    The function tries multiple patterns in priority order, selecting the match
    with the highest priority and shortest length if multiple matches found.

    Args:
        text: Response text containing ABCD answer
    Returns:
        "choice" key (see GQPA dataset columns) or empty string if no answer is found.
    Examples:
        >>> ABCDExtractor.extract("The answer is B")
        'choice2'
        >>> ABCDExtractor.extract("**Answer:** C")
        'choice3'
        >>> ABCDExtractor.extract("\\\\boxed{D}")
        'choice4'
    """

    PATTERNS = [
        # 0) "**Answer:** A" or "*Answers* – B", i.e. markdown-wrapped "Answer(s)" with an unwrapped letter.
        re.compile(
            r"""(?ix)                   # case-insensitive, ignore-space
            (?:\*{1,2}|_{1,2})          # leading *…*  or _…_
            Answer[s]?                  #   Answer or Answers
            \s*[:\-–]?                  #   optional separator
            (?:\*{1,2}|_{1,2})          # closing wrapper
            \s*                         # optional space
            ([ABCD])\b                  # the actual letter
            """,
            re.X,
        ),
        # 0.1) Answer with optional markdown and colons
        re.compile(
            r"""(?ix)           # ignore case, allow verbose mode
            ^\s*                      # optional leading whitespace
            (?:\*{1,2}|_{1,2})?       # optional markdown wrapper
            Answer:?                   # the word 'answer' with an optional colon
            (?:\*{1,2}|_{1,2})?       # optional markdown wrapper again
            \s*:?\s*                  # optional colon with optional spaces
            (?:\*{1,2}|_{1,2})?       # optional markdown wrapper before letter
            ([ABCD])                 # capture the letter
            (?:\*{1,2}|_{1,2})?       # optional markdown wrapper after letter
            \s*                     # optional trailing whitespace, end of line
        """,
            re.MULTILINE,
        ),
        # 1) Answer: (C)   or   Answers: (B)
        re.compile(r"(?ix)\bAnswer[s]?\b\s*[:\-–]?\s*\(\s*([ABCD])\s*\)"),
        # 2) Answer: C    or   Answers – D
        re.compile(r"(?ix)\bAnswer[s]?\b\s*[:\-–]?\s*([ABCD])\b"),
        # 3) Option B   or   Choice: C
        re.compile(r"(?ix)\b(?:Option|Choice)\b\s*[:\-–]?\s*([ABCD])\b"),
        # 7) LaTeX \boxed{...A...}, catches both \boxed{A} and
        #    \boxed{\text{A } 2.08\times10^{-6}\,\mathrm{m}} etc.
        re.compile(r"(?x)\\boxed\{[^}]*?([ABCD])[^}]*\}", re.MULTILINE),
        # 7.5) LaTeX \boxed{\textbf{...C...}}
        re.compile(
            r"(?x)\\boxed\{[^}]*?\\textbf\{[^}]*?([ABCD])[^}]*\}[^}]*\}", re.MULTILINE
        ),
        # 7.51) LaTeX \boxed{\text{...C...}}
        re.compile(
            r"(?x)\\boxed\{[^}]*?\\text\{[^}]*?([ABCD])[^}]*\}[^}]*\}", re.MULTILINE
        ),
        # 4) bare singletons:  (A)  [B]
        re.compile(r"(?x)(?<![A-Za-z0-9])[\(\[]\s*([ABCD])\s*[\)\]](?![A-Za-z0-9])"),
        # 5) Markdown-wrapped: *A*  **B**  _C_  __D__
        re.compile(
            r"(?x)(?<![A-Za-z0-9])(?:\*{1,2}|_{1,2})([ABCD])(?:\*{1,2}|_{1,2})(?![A-Za-z0-9])"
        ),
        # 6) LaTeX \textbf{...C...}
        re.compile(r"(?x)\\textbf\{[^}]*?([ABCD])[^}]*\}"),
        # 8) markdown-wrapped answer plus ")" plus description, e.g. **D) …**
        re.compile(r"""(?x)                        # ignore whitespace in pattern
            (?<![A-Za-z0-9])            # not preceded by word-char
            (?:\*{1,2}|_{1,2})          # opening ** or __ or * or _
            \s*([ABCD])\)               # capture letter plus ")"
            [^*_\n]+?                   # some text inside wrapper
            (?:\*{1,2}|_{1,2})          # closing wrapper
            (?![A-Za-z0-9])             # not followed by word-char
        """),
        # 9) final fallback: a line that's exactly "A", "B.", "C)", "**D**", etc.
        re.compile(
            r"""(?x)^\s*
            (?:\*{1,2}|_{1,2})?     # optional markdown wrapper
            ([ABCD])                # capture group for letter
            (?:\*{1,2}|_{1,2})?     # optional closing markdown
            \s*[\.\)\-–:]?          # optional separator after the letter
            \s*.*$                  # allow any following text
        """,
            re.MULTILINE,
        ),
    ]

    @classmethod
    def extract(cls, text: str, default: str | None = None) -> str | None:
        matches = []
        for prio, pat in enumerate(cls.PATTERNS):
            m = pat.search(text)
            if m:
                letter = m.group(1).upper()
                if letter in "ABCD":
                    matches.append((prio, m, letter))

        # Sort by priority (lower is better) and then by match length (shorter is better)
        matches.sort(key=lambda triple: (triple[0], len(triple[1].group(0))))

        choice_map = {
            "A": "choice1",
            "B": "choice2",
            "C": "choice3",
            "D": "choice4",
        }

        # Return the best match
        for _, _, letter in matches:
            return choice_map[letter]

        # Final fallback from OpenAI: take first character after stripping markdown
        # This is a last resort if no patterns matched
        stripped = text.removeprefix("**")
        if stripped and stripped[0].upper() in "ABCD":
            abcd_choice = stripped[0].upper()
            return choice_map[abcd_choice]

        return default if default is not None else ""


class BoxedMathExtractor(Extractor, extractor_id="boxed_math_extractor"):
    """Extract boxed math answer from response text.
    Based on OpenAI's extract_boxed_math function from GPT-OSS.
    https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/aime_eval.py
    """

    @classmethod
    def extract(cls, text: str, default: str | None = None) -> str | None:
        pattern = r"boxed{(.*?)}|framebox{(.*?)}"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            for match in matches[::-1]:
                for group in match:
                    if group != "":
                        retval = group.split(",")[-1].strip()
                        return retval
        pattern = r"\d+"  # get the last integer if no pattern found
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1]
        return default


class IdentityExtractor(Extractor, extractor_id="identity_extractor"):
    """Extract identity answer from response text."""

    @classmethod
    def extract(cls, text: str, _: str | None = None) -> str | None:
        return text


class PythonCodeExtractor(Extractor, extractor_id="python_code_extractor"):
    """Extract Python code from markdown code blocks.
    Based on parse_code function from GPT-OSS livecodebench_eval.py.
    Reference: https://github.com/openai/gpt-oss/blob/main/gpt_oss/evals/livecodebench_eval.py

    Extracts Python code from ```python or ``` code blocks.
    Priority:
    1. Last ```python block
    2. Last plain ``` block

    Args:
        text: Response text containing code blocks
        default: Default value to return if extraction fails (instead of None)

    Returns:
        Extracted Python code, or default if provided and extraction fails, or None otherwise

    Examples:
        >>> PythonCodeExtractor.extract("```python\\ndef foo():\\n    pass\\n```")
        'def foo():\\n    pass'
        >>> PythonCodeExtractor.extract("```\\nprint('hello')\\n```")
        "print('hello')"
        >>> PythonCodeExtractor.extract("no code here", default="# FAILED")
        '# FAILED'
    """

    @classmethod
    def extract(cls, text: str, default: str | None = None) -> str | None:
        if not text or not isinstance(text, str):
            return default

        text = text.strip()
        if not text:
            return default

        # Try ```python blocks first (most specific)
        python_matches = list(re.finditer(r"```python(.*?)```", text, re.DOTALL))
        if python_matches:
            return python_matches[-1].group(1).strip()

        # Fall back to plain ``` blocks
        plain_matches = list(re.finditer(r"```(.*?)```", text, re.DOTALL))
        if plain_matches:
            # Get the last match
            code = plain_matches[-1].group(1).strip()
            # Remove language tag if present (e.g., ```python\n or ```py\n)
            code = re.sub(r"^(?:python|py)\s*\n", "", code, flags=re.IGNORECASE)
            return code

        return default


class FunctionCallExtractor(Extractor, extractor_id="function_call_extractor"):
    """Extract function call(s) from model output for BFCL evaluation.

    Extraction priority (highest to lowest fidelity):
    1. Native tool_calls JSON (serialized by the adapter when the endpoint returns
       structured tool_calls -- this is the most reliable path)
    2. JSON array of {"name": "...", "arguments": {...}} objects
    3. Text-based function calls (LAST RESORT -- regex heuristic, may produce
       false positives from explanatory text or fail on nested parentheses)

    The extractor normalizes all formats into a JSON string representation
    suitable for comparison by the bfcl-eval scoring library.

    Examples:
        >>> # Native tool_calls (serialized by adapter) -- preferred path
        >>> text = '[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\\"city\\":\\"London\\"}"}}]'
        >>> FunctionCallExtractor.extract(text)
        '[{"name": "get_weather", "arguments": {"city": "London"}}]'

        >>> # Text-based function call (last resort fallback)
        >>> text = "get_weather(city='London')"
        >>> FunctionCallExtractor.extract(text)
        '[{"name": "get_weather", "arguments": {"city": "London"}}]'
    """

    _SKIP_NAMES = frozenset(
        {
            "print",
            "return",
            "if",
            "for",
            "while",
            "def",
            "class",
            "import",
            "from",
            "with",
            "assert",
            "raise",
            "try",
            "except",
            "len",
            "str",
            "int",
            "float",
            "list",
            "dict",
            "set",
            "tuple",
            "type",
            "isinstance",
            "range",
            "enumerate",
            "zip",
            "map",
            "filter",
            "Note",
            "Example",
            "Here",
            "This",
            "The",
            "For",
        }
    )

    _FUNC_START_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

    @classmethod
    def extract(cls, text: str, default: str | None = None) -> str | None:
        """Extract function calls from model output text.

        Args:
            text: Raw model output (either serialized tool_calls JSON or text)
            default: Default value if extraction fails

        Returns:
            JSON string of normalized function calls, or default on failure
        """
        if not text or not isinstance(text, str):
            return default

        text = text.strip()
        if not text:
            return default

        # Priority 1: Native tool_calls JSON (from adapter serialization)
        result = cls._try_parse_tool_calls_json(text)
        if result is not None:
            return result

        # Priority 2: JSON array of function call objects
        result = cls._try_parse_function_array(text)
        if result is not None:
            return result

        # Priority 3 (last resort): Text-based function calls.
        # This path is inherently heuristic and may not handle all edge cases
        # (e.g., deeply nested parentheses, function-like text in explanations).
        result = cls._try_parse_text_function_calls(text)
        if result is not None:
            return result

        return default

    @classmethod
    def has_native_tool_calls(cls, text: str) -> bool:
        """Whether ``text`` parses as serialized OpenAI ``tool_calls`` JSON.

        Stable public entry point for consumers (e.g. hallucination scoring)
        that only need to know whether the model emitted a structured tool call,
        without reaching into the extractor's private parsing helpers.
        """
        return cls._try_parse_tool_calls_json(text) is not None

    @classmethod
    def _try_parse_tool_calls_json(cls, text: str) -> str | None:
        """Parse serialized OpenAI tool_calls format.

        Expected format: [{"id":"...","type":"function","function":{"name":"...","arguments":"..."}}]
        """
        try:
            data = json.loads(text)
            if not isinstance(data, list):
                return None

            calls: list[dict[str, Any]] = []
            for item in data:
                if not isinstance(item, dict):
                    return None
                func_data = item.get("function")
                if not isinstance(func_data, dict):
                    return None

                name = func_data.get("name", "")
                arguments = func_data.get("arguments", "{}")

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except (json.JSONDecodeError, ValueError):
                        pass

                calls.append({"name": name, "arguments": arguments})

            if calls:
                return json.dumps(calls)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return None

    @classmethod
    def _try_parse_function_array(cls, text: str) -> str | None:
        """Parse a JSON array of function call dicts: [{"name": "...", "arguments": {...}}]"""
        try:
            data = json.loads(text)
            if not isinstance(data, list):
                return None

            calls: list[dict[str, Any]] = []
            for item in data:
                if not isinstance(item, dict):
                    return None
                if "name" not in item:
                    return None
                calls.append(
                    {
                        "name": item["name"],
                        "arguments": item.get("arguments", {}),
                    }
                )

            if calls:
                return json.dumps(calls)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return None

    @classmethod
    def _find_balanced_parens(cls, text: str, start: int) -> int | None:
        """Find the closing paren matching the opening paren at `start`.

        Returns the index of the matching ')' or None if unbalanced.
        Handles nested parentheses, strings (single/double quotes), and brackets.
        """
        depth = 0
        i = start
        in_single_quote = False
        in_double_quote = False
        length = len(text)

        while i < length:
            ch = text[i]
            if ch == "\\" and (in_single_quote or in_double_quote):
                i += 2
                continue
            if ch == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif ch == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif not in_single_quote and not in_double_quote:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        return i
            i += 1
        return None

    @classmethod
    def _try_parse_text_function_calls(cls, text: str) -> str | None:
        """Parse text-based function calls like func_name(arg1='val1', arg2=42).

        Uses balanced parenthesis matching to handle nested calls.
        Applies heuristics to reject likely false positives from natural language.
        Tracks consumed ranges to avoid extracting inner calls that are actually
        arguments of an outer call.
        """
        calls: list[dict[str, Any]] = []
        consumed_ranges: list[tuple[int, int]] = []

        for match in cls._FUNC_START_PATTERN.finditer(text):
            match_start = match.start()

            if any(s <= match_start < e for s, e in consumed_ranges):
                continue

            func_name = match.group(1)
            if func_name in cls._SKIP_NAMES:
                continue

            paren_open = match.end() - 1
            paren_close = cls._find_balanced_parens(text, paren_open)
            if paren_close is None:
                continue

            consumed_ranges.append((paren_open, paren_close + 1))

            args_str = text[paren_open + 1 : paren_close].strip()

            if args_str and cls._looks_like_signature(args_str):
                continue

            arguments = cls._parse_arguments_string(args_str)
            calls.append({"name": func_name, "arguments": arguments})

        if calls:
            return json.dumps(calls)
        return None

    @classmethod
    def _looks_like_signature(cls, args_str: str) -> bool:
        """Return True if args look like a type signature rather than real arguments.

        Bare identifiers without '=' or literal values (e.g., 'city', 'x, y')
        suggest a declaration or documentation reference, not an actual invocation.
        """
        parts = [p.strip() for p in args_str.split(",")]
        return all(re.fullmatch(r"[A-Za-z_]\w*(?:\s*:\s*\w+)?", p) for p in parts if p)

    @classmethod
    def _parse_arguments_string(cls, args_str: str) -> dict[str, Any]:
        """Parse a function arguments string into a dict.

        Handles: key='value', key=123, key=True, key=None, key=[1,2,3]
        Falls back to returning the raw string if parsing fails.
        """
        if not args_str:
            return {}

        arguments: dict[str, Any] = {}
        try:
            fake_dict = f"dict({args_str})"
            tree = ast.parse(fake_dict, mode="eval")
            if isinstance(tree.body, ast.Call):
                for kw in tree.body.keywords:
                    if kw.arg is not None:
                        try:
                            arguments[kw.arg] = ast.literal_eval(kw.value)
                        except (ValueError, SyntaxError):
                            arguments[kw.arg] = ast.unparse(kw.value)
        except (SyntaxError, ValueError):
            arguments = {"_raw": args_str}

        return arguments
