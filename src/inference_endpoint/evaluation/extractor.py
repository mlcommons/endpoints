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


import inspect
import re
from abc import ABC, abstractmethod
from typing import ClassVar


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

    CHOICE_MAP = {
        "A": "choice1",
        "B": "choice2",
        "C": "choice3",
        "D": "choice4",
    }

    # Each pattern below captures the answer letter as group 1.
    FINAL_ANSWER_PATTERNS = [
        # JSON-ish final responses.
        # Examples: {"answer": "A"}, 'answer': 'C'
        re.compile(r"""(?is)["']answer["']\s*:\s*["']?\s*([ABCD])\b"""),
        # Explicit final-answer statements.
        # Examples: "Final answer: B", "answer is (D)", "Answer = **C**"
        re.compile(
            r"""(?ix)
            \b(?:final\s+answer|answer)\b
            \s*(?:is|:|=)?\s*
            (?:\\boxed\{\s*)?
            (?:\*{1,2}|_{1,2})?
            \(?\s*([ABCD])\b
            """
        ),
        # Explicit option/choice statements near the end of the response.
        # Examples: "option C", "Choice: (A)", "the correct choice is **B**"
        re.compile(
            r"""(?ix)
            \b(?:option|choice)\b
            \s*(?:is|:|=)?\s*
            (?:\*{1,2}|_{1,2})?
            \(?\s*([ABCD])\b
            """
        ),
        # Boxed answers.
        # Examples: "\\boxed{D}", "\\boxed{\\text{A}}", "\\boxed{\\textbf{C}}"
        re.compile(
            r"""(?is)\\boxed\{\s*(?:\\(?:text|textbf)\{\s*)?([ABCD])\b"""
        ),
        # A final standalone line such as "C", "**D**", or "(B)".
        # Examples: final line "A", final line "**D**", final line "(B)"
        re.compile(
            r"""(?im)^\s*
            (?:\*{1,2}|_{1,2})?
            \(?\s*([ABCD])\s*\)?
            (?:\*{1,2}|_{1,2})?
            \s*[\.\)]?\s*$
            """
        ),
        # A final answer line with the option text included, e.g. "(D) foo".
        # Examples: "(D) all of the above", "**(B)** pressure increases"
        re.compile(
            r"""(?im)^\s*
            (?:\*{1,2}|_{1,2})?
            \(\s*([ABCD])\s*\)
            (?:\*{1,2}|_{1,2})?
            \s+\S
            """
        ),
    ]

    # Each fallback pattern below also captures the answer letter as group 1.
    PATTERNS = [
        # 0) "**Answer:** A" or "*Answers* – B", i.e. markdown-wrapped "Answer(s)" with an unwrapped letter.
        # Examples: "**Answer:** A", "*Answers* - B", "__Answer__ C"
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
        # Examples: "Answer: **D**", "**Answer:** C", "answer B"
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
        # Examples: "Answer: (C)", "Answers - (B)"
        re.compile(r"(?ix)\bAnswer[s]?\b\s*[:\-–]?\s*\(\s*([ABCD])\s*\)"),
        # 2) Answer: C    or   Answers – D
        # Examples: "Answer: C", "Answers - D"
        re.compile(r"(?ix)\bAnswer[s]?\b\s*[:\-–]?\s*([ABCD])\b"),
        # 3) Option B   or   Choice: C
        # Examples: "Option B", "Choice: C"
        re.compile(r"(?ix)\b(?:Option|Choice)\b\s*[:\-–]?\s*([ABCD])\b"),
        # 7) LaTeX \boxed{...A...}, catches both \boxed{A} and
        #    \boxed{\text{A } 2.08\times10^{-6}\,\mathrm{m}} etc.
        # Examples: "\\boxed{A}", "\\boxed{the answer is C}"
        re.compile(r"(?x)\\boxed\{[^}]*?([ABCD])[^}]*\}", re.MULTILINE),
        # 7.5) LaTeX \boxed{\textbf{...C...}}
        # Examples: "\\boxed{\\textbf{C}}", "\\boxed{\\textbf{choice D}}"
        re.compile(
            r"(?x)\\boxed\{[^}]*?\\textbf\{[^}]*?([ABCD])[^}]*\}[^}]*\}", re.MULTILINE
        ),
        # 7.51) LaTeX \boxed{\text{...C...}}
        # Examples: "\\boxed{\\text{B}}", "\\boxed{\\text{Answer: A}}"
        re.compile(
            r"(?x)\\boxed\{[^}]*?\\text\{[^}]*?([ABCD])[^}]*\}[^}]*\}", re.MULTILINE
        ),
        # 4) bare singletons:  (A)  [B]
        # Examples: "(A)", "[B]"
        re.compile(r"(?x)(?<![A-Za-z0-9])[\(\[]\s*([ABCD])\s*[\)\]](?![A-Za-z0-9])"),
        # 5) Markdown-wrapped: *A*  **B**  _C_  __D__
        # Examples: "*A*", "**B**", "_C_", "__D__"
        re.compile(
            r"(?x)(?<![A-Za-z0-9])(?:\*{1,2}|_{1,2})([ABCD])(?:\*{1,2}|_{1,2})(?![A-Za-z0-9])"
        ),
        # 6) LaTeX \textbf{...C...}
        # Examples: "\\textbf{C}", "\\textbf{Answer D}"
        re.compile(r"(?x)\\textbf\{[^}]*?([ABCD])[^}]*\}"),
        # 8) markdown-wrapped answer plus ")" plus description, e.g. **D) …**
        # Examples: "**D) all of the above**", "_B) pressure increases_"
        re.compile(r"""(?x)                        # ignore whitespace in pattern
            (?<![A-Za-z0-9])            # not preceded by word-char
            (?:\*{1,2}|_{1,2})          # opening ** or __ or * or _
            \s*([ABCD])\)               # capture letter plus ")"
            [^*_\n]+?                   # some text inside wrapper
            (?:\*{1,2}|_{1,2})          # closing wrapper
            (?![A-Za-z0-9])             # not followed by word-char
        """),
        # 9) final fallback: a line that's exactly "A", "B.", "C)", "**D**", etc.
        # Examples: "A", "B.", "C)", "**D**", "A - because ..."
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
        if not text or not isinstance(text, str):
            return default if default is not None else ""

        # Reasoning models often produce a long rationale followed by a concise
        # final answer. Prefer explicit answer forms near the response tail before
        # broad option-list fallbacks such as "(A)" or "(B)".
        #
        # Rightmost-match rule: scan only the last 6000 characters, collect every
        # final-answer regex match as (start_offset, end_offset, letter), then
        # select max(..., key=(start_offset, end_offset)). This guarantees that
        # the latest regex match inside the scanned tail wins, e.g. an earlier
        # "Choice: A" is overridden by a later "Final answer: D". This is still a
        # regex-position guarantee, not a semantic guarantee: if the true answer
        # is outside the tail, does not match these regexes, or is followed by a
        # later false-positive answer-like string, the extractor can still choose
        # the wrong letter.
        tail = text.strip()[-6000:]
        final_matches: list[tuple[int, int, str]] = []
        for pat in cls.FINAL_ANSWER_PATTERNS:
            for m in pat.finditer(tail):
                letter = m.group(1).upper()
                if letter in cls.CHOICE_MAP:
                    final_matches.append((m.start(), m.end(), letter))
        if final_matches:
            _, _, letter = max(final_matches, key=lambda item: (item[0], item[1]))
            return cls.CHOICE_MAP[letter]

        matches = []
        for prio, pat in enumerate(cls.PATTERNS):
            m = pat.search(text)
            if m:
                letter = m.group(1).upper()
                if letter in "ABCD":
                    matches.append((prio, m, letter))

        # Sort by priority (lower is better) and then by match length (shorter is better)
        matches.sort(key=lambda triple: (triple[0], len(triple[1].group(0))))

        # Return the best match
        for _, _, letter in matches:
            return cls.CHOICE_MAP[letter]

        # Final fallback from OpenAI: take first character after stripping markdown
        # This is a last resort if no patterns matched
        stripped = text.removeprefix("**")
        if stripped and stripped[0].upper() in "ABCD":
            abcd_choice = stripped[0].upper()
            return cls.CHOICE_MAP[abcd_choice]

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

        # Collapse repeated/empty language openers (e.g. a stray
        # "```python\n\n```python\n<code>") that otherwise make the middle
        # fence look like a closing fence and yield an empty block.
        text = re.sub(
            r"(?:```[ \t]*(?:python|py)[ \t]*\r?\n\s*)+?(```[ \t]*(?:python|py)[ \t]*\r?\n)",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )

        # Try ```python blocks first (most specific). Pick the last NON-EMPTY
        # block: models often emit a trailing empty ```python``` (e.g. in a
        # meta-sentence) after the real solution.
        python_blocks = [
            m.group(1).strip()
            for m in re.finditer(r"```[ \t]*python[ \t]*\r?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        ]
        python_blocks = [c for c in python_blocks if c]
        if python_blocks:
            return python_blocks[-1]

        # Fall back to plain ``` blocks (last non-empty, strip any lang tag).
        plain_blocks = []
        for m in re.finditer(r"```(.*?)```", text, re.DOTALL):
            code = re.sub(r"^(?:python|py)\s*\r?\n", "", m.group(1).strip(), flags=re.IGNORECASE).strip()
            if code:
                plain_blocks.append(code)
        if plain_blocks:
            return plain_blocks[-1]

        # Last resort: an unclosed final fence (truncated response). Take from
        # the last opening fence to the end of text.
        m = re.search(
            r"```[ \t]*(?:python|py)?[ \t]*\r?\n(.*)$", text, re.DOTALL | re.IGNORECASE
        )
        if m:
            code = m.group(1).strip().rstrip("`").strip()
            if code:
                return code

        return default
