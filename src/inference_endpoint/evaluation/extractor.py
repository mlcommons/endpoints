# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import re
from abc import ABC, abstractmethod


class Extractor(ABC):
    """An Extractor is used to extract phrases or substrings from the model's outputs using
    multiple regex patterns with a priority system. This is useful for extracting values from
    strings with the same general format but small variations, such as a model outputting a
    numeric value plain or inside a LaTeX block.
    """

    @abstractmethod
    @classmethod
    def extract(cls, text: str) -> str | None:
        raise NotImplementedError


class ABCDExtractor(Extractor):
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
    def extract(cls, text: str) -> str | None:
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

        return ""
