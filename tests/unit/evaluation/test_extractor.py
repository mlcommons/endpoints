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


import pytest
from inference_endpoint.evaluation.extractor import (
    Extractor,
    LetterExtractor,
    PythonCodeExtractor,
)


@pytest.mark.unit
class TestPythonCodeExtractor:
    """Test cases for PythonCodeExtractor."""

    def test_extract_python_block(self):
        """Test extraction from ```python block."""
        text = "Here's the solution:\n```python\ndef foo():\n    pass\n```"
        result = PythonCodeExtractor.extract(text)
        assert result == "def foo():\n    pass"

    def test_extract_plain_block(self):
        """Test extraction from plain ``` block."""
        text = "Solution:\n```\nprint('hello')\n```"
        result = PythonCodeExtractor.extract(text)
        assert result == "print('hello')"

    def test_extract_last_python_block(self):
        """Test that it extracts the last ```python block."""
        text = """First attempt:
```python
def wrong():
    pass
```

Better solution:
```python
def correct():
    return True
```"""
        result = PythonCodeExtractor.extract(text)
        assert result == "def correct():\n    return True"

    def test_extract_plain_with_language_tag(self):
        """Test extraction from plain block with language tag."""
        text = "```\npython\nprint('test')\n```"
        result = PythonCodeExtractor.extract(text)
        # Should remove the language tag line
        assert result == "print('test')"

    def test_extract_py_tag(self):
        """Test extraction with ```py tag."""
        text = "```\npy\nprint('test')\n```"
        result = PythonCodeExtractor.extract(text)
        assert result == "print('test')"

    def test_extract_none_empty_string(self):
        """Test that empty string returns None."""
        assert PythonCodeExtractor.extract("") is None

    def test_extract_none_no_code_block(self):
        """Test that text without code block returns None."""
        text = "This is just plain text without any code blocks."
        assert PythonCodeExtractor.extract(text) is None

    def test_extract_none_null_input(self):
        """Test that None input returns None."""
        assert PythonCodeExtractor.extract(None) is None

    def test_extract_none_non_string(self):
        """Test that non-string input returns None."""
        assert PythonCodeExtractor.extract(123) is None

    def test_extract_multiline_code(self):
        """Test extraction of multiline code."""
        text = """```python
class Solution:
    def solve(self, n: int) -> int:
        result = 0
        for i in range(n):
            result += i
        return result
```"""
        expected = """class Solution:
    def solve(self, n: int) -> int:
        result = 0
        for i in range(n):
            result += i
        return result"""
        result = PythonCodeExtractor.extract(text)
        assert result == expected

    def test_extract_with_markdown_formatting(self):
        """Test extraction from text with markdown formatting."""
        text = """**Solution:**

Here's the code:

```python
def foo():
    return 42
```

*This works!*"""
        result = PythonCodeExtractor.extract(text)
        assert result == "def foo():\n    return 42"

    def test_priority_python_over_plain(self):
        """Test that ```python blocks have priority over plain blocks."""
        text = """Plain block:
```
def plain():
    pass
```

Python block:
```python
def python():
    return True
```"""
        result = PythonCodeExtractor.extract(text)
        # Should extract the last python block
        assert result == "def python():\n    return True"

    def test_extract_with_inline_code(self):
        """Test that inline code is not extracted."""
        text = "Use `print()` to output. No code block here."
        assert PythonCodeExtractor.extract(text) is None

    def test_extract_whitespace_handling(self):
        """Test that whitespace is properly handled."""
        text = "  \n\n  ```python\n  def test():\n      pass\n  ```  \n\n  "
        result = PythonCodeExtractor.extract(text)
        assert result == "def test():\n      pass"

    def test_registered_in_extractor_registry(self):
        """Test that PythonCodeExtractor is registered."""
        assert "python_code_extractor" in Extractor.PREDEFINED
        assert Extractor.get("python_code_extractor") == PythonCodeExtractor

    def test_extractor_get_method(self):
        """Test that we can retrieve PythonCodeExtractor by name."""
        extractor_cls = Extractor.get("python_code_extractor")
        text = "```python\nprint('test')\n```"
        result = extractor_cls.extract(text)
        assert result == "print('test')"


@pytest.mark.unit
class TestLetterExtractor:
    """Tests for LetterExtractor — returns raw letter (A–J) for MCQ datasets
    where ground truth is stored as a letter (e.g. MLPerf GPQA/MMLU-Pro)."""

    def test_answer_colon(self):
        assert LetterExtractor.extract("Answer: B") == "B"

    def test_markdown_answer(self):
        assert LetterExtractor.extract("**Answer:** C") == "C"

    def test_extended_range(self):
        # E–J range needed for MMLU-Pro (10 choices); check boundary letters
        assert LetterExtractor.extract("Answer: E") == "E"
        assert LetterExtractor.extract("Answer: J") == "J"

    def test_boxed_letter(self):
        assert LetterExtractor.extract(r"\boxed{F}") == "F"

    def test_parenthesised_singleton(self):
        assert LetterExtractor.extract("The correct choice is (H)") == "H"

    def test_mlperf_style_output(self):
        # Mirrors the few-shot prompt format: model outputs "Answer: X"
        text = "Let me think... the ring characteristic is 0.\nAnswer: A"
        assert LetterExtractor.extract(text) == "A"

    def test_no_match_returns_empty_string(self):
        assert LetterExtractor.extract("No answer here.") == ""

    def test_default_on_no_match(self):
        assert LetterExtractor.extract("Nothing", default="X") == "X"

    def test_registered(self):
        assert "letter_extractor" in Extractor.PREDEFINED
        assert Extractor.get("letter_extractor") is LetterExtractor
