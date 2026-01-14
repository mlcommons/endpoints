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


from inference_endpoint.evaluation.extractor import PythonCodeExtractor


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
        from inference_endpoint.evaluation.extractor import Extractor

        assert "python_code_extractor" in Extractor.PREDEFINED
        assert Extractor.get("python_code_extractor") == PythonCodeExtractor

    def test_extractor_get_method(self):
        """Test that we can retrieve PythonCodeExtractor by name."""
        from inference_endpoint.evaluation.extractor import Extractor

        extractor_cls = Extractor.get("python_code_extractor")
        text = "```python\nprint('test')\n```"
        result = extractor_cls.extract(text)
        assert result == "print('test')"
