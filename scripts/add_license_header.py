#!/usr/bin/env python3
"""Add license headers to Python files."""

import argparse
import sys
from pathlib import Path

LICENSE_HEADER = """# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""


def has_license_header(content: str) -> bool:
    """Check if file already has a license header."""
    return "SPDX-License-Identifier" in content or "Copyright" in content[:500]


def add_header_to_file(filepath: Path) -> bool:
    """Add license header to a file if it doesn't have one.

    Returns:
        True if file was modified, False otherwise.
    """
    content = filepath.read_text()

    # Skip if already has header
    if has_license_header(content):
        return False

    # Handle shebang
    if content.startswith("#!"):
        lines = content.split("\n", 1)
        shebang = lines[0] + "\n"
        rest = lines[1] if len(lines) > 1 else ""
        new_content = shebang + LICENSE_HEADER + rest
    else:
        new_content = LICENSE_HEADER + content

    filepath.write_text(new_content)
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Add license headers to files")
    parser.add_argument("files", nargs="+", help="Files to add headers to")
    args = parser.parse_args()

    modified = []
    for filepath in args.files:
        path = Path(filepath)
        if path.suffix == ".py" and add_header_to_file(path):
            modified.append(filepath)

    if modified:
        print(f"Added license headers to {len(modified)} file(s):")
        for f in modified:
            print(f"  - {f}")
        return 1  # Return 1 to indicate files were modified

    return 0


if __name__ == "__main__":
    sys.exit(main())
