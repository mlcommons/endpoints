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

"""Pre-fetch VBench model weights using Python stdlib only (no wget/unzip).

Run at image-build time to warm the VBench cache for the MLPerf WAN 2.2
dimensions. Three dims download via wget at evaluation time; this script
fetches them upfront so the image is self-contained:
  - motion_smoothness  (AMT checkpoint)
  - dynamic_degree     (RAFT models zip, extracted in place)
  - scene              (Tag2Text checkpoint)

The other three MLPerf dims (subject_consistency, background_consistency,
appearance_style) use torch.hub / CLIP and handle their own caching.
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile

_CACHE_DIR = Path(
    os.environ.get("VBENCH_CACHE_DIR") or Path.home() / ".cache" / "vbench"
)

# (url, dest, unzip_into_or_None, cache_guard_or_None)
# cache_guard: existence check path. For RAFT, the zip is deleted after
# extraction, so guard on raft-things.pth instead of the missing zip.
_DOWNLOADS = [
    (
        "https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth",
        _CACHE_DIR / "amt_model" / "amt-s.pth",
        None,
        None,
    ),
    (
        "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip",
        _CACHE_DIR / "raft_model" / "models.zip",
        _CACHE_DIR / "raft_model",
        _CACHE_DIR / "raft_model" / "models" / "raft-things.pth",
    ),
    (
        "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth",
        _CACHE_DIR / "caption_model" / "tag2text_swin_14m.pth",
        None,
        None,
    ),
]


def _fetch(
    url: str, dest: Path, unzip_into: "Path | None", cache_guard: "Path | None"
) -> None:
    guard = cache_guard if cache_guard is not None else dest
    if guard.exists():
        print(f"  cached: {guard.name}", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url}", flush=True)
    # Atomic write: interrupted builds leave a .tmp, not a partial artifact.
    with NamedTemporaryFile(dir=dest.parent, delete=False, suffix=".tmp") as tf:
        tmp = Path(tf.name)
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    size_mb = dest.stat().st_size // (1024 * 1024)
    print(f"  saved {dest} ({size_mb} MB)", flush=True)
    if unzip_into is not None:
        unzip_into.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(unzip_into)
        dest.unlink()
        print(f"  extracted to {unzip_into}", flush=True)


def main() -> int:
    print(f"VBench cache dir: {_CACHE_DIR}", flush=True)
    for url, dest, unzip_into, cache_guard in _DOWNLOADS:
        try:
            _fetch(url, dest, unzip_into, cache_guard)
        except Exception as exc:
            print(f"ERROR fetching {url}: {exc}", file=sys.stderr, flush=True)
            return 1
    print("VBench weight prefetch complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
