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

"""WAN2.2 prompt dataset for MLPerf inference benchmarking."""

from pathlib import Path
from typing import Any

import pandas as pd

from inference_endpoint.dataset_manager.dataset import Dataset


class Wan22Dataset(Dataset, dataset_id="wan22_mlperf"):
    """Dataset that loads MLPerf WAN2.2 prompt text files.

    Each non-blank line in the file is one prompt. MLPerf endpoints run perf
    and accuracy in a single pass, so Wan22Adapter always requests video_bytes.
    """

    COLUMN_NAMES = ["prompt"]

    @classmethod
    def get_dataloader(  # type: ignore[override]
        cls,
        path: Path | str | None = None,
        negative_prompt: str = "",
        **kwargs: Any,
    ) -> "Wan22Dataset":
        """Create a Wan22Dataset from a prompts file path.

        Called by DataLoaderFactory when ``--dataset <path>`` is used with
        ``name=wan22_mlperf``.  The ``path`` argument maps directly to
        ``prompts_path``.
        """
        if path is None:
            raise ValueError(
                "Wan22Dataset requires a prompts file path. "
                "Pass --dataset <path/to/prompts.txt> or set path= in the dataset config."
            )
        return cls(prompts_path=path, negative_prompt=negative_prompt)

    def __init__(
        self,
        prompts_path: Path | str,
        negative_prompt: str = "",
    ) -> None:
        prompts = [
            line.strip()
            for line in Path(prompts_path).read_text().splitlines()
            if line.strip()
        ]
        super().__init__(dataframe=pd.DataFrame({"prompt": prompts}))
        self.negative_prompt = negative_prompt

    def load(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Build self.data from the loaded dataframe. No transforms needed."""
        assert self.dataframe is not None
        self.data = [
            {
                "prompt": row["prompt"],
                "negative_prompt": self.negative_prompt,
                "sample_id": str(i),
                "sample_index": i,
            }
            for i, row in self.dataframe.iterrows()
        ]

    def load_sample(self, index: int) -> dict[str, Any]:
        assert self.data is not None, "Dataset not loaded. Call load() first."
        return dict(self.data[index % len(self.data)])

    def num_samples(self) -> int:
        assert self.data is not None, "Dataset not loaded. Call load() first."
        return len(self.data)
