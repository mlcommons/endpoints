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

# MLPerf canonical negative prompt for WAN2.2-T2V-A14B.
# Injected into every sample's query.data so the server receives the exact
# string MLPerf expects, rather than relying on the server's internal default.
_MLPERF_NEGATIVE_PROMPT = (
    "vivid colors, overexposed, static, blurry details, subtitles, style, "
    "work of art, painting, picture, still, overall grayish, worst quality, "
    "low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, "
    "poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, "
    "static image, cluttered background, three legs, many people in the background, "
    "walking backwards"
)


class VideoGenDataset(Dataset, dataset_id="wan22_mlperf"):
    """Dataset that loads MLPerf WAN2.2 prompt text files.

    Each non-blank line in the file is one prompt. MLPerf endpoints run perf
    and accuracy in a single pass, so VideoGenAdapter always requests video_bytes.

    By default, the MLPerf canonical negative prompt is injected into every sample.
    Pass ``negative_prompt=None`` to omit the field and let the server apply its
    own default. Pass ``latent_path=<path>`` to use a fixed pre-computed latent
    tensor for reproducibility.
    """

    COLUMN_NAMES = ["prompt"]

    @classmethod
    def get_dataloader(  # type: ignore[override]
        cls,
        path: Path | str | None = None,
        negative_prompt: str | None = _MLPERF_NEGATIVE_PROMPT,
        latent_path: Path | str | None = None,
        **kwargs: Any,
    ) -> "VideoGenDataset":
        """Create a VideoGenDataset from a prompts file path.

        Called by DataLoaderFactory when ``--dataset <path>`` is used with
        ``name=wan22_mlperf``.  The ``path`` argument maps directly to
        ``prompts_path``.
        """
        if path is None:
            raise ValueError(
                "VideoGenDataset requires a prompts file path. "
                "Pass --dataset <path/to/prompts.txt> or set path= in the dataset config."
            )
        return cls(
            prompts_path=path,
            negative_prompt=negative_prompt,
            latent_path=latent_path,
        )

    def __init__(
        self,
        prompts_path: Path | str,
        negative_prompt: str | None = _MLPERF_NEGATIVE_PROMPT,
        latent_path: Path | str | None = None,
    ) -> None:
        prompts = [
            line.strip()
            for line in Path(prompts_path).read_text().splitlines()
            if line.strip()
        ]
        super().__init__(dataframe=pd.DataFrame({"prompt": prompts}))
        self.negative_prompt = negative_prompt
        self.latent_path = str(latent_path) if latent_path is not None else None

    def load(self, **kwargs: Any) -> None:  # type: ignore[override]
        """Build self.data from the loaded dataframe. No transforms needed.

        Optional fields (``negative_prompt``, ``latent_path``) are omitted from
        each sample dict when their dataset-level value is ``None``, so the
        adapter's ``exclude_none=True`` serialisation falls back to server-side
        defaults.
        """
        assert self.dataframe is not None
        self.data = [
            {
                "prompt": row["prompt"],
                **(
                    {"negative_prompt": self.negative_prompt}
                    if self.negative_prompt is not None
                    else {}
                ),
                **(
                    {"latent_path": self.latent_path}
                    if self.latent_path is not None
                    else {}
                ),
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
