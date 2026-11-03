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

"""
Test utility functions and helper classes for pytest tests.

This module provides reusable test utilities that can be imported directly
instead of using pytest fixtures with factory patterns.
"""

import hashlib
import json
import random
import re
import string
import uuid
from pathlib import Path
from typing import Any

import msgspec
import zmq
from inference_endpoint.config.schema import Settings
from inference_endpoint.core.types import (
    TOKEN_HDR_HIGH,
    Query,
    TextModelOutput,
)
from inference_endpoint.dataset_manager.dataset import Dataset
from inference_endpoint.metrics.report import OtherBasisMetrics, Report
from inference_endpoint.metrics.token_count_basis import ServerUsage

# The token-count basis a run uses when none is configured. Basis-dependent tests
# take it from here instead of pinning a literal.
DEFAULT_SOURCE = Settings.model_fields["token_count_source"].default

# One table pins the usable-server-usage rule for every consumer:
# (id, prompt_tokens, completion_tokens, output, expected classification).
SERVER_USAGE_CASES: list[tuple[str, object, object, TextModelOutput, ServerUsage]] = [
    ("usable", 10, 5, TextModelOutput(output="hi"), ServerUsage.USABLE),
    (
        "missing_both",
        msgspec.UNSET,
        msgspec.UNSET,
        TextModelOutput(output="hi"),
        ServerUsage.MISSING,
    ),
    (
        "missing_prompt",
        msgspec.UNSET,
        5,
        TextModelOutput(output="hi"),
        ServerUsage.MISSING,
    ),
    ("zero_beside_output", 10, 0, TextModelOutput(output="hi"), ServerUsage.MISSING),
    ("true_zero", 10, 0, TextModelOutput(output=""), ServerUsage.TRUE_ZERO),
    ("zero_prompt", 0, 5, TextModelOutput(output="hi"), ServerUsage.MISSING),
    ("negative", 10, -1, TextModelOutput(output="hi"), ServerUsage.MISSING),
    ("bool", True, 5, TextModelOutput(output="hi"), ServerUsage.MISSING),
    (
        "above_bound",
        10,
        TOKEN_HDR_HIGH + 1,
        TextModelOutput(output="hi"),
        ServerUsage.MISSING,
    ),
    ("float", 10.0, 5, TextModelOutput(output="hi"), ServerUsage.MISSING),
]

# The cases an event log can carry. It stores counts as integers, so a bool or
# float count reaches a post-run reader as an undecodable line, not a usage case.
LOGGED_SERVER_USAGE_CASES = [
    case
    for case in SERVER_USAGE_CASES
    if all(
        value is msgspec.UNSET
        or (isinstance(value, int) and not isinstance(value, bool))
        for value in case[1:3]
    )
]


def write_large_prompt_dataset(directory: Path, n_samples: int) -> Path:
    path = directory / "big_prompts.jsonl"
    prompt = "lorem ipsum " * 21_000
    with path.open("w") as f:
        for i in range(n_samples):
            f.write(json.dumps({"prompt": f"{i} {prompt}"}) + "\n")
    return path


def _generate_random_word(
    rng: random.Random, mean_length: float = 5.0, std_dev: float = 2.0
) -> str:
    """Generate a random word with length following a normal distribution.

    Args:
        rng: Random number generator instance
        mean_length: Average word length (default: 5 characters)
        std_dev: Standard deviation for length distribution (default: 2.0)

    Returns:
        A random word with length drawn from a normal distribution, clamped to [1, 15]
    """
    length = int(rng.gauss(mean_length, std_dev))
    length = max(1, min(15, length))  # Clamp to reasonable bounds
    return "".join(rng.choices(string.ascii_lowercase, k=length))


def create_test_query(
    prompt_size: int = 100,
    stream: bool = False,
    query_id: str | None = None,
    seed: int | None = None,
) -> Query:
    """Create a test query with specified parameters.

    This is a flexible factory that can create queries with different sizes,
    streaming modes, and custom IDs for testing purposes. The prompt is generated
    using randomly generated words with realistic length distributions.

    Args:
        prompt_size: Target size of the prompt in characters (default: 100).
                    The actual size may vary slightly due to word boundaries.
        stream: Whether to enable streaming mode (default: False)
        query_id: Custom query ID, or None to generate a UUID (default: None)
        seed: Random seed for reproducible prompt generation (default: None)

    Returns:
        Query: A test query with the specified parameters

    Examples:
        # Create a simple query
        query = create_test_query()

        # Create a large query for performance testing
        large_query = create_test_query(prompt_size=1000, stream=False)

        # Create a streaming query with custom ID
        streaming_query = create_test_query(stream=True, query_id="test-123")

        # Create a query with reproducible prompt
        query = create_test_query(prompt_size=500, seed=42)
    """
    # Use a local random instance for reproducibility if seed is provided
    rng = random.Random(seed) if seed is not None else random.Random()

    # Generate prompt from random words until we reach approximately the target size
    words = []
    current_length = 0

    while current_length < prompt_size:
        word = _generate_random_word(rng)
        words.append(word)
        # Add 1 for the space character (except for the first word)
        current_length += len(word) + (1 if words else 0)

    prompt = " ".join(words)

    # Trim to exact size if we overshot
    if len(prompt) > prompt_size:
        prompt = prompt[:prompt_size].rstrip()

    return Query(
        id=query_id or str(uuid.uuid4()),
        data={
            "model": "test-model",
            "prompt": prompt,
            "stream": stream,
        },
    )


class DummyDataLoader(Dataset):
    """Simple dataloader for testing that returns sample indices directly.

    This dataloader is useful for unit tests where you need a simple DataLoader
    implementation that returns predictable values.

    Args:
        n_samples: Number of samples in the dataset (default: 100)

    Examples:
        # Create a dataloader with 100 samples
        loader = DummyDataLoader()
        assert loader.num_samples() == 100
        assert loader.load_sample(5) == 5

        # Create a dataloader with custom sample count
        loader = DummyDataLoader(n_samples=50)
        assert loader.num_samples() == 50
    """

    def __init__(self, n_samples: int = 100):
        super().__init__(None)
        self.n_samples = n_samples

    def load_sample(self, sample_index: int) -> int:
        """Load a sample by its index (returns the index itself)."""
        assert sample_index >= 0 and sample_index < self.n_samples
        return sample_index

    def num_samples(self) -> int:
        """Return the total number of samples."""
        return self.n_samples


def get_test_socket_path(tmp_path: Path, test_name: str, suffix: str = "") -> str:
    """Generate a short socket name using hash to avoid path length limits.

    This avoids Unix domain socket path length limits (typically 108 chars)
    when pytest runs tests in parallel with long temporary directory paths.

    Args:
        tmp_path: The pytest tmp_path fixture or any Path object
        test_name: A unique identifier for the test
        suffix: Optional suffix to append (e.g., "_req", "_resp")

    Returns:
        A short IPC socket path like "ipc:///tmp/.../a1b2c3d4_req"

    Examples:
        # In a test function
        socket_path = get_test_socket_path(tmp_path, "test_zmq_client", "_req")
        # Returns: "ipc:///tmp/pytest-xxx/a1b2c3d4_req"
    """
    # Create a hash of the test name to ensure uniqueness
    hash_val = hashlib.md5(test_name.encode()).hexdigest()[:8]
    # Combine with a short suffix
    name = f"{hash_val}{suffix}"
    socket_path = f"ipc://{tmp_path}/{name}"
    assert (
        len(socket_path) <= zmq.IPC_PATH_MAX_LEN
    ), "socket path is too long for ZMQ IPC"
    return socket_path


def parse_sse_frames(body: bytes) -> list[dict[str, Any] | bytes]:
    """Parse SSE ``data:`` frames from a raw or decoded streaming body.

    JSON frames decode to dicts and the ``[DONE]`` sentinel stays ``b"[DONE]"``.
    HTTP chunk-size lines are ignored, so this accepts raw chunked bytes as well
    as bodies already decoded by an HTTP client.
    """
    return [
        m if m == b"[DONE]" else json.loads(m)
        for m in re.findall(rb"data: (\{[^\n]+\}|\[DONE\])", body)
    ]


_BASE_REPORT: dict[str, Any] = {
    "version": "t",
    "git_sha": None,
    "test_started_at": 0,
    "n_samples_issued": 10,
    "n_samples_completed": 10,
    "n_samples_failed": 0,
    "duration_ns": 1,
    "state": "complete",
    "complete": True,
    "ttft": {},
    "tpot": {},
    "latency": {},
    "input_sequence_lengths": {},
    "output_sequence_lengths": {},
    "token_count_source": DEFAULT_SOURCE,
    "other_basis": OtherBasisMetrics(
        input_sequence_lengths=None,
        output_sequence_lengths=None,
        tpot=None,
        tps=None,
    ),
}


def make_report(**over: Any) -> Report:
    """A complete 10-sample ``Report`` with ``over`` applied to its fields."""
    return Report(**(_BASE_REPORT | over))
