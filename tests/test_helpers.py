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
import random
import string
import uuid
from pathlib import Path

import zmq
from inference_endpoint.core.types import (
    Query,
)
from inference_endpoint.dataset_manager.dataset import Dataset


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
