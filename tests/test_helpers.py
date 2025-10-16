"""
Test utility functions and helper classes for pytest tests.

This module provides reusable test utilities that can be imported directly
instead of using pytest fixtures with factory patterns.
"""

import hashlib
import random
import string
import uuid
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from inference_endpoint.core.types import Query
from inference_endpoint.dataset_manager.dataloader import DataLoader
from inference_endpoint.load_generator.events import SampleEvent
from inference_endpoint.load_generator.load_generator import SampleIssuer
from inference_endpoint.load_generator.sample import SampleFactory


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
    rng = random.Random(seed) if seed is not None else random

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


class DummyDataLoader(DataLoader):
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
    return f"ipc://{tmp_path}/{name}"


class SerialSampleIssuer(SampleIssuer):
    """SampleIssuer for testing. No threading, and is blocking. Whenever issue is called,
    it performs the provided compute function, calling callbacks when necessary.

    The compute function should be a generator, yielding the 'chunks' of the supposed
    response.
    """

    def __init__(self, compute_func=None):
        if compute_func is None:
            self.compute_func = lambda x: x
        else:
            self.compute_func = compute_func

    def issue(self, sample):
        dat = sample.get_bytes()
        sample.callbacks[SampleEvent.REQUEST_SENT](None)
        first = True
        chunks = []
        for chunk in self.compute_func(dat):
            chunks.append(chunk)
            if first:
                sample.callbacks[SampleEvent.FIRST_CHUNK](chunk)
                first = False
            else:
                sample.callbacks[SampleEvent.NON_FIRST_CHUNK](chunk)
        sample.callbacks[SampleEvent.COMPLETE](chunks)


class PooledSampleIssuer(SampleIssuer):
    """SampleIssuer that has a non-blocking issue() method. Has a pool of workers which compute
    the samples in parallel.

    Uses ThreadPoolExecutor to properly propagate exceptions from worker threads to the main thread.
    Call check_errors() to raise any exceptions that occurred in workers and clean up completed futures.
    """

    def __init__(self, compute_func=None, n_workers: int = 4):
        self.n_workers = n_workers
        if compute_func is None:
            self.compute_func = lambda x: x
        else:
            self.compute_func = compute_func
        self.executor = ThreadPoolExecutor(max_workers=n_workers)
        self.futures = []

    def shutdown(self):
        """Shutdown the executor and wait for all tasks to complete.

        Raises any exceptions that occurred in worker threads.
        """
        self.executor.shutdown(wait=True)
        # Check all futures for exceptions
        for future in self.futures:
            future.result()  # This will raise if the worker raised an exception
        self.futures.clear()

    def handle_sample(self, sample):
        dat = sample.get_bytes()
        first = True
        chunks = []
        for chunk in self.compute_func(dat):
            chunks.append(chunk)
            if first:
                sample.callbacks[SampleEvent.FIRST_CHUNK](chunk)
                first = False
            else:
                sample.callbacks[SampleEvent.NON_FIRST_CHUNK](chunk)
        sample.callbacks[SampleEvent.COMPLETE](chunks)

    def check_errors(self):
        """Check if any worker thread has raised an exception and re-raise it.

        This checks completed futures without blocking and removes them from the list
        to prevent unbounded memory growth.
        """
        remaining_futures = []
        for future in self.futures:
            if future.done():
                # This will raise if the worker raised an exception
                future.result()
                # Don't keep completed futures
            else:
                # Keep incomplete futures
                remaining_futures.append(future)
        self.futures = remaining_futures

    def issue(self, sample):
        """Submit a sample to be processed by the worker pool."""
        sample.callbacks[SampleEvent.REQUEST_SENT](None)
        future = self.executor.submit(self.handle_sample, sample)
        self.futures.append(future)

        # Periodically clean up completed futures to prevent unbounded growth
        # Check every 100 submissions to balance cleanup overhead vs memory usage
        if len(self.futures) >= 100:
            self.check_errors()


class HistogramSampleFactory(SampleFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(None, None)

        self.sent_hist = defaultdict(int)
        self.completed_hist = defaultdict(int)
        self.completed_chunks = {}
        self.uuid_to_idx = {}

    @staticmethod
    def sample_get_bytes(dataloader, sample_index):
        return sample_index

    def inc_sent(self, sample_index: int, sample_uuid: str, val):
        self.sent_hist[sample_index] += 1

    def inc_completed(self, sample_index: int, sample_uuid: str, val):
        self.completed_hist[sample_index] += 1
        self.completed_chunks[sample_uuid] = val

    def get_sample_callbacks(
        self, sample_index: int, sample_uuid: str
    ) -> dict[SampleEvent, Callable]:
        self.uuid_to_idx[sample_uuid] = sample_index
        return {
            SampleEvent.REQUEST_SENT: partial(self.inc_sent, sample_index, sample_uuid),
            SampleEvent.FIRST_CHUNK: (lambda val: None),
            SampleEvent.NON_FIRST_CHUNK: (lambda val: None),
            SampleEvent.COMPLETE: partial(
                self.inc_completed, sample_index, sample_uuid
            ),
        }
