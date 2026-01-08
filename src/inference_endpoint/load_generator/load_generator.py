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

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from ..dataset_manager.dataset import Dataset
from ..metrics.recorder import EventRecorder
from ..utils import sleep_ns
from .events import SessionEvent
from .sample import IssuedSample, Sample
from .scheduler import Scheduler


class SampleIssuer(ABC):
    """Abstract base class for components that send samples to inference endpoints.

    SampleIssuers are responsible for the complete workflow of sending a sample
    to a System Under Test (SUT):
    1. Ingest a Sample object from the Load Generator
    2. Build the appropriate request format (HTTP, gRPC, etc.)
    3. Send the request to the endpoint
    4. Handle the response asynchronously (results arrive via callbacks)

    Implementations must handle:
    - Request formatting (converting Sample.data to endpoint-specific format)
    - Network communication (HTTP, gRPC, WebSocket, etc.)
    - Error handling (timeouts, connection errors, etc.)
    - Response routing (back to metrics collector via events)

    Lifecycle:
    1. start() - Initialize connections, setup resources
    2. issue(sample) - Send samples (called repeatedly during benchmark)
    3. shutdown() - Clean up connections, release resources

    Example implementations:
    - HttpClientSampleIssuer: HTTP/REST endpoints (OpenAI-compatible)
    - GrpcSampleIssuer: gRPC endpoints (future)
    """

    def start(self):  # noqa: B027
        """Initialize resources and establish connections.

        Called once after instantiation to set up any dependency components
        like HTTP client pools, authentication, or connection pooling.

        Optional implementation - default does nothing.

        Raises:
            SetupError: If initialization fails.
        """
        pass

    @abstractmethod
    def issue(self, sample: Sample):
        """Send a sample to the SUT endpoint.

        This is the core method that sends a single sample/query to the endpoint.
        It should be non-blocking and return quickly - actual response handling
        happens asynchronously via the event system.

        The implementation must:
        1. Convert Sample.data to the endpoint's request format
        2. Send the request (typically async/non-blocking)
        3. Ensure response triggers appropriate events (COMPLETE, STREAM_CHUNK, etc.)

        Args:
            sample: Sample object containing request data and metadata.

        Raises:
            ExecutionError: If request cannot be sent.
        """
        raise NotImplementedError

    def shutdown(self):  # noqa: B027
        """Clean up resources and close connections.

        Called once when the issuer is no longer needed. Should gracefully
        shutdown connections, flush pending requests, and release resources.

        Optional implementation - default does nothing.
        """
        pass


class LoadGenerator(ABC):
    """Abstract base class for load generation strategies.

    LoadGenerators control WHEN samples are issued to the SUT. They coordinate:
    - Sample selection from the dataset (via DataLoader)
    - Timing and scheduling (via Scheduler)
    - Actual sample issuance (via SampleIssuer)
    - Event recording for metrics

    Key responsibilities:
    - Load sample data from dataset at the right time
    - Apply scheduling/timing delays
    - Issue samples via the SampleIssuer
    - Record timing events for metrics

    LoadGenerators are iterators - each iteration issues one sample and
    returns information about what was issued.

    Attributes:
        sample_issuer: Component that sends samples to endpoints.
        dataloader: Component that loads sample data from datasets.
    """

    def __init__(
        self,
        sample_issuer: SampleIssuer,
        dataloader: Dataset,
        name: str | None = None,
    ):
        """Initialize load generator with required dependencies.

        Args:
            sample_issuer: SampleIssuer to send samples to endpoint.
            dataloader: DataLoader to retrieve sample data from dataset.
        """
        self.sample_issuer = sample_issuer
        self.dataloader = dataloader
        self.name = name
        self.uuid_to_index_map = {}

    @abstractmethod
    def __next__(self) -> tuple[Sample, int]:
        """Issue the next sample according to the load generation strategy.

        This method should:
        1. Determine which sample to issue next
        2. Load the sample data from dataloader
        3. Apply any scheduling delays (blocking)
        4. Issue the sample via sample_issuer
        5. Return the sample and timestamp

        Note: This method MAY block to implement delays/scheduling.
        It should only return AFTER the sample has been issued.

        Returns:
            Tuple of (sample, timestamp_ns):
            - sample: The Sample object that was issued
            - timestamp_ns: Monotonic nanosecond timestamp when issued

        Raises:
            StopIteration: When all samples have been issued.
        """
        raise NotImplementedError

    def __iter__(self):
        """Return self as an iterator."""
        self.uuid_to_index_map = {}
        return self

    def load_sample_data(self, sample_index: int, sample_uuid: str) -> Any:
        """Load sample data from dataloader and record event.

        Helper method that loads sample data and records the data load event
        for accurate timing measurements.

        Args:
            sample_index: Index of sample in dataset.
            sample_uuid: UUID of the sample being created.

        Returns:
            Sample data loaded from dataloader (format depends on dataset).
        """
        sample_data = self.dataloader.load_sample(sample_index)
        EventRecorder.record_event(
            SessionEvent.LOADGEN_DATA_LOAD,
            time.monotonic_ns(),
            sample_uuid=sample_uuid,
        )
        return sample_data

    def issue_sample(self, sample: Sample) -> int:
        """Issue a sample via the SampleIssuer and record timing event.

        Helper method that:
        1. Records the current timestamp
        2. Records LOADGEN_ISSUE_CALLED event for metrics
        3. Invokes sample_issuer.issue(sample)
        4. Returns the timestamp

        The timestamp is recorded BEFORE issuing to ensure accurate timing
        even if the issue() call is slow or triggers immediate callbacks.

        Args:
            sample: Sample to issue to the endpoint.

        Returns:
            Monotonic nanosecond timestamp when issue was called.
        """
        timestamp_ns = time.monotonic_ns()

        # Currently, EventRecorder will raise an Exception if the in-flight sample
        # counter is negative. This happens if the SampleIssuer somehow invokes a
        # SampleEvent.COMPLETE event before the record_event call for LOADGEN_ISSUE_CALLED
        # goes off.
        # This can be solved by just recording the issue() call right before actually
        # invoking it. If this timing mechanism is a problem, we can remove the
        # negative check in EventRecorder, since the order of insertions doesn't matter
        # as much if the timestamps are correct.
        EventRecorder.record_event(
            SessionEvent.LOADGEN_ISSUE_CALLED,
            timestamp_ns,
            sample_uuid=sample.uuid,
        )
        logging.debug(f"Issuing sample {sample.uuid} at {timestamp_ns}")
        self.sample_issuer.issue(sample)
        return timestamp_ns


class SchedulerBasedLoadGenerator(LoadGenerator):
    """LoadGenerator that uses a Scheduler to control sample timing.

    This is the primary LoadGenerator implementation, delegating timing decisions
    to a pluggable Scheduler. It handles:
    - Sample ordering (via scheduler's sample_order)
    - Timing delays (via scheduler's delay_fn)
    - Sample loading and issuance
    - Timing measurements

    The scheduler determines:
    - Which sample to issue next (sample index)
    - How long to wait before issuing (delay in nanoseconds)

    This enables different load patterns (Poisson, max throughput, burst, etc.)
    without changing the LoadGenerator code.

    Attributes:
        scheduler: Scheduler controlling sample timing.
        _iterator: Iterator over scheduler (sample_index, delay) pairs.
        last_issue_timestamp_ns: Timestamp of last issued sample (for delay calculation).
    """

    def __init__(
        self,
        sample_issuer: SampleIssuer,
        dataloader: Dataset,
        scheduler: Scheduler,
    ):
        """Initialize scheduler-based load generator.

        Args:
            sample_issuer: SampleIssuer to send samples to endpoint.
            dataloader: DataLoader to retrieve sample data.
            scheduler: Scheduler controlling timing and sample order.
        """
        super().__init__(sample_issuer, dataloader)

        self.scheduler = scheduler
        self._iterator = None
        self.last_issue_timestamp_ns = 0

    def __next__(self) -> IssuedSample:
        """Issue next sample according to scheduler timing.

        This method:
        1. Gets next (sample_index, delay_ns) from scheduler
        2. Loads sample data from dataloader
        3. Waits for scheduled time (busy-wait for precision)
        4. Issues sample via sample_issuer
        5. Returns IssuedSample with timing info

        The busy-wait ensures precise timing even for high QPS scenarios
        where sleep() precision would be insufficient.

        Returns:
            IssuedSample containing sample, index, and actual issue timestamp.

        Raises:
            StopIteration: When scheduler has no more samples to issue.
        """
        # Let raised StopIteration be propagated up the stack
        s_idx, delay_ns = next(self._iterator)

        # Data loading is not timed for Time-to-Token metrics. It is assumed that the
        # hypothetical user would have put the data into memory available for a network
        # request beforehand.
        sample = Sample(None)  # Create sample object first to generate uuid
        sample.data = self.load_sample_data(s_idx, sample.uuid)

        self.uuid_to_index_map[sample.uuid] = s_idx

        scheduled_issue_timestamp_ns = self.last_issue_timestamp_ns + delay_ns
        while (now := time.monotonic_ns()) < scheduled_issue_timestamp_ns:
            sleep_ns(scheduled_issue_timestamp_ns - now)
        self.last_issue_timestamp_ns = self.issue_sample(sample)
        return IssuedSample(sample, s_idx, self.last_issue_timestamp_ns)

    def __iter__(self):
        if self._iterator is not None:
            raise RuntimeError(
                "SchedulerBasedLoadGenerator can only be iterated over once"
            )
        self._iterator = iter(self.scheduler)
        return super().__iter__()
