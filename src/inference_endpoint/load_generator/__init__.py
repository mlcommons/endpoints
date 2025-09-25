"""
Load Generator for the MLPerf Inference Endpoint Benchmarking System.

This module handles load pattern generation and query lifecycle management.
Status: To be implemented by the development team.
"""

import queue
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

from ..config.ruleset import RuntimeSettings
from ..utils import sleep_ns
from .scheduler import Sample, SampleEvent, Scheduler


class SampleIssuer(ABC):
    def __init__(self):
        self.threads = {}
        self.queues = defaultdict(queue.Queue)

    def response_processor(self, sample: Sample):
        chunks = []
        while True:
            chunk = self.get_next_response_chunk(sample.uuid)
            if chunk is None or chunk == SampleEvent.COMPLETE:
                break
            # Trigger callback if exists
            if len(chunks) == 0 and SampleEvent.FIRST_CHUNK in sample.callbacks:
                sample.callbacks[SampleEvent.FIRST_CHUNK](chunk)
            elif len(chunks) > 0 and SampleEvent.NON_FIRST_CHUNK in sample.callbacks:
                sample.callbacks[SampleEvent.NON_FIRST_CHUNK](chunk)
            chunks.append(chunk)
        sample.callbacks[SampleEvent.COMPLETE](chunks)

    def get_next_response_chunk(self, s_uuid: int):
        return self.queues[s_uuid].get()

    def push_response_chunk(self, s_uuid: int, chunk: Any):
        self.queues[s_uuid].put(chunk)

    def setup_response_processor(self, sample: Sample):
        self.queues[sample.uuid] = queue.Queue()

    def issue(self, sample: Sample) -> dict[str, int]:
        self.setup_response_processor(sample)
        # Start response thread
        _t = threading.Thread(target=self.response_processor, args=(sample,))
        _t.start()
        self.threads[sample.uuid] = _t

        sample_data = sample.get_bytes()
        sample_process_start_ns = time.monotonic_ns()
        self.process_sample_data(sample.uuid, sample_data)
        sample_process_end_ns = time.monotonic_ns()
        # Trigger REQUEST_SENT event
        if SampleEvent.REQUEST_SENT in sample.callbacks:
            sample.callbacks[SampleEvent.REQUEST_SENT]()

        return {
            "sample_process_start_ns": sample_process_start_ns,
            "sample_process_end_ns": sample_process_end_ns,
        }

    def stop_response_thread(self, s_uuid: int):
        self.push_response_chunk(s_uuid, None)
        self.threads[s_uuid].join()
        del self.threads[s_uuid]
        del self.queues[s_uuid]

    @abstractmethod
    def process_sample_data(self, s_uuid: int, sample_data: Any):
        raise NotImplementedError


class BenchmarkSession:
    def __init__(self, rt_settings: RuntimeSettings):
        self.rt_settings = rt_settings
        self.start_time_ns = None

        self.test_end_event = threading.Event()
        self.thread: threading.Thread | None = None

    @property
    def min_end_time_ns(self):
        if self.start_time_ns is None:
            return None
        return int(self.start_time_ns + self.rt_settings.min_duration_ms * 1e6)

    @property
    def max_end_time_ns(self):
        if self.start_time_ns is None:
            return None
        return int(self.start_time_ns + self.rt_settings.max_duration_ms * 1e6)

    def mark_start_timestamp(self):
        self.start_time_ns = time.monotonic_ns()

    def wait_for_test_end(self):
        self.test_end_event.wait()
        self.thread.join(timeout=10)


class LoadGenerator:
    def __init__(self, scheduler: Scheduler, sample_issuer: SampleIssuer):
        self.scheduler = scheduler
        self.rt_settings = scheduler.runtime_settings
        self.sample_issuer = sample_issuer

    def start_test(self):
        sess = BenchmarkSession(self.rt_settings)
        sess.thread = threading.Thread(target=self._run_session, args=(sess,))
        sess.thread.start()
        return sess

    def _run_session(self, sess: BenchmarkSession):
        sess.mark_start_timestamp()
        last_issue_timestamp_ns = 0
        for sample, delay_ns in self.scheduler:
            scheduled_issue_time_ns = last_issue_timestamp_ns + delay_ns
            while True:
                now = time.monotonic_ns()
                if now >= scheduled_issue_time_ns:
                    break
                sleep_ns(scheduled_issue_time_ns - now)
            # TODO: Log return value of .issue()to session live statistics
            self.sample_issuer.issue(sample)
            last_issue_timestamp_ns = time.monotonic_ns()

            if last_issue_timestamp_ns >= sess.max_end_time_ns:
                warnings.warn(
                    "Session timeout reached. Stopping benchmark.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break
        sess.test_end_event.set()
