# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import logging
import threading
import time
import uuid

from ..config.ruleset import RuntimeSettings
from ..dataset_manager.dataloader import DataLoader
from ..metrics.recorder import EventRecorder
from .events import SessionEvent
from .load_generator import LoadGenerator, SampleIssuer, SchedulerBasedLoadGenerator


class BenchmarkSession:
    def __init__(
        self,
        runtime_settings: RuntimeSettings,
        session_id: str | None = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.runtime_settings = runtime_settings
        self.session_id = session_id if session_id else uuid.uuid4().hex

        self.end_event = threading.Event()
        self.thread = None

        self.sample_uuid_map = {}
        self.event_recorder = EventRecorder(
            session_id=self.session_id, notify_idle=self.end_event
        )

    @property
    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def _run_test(
        self,
        load_generator: LoadGenerator,
        stop_sample_issuer_on_test_end: bool = True,
        max_shutdown_timeout_s: float = 300.0,
    ):
        with self.event_recorder:
            EventRecorder.record_event(SessionEvent.TEST_STARTED, time.monotonic_ns())
            for issued_sample in load_generator:
                # In the future, we'll want to push this to some thread or process that
                # performs output verification / accuracy checks.
                self.sample_uuid_map[issued_sample.sample.uuid] = issued_sample

            self.event_recorder.should_check_idle = True
            EventRecorder.record_event(SessionEvent.LOADGEN_STOP, time.monotonic_ns())
            start_time = time.monotonic()
            while self.event_recorder.n_inflight_samples != 0:
                if (
                    max_shutdown_timeout_s is not None
                    and time.monotonic() - start_time > max_shutdown_timeout_s
                ):
                    raise TimeoutError(
                        f"Max shutdown timeout of {max_shutdown_timeout_s}s reached"
                    )
                self.end_event.wait(timeout=10.0)
                self.logger.info(
                    f"Waiting for the test to end... {self.event_recorder.n_inflight_samples} samples remaining"
                )

            if stop_sample_issuer_on_test_end:
                load_generator.sample_issuer.shutdown()
            EventRecorder.record_event(SessionEvent.TEST_ENDED, time.monotonic_ns())

    def wait_for_test_end(self, timeout: float | None = None) -> bool:
        """
        Join the test thread and return True if the test completed, False if it timed out.

        Args:
            timeout: The maximum time to wait for the test to complete. If None, wait indefinitely.

        Returns:
            bool: True if the test thread has completed, False if it timed out.
        """
        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()

    @classmethod
    def start(
        cls,
        runtime_settings: RuntimeSettings,
        dataloader: DataLoader,
        sample_issuer: SampleIssuer,
        *args,
        load_generator_cls: type[LoadGenerator] = SchedulerBasedLoadGenerator,
        name: str | None = None,
        stop_sample_issuer_on_test_end: bool = True,
        max_shutdown_timeout_s: float = 300.0,
    ) -> BenchmarkSession:
        """Start a new BenchmarkSession in a thread.

        Args:
            runtime_settings: The runtime settings to use for the session.
            dataloader: The dataloader to use for the session.
            sample_issuer: The sample issuer to use for the session.
            load_generator_cls: The load generator class to use for the session.
            name: The name of the session.
            stop_sample_issuer_on_test_end: Whether to stop the sample issuer on test end.
            max_shutdown_timeout_s: The maximum timeout to wait for the test to complete after all samples have been issued.
                                    If None, wait indefinitely. (Default: 300.0 seconds)

        Returns:
            The new BenchmarkSession.
        """
        session = cls(runtime_settings, session_id=name)
        load_generator = load_generator_cls(sample_issuer, dataloader, *args)
        session.thread = threading.Thread(
            target=session._run_test,
            args=(
                load_generator,
                stop_sample_issuer_on_test_end,
                max_shutdown_timeout_s,
            ),
        )
        session.thread.start()
        return session
