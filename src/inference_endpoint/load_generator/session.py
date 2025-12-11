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
import os
import threading
import time
import uuid
from pathlib import Path

import orjson
from transformers import AutoTokenizer

from ..config.runtime_settings import RuntimeSettings
from ..dataset_manager.dataloader import DataLoader
from ..metrics.recorder import EventRecorder
from ..metrics.reporter import MetricsReporter
from .events import SessionEvent
from .load_generator import LoadGenerator, SampleIssuer, SchedulerBasedLoadGenerator

logger = logging.getLogger(__name__)


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

        self.event_recorder = EventRecorder(
            session_id=self.session_id, notify_idle=self.end_event
        )

        self.sample_uuid_map = None

    @property
    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def _run_test(
        self,
        perf_test_generator: LoadGenerator,
        accuracy_test_generators: dict[str, LoadGenerator] | None = None,
        stop_sample_issuer_on_test_end: bool = True,
        max_shutdown_timeout_s: float = 300.0,
        report_dir: os.PathLike | None = None,
        tokenizer_override: AutoTokenizer | None = None,
        dump_events_csv: bool = False,
    ):
        with self.event_recorder:
            try:
                EventRecorder.record_event(
                    SessionEvent.TEST_STARTED, time.monotonic_ns()
                )

                for _ in perf_test_generator:
                    pass

                EventRecorder.record_event(
                    SessionEvent.STOP_PERFORMANCE_TRACKING, time.monotonic_ns()
                )

                if accuracy_test_generators:
                    for _, generator in accuracy_test_generators.items():
                        for _ in generator:
                            pass

                self.event_recorder.should_check_idle = True
                EventRecorder.record_event(
                    SessionEvent.LOADGEN_STOP, time.monotonic_ns()
                )
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
            except Exception as e:
                logger.error(f"Error running benchmark session: {e}")
                raise e
            finally:
                if stop_sample_issuer_on_test_end:
                    perf_test_generator.sample_issuer.shutdown()
                EventRecorder.record_event(SessionEvent.TEST_ENDED, time.monotonic_ns())

            self.event_recorder.wait_for_writes()

            # Handle reporting
            with MetricsReporter(self.event_recorder.connection_name) as reporter:
                has_model = hasattr(self.runtime_settings, "model")
                tokenizer = None
                if tokenizer_override is not None:
                    tokenizer = tokenizer_override
                if has_model:
                    model = self.runtime_settings.model
                    if tokenizer is None:
                        try:
                            tokenizer = AutoTokenizer.from_pretrained(
                                model if isinstance(model, str) else model.name
                            )
                        except Exception as e:
                            logger.error(
                                f"Error loading tokenizer for model {model}: {e}"
                            )
                            tokenizer = None
                report = reporter.create_report(tokenizer)

                # Consolidate UUID->index mappings
                sample_idx_map = {
                    "performance": perf_test_generator.uuid_to_index_map,
                }
                if accuracy_test_generators:
                    for name, generator in accuracy_test_generators.items():
                        sample_idx_map[name] = generator.uuid_to_index_map
                self.sample_uuid_map = sample_idx_map

                # Save to report directory if provided
                if report_dir:
                    Path(report_dir).mkdir(parents=True, exist_ok=True)
                    report.to_json(save_to=Path(report_dir) / "result_summary.json")

                    # Dump runtime settings to report directory
                    rt_settings_data = {
                        "min_duration_ms": self.runtime_settings.min_duration_ms,
                        "max_duration_ms": self.runtime_settings.max_duration_ms,
                        "n_samples_from_dataset": self.runtime_settings.n_samples_from_dataset,
                        "n_samples_to_issue": self.runtime_settings.n_samples_to_issue,
                        "min_sample_count": self.runtime_settings.min_sample_count,
                        "total_samples_to_issue": self.runtime_settings.total_samples_to_issue(),
                    }
                    # TODO: Since RuntimeSettings stores the random.Random objects directly, there is no way
                    # to retrieve the seed values. The best way to do this is probably a custom random.Random
                    # class that stores the original seed as a read-only property, and unable to set the seed
                    # after initialization.
                    if has_model:
                        rt_settings_data["model"] = (
                            model if isinstance(model, str) else model.name
                        )

                    # TODO: After Zhihan's MR is merged, grab the scheduler class and other LG init settings
                    # from the runtime settings object
                    with (Path(report_dir) / "runtime_settings.json").open("w") as f:
                        f.write(
                            orjson.dumps(
                                rt_settings_data,
                                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
                            ).decode("utf-8")
                        )

                    # Save the UUID mapping for output verification
                    with (Path(report_dir) / "sample_idx_map.json").open("w") as f:
                        f.write(orjson.dumps(self.sample_uuid_map).decode("utf-8"))

                    if dump_events_csv:
                        reporter.dump_to_csv(Path(report_dir) / "events.csv")

                # Print summary
                report.display()

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
        report_dir: os.PathLike | None = None,
        tokenizer_override: AutoTokenizer | None = None,
        dump_events_csv: bool = False,
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
            report_dir: The path to save the report to. If None, no report will be saved.
            tokenizer_override: The tokenizer to use for the session. If None, a tokenizer will be automatically selected
                                based on the model name in the runtime settings.
            dump_events_csv: Whether to dump the events to a CSV file. Only use for debugging
                             purposes, as the events database can get quite large.

        Returns:
            The new BenchmarkSession.
        """
        session = cls(runtime_settings, session_id=name)
        load_generator = load_generator_cls(sample_issuer, dataloader, *args)
        session.thread = threading.Thread(
            target=session._run_test,
            args=(load_generator,),
            kwargs={
                "stop_sample_issuer_on_test_end": stop_sample_issuer_on_test_end,
                "max_shutdown_timeout_s": max_shutdown_timeout_s,
                "report_dir": report_dir,
                "tokenizer_override": tokenizer_override,
                "dump_events_csv": dump_events_csv,
            },
        )
        session.thread.start()
        return session
