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

from __future__ import annotations

import inspect
import logging
import os
import threading
import time
import uuid
from pathlib import Path

import orjson
from transformers import AutoTokenizer

from ..config.runtime_settings import RuntimeSettings
from ..dataset_manager.dataset import Dataset
from ..metrics.recorder import EventRecorder
from ..metrics.reporter import MetricsReporter
from .events import SessionEvent
from .load_generator import LoadGenerator, SampleIssuer, SchedulerBasedLoadGenerator
from .scheduler import Scheduler, WithoutReplacementSampleOrder

logger = logging.getLogger(__name__)

# poll interval for checking if test-session should end
SHUTDOWN_POLL_INTERVAL_S = 10.0


class BenchmarkSession:
    def __init__(
        self,
        runtime_settings: RuntimeSettings,
        session_id: str | None = None,
        db_backend: str = "sqlite",
        db_conninfo: str | None = None,
    ):
        self.logger = logging.getLogger(__name__)
        self.runtime_settings = runtime_settings
        self.session_id = session_id if session_id else uuid.uuid4().hex
        self.db_backend = db_backend
        self.db_conninfo = db_conninfo

        # EventRecorder will set this when all samples complete, helps avoid busy-waiting
        self.end_event = threading.Event()
        self.thread = None

        # CPython GIL provides atomic boolean writes, no need for threading.Event()
        self.stop_requested = False

        pg_storage = None
        if db_backend == "postgres":
            from ..storage.db import PostgresBackend

            pg_storage = PostgresBackend(conninfo=db_conninfo)  # WRITER

        self.event_recorder = EventRecorder(
            session_id=self.session_id,
            notify_idle=self.end_event,
            backend=db_backend,
            storage=pg_storage,
        )
        # Will be populated after the test finishes by _run_test
        self.report = None

        self.sample_uuid_map = None

    @property
    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def stop(self) -> None:
        """Signal the session to stop early."""
        self.stop_requested = True
        # wakeup _run_test if needed, short-circuit SHUTDOWN_POLL_INTERVAL_S
        self.end_event.set()

    def _run_test(  # _run_thread called as param to a thread at the base of this file
        self,
        perf_test_generator: LoadGenerator,
        accuracy_test_generators: dict[str, LoadGenerator] | None = None,
        max_shutdown_timeout_s: float = 300.0,
        report_dir: os.PathLike | None = None,
        tokenizer_override: AutoTokenizer | None = None,
        dump_events_log: bool = False,
    ):
        print("\nSTART of _run_test\n")

        with self.event_recorder:
            try:
                EventRecorder.record_event(
                    SessionEvent.TEST_STARTED, time.monotonic_ns()
                )

                # it looks like this issues requests to host LLM
                #
                for _ in (
                    perf_test_generator
                ):  # LoadGenerator is passed in and run here; generates some data
                    # Actual issue is done during next(generator). Nothing else to do here, just pass.
                    pass

                EventRecorder.record_event(
                    SessionEvent.STOP_PERFORMANCE_TRACKING, time.monotonic_ns()
                )
                self.logger.info("All performance samples issued")

                if accuracy_test_generators:
                    for _, generator in accuracy_test_generators.items():
                        for _ in generator:
                            # Actual issue is done during next(generator). Nothing else to do here, just pass.
                            pass

                self.logger.info("All accuracy samples issued")

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

                    if self.stop_requested:
                        self.logger.info(
                            f"Early stop requested (pending={self.event_recorder.n_inflight_samples}), shutting down test..."
                        )
                        break

                    self.end_event.wait(timeout=SHUTDOWN_POLL_INTERVAL_S)
                    self.logger.info(
                        f"Waiting for the test to end... {self.event_recorder.n_inflight_samples} samples remaining"
                    )

            except Exception as e:
                logger.error(f"Error running benchmark session: {e}")
                raise e
            finally:
                EventRecorder.record_event(SessionEvent.TEST_ENDED, time.monotonic_ns())

            print("before _runtest.wait_for_write()")  # we are here
            self.event_recorder.wait_for_writes()
            print("after .wait_for_write()")  # we see this

            # Handle reporting
            if self.db_backend == "postgres":
                reporter_client = "postgres"
                reporter_table = self.event_recorder.table_name
                from ..storage.db import PostgresBackend

                reporter_storage = PostgresBackend(conninfo=self.db_conninfo)  # READER
            else:
                reporter_client = "duckdb"
                reporter_table = "events"
                reporter_storage = None

            with MetricsReporter(
                self.event_recorder.connection_name,
                client_type=reporter_client,
                table_name=reporter_table,
                storage=reporter_storage,
            ) as reporter:  # reporter has JSON data we will write down for
                has_model = hasattr(
                    self.runtime_settings, "model"
                )  #      result_summary.json
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

                # here is where report is created
                report = reporter.create_report(
                    tokenizer
                )  # call into reporter.create_report()  I think this is all
                #    created using DB. Used below to create result_summary.json

                # Store report on session so external callers can use it
                self.report = report

                # Consolidate UUID->index mappings
                perf_name = (
                    perf_test_generator.name
                    if perf_test_generator.name
                    else "performance"
                )

                # perf_test_generator data stored here and
                # eventually written down to   <>.json           AGAIN it does not look like this comes from DB. Inline as test is run.
                #
                #####################################################
                sample_idx_map = {  # data written to 'sample_idx_map.json'
                    perf_name: perf_test_generator.uuid_to_index_map,  # not clear where this data comes from - PostGres ??  passed in.
                }
                if accuracy_test_generators:
                    for default_name, generator in accuracy_test_generators.items():
                        name = generator.name if generator.name else default_name
                        sample_idx_map[name] = generator.uuid_to_index_map
                self.sample_uuid_map = sample_idx_map

                # Save to report directory if provided
                if report_dir:
                    print("\nsave files to report dir /tmp/report* \n")
                    Path(report_dir).mkdir(parents=True, exist_ok=True)

                    curr_method = inspect.currentframe().f_code.co_name
                    print(f"SAVE result_summary.json call from {curr_method}")
                    report.to_json(
                        save_to=Path(report_dir) / "result_summary.json"
                    )  ## save result_summary.json   TTFT, TPOT, etc

                    # Dump runtime settings to report directory
                    rt_settings_data = {  # this is the data to be saved in 'runtime_settings.json'
                        "min_duration_ms": self.runtime_settings.min_duration_ms,  # DB not required for this; defined run params
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

                    curr_method = inspect.currentframe().f_code.co_name
                    print(f"SAVE runtime_settings.json call from {curr_method}")
                    with (Path(report_dir) / "runtime_settings.json").open("w") as f:
                        f.write(
                            orjson.dumps(  # orjson - fast json    Passed in; NOT from DB
                                rt_settings_data,  # this is the data written down to 'runtime_settings.json'
                                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS,
                            ).decode("utf-8")
                        )

                    # Save the UUID mapping for output verification       Data for this JSON is a map UUID -> index; created by the generator as it
                    #                                                       is creating requests    IE not read from DB
                    curr_method = inspect.currentframe().f_code.co_name
                    print(f"SAVE sample_idx_map.json call from {curr_method}")
                    with (Path(report_dir) / "sample_idx_map.json").open("w") as f:
                        f.write(orjson.dumps(self.sample_uuid_map).decode("utf-8"))  #

                    curr_method = inspect.currentframe().f_code.co_name
                    print(f"SAVE events.jsonl call from {curr_method}")
                    if dump_events_log:
                        reporter.dump_to_json(
                            Path(report_dir) / "events.jsonl"
                        )  # biggest file; looks like the individual transactions;
                        #   comes FROM DB
                # Print summary
                print("dump out summary data which in turn should save files to /tmp")
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
        dataset: Dataset,
        sample_issuer: SampleIssuer,
        scheduler: Scheduler,
        *args,
        accuracy_datasets: list[Dataset] | None = None,
        load_generator_cls: type[
            LoadGenerator
        ] = SchedulerBasedLoadGenerator,  # see load_generator.py
        name: str | None = None,
        max_shutdown_timeout_s: float = 300.0,
        report_dir: os.PathLike | None = None,
        tokenizer_override: AutoTokenizer | None = None,
        dump_events_log: bool = False,
        db_backend: str = "sqlite",
        db_conninfo: str | None = None,
    ) -> BenchmarkSession:
        """Start a new BenchmarkSession in a thread.

        Args:
            runtime_settings: The runtime settings to use for the session.
            dataset: The dataset to use for the performance test.
            sample_issuer: The sample issuer to use for the session.
            scheduler: The scheduler to use for the session.
            accuracy_datasets: The datasets to use for the accuracy tests. If None, no accuracy tests will be run.
            load_generator_cls: The load generator class to use for the session.
            name: The name of the session.
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
        session = cls(
            runtime_settings,
            session_id=name,
            db_backend=db_backend,
            db_conninfo=db_conninfo,
        )
        load_generator = load_generator_cls(sample_issuer, dataset, scheduler, *args)

        # Create accuracy test generators
        accuracy_test_generators = None
        if accuracy_datasets:
            accuracy_test_generators = {}
            for ds in accuracy_datasets:
                if hasattr(ds.__class__, "DATASET_ID"):
                    ds_name = ds.__class__.DATASET_ID
                else:
                    ds_name = ds.__class__.__name__

                # Create accuracy dataset specific runtime settings
                acc_rt_settings = RuntimeSettings(
                    metric_target=runtime_settings.metric_target,
                    reported_metrics=runtime_settings.reported_metrics,
                    min_duration_ms=0,
                    max_duration_ms=None,
                    n_samples_from_dataset=ds.num_samples(),
                    n_samples_to_issue=ds.num_samples() * ds.repeats,
                    min_sample_count=ds.num_samples() * ds.repeats,
                    rng_sched=runtime_settings.rng_sched,
                    rng_sample_index=runtime_settings.rng_sample_index,
                    load_pattern=runtime_settings.load_pattern,
                )
                acc_sched = scheduler.__class__(
                    acc_rt_settings, WithoutReplacementSampleOrder
                )

                accuracy_test_generators[ds_name] = load_generator_cls(
                    sample_issuer, ds, acc_sched, *args
                )

        session.thread = threading.Thread(
            target=session._run_test,  # _run_test started here
            args=(load_generator,),  # pass iterator in
            kwargs={
                "accuracy_test_generators": accuracy_test_generators,
                "max_shutdown_timeout_s": max_shutdown_timeout_s,
                "report_dir": report_dir,
                "tokenizer_override": tokenizer_override,
                "dump_events_log": dump_events_log,
            },
        )
        session.thread.start()
        return session
