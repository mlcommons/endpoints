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

import logging
import os
import threading
import uuid
from pathlib import Path

import msgspec.json

from ..config.runtime_settings import RuntimeSettings
from ..dataset_manager.dataset import Dataset
from .load_generator import LoadGenerator, SampleIssuer, SchedulerBasedLoadGenerator
from .scheduler import Scheduler, WithoutReplacementSampleOrder

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
        self.thread: threading.Thread | None = None

        # CPython GIL provides atomic boolean writes, no need for threading.Event()
        self.stop_requested = False

        # Will be populated after the test finishes by _run_test
        self.report = None

        self.sample_uuid_map: dict[str, dict[str, int]] | None = None

    @property
    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def stop(self) -> None:
        """Signal the session to stop early."""
        self.stop_requested = True
        # wakeup _run_test if needed, short-circuit SHUTDOWN_POLL_INTERVAL_S
        self.end_event.set()

    def _run_test(
        self,
        perf_test_generator: LoadGenerator,
        accuracy_test_generators: dict[str, LoadGenerator] | None = None,
        report_dir: os.PathLike | None = None,
    ):
        try:
            for _ in perf_test_generator:
                pass

            self.logger.info("All performance samples issued")

            if accuracy_test_generators:
                for _, generator in accuracy_test_generators.items():
                    for _ in generator:
                        pass

            self.logger.info("All accuracy samples issued")

            # TODO: Wire in EventPublisherService + ServiceLauncher (Phase 5)
            # For now, no event recording or report generation.

        except Exception as e:
            logger.error(f"Error running benchmark session: {e}")
            raise e

        # Consolidate UUID->index mappings
        perf_name = (
            perf_test_generator.name if perf_test_generator.name else "performance"
        )
        sample_idx_map = {
            perf_name: perf_test_generator.uuid_to_index_map,
        }
        if accuracy_test_generators:
            for default_name, generator in accuracy_test_generators.items():
                name = generator.name if generator.name else default_name
                sample_idx_map[name] = generator.uuid_to_index_map
        self.sample_uuid_map = sample_idx_map

        # Save runtime settings and UUID map if report_dir provided
        if report_dir:
            Path(report_dir).mkdir(parents=True, exist_ok=True)

            rt_settings_data: dict[str, int | str | None] = {
                "min_duration_ms": self.runtime_settings.min_duration_ms,
                "max_duration_ms": self.runtime_settings.max_duration_ms,
                "n_samples_from_dataset": self.runtime_settings.n_samples_from_dataset,
                "n_samples_to_issue": self.runtime_settings.n_samples_to_issue,
                "min_sample_count": self.runtime_settings.min_sample_count,
                "total_samples_to_issue": self.runtime_settings.total_samples_to_issue(),
            }
            has_model = hasattr(self.runtime_settings, "model")
            if has_model:
                model = getattr(self.runtime_settings, "model", None)
                if model is not None:
                    rt_settings_data["model"] = (
                        model if isinstance(model, str) else str(model.name)
                    )

            with (Path(report_dir) / "runtime_settings.json").open("w") as f:
                f.write(
                    msgspec.json.format(
                        msgspec.json.encode(dict(sorted(rt_settings_data.items()))),
                        indent=2,
                    ).decode("utf-8")
                )

            with (Path(report_dir) / "sample_idx_map.json").open("w") as f:
                f.write(msgspec.json.encode(self.sample_uuid_map).decode("utf-8"))

    def wait_for_test_end(self, timeout: float | None = None) -> bool:
        """
        Join the test thread and return True if the test completed, False if it timed out.

        Args:
            timeout: The maximum time to wait for the test to complete. If None, wait indefinitely.

        Returns:
            bool: True if the test thread has completed, False if it timed out.
        """
        if not self.thread:
            return False
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
        load_generator_cls: type[LoadGenerator] = SchedulerBasedLoadGenerator,
        name: str | None = None,
        report_dir: os.PathLike | None = None,
    ) -> BenchmarkSession:
        """Start a new BenchmarkSession in a thread.

        Args:
            runtime_settings: The runtime settings to use for the session.
            dataset: The dataset to use for the performance test.
            sample_issuer: The sample issuer to use for the session.
            scheduler: The scheduler to use for the session.
            accuracy_datasets: The datasets to use for the accuracy tests.
            load_generator_cls: The load generator class to use for the session.
            name: The name of the session.
            report_dir: The path to save the report to. If None, no report will be saved.

        Returns:
            The new BenchmarkSession.
        """
        session = cls(runtime_settings, session_id=name)
        load_generator = load_generator_cls(sample_issuer, dataset, scheduler, *args)  # type: ignore[arg-type]

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
                    sample_issuer,
                    ds,
                    acc_sched,  # type: ignore[arg-type]
                    *args,
                )

        session.thread = threading.Thread(
            target=session._run_test,
            args=(load_generator,),
            kwargs={
                "accuracy_test_generators": accuracy_test_generators,
                "report_dir": report_dir,
            },
        )
        session.thread.start()
        return session
