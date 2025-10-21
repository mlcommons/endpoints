from __future__ import annotations

import threading
import time
import uuid

from ..config.ruleset import RuntimeSettings
from ..dataset_manager.dataloader import DataLoader
from ..metrics.recorder import EventRecorder
from .events import SessionEvent
from .load_generator import LoadGenerator, SampleIssuer, SchedulerBasedLoadGenerator
from .sample import Sample


class BenchmarkSession:
    def __init__(
        self,
        runtime_settings: RuntimeSettings,
        session_id: str | None = None,
    ):
        self.runtime_settings = runtime_settings
        if session_id:
            self.session_id = session_id
        else:
            self.session_id = uuid.uuid4().hex

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
    ):
        with self.event_recorder:
            for issued_sample in load_generator:
                # In the future, we'll want to push this to some thread or process that
                # performs output verification / accuracy checks.
                self.sample_uuid_map[issued_sample.sample.uuid] = issued_sample

            self.event_recorder.should_check_idle = True
            EventRecorder.record_event(SessionEvent.LOADGEN_STOP, time.monotonic_ns())
            while self.event_recorder.n_inflight_samples != 0:
                self.end_event.wait(timeout=10.0)

            if stop_sample_issuer_on_test_end:
                load_generator.sample_issuer.shutdown()

    def wait_for_test_end(self):
        self.thread.join()
        self.thread = None

    @classmethod
    def start(
        cls,
        runtime_settings: RuntimeSettings,
        dataloader: DataLoader,
        sample_issuer: SampleIssuer,
        *args,
        sample_class: type[Sample] = Sample,
        load_generator_cls: type[LoadGenerator] = SchedulerBasedLoadGenerator,
        name: str | None = None,
        stop_sample_issuer_on_test_end: bool = True,
    ) -> BenchmarkSession:
        session = cls(runtime_settings, session_id=name)
        load_generator = load_generator_cls(
            sample_issuer, sample_class, dataloader, *args
        )
        session.thread = threading.Thread(
            target=session._run_test,
            args=(load_generator, stop_sample_issuer_on_test_end),
        )
        session.thread.start()
        return session
