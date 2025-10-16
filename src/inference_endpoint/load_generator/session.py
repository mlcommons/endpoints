from __future__ import annotations

import threading
import uuid

from ..config.ruleset import RuntimeSettings
from ..dataset_manager.dataloader import DataLoader
from ..metrics.recorder import EventRecorder
from .events import SessionEvent
from .load_generator import LoadGenerator, SampleIssuer, SchedulerBasedLoadGenerator
from .sample import SampleFactory


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
        self.event_recorder = EventRecorder(
            session_id=self.session_id, idle_notify_th_ev=self.end_event
        )
        self.thread = None

    @property
    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def _run_test(
        self, load_generator: LoadGenerator, stop_sample_issuer_on_test_end: bool = True
    ):
        with self.event_recorder:
            for sample, issue_timestamp_ns in load_generator:
                self.event_recorder.record_event(
                    ev_type=SessionEvent.LG_ISSUE_CALLED,
                    timestamp_ns=issue_timestamp_ns,
                    sample_uuid=sample.uuid,
                )

            self.event_recorder.should_check_idle = True
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
        sample_factory_cls: type[SampleFactory] = SampleFactory,
        load_generator_cls: type[LoadGenerator] = SchedulerBasedLoadGenerator,
        name: str | None = None,
        stop_sample_issuer_on_test_end: bool = True,
    ) -> BenchmarkSession:
        session = cls(runtime_settings, session_id=name)
        sample_factory = sample_factory_cls(dataloader, session.event_recorder)
        load_generator = load_generator_cls(sample_issuer, sample_factory, *args)
        session.thread = threading.Thread(
            target=session._run_test,
            args=(load_generator, stop_sample_issuer_on_test_end),
        )
        session.thread.start()
        return session
