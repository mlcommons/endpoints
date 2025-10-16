import random
from collections.abc import Callable

from inference_endpoint.load_generator.events import SampleEvent
from inference_endpoint.load_generator.sample import SampleFactory

from tests.test_helpers import DummyDataLoader


def test_sample_factory():
    completed = []

    # Create dataloader instance
    dummy_dataloader = DummyDataLoader(n_samples=100)

    class TestingFactory(SampleFactory):
        def get_sample_callbacks(
            self, sample_index: int, sample_uuid: int
        ) -> dict[SampleEvent, Callable]:
            return {SampleEvent.COMPLETE: lambda _: completed.append(sample_index)}

    factory = TestingFactory(dummy_dataloader, None)

    indices = list(range(dummy_dataloader.n_samples))
    random.shuffle(indices)

    uuids = set()
    for idx in indices:
        sample = factory(idx)
        assert sample.uuid not in uuids, "UUIDs should be unique but found duplicate"
        uuids.add(sample.uuid)

        obj = sample.get_bytes()
        assert obj == idx, "Sample 'bytes' should be equal to index for DummyDataLoader"

        sample.callbacks[SampleEvent.COMPLETE](
            None
        )  # Value is ignored for testing callback
        assert len(completed) == len(
            uuids
        ), "Completed callback should be called for each sample"
        assert (
            completed[-1] == idx
        ), "Completed callback should be called with correct sample index"
