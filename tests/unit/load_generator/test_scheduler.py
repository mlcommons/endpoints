import random

from inference_endpoint.core.types import QueryResult
from inference_endpoint.dataset_manager.dataloader import DataLoader
from inference_endpoint.load_generator.scheduler import (
    SampleEvent,
    SampleFactory,
    WithoutReplacementSampleOrder,
    WithReplacementSampleOrder,
)


def test_sample_factory(dummy_dataloader: DataLoader):
    completed = []

    class TestingFactory(SampleFactory):
        @staticmethod
        def sample_complete_callback(output, sid=None):
            completed.append(sid)

    factory = TestingFactory(dummy_dataloader)

    indices = list(range(dummy_dataloader.n_samples))
    random.shuffle(indices)

    uuids = set()
    for idx in indices:
        sample = factory(idx)
        assert sample.uuid not in uuids, "UUIDs should be unique but found duplicate"
        uuids.add(sample.uuid)

        obj = sample.get_bytes()
        assert obj == idx, "Sample 'bytes' should be equal to index for DummyDataLoader"

        result = QueryResult(id=sample.uuid, response_output=None)
        sample.callbacks[SampleEvent.COMPLETE](result)
        assert len(completed) == len(
            uuids
        ), "Completed callback should be called for each sample"
        assert (
            completed[-1] == idx
        ), "Completed callback should be called with correct sample index"


def test_without_replacement_sample_order():
    ordering = WithoutReplacementSampleOrder(12345, 100)
    indices = list(iter(ordering))
    for i in range(0, 12345, 100):
        assert len(set(indices[i : i + 100])) == min(
            100, 12345 - i
        ), "Indices should be unique, and occur at least once"

    # Assert that order is different in each pass of the dataset
    assert (
        indices[:100] != indices[100:200]
    ), "Order should be different in each pass of the dataset"


def test_with_replacement_sample_order():
    ordering = WithReplacementSampleOrder(12345, 100, rng=random.Random(42))
    indices = list(iter(ordering))

    # With Python random.Random(42), the order can be deterministic
    assert indices[:10] == [
        81,
        14,
        3,
        94,
        35,
        31,
        28,
        17,
        94,
        13,
    ], "Order does not match expected deterministic order"
    # Note with this specific seed and order, 94 occurs twice in the first 10 indices
    assert indices[:10].count(94) == 2, "94 should occur twice in the first 10 indices"
