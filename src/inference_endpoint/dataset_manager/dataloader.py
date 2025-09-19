from abc import ABC, abstractmethod
from typing import Any


class DataLoader(ABC):
    """Implementation for loading a dataset into memory and handling memory management for the dataset.

    For large datasets that cannot entirely fit into memory, the dataloader should implement some strategy for loading and unloading
    required samples into memory as needed.

    For any given sample index, the sample can be required to be in memory any number of times during the lifetime of a benchmark run.
    """

    def __init__(self, max_memory_usage_bytes: int | None = None):
        """
        Initializes the dataloader.

        Args:
            max_memory_usage_bytes (int | None): The maximum memory usage in bytes. If None, the dataloader will not limit memory usage artificially.
        """
        self.max_memory_usage_bytes = max_memory_usage_bytes

    @abstractmethod
    def load_sample(self, index: int) -> Any:
        """
        Loads a sample from the dataset.
        """
        raise NotImplementedError

    def mark_unneeded(self, index: int):
        """
        Marks a sample as no longer needed. The DataLoader implementation should implement some strategy for unloading this sample from memory
        as necessary (i.e. when the memory limit is reached and load_sample is called).

        It is not necessary to implement this method if the DataLoader implementation requires or assumes the entire dataset can fit in memory.
        """
        pass
