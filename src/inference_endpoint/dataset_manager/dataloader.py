import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from logging import getLogger
from typing import Any

import pandas
from datasets import load_dataset, load_from_disk


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

    def mark_unneeded(self, index: int):  # noqa: B027
        """
        Marks a sample as no longer needed. The DataLoader implementation should implement some strategy for unloading this sample from memory
        as necessary (i.e. when the memory limit is reached and load_sample is called).

        It is not necessary to implement this method if the DataLoader implementation requires or assumes the entire dataset can fit in memory.
        """
        pass

    @abstractmethod
    def num_samples(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        raise NotImplementedError


class PickleReader(DataLoader):
    def __init__(
        self,
        file_path,
        parser: Callable[[Any], Any] = None,
        max_memory_usage_bytes: int | None = None,
    ):
        """
        Initializes the PickleReader.
        This does not load the data from the pickle file. Call load() to load the data.

        Args:
            file_path (str): The path to the pickle file.
            parser (Callable[[Any], Any]): A function to extract the data from the row.
        """
        super().__init__(max_memory_usage_bytes)
        self.file_path = file_path
        self.data = []
        # self.text_inputs = []
        self.loaded = False
        self.parser = parser
        self.logger = getLogger(__name__)
        if parser is None:

            def extract_text_input(row):
                return row.text_input

            self.parser = extract_text_input

    def load(self, force: bool = False):
        """
        Loads the data from the pickle file.
        If force is True, it will reload the data even if it is already loaded.
        """
        if self.loaded and not force:
            return
        with open(self.file_path, "rb") as file:
            self.data = pickle.load(file)
            self.logger.info(
                f"Loading data from {self.file_path} with columns: {self.data.columns}"
            )
            assert "text_input" in self.data.columns
            self.text_inputs = [None] * len(self.data)
            # this preloads the  data in source
            # for idx, data in self.data.iterrows():
            #     # idx is not passed to the parser since it should _not_ be used in the parser
            #     self.text_inputs[idx] = self.parser(data)
        self.loaded = True

    def num_samples(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

    def write_samples(self, file_path: str, indices: list[int]):
        """
        Utility function - writes the samples to a pickle file.
        """
        if not self.loaded:
            self.load()
        samples = self.data.iloc[indices]
        with open(file_path, "wb") as file:
            # for index in indices:
            pickle.dump(samples, file)

    def write_unique_samples(self, file_path: str):
        """ "
        Utility function - writes the unique samples to a pickle file.
        """
        dataset_sources = {"math500", "aime1983", "livecodebench", "gpqa", "mmlu_pro"}
        samples = pandas.DataFrame(columns=self.data.columns)
        for dataset_source in dataset_sources:
            filtered = self.data[self.data["dataset"] == dataset_source]
            samples = pandas.concat([samples, filtered.iloc[[0]]], ignore_index=True)

        with open(file_path, "wb") as file:
            # for sample in samples:
            pickle.dump(samples, file)

    def load_sample(self, index: int) -> Any:
        """
        Loads a sample from the data.
        """
        assert self.loaded, "Data is not loaded. Call load() to load the data."
        return self.parser(self.data.iloc[index])

    def get_column_names(self):
        return self.data.columns


class HFDataLoader(DataLoader):
    def __init__(
        self,
        dataset_name,
        *,
        parser: Callable[[Any], Any] = None,
        split: str = "train",
        format: str | None = None,
        max_memory_usage_bytes: int | None = None,
    ):
        super().__init__(max_memory_usage_bytes)
        self.logger = getLogger(__name__)
        self.dataset_name = dataset_name
        self.data = []
        self.text_inputs = []
        self.parser = parser
        self.split = split
        self.format = format
        if parser is None:

            def extract_row(row):
                return row  # by default, return the training data which is a dictionary

            self.parser = extract_row

    def load(self):
        if self.format is None:
            self.data = load_dataset(self.dataset_name)
        else:
            # huggingface uses a different method to load local arrow datasets
            self.data = load_from_disk(self.dataset_name)

    def load_sample(self, index: int) -> Any:
        return self.parser(self.data[self.split][index])

    def num_samples(self):
        return len(self.data[self.split])
