from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class _Dataset:
    # TODO: Expand this class to include more dataset metadata
    description: str
    """The name of the dataset, or a description if it is a custom dataset"""

    size: int
    """The number of unique samples in the dataset"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the dataset"""

    def __hash__(self) -> int:
        # Datasets are used as pseudo-'Enum' values, so we can use id as hash
        return id(self)


OpenOrca = _Dataset(
    description="OpenOrca",
    size=24576,
    metadata={
        "max_seq_len": 1024
    }
)

CNNDailyMail = _Dataset(
    description="CNNDailyMail v3.0.0",
    size=13368,
    metadata={
        "max_seq_len": 2048
    }
)

TextGenLongSeqLen = _Dataset(
    description="Subset of LongBench, LongDataCollections, Ruler, GovReport",
    size=8313,
)

TextGenComplex = _Dataset(
    description="Subset of OpenOrca (5k samples), GSM8K (5k samples from train split), MBXP (5k samples) for QA, Math, and Code Generation",
    size=15000,
    metadata={
        "max_seq_len": 2048
    }
)

MLPerfDeepseekR1 = _Dataset(
    description="Custom dataset curated by MLCommons for DeepSeek R1, specifically for the MLPerf Inference benchmark",
    size=4388,
)


# Note this isn't completely robust, but will prevent simple cases of defining new instances
def _disallow_instantiation(cls, *args, **kwargs):
    raise TypeError("Cannot instantiate _Dataset directly. Use a pre-defined dataset from this module.")
_Dataset.__new__ = _disallow_instantiation
