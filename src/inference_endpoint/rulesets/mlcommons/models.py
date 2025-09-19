from dataclasses import dataclass

from . import datasets


@dataclass(frozen=True)
class _Model:
    name: str
    """The name of the benchmark model"""

    golden_accuracy: tuple[str, dict[str, float | tuple[float, float]]]
    """The accuracy(ies) of the model with full precision, without any modifications to the model (i.e. quantization).
    The first element is a string indicating the precision used to obtain the accuracy values.
    The second element is a dictionary of metric names and their accuracy values.
    Accuracy values can either be a float or a 2-element tuple indicating an inclusive range of acceptable values."""

    accuracy_target_settings: list[dict[str, tuple[float, ...]]]
    """Possible accuracy settings for the model to run in. Schema:
    {
        "metric_name1": (lower_bound_factor,),  # Implied no upper bound
        "metric_name2": (lower_bound_factor, upper_bound_factor), # Exceeding upper bound is considered invalid,
        ...
    }
    where lower_bound_factor and upper_bound_factor are floats representing factors to multiply with the fp32 golden accuracy to get the lower and upper bounds of the metric.
    """

    dataset: datasets._Dataset

    use_equal_issue_mode: bool = True
    """Whether to use equal issue mode for benchmarks with non-uniform inputs/outputs. Taken from 'sample_concatenate_permutation' in mlperf.conf.
    By default, this is set to True, since most benchmarks are LLMs with variable length texts."""

    def __hash__(self) -> int:
        # Benchmarks are used as pseudo-'Enum' values, so we can use id as hash
        return id(self)


# TODO: Double check the precision used for golden accuracy values
DeepSeek_R1 = _Model(
    "deepseek-r1",
    golden_accuracy=("fp32",
                     {"exact_match": 81.3582,
                      "TOKENS_PER_SAMPLE": 3886.2274}),
    accuracy_target_settings=[{"exact_match": (0.99,),
                               "TOKENS_PER_SAMPLE": (0.9, 1.1)}],
    dataset=datasets.MLPerfDeepseekR1)
Llama3_1_8b = _Model(
    "llama3-1-8b",
    golden_accuracy=("fp32", {"ROUGE1": 38.7792,
                              "ROUGE2": 15.9075,
                              "ROUGEL": 24.4957,
                              "ROUGELSUM": 35.793,
                              "GEN_LEN": 8167644}),
    accuracy_target_settings=[{"ROUGE1": (0.99,),
                                "ROUGE2": (0.99,),
                                "ROUGEL": (0.99,),
                                "ROUGELSUM": (0.99,),
                                "GEN_LEN": (0.9, 1.1)}],
    dataset=datasets.CNNDailyMail)
Llama2_70b = _Model(
    "llama2-70b",
    golden_accuracy=("fp32", {"ROUGE1": 44.4312,
                              "ROUGE2": 22.0352,
                              "ROUGEL": 28.6162,
                              "TOKENS_PER_SAMPLE": 294.45}),
    accuracy_target_settings=[{"ROUGE1": (0.99,),
                                "ROUGE2": (0.99,),
                                "ROUGEL": (0.99,),
                                "TOKENS_PER_SAMPLE": (0.9, 1.1)},
                                {"ROUGE1": (0.999,),
                                "ROUGE2": (0.999,),
                                "ROUGEL": (0.999,),
                                "TOKENS_PER_SAMPLE": (0.9, 1.1)}],
    dataset=datasets.OpenOrca)
Llama3_1_405b = _Model(
    "llama3.1-405b",
    golden_accuracy=("fp32", {"ROUGEL": 21.6666,
                              "exact_match": 90.1335,
                              "TOKENS_PER_SAMPLE": 684.68}),
    accuracy_target_settings=[{"ROUGEL": (0.99,),
                               "exact_match": (0.99,),
                               "TOKENS_PER_SAMPLE": (0.9, 1.1)}],
    dataset=datasets.TextGenLongSeqLen)
Mixtral8x7B = _Model(
    "mixtral-8x7b",
    golden_accuracy=("fp32", {"ROUGE1": 45.5989,
                              "ROUGE2": 23.3526,
                              "ROUGEL": 30.4608,
                              "TOKENS_PER_SAMPLE": 144.84,
                              "gsm8k_accuracy": 73.66,
                              "mbxp_accuracy": 60.16}),
    accuracy_target_settings=[{"ROUGE1": (0.99,),
                               "ROUGE2": (0.99,),
                               "ROUGEL": (0.99,),
                               "TOKENS_PER_SAMPLE": (0.9, 1.1),
                               "gsm8k_accuracy": (0.99,),
                               "mbxp_accuracy": (0.99,)}],
    dataset=datasets.TextGenComplex)


# Note this isn't completely robust, but will prevent simple cases of defining new instances
def _disallow_instantiation(cls, *args, **kwargs):
    raise TypeError("Cannot instantiate _Model directly. Use a pre-defined model from this module.")
_Model.__new__ = _disallow_instantiation
