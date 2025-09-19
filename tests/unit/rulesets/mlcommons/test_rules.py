import pytest

from inference_endpoint import metrics
from inference_endpoint.rulesets.mlcommons.rules import CURRENT, OptimizationPriority
from inference_endpoint.rulesets.mlcommons import models
from inference_endpoint.config.user_config import UserConfig


def test_apply_user_config():
    user_config = UserConfig(1234.5,
                             max_duration_ms=42 * 60 * 1000)
    rt_settings = CURRENT.apply_user_config(model=models.Llama3_1_8b,
                                            user_config=user_config,
                                            opt_prio=OptimizationPriority.LOW_LATENCY_INTERACTIVE)

    assert rt_settings.model is models.Llama3_1_8b
    assert rt_settings.optimization_priority is OptimizationPriority.LOW_LATENCY_INTERACTIVE

    assert isinstance(rt_settings.metric_target, metrics.Throughput)
    assert rt_settings.metric_target.target == 1234.5

    assert len(rt_settings.reported_metrics) == 3
    assert isinstance(rt_settings.reported_metrics[0], metrics.Throughput)
    assert rt_settings.reported_metrics[0].target == 1234.5
    assert isinstance(rt_settings.reported_metrics[1], metrics.TTFT)
    assert rt_settings.reported_metrics[1].target == 500
    assert isinstance(rt_settings.reported_metrics[2], metrics.TPOT)
    assert rt_settings.reported_metrics[2].target == 30

    assert rt_settings.min_duration_ms == 10 * 60 * 1000
    assert rt_settings.max_duration_ms == 42 * 60 * 1000
    assert rt_settings.n_samples_from_dataset == 13368
    assert rt_settings.n_samples_to_issue is None
    assert rt_settings.rules is CURRENT.benchmark_rulesets[models.Llama3_1_8b][OptimizationPriority.LOW_LATENCY_INTERACTIVE]
