import numpy as np

from methylation_predictor.full_coverage_sampler import build_epoch_schedule
from methylation_predictor.full_suite.ablation_runner import (
    checkpoint_metric_value,
    interleave_full_coverage_schedules,
)


def test_checkpoint_metric_direction_supports_mas_pcc():
    value, higher = checkpoint_metric_value({"mas_pcc": 0.42}, "mas_pcc")
    assert value == 0.42
    assert higher is True


def test_checkpoint_metric_direction_supports_mse():
    value, higher = checkpoint_metric_value({"mse": 0.02}, "mse")
    assert value == 0.02
    assert higher is False


def test_interleaved_plan_consumes_every_source_schedule_once():
    schedules = [
        build_epoch_schedule(np.arange(11), np.arange(5), 4, 3, epoch=1, seed=17),
        build_epoch_schedule(np.arange(23), np.arange(4), 6, 4, epoch=1, seed=1026),
        build_epoch_schedule(np.arange(41), np.arange(2), 8, 2, epoch=1, seed=2035),
    ]
    plan = interleave_full_coverage_schedules(schedules)
    assert len(plan) == sum(len(s) for s in schedules)
    for source_index, schedule in enumerate(schedules):
        local_steps = sorted(step for source, step in plan if source == source_index)
        assert local_steps == list(range(len(schedule)))
