import math

import numpy as np

from methylation_predictor.benchmark.table5.trainer import (
    CartesianSourceSchedule,
    interleave_cartesian_schedules,
    pair_weight_scale,
    resolve_final_epochs,
)


def test_cartesian_schedule_visits_every_pair_slot_once():
    schedule = CartesianSourceSchedule(
        n_rows=7,
        n_cpgs=11,
        row_block_size=3,
        cpg_block_size=4,
        epoch=1,
        seed=17,
    )
    seen = np.zeros((7, 11), dtype=np.int8)
    for step in range(len(schedule)):
        rows, cpgs = schedule[step]
        seen[np.ix_(rows, cpgs)] += 1
    assert np.all(seen == 1)
    assert len(schedule) == math.ceil(7 / 3) * math.ceil(11 / 4)
    assert schedule.coverage_report()["pair_slots"] == 77


def test_interleave_consumes_every_source_step_once():
    schedules = [
        CartesianSourceSchedule(7, 11, 3, 4, 1, 1),
        CartesianSourceSchedule(5, 13, 2, 5, 1, 2),
        CartesianSourceSchedule(2, 17, 2, 4, 1, 3),
    ]
    plan = interleave_cartesian_schedules(schedules, seed=17, epoch=1)
    assert len(plan) == sum(map(len, schedules))
    assert sorted(plan) == sorted(
        (source, step)
        for source, schedule in enumerate(schedules)
        for step in range(len(schedule))
    )


def test_pair_weight_scale_has_unit_average_when_all_expected_pairs_seen():
    observed = [100, 200, 50, 150]
    total = sum(observed)
    scales = [pair_weight_scale(x, total, len(observed)) for x in observed]
    assert np.isclose(np.mean(scales), 1.0)
    # A batch with twice as many finite training pairs has twice the epoch weight.
    assert np.isclose(scales[1] / scales[0], 2.0)


def test_final_epoch_resolution_preserves_update_budget():
    epochs = resolve_final_epochs(20, 117, 1480)
    assert epochs == 2
    assert abs(epochs * 1480 - 20 * 117) <= 1480 / 2
