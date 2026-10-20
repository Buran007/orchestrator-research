"""Награда: отжиг лямбд, r_parallel и r_finish в [0, 1], сборка total для обучения и валидации."""

import dataclasses
import itertools
import json
import math

import pytest

from orchestrator_research.config import RewardConfig
from orchestrator_research.domain import Stage, SubagentRun
from orchestrator_research.reward import anneal, compose_reward, r_finish, r_parallel


def run(*, finished: bool = True, infra: bool = False) -> SubagentRun:
    return SubagentRun(
        name="a",
        role="retriever",
        model_alias="small",
        steps=1,
        finished=finished,
        stop_reason=None,
        tokens=10,
        infra_failure=infra,
    )


def stage(*runs: SubagentRun, turn: int = 0) -> Stage:
    return Stage(turn_index=turn, subagent_runs=tuple(runs))


STAGE_SETS = (
    (),
    (stage(),),
    (stage(*[run()] * 10),),
    (stage(run()), stage(), stage(run(), run())),
    (stage(*[run()] * 8), stage(run())),
)


# ----- отжиг -----------------------------------------------------------------


@pytest.mark.parametrize("mode", ["linear", "cosine"])
def test_anneal_endpoints_midpoint_and_zero_after(mode):
    assert anneal(0.2, 0, 10, mode) == pytest.approx(0.2)
    assert anneal(0.2, 5, 10, mode) == pytest.approx(0.1)  # оба режима проходят половину при T/2
    assert anneal(0.2, 10, 10, mode) == 0.0
    assert anneal(0.2, 1000, 10, mode) == 0.0


@pytest.mark.parametrize("mode", ["linear", "cosine"])
def test_anneal_is_non_increasing_and_negative_steps_count_as_zero(mode):
    steps = range(-3, 30)
    values = [anneal(0.5, step, 7, mode) for step in steps]
    assert values == sorted(values, reverse=True)
    assert all(anneal(0.5, step, 7, mode) == anneal(0.5, 0, 7, mode) for step in range(-3, 1))
    assert all(anneal(0.5, step, 7, mode) == 0.0 for step in range(7, 30))


# ----- r_parallel / r_finish -------------------------------------------------


@pytest.mark.parametrize(
    ("stages", "mode"),
    list(itertools.product(STAGE_SETS, ["fraction_of_stages", "count_normalised", "cohort_size_avg"])),
)
def test_r_parallel_stays_in_unit_interval(stages, mode):
    assert 0.0 <= r_parallel(stages, mode, max_subagents_init=2) <= 1.0


def test_r_parallel_modes_match_their_formulas():
    stages = (stage(), stage(run()), stage(run(), run()))
    assert r_parallel(stages, "fraction_of_stages", 8) == pytest.approx(2 / 3)
    dense = (stage(run(), run(), run(), run()),)
    assert r_parallel(dense, "count_normalised", 2) == 1.0
    assert r_parallel(dense, "cohort_size_avg", 8) == 0.5


@pytest.mark.parametrize("mode", ["fraction_of_stages", "count_normalised", "cohort_size_avg"])
def test_r_parallel_is_zero_without_spawns(mode):
    assert r_parallel((), mode, 8) == 0.0
    assert r_parallel((stage(), stage()), mode, 8) == 0.0


def test_r_finish_zero_when_nothing_spawned():
    assert r_finish(()) == 0.0
    assert r_finish((stage(), stage())) == 0.0


def test_r_finish_excludes_infra_failures_from_denominator():
    assert r_finish((stage(run(infra=True, finished=False), run(infra=True, finished=False)),)) == 1.0
    assert r_finish((stage(run(finished=True), run(infra=True, finished=False)),)) == 1.0
    assert r_finish(
        (stage(run(finished=True), run(finished=False), run(infra=True, finished=False)),)
    ) == pytest.approx(0.5)


@pytest.mark.parametrize(("finished", "infra"), list(itertools.product((True, False), repeat=2)))
def test_r_finish_within_unit_interval(finished, infra):
    assert 0.0 <= r_finish((stage(run(finished=finished, infra=infra), run(finished=True)),)) <= 1.0


# ----- сборка награды ---------------------------------------------------------


@pytest.fixture
def cfg():
    return RewardConfig()  # умолчания: lambda_parallel 0.1, lambda_finish 0.05, отжиг 7 шагов, linear


@pytest.fixture
def one_run_stages():
    return (stage(run()),)


def test_validation_total_is_task_outcome_only(cfg, one_run_stages):
    breakdown = compose_reward(1.0, one_run_stages, step=0, is_validation=True, cfg=cfg)
    assert breakdown.total == 1.0
    assert breakdown.r_parallel == 1.0  # компоненты всё равно попадают в отчёт


def test_training_total_matches_formula(cfg, one_run_stages):
    breakdown = compose_reward(1.0, one_run_stages, step=0, is_validation=False, cfg=cfg)
    assert breakdown.total == pytest.approx(1.0 + 0.1 * 1.0 + 0.05 * 1.0)
    assert breakdown.lambda_parallel == pytest.approx(0.1)
    assert breakdown.lambda_finish == pytest.approx(0.05)


def test_lambdas_vanish_after_anneal(cfg, one_run_stages):
    breakdown = compose_reward(0.0, one_run_stages, step=7, is_validation=False, cfg=cfg)
    assert (breakdown.lambda_parallel, breakdown.lambda_finish, breakdown.total) == (0.0, 0.0, 0.0)


def test_negative_step_uses_step_zero_lambdas(cfg, one_run_stages):
    at_zero = compose_reward(0.0, one_run_stages, step=0, is_validation=False, cfg=cfg)
    negative = compose_reward(0.0, one_run_stages, step=-1, is_validation=False, cfg=cfg)
    assert negative.total == at_zero.total
    assert negative.step == 0


def test_breakdown_is_json_ready(cfg, one_run_stages):
    payload = dataclasses.asdict(compose_reward(1.0, one_run_stages, step=3, is_validation=False, cfg=cfg))
    json.dumps(payload)


def test_rejects_non_finite_task_reward(cfg, one_run_stages):
    with pytest.raises(ValueError):
        compose_reward(math.nan, one_run_stages, step=0, is_validation=False, cfg=cfg)
