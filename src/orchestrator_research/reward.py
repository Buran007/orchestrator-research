"""Награда оркестратора в духе PARL: результат задачи плюс затухающие бонусы за делегирование.

Формулы (``t`` — глобальный шаг обучения, ``T`` — ``anneal_steps``)::

    r_PARL(t)      = r_perf + lambda1(t) * r_parallel + lambda2(t) * r_finish     (training)
    r_PARL(t)      = r_perf                                                     (validation)

    linear anneal  lambda(t) = init * (1 - t / T)            for t < T, else 0
    cosine anneal  lambda(t) = init * 0.5 * (1 + cos(pi t / T))  for t < T, else 0

    режимы r_parallel (все в [0, 1]); stage — ход с вызовами инструментов:
        fraction_of_stages   доля ходов, где запущен хотя бы один субагент
        count_normalised     min(1, всего запущено / max_subagents_init)
        cohort_size_avg      min(1, среднее число запущенных за ход / max_subagents_init)

    r_finish = finished_ok / observable,  observable = запущено − упавших по инфраструктуре;
               ничего не запущено → 0; все упали по инфраструктуре → 1 (поведение не наблюдалось).

Режимы и имена расписаний проверяет ``RewardConfig``; здесь входы не перепроверяются.

Источник: Kimi K2.5 PARL §3 (разложение награды и затухание); перенесено из форка verl в MultiAgents-main —
``verl/workers/reward_manager/parl.py`` (``_r_parallel``, ``_r_finish``) и ``parl_lambda.py`` (``LambdaScheduler``).
Отличие от форка: счётчики берутся из ``Stage.subagent_runs``, а сбои инфраструктуры исключены из r_finish.
Инварианты — ``tests/test_reward.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import RewardConfig
    from .domain import Stage, SubagentRun


def anneal(init: float, step: int, anneal_steps: int, mode: str) -> float:
    """Вес, затухающий от ``init`` на шаге 0 до ровно 0 на шаге ``anneal_steps`` (линейно или по косинусу)."""
    step = max(0, step)
    if step >= anneal_steps:
        return 0.0
    progress = step / anneal_steps
    if mode == "cosine":
        return init * 0.5 * (1.0 + math.cos(math.pi * progress))
    return init * (1.0 - progress)


def _spawned_runs(stages: Sequence[Stage]) -> list[SubagentRun]:
    return [run for stage in stages for run in stage.subagent_runs]


def r_parallel(stages: Sequence[Stage], mode: str, max_subagents_init: int) -> float:
    """Бонус за делегирование в ``[0, 1]``: за само решение запустить субагентов, а не за их результат."""
    if not stages:
        return 0.0
    if mode == "fraction_of_stages":
        return sum(1 for stage in stages if stage.subagent_runs) / len(stages)
    if mode == "count_normalised":
        return min(1.0, len(_spawned_runs(stages)) / max_subagents_init)
    average = sum(len(stage.subagent_runs) for stage in stages) / len(stages)
    return min(1.0, average / max_subagents_init)


def r_finish(stages: Sequence[Stage]) -> float:
    """Доля субагентов, доведших задание до ответа, в ``[0, 1]``; сбои инфраструктуры не считаются."""
    runs = _spawned_runs(stages)
    if not runs:
        return 0.0
    observable = [run for run in runs if not run.infra_failure]
    if not observable:
        # Все упали по вине окружения: поведение модели не наблюдалось, не штрафуем.
        return 1.0
    return sum(1 for run in observable if run.finished) / len(observable)


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    total: float
    r_perf: float
    r_parallel: float
    r_finish: float
    lambda_parallel: float
    lambda_finish: float
    step: int
    is_validation: bool


def compose_reward(
    r_perf: float,
    stages: Sequence[Stage],
    *,
    step: int,
    is_validation: bool,
    cfg: RewardConfig,
) -> RewardBreakdown:
    """Итоговая награда: ``r_perf`` плюс затухающие бонусы.

    На валидации ``total == r_perf``; бонусы всё равно считаются, чтобы в логах было видно их значения.
    """
    if not math.isfinite(r_perf):
        raise ValueError(f"r_perf must be finite; got {r_perf}")
    step = max(0, step)
    lambda_parallel = anneal(cfg.lambda_parallel, step, cfg.anneal_steps, cfg.anneal_mode)
    lambda_finish = anneal(cfg.lambda_finish, step, cfg.anneal_steps, cfg.anneal_mode)
    par = r_parallel(stages, cfg.r_parallel_mode, cfg.max_subagents_init)
    fin = r_finish(stages)
    total = r_perf if is_validation else r_perf + lambda_parallel * par + lambda_finish * fin
    return RewardBreakdown(
        total=float(total),
        r_perf=float(r_perf),
        r_parallel=par,
        r_finish=fin,
        lambda_parallel=lambda_parallel,
        lambda_finish=lambda_finish,
        step=step,
        is_validation=is_validation,
    )
