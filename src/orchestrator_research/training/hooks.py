"""Хук Agent Lightning: проставляет ``train_step`` каждому роллауту при постановке в очередь.

agl грузит его по ПУТИ к файлу (``agentlightning.hooks``) и требует ровно один подкласс ``RolloutHooks``.
``on_enqueue`` выполняется в процессе тренера, поэтому ``ORCH_RESEARCH_TRAIN_ROLLOUTS_PER_STEP`` должна быть
в его окружении — её ставит ``train.py`` перед ``run_ppo``.

Корректно только в синхронном режиме без динамического сэмплирования (ADR 0005). Продолжение прогона
выключено: счётчик начнётся с 0, и затухание бонусов награды начнётся заново.
"""

from __future__ import annotations

from agentlightning.hooks import RolloutHooks
from agentlightning.schemas import RolloutCreate

from orchestrator_research.training.overrides import StepCounter


class StepCounterHooks(RolloutHooks):
    def __init__(self) -> None:
        self._counter = StepCounter.from_env()

    def on_enqueue(self, request: RolloutCreate) -> RolloutCreate:
        if not isinstance(request.input, dict):
            raise TypeError(f"rollout input must be a dict; got {type(request.input).__name__}")
        step = self._counter.step_for(request.is_train)
        return request.model_copy(update={"input": {**request.input, "train_step": step}})
