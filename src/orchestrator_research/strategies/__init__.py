"""Стратегии по имени: ``get_strategy(name)(env)`` → ``Strategy``."""

from __future__ import annotations

from ..domain import Strategy
from .base import EpisodeEnvironment, OrchestratorLoop, StrategyFactory


def build_swarm(env: EpisodeEnvironment) -> Strategy:
    """Оркестратор может нанимать субагентов и делегировать им задачи."""
    return OrchestratorLoop(env, name="swarm", subagents=True)


def build_single(env: EpisodeEnvironment) -> Strategy:
    """Оркестратор ищет сам, без субагентов."""
    return OrchestratorLoop(env, name="single", subagents=False)


STRATEGIES: dict[str, StrategyFactory] = {"swarm": build_swarm, "single": build_single}


def get_strategy(name: str) -> StrategyFactory:
    try:
        return STRATEGIES[name]
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; available: {sorted(STRATEGIES)}") from None


__all__ = ["STRATEGIES", "EpisodeEnvironment", "StrategyFactory", "get_strategy"]
