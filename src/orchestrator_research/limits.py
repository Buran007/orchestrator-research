"""Лимиты токенов и шагов на один вопрос; учёт по схеме «бронь → фактический расход».

Обозначения (считаются только сгенерированные токены; лимит ``None`` — без ограничения)::

    R         = final_answer_reserve
    room_o    = orchestrator_tokens_per_task - (orch_used + orch_reserved)
    room_s_i  = subagent_tokens_per_task - (used_i + reserved_i)
    room_t    = total_tokens_per_task - (total_used + total_reserved)

    orchestrator turn: stop iff turns >= max_turns or room_o - R <= 0 or room_t - R <= 0
                       otherwise reserve = min(orchestrator_tokens_per_turn, room_o - R, room_t - R)
    final answer:      reserve = max(0, min(base, room_o, room_t)),  base = R if R > 0 else per_turn
    subagent step:     stop iff steps_i >= subagent_max_steps or room_s_i <= 0 or room_t - R <= 0
                       otherwise reserve = min(subagent_tokens_per_step, room_s_i, room_t - R)

``reserve_*`` возвращает 0 ровно тогда, когда соответствующий ``*_stop_reason()`` не None, иначе бронирует
по формуле. Ход оркестратора засчитывается в момент брони; финальный ответ ходом не считается.

Источник: собственная политика бюджета проекта, внешней ссылки нет. Инварианты — tests/test_limits.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskLimits:
    """Лимиты на ОДИН вопрос. None — без ограничения; токены — сгенерированные (completion)."""

    orchestrator_tokens_per_task: int | None = None
    orchestrator_tokens_per_turn: int = 2048
    subagent_tokens_per_task: int | None = None  # на КАЖДЫЙ экземпляр субагента, сумма по всем его шагам
    subagent_tokens_per_step: int = 1024
    total_tokens_per_task: int | None = None  # оркестратор + все субагенты
    final_answer_reserve: int = 512
    max_turns: int = 15
    max_tool_calls_per_turn: int = 8
    subagent_max_steps: int = 10
    max_subagents_per_task: int = 8
    max_parallel: int = 8

    def __post_init__(self) -> None:
        for name in (
            "orchestrator_tokens_per_turn",
            "subagent_tokens_per_step",
            "max_turns",
            "max_tool_calls_per_turn",
            "subagent_max_steps",
            "max_subagents_per_task",
            "max_parallel",
            "orchestrator_tokens_per_task",
            "subagent_tokens_per_task",
            "total_tokens_per_task",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if self.final_answer_reserve < 0:
            raise ValueError(f"final_answer_reserve must be >= 0, got {self.final_answer_reserve}")
        for name in ("orchestrator_tokens_per_task", "total_tokens_per_task"):
            value = getattr(self, name)
            if value is not None and value <= self.final_answer_reserve:
                raise ValueError(f"{name} must be greater than final_answer_reserve ({self.final_answer_reserve})")


class StopReason(StrEnum):
    ORCHESTRATOR_BUDGET = "orchestrator_budget"
    TOTAL_BUDGET = "total_budget"
    SUBAGENT_BUDGET = "subagent_budget"
    TURNS = "turns"
    SUBAGENT_STEPS = "subagent_steps"


@dataclass(slots=True)
class _SubagentAccount:
    used: int = 0
    reserved: int = 0
    steps: int = 0


def _room(limit: int | None, spent: int) -> int | None:
    """Сколько осталось до ``limit``; None — без лимита."""
    return None if limit is None else limit - spent


def _exhausted(room: int | None) -> bool:
    return room is not None and room <= 0


def _cap(cap: int, *rooms: int | None) -> int:
    """Минимум из ``cap`` и всех ограниченных остатков."""
    return min([cap, *(room for room in rooms if room is not None)])


class TaskBudget:
    """Счёт расхода на один вопрос.

    Каждая генерация сначала бронируется (``reserve_*``), и бронь сразу вычитается из остатка — поэтому
    параллельные субагенты не могут вместе превысить лимит. ``commit_*`` снимает бронь и записывает факт.

    Инварианты:
      * ``total_tokens == orchestrator_tokens + сумма токенов субагентов``;
      * пока провайдер соблюдает ``max_tokens``, ``actual <= reserved`` и лимиты не превышаются
        (``snapshot()["overshoot_tokens"] == 0``);
      * ни ход, ни шаг субагента не занимают ``final_answer_reserve`` из общего бюджета — на финальный
        ответ место есть всегда;
      * у одного субагента не должно быть двух шагов одновременно: лимит шагов считает только закрытые.
    """

    def __init__(self, limits: TaskLimits) -> None:
        self._limits = limits
        self._orch_used = 0
        self._orch_reserved = 0
        self._orch_turns = 0
        self._overshoot = 0
        self._subagents: dict[str, _SubagentAccount] = {}

    # ----- чтение -------------------------------------------------------------

    @property
    def orchestrator_tokens(self) -> int:
        return self._orch_used

    @property
    def total_tokens(self) -> int:
        return self._orch_used + sum(acc.used for acc in self._subagents.values())

    @property
    def orchestrator_turns(self) -> int:
        return self._orch_turns

    def snapshot(self) -> dict[str, Any]:
        return {
            "orchestrator_tokens": self._orch_used,
            "orchestrator_turns": self._orch_turns,
            "subagent_tokens": {name: acc.used for name, acc in self._subagents.items()},
            "subagent_steps": {name: acc.steps for name, acc in self._subagents.items()},
            "subagents_created": len(self._subagents),
            "total_tokens": self.total_tokens,
            "overshoot_tokens": self._overshoot,
        }

    # ----- оркестратор --------------------------------------------------------

    def reserve_orchestrator_turn(self) -> int:
        """Забронировать ``max_tokens`` на ход и засчитать ход; 0 — ход делать нельзя."""
        if self.orchestrator_stop_reason() is not None:
            return 0
        reserve = self._limits.final_answer_reserve
        amount = _cap(self._limits.orchestrator_tokens_per_turn, self._orch_room(reserve), self._total_room(reserve))
        self._orch_reserved += amount
        self._orch_turns += 1
        return amount

    def reserve_final_answer(self) -> int:
        """Забронировать ``max_tokens`` на финальный ответ; может брать отложенный резерв. Ходом не считается."""
        limits = self._limits
        base = limits.final_answer_reserve if limits.final_answer_reserve > 0 else limits.orchestrator_tokens_per_turn
        amount = max(0, _cap(base, self._orch_room(), self._total_room()))
        self._orch_reserved += amount
        return amount

    def commit_orchestrator(self, reserved: int, actual: int) -> None:
        _check_amounts(reserved, actual)
        if reserved > self._orch_reserved:
            raise ValueError(f"reserved {reserved} exceeds outstanding orchestrator reservation {self._orch_reserved}")
        self._orch_reserved -= reserved
        self._orch_used += actual
        self._overshoot += max(0, actual - reserved)

    def orchestrator_stop_reason(self) -> StopReason | None:
        """Почему ``reserve_orchestrator_turn`` вернёт 0; None — ход можно делать."""
        if self._orch_turns >= self._limits.max_turns:
            return StopReason.TURNS
        reserve = self._limits.final_answer_reserve
        if _exhausted(self._orch_room(reserve)):
            return StopReason.ORCHESTRATOR_BUDGET
        if _exhausted(self._total_room(reserve)):
            return StopReason.TOTAL_BUDGET
        return None

    # ----- субагенты ----------------------------------------------------------

    def can_create_subagent(self) -> bool:
        return len(self._subagents) < self._limits.max_subagents_per_task

    def register_subagent(self, name: str) -> None:
        if name in self._subagents:
            raise ValueError(f"duplicate subagent name: {name!r}")
        if not self.can_create_subagent():
            raise ValueError(f"max_subagents_per_task ({self._limits.max_subagents_per_task}) reached")
        self._subagents[name] = _SubagentAccount()

    def reserve_subagent_step(self, name: str) -> int:
        """Забронировать ``max_tokens`` на шаг ``name``; 0 — шаг делать нельзя."""
        if self.subagent_stop_reason(name) is not None:
            return 0
        acc = self._account(name)
        amount = _cap(
            self._limits.subagent_tokens_per_step,
            self._subagent_room(acc),
            self._total_room(self._limits.final_answer_reserve),
        )
        acc.reserved += amount
        return amount

    def commit_subagent(self, name: str, reserved: int, actual: int) -> None:
        _check_amounts(reserved, actual)
        acc = self._account(name)
        if reserved > acc.reserved:
            raise ValueError(f"reserved {reserved} exceeds outstanding reservation {acc.reserved} of {name!r}")
        acc.reserved -= reserved
        acc.used += actual
        acc.steps += 1
        self._overshoot += max(0, actual - reserved)

    def subagent_stop_reason(self, name: str) -> StopReason | None:
        """Почему ``reserve_subagent_step(name)`` вернёт 0; None — шаг можно делать."""
        acc = self._account(name)
        if acc.steps >= self._limits.subagent_max_steps:
            return StopReason.SUBAGENT_STEPS
        if _exhausted(self._subagent_room(acc)):
            return StopReason.SUBAGENT_BUDGET
        if _exhausted(self._total_room(self._limits.final_answer_reserve)):
            return StopReason.TOTAL_BUDGET
        return None

    # ----- внутреннее ---------------------------------------------------------

    def _account(self, name: str) -> _SubagentAccount:
        try:
            return self._subagents[name]
        except KeyError:
            raise KeyError(f"unknown subagent: {name!r}") from None

    def _orch_room(self, reserve: int = 0) -> int | None:
        return _room(self._limits.orchestrator_tokens_per_task, self._orch_used + self._orch_reserved + reserve)

    def _total_room(self, reserve: int = 0) -> int | None:
        total_reserved = self._orch_reserved + sum(acc.reserved for acc in self._subagents.values())
        return _room(self._limits.total_tokens_per_task, self.total_tokens + total_reserved + reserve)

    def _subagent_room(self, acc: _SubagentAccount) -> int | None:
        return _room(self._limits.subagent_tokens_per_task, acc.used + acc.reserved)


def _check_amounts(reserved: int, actual: int) -> None:
    if reserved < 0 or actual < 0:
        raise ValueError(f"reserved and actual must be >= 0, got reserved={reserved}, actual={actual}")
