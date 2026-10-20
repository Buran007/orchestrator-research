"""Цикл оркестратора на одном вопросе: ходы модели, вызовы инструментов, финальный ответ.

Вход — ``OrchestratorLoop.run``; ходы и причины остановки — ``_loop``; исполнение инструментов — ``_run_tools``.
``messages`` только дописываются: Agent Lightning склеивает ходы эпизода в одну траекторию по общему префиксу.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from ..domain import (
    ChatModel,
    EpisodeResult,
    EpisodeStatus,
    EventSource,
    ModelTurn,
    Observation,
    Question,
    Retriever,
    Stage,
    Strategy,
    SubagentRun,
    ToolCall,
    TraceEvent,
)
from ..limits import TaskBudget, TaskLimits
from ..subagents import CONTEXT_STOP, ContextMeter, SubagentPool, _chars_tokens, is_context_length_error
from ..tools import (
    ASSIGN_TASK_TOOL,
    CREATE_SUBAGENT_TOOL,
    FINALIZER_SYSTEM_PROMPT,
    SEARCH_TOOL_NAME,
    assistant_message,
    orchestrator_system_prompt,
    orchestrator_tools,
    run_search,
)


class EpisodeStop(StrEnum):
    """Причина остановки цикла, кроме бюджетных (те — ``limits.StopReason``)."""

    ANSWERED = "answered"
    NO_ANSWER = "no_answer"
    CONTEXT = CONTEXT_STOP  # следующий ход не влезет в окно


@dataclass
class EpisodeEnvironment:
    """Зависимости одного вопроса; на каждый вопрос — новый экземпляр (бюджет считается на вопрос).

    ``budget`` — тот же объект, с которым создан ``pool``: субагенты тратят общий бюджет.
    ``context_limit`` — окно оркестратора в токенах; None — не следим, ограничивает сервер модели.
    """

    orchestrator: ChatModel
    pool: SubagentPool | None
    retriever: Retriever
    budget: TaskBudget
    limits: TaskLimits
    top_k: int
    context_limit: int | None = None


StrategyFactory = Callable[[EpisodeEnvironment], Strategy]


class _ModelFailure(Exception):
    """Запрос к оркестратору упал; эпизод получает статус MODEL_ERROR."""


class _ContextOverflow(Exception):
    """Сервер модели отклонил промпт как слишком длинный; причина остановки CONTEXT."""


class OrchestratorLoop:
    """Оркестратор с бюджетом. ``subagents=False`` — только поиск (стратегия single)."""

    def __init__(self, env: EpisodeEnvironment, *, name: str, subagents: bool) -> None:
        if subagents and env.pool is None:
            raise ValueError(f"strategy {name!r} needs a SubagentPool")
        self._env = env
        self._name = name
        self._pool = env.pool if subagents else None
        roles = env.pool.role_names if env.pool is not None else ()
        aliases = env.pool.model_aliases if env.pool is not None else ()
        self._tools = orchestrator_tools(subagents=subagents, role_names=roles, model_aliases=aliases)
        self._system_prompt = orchestrator_system_prompt(subagents=subagents, role_names=roles, model_aliases=aliases)

    async def run(self, question: Question) -> EpisodeResult:
        """Прогнать эпизод: цикл ходов, при необходимости финальный ход, сборка ``EpisodeResult``.

        Правильность ответа здесь не проверяется — это делает судья.
        """
        env = self._env
        budget = env.budget
        started = time.perf_counter()
        events: list[TraceEvent] = []
        stages: list[Stage] = []  # субагенты по ходам — для r_parallel в награде
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": f"QUESTION:\n{question.text}"},
        ]
        final_answer: str | None = None
        stop_reason: str | None = None
        error: str | None = None
        meter = ContextMeter()
        try:
            final_answer, stop_reason = await self._loop(messages, events, stages, meter)
            if final_answer is None:
                final_answer = await self._finalize(messages, events, meter)
        except _ModelFailure as exc:
            error = str(exc)
        except _ContextOverflow:
            final_answer = None
            stop_reason = EpisodeStop.CONTEXT.value

        if error is not None:
            status = EpisodeStatus.MODEL_ERROR
        elif final_answer is not None:
            status = EpisodeStatus.COMPLETED
        else:
            status = EpisodeStatus.NO_FINAL_ANSWER
        observed = [event for event in events if event.source is EventSource.OBSERVATION]
        return EpisodeResult(
            question_id=question.question_id,
            final_answer=final_answer,
            status=status,
            stop_reason=stop_reason,
            error=error,
            strategy=self._name,
            elapsed_seconds=max(0.0, time.perf_counter() - started),
            events=tuple(events),
            stages=tuple(stages),
            budget=budget.snapshot(),
            tool_errors=sum(event.is_error for event in observed),
            infra_errors=sum(_is_infra(event) for event in observed),
        )

    async def _loop(
        self, messages: list[dict[str, Any]], events: list[TraceEvent], stages: list[Stage], meter: ContextMeter
    ) -> tuple[str | None, str]:
        """Ходы до ответа, исчерпания бюджета или окна.

        Возвращает ``(answer, stop)``; ``answer=None`` — ответа нет, ``run`` вызовет ``_finalize``.
        """
        env = self._env
        budget = env.budget
        limits = env.limits
        while True:
            stop = budget.orchestrator_stop_reason()
            if stop is not None:
                return None, stop.value
            # После хода должно остаться место на финальный ответ.
            if env.context_limit is not None and (
                meter.estimate(messages) + limits.orchestrator_tokens_per_turn + limits.final_answer_reserve
                > env.context_limit
            ):
                return None, EpisodeStop.CONTEXT.value
            turn = await self._call(messages, budget.reserve_orchestrator_turn(), self._tools)
            events.append(_orchestrator_event(turn))
            messages.append(assistant_message(turn.text, turn.tool_calls))
            meter.record(turn, len(messages))
            if not turn.tool_calls:
                answer = (turn.final_answer or "").strip()
                # Обрезанный по max_tokens ход — не ответ: текст может оборваться внутри <think>.
                if answer and turn.finish_reason != "length":
                    return answer, EpisodeStop.ANSWERED.value
                return None, EpisodeStop.NO_ANSWER.value
            observations, runs = await self._run_tools(turn.tool_calls)
            stages.append(Stage(turn_index=budget.orchestrator_turns - 1, subagent_runs=tuple(runs)))
            room = self._observation_room(messages, meter)
            for call, observation in zip(turn.tool_calls, observations, strict=True):
                content, room = fit_observation(observation.content, room)
                observation = replace(observation, content=content)
                messages.append(
                    {"role": "tool", "tool_call_id": call.call_id, "name": call.name, "content": observation.content}
                )
                events.append(_observation_event(call, observation))

    def _observation_room(self, messages: list[dict[str, Any]], meter: ContextMeter) -> int | None:
        """Сколько символов вывода инструментов влезет, не отняв место у финального хода; None — без лимита.

        ``room = 3 * (context_limit - estimate - final_answer_reserve - finalizer_prompt)``, 3 символа на токен,
        как в ``ContextMeter``. Без обрезки несколько поисков за ход заполняли окно целиком.
        """
        limit = self._env.context_limit
        if limit is None:
            return None
        finalizer = _chars_tokens([{"content": FINALIZER_SYSTEM_PROMPT}])
        tokens = limit - meter.estimate(messages) - self._env.limits.final_answer_reserve - finalizer
        return max(0, 3 * tokens)

    async def _finalize(
        self, messages: list[dict[str, Any]], events: list[TraceEvent], meter: ContextMeter
    ) -> str | None:
        """Финальный ход без инструментов в пределах ``final_answer_reserve``; None — ответа нет."""
        budget = self._env.budget
        reserved = budget.reserve_final_answer()
        if reserved == 0:
            return None
        # Новым сообщением в конец, чтобы не ломать общий префикс.
        prompt = [*messages, {"role": "user", "content": FINALIZER_SYSTEM_PROMPT}]
        max_tokens = reserved
        if self._env.context_limit is not None:
            max_tokens = min(reserved, self._env.context_limit - meter.estimate(prompt))
            if max_tokens <= 0:
                budget.commit_orchestrator(reserved, 0)  # запрос не отправлен, бронь снимаем
                return None
        # Без thinking: иначе весь резерв уходит в <think> и ответа нет.
        turn = await self._call(prompt, reserved, None, max_tokens=max_tokens, thinking=False)
        events.append(_orchestrator_event(turn))
        # Без thinking обрезанный ход — длинный, но ответ; не берём, только если модель всё же начала <think>.
        if turn.finish_reason == "length" and "<think>" in (turn.text or ""):
            return None
        return (turn.final_answer or "").strip() or None

    async def _call(
        self,
        messages: Sequence[Mapping[str, Any]],
        reserved: int,
        tools: Sequence[Mapping[str, Any]] | None,
        *,
        max_tokens: int | None = None,
        thinking: bool | None = None,
    ) -> ModelTurn:
        """Запрос к оркестратору. Бронь ``reserved`` закрывается фактическим расходом всегда, даже при ошибке.

        ``max_tokens`` — потолок генерации (по умолчанию ``reserved``); ``thinking`` — переопределение на запрос.
        """
        actual = 0
        try:
            extra = {} if thinking is None else {"thinking": thinking}
            turn = await self._env.orchestrator.complete(
                messages, tools=tools, max_tokens=reserved if max_tokens is None else max_tokens, **extra
            )
            actual = turn.output_tokens
        except Exception as exc:
            if is_context_length_error(exc):
                raise _ContextOverflow(str(exc)) from exc
            raise _ModelFailure(f"{type(exc).__name__}: {exc}") from exc
        finally:
            self._env.budget.commit_orchestrator(reserved, actual)
        return turn

    async def _run_tools(self, calls: tuple[ToolCall, ...]) -> tuple[list[Observation], list[SubagentRun]]:
        """Исполнить вызовы одного хода: ровно один ``Observation`` на вызов, в порядке вызовов.

        ``create_subagent`` — сразу; поиски и ``assign_task`` — параллельно.
        """
        limit = self._env.limits.max_tool_calls_per_turn
        pool = self._pool
        results: dict[int, Observation] = {}
        searches: list[int] = []
        assigns: list[tuple[int, str, str]] = []  # (index, agent, prompt)
        for index, call in enumerate(calls):
            if index >= limit:
                results[index] = _error(call, f"too many tool calls in one turn; limit is {limit}, not run")
            elif call.name == SEARCH_TOOL_NAME:
                searches.append(index)
            elif call.name == CREATE_SUBAGENT_TOOL and pool is not None:
                results[index] = self._create(pool, call)
            elif call.name == ASSIGN_TASK_TOOL and pool is not None:
                parsed = _assign_arguments(call)
                if parsed is None:
                    results[index] = _error(call, "assign_task requires exactly agent and prompt")
                else:
                    assigns.append((index, *parsed))
            else:
                results[index] = _error(call, f"unknown or unavailable tool: {call.name}")

        # create уже выполнены выше, так что assign в этом же ходе видит новых субагентов.
        search_batch = asyncio.gather(
            *(run_search(self._env.retriever, calls[i].arguments, self._env.top_k) for i in searches)
        )
        # Пустой gather() сразу возвращает [].
        assign_batch = (
            pool.assign_many([(calls[i].call_id, agent, prompt) for i, agent, prompt in assigns])
            if pool is not None
            else asyncio.gather()
        )
        search_out, assign_out = await asyncio.gather(search_batch, assign_batch)

        for index, (content, is_error, infra_error) in zip(searches, search_out, strict=True):
            results[index] = Observation(
                calls[index].call_id, SEARCH_TOOL_NAME, content, is_error=is_error, infra_error=infra_error
            )
        runs: list[SubagentRun] = []
        for (index, _, _), (observation, run) in zip(assigns, assign_out, strict=True):
            results[index] = observation
            if run is not None:
                runs.append(run)
        return [results[index] for index in range(len(calls))], runs

    @staticmethod
    def _create(pool: SubagentPool, call: ToolCall) -> Observation:
        args = call.arguments
        if set(args) != {"name", "role", "model"}:
            return _error(call, "create_subagent requires exactly name, role and model")
        return pool.create(call.call_id, args["name"], args["role"], args["model"])


def _assign_arguments(call: ToolCall) -> tuple[str, str] | None:
    args = call.arguments
    if set(args) != {"agent", "prompt"}:
        return None
    return args["agent"], args["prompt"]


def _error(call: ToolCall, message: str) -> Observation:
    return Observation(call.call_id, call.name, message, is_error=True)


def _is_infra(event: TraceEvent) -> bool:
    return bool(event.metadata.get("infra_error", False))


def _orchestrator_event(turn: ModelTurn) -> TraceEvent:
    """Ход модели; единственный вид событий с ``trainable=True``."""
    return TraceEvent(
        source=EventSource.ORCHESTRATOR,
        content=turn.text,
        token_count=turn.output_tokens,
        trainable=True,
        tool_calls=turn.tool_calls,
        metadata={"finish_reason": turn.finish_reason, "elapsed_seconds": turn.elapsed_seconds},
    )


def _observation_event(call: ToolCall, observation: Observation) -> TraceEvent:
    """Вывод инструмента: контекст для следующих ходов, в лосс не идёт."""
    return TraceEvent(
        source=EventSource.OBSERVATION,
        content=observation.content,
        token_count=0,
        trainable=False,
        tool_call_id=call.call_id,
        tool_name=call.name,
        is_error=observation.is_error,
        metadata={**observation.metadata, "infra_error": observation.infra_error},
    )


CONTEXT_CUT_NOTE = "\n[cut: context limit]"


def fit_observation(content: str, room: int | None) -> tuple[str, int | None]:
    """Обрезать ``content`` до ``room`` символов с пометкой; вернуть текст и остаток места (None — без лимита)."""
    if room is None:
        return content, None
    if len(content) <= room:
        return content, room - len(content)
    return content[: max(0, room - len(CONTEXT_CUT_NOTE))] + CONTEXT_CUT_NOTE, 0
