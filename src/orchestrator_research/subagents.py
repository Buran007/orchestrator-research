"""Субагенты (не обучаются): цикл одного задания — ``run_subagent``; пул экземпляров на эпизод — ``SubagentPool``.

Задание — цикл «ход модели → поиск» со своим счётом в ``TaskBudget``. Бронь токенов на шаг закрывается
в ``finally``, поэтому остановленный, упавший или отменённый субагент не оставляет висящих броней.
Пул не бросает исключений в оркестратор: неверный вызов возвращается ошибкой в ``Observation``.
Экземпляр не помнит диалог между заданиями; задания одному экземпляру выполняются по очереди.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .domain import ChatModel, ModelTurn, Observation, Retriever, SubagentRun
from .limits import TaskBudget, TaskLimits
from .tools import (
    ASSIGN_TASK_TOOL,
    CREATE_SUBAGENT_TOOL,
    SEARCH_TOOL_NAME,
    assistant_message,
    executed,
    run_search,
    search_tool_schema,
)

INFRA_FAILURE_MARKER = "[SUBAGENT_INFRA_FAILURE]"
CONTEXT_STOP = "context"
CONTEXT_LENGTH_MARKER = "maximum context length"  # подстрока ответа 400 от vLLM/OpenAI на слишком длинный промпт
_MAX_ERROR_CHARS = 200


def is_context_length_error(exc: BaseException) -> bool:
    """Провайдер отклонил запрос, потому что промпт длиннее окна."""
    return CONTEXT_LENGTH_MARKER in str(exc)


def _chars_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    """Оценка с запасом: токен на три символа текста и JSON вызовов."""
    chars = sum(len(str(m.get("content") or "")) + len(str(m.get("tool_calls") or "")) for m in messages)
    return chars // 3


@dataclass(slots=True)
class ContextMeter:
    """Оценка размера промпта следующего запроса в диалоге.

    ``next = prompt_tokens(last) + output_tokens(last) + chars(сообщения после last) // 3``: точные числа
    провайдера за уже отправленное плюс оценка по символам за новое. Пока провайдер ничего не сообщил,
    вся история считается по символам. ``record`` вызывать после того, как ход дописан в историю.
    """

    prompt_tokens: int = 0
    output_tokens: int = 0
    counted_from: int = 0  # первое сообщение, не вошедшее в prompt_tokens

    def record(self, turn: ModelTurn, messages_len: int) -> None:
        self.prompt_tokens = turn.prompt_tokens
        self.output_tokens = turn.output_tokens
        self.counted_from = messages_len

    def estimate(self, messages: Sequence[Mapping[str, Any]]) -> int:
        if self.prompt_tokens <= 0:
            return _chars_tokens(messages)
        return self.prompt_tokens + self.output_tokens + _chars_tokens(messages[self.counted_from :])


async def run_subagent(
    *,
    name: str,
    role: str,
    model_alias: str,
    client: ChatModel,
    system_prompt: str,
    task: str,
    budget: TaskBudget,
    retriever: Retriever,
    top_k: int,
    return_max_chars: int,
    context_limit: int | None = None,
) -> tuple[str, SubagentRun]:
    """Выполнить одно задание; вернуть ``(текст для оркестратора, SubagentRun)``.

    Текст — ответ субагента; если ответа нет — последний текст с префиксом ``[budget exhausted: <причина>]``;
    при сбое провайдера — ``[SUBAGENT_INFRA_FAILURE] ...``. Всегда обрезается до ``return_max_chars``.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]
    tools = [search_tool_schema()]
    steps = 0  # попытки вызова модели, включая упавшую
    tokens = 0
    search_calls = 0
    last_text = ""
    meter = ContextMeter()

    def finish(text: str, **flags: Any) -> tuple[str, SubagentRun]:
        return _result(
            name=name,
            role=role,
            model_alias=model_alias,
            steps=steps,
            tokens=tokens,
            search_calls=search_calls,
            text=text,
            max_chars=return_max_chars,
            **flags,
        )

    def context_stop() -> tuple[str, SubagentRun]:
        return finish(f"[budget exhausted: {CONTEXT_STOP}] {last_text}".strip(), stop_reason=CONTEXT_STOP)

    while True:
        stop = budget.subagent_stop_reason(name)
        if stop is not None:
            return finish(f"[budget exhausted: {stop.value}] {last_text}".strip(), stop_reason=stop.value)

        estimate = meter.estimate(messages)
        if context_limit is not None and estimate >= context_limit:
            return context_stop()

        reserved = budget.reserve_subagent_step(name)
        # Потолок не больше брони: закрывается именно бронь.
        max_tokens = reserved if context_limit is None else min(reserved, context_limit - estimate)
        actual = 0
        try:
            turn = await client.complete(messages, tools=tools, max_tokens=max_tokens)
            actual = turn.output_tokens
        except Exception as exc:
            steps += 1
            if is_context_length_error(exc):
                return context_stop()
            detail = f"{type(exc).__name__}: {exc}"[:_MAX_ERROR_CHARS]
            return finish(f"{INFRA_FAILURE_MARKER} {detail}", stop_reason="infra_failure", infra_failure=True)
        finally:
            budget.commit_subagent(name, reserved, actual)

        steps += 1
        tokens += actual
        if turn.text.strip():
            last_text = turn.text.strip()

        if not turn.tool_calls:
            answer = (turn.final_answer or "").strip()
            if answer and turn.finish_reason != "length":
                return finish(answer, finished=True)
            stop_text = "max_tokens" if turn.finish_reason == "length" else "empty_answer"
            return finish(f"[budget exhausted: {stop_text}] {last_text}".strip(), stop_reason=stop_text)

        messages.append(assistant_message(turn.text, turn.tool_calls))
        meter.record(turn, len(messages))
        for call in turn.tool_calls:
            if call.name == SEARCH_TOOL_NAME:
                content, is_error, infra_error = await run_search(retriever, call.arguments, top_k)
                if executed(is_error, infra_error):
                    search_calls += 1
            else:
                content = f"subagents may only use {SEARCH_TOOL_NAME}"
            messages.append({"role": "tool", "tool_call_id": call.call_id, "name": call.name, "content": content})


def _result(
    *,
    name: str,
    role: str,
    model_alias: str,
    steps: int,
    tokens: int,
    search_calls: int,
    text: str,
    max_chars: int,
    finished: bool = False,
    stop_reason: str | None = None,
    infra_failure: bool = False,
) -> tuple[str, SubagentRun]:
    run = SubagentRun(
        name=name,
        role=role,
        model_alias=model_alias,
        steps=steps,
        finished=finished,
        stop_reason=stop_reason,
        tokens=tokens,
        infra_failure=infra_failure,
        search_calls=search_calls,
    )
    return text[:max_chars], run


# --- пул экземпляров эпизода ---


@dataclass(frozen=True, slots=True)
class _Instance:
    name: str
    role: str
    model_alias: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SubagentPool:
    """Экземпляры субагентов одного эпизода и исполнение их заданий в рамках бюджета вопроса.

    ``clients`` — алиас модели → клиент; ``semaphore`` — общий для всех эпизодов лимит одновременных
    субагентов (бережёт квоту API), ``limits.max_parallel`` — лимит на один ход.
    """

    def __init__(
        self,
        *,
        clients: Mapping[str, ChatModel],
        role_prompts: Mapping[str, str],
        budget: TaskBudget,
        limits: TaskLimits,
        retriever: Retriever,
        top_k: int,
        return_max_chars: int,
        semaphore: asyncio.Semaphore,
        context_limit: int | None = None,
    ) -> None:
        if not clients:
            raise ValueError("SubagentPool needs at least one model alias")
        if not role_prompts:
            raise ValueError("SubagentPool needs at least one role prompt")
        self._clients = dict(clients)
        self._role_prompts = dict(role_prompts)
        self._budget = budget
        self._limits = limits
        self._retriever = retriever
        self._top_k = top_k
        self._return_max_chars = return_max_chars
        self._semaphore = semaphore
        self._context_limit = context_limit  # None — окно не отслеживаем
        self._instances: dict[str, _Instance] = {}

    @property
    def role_names(self) -> tuple[str, ...]:
        return tuple(self._role_prompts)

    @property
    def model_aliases(self) -> tuple[str, ...]:
        return tuple(self._clients)

    def create(self, call_id: str, name: str, role: str, model: str) -> Observation:
        """Зарегистрировать экземпляр; при ошибке — ``Observation`` с ошибкой, не исключение."""
        if not isinstance(name, str) or not name.strip():
            return _error(call_id, CREATE_SUBAGENT_TOOL, "create_subagent requires a non-empty name")
        if not isinstance(role, str) or role not in self._role_prompts:
            return _error(
                call_id, CREATE_SUBAGENT_TOOL, f"unknown subagent role: {role!r}; allowed: {list(self.role_names)}"
            )
        if not isinstance(model, str) or model not in self._clients:
            return _error(
                call_id, CREATE_SUBAGENT_TOOL, f"unknown model alias: {model!r}; allowed: {list(self.model_aliases)}"
            )
        try:
            self._budget.register_subagent(name)
        except ValueError as exc:  # дубль имени или лимит субагентов — проверяет бюджет
            return _error(call_id, CREATE_SUBAGENT_TOOL, str(exc))
        self._instances[name] = _Instance(name=name, role=role, model_alias=model)
        content = json.dumps({"agent": name, "role": role, "model": model}, ensure_ascii=False)
        return Observation(call_id=call_id, tool_name=CREATE_SUBAGENT_TOOL, content=content)

    async def assign_many(self, calls: Sequence[tuple[str, str, str]]) -> list[tuple[Observation, SubagentRun | None]]:
        """Выполнить задания ``(call_id, agent, prompt)`` параллельно; результаты — в порядке входа.

        ``run`` = None, если задание отклонено до запуска (неизвестный субагент, пустой prompt).
        """
        gate = asyncio.Semaphore(self._limits.max_parallel)

        async def one(call_id: str, agent: str, prompt: str) -> tuple[Observation, SubagentRun | None]:
            instance = self._instances.get(agent) if isinstance(agent, str) else None
            if instance is None:
                return _error(call_id, ASSIGN_TASK_TOOL, f"unknown subagent instance: {agent!r}"), None
            if not isinstance(prompt, str) or not prompt.strip():
                return _error(call_id, ASSIGN_TASK_TOOL, "assign_task requires a non-empty prompt"), None
            async with gate, instance.lock, self._semaphore:
                text, run = await run_subagent(
                    name=instance.name,
                    role=instance.role,
                    model_alias=instance.model_alias,
                    client=self._clients[instance.model_alias],
                    system_prompt=self._role_prompts[instance.role],
                    task=prompt.strip(),
                    budget=self._budget,
                    retriever=self._retriever,
                    top_k=self._top_k,
                    return_max_chars=self._return_max_chars,
                    context_limit=self._context_limit,
                )
            observation = Observation(
                call_id=call_id,
                tool_name=ASSIGN_TASK_TOOL,
                content=text,
                is_error=run.infra_failure,
                infra_error=run.infra_failure,
                metadata={
                    "subagent_steps": run.steps,
                    "subagent_tokens": run.tokens,
                    "subagent_search_calls": run.search_calls,
                },
            )
            return observation, run

        return list(await asyncio.gather(*(one(*call) for call in calls)))


def _error(call_id: str, tool_name: str, message: str) -> Observation:
    return Observation(call_id=call_id, tool_name=tool_name, content=message, is_error=True)
