"""Типы ядра: неизменяемые записи эпизода и протоколы зависимостей (внизу файла).

Ничего от провайдеров и SDK: клиенты, HF-объекты и ответы API живут в :mod:`adapters`.
Эпизод — один вопрос и вся попытка на него ответить; трасса чередует ходы оркестратора и выводы инструментов.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class EventSource(StrEnum):
    ORCHESTRATOR = "orchestrator"
    OBSERVATION = "observation"


class EpisodeStatus(StrEnum):
    COMPLETED = "completed"
    NO_FINAL_ANSWER = "no_final_answer"
    MODEL_ERROR = "model_error"


class GradeStatus(StrEnum):
    GRADED = "graded"
    SKIPPED_NO_ANSWER = "skipped_no_answer"
    TRANSPORT_ERROR = "judge_transport_error"
    PARSE_ERROR = "judge_parse_error"


@dataclass(frozen=True, slots=True)
class Question:
    question_id: str
    text: str
    gold_answers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError("question_id must not be empty")
        if not self.text.strip():
            raise ValueError("question text must not be empty")


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("tool call_id must not be empty")
        if not self.name.strip():
            raise ValueError("tool name must not be empty")


@dataclass(frozen=True, slots=True)
class ModelTurn:
    text: str
    output_tokens: int
    finish_reason: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    final_answer: str | None = None  # текст хода без вызовов инструментов; при вызовах всегда None
    elapsed_seconds: float = 0.0
    prompt_tokens: int = 0  # размер промпта по данным провайдера; 0 — провайдер не сообщил

    def __post_init__(self) -> None:
        if self.output_tokens < 0:
            raise ValueError("output_tokens must be >= 0")
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be >= 0")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be >= 0")
        if self.tool_calls and self.final_answer is not None:
            raise ValueError("a turn cannot contain both tool calls and a final answer")


@dataclass(frozen=True, slots=True)
class Observation:
    """Вывод инструмента, привязанный к вызову по ``call_id``."""

    call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    infra_error: bool = False  # упало окружение (поиск, API); при ошибке модели остаётся False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """Элемент трассы эпизода. ``trainable`` — только у ходов оркестратора: выводы инструментов в лосс не идут."""

    source: EventSource
    content: str
    token_count: int
    trainable: bool
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    tool_name: str | None = None
    is_error: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise ValueError("token_count must be >= 0")
        expected = self.source is EventSource.ORCHESTRATOR
        if self.trainable is not expected:
            raise ValueError("only orchestrator events may be trainable")
        if self.source is EventSource.OBSERVATION and not self.tool_call_id:
            raise ValueError("an observation must reference its tool call")


@dataclass(frozen=True, slots=True)
class SubagentRun:
    """Стоимость и исход одного ``assign_task``."""

    name: str
    role: str
    model_alias: str
    steps: int
    finished: bool
    stop_reason: str | None
    tokens: int
    infra_failure: bool = False
    search_calls: int = 0

    def __post_init__(self) -> None:
        if self.steps < 0 or self.tokens < 0 or self.search_calls < 0:
            raise ValueError("subagent steps, tokens and search_calls must be >= 0")


@dataclass(frozen=True, slots=True)
class Stage:
    """Ход оркестратора с вызовами инструментов и запущенные в нём субагенты (вход для ``r_parallel``)."""

    turn_index: int
    subagent_runs: tuple[SubagentRun, ...] = ()


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Итог эпизода: ответ, статус, трасса и счётчики."""

    question_id: str
    final_answer: str | None
    status: EpisodeStatus
    stop_reason: str | None
    error: str | None
    strategy: str
    elapsed_seconds: float
    events: tuple[TraceEvent, ...]
    stages: tuple[Stage, ...]
    budget: Mapping[str, Any]  # TaskBudget.snapshot(): ходы, токены и т.п.
    tool_errors: int
    infra_errors: int


@dataclass(frozen=True, slots=True)
class GradeResult:
    status: GradeStatus
    correct: bool | None
    confidence: float | None = None
    extracted_final_answer: str | None = None
    reasoning: str | None = None
    attempts: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SearchResult:
    doc_id: str
    text: str
    score: float


# --- протоколы ---


class ChatModel(Protocol):
    """Chat-completion без стриминга: оркестратор или модель субагента."""

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None,
        max_tokens: int,
        thinking: bool | None = None,
    ) -> ModelTurn:
        """``max_tokens`` — жёсткий потолок генерации; ``thinking`` переопределяет режим на один запрос."""


class Retriever(Protocol):
    """Синхронный поиск; стратегии зовут его через ``asyncio.to_thread``."""

    def search(self, query: str, top_k: int) -> Sequence[SearchResult]:
        """Документы по убыванию релевантности."""


class Judge(Protocol):
    async def grade(self, question: Question, answer: str | None) -> GradeResult:
        """Оценивается только ответ, без траектории."""


class Strategy(Protocol):
    """Полный эпизод на один вопрос."""

    async def run(self, question: Question) -> EpisodeResult:
        """Прогнать эпизод."""
