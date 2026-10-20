"""Общие фейки и фикстуры тестов. Все фейки детерминированы и в сеть не ходят."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from orchestrator_research import assembly
from orchestrator_research.adapters.fake import ScriptedChatModel, StaticRetriever
from orchestrator_research.config import ExperimentConfig, load_config
from orchestrator_research.domain import ChatModel, GradeResult, GradeStatus, Question

ROOT = Path(__file__).parents[1]
EXPERIMENT = ROOT / "experiments/template/config.toml"


def correct_grade() -> GradeResult:
    return GradeResult(status=GradeStatus.GRADED, correct=True, confidence=90.0, attempts=1)


class FakeJudge:
    """Судья: на любой ответ ``grade``; с ``error`` — бросает его; id из ``transport_fail`` падают по одному разу."""

    def __init__(
        self,
        grade: GradeResult | None = None,
        *,
        error: BaseException | None = None,
        transport_fail: tuple[str, ...] = (),
    ) -> None:
        self._grade = grade or correct_grade()
        self._error = error
        self._transport_fail = set(transport_fail)
        self.calls: list[str] = []

    async def grade(self, question: Question, answer: str | None) -> GradeResult:
        self.calls.append(question.question_id)
        if self._error is not None:
            raise self._error
        if question.question_id in self._transport_fail:
            self._transport_fail.discard(question.question_id)
            return GradeResult(status=GradeStatus.TRANSPORT_ERROR, correct=None, error="timeout")
        return self._grade


@pytest.fixture
def experiment_config() -> ExperimentConfig:
    return load_config(EXPERIMENT)


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Подменить сетевые билдеры в ``assembly`` фейками; код эпизода и награды остаётся настоящим."""

    def _wire(
        *,
        orchestrator: ChatModel,
        judge: Any,
        subagent: ChatModel | None = None,
        retriever: Any = None,
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        monkeypatch.setattr(assembly, "build_orchestrator_client", lambda cfg, **kwargs: orchestrator)
        monkeypatch.setattr(assembly, "build_retriever", lambda cfg: retriever or StaticRetriever())
        monkeypatch.setattr(assembly, "build_judge", lambda cfg: judge)
        monkeypatch.setattr(assembly, "build_subagent_client", lambda cfg, model_id: subagent or ScriptedChatModel([]))

    return _wire
