"""Судья BrowseComp-Plus: LLM сравнивает ответ с эталоном.

Сетевые повторы делает ``ChatClient``; здесь переспрашиваем только при неразборчивом JSON вердикта.
Любой сбой — явный статус оценки, а не вердикт «неверно».
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Protocol

from ..config import JudgeConfig
from ..domain import GradeResult, GradeStatus, ModelTurn, Question
from .chat import ChatClient, ModelProtocolError, ModelRequestError, api_key_from_env

USER_PROMPT = """Judge whether the following [response] to [question] is correct based on the [correct_answer].

[question]: {question}

[response]: {answer}

[correct_answer]: {correct_answer}

Respond with a JSON object containing exactly these fields:
- "extracted_final_answer": the exact final answer extracted from the response (string, or "None" if no answer found)
- "reasoning": brief explanation of whether the extracted answer matches the correct answer (string)
- "correct": whether the answer is correct ("yes" or "no")
- "confidence": the confidence score from the response as a number 0-100 (integer, default 100)

Respond ONLY with the JSON object, no other text."""

JSON_OBJECT_FORMAT = {"type": "json_object"}


class JudgeModel(Protocol):
    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        max_tokens: int,
        response_format: Mapping[str, Any] | None = None,
    ) -> ModelTurn: ...


class OpenAICompatibleJudge:
    """Оценить ответ; сбой запроса никогда не превращается в «неверно»."""

    def __init__(self, config: JudgeConfig, *, client: JudgeModel | None = None) -> None:
        self._config = config
        self._client = client or ChatClient(
            base_url=config.base_url,
            model=config.model,
            api_key=api_key_from_env(config.api_key_env),
            temperature=config.temperature,
            top_p=1.0,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            extra_body=dict(config.extra_body),
        )

    async def grade(self, question: Question, answer: str | None) -> GradeResult:
        candidate = (answer or "").strip()
        if not candidate:
            return GradeResult(status=GradeStatus.SKIPPED_NO_ANSWER, correct=None)
        messages = _messages(question, candidate)
        attempts = self._config.max_retries + 1
        last = GradeResult(status=GradeStatus.PARSE_ERROR, correct=None)
        for attempt in range(1, attempts + 1):
            try:
                turn = await self._client.complete(
                    messages,
                    max_tokens=self._config.max_tokens,
                    response_format=JSON_OBJECT_FORMAT,
                )
            except (ModelRequestError, ModelProtocolError) as exc:
                # Упавший запрос ничего не говорит о качестве ответа — отдельно от вердикта «no».
                return GradeResult(
                    status=GradeStatus.TRANSPORT_ERROR,
                    correct=None,
                    attempts=attempt,
                    error=f"{type(exc).__name__}: {exc}",
                )
            last = parse_judge_response(turn.text)
            if last.status is GradeStatus.GRADED:
                return replace(last, attempts=attempt)
        # Битый вывод бывает случайным (например, обрезка у провайдера): переспрашиваем в пределах лимита,
        # но в вердикт молча не превращаем.
        return replace(
            last,
            attempts=attempts,
            error="judge response was not valid JSON with a yes/no correctness verdict",
        )


def _messages(question: Question, candidate: str) -> list[dict[str, str]]:
    """Запрос судье: вопрос, ответ модели и один эталонный ответ."""
    prompt = USER_PROMPT.format(
        question=question.text,
        answer=candidate,
        correct_answer=question.gold_answers[0] if question.gold_answers else "",
    )
    return [{"role": "user", "content": prompt}]


def parse_judge_response(text: str) -> GradeResult:
    """Первый JSON-объект в тексте; допускается текст или ``` вокруг."""
    start = text.find("{")
    if start < 0:
        return GradeResult(status=GradeStatus.PARSE_ERROR, correct=None)
    try:
        payload, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError:
        return GradeResult(status=GradeStatus.PARSE_ERROR, correct=None)
    verdict = payload.get("correct")
    if isinstance(verdict, str):
        normalized = verdict.strip().lower()
        correct = True if normalized == "yes" else False if normalized == "no" else None
    elif isinstance(verdict, bool):
        correct = verdict
    else:
        correct = None
    if correct is None:
        return GradeResult(status=GradeStatus.PARSE_ERROR, correct=None)
    extracted = payload.get("extracted_final_answer")
    reasoning = payload.get("reasoning")
    return GradeResult(
        status=GradeStatus.GRADED,
        correct=correct,
        confidence=_confidence(payload.get("confidence")),
        extracted_final_answer=str(extracted) if extracted is not None else None,
        reasoning=str(reasoning) if reasoning is not None else None,
        attempts=1,
    )


def _confidence(value: Any) -> float | None:
    try:
        numeric = float(str(value).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    if not 0 <= numeric <= 100:
        return None
    return numeric
