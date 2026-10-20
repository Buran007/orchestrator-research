"""Судья: разбор вердикта, повтор при нечитаемом ответе, ошибки транспорта без вердикта."""

from __future__ import annotations

import asyncio

import pytest

from orchestrator_research.adapters.chat import ModelRequestError
from orchestrator_research.adapters.judge import OpenAICompatibleJudge, parse_judge_response
from orchestrator_research.config import JudgeConfig
from orchestrator_research.domain import GradeStatus, ModelTurn, Question

GRADED_NO = '{"correct":"no","confidence":100,"reasoning":"wrong"}'
QUESTION = Question("q1", "Question?", ("gold",))


def make_config(retries: int = 1) -> JudgeConfig:
    return JudgeConfig(
        base_url="http://localhost:8000/v1",
        model="judge-test",
        api_key_env="",
        temperature=0,
        timeout_seconds=5,
        max_retries=retries,
        max_tokens=128,
    )


class ScriptedJudgeModel:
    """Отдаёт заготовленные ответы по порядку, последний повторяется; исключение бросается вместо ответа."""

    def __init__(self, *outcomes):
        self._outcomes = list(outcomes)
        self.requests: list[dict] = []

    async def complete(self, messages, *, tools=None, max_tokens, response_format=None):
        self.requests.append({"max_tokens": max_tokens, "response_format": response_format})
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return ModelTurn(text=outcome, output_tokens=1, finish_reason="stop")


def grade(judge: OpenAICompatibleJudge, answer: str | None):
    return asyncio.run(judge.grade(QUESTION, answer))


def test_parses_object_wrapped_in_prose_and_fences():
    verdict = '{"correct":"yes","confidence":"91%","extracted_final_answer":"Paris","reasoning":"matches"}'
    text = f"prefix ```json\n{verdict}\n``` suffix"
    result = parse_judge_response(text)
    assert result.status is GradeStatus.GRADED
    assert result.correct is True
    assert result.confidence == 91
    assert result.extracted_final_answer == "Paris"


@pytest.mark.parametrize("text", ["not json", '{"correct":"maybe"}', '{"correct":1}', '{"correct":"yes"'])
def test_malformed_or_ambiguous_verdict_is_parse_failure(text):
    assert parse_judge_response(text).status is GradeStatus.PARSE_ERROR


def test_empty_answer_is_never_sent_to_the_judge():
    model = ScriptedJudgeModel("unused")
    result = grade(OpenAICompatibleJudge(make_config(), client=model), "  ")
    assert result.status is GradeStatus.SKIPPED_NO_ANSWER
    assert model.requests == []


def test_judge_requests_json_object_with_configured_token_cap():
    model = ScriptedJudgeModel(GRADED_NO)
    grade(OpenAICompatibleJudge(make_config(), client=model), "candidate")
    assert model.requests == [{"max_tokens": 128, "response_format": {"type": "json_object"}}]


def test_reasks_after_unparseable_output_and_reports_attempts():
    model = ScriptedJudgeModel("bad response", GRADED_NO)
    result = grade(OpenAICompatibleJudge(make_config(), client=model), "candidate")
    assert result.status is GradeStatus.GRADED
    assert result.correct is False
    assert result.attempts == 2


def test_persistent_parse_failure_stays_distinct_from_a_verdict():
    model = ScriptedJudgeModel("bad response")
    result = grade(OpenAICompatibleJudge(make_config(), client=model), "candidate")
    assert result.status is GradeStatus.PARSE_ERROR
    assert result.correct is None
    assert result.attempts == 2
    assert len(model.requests) == 2


def test_transport_failure_fails_closed_without_a_verdict():
    model = ScriptedJudgeModel(ModelRequestError("model request failed: ConnectionError: offline"))
    result = grade(OpenAICompatibleJudge(make_config(), client=model), "candidate")
    assert result.status is GradeStatus.TRANSPORT_ERROR
    assert result.correct is None
    assert len(model.requests) == 1
