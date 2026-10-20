"""Цикл оркестратора (стратегии single и swarm) на заскриптованных моделях: бюджет, финальный ход,
ошибки инструментов и модели, страж контекстного окна.
"""

from __future__ import annotations

import asyncio

import pytest

from orchestrator_research.adapters.chat import ModelRequestError
from orchestrator_research.adapters.fake import ScriptedChatModel, StaticRetriever
from orchestrator_research.domain import EpisodeStatus, EventSource, ModelTurn, Question, SearchResult, ToolCall
from orchestrator_research.limits import TaskBudget, TaskLimits
from orchestrator_research.strategies import STRATEGIES, get_strategy
from orchestrator_research.strategies.base import EpisodeEnvironment
from orchestrator_research.subagents import SubagentPool
from orchestrator_research.tools import ASSIGN_TASK_TOOL, FINALIZER_SYSTEM_PROMPT, SEARCH_TOOL_NAME, executed

BUDGET_STOPS = {"orchestrator_budget", "total_budget", "turns"}

ROLES = {"retriever": "prompt:retriever", "skeptic": "prompt:skeptic"}
MODELS = {"small": "provider/small", "big": "provider/big"}
HITS = (SearchResult("d1", "evidence text", 0.9),)
QUESTION = Question("q1", "What is the answer?")


def turn(text="", output_tokens=3, calls=(), finish="stop"):
    return ModelTurn(
        text=text,
        output_tokens=output_tokens,
        finish_reason="tool_calls" if calls else finish,
        tool_calls=tuple(calls),
        final_answer=None if calls or not text else text,
    )


def search_call(call_id, query="x"):
    return ToolCall(call_id, "browsecomp_search", {"query": query})


def create_call(call_id, name="lead", role="retriever", model="small"):
    return ToolCall(call_id, "create_subagent", {"name": name, "role": role, "model": model})


def assign_call(call_id, agent="lead", prompt="find"):
    return ToolCall(call_id, "assign_task", {"agent": agent, "prompt": prompt})


class FailingRetriever:
    def search(self, query, top_k):
        raise ConnectionError("retrieval server down")


def observations(result):
    return [event for event in result.events if event.source is EventSource.OBSERVATION]


def ran_calls(result, tool_name):
    """Исполненные вызовы ``tool_name``; отказы валидации и вызовы сверх лимита не в счёт."""
    return [
        event
        for event in observations(result)
        if event.tool_name == tool_name and executed(event.is_error, event.metadata.get("infra_error", False))
    ]


def make_env(orchestrator_turns, *, limits=None, subagent_turns=(), retriever=None):
    limits = limits or TaskLimits()
    budget = TaskBudget(limits)
    retriever = retriever or StaticRetriever(HITS)
    orchestrator = ScriptedChatModel(orchestrator_turns)
    subagent_model = ScriptedChatModel(subagent_turns)
    pool = SubagentPool(
        clients={alias: subagent_model for alias in MODELS},
        role_prompts=ROLES,
        budget=budget,
        limits=limits,
        retriever=retriever,
        top_k=5,
        return_max_chars=300,
        semaphore=asyncio.Semaphore(8),
    )
    env = EpisodeEnvironment(
        orchestrator=orchestrator,
        pool=pool,
        retriever=retriever,
        budget=budget,
        limits=limits,
        top_k=5,
    )
    return env, orchestrator, subagent_model


def run_strategy(name, env, question=QUESTION):
    return asyncio.run(get_strategy(name)(env).run(question))


def tool_names(orchestrator: ScriptedChatModel, index: int = 0) -> list[str]:
    return [tool["function"]["name"] for tool in orchestrator.calls[index]["tools"]]


# ----- реестр и контракт -----------------------------------------------------


@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_every_registered_strategy_completes_on_a_scripted_answer(name):
    env, _, _ = make_env([turn("Paris", 2)])
    result = run_strategy(name, env)
    assert result.status is EpisodeStatus.COMPLETED
    assert result.final_answer == "Paris"
    assert result.strategy == name


def test_unknown_strategy_lists_available_names():
    with pytest.raises(ValueError, match="unknown strategy.*single.*swarm"):
        get_strategy("does_not_exist")


# ----- swarm -----------------------------------------------------------------


def test_delegation_records_stages_subagent_cost_and_append_only_history():
    env, orchestrator, _ = make_env(
        [
            turn(output_tokens=4, calls=(search_call("s1"),)),
            turn(output_tokens=6, calls=(create_call("c1"), assign_call("a1"))),
            turn("Paris", 3),
        ],
        subagent_turns=[turn("lead facts", output_tokens=4)],
    )
    result = run_strategy("swarm", env)

    assert result.final_answer == "Paris"
    assert result.stop_reason == "answered"
    assert result.budget["orchestrator_turns"] == 3
    assert result.budget["orchestrator_tokens"] == 13
    assert [stage.turn_index for stage in result.stages] == [0, 1]
    assert result.stages[0].subagent_runs == ()
    (run,) = result.stages[1].subagent_runs
    assert (run.name, run.role, run.model_alias) == ("lead", "retriever", "small")
    assert run.finished and run.tokens == 4
    assert (
        len(observations(result)),
        len(ran_calls(result, SEARCH_TOOL_NAME)),
        len(ran_calls(result, ASSIGN_TASK_TOOL)),
    ) == (3, 1, 1)
    assert sum(result.budget["subagent_tokens"].values()) == 4
    observation = next(e for e in result.events if e.tool_call_id == "a1")
    assert observation.source is EventSource.OBSERVATION
    assert observation.content == "lead facts"
    assert set(tool_names(orchestrator)) == {"browsecomp_search", "create_subagent", "assign_task"}
    # История только дописывается: каждый запрос продолжает предыдущий (общий префикс для Agent Lightning).
    assert (
        orchestrator.calls[1]["messages"][: len(orchestrator.calls[0]["messages"])] == orchestrator.calls[0]["messages"]
    )
    assert (
        orchestrator.calls[2]["messages"][: len(orchestrator.calls[1]["messages"])] == orchestrator.calls[1]["messages"]
    )
    last_tool = orchestrator.calls[2]["messages"][-1]
    assert (last_tool["role"], last_tool["tool_call_id"]) == ("tool", "a1")


def test_finalizer_uses_only_the_reserved_answer_budget_and_is_not_a_turn():
    limits = TaskLimits(orchestrator_tokens_per_task=100, orchestrator_tokens_per_turn=60, final_answer_reserve=20)
    env, orchestrator, _ = make_env(
        [
            turn(output_tokens=60, calls=(search_call("s1"),)),
            turn(output_tokens=20, calls=(search_call("s2"),)),
            turn("final answer", output_tokens=20),
        ],
        limits=limits,
    )
    result = run_strategy("swarm", env)

    assert result.final_answer == "final answer"
    assert result.stop_reason == "orchestrator_budget"
    assert result.stop_reason in BUDGET_STOPS
    assert result.budget["orchestrator_tokens"] == 100
    assert result.budget["orchestrator_turns"] == 2  # финальный ход ходом не считается
    assert orchestrator.calls[-1]["max_tokens"] == 20
    assert orchestrator.calls[-1]["tools"] == []
    assert orchestrator.calls[-1]["thinking"] is False  # в резерв должен влезть ответ, а не <think>
    assert all(call["thinking"] is None for call in orchestrator.calls[:-1])  # обычные ходы — режим из конфига
    assert env.budget.snapshot()["overshoot_tokens"] == 0


def test_total_budget_including_subagents_is_never_exceeded():
    limits = TaskLimits(
        total_tokens_per_task=60, final_answer_reserve=10, orchestrator_tokens_per_turn=30, subagent_tokens_per_step=30
    )
    env, _, _ = make_env(
        [turn(output_tokens=20, calls=(create_call("c1"), assign_call("a1"))), turn("ans", output_tokens=10)],
        limits=limits,
        subagent_turns=[turn(output_tokens=30, calls=(search_call("s1"),))],
    )
    result = run_strategy("swarm", env)

    assert result.final_answer == "ans"
    assert result.stop_reason == "total_budget"
    assert env.budget.total_tokens == 60
    assert env.budget.snapshot()["overshoot_tokens"] == 0
    (run,) = result.stages[0].subagent_runs
    assert (run.stop_reason, run.finished) == ("total_budget", False)


def test_too_many_tool_calls_get_error_observations_and_are_not_counted_as_searches():
    env, _, _ = make_env(
        [
            turn(output_tokens=3, calls=(search_call("s1"), search_call("s2"), search_call("s3"))),
            turn("ok", output_tokens=2),
        ],
        limits=TaskLimits(max_tool_calls_per_turn=2),
    )
    result = run_strategy("swarm", env)

    assert result.status is EpisodeStatus.COMPLETED
    assert result.final_answer == "ok"
    rejected = next(e for e in result.events if e.tool_call_id == "s3")
    assert rejected.is_error and "too many tool calls" in rejected.content
    assert (result.tool_errors, len(ran_calls(result, SEARCH_TOOL_NAME)), len(observations(result))) == (1, 2, 3)


def test_assignment_to_unknown_instance_is_an_error_observation():
    env, _, _ = make_env([turn(output_tokens=3, calls=(assign_call("a1", agent="nobody"),)), turn("ok", 2)])
    result = run_strategy("swarm", env)
    observation = next(e for e in result.events if e.tool_call_id == "a1")
    assert observation.is_error
    assert len(ran_calls(result, ASSIGN_TASK_TOOL)) == 0
    assert result.final_answer == "ok"


def test_model_failure_is_recorded_as_episode_status():
    env, _, _ = make_env([ModelRequestError("provider down")])
    result = run_strategy("swarm", env)
    assert result.status is EpisodeStatus.MODEL_ERROR
    assert "ModelRequestError" in result.error
    assert result.final_answer is None
    assert result.budget["orchestrator_turns"] == 1  # бронь упавшего запроса всё равно закрыта
    assert env.budget.orchestrator_tokens == 0


def test_retriever_outage_is_an_infrastructure_error_and_still_counts_as_a_search():
    env, _, _ = make_env(
        [turn(output_tokens=3, calls=(search_call("s1"),)), turn("ok", 2)],
        retriever=FailingRetriever(),
    )
    result = run_strategy("swarm", env)
    observation = next(e for e in result.events if e.tool_call_id == "s1")
    assert observation.is_error
    assert observation.metadata["infra_error"] is True
    assert (len(ran_calls(result, SEARCH_TOOL_NAME)), result.tool_errors, result.infra_errors) == (1, 1, 1)
    assert result.final_answer == "ok"


def test_search_with_bad_arguments_is_a_model_mistake_not_an_infra_error():
    bad = ToolCall("s1", "browsecomp_search", {"query": ""})
    env, _, _ = make_env([turn(output_tokens=3, calls=(bad,)), turn("ok", 2)])
    result = run_strategy("swarm", env)
    observation = next(e for e in result.events if e.tool_call_id == "s1")
    assert observation.is_error
    assert observation.metadata["infra_error"] is False
    assert (len(ran_calls(result, SEARCH_TOOL_NAME)), result.infra_errors) == (0, 0)


def test_answer_cut_off_at_max_tokens_is_not_final_and_goes_to_finalizer():
    env, orchestrator, _ = make_env(
        [turn("<think>still reasoning", output_tokens=5, finish="length"), turn("Paris", output_tokens=2)],
    )
    result = run_strategy("swarm", env)
    assert result.final_answer == "Paris"
    assert result.stop_reason == "no_answer"
    assert orchestrator.calls[-1]["tools"] == []


def test_finalizer_cut_off_at_max_tokens_yields_no_final_answer():
    limits = TaskLimits(orchestrator_tokens_per_task=100, orchestrator_tokens_per_turn=60, final_answer_reserve=20)
    env, _, _ = make_env(
        [
            turn(output_tokens=60, calls=(search_call("s1"),)),
            turn(output_tokens=20, calls=(search_call("s2"),)),
            turn("<think>unclosed", output_tokens=20, finish="length"),
        ],
        limits=limits,
    )
    result = run_strategy("swarm", env)
    assert result.final_answer is None
    assert result.status is EpisodeStatus.NO_FINAL_ANSWER


# ----- single ----------------------------------------------------------------


def test_single_exposes_only_search_and_rejects_subagent_calls():
    env, orchestrator, _ = make_env([turn(output_tokens=3, calls=(create_call("c1"),)), turn("ok", 2)])
    result = run_strategy("single", env)

    assert tool_names(orchestrator) == ["browsecomp_search"]
    assert len(ran_calls(result, ASSIGN_TASK_TOOL)) == 0
    rejected = next(e for e in result.events if e.tool_call_id == "c1")
    assert rejected.is_error
    assert result.final_answer == "ok"
    assert result.stages[0].subagent_runs == ()


def test_single_without_pool_runs_on_search_only():
    env, _, _ = make_env([turn(output_tokens=3, calls=(search_call("s1"),)), turn("ok", 2)])
    env = EpisodeEnvironment(
        orchestrator=env.orchestrator,
        pool=None,
        retriever=env.retriever,
        budget=env.budget,
        limits=env.limits,
        top_k=env.top_k,
    )
    result = run_strategy("single", env)
    assert len(ran_calls(result, SEARCH_TOOL_NAME)) == 1
    assert result.final_answer == "ok"


# ----- страж контекста -------------------------------------------------------

# Оценка стартового промпта ~373 токена (system prompt 1092 символа + вопрос); 532 вмещает ровно один
# ход в 60 токенов плюс резерв финального ответа в 20, не больше.
TIGHT = TaskLimits(orchestrator_tokens_per_turn=60, final_answer_reserve=20, max_turns=10)
CONTEXT_LIMIT = 532
LONG_HITS = (SearchResult("d1", "x" * 3000, 0.9),)


def prompt_turn(prompt_tokens, *, output_tokens=20, calls=(), text=""):
    """Ход с размером промпта от провайдера — его и читает страж контекста."""
    return ModelTurn(
        text=text,
        output_tokens=output_tokens,
        finish_reason="tool_calls" if calls else "stop",
        tool_calls=tuple(calls),
        prompt_tokens=prompt_tokens,
    )


def test_context_guard_cuts_a_tool_output_that_would_overflow_instead_of_ending_the_episode():
    from orchestrator_research.strategies.base import CONTEXT_CUT_NOTE

    env, orchestrator, _ = make_env(
        [prompt_turn(400, calls=(search_call("s1"),)), ModelTurn("Paris", 3, "stop", final_answer="Paris")],
        limits=TIGHT,
        retriever=StaticRetriever(LONG_HITS),
    )
    env.context_limit = CONTEXT_LIMIT
    result = run_strategy("swarm", env)
    # Целиком хит в 3000 символов переполнил бы окно; обрезанный до пометки, оставляет место ещё на ход.
    assert orchestrator.calls[-1]["messages"][-1]["content"] == CONTEXT_CUT_NOTE
    assert (result.status, result.stop_reason, result.final_answer) == (EpisodeStatus.COMPLETED, "answered", "Paris")


def test_finalizer_is_capped_to_the_room_left_under_the_context_limit():
    # Оценка перед финальным ходом: последний промпт (400) + последний вывод (20) + инструкция (len // 3).
    finalizer_estimate = 400 + 20 + len(FINALIZER_SYSTEM_PROMPT) // 3
    env, orchestrator, _ = make_env(
        [prompt_turn(400), ModelTurn("Paris", 3, "stop", final_answer="Paris")],
        limits=TIGHT,
    )
    env.context_limit = finalizer_estimate + 5
    result = run_strategy("swarm", env)
    assert orchestrator.calls[-1]["max_tokens"] == 5  # меньше резерва в 20 — сработал потолок окна
    assert (result.status, result.final_answer) == (EpisodeStatus.COMPLETED, "Paris")
    assert result.budget["orchestrator_tokens"] == 23


def test_provider_context_error_ends_the_episode_with_context_stop_and_commits_the_reservation():
    env, orchestrator, _ = make_env(
        [ModelRequestError("model request failed: BadRequestError: This model's maximum context length is 8192 tokens")]
    )
    result = run_strategy("swarm", env)
    assert (result.status, result.stop_reason, result.error) == (EpisodeStatus.NO_FINAL_ANSWER, "context", None)
    assert result.final_answer is None
    assert len(orchestrator.calls) == 1  # после переполнения финального хода нет
    assert result.budget["orchestrator_turns"] == 1
    assert env.budget.orchestrator_tokens == 0


def test_fit_observation_invariants():
    from orchestrator_research.strategies.base import CONTEXT_CUT_NOTE, fit_observation

    assert fit_observation("abc", None) == ("abc", None)
    assert fit_observation("abc", 10) == ("abc", 7)
    cut, left = fit_observation("x" * 100, 40)
    assert (len(cut), left) == (40, 0) and cut.endswith(CONTEXT_CUT_NOTE)
    assert fit_observation("x" * 100, 0) == (CONTEXT_CUT_NOTE, 0)


def test_tool_output_is_cut_so_the_finalizer_keeps_its_reserve():
    from orchestrator_research.strategies.base import CONTEXT_CUT_NOTE

    env, orchestrator, _ = make_env(
        [prompt_turn(400, calls=(search_call("s1"),)), ModelTurn("Paris", 3, "stop", final_answer="Paris")],
        limits=TIGHT,
        retriever=StaticRetriever(LONG_HITS),
    )
    env.context_limit = 700
    result = run_strategy("swarm", env)
    tool_message = orchestrator.calls[-1]["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["content"].endswith(CONTEXT_CUT_NOTE)
    assert len(tool_message["content"]) < 3000  # хит в 3000 символов целиком не вошёл
    assert result.final_answer == "Paris"


def test_finalizer_cut_off_without_thinking_keeps_the_answer():
    limits = TaskLimits(orchestrator_tokens_per_task=100, orchestrator_tokens_per_turn=60, final_answer_reserve=20)
    env, _, _ = make_env(
        [
            turn(output_tokens=60, calls=(search_call("s1"),)),
            turn(output_tokens=20, calls=(search_call("s2"),)),
            turn("Carter Woodson, because the", output_tokens=20, finish="length"),
        ],
        limits=limits,
    )
    result = run_strategy("swarm", env)
    assert result.final_answer == "Carter Woodson, because the"
    assert result.status is EpisodeStatus.COMPLETED
