"""Субагенты: цикл одного субагента (``run_subagent``), пул с лимитами и параллелизмом, страж контекста."""

from __future__ import annotations

import asyncio

import pytest

from orchestrator_research.adapters.chat import ModelRequestError
from orchestrator_research.adapters.fake import ScriptedChatModel, StaticRetriever
from orchestrator_research.domain import ModelTurn, SearchResult, ToolCall
from orchestrator_research.limits import TaskBudget, TaskLimits
from orchestrator_research.subagents import INFRA_FAILURE_MARKER, SubagentPool, run_subagent

ROLES = {"retriever": "prompt:retriever", "skeptic": "prompt:skeptic"}
HITS = (SearchResult("d1", "evidence text", 0.9),)


def turn(text="", output_tokens=5, calls=(), finish="stop"):
    return ModelTurn(
        text=text,
        output_tokens=output_tokens,
        finish_reason="tool_calls" if calls else finish,
        tool_calls=tuple(calls),
        final_answer=None if calls or not text else text,
    )


def search(call_id, query="q"):
    return ToolCall(call_id, "browsecomp_search", {"query": query})


def worker(model, budget, *, name="a"):
    return run_subagent(
        name=name,
        role="retriever",
        model_alias="small",
        client=model,
        system_prompt="p",
        task="find",
        budget=budget,
        retriever=StaticRetriever(HITS),
        top_k=5,
        return_max_chars=500,
    )


def make_pool(clients, *, limits=None, semaphore=None):
    limits = limits or TaskLimits()
    budget = TaskBudget(limits)
    pool = SubagentPool(
        clients=clients,
        role_prompts=ROLES,
        budget=budget,
        limits=limits,
        retriever=StaticRetriever(HITS),
        top_k=5,
        return_max_chars=300,
        semaphore=semaphore or asyncio.Semaphore(64),
    )
    return pool, budget


class ConcurrencyProbe:
    """Модель, которая ненадолго засыпает и запоминает пик одновременных вызовов."""

    def __init__(self, delay=0.005):
        self.in_flight = 0
        self.peak = 0
        self.delay = delay

    async def complete(self, messages, *, tools, max_tokens):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(self.delay)
        self.in_flight -= 1
        return turn(text="done", output_tokens=1)


# ----- один субагент ---------------------------------------------------------


def test_budget_stop_returns_partial_text_with_marker():
    budget = TaskBudget(TaskLimits(subagent_tokens_per_step=20, subagent_tokens_per_task=30))
    budget.register_subagent("a")
    model = ScriptedChatModel(
        [
            turn(output_tokens=20, calls=(search("c1", "alpha"),)),
            turn(text="partial facts", output_tokens=10, calls=(search("c2", "beta"),)),
        ]
    )
    text, run = asyncio.run(worker(model, budget))
    assert text.startswith("[budget exhausted: subagent_budget]")
    assert "partial facts" in text
    assert (run.finished, run.stop_reason) == (False, "subagent_budget")
    assert (run.steps, run.tokens, run.search_calls) == (2, 30, 2)
    assert budget.snapshot()["subagent_tokens"]["a"] == 30
    assert budget.snapshot()["overshoot_tokens"] == 0


def test_infra_failure_is_reported_not_raised_and_counts_the_attempted_step():
    budget = TaskBudget(TaskLimits())
    budget.register_subagent("a")
    model = ScriptedChatModel([ModelRequestError("provider down after retries")])
    text, run = asyncio.run(worker(model, budget))
    assert text.startswith(INFRA_FAILURE_MARKER)
    assert (run.infra_failure, run.finished, run.tokens, run.steps) == (True, False, 0, 1)
    assert budget.snapshot()["subagent_tokens"]["a"] == 0
    assert budget.snapshot()["subagent_steps"]["a"] == 1  # упавший вызов засчитан в лимит шагов


def test_final_answer_is_truncated_to_return_limit():
    budget = TaskBudget(TaskLimits())
    budget.register_subagent("a")
    model = ScriptedChatModel([turn(text="x" * 50, output_tokens=4)])
    text, run = asyncio.run(
        run_subagent(
            name="a",
            role="retriever",
            model_alias="small",
            client=model,
            system_prompt="p",
            task="find",
            budget=budget,
            retriever=StaticRetriever(HITS),
            top_k=5,
            return_max_chars=10,
        )
    )
    assert text == "x" * 10
    assert (run.finished, run.stop_reason) == (True, None)


def test_truncated_generation_is_not_reported_as_finished():
    budget = TaskBudget(TaskLimits())
    budget.register_subagent("a")
    model = ScriptedChatModel([turn(text="cut off", output_tokens=4, finish="length")])
    text, run = asyncio.run(worker(model, budget))
    assert (run.finished, run.stop_reason) == (False, "max_tokens")
    assert text.startswith("[budget exhausted: max_tokens]")


def test_rejected_search_arguments_do_not_count_as_search_calls():
    budget = TaskBudget(TaskLimits())
    budget.register_subagent("a")
    bad = ToolCall("c1", "browsecomp_search", {"query": "  "})
    model = ScriptedChatModel([turn(output_tokens=3, calls=(bad, search("c2"))), turn(text="ok", output_tokens=2)])
    text, run = asyncio.run(worker(model, budget))
    assert text == "ok"
    assert run.search_calls == 1


# ----- пул -------------------------------------------------------------------


def test_invalid_role_or_alias_returns_error_observation_and_creates_nothing():
    pool, budget = make_pool({"small": ScriptedChatModel([])})
    bad_role = pool.create("c1", "a", "admin", "small")
    bad_alias = pool.create("c2", "a", "retriever", "huge")
    assert bad_role.is_error and "unknown subagent role" in bad_role.content
    assert bad_alias.is_error and "unknown model alias" in bad_alias.content
    assert (bad_role.call_id, bad_alias.call_id) == ("c1", "c2")
    assert budget.snapshot()["subagents_created"] == 0


def test_duplicate_name_is_an_error_observation():
    pool, _ = make_pool({"small": ScriptedChatModel([])})
    assert not pool.create("c1", "a", "retriever", "small").is_error
    assert pool.create("c2", "a", "skeptic", "small").is_error


def test_assignment_to_unknown_instance_is_an_error_observation():
    pool, _ = make_pool({"small": ScriptedChatModel([])})
    ((observation, run),) = asyncio.run(pool.assign_many([("c1", "missing", "task")]))
    assert observation.is_error and observation.call_id == "c1"
    assert run is None


def test_subagent_cap_is_enforced_by_the_ledger():
    pool, budget = make_pool({"small": ScriptedChatModel([])}, limits=TaskLimits(max_subagents_per_task=2))
    assert not pool.create("c1", "a", "retriever", "small").is_error
    assert not pool.create("c2", "b", "retriever", "small").is_error
    capped = pool.create("c3", "c", "skeptic", "small")
    assert capped.is_error
    assert "max_subagents_per_task" in capped.content
    assert budget.snapshot()["subagents_created"] == 2


def test_parallelism_never_exceeds_max_parallel():
    probe = ConcurrencyProbe()
    pool, _ = make_pool({"small": probe}, limits=TaskLimits(max_parallel=2))
    for index in range(5):
        pool.create(f"c{index}", f"a{index}", "retriever", "small")
    results = asyncio.run(pool.assign_many([(f"c{i}", f"a{i}", "task") for i in range(5)]))
    assert len(results) == 5
    assert all(run is not None and run.finished for _, run in results)
    assert probe.peak == 2


def test_global_semaphore_bounds_provider_calls_across_pools():
    probe = ConcurrencyProbe()
    shared = asyncio.Semaphore(1)
    pool_one, _ = make_pool({"small": probe}, semaphore=shared)
    pool_two, _ = make_pool({"small": probe}, semaphore=shared)
    pool_one.create("c1", "a", "retriever", "small")
    pool_two.create("c2", "a", "retriever", "small")

    async def both():
        await asyncio.gather(pool_one.assign_many([("c3", "a", "x")]), pool_two.assign_many([("c4", "a", "y")]))

    asyncio.run(both())
    assert probe.peak == 1


def test_same_instance_assignments_are_serialized():
    probe = ConcurrencyProbe()
    pool, _ = make_pool({"small": probe}, limits=TaskLimits(max_parallel=4))
    pool.create("c0", "a", "retriever", "small")
    asyncio.run(pool.assign_many([("c1", "a", "one"), ("c2", "a", "two"), ("c3", "a", "three")]))
    assert probe.peak == 1


def test_tokens_accumulate_per_subagent_without_carrying_dialogue():
    model = ScriptedChatModel([turn(text="first", output_tokens=7), turn(text="second", output_tokens=5)])
    pool, budget = make_pool({"small": model})
    pool.create("c0", "a", "retriever", "small")

    async def two_assignments():
        first = await pool.assign_many([("c1", "a", "task one")])
        second = await pool.assign_many([("c2", "a", "task two")])
        return first[0], second[0]

    (first, first_run), (second, second_run) = asyncio.run(two_assignments())
    assert (first.content, second.content) == ("first", "second")
    assert budget.snapshot()["subagent_tokens"]["a"] == 12
    assert first_run.tokens + second_run.tokens == 12
    assert len(model.calls[1]["messages"]) == 2  # только system и новая задача, прошлый диалог не тянется
    assert budget.snapshot()["subagent_steps"]["a"] == 2


def test_invalid_assignment_does_not_consume_budget():
    pool, budget = make_pool({"small": ScriptedChatModel([])})
    pool.create("c0", "a", "retriever", "small")
    ((observation, run),) = asyncio.run(pool.assign_many([("c1", "a", "   ")]))
    assert observation.is_error
    assert run is None
    assert budget.snapshot()["subagent_tokens"]["a"] == 0


def test_pool_needs_at_least_one_model_alias():
    with pytest.raises(ValueError, match="model alias"):
        make_pool({})


# ----- страж контекста -------------------------------------------------------


def prompt_turn(prompt_tokens, *, output_tokens=5, calls=()):
    """Ход с размером промпта от провайдера — его и читает страж контекста."""
    return ModelTurn(
        text="",
        output_tokens=output_tokens,
        finish_reason="tool_calls" if calls else "stop",
        tool_calls=tuple(calls),
        prompt_tokens=prompt_tokens,
    )


def test_context_guard_stops_the_subagent_before_its_prompt_would_overflow():
    budget = TaskBudget(TaskLimits(subagent_tokens_per_step=1024))
    budget.register_subagent("a")
    model = ScriptedChatModel([prompt_turn(100, calls=(search("c1"),)), turn("never reached", 3)])
    text, run_record = asyncio.run(
        run_subagent(
            name="a",
            role="retriever",
            model_alias="small",
            client=model,
            system_prompt="p",
            task="find",
            budget=budget,
            retriever=StaticRetriever(HITS),
            top_k=5,
            return_max_chars=500,
            context_limit=100,
        )
    )
    assert len(model.calls) == 1
    # Первый запрос ограничен остатком окна, а не бронью шага (1024).
    assert model.calls[0]["max_tokens"] == 99
    assert text.startswith("[budget exhausted: context]")
    assert (run_record.finished, run_record.stop_reason, run_record.steps) == (False, "context", 1)
    assert budget.snapshot()["overshoot_tokens"] == 0


def test_provider_context_length_error_is_a_context_stop_not_an_infra_failure():
    budget = TaskBudget(TaskLimits())
    budget.register_subagent("a")
    model = ScriptedChatModel(
        [ModelRequestError("model request failed: BadRequestError: maximum context length is 8192 tokens")]
    )
    text, run_record = asyncio.run(
        run_subagent(
            name="a",
            role="retriever",
            model_alias="small",
            client=model,
            system_prompt="p",
            task="find",
            budget=budget,
            retriever=StaticRetriever(HITS),
            top_k=5,
            return_max_chars=500,
        )
    )
    assert text.startswith("[budget exhausted: context]")
    assert (run_record.stop_reason, run_record.infra_failure) == ("context", False)
    assert budget.snapshot()["subagent_tokens"]["a"] == 0


def test_context_meter_uses_provider_counts_plus_new_characters():
    from orchestrator_research.subagents import ContextMeter

    meter = ContextMeter()
    history = [{"role": "system", "content": "s" * 30}, {"role": "user", "content": "u" * 30}]
    assert meter.estimate(history) == 20  # счётчика от провайдера ещё нет: 60 символов // 3
    history.append({"role": "assistant", "content": "a" * 9})
    meter.record(prompt_turn(500, output_tokens=7), len(history))
    history.append({"role": "tool", "content": "t" * 300})
    assert meter.estimate(history) == 500 + 7 + 100
