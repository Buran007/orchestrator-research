"""Инварианты ``TaskLimits`` и ``TaskBudget`` из docstring ``limits.py``: формулы брони, причины остановки,
неприкосновенность ``final_answer_reserve``, отсутствие перерасхода при параллельных бронях.
"""

import random

import pytest

from orchestrator_research.limits import StopReason, TaskBudget, TaskLimits

# ----- валидация TaskLimits --------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
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
    ],
)
def test_positive_fields_reject_zero_and_negative(field):
    for bad in (0, -1):
        with pytest.raises(ValueError, match=field):
            TaskLimits(**{field: bad})


@pytest.mark.parametrize("field", ["orchestrator_tokens_per_task", "total_tokens_per_task"])
def test_task_limit_must_exceed_final_answer_reserve(field):
    with pytest.raises(ValueError, match=field):
        TaskLimits(**{field: 512, "final_answer_reserve": 512})
    TaskLimits(**{field: 513, "final_answer_reserve": 512})


def test_final_answer_reserve_may_be_zero_but_not_negative():
    assert TaskLimits(final_answer_reserve=0).final_answer_reserve == 0
    with pytest.raises(ValueError, match="final_answer_reserve"):
        TaskLimits(final_answer_reserve=-1)


# ----- оркестратор -----------------------------------------------------------


def test_per_turn_cap_applies_when_unlimited():
    budget = TaskBudget(TaskLimits(orchestrator_tokens_per_turn=100, final_answer_reserve=10))
    assert budget.reserve_orchestrator_turn() == 100
    budget.commit_orchestrator(100, 100)
    assert budget.reserve_orchestrator_turn() == 100


def test_max_turns_stops_orchestrator():
    budget = TaskBudget(TaskLimits(max_turns=2, orchestrator_tokens_per_turn=10))
    for _ in range(2):
        reserved = budget.reserve_orchestrator_turn()
        assert reserved == 10
        budget.commit_orchestrator(reserved, 10)
    assert budget.reserve_orchestrator_turn() == 0
    assert budget.orchestrator_stop_reason() is StopReason.TURNS


def test_orchestrator_budget_exhaustion_leaves_exactly_the_final_reserve():
    limits = TaskLimits(
        orchestrator_tokens_per_task=1000,
        orchestrator_tokens_per_turn=300,
        final_answer_reserve=200,
        max_turns=10,
    )
    budget = TaskBudget(limits)
    amounts = []
    for _ in range(3):
        reserved = budget.reserve_orchestrator_turn()
        amounts.append(reserved)
        budget.commit_orchestrator(reserved, reserved)
    assert amounts == [300, 300, 200]
    assert budget.reserve_orchestrator_turn() == 0
    assert budget.orchestrator_stop_reason() is StopReason.ORCHESTRATOR_BUDGET

    final = budget.reserve_final_answer()
    assert final == 200
    budget.commit_orchestrator(final, final)
    assert budget.orchestrator_tokens == 1000
    assert budget.snapshot()["total_tokens"] == 1000


def test_final_answer_is_not_counted_as_an_orchestrator_turn():
    budget = TaskBudget(TaskLimits(orchestrator_tokens_per_task=1000, orchestrator_tokens_per_turn=300, max_turns=5))
    for _ in range(2):
        reserved = budget.reserve_orchestrator_turn()
        budget.commit_orchestrator(reserved, reserved)
    final = budget.reserve_final_answer()
    budget.commit_orchestrator(final, final)
    assert budget.orchestrator_turns == 2


def test_final_answer_with_zero_reserve_uses_per_turn_cap():
    limits = TaskLimits(orchestrator_tokens_per_task=150, orchestrator_tokens_per_turn=100, final_answer_reserve=0)
    budget = TaskBudget(limits)
    reserved = budget.reserve_orchestrator_turn()
    assert reserved == 100
    budget.commit_orchestrator(reserved, 100)
    assert budget.reserve_final_answer() == 50


# ----- субагенты -------------------------------------------------------------


def test_subagents_never_consume_total_final_answer_reserve():
    limits = TaskLimits(
        total_tokens_per_task=1000,
        final_answer_reserve=200,
        orchestrator_tokens_per_turn=300,
        subagent_tokens_per_step=300,
        subagent_max_steps=10,
    )
    budget = TaskBudget(limits)
    budget.register_subagent("a")
    budget.register_subagent("b")
    amounts = []
    for _ in range(3):
        reserved = budget.reserve_subagent_step("a")
        amounts.append(reserved)
        budget.commit_subagent("a", reserved, reserved)
    assert amounts == [300, 300, 200]
    assert budget.total_tokens == 800
    assert budget.reserve_subagent_step("a") == 0
    assert budget.subagent_stop_reason("a") is StopReason.TOTAL_BUDGET
    assert budget.reserve_subagent_step("b") == 0
    assert budget.orchestrator_stop_reason() is StopReason.TOTAL_BUDGET
    assert budget.reserve_final_answer() == 200


def test_exhausting_one_subagent_does_not_affect_another():
    budget = TaskBudget(TaskLimits(subagent_tokens_per_task=500, subagent_tokens_per_step=200))
    budget.register_subagent("a")
    budget.register_subagent("b")
    for expected in (200, 200, 100):
        reserved = budget.reserve_subagent_step("a")
        assert reserved == expected
        budget.commit_subagent("a", reserved, reserved)
    assert budget.reserve_subagent_step("a") == 0
    assert budget.subagent_stop_reason("a") is StopReason.SUBAGENT_BUDGET
    assert budget.snapshot()["subagent_tokens"]["a"] == 500
    assert budget.subagent_stop_reason("b") is None
    assert budget.reserve_subagent_step("b") == 200


def test_subagent_max_steps_stop():
    budget = TaskBudget(TaskLimits(subagent_max_steps=2, subagent_tokens_per_step=10))
    budget.register_subagent("a")
    budget.register_subagent("b")
    for _ in range(2):
        reserved = budget.reserve_subagent_step("a")
        budget.commit_subagent("a", reserved, reserved)
    assert budget.reserve_subagent_step("a") == 0
    assert budget.subagent_stop_reason("a") is StopReason.SUBAGENT_STEPS
    assert budget.reserve_subagent_step("b") == 10


def test_max_subagents_cap_and_duplicate_name():
    budget = TaskBudget(TaskLimits(max_subagents_per_task=2))
    budget.register_subagent("a")
    with pytest.raises(ValueError, match="duplicate"):
        budget.register_subagent("a")
    budget.register_subagent("b")
    assert not budget.can_create_subagent()
    with pytest.raises(ValueError, match="max_subagents_per_task"):
        budget.register_subagent("c")
    assert tuple(budget.snapshot()["subagent_tokens"]) == ("a", "b")


def test_unknown_subagent_raises_key_error():
    budget = TaskBudget(TaskLimits())
    with pytest.raises(KeyError):
        budget.reserve_subagent_step("ghost")
    with pytest.raises(KeyError):
        budget.commit_subagent("ghost", 0, 0)


def test_parallel_reservations_never_exceed_total():
    limits = TaskLimits(
        total_tokens_per_task=1000,
        final_answer_reserve=100,
        orchestrator_tokens_per_turn=300,
        subagent_tokens_per_step=200,
        max_subagents_per_task=5,
    )
    budget = TaskBudget(limits)
    names = [f"s{i}" for i in range(5)]
    for name in names:
        budget.register_subagent(name)
    reservations = [budget.reserve_subagent_step(name) for name in names]
    assert sum(reservations) <= 1000 - 100
    assert reservations == [200, 200, 200, 200, 100]

    final = budget.reserve_final_answer()
    assert sum(reservations) + final <= 1000
    assert final == 100
    assert budget.reserve_orchestrator_turn() == 0
    assert budget.orchestrator_stop_reason() is StopReason.TOTAL_BUDGET

    for name, reserved in zip(names, reservations, strict=True):
        budget.commit_subagent(name, reserved, reserved)
    budget.commit_orchestrator(final, final)
    assert budget.snapshot()["total_tokens"] == 1000


# ----- учёт ------------------------------------------------------------------


def test_overshoot_is_recorded_without_exception():
    budget = TaskBudget(TaskLimits(orchestrator_tokens_per_turn=100, subagent_tokens_per_step=100))
    budget.register_subagent("a")
    reserved = budget.reserve_orchestrator_turn()
    budget.commit_orchestrator(reserved, 103)
    assert budget.snapshot()["overshoot_tokens"] == 3
    assert budget.orchestrator_tokens == 103

    reserved = budget.reserve_orchestrator_turn()
    budget.commit_orchestrator(reserved, 50)
    assert budget.snapshot()["overshoot_tokens"] == 3

    reserved = budget.reserve_subagent_step("a")
    budget.commit_subagent("a", reserved, 110)
    assert budget.snapshot()["overshoot_tokens"] == 13
    assert budget.total_tokens == 103 + 50 + 110


@pytest.mark.parametrize(("reserved", "actual"), [(-1, 0), (10, -1), (5, 0)])
def test_commit_rejects_negative_or_unreserved_amounts(reserved, actual):
    budget = TaskBudget(TaskLimits())
    with pytest.raises(ValueError):
        budget.commit_orchestrator(reserved, actual)


def test_snapshot_reports_ledger():
    budget = TaskBudget(TaskLimits(orchestrator_tokens_per_turn=10, subagent_tokens_per_step=5))
    budget.register_subagent("a")
    reserved = budget.reserve_orchestrator_turn()
    budget.commit_orchestrator(reserved, 7)
    reserved = budget.reserve_subagent_step("a")
    budget.commit_subagent("a", reserved, 4)
    assert budget.snapshot() == {
        "orchestrator_tokens": 7,
        "orchestrator_turns": 1,
        "subagent_tokens": {"a": 4},
        "subagent_steps": {"a": 1},
        "subagents_created": 1,
        "total_tokens": 11,
        "overshoot_tokens": 0,
    }


def test_all_none_limits_only_apply_per_turn_and_step_caps():
    limits = TaskLimits(
        orchestrator_tokens_per_task=None,
        subagent_tokens_per_task=None,
        total_tokens_per_task=None,
        orchestrator_tokens_per_turn=50,
        subagent_tokens_per_step=40,
        final_answer_reserve=512,
        max_turns=100,
        subagent_max_steps=100,
    )
    budget = TaskBudget(limits)
    budget.register_subagent("a")
    for _ in range(100):
        reserved = budget.reserve_orchestrator_turn()
        assert reserved == 50
        budget.commit_orchestrator(reserved, reserved)
        reserved = budget.reserve_subagent_step("a")
        assert reserved == 40
        budget.commit_subagent("a", reserved, reserved)
    assert budget.reserve_orchestrator_turn() == 0
    assert budget.orchestrator_stop_reason() is StopReason.TURNS
    assert budget.reserve_final_answer() == 512


# ----- случайный property-тест -----------------------------------------------


def _assert_invariants(budget, limits, pending_total):
    """Инварианты ``TaskBudget``; ``pending_total`` — открытые брони, они тоже не должны пробить общий лимит."""
    snap = budget.snapshot()
    sub_tokens = snap["subagent_tokens"]
    assert budget.total_tokens == budget.orchestrator_tokens + sum(sub_tokens.values())
    assert snap["overshoot_tokens"] == 0
    assert budget.orchestrator_turns <= limits.max_turns
    assert len(sub_tokens) <= limits.max_subagents_per_task
    if limits.orchestrator_tokens_per_task is not None:
        assert budget.orchestrator_tokens <= limits.orchestrator_tokens_per_task
    if limits.subagent_tokens_per_task is not None:
        assert all(tokens <= limits.subagent_tokens_per_task for tokens in sub_tokens.values())
    if limits.total_tokens_per_task is not None:
        assert budget.total_tokens <= limits.total_tokens_per_task
        assert budget.total_tokens + pending_total <= limits.total_tokens_per_task


@pytest.mark.parametrize("seed", range(50))
def test_random_interleaving_preserves_invariants(seed):
    """Случайное чередование броней и коммитов; ``actual <= reserved``, у владельца не больше одной открытой брони."""
    rng = random.Random(seed)
    reserve = rng.randint(0, 200)
    limits = TaskLimits(
        orchestrator_tokens_per_task=rng.choice([None, reserve + rng.randint(1, 800)]),
        orchestrator_tokens_per_turn=rng.randint(1, 300),
        subagent_tokens_per_task=rng.choice([None, rng.randint(1, 400)]),
        subagent_tokens_per_step=rng.randint(1, 200),
        total_tokens_per_task=rng.choice([None, reserve + rng.randint(1, 2000)]),
        final_answer_reserve=reserve,
        max_turns=rng.randint(1, 30),
        subagent_max_steps=rng.randint(1, 10),
        max_subagents_per_task=rng.randint(1, 5),
    )
    budget = TaskBudget(limits)
    pending: list[tuple[str | None, int]] = []  # (имя субагента или None для оркестратора, reserved)

    def pending_total():
        return sum(r for _, r in pending)

    def busy(owner):
        return any(o == owner for o, _ in pending)

    for _ in range(300):
        roll = rng.random()
        if pending and roll < 0.5:
            owner, reserved = pending.pop(rng.randrange(len(pending)))
            actual = rng.randint(0, reserved)
            if owner is None:
                budget.commit_orchestrator(reserved, actual)
            else:
                budget.commit_subagent(owner, reserved, actual)
        elif roll < 0.65:
            if not busy(None):
                stop = budget.orchestrator_stop_reason()
                reserved = budget.reserve_orchestrator_turn()
                assert (stop is None) == (reserved > 0)
                assert reserved <= limits.orchestrator_tokens_per_turn
                if reserved:
                    pending.append((None, reserved))
        else:
            names = list(budget.snapshot()["subagent_tokens"])
            if names and (not budget.can_create_subagent() or rng.random() < 0.7):
                name = rng.choice(names)
            elif budget.can_create_subagent():
                name = f"s{len(names)}"
                budget.register_subagent(name)
            else:
                continue
            if not busy(name):
                stop = budget.subagent_stop_reason(name)
                reserved = budget.reserve_subagent_step(name)
                assert (stop is None) == (reserved > 0)
                assert reserved <= limits.subagent_tokens_per_step
                if reserved:
                    pending.append((name, reserved))
        _assert_invariants(budget, limits, pending_total())

    while pending:
        owner, reserved = pending.pop()
        actual = rng.randint(0, reserved)
        if owner is None:
            budget.commit_orchestrator(reserved, actual)
        else:
            budget.commit_subagent(owner, reserved, actual)
    _assert_invariants(budget, limits, 0)

    final = budget.reserve_final_answer()
    remaining = [reserve]
    if limits.total_tokens_per_task is not None:
        remaining.append(limits.total_tokens_per_task - budget.total_tokens)
    if limits.orchestrator_tokens_per_task is not None:
        remaining.append(limits.orchestrator_tokens_per_task - budget.orchestrator_tokens)
    assert final >= min(remaining)
