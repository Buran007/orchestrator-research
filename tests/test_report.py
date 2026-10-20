"""Отчёт по прогону: точность без сбоев судьи, счётчики вызовов и субагентов, парное сравнение двух прогонов."""

import pytest

from orchestrator_research.report import format_report, paired_comparison, summarize_group, summarize_groups


def _observation(tool_name, *, is_error=False, infra_error=False):
    return {
        "source": "observation",
        "tool_name": tool_name,
        "is_error": is_error,
        "metadata": {"infra_error": infra_error},
    }


def _run(*, finished=True, infra=False, tokens=5):
    return {
        "name": "a",
        "role": "fact_checker",
        "model_alias": "small",
        "steps": 1,
        "finished": finished,
        "tokens": tokens,
        "search_calls": 0,
        "infra_failure": infra,
    }


def _episode(*, stop="turns", tokens=10, sub_tokens=0, events=(), stages=(), status="completed", infra_errors=0):
    return {
        "status": status,
        "stop_reason": stop,
        "elapsed_seconds": 1.5,
        "budget": {"orchestrator_tokens": tokens, "subagent_tokens": {"a": sub_tokens} if sub_tokens else {}},
        "events": list(events),
        "stages": list(stages),
        "infra_errors": infra_errors,
    }


def _row(question_id="q1", *, correct, grade=None, split="val_test", step=-1, **episode):
    status = grade or ("graded" if correct is not None else "skipped_no_answer")
    return {
        "source": "eval",
        "question_id": question_id,
        "split": split,
        "step": step,
        "reward": {"r_perf": 1.0 if correct else 0.0, "total": 1.0 if correct else 0.0},
        "grade": {"status": status, "correct": correct},
        "episode": {"question_id": question_id, **_episode(**episode)},
    }


def test_judge_failures_leave_accuracy_and_are_counted_separately():
    group = summarize_group(
        [
            _row("q1", correct=True),
            _row("q2", correct=False),
            _row("q3", correct=None, grade="judge_transport_error"),
            _row("q4", correct=None, grade="skipped_no_answer"),
        ]
    )

    assert group["n"] == 4
    assert group["evaluable"] == 3
    assert group["judge_failures"] == 1
    assert group["accuracy"] == pytest.approx(1 / 3)  # нет ответа — неверно; сбой судьи не учитывается вовсе
    assert group["infra_failures"] == 1


def test_accuracy_is_none_when_nothing_was_evaluable():
    assert summarize_group([])["accuracy"] is None
    assert summarize_group([_row("q1", correct=None, grade="judge_parse_error")])["accuracy"] is None


def test_counters_include_only_calls_that_actually_ran():
    events = [
        _observation("browsecomp_search"),
        _observation("browsecomp_search", is_error=True),  # аргументы отклонены: вызова не было
        _observation("browsecomp_search", is_error=True, infra_error=True),  # ретривер недоступен: вызов был и упал
        _observation("assign_task"),
    ]
    group = summarize_group([_row("q1", correct=True, events=events)])

    assert group["search_calls_mean"] == 2
    assert group["subagent_calls_mean"] == 1


def test_subagent_finish_rate_and_tokens_come_from_stages_and_budget():
    stages = [{"turn_index": 0, "subagent_runs": [_run(finished=True), _run(finished=False)]}]
    group = summarize_group([_row("q1", correct=True, stages=stages, sub_tokens=12, tokens=20)])

    assert group["subagents_created"] == 2
    assert group["subagents_finished"] == 1
    assert group["subagent_finish_rate"] == 0.5
    assert group["subagent_tokens_mean"] == 12
    assert group["total_tokens_mean"] == 32


def test_budget_exhaustion_is_a_budget_stop_not_an_answer_stop():
    group = summarize_group(
        [
            _row("q1", correct=False, stop="orchestrator_budget"),
            _row("q2", correct=True, stop="answered"),
        ]
    )
    assert group["budget_exhausted"] == 1


def test_groups_by_split_and_step_and_formats_one_line_per_group():
    rows = [
        _row("q1", correct=True, split="train", step=0),
        _row("q2", correct=False, split="train", step=0),
        _row("q3", correct=True, split="val_test", step=-1),
    ]
    groups = summarize_groups(rows)

    assert list(groups) == [("train", 0), ("val_test", -1)]
    assert groups[("train", 0)]["n"] == 2
    text = format_report(groups)
    assert text.splitlines()[0].startswith("split")
    assert "val_test" in text and "turns=2" in text and "turns=1" in text
    assert "(no episodes found)" in format_report({})


def test_paired_comparison_uses_the_same_questions():
    baseline = [_row("a", correct=True), _row("b", correct=False), _row("c", correct=False)]
    candidate = [_row("a", correct=True), _row("b", correct=True), _row("c", correct=False)]
    comparison = paired_comparison(baseline, candidate, bootstrap_samples=100, seed=7)

    assert comparison["paired_question_count"] == 3
    assert comparison["accuracy_delta"] == pytest.approx(1 / 3)
    assert comparison["baseline_only_correct"] == 0
    assert comparison["candidate_only_correct"] == 1
    assert comparison["mcnemar_exact_p_two_sided"] == 1.0


def test_latest_retry_supersedes_an_earlier_judge_failure():
    baseline = [_row("q1", correct=None, grade="judge_transport_error"), _row("q1", correct=False)]
    comparison = paired_comparison(baseline, [_row("q1", correct=True)], bootstrap_samples=10)

    assert comparison["accuracy_delta"] == 1.0


def test_no_answer_is_wrong_but_an_unresolved_judge_blocks_the_comparison():
    comparison = paired_comparison(
        [_row("q1", correct=None, grade="skipped_no_answer")], [_row("q1", correct=True)], bootstrap_samples=10
    )
    assert comparison["accuracy_delta"] == 1.0

    with pytest.raises(ValueError, match="judge failures"):
        paired_comparison([_row("q1", correct=None, grade="judge_parse_error")], [_row("q1", correct=True)])


def test_paired_comparison_rejects_different_graded_sets():
    with pytest.raises(ValueError, match="identical"):
        paired_comparison([_row("a", correct=True)], [_row("b", correct=False)])
