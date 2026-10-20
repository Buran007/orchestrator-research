"""Метрики по сохранённым эпизодам и парное сравнение двух прогонов.

Строки — из ``storage.collect_run_rows``. Все счётчики выводятся из самого эпизода (события, стадии, бюджет),
поэтому в файле эпизода ничего производного не хранится.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .domain import EventSource, GradeStatus
from .storage import latest_records
from .tools import ASSIGN_TASK_TOOL, SEARCH_TOOL_NAME, executed

_JUDGE_FAILURES = frozenset({GradeStatus.TRANSPORT_ERROR.value, GradeStatus.PARSE_ERROR.value})
_BUDGET_STOPS = frozenset({"orchestrator_budget", "total_budget", "turns"})
_EVALUABLE = frozenset({GradeStatus.GRADED.value, GradeStatus.SKIPPED_NO_ANSWER.value})


def summarize_groups(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    """Метрики на каждую пару ``(split, step)``."""
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["split"]), int(row["step"])), []).append(row)
    return {key: summarize_group(groups[key]) for key in sorted(groups)}


def summarize_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Точность, награда, токены, причины остановки, доля завершённых субагентов, сбои.

    Точность — среднее ``r_perf`` по оцениваемым строкам. Сбой судьи — не ответ: в среднее не входит,
    считается в ``judge_failures``. Отсутствие ответа оценивается как неверный ответ.
    """
    episodes = [row["episode"] for row in rows]
    runs = [
        run for episode in episodes for stage in episode.get("stages", ()) for run in stage.get("subagent_runs", ())
    ]
    created = len(runs)
    finished = sum(bool(run.get("finished")) for run in runs)
    orchestrator_tokens = [int(episode["budget"].get("orchestrator_tokens", 0)) for episode in episodes]
    subagent_tokens = [sum(episode["budget"].get("subagent_tokens", {}).values()) for episode in episodes]
    rewards = [float(row["reward"]["total"]) for row in rows if "total" in row.get("reward", {})]
    evaluable = [row for row in rows if row["grade"].get("status") in _EVALUABLE]
    return {
        "n": len(rows),
        "evaluable": len(evaluable),
        "judge_failures": sum(row["grade"].get("status") in _JUDGE_FAILURES for row in rows),
        "accuracy": _mean([_r_perf(row) for row in evaluable]),
        "reward_total_mean": _mean(rewards),
        "orchestrator_tokens_mean": _mean(orchestrator_tokens),
        "subagent_tokens_mean": _mean(subagent_tokens),
        "total_tokens_mean": _mean([o + s for o, s in zip(orchestrator_tokens, subagent_tokens, strict=True)]),
        "stop_reasons": dict(Counter(str(episode.get("stop_reason") or "none") for episode in episodes)),
        "subagents_created": created,
        "subagents_finished": finished,
        "subagent_finish_rate": _ratio(finished, created),
        "infra_failures": sum(_is_infra_failure(row) for row in rows),
        "search_calls_mean": _mean([_episode_counts(episode)["search_calls"] for episode in episodes]),
        "subagent_calls_mean": _mean([_episode_counts(episode)["subagent_calls"] for episode in episodes]),
        "budget_exhausted": sum(episode.get("stop_reason") in _BUDGET_STOPS for episode in episodes),
    }


def format_report(groups: Mapping[tuple[str, int], Mapping[str, Any]]) -> str:
    """Текстовая таблица: строка на split и step."""
    header = (
        f"{'split':<10} {'step':>5} {'n':>5} {'acc':>7} {'judge_err':>9} {'reward':>8} {'orch_tok':>9} "
        f"{'sub_tok':>9} {'sub fin/made':>13} {'infra':>6}  stop_reasons"
    )
    lines = [header, "-" * len(header)]
    for (split, step), metrics in groups.items():
        finished = f"{metrics['subagents_finished']}/{metrics['subagents_created']}"
        stops = ", ".join(f"{name}={count}" for name, count in sorted(metrics["stop_reasons"].items()))
        lines.append(
            f"{split:<10} {step:>5} {metrics['n']:>5} {_fmt(metrics['accuracy']):>7} "
            f"{metrics['judge_failures']:>9} {_fmt(metrics['reward_total_mean']):>8} "
            f"{_fmt(metrics['orchestrator_tokens_mean']):>9} {_fmt(metrics['subagent_tokens_mean']):>9} "
            f"{finished:>13} {metrics['infra_failures']:>6}  {stops}"
        )
    if not groups:
        lines.append("(no episodes found)")
    return "\n".join(lines)


def paired_comparison(
    baseline_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    seed: int = 0,
    bootstrap_samples: int = 5000,
) -> dict[str, Any]:
    """Сравнить точность на общих вопросах: точный тест Макнемара и бутстрэп-интервал."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be > 0")
    baseline = _benchmark_outcomes(baseline_records)
    candidate = _benchmark_outcomes(candidate_records)
    if set(baseline) != set(candidate):
        missing_candidate = sorted(set(baseline) - set(candidate))
        missing_baseline = sorted(set(candidate) - set(baseline))
        raise ValueError(
            "paired comparison requires identical successfully graded question IDs; "
            f"missing candidate={missing_candidate}, missing baseline={missing_baseline}"
        )
    ids = sorted(baseline)
    if not ids:
        raise ValueError("no paired graded questions available")
    deltas = [int(candidate[qid]) - int(baseline[qid]) for qid in ids]
    baseline_only = sum(delta == -1 for delta in deltas)
    candidate_only = sum(delta == 1 for delta in deltas)
    point = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    boot_means = []
    for _ in range(bootstrap_samples):
        boot_means.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
    boot_means.sort()
    return {
        "paired_question_count": len(ids),
        "baseline_accuracy": sum(baseline.values()) / len(ids),
        "candidate_accuracy": sum(candidate.values()) / len(ids),
        "accuracy_delta": point,
        "accuracy_delta_ci95": [_percentile(boot_means, 0.025), _percentile(boot_means, 0.975)],
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "mcnemar_exact_p_two_sided": _mcnemar_exact(baseline_only, candidate_only),
    }


def _episode_counts(episode: Mapping[str, Any]) -> dict[str, int]:
    """Реально выполненные вызовы (без отклонённых и сверх лимита) и расход субагентов."""
    observed = [event for event in episode.get("events", ()) if event["source"] == EventSource.OBSERVATION.value]
    ran = [
        event
        for event in observed
        if executed(bool(event["is_error"]), bool(event["metadata"].get("infra_error", False)))
    ]
    return {
        "search_calls": sum(event["tool_name"] == SEARCH_TOOL_NAME for event in ran),
        "subagent_calls": sum(event["tool_name"] == ASSIGN_TASK_TOOL for event in ran),
    }


def _benchmark_outcomes(records: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    latest = latest_records([dict(record) for record in records])
    outcomes: dict[str, bool] = {}
    failed: list[str] = []
    for question_id, row in latest.items():
        grade = row.get("grade", {})
        status = grade.get("status")
        if status == GradeStatus.GRADED.value:
            outcomes[question_id] = bool(grade.get("correct"))
        elif status == GradeStatus.SKIPPED_NO_ANSWER.value:
            outcomes[question_id] = False
        else:
            failed.append(question_id)
    if failed:
        raise ValueError(
            f"paired comparison requires resolved grades for every question; judge failures={sorted(failed)}"
        )
    return outcomes


def _r_perf(row: Mapping[str, Any]) -> float:
    reward = row.get("reward", {})
    if "r_perf" in reward:
        return float(reward["r_perf"])
    grade = row["grade"]
    return 1.0 if grade.get("status") == GradeStatus.GRADED.value and grade.get("correct") is True else 0.0


def _is_infra_failure(row: Mapping[str, Any]) -> bool:
    episode, grade = row["episode"], row["grade"]
    if episode.get("status") == "model_error" or int(episode.get("infra_errors", 0)) > 0:
        return True
    if grade.get("status") in _JUDGE_FAILURES:
        return True
    return any(
        run.get("infra_failure") for stage in episode.get("stages", ()) for run in stage.get("subagent_runs", ())
    )


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}" if abs(value) < 1000 else f"{value:.0f}"


def _mcnemar_exact(left: int, right: int) -> float:
    discordant = left + right
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(left, right) + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(sorted_values: Sequence[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    index = max(0, min(len(sorted_values) - 1, math.ceil(percentile * len(sorted_values)) - 1))
    return float(sorted_values[index])
