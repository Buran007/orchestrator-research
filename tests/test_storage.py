"""Хранилище результатов: JSONL с дописыванием, восстановление оборванного хвоста, что считается готовым."""

import json

import pytest

from orchestrator_research.domain import EpisodeResult, EpisodeStatus, GradeResult, GradeStatus, Stage, SubagentRun
from orchestrator_research.storage import (
    JsonlResultStore,
    atomic_write_json,
    canonical_json,
    collect_run_rows,
    is_usable,
    read_jsonl_readonly,
    write_immutable_json,
)


def document(question_id="q1", *, grade="graded", correct=True, episode_status="completed", infra_errors=0, **extra):
    """Документ эпизода в том виде, в каком его пишет ``assembly.episode_document``, уже как JSON-значения."""
    return {
        "schema_version": 1,
        "question_id": question_id,
        "split": extra.pop("split", "val_test"),
        "step": extra.pop("step", -1),
        "reward": {"r_perf": 1.0 if correct else 0.0, "total": 1.0 if correct else 0.0},
        "grade": {"status": grade, "correct": correct},
        "episode": {
            "question_id": question_id,
            "status": episode_status,
            "infra_errors": infra_errors,
            "tool_errors": 0,
            "stop_reason": "answered",
            "stages": [],
            "events": [],
            "budget": {},
        },
        **extra,
    }


def test_appends_one_json_line_per_document_and_reads_them_back(tmp_path):
    store = JsonlResultStore(tmp_path / "nested" / "val_test.jsonl")
    store.append(document("q1"))
    store.append(document("q2"))

    lines = store.path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["question_id"] for line in lines] == ["q1", "q2"]
    assert [record["question_id"] for record in store.records()] == ["q1", "q2"]


def test_episode_round_trip_keeps_stages_budget_stop_reason_and_strategy(tmp_path):
    run = SubagentRun(
        name="checker",
        role="fact_checker",
        model_alias="small",
        steps=2,
        finished=True,
        stop_reason="answered",
        tokens=7,
        search_calls=1,
    )
    episode = EpisodeResult(
        question_id="q9",
        final_answer="Paris",
        status=EpisodeStatus.COMPLETED,
        stop_reason="turns",
        error=None,
        strategy="swarm",
        elapsed_seconds=0.25,
        events=(),
        stages=(Stage(turn_index=0, subagent_runs=(run,)), Stage(turn_index=1)),
        budget={"orchestrator_tokens": 25, "subagent_tokens": {"checker": 7}, "overshoot_tokens": 0},
        tool_errors=0,
        infra_errors=0,
    )
    store = JsonlResultStore(tmp_path / "x.jsonl")
    store.append(
        {"schema_version": 1, "question_id": "q9", "episode": episode, "grade": GradeResult(GradeStatus.GRADED, True)}
    )

    stored = store.records()[0]["episode"]
    assert stored["stages"][0]["subagent_runs"][0]["name"] == "checker"
    assert stored["stop_reason"] == "turns"
    assert stored["strategy"] == "swarm"
    assert stored["budget"]["subagent_tokens"] == {"checker": 7}
    assert json.loads(canonical_json(episode))["stages"] == stored["stages"]


def test_recovers_only_a_torn_final_line(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_bytes(b'{"schema_version":1,"question_id":"q1"}\n{"broken":')
    store = JsonlResultStore(path)

    store.recover_tail()

    assert [row["question_id"] for row in store.records()] == ["q1"]


def test_rejects_corruption_in_a_complete_line(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_bytes(b'{"broken":}\n')
    with pytest.raises(ValueError, match="line 1"):
        JsonlResultStore(path).recover_tail()


@pytest.mark.parametrize(
    ("record", "usable"),
    [
        (document(grade="graded", correct=True), True),
        (document(grade="skipped_no_answer", correct=None), True),  # отсутствие ответа — тоже результат
        (document(grade="judge_transport_error", correct=None), False),
        (document(grade="judge_parse_error", correct=None), False),
        (document(episode_status="model_error"), False),
        (document(infra_errors=1), False),
    ],
    ids=["graded", "no-answer", "transport", "parse", "model-error", "infra-error"],
)
def test_is_usable_separates_failures_from_answers(record, usable):
    assert is_usable(record) is usable


def test_model_and_tool_mistakes_do_not_force_a_retry(tmp_path):
    record = document()
    record["episode"]["tool_errors"] = 3  # отклонённые вызовы — ошибка модели, не среды
    store = JsonlResultStore(tmp_path / "x.jsonl")
    store.append(record)

    assert store.completed_ids() == {"q1"}


def test_judge_error_is_retried_then_superseded_by_the_graded_attempt(tmp_path):
    store = JsonlResultStore(tmp_path / "samples.jsonl")
    store.append(document("q1", grade="judge_transport_error", correct=None))
    assert store.completed_ids() == set()

    store.append(document("q1"))
    assert store.completed_ids() == {"q1"}
    assert len(store.records()) == 2


def test_write_immutable_json_accepts_same_content_and_refuses_changes(tmp_path):
    path = tmp_path / "eval" / "config.json"
    write_immutable_json(path, {"config_hash": "a", "config": {"x": 1}})
    write_immutable_json(path, {"config": {"x": 1}, "config_hash": "a"})

    with pytest.raises(ValueError, match="different configuration"):
        write_immutable_json(path, {"config_hash": "b", "config": {"x": 2}})


def test_collect_run_rows_keeps_every_training_rollout_and_the_latest_eval_attempt(tmp_path):
    atomic_write_json(
        tmp_path / "episodes" / "train" / "0" / "r1-a1.json",
        {**document("q1", split="train", step=0), "episode": {"status": "completed"}},
    )
    atomic_write_json(
        tmp_path / "episodes" / "train" / "0" / "r1-a2.json",
        {**document("q1", split="train", step=0), "episode": {"status": "completed"}},
    )
    eval_path = tmp_path / "eval" / "val_test.jsonl"
    store = JsonlResultStore(eval_path)
    store.append(document("q2", grade="judge_transport_error", correct=None))
    store.append(document("q2"))

    rows = collect_run_rows(tmp_path)

    assert [row["source"] for row in rows] == ["train", "train", "eval"]
    assert (rows[2]["split"], rows[2]["step"], rows[2]["grade"]["status"]) == ("val_test", -1, "graded")


def test_collect_run_rows_never_repairs_a_torn_eval_file(tmp_path):
    eval_path = tmp_path / "eval" / "val_test.jsonl"
    JsonlResultStore(eval_path).append(document("q1"))
    with eval_path.open("ab") as stream:
        stream.write(b'{"torn":')  # обрыв при записи; читатель не должен его чинить
    size_before = eval_path.stat().st_size

    rows = collect_run_rows(tmp_path)

    assert size_before == eval_path.stat().st_size
    assert len(read_jsonl_readonly(eval_path)) == 1
    assert [row["episode"]["question_id"] for row in rows] == ["q1"]
