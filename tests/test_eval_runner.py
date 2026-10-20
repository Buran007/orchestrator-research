"""Прогон оценки: одна запись на вопрос, возобновление, повтор только после сбоя судьи, неизменность конфига."""

import asyncio
import json

import pytest
from conftest import FakeJudge

from orchestrator_research import assembly
from orchestrator_research.adapters.fake import ScriptedChatModel
from orchestrator_research.domain import ModelTurn, Question
from orchestrator_research.eval_runner import eval_rows_for_split, question_from_row, run_eval
from orchestrator_research.storage import collect_run_rows

QUESTIONS = [Question("q1", "Who wrote it?", ("Ada",)), Question("q2", "Which city?", ("Paris",))]


def _answers(count: int) -> list[ModelTurn]:
    return [ModelTurn("done", 4, "stop", final_answer="Ada") for _ in range(count)]


@pytest.fixture
def judge() -> FakeJudge:
    return FakeJudge()


@pytest.fixture
def orchestrator() -> ScriptedChatModel:
    return ScriptedChatModel(_answers(4))


@pytest.fixture(autouse=True)
def _network(wire, orchestrator, judge):
    """Настоящий код эпизода и награды поверх фейков; тест может перепривязать их своим вызовом ``wire``."""
    wire(orchestrator=orchestrator, judge=judge)


def _run(cfg, tmp_path, **kwargs):
    options = {"split": "val_test", "run_id": "r1", "artifact_root": tmp_path, "questions": QUESTIONS, "concurrency": 2}
    options.update(kwargs)
    return asyncio.run(run_eval(cfg, **options))


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_writes_one_validation_record_per_question(experiment_config, tmp_path):
    summary = _run(experiment_config, tmp_path)

    assert (summary.evaluated, summary.resumed, summary.errors) == (2, 0, ())
    assert summary.path == tmp_path / "runs" / experiment_config.experiment.name / "r1" / "eval" / "val_test.jsonl"
    rows = _lines(summary.path)
    assert sorted(row["question_id"] for row in rows) == ["q1", "q2"]
    for row in rows:
        assert (row["split"], row["step"], row["is_validation"]) == ("val_test", -1, True)
        assert row["experiment"] == experiment_config.experiment.name
        assert row["reward"]["r_perf"] == 1.0
        assert row["reward"]["total"] == 1.0  # на валидации total — только исход задачи
    config_file = json.loads((summary.run_dir / "eval" / "config.json").read_text(encoding="utf-8"))
    assert config_file["config_hash"] == experiment_config.config_hash()


def test_resume_skips_questions_that_already_have_an_answer(experiment_config, judge, tmp_path):
    _run(experiment_config, tmp_path)
    summary = _run(experiment_config, tmp_path)

    assert (summary.evaluated, summary.resumed, summary.errors) == (0, 2, ())
    assert judge.calls == ["q1", "q2"]  # второй прогон к судье не обращался


def test_judge_transport_failure_is_retried_on_resume_only_for_that_question(experiment_config, wire, tmp_path):
    judge = FakeJudge(transport_fail=("q1",))
    orchestrator = ScriptedChatModel(_answers(3))
    wire(orchestrator=orchestrator, judge=judge)

    first = _run(experiment_config, tmp_path)
    second = _run(experiment_config, tmp_path)

    assert (first.evaluated, first.resumed) == (2, 0)
    assert (second.evaluated, second.resumed) == (1, 1)
    assert judge.calls == ["q1", "q2", "q1"]


def test_changed_config_under_the_same_run_id_is_refused(experiment_config, tmp_path):
    _run(experiment_config, tmp_path)
    changed = experiment_config.model_copy(
        update={"experiment": experiment_config.experiment.model_copy(update={"strategy": "changed"})}
    )

    with pytest.raises(ValueError, match="different configuration"):
        _run(changed, tmp_path)


def test_max_questions_limits_the_selection(experiment_config, judge, tmp_path):
    summary = _run(experiment_config, tmp_path, max_questions=1)

    assert summary.evaluated == 1
    assert judge.calls == ["q1"]


def test_a_crashed_episode_is_reported_and_left_for_the_next_resume(experiment_config, monkeypatch, tmp_path):
    original = assembly.run_episode

    async def crash_on_q2(cfg, question, deps, **kwargs):
        if question.question_id == "q2":
            raise RuntimeError("boom")
        return await original(cfg, question, deps, **kwargs)

    monkeypatch.setattr(assembly, "run_episode", crash_on_q2)
    summary = _run(experiment_config, tmp_path)

    assert summary.evaluated == 1
    assert len(summary.errors) == 1 and summary.errors[0].startswith("q2: RuntimeError")
    recorded = [row["episode"]["question_id"] for row in collect_run_rows(summary.run_dir) if row["source"] == "eval"]
    assert recorded == ["q1"]


def test_duplicate_question_ids_are_rejected(experiment_config, tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        _run(experiment_config, tmp_path, questions=[QUESTIONS[0], QUESTIONS[0]])


def test_eval_rows_for_split_selects_only_that_split(experiment_config, tmp_path):
    summary = _run(experiment_config, tmp_path)

    rows = eval_rows_for_split(collect_run_rows(summary.run_dir), "val_test")
    assert len(rows) == 2
    assert eval_rows_for_split(collect_run_rows(summary.run_dir), "train") == []


def test_question_from_row_maps_the_prepared_schema():
    row = {"question_id": 17, "question": "Q?", "answers": ["A"], "split": "val_test", "train_step": -1}
    assert question_from_row(row) == Question("17", "Q?", ("A",))
