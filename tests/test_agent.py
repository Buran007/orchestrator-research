"""``OrchestratorAgent`` в одном agl-роллауте: награда в событии, запись эпизода, отказ судьи, сборка зависимостей.

Сеть подменена фикстурой ``wire``; эпизод, награда и запись на диск — настоящие.
"""

import asyncio
import json

import pytest
from conftest import EXPERIMENT, FakeJudge

from orchestrator_research import agent, assembly
from orchestrator_research.adapters.fake import ScriptedChatModel
from orchestrator_research.domain import GradeResult, GradeStatus, ModelTurn, Question, ToolCall

TRAIN_URL = "http://127.0.0.1:9999/v1/mode/train/rollout/roll-1/attempt/att-2/v1"
VAL_URL = "http://127.0.0.1:9999/v1/mode/val/rollout/roll-9/attempt/att-4/v1"
SECRET = "agl-secret-key-value"
TRAIN_RECORD = "episodes/train/0/roll-1-att-2.json"


def orchestrator_script() -> ScriptedChatModel:
    """Ход 1 — create и assign одного субагента в одном ходе, ход 2 — ответ."""
    delegate = ModelTurn(
        text="",
        output_tokens=20,
        finish_reason="tool_calls",
        tool_calls=(
            ToolCall("c1", "create_subagent", {"name": "alpha", "role": "retriever", "model": "small"}),
            ToolCall("c2", "assign_task", {"agent": "alpha", "prompt": "find the author"}),
        ),
    )
    answer = ModelTurn(text="Ada", output_tokens=5, finish_reason="stop", final_answer="Ada")
    return ScriptedChatModel([delegate, answer])


def subagent_script() -> ScriptedChatModel:
    return ScriptedChatModel(
        [ModelTurn(text="author is Ada", output_tokens=5, finish_reason="stop", final_answer="Ada")]
    )


@pytest.fixture
def rollout(monkeypatch, tmp_path):
    """Выставить окружение, которое получает agl-роллаут; вернуть список отправленных наград."""
    posted: list[tuple[str, str, dict]] = []

    def configure(base_url: str, split: str) -> list[tuple[str, str, dict]]:
        monkeypatch.setenv("ORCH_RESEARCH_CONFIG", str(EXPERIMENT))
        monkeypatch.setenv("ORCH_RESEARCH_RUN_DIR", str(tmp_path / "run"))
        monkeypatch.setenv("QUESTION_ID", "17")
        monkeypatch.setenv("QUESTION", "Who wrote the book?")
        monkeypatch.setenv("ANSWERS", json.dumps(["Ada"]))
        monkeypatch.setenv("SPLIT", split)
        monkeypatch.setenv("TRAIN_STEP", "0")
        monkeypatch.setenv("AGL_OPENAI_BASE_URL", base_url)
        monkeypatch.setenv("AGL_EVENT_URL", "http://127.0.0.1:9999/event")
        monkeypatch.setenv("AGL_KEY", SECRET)
        monkeypatch.setattr(agent, "post_reward", lambda url, key, payload: posted.append((url, key, payload)))
        return posted

    return configure


def _run() -> None:
    asyncio.run(agent.OrchestratorAgent().run())


def _record(tmp_path, relative: str = TRAIN_RECORD) -> dict:
    return json.loads((tmp_path / "run" / relative).read_text(encoding="utf-8"))


def test_train_rollout_posts_reward_with_step_zero_lambdas(rollout, wire, experiment_config, tmp_path):
    posted = rollout(TRAIN_URL, "train")
    wire(orchestrator=orchestrator_script(), judge=FakeJudge(), subagent=subagent_script())

    _run()

    assert len(posted) == 1
    url, key, payload = posted[0]
    assert url == "http://127.0.0.1:9999/event" and key == SECRET
    assert payload["event_type"] == "reward"
    data = payload["data"]
    cfg = experiment_config
    breakdown = data["breakdown"]
    assert breakdown["r_perf"] == 1.0
    assert breakdown["lambda_parallel"] == pytest.approx(cfg.reward.lambda_parallel)
    assert breakdown["lambda_finish"] == pytest.approx(cfg.reward.lambda_finish)
    assert breakdown["r_parallel"] == 1.0  # единственный этап делегировал
    assert breakdown["r_finish"] == 1.0  # единственный субагент закончил сам
    assert data["value"] == pytest.approx(1 + cfg.reward.lambda_parallel + cfg.reward.lambda_finish)
    assert data["split"] == "train"
    assert data["reason"] == "completed/graded"


def test_episode_record_is_written_under_run_dir_without_secrets(rollout, wire, tmp_path):
    posted = rollout(TRAIN_URL, "train")
    wire(orchestrator=orchestrator_script(), judge=FakeJudge(), subagent=subagent_script())

    _run()

    text = (tmp_path / "run" / TRAIN_RECORD).read_text(encoding="utf-8")
    assert SECRET not in text
    record = json.loads(text)
    assert record["question_id"] == "17"
    assert record["split"] == "train"
    assert record["experiment"] == "template"
    assert record["rollout_id"] == "roll-1" and record["attempt_id"] == "att-2"
    assert record["grade"]["correct"] is True
    assert record["reward"]["total"] == pytest.approx(posted[0][2]["data"]["value"])
    assert record["episode"]["budget"]["orchestrator_turns"] == 2
    assert record["episode"]["stages"] and record["episode"]["events"]


def test_validation_rollout_reward_is_task_outcome_only(rollout, wire, tmp_path):
    posted = rollout(VAL_URL, "val_test")
    wire(orchestrator=orchestrator_script(), judge=FakeJudge(), subagent=subagent_script())

    _run()

    data = posted[0][2]["data"]
    assert data["value"] == 1.0
    assert data["breakdown"]["is_validation"] is True
    assert (tmp_path / "run" / "episodes" / "val" / "0" / "roll-9-att-4.json").is_file()


@pytest.mark.parametrize(
    "judge",
    [
        FakeJudge(error=RuntimeError("judge endpoint down")),
        FakeJudge(grade=GradeResult(status=GradeStatus.TRANSPORT_ERROR, correct=None, error="timeout")),
    ],
    ids=["judge-raises", "judge-transport-error"],
)
def test_judge_failure_fails_closed_to_zero_task_reward(rollout, wire, experiment_config, tmp_path, judge):
    posted = rollout(TRAIN_URL, "train")
    wire(orchestrator=orchestrator_script(), judge=judge, subagent=subagent_script())

    _run()

    data = posted[0][2]["data"]
    assert data["breakdown"]["r_perf"] == 0.0
    assert data["value"] == pytest.approx(
        experiment_config.reward.lambda_parallel + experiment_config.reward.lambda_finish
    )
    assert _record(tmp_path)["grade"]["status"] == "judge_transport_error"


def test_judge_exception_is_recorded_with_its_type(rollout, wire, tmp_path):
    rollout(TRAIN_URL, "train")
    wire(orchestrator=orchestrator_script(), judge=FakeJudge(error=RuntimeError("judge endpoint down")))

    _run()

    assert "RuntimeError" in _record(tmp_path)["grade"]["error"]


def test_reward_post_failure_raises(rollout, wire, monkeypatch):
    posted = rollout(TRAIN_URL, "train")
    wire(orchestrator=orchestrator_script(), judge=FakeJudge(), subagent=subagent_script())

    def failing_post(url, key, payload):
        raise OSError("connection refused")

    monkeypatch.setattr(agent, "post_reward", failing_post)
    with pytest.raises(OSError):
        _run()
    assert posted == []


def test_parse_rollout_ids_and_mode():
    assert agent.parse_rollout_ids(TRAIN_URL) == ("roll-1", "att-2")
    assert agent.is_validation_url(VAL_URL) is True
    assert agent.is_validation_url(TRAIN_URL) is False
    with pytest.raises(ValueError):
        agent.parse_rollout_ids("http://127.0.0.1:9999/v1")
    with pytest.raises(ValueError):
        agent.is_validation_url("http://127.0.0.1:9999/v1/rollout/a/attempt/b/v1")


def test_orchestrator_client_passes_thinking_flag(experiment_config):
    cfg = experiment_config
    client = assembly.build_orchestrator_client(
        cfg, base_url=TRAIN_URL, model="auto", api_key=SECRET, temperature=cfg.orchestrator.temperature
    )
    assert client._extra_body == {"chat_template_kwargs": {"enable_thinking": cfg.orchestrator.enable_thinking}}


def test_build_retriever_is_a_cheap_http_client_that_loads_no_index(monkeypatch, experiment_config):
    """Роллаут не грузит FAISS и корпус — это один раз делает общий retrieval-сервер."""
    import orchestrator_research.adapters.retrieval as in_process_retrieval

    def must_not_load(*args, **kwargs):
        raise AssertionError("the index must not be loaded in the rollout process")

    monkeypatch.setattr(in_process_retrieval.BrowseCompRetriever, "from_shards", must_not_load)
    retriever = assembly.build_retriever(experiment_config)
    assert retriever.endpoint == experiment_config.retrieval.server_url + "/search"
    assert retriever.max_retries == experiment_config.retrieval.max_retries


def test_question_from_env_rejects_non_string_answers():
    with pytest.raises(ValueError):
        agent._question_from_env({"QUESTION_ID": "1", "QUESTION": "q", "ANSWERS": json.dumps([1])})
    question = agent._question_from_env({"QUESTION_ID": "1", "QUESTION": "q", "ANSWERS": json.dumps(["a"])})
    assert question == Question(question_id="1", text="q", gold_answers=("a",))


def test_missing_subagent_key_fails_before_any_model_call(monkeypatch, experiment_config):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        assembly.build_episode_deps(
            experiment_config, base_url=TRAIN_URL, model="auto", api_key=SECRET, temperature=0.0
        )
