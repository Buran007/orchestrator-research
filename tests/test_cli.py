"""CLI: аргументы и умолчания подкоманд, проброс в исполнителей, коды выхода."""

from pathlib import Path

import pytest

from orchestrator_research import application, cli
from orchestrator_research.application import Check
from orchestrator_research.benchmark import SPLIT_NAMES
from orchestrator_research.eval_runner import EvalSummary

EXPERIMENT = Path(__file__).parents[1] / "experiments/template/config.toml"


def test_eval_defaults_match_the_documented_contract():
    args = cli.build_parser().parse_args(["eval", "--config", "x.toml", "--run-id", "r1"])
    assert args.split == "val_test"
    assert args.checkpoint_url is None
    assert args.max_questions is None
    assert args.concurrency == cli.DEFAULT_CONCURRENCY == 8


def test_eval_accepts_the_three_splits_and_rejects_others():
    parser = cli.build_parser()
    for split in SPLIT_NAMES:
        assert parser.parse_args(["eval", "--config", "x", "--run-id", "r", "--split", split]).split == split
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "--config", "x", "--run-id", "r", "--split", "heldout"])


def test_eval_requires_run_id_and_positive_counts():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "--config", "x"])
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "--config", "x", "--run-id", "r", "--max-questions", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "--config", "x", "--run-id", "r", "--concurrency", "-1"])


def test_every_documented_command_is_registered():
    parser = cli.build_parser()
    for argv in (
        ["validate-config", "--config", "x"],
        ["prepare-data", "--config", "x"],
        ["serve-retrieval", "--config", "x", "--port", "9000"],
        ["doctor", "--config", "x"],
        ["train", "--config", "x", "--smoke", "--dry-run"],
        ["report", "runs/a"],
        ["compare", "a.jsonl", "b.jsonl"],
        ["demo"],
    ):
        args, _ = parser.parse_known_args(argv)  # train разбирает флаги сам, поэтому неизвестные допустимы
        assert args.command == argv[0]


def test_train_passes_its_arguments_through(monkeypatch):
    captured = {}

    def fake_train(argv):
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr("orchestrator_research.training.train.main", fake_train)
    assert cli.main(["train", "--config", str(EXPERIMENT), "--run-id", "r9", "--smoke", "--dry-run"]) == 0
    assert captured["argv"] == ["--config", str(EXPERIMENT), "--run-id", "r9", "--smoke", "--dry-run"]


def test_serve_retrieval_passes_host_and_port(monkeypatch):
    captured = {}

    def fake_serve(argv):
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr("orchestrator_research.adapters.retrieval_server.main", fake_serve)
    assert cli.main(["serve-retrieval", "--config", "x.toml", "--port", "9001"]) == 0
    assert captured["argv"] == ["--config", "x.toml", "--host", "127.0.0.1", "--port", "9001"]


def test_eval_command_hands_arguments_to_the_runner(monkeypatch, tmp_path):
    captured = {}

    async def fake_run_eval(cfg, **kwargs):
        captured.update(kwargs)
        return EvalSummary(run_dir=tmp_path, path=tmp_path / "val_test.jsonl", evaluated=0, resumed=0, errors=())

    monkeypatch.setattr("orchestrator_research.eval_runner.run_eval", fake_run_eval)
    code = cli.main(
        [
            "eval",
            "--config",
            str(EXPERIMENT),
            "--run-id",
            "r1",
            "--checkpoint-url",
            "http://127.0.0.1:9000/v1",
            "--max-questions",
            "3",
            "--concurrency",
            "4",
        ]
    )
    assert code == 0
    assert captured["split"] == "val_test"
    assert captured["run_id"] == "r1"
    assert captured["checkpoint_url"] == "http://127.0.0.1:9000/v1"
    assert captured["max_questions"] == 3
    assert captured["concurrency"] == 4
    assert captured["artifact_root"].name == "artifacts"


def test_eval_exit_code_is_one_when_episodes_crashed(monkeypatch, tmp_path):
    async def fake_run_eval(cfg, **kwargs):
        return EvalSummary(run_dir=tmp_path, path=tmp_path / "x.jsonl", evaluated=0, resumed=0, errors=("q1: boom",))

    monkeypatch.setattr("orchestrator_research.eval_runner.run_eval", fake_run_eval)
    assert cli.main(["eval", "--config", str(EXPERIMENT), "--run-id", "r1"]) == 1


def test_doctor_exit_code_follows_fail_checks(monkeypatch):
    monkeypatch.setattr(application, "doctor", lambda cfg: [Check("OK", "a", "fine")])
    assert cli.main(["doctor", "--config", str(EXPERIMENT)]) == 0
    monkeypatch.setattr(application, "doctor", lambda cfg: [Check("FAIL", "a", "broken")])
    assert cli.main(["doctor", "--config", str(EXPERIMENT)]) == 1


def test_user_errors_exit_with_two_and_a_message(capsys):
    assert cli.main(["validate-config", "--config", "does-not-exist.toml"]) == 2
    assert "error:" in capsys.readouterr().err


def test_validate_config_prints_name_strategy_limits_pool_and_hash(capsys):
    assert cli.main(["validate-config", "--config", str(EXPERIMENT)]) == 0
    out = capsys.readouterr().out
    assert "experiment: template" in out
    assert "strategy: swarm" in out
    assert "max_turns=15" in out
    assert "subagent pool:" in out
    assert "config_hash: " in out
