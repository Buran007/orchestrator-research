"""Учебный прогон `demo`: идёт через настоящий код и печатает все разделы."""

from orchestrator_research.cli import main
from orchestrator_research.demo import run_demo


def test_walkthrough_runs_the_real_code_and_prints_every_section(capsys):
    assert run_demo() == 0
    text = capsys.readouterr().out
    for marker in (
        "1. TaskBudget",
        "ход 2: зарезервировано 7",  # вторая бронь — остаток бюджета задачи в 20 токенов
        "2. Один эпизод swarm",
        "статус=completed, ответ='Paris'",
        "3. Награда",
        "validation: total=1.000",
        "4. Будущий RL",
        "advantages=[0.5, 0.5, -0.5, -0.5]",
        "Веса здесь не обновляются",
    ):
        assert marker in text


def test_cli_demo_has_no_json_mode(capsys):
    assert main(["demo"]) == 0
    assert "Учебный запуск" in capsys.readouterr().out
