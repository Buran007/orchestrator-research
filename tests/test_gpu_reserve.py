"""Удержание памяти GPU: по умолчанию выключено, срабатывает один раз до `init_model`, атрибуты verl сохраняются."""

import pytest

from orchestrator_research.training.gpu_reserve import RESERVE_ENV, install, reserve_gb_from_env, wrap_init_model


def _worker_cls(calls):
    class Worker:
        def init_model(self, x):
            calls.append(("init", x))
            return "ok"

    Worker.init_model.dispatch_mode = "ONE_TO_ALL"  # вместо атрибута от `@register` verl
    return Worker


def test_env_parsing(monkeypatch):
    monkeypatch.delenv(RESERVE_ENV, raising=False)
    assert reserve_gb_from_env() == 0
    monkeypatch.setenv(RESERVE_ENV, "30.5")
    assert reserve_gb_from_env() == 30.5
    monkeypatch.setenv(RESERVE_ENV, "-1")
    with pytest.raises(ValueError):
        reserve_gb_from_env()


def test_install_is_noop_when_off(monkeypatch):
    monkeypatch.setenv(RESERVE_ENV, "0")
    install()  # не должен импортировать verl


def test_hold_runs_before_init_once_and_keeps_attributes():
    calls = []
    cls = _worker_cls(calls)
    hold = lambda gb: calls.append(("hold", gb))
    wrap_init_model(cls, 30.0, hold)
    wrap_init_model(cls, 30.0, hold)  # повторный вызов не удерживает память второй раз
    assert cls.init_model.dispatch_mode == "ONE_TO_ALL"
    assert cls().init_model(1) == "ok"
    assert calls == [("hold", 30.0), ("init", 1)]
