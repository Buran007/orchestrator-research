"""Лаунчер agl-server подменяет только таймаут upstream; класс клиента остаётся родным классом httpx."""

import httpx

from orchestrator_research.training.agl_server import httpx_with_timeout


def test_every_timeout_becomes_the_configured_one():
    shim = httpx_with_timeout(httpx, 1800.0)
    assert shim.AsyncClient is httpx.AsyncClient
    timeout = shim.Timeout(timeout=300.0)
    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (1800.0,) * 4
