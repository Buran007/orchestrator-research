"""``agl-server`` с настраиваемым тайм-аутом до vLLM: ``python -m orchestrator_research.training.agl_server ...``.

В Agent Lightning 1.0.2 тайм-аут прокси зашит (300 с) и при тайм-ауте запрос повторяется до 6 раз. На общей GPU
длинный ход идёт дольше 300 с, и прокси перезапускал ту же генерацию, пока роллаут не упирался в свой лимит.
Здесь тайм-аут берётся из ``OR_AGL_UPSTREAM_TIMEOUT`` (по умолчанию 1800 с — как у роллаута); остальное —
штатный ``main`` agl с теми же аргументами Hydra.
"""

from __future__ import annotations

import os
import types

TIMEOUT_ENV = "OR_AGL_UPSTREAM_TIMEOUT"
DEFAULT_TIMEOUT = 1800.0


def httpx_with_timeout(httpx_module: types.ModuleType, seconds: float) -> types.SimpleNamespace:
    """Подмена ``httpx`` внутри ``app.py`` agl: тот же ``AsyncClient``, любой ``Timeout`` равен ``seconds``."""
    return types.SimpleNamespace(
        AsyncClient=httpx_module.AsyncClient,
        Timeout=lambda *args, **kwargs: httpx_module.Timeout(timeout=seconds),
    )


def main() -> None:
    import httpx
    from agentlightning.server import app as agl_app
    from agentlightning.server.__main__ import main as agl_main

    seconds = float(os.environ.get(TIMEOUT_ENV, "") or DEFAULT_TIMEOUT)
    agl_app.httpx = httpx_with_timeout(httpx, seconds)
    agl_main()


if __name__ == "__main__":
    main()
