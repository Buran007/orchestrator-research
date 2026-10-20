"""Детерминированные заглушки для тестов и смоуков: без сети и без весов."""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..domain import ModelTurn, SearchResult


class ScriptedChatModel:
    """Отдаёт заданные ходы по очереди (исключение в сценарии — бросает); соблюдает ``max_tokens``,
    все запросы пишет в ``calls``.
    """

    def __init__(self, turns: Iterable[ModelTurn | BaseException]) -> None:
        self._turns: deque[ModelTurn | BaseException] = deque(turns)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        max_tokens: int,
        thinking: bool | None = None,
    ) -> ModelTurn:
        recorded = {
            "messages": copy.deepcopy([dict(m) for m in messages]),
            "tools": list(tools or ()),
            "max_tokens": max_tokens,
            "thinking": thinking,
        }
        self.calls.append(recorded)
        if not self._turns:
            raise RuntimeError("ScriptedChatModel has no scripted turn left")
        item = self._turns.popleft()
        if isinstance(item, BaseException):
            raise item
        if item.output_tokens > max_tokens:
            raise ValueError(f"scripted turn uses {item.output_tokens} tokens but max_tokens was {max_tokens}")
        return item


class StaticRetriever:
    """На любой запрос — одни и те же документы (до ``top_k``); запросы пишет в ``queries``."""

    def __init__(self, hits: Sequence[SearchResult] = ()) -> None:
        self._hits = tuple(hits)
        self.queries: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> Sequence[SearchResult]:
        self.queries.append((query, top_k))
        return self._hits[:top_k]
