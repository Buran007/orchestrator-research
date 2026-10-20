"""Отдельный сервер поиска BrowseComp-Plus: индекс FAISS и корпус держит один процесс.

Agent Lightning запускает процесс на каждый роллаут (до 64 параллельно), поэтому индекс, корпус и токенизатор
сниппетов грузятся здесь один раз, а роллауты ходят в ``POST /search`` через ``HttpRetriever``
(как в Search-R1).

    python -m orchestrator_research.adapters.retrieval_server --config CFG [--host H] [--port P]

Endpoints::

    GET  /healthz  -> {"status": "ok", "documents": N, "vectors": M}
    POST /search   {"query": str, "top_k": int}  -> {"results": [{"doc_id", "text", "score"}, ...]}

Коды: 400 — плохое или слишком большое тело, 404 — неизвестный путь, 503 — сервис эмбеддингов временно
недоступен (клиент повторит), 500 — всё прочее.

Каждый запрос — в своём потоке (``ThreadingHTTPServer``); ``IndexFlatIP.search`` только читает, блокировка не нужна.
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from ..config import ExperimentConfig, load_config
from ..domain import SearchResult
from .chat import api_key_from_env
from .retrieval import (
    DEFAULT_TOP_K,
    BrowseCompRetriever,
    EmbeddingError,
    HuggingFaceBrowseCompCorpus,
    OpenAICompatibleEmbedder,
    TokenizerSnippet,
)

MIN_TOP_K = 1
MAX_TOP_K = 50
MAX_QUERY_CHARS = 4000
MAX_BODY_BYTES = 64 * 1024

SearchFn = Callable[[str, int], Sequence[SearchResult]]


class BadRequest(ValueError):
    """Тело запроса не является корректным запросом поиска → 400."""


def parse_search_request(raw: bytes) -> tuple[str, int]:
    """Проверить тело ``/search``; вернуть ``(query, top_k)``, ``top_k`` ограничен 1..50."""
    if len(raw) > MAX_BODY_BYTES:
        raise BadRequest(f"request body exceeds {MAX_BODY_BYTES} bytes")
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BadRequest("request body must be UTF-8 JSON") from exc
    if not isinstance(body, dict):
        raise BadRequest("request body must be a JSON object")
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        raise BadRequest("'query' must be a non-empty string")
    if len(query) > MAX_QUERY_CHARS:
        raise BadRequest(f"'query' must be at most {MAX_QUERY_CHARS} characters")
    top_k = body.get("top_k", DEFAULT_TOP_K)
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise BadRequest("'top_k' must be an integer")
    return query.strip(), min(MAX_TOP_K, max(MIN_TOP_K, top_k))


class RetrievalHandler(BaseHTTPRequestHandler):
    server_version = "orchestrator-retrieval/1.0"

    def do_GET(self) -> None:  # noqa: N802 - имя метода задаёт stdlib
        if self.path == "/healthz":
            self._send(200, {"status": "ok", **self.server.stats})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - имя метода задаёт stdlib
        if self.path != "/search":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send(400, {"error": "Content-Length header is required"})
            return
        # Читаем максимум на байт больше лимита: слишком большое тело отклоняем, не читая целиком.
        raw = self.rfile.read(min(max(0, length), MAX_BODY_BYTES + 1))
        try:
            query, top_k = parse_search_request(raw)
        except BadRequest as exc:
            self._send(400, {"error": str(exc)})
            return
        try:
            hits = self.server.search(query, top_k)
        except EmbeddingError:
            self._send(503, {"error": "retrieval backend temporarily unavailable"})
            return
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
            return
        except Exception:  # наружу подробностей не отдаём, traceback — в лог сервера
            self.log_error("search failed:\n%s", traceback.format_exc())
            self._send(500, {"error": "internal retrieval error"})
            return
        self._send(200, {"results": [{"doc_id": hit.doc_id, "text": hit.text, "score": hit.score} for hit in hits]})

    def _send(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_server(
    search: SearchFn, stats: Mapping[str, int], host: str = "127.0.0.1", port: int = 0
) -> ThreadingHTTPServer:
    """Создать сервер; ``port=0`` — свободный порт (смотреть ``server_address``)."""
    server = ThreadingHTTPServer((host, port), RetrievalHandler)
    server.search = search
    server.stats = dict(stats)
    return server


def load_backend(cfg: ExperimentConfig) -> tuple[SearchFn, dict[str, int]]:
    """Загрузить эмбеддер, корпус, шарды FAISS и токенизатор сниппетов (минуты и несколько GB).

    Возвращает функцию поиска (со сниппетами) и размеры для ``/healthz``.
    """
    retrieval, embedding = cfg.retrieval, cfg.embedding
    embedder = OpenAICompatibleEmbedder(
        base_url=embedding.base_url,
        model_id=embedding.model,
        api_key=api_key_from_env(embedding.api_key_env),
        timeout_seconds=embedding.timeout_seconds,
        max_retries=embedding.max_retries,
    )
    corpus = HuggingFaceBrowseCompCorpus(repo_id=retrieval.corpus_repo_id, revision=retrieval.corpus_revision)
    retriever = BrowseCompRetriever.from_shards(retrieval.index_path, embedder=embedder, corpus=corpus)
    snippet = TokenizerSnippet(retrieval.snippet_tokenizer, max_tokens=retrieval.snippet_tokens)
    stats = {"documents": corpus.document_count, "vectors": retriever.vector_count}
    return partial(retriever.search, snippet=snippet), stats


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve BrowseComp-Plus dense retrieval over HTTP.")
    parser.add_argument("--config", required=True, help="experiment TOML file")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default: port of retrieval.server_url)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    port = args.port if args.port is not None else (urlsplit(cfg.retrieval.server_url).port or 8100)
    search, stats = load_backend(cfg)
    server = make_server(search, stats, args.host, port)
    host, bound = server.server_address[0], server.server_address[1]
    print(f"retrieval server on http://{host}:{bound} stats={stats}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
