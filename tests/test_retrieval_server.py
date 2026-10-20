"""HTTP-сервер поиска и его клиенты: коды ответа, повторы `HttpRetriever`, эмбеддер поверх OpenAI SDK.

Сервер поднимается на случайном порту 127.0.0.1, FAISS заменён `FakeSearch`.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestrator_research import assembly
from orchestrator_research.adapters.retrieval import (
    EmbeddingError,
    HttpRetriever,
    OpenAICompatibleEmbedder,
    RetrievalError,
)
from orchestrator_research.adapters.retrieval_server import (
    DEFAULT_TOP_K,
    MAX_BODY_BYTES,
    MAX_TOP_K,
    BadRequest,
    make_server,
    parse_search_request,
)
from orchestrator_research.config import load_config
from orchestrator_research.domain import SearchResult

EXPERIMENT = Path(__file__).parents[1] / "experiments/template/config.toml"
STATS = {"documents": 830, "vectors": 2}


class FakeSearch:
    """Вместо поиска по FAISS: пишет вызовы; первые ``unavailable_times`` раз — `EmbeddingError`, ``error`` — всегда."""

    def __init__(self, *, unavailable_times=0, error=None):
        self.calls = []
        self._unavailable = unavailable_times
        self._error = error

    def __call__(self, query, top_k):
        self.calls.append((query, top_k))
        if self._error is not None:
            raise self._error
        if self._unavailable > 0:
            self._unavailable -= 1
            raise EmbeddingError("embedding server warming up")
        return [SearchResult("doc-1", "Paris is the capital.", 0.9), SearchResult("doc-2", "second", 0.5)][:top_k]


def _start(search):
    server = make_server(search, STATS, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}"


def _stop(server):
    server.shutdown()
    server.server_close()


@pytest.fixture
def running_server():
    search = FakeSearch()
    server, url = _start(search)
    yield search, url
    _stop(server)


def _post(url, body: bytes):
    request = urllib.request.Request(url + "/search", data=body, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_healthz_reports_index_sizes(running_server):
    _, url = running_server
    with urllib.request.urlopen(url + "/healthz", timeout=5) as response:
        assert response.status == 200
        assert json.loads(response.read()) == {"status": "ok", **STATS}


def test_search_happy_path_returns_doc_id_text_and_score(running_server):
    search, url = running_server
    status, body = _post(url, json.dumps({"query": "  capital  ", "top_k": 1}).encode())
    assert status == 200
    assert body == {"results": [{"doc_id": "doc-1", "text": "Paris is the capital.", "score": 0.9}]}
    assert search.calls == [("capital", 1)]


@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"not json",
        b"[1, 2]",
        b'{"query": ""}',
        b'{"top_k": 3}',
        b'{"query": "x", "top_k": "3"}',
        b'{"query": "x", "top_k": true}',
    ],
)
def test_bad_search_bodies_get_400_and_never_reach_the_backend(running_server, body):
    search, url = running_server
    status, payload = _post(url, body)
    assert status == 400
    assert "error" in payload
    assert search.calls == []


def test_unknown_path_is_404(running_server):
    _, url = running_server
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(urllib.request.Request(url + "/nope"), timeout=5)
    assert caught.value.code == 404


def test_embedding_unavailable_maps_to_503():
    server, url = _start(FakeSearch(unavailable_times=1))
    try:
        status, payload = _post(url, json.dumps({"query": "x"}).encode())
    finally:
        _stop(server)
    assert status == 503
    assert payload == {"error": "retrieval backend temporarily unavailable"}


def test_parse_search_request_defaults_and_clamps_top_k():
    assert parse_search_request(b'{"query": "q"}') == ("q", DEFAULT_TOP_K)
    assert parse_search_request(b'{"query": "q", "top_k": 999}') == ("q", MAX_TOP_K)
    assert parse_search_request(b'{"query": "q", "top_k": 0}') == ("q", 1)
    with pytest.raises(BadRequest):
        parse_search_request(b'{"query": "q", "top_k": 1.5}')


def test_oversized_body_is_a_bad_request_not_a_separate_status():
    with pytest.raises(BadRequest, match="exceeds"):
        parse_search_request(b"x" * (MAX_BODY_BYTES + 1))


def test_http_retriever_retries_after_one_503_then_succeeds():
    search = FakeSearch(unavailable_times=1)
    server, url = _start(search)
    sleeps = []
    try:
        hits = HttpRetriever(url, max_retries=3, sleep=sleeps.append).search("capital", 2)
    finally:
        _stop(server)
    assert [hit.doc_id for hit in hits] == ["doc-1", "doc-2"]
    assert hits[0] == SearchResult("doc-1", "Paris is the capital.", 0.9)
    assert search.calls == [("capital", 2), ("capital", 2)]
    assert sleeps == [0.5]


def test_http_retriever_does_not_retry_client_errors():
    search = FakeSearch(error=ValueError("query embedding is a zero vector"))
    server, url = _start(search)
    sleeps = []
    try:
        with pytest.raises(RetrievalError, match="HTTP 400"):
            HttpRetriever(url, max_retries=3, sleep=sleeps.append).search("q", 1)
    finally:
        _stop(server)
    assert len(search.calls) == 1
    assert sleeps == []


def test_http_retriever_gives_up_after_max_retries_on_connection_refused():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
    sleeps = []
    retriever = HttpRetriever(f"http://127.0.0.1:{closed_port}", max_retries=2, sleep=sleeps.append)
    with pytest.raises(RetrievalError, match="after 3 attempts"):
        retriever.search("q", 1)
    assert sleeps == [0.5, 1.0]


def test_http_retriever_retries_a_truncated_response(monkeypatch):
    class _Truncated:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            raise http.client.IncompleteRead(b"{")

    class _Ok(_Truncated):
        def read(self):
            return json.dumps({"results": []}).encode()

    responses = iter([_Truncated(), _Ok()])
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: next(responses))
    sleeps = []
    assert HttpRetriever("http://x", max_retries=1, sleep=sleeps.append).search("q", 1) == []
    assert sleeps == [0.5]


def test_build_retriever_is_the_http_client_for_the_configured_server():
    cfg = load_config(EXPERIMENT)
    retriever = assembly.build_retriever(cfg)
    assert isinstance(retriever, HttpRetriever)
    assert retriever.endpoint == cfg.retrieval.server_url + "/search"
    assert retriever.max_retries == cfg.retrieval.max_retries


def _embedding_reply(*records):
    return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=e) for i, e in records])


class _FakeEmbeddings:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply(kwargs)


def test_embedder_reorders_by_index_and_normalises_vectors():
    np = pytest.importorskip("numpy")
    embeddings = _FakeEmbeddings(lambda _kw: _embedding_reply((1, [0.0, 2.0]), (0, [3.0, 4.0])))
    embedder = OpenAICompatibleEmbedder(base_url="http://x/v1", model_id="m", api_key="k", embeddings=embeddings)

    vectors = embedder.encode(["first", "second"])

    assert embeddings.calls[0]["encoding_format"] == "float"
    assert embeddings.calls[0]["input"][0].endswith("first")
    assert np.allclose(vectors, [[0.6, 0.8], [0.0, 1.0]])


def test_embedder_maps_sdk_failures_to_embedding_error():
    pytest.importorskip("numpy")
    openai = pytest.importorskip("openai")
    import httpx

    request = httpx.Request("POST", "http://x/v1/embeddings")

    def fail(_kwargs):
        raise openai.APIConnectionError(request=request)

    embedder = OpenAICompatibleEmbedder(
        base_url="http://x/v1", model_id="m", api_key="k", embeddings=_FakeEmbeddings(fail)
    )
    with pytest.raises(EmbeddingError, match="embedding endpoint request failed"):
        embedder.encode(["q"])
