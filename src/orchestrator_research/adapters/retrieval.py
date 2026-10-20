"""Поиск BrowseComp-Plus: плотный поиск FAISS, эмбеддинги запросов и HTTP-клиент сервера поиска.

faiss, numpy, datasets, transformers и huggingface_hub импортируются внутри функций: процессам агента
нужен только ``HttpRetriever``, и импорт модуля для них дешёвый.
"""

from __future__ import annotations

import glob
import http.client
import json
import os
import pickle
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from ..config import EMBEDDING_MODEL_BY_VARIANT
from ..domain import SearchResult

DEFAULT_TOP_K = 5
QUERY_INSTRUCTION = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_CAP_SECONDS = 8.0


class CorpusLookup(Protocol):
    def get_text(self, docid: str) -> str | None:
        """Текст документа по его id."""


class QueryEmbedder(Protocol):
    def encode(self, texts: Sequence[str]) -> object:
        """Матрица: строка-вектор на каждый текст."""


class EmbeddingError(RuntimeError):
    """Сервис эмбеддингов не вернул корректные векторы."""


class RetrievalError(RuntimeError):
    """Сервер поиска отклонил запрос или так и не ответил после всех повторов."""


class _TransientRetrievalError(RetrievalError):
    """Ошибка, которую может вылечить повтор."""


def download_index(
    index_dir: str | Path,
    *,
    variant: str,
    repo_id: str,
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Скачать один вариант готового индекса; вернуть локальный каталог с шардами."""
    if variant not in EMBEDDING_MODEL_BY_VARIANT:
        raise ValueError(f"unsupported BrowseComp index variant {variant!r}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Index download requires the optional 'huggingface_hub' dependency") from exc

    root = Path(index_dir).expanduser()
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{variant}/*"],
        local_dir=str(root),
        revision=revision,
        token=token,
    )
    shard_dir = root / variant
    if not list(shard_dir.glob("corpus.shard*.pkl")):
        raise FileNotFoundError(f"No FAISS index shards found under {shard_dir}")
    return shard_dir


class HuggingFaceBrowseCompCorpus:
    """Корпус через memory-map: индекс docid без загрузки всех текстов в RAM."""

    def __init__(self, *, repo_id: str, revision: str | None = None) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("Corpus loading requires the optional 'datasets' dependency") from exc
        self._dataset = load_dataset(repo_id, split="train", revision=revision).select_columns(["docid", "text"])
        self._row_for_docid: dict[str, int] = {}
        for row_index, docid in enumerate(self._dataset["docid"]):
            key = str(docid)
            if key in self._row_for_docid:
                raise ValueError(f"duplicate docid in BrowseComp corpus: {key}")
            self._row_for_docid[key] = row_index

    def get_text(self, docid: str) -> str | None:
        row_index = self._row_for_docid.get(str(docid))
        if row_index is None:
            return None
        return str(self._dataset[row_index]["text"])

    @property
    def document_count(self) -> int:
        return len(self._row_for_docid)


class BrowseCompRetriever:
    """Поиск FAISS по скалярному произведению над pickle-шардами Tevatron.

    Индекс передаётся снаружи: тяжёлые библиотеки не нужны ядру, а тесты подставляют маленький индекс.
    """

    def __init__(
        self,
        *,
        index: Any,
        lookup: Sequence[object],
        embedder: QueryEmbedder,
        corpus: CorpusLookup,
    ) -> None:
        self._index = index
        self._lookup = tuple(str(docid) for docid in lookup)
        self._embedder = embedder
        self._corpus = corpus
        if not self._lookup:
            raise ValueError("FAISS lookup must not be empty")
        if self.vector_count != len(self._lookup):
            raise ValueError(f"index has {self.vector_count} vectors but lookup has {len(self._lookup)} ids")

    @classmethod
    def from_shards(
        cls, index_dir: str | Path, *, embedder: QueryEmbedder, corpus: CorpusLookup
    ) -> BrowseCompRetriever:
        """Загрузить шарды ``(passage_vectors, docids)`` в IndexFlatIP."""
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Dense retrieval requires optional 'faiss-cpu' and 'numpy' dependencies") from exc

        directory = Path(index_dir).expanduser()
        paths = sorted(glob.glob(str(directory / "corpus.shard*.pkl")))
        if not paths:
            raise FileNotFoundError(f"No corpus.shard*.pkl files found under {directory}")
        faiss_index, lookup = cls._read_index_shards(paths, faiss, np)
        return cls(index=faiss_index, lookup=lookup, embedder=embedder, corpus=corpus)

    @staticmethod
    def _read_index_shards(paths: Sequence[str], faiss: Any, np: Any) -> tuple[object, list[object]]:
        """Векторы шардов в порядке имён файлов → один плоский индекс."""
        faiss_index = None
        lookup: list[object] = []
        for shard_path in paths:
            # pickle при загрузке может выполнить код: грузим только из зафиксированной ревизии репозитория Tevatron.
            with open(shard_path, "rb") as stream:
                reps, docids = pickle.load(stream)  # noqa: S301
            vectors = np.ascontiguousarray(np.asarray(reps), dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[0] != len(docids):
                raise ValueError(f"malformed embedding shard {os.path.basename(shard_path)}")
            if faiss_index is None:
                faiss_index = faiss.IndexFlatIP(int(vectors.shape[1]))
            elif int(vectors.shape[1]) != int(faiss_index.d):
                raise ValueError("embedding shards have inconsistent vector dimensions")
            faiss_index.add(vectors)
            lookup.extend(docids)
        return faiss_index, lookup

    @property
    def vector_dimension(self) -> int:
        return int(self._index.d)

    @property
    def vector_count(self) -> int:
        return int(self._index.ntotal)

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        *,
        snippet: Callable[[str], str] | None = None,
    ) -> list[SearchResult]:
        """Лучшие ``top_k`` документов; ``snippet`` обрезает текст каждого (id и оценка сохраняются)."""
        import numpy as np

        query_vectors = self._query_vector(query, top_k, np)
        scores, indices = self._index.search(query_vectors, min(top_k, self.vector_count))
        results: list[SearchResult] = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if int(index) < 0 or int(index) >= len(self._lookup):
                continue
            docid = self._lookup[int(index)]
            text = self._corpus.get_text(docid)
            if text is None:
                raise KeyError(
                    f"retrieved docid {docid!r} is missing from the pinned corpus; index/corpus revisions may not match"
                )
            if snippet is not None:
                text = snippet(text)
            results.append(SearchResult(doc_id=docid, text=str(text), score=float(score)))
        return results

    def _query_vector(self, query: str, top_k: int, np: Any) -> object:
        """Вектор запроса, нормированный; размерность сверяется с индексом."""
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vectors = np.asarray(self._embedder.encode([query]), dtype=np.float32)
        if query_vectors.shape != (1, self.vector_dimension):
            raise ValueError(
                f"query vector shape {query_vectors.shape} does not match index dimension "
                f"{self.vector_dimension}; embedding model and index must match"
            )
        if not np.isfinite(query_vectors).all():
            raise ValueError("query embedding contains non-finite values")
        norm = float(np.linalg.norm(query_vectors[0]))
        if norm <= 1e-12:
            raise ValueError("query embedding is a zero vector")
        return np.ascontiguousarray(query_vectors / norm, dtype=np.float32)


class TokenizerSnippet:
    """Обрезка документа до ``max_tokens`` токенов токенизатором бенчмарка — это и есть сниппет."""

    def __init__(self, tokenizer_id: str, max_tokens: int) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Token-accurate snippets require the optional 'transformers' dependency") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self._max_tokens = max_tokens

    def __call__(self, text: str) -> str:
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= self._max_tokens:
            return text
        return self._tokenizer.decode(token_ids[: self._max_tokens], skip_special_tokens=True)


# --- эмбеддинги запросов (OpenAI-совместимый API) ---


class EmbeddingsApi(Protocol):
    """``client.embeddings`` из OpenAI SDK."""

    def create(self, **kwargs: Any) -> Any: ...


def browsecomp_query_text(query: str) -> str:
    """Добавить к запросу ровно тот префикс-инструкцию, с которым строился индекс BrowseComp-Plus."""
    if not query.strip():
        raise ValueError("query must not be empty")
    return QUERY_INSTRUCTION + query


def _normalize(vectors: object) -> object:
    import numpy as np

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise EmbeddingError("embedding backend returned a non-finite or non-matrix result")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise EmbeddingError("embedding backend returned a zero vector")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


class OpenAICompatibleEmbedder:
    """Эмбеддинги запросов пачкой через ``embeddings.create``; повторы делает SDK, остальное — ``EmbeddingError``."""

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        api_key: str,
        timeout_seconds: float = 60.0,
        expected_dimension: int | None = None,
        max_retries: int = 3,
        embeddings: EmbeddingsApi | None = None,
    ) -> None:
        if embeddings is None:
            from openai import OpenAI

            embeddings = OpenAI(
                base_url=base_url, api_key=api_key, timeout=timeout_seconds, max_retries=max_retries
            ).embeddings
        self._embeddings = embeddings
        self._model_id = model_id
        self.expected_dimension = expected_dimension

    def encode(self, texts: Sequence[str]) -> object:
        import numpy as np
        import openai

        if not texts:
            return np.empty((0, self.expected_dimension or 0), dtype=np.float32)
        inputs = [browsecomp_query_text(text) for text in texts]
        try:
            response = self._embeddings.create(model=self._model_id, input=inputs, encoding_format="float")
        except openai.APIError as exc:
            raise EmbeddingError(f"embedding endpoint request failed: {type(exc).__name__}: {exc}") from exc
        records = list(response.data)
        if len(records) != len(texts):
            raise EmbeddingError(f"embedding count {len(records)} does not match input count {len(texts)}")
        # Некоторые провайдеры возвращают векторы не по порядку — сопоставляем по полю ``index``.
        vectors = np.asarray([record.embedding for record in sorted(records, key=lambda r: r.index)], dtype=np.float32)
        if self.expected_dimension is not None and (vectors.ndim != 2 or vectors.shape[1] != self.expected_dimension):
            raise EmbeddingError(
                f"embedding dimension {vectors.shape[-1] if vectors.ndim else '?'} does not match "
                f"index dimension {self.expected_dimension}; model and index must match"
            )
        return _normalize(vectors)


# --- HTTP-клиент сервера поиска ---


class HttpRetriever:
    """``POST {server_url}/search``; реализует ``Retriever``.

    429, 5xx и сбои соединения повторяются с экспоненциальной паузой (с потолком); прочие ошибки — сразу.
    """

    def __init__(
        self,
        server_url: str,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not server_url.strip():
            raise ValueError("server_url must not be empty")
        self.endpoint = server_url.rstrip("/") + "/search"
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._sleep = sleep

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        payload = json.dumps({"query": query, "top_k": top_k}, ensure_ascii=False).encode("utf-8")
        attempts = self.max_retries + 1
        last: _TransientRetrievalError | None = None
        for attempt in range(attempts):
            try:
                return _parse_results(self._post_once(payload))
            except _TransientRetrievalError as exc:
                last = exc
                if attempt + 1 < attempts:
                    self._sleep(min(_BACKOFF_CAP_SECONDS, _BACKOFF_BASE_SECONDS * (2**attempt)))
        raise RetrievalError(f"retrieval server {self.endpoint} unavailable after {attempts} attempts: {last}")

    def _post_once(self, payload: bytes) -> Any:
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429 or exc.code >= 500:
                raise _TransientRetrievalError(f"HTTP {exc.code}") from exc
            raise RetrievalError(f"retrieval server rejected the search: HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
            raise _TransientRetrievalError(str(exc)) from exc
        try:
            return json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RetrievalError("retrieval server returned non-JSON content") from exc


def _parse_results(body: Any) -> list[SearchResult]:
    if not isinstance(body, dict) or not isinstance(body.get("results"), list):
        raise RetrievalError("retrieval response must be an object with a 'results' list")
    results: list[SearchResult] = []
    for item in body["results"]:
        if not isinstance(item, dict):
            raise RetrievalError("retrieval result must be an object")
        doc_id, text, score = item.get("doc_id"), item.get("text"), item.get("score")
        if not isinstance(doc_id, str) or not isinstance(text, str) or isinstance(score, bool):
            raise RetrievalError("retrieval result needs string doc_id and text and a numeric score")
        if not isinstance(score, (int, float)):
            raise RetrievalError("retrieval result needs string doc_id and text and a numeric score")
        results.append(SearchResult(doc_id=doc_id, text=text, score=float(score)))
    return results
