"""Ретривер BrowseComp: префикс инструкции к запросу, допустимые варианты индекса, обрезка сниппетов."""

from __future__ import annotations

import pytest

from orchestrator_research.adapters.retrieval import (
    QUERY_INSTRUCTION,
    BrowseCompRetriever,
    browsecomp_query_text,
    download_index,
)
from orchestrator_research.domain import SearchResult


def test_query_text_gets_the_exact_instruction_prefix():
    query = "who wrote the origin of species"
    assert browsecomp_query_text(query) == QUERY_INSTRUCTION + query


def test_empty_query_is_rejected_before_embedding():
    with pytest.raises(ValueError, match="must not be empty"):
        browsecomp_query_text("   ")


def test_only_supported_index_variants_can_be_downloaded():
    with pytest.raises(ValueError, match="unsupported"):
        download_index("unused", variant="unknown", repo_id="unused")


def test_search_trims_snippets_but_keeps_ids_and_scores():
    np = pytest.importorskip("numpy")

    class FlatIndex:
        d = 2
        ntotal = 2

        def search(self, vectors, k):
            assert vectors.shape == (1, 2)
            return np.array([[0.9, 0.5]], dtype=np.float32), np.array([[0, 1]])

    class Embedder:
        def encode(self, texts):
            return np.array([[3.0, 4.0]], dtype=np.float32)

    class Corpus:
        def get_text(self, docid):
            return {"d1": "abcdefghij", "d2": "klmnop"}[docid]

    retriever = BrowseCompRetriever(index=FlatIndex(), lookup=["d1", "d2"], embedder=Embedder(), corpus=Corpus())
    hits = retriever.search("q", 2, snippet=lambda text: text[:4])

    assert hits == [
        SearchResult(doc_id="d1", text="abcd", score=pytest.approx(0.9)),
        SearchResult(doc_id="d2", text="klmn", score=pytest.approx(0.5)),
    ]
