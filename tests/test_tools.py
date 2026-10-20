"""Инструмент поиска: разбор аргументов, рендер выдачи, разделение ошибок модели и инфраструктуры."""

import asyncio

import pytest

from orchestrator_research.adapters.fake import StaticRetriever
from orchestrator_research.domain import SearchResult
from orchestrator_research.tools import (
    NO_RESULTS_TEXT,
    executed,
    parse_search_arguments,
    render_hits,
    run_search,
    search_tool_schema,
)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"query": "  alpha  "}, ("alpha", 5)),
        ({"query": "beta", "num_results": 2}, ("beta", 2)),
        ({"query": "gamma", "num_results": 50}, ("gamma", 5)),  # не больше top_k
    ],
)
def test_valid_arguments_are_stripped_and_capped(arguments, expected):
    assert parse_search_arguments(arguments, top_k=5) == expected


@pytest.mark.parametrize(
    "arguments",
    [
        "not a mapping",
        {"query": ""},
        {"query": 3},
        {"query": "ok", "num_results": 0},
        {"query": "ok", "num_results": True},
        {"query": "ok", "extra": 1},
    ],
)
def test_invalid_arguments_raise_value_error(arguments):
    with pytest.raises(ValueError):
        parse_search_arguments(arguments, top_k=5)


def test_render_hits_numbers_passages_from_rank_one():
    text = render_hits([SearchResult("d1", "first", 0.5), SearchResult("d2", "second", 0.25)])
    assert text == "1. [DocID: d1] (Score: 0.5000)\n   first\n\n2. [DocID: d2] (Score: 0.2500)\n   second"


def test_run_search_happy_path_returns_rendered_hits():
    retriever = StaticRetriever([SearchResult("d1", "passage", 1.0)])
    content, is_error, infra_error = asyncio.run(run_search(retriever, {"query": "q"}, top_k=3))
    assert "[DocID: d1]" in content
    assert (is_error, infra_error) == (False, False)
    assert retriever.queries == [("q", 3)]


def test_run_search_with_no_hits_says_so():
    content, is_error, _ = asyncio.run(run_search(StaticRetriever(()), {"query": "q"}, top_k=3))
    assert content == NO_RESULTS_TEXT
    assert not is_error


def test_run_search_bad_arguments_are_a_model_error_without_touching_the_retriever():
    retriever = StaticRetriever([SearchResult("d1", "passage", 1.0)])
    content, is_error, infra_error = asyncio.run(run_search(retriever, {"query": ""}, top_k=3))
    assert content.startswith("search failed:")
    assert (is_error, infra_error) == (True, False)
    assert retriever.queries == []


class _DownRetriever:
    def search(self, query, top_k):
        raise ConnectionError("down")


def test_run_search_retriever_exception_is_an_infrastructure_error():
    content, is_error, infra_error = asyncio.run(run_search(_DownRetriever(), {"query": "q"}, top_k=3))
    assert content == "search failed: ConnectionError: down"
    assert (is_error, infra_error) == (True, True)


@pytest.mark.parametrize(
    ("is_error", "infra_error", "ran"),
    [(False, False, True), (True, True, True), (True, False, False)],
)
def test_executed_distinguishes_rejections_from_environment_failures(is_error, infra_error, ran):
    assert executed(is_error, infra_error) is ran


def test_search_schema_has_no_hint_and_requires_query():
    schema = search_tool_schema()["function"]
    assert schema["parameters"]["required"] == ["query"]
    assert schema["parameters"]["properties"]["num_results"]["description"] == "Maximum number of results."
