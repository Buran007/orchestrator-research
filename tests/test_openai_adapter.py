"""`ChatClient`: разбор ответа OpenAI-совместимого API, проверка tool calls и usage, сборка запроса."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from orchestrator_research.adapters.chat import (
    ChatClient,
    MissingUsageError,
    ModelProtocolError,
    ModelRequestError,
    api_key_from_env,
)
from orchestrator_research.domain import ToolCall
from orchestrator_research.tools import orchestrator_tools


def make_client(transport, **overrides) -> ChatClient:
    fields = {
        "base_url": "https://api.example.test/v1",
        "model": "orchestrator-test",
        "api_key": "EMPTY",
        "temperature": 0.0,
        "top_p": 1.0,
        "timeout_seconds": 12.0,
        "max_retries": 2,
        "transport": transport,
    }
    fields.update(overrides)
    return ChatClient(**fields)


def response(*, content="answer", tool_calls=(), usage="default", finish_reason="stop"):
    msg = NS(content=content, tool_calls=list(tool_calls))
    if usage == "default":
        usage = usage_obj()
    return NS(choices=[NS(message=msg, finish_reason=finish_reason)], usage=usage)


def usage_obj(prompt=12, completion=5, total=17):
    return NS(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def tool_call(call_id="call_1", name="browsecomp_search", arguments='{"query":"Ada"}'):
    return NS(id=call_id, type="function", function=NS(name=name, arguments=arguments))


class FakeTransport:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def run(coro):
    return asyncio.run(coro)


def test_reads_output_tokens_and_strips_inline_think():
    transport = FakeTransport(response(content="<think>compare sources</think>Paris"))
    turn = run(make_client(transport).complete([{"role": "user", "content": "Where?"}], max_tokens=30))

    assert turn.text == "Paris"
    assert turn.final_answer == "Paris"
    assert turn.output_tokens == 5
    assert transport.requests[0]["max_tokens"] == 30
    assert "tools" not in transport.requests[0]


def test_parses_native_tool_calls_and_allows_null_content():
    transport = FakeTransport(response(content=None, tool_calls=[tool_call()], finish_reason="tool_calls"))
    turn = run(
        make_client(transport).complete(
            [{"role": "user", "content": "Search"}],
            max_tokens=50,
            tools=orchestrator_tools(subagents=True, role_names=("retriever",), model_aliases=("small",)),
        )
    )

    assert turn.text == ""
    assert turn.final_answer is None
    assert turn.tool_calls == (ToolCall("call_1", "browsecomp_search", {"query": "Ada"}),)
    assert transport.requests[0]["tool_choice"] == "auto"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [("not-json", "malformed JSON"), ('["x"]', "JSON object")],
)
def test_rejects_malformed_or_non_object_tool_arguments(arguments, message):
    transport = FakeTransport(response(content=None, tool_calls=[tool_call(arguments=arguments)]))
    with pytest.raises(ModelProtocolError, match=message):
        run(make_client(transport).complete([], max_tokens=10, tools=orchestrator_tools(subagents=False)))


@pytest.mark.parametrize("tools", [None, orchestrator_tools(subagents=False)])
def test_rejects_tool_call_that_was_not_declared(tools):
    transport = FakeTransport(response(content=None, tool_calls=[tool_call(name="shell")]))
    with pytest.raises(ModelProtocolError, match="undeclared tool"):
        run(make_client(transport).complete([], max_tokens=10, tools=tools))


def test_missing_usage_is_a_protocol_error():
    transport = FakeTransport(response(content="abc", usage=None))
    with pytest.raises(MissingUsageError):
        run(make_client(transport).complete([], max_tokens=10))


def test_provider_failure_becomes_model_request_error_without_client_retries():
    transport = FakeTransport(RuntimeError("rate limited"))
    with pytest.raises(ModelRequestError, match="model request failed: RuntimeError: rate limited"):
        run(make_client(transport).complete([], max_tokens=10))
    assert len(transport.requests) == 1


def test_response_format_is_forwarded_only_when_given():
    transport = FakeTransport(response(), response())
    run(make_client(transport).complete([], max_tokens=10, response_format={"type": "json_object"}))
    run(make_client(transport).complete([], max_tokens=10))
    assert transport.requests[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in transport.requests[1]


def test_extra_body_is_forwarded_only_when_configured():
    transport = FakeTransport(response(), response())
    run(
        make_client(transport, extra_body={"chat_template_kwargs": {"enable_thinking": False}}).complete(
            [], max_tokens=10
        )
    )
    run(make_client(transport).complete([], max_tokens=10))
    assert transport.requests[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "extra_body" not in transport.requests[1]


def test_rejects_non_positive_max_tokens_before_any_request():
    transport = FakeTransport()
    with pytest.raises(ValueError):
        run(make_client(transport).complete([], max_tokens=0))
    assert transport.requests == []


def test_empty_api_key_name_means_keyless_endpoint():
    assert api_key_from_env("") == "EMPTY"


def test_missing_key_variable_is_named_without_leaking_values(monkeypatch):
    name = "ORCHESTRATOR_RESEARCH_TEST_MISSING_KEY"
    monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match=name):
        api_key_from_env(name)


def test_reads_present_key_variable(monkeypatch):
    name = "ORCHESTRATOR_RESEARCH_TEST_PRESENT_KEY"
    monkeypatch.setenv(name, "not-a-real-key")
    assert api_key_from_env(name) == "not-a-real-key"


def test_prompt_tokens_are_read_from_usage():
    transport = FakeTransport(response(usage=usage_obj(prompt=812, completion=5)))
    turn = run(make_client(transport).complete([], max_tokens=10))
    assert (turn.prompt_tokens, turn.output_tokens) == (812, 5)


def test_missing_prompt_tokens_default_to_zero_without_failing():
    transport = FakeTransport(response(usage=NS(completion_tokens=7)))
    turn = run(make_client(transport).complete([], max_tokens=10))
    assert (turn.prompt_tokens, turn.output_tokens) == (0, 7)


def test_thinking_override_merges_into_chat_template_kwargs():
    transport = FakeTransport(response())
    client = make_client(transport, extra_body={"chat_template_kwargs": {"enable_thinking": True, "x": 1}, "y": 2})
    run(client.complete([], max_tokens=5, thinking=False))
    assert transport.requests[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False, "x": 1}, "y": 2}
    assert client._extra_body["chat_template_kwargs"]["enable_thinking"] is True  # умолчание клиента не испорчено
