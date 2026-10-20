"""Асинхронный OpenAI-совместимый клиент: оркестратор, субагенты и судья.

Повторы при 408/409/429/5xx отдаются OpenAI SDK (``max_retries``). Объекты SDK наружу не выходят —
ядро получает ``ModelTurn``. SDK импортируется лениво, тестам сеть не нужна. Без стриминга.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from uuid import uuid4

from ..domain import ModelTurn, ToolCall


class CompletionTransport(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class ModelRequestError(RuntimeError):
    """Запрос к модели не удался после повторов SDK."""


class ModelProtocolError(RuntimeError):
    """Ответ провайдера нельзя безопасно положить в диалог."""


class MissingUsageError(ModelProtocolError):
    """Провайдер не вернул usage — без него нельзя вести бюджет токенов."""


def api_key_from_env(env_name: str) -> str:
    """Ключ API из переменной окружения; пустое имя — локальный сервер без ключа. Значение в ошибки не попадает."""
    if not env_name:
        return "EMPTY"
    value = os.environ.get(env_name)
    if not value:
        raise ValueError(f"environment variable {env_name!r} is not set or empty; export it before running")
    return value


def _completion_tokens(response: Any) -> int | None:
    usage = getattr(response, "usage", None)
    value = getattr(usage, "completion_tokens", None) if usage is not None else None
    try:
        return None if value is None else max(0, int(value))
    except (TypeError, ValueError):
        return None


def _prompt_tokens(response: Any) -> int:
    """Размер промпта от провайдера; 0, если нет — тогда ``ContextMeter`` считает по символам."""
    usage = getattr(response, "usage", None)
    value = getattr(usage, "prompt_tokens", None) if usage is not None else None
    try:
        return 0 if value is None else max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _strip_inline_reasoning(content: str) -> str:
    """Убрать блоки ``<think>...</think>``; остаток — текст ответа."""
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


class ChatClient:
    """Chat-completions клиент. ``ModelTurn.output_tokens`` — ``completion_tokens`` провайдера;
    ответ без usage — ``MissingUsageError``.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float,
        top_p: float,
        timeout_seconds: float,
        max_retries: int,
        extra_body: dict[str, Any] | None = None,
        transport: CompletionTransport | None = None,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._top_p = top_p
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._extra_body = dict(extra_body or {})
        self._transport = transport

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        max_tokens: int,
        response_format: Mapping[str, Any] | None = None,
        thinking: bool | None = None,
    ) -> ModelTurn:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        loop = asyncio.get_running_loop()
        started = loop.time()
        tool_list = [dict(tool) for tool in tools or ()]
        params = self._request_params(messages, max_tokens, tool_list, response_format, thinking)
        allowed_tools = {tool["function"]["name"] for tool in tool_list}
        try:
            response = await self._client().create(**params)
        except Exception as exc:
            raise ModelRequestError(f"model request failed: {type(exc).__name__}: {exc}") from exc
        return self._parse_response(response, max(0.0, loop.time() - started), allowed_tools=allowed_tools)

    def _request_params(
        self,
        messages: Sequence[Mapping[str, Any]],
        max_tokens: int,
        tools: list[dict[str, Any]],
        response_format: Mapping[str, Any] | None,
        thinking: bool | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self._model,
            "messages": [dict(message) for message in messages],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": max_tokens,
        }
        if tools:
            # Модель видит JSON-схему, но аргументы всё равно проверяются при разборе ответа.
            params["tools"] = tools
            params["tool_choice"] = "auto"
        if response_format is not None:
            params["response_format"] = dict(response_format)
        extra_body = self._extra_body
        if thinking is not None:
            # переопределить thinking в chat template на один запрос (chat_template_kwargs vLLM/Qwen)
            kwargs = {**extra_body.get("chat_template_kwargs", {}), "enable_thinking": thinking}
            extra_body = {**extra_body, "chat_template_kwargs": kwargs}
        if extra_body:
            params["extra_body"] = extra_body
        return params

    def _client(self) -> CompletionTransport:
        if self._transport is None:
            from openai import AsyncOpenAI

            self._transport = AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                timeout=self._timeout,
                max_retries=self._max_retries,
            ).chat.completions
        return self._transport

    def _parse_response(self, response: Any, elapsed: float, *, allowed_tools: set[str]) -> ModelTurn:
        choices = getattr(response, "choices", None) or ()
        if not choices:
            raise ModelProtocolError("chat completion returned no choices")
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise ModelProtocolError("chat completion choice has no message")

        raw_content = getattr(message, "content", None)
        content = raw_content if isinstance(raw_content, str) else ""
        content = _strip_inline_reasoning(content)
        calls = self._parse_tool_calls(message, allowed_tools)

        output_tokens = _completion_tokens(response)
        if output_tokens is None:
            raise MissingUsageError("provider omitted usage.completion_tokens; generated tokens cannot be budgeted")

        answer = content if not calls and content else None
        return ModelTurn(
            text=content,
            output_tokens=output_tokens,
            finish_reason=getattr(choice, "finish_reason", None),
            tool_calls=tuple(calls),
            final_answer=answer,
            elapsed_seconds=elapsed,
            prompt_tokens=_prompt_tokens(response),
        )

    @staticmethod
    def _parse_tool_calls(message: Any, allowed_tools: set[str]) -> list[ToolCall]:
        """Вызовы функций от провайдера → проверенные ``ToolCall``."""
        calls: list[ToolCall] = []
        for raw in getattr(message, "tool_calls", None) or ():
            function = getattr(raw, "function", None)
            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", None)
            if not isinstance(name, str) or not name.strip():
                raise ModelProtocolError("tool call has no function name")
            if name not in allowed_tools:
                raise ModelProtocolError(f"model called undeclared tool {name!r}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ModelProtocolError(f"tool {name!r} returned malformed JSON arguments") from exc
            if not isinstance(arguments, Mapping):
                raise ModelProtocolError(f"tool {name!r} arguments must be a JSON object")
            call_id = getattr(raw, "id", None) or f"call_{uuid4().hex}"
            calls.append(ToolCall(call_id=str(call_id), name=name, arguments=dict(arguments)))
        return calls
