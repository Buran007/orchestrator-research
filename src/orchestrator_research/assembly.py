"""Сборка: здесь и только здесь адаптеры, стратегии и награда соединяются по ``ExperimentConfig``.

Доменные модули этот файл не импортируют. Поиск — общий HTTP-сервис (``adapters/retrieval_server.py``),
отсюда создаётся только клиент: процессы роллаутов не грузят ни FAISS-индекс, ни корпус.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .adapters.chat import ChatClient, api_key_from_env
from .adapters.judge import OpenAICompatibleJudge
from .adapters.retrieval import HttpRetriever
from .config import ExperimentConfig, load_role_prompts
from .domain import ChatModel, EpisodeResult, GradeResult, GradeStatus, Judge, Question, Retriever, Strategy
from .limits import TaskBudget
from .reward import RewardBreakdown, compose_reward
from .storage import SCHEMA_VERSION
from .strategies import get_strategy
from .strategies.base import EpisodeEnvironment
from .subagents import SubagentPool


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """Итог эпизода: трасса, оценка судьи, награда."""

    episode: EpisodeResult
    grade: GradeResult
    reward: RewardBreakdown


@dataclass(frozen=True, slots=True)
class EpisodeDeps:
    """Клиенты и общие семафоры процесса; создаются один раз, все эпизоды процесса их переиспользуют."""

    orchestrator: ChatModel
    retriever: Retriever
    judge: Judge
    subagent_clients: Mapping[str, ChatModel]
    role_prompts: Mapping[str, str]
    semaphore: asyncio.Semaphore


def build_episode_deps(
    cfg: ExperimentConfig,
    *,
    base_url: str,
    model: str,
    api_key: str,
    temperature: float,
) -> EpisodeDeps:
    """Создать все клиенты; при отсутствии ключа падает сразу, до первого запроса к модели."""
    api_key_from_env(cfg.subagents.api_key_env)
    return EpisodeDeps(
        orchestrator=build_orchestrator_client(
            cfg, base_url=base_url, model=model, api_key=api_key, temperature=temperature
        ),
        retriever=build_retriever(cfg),
        judge=build_judge(cfg),
        subagent_clients={
            alias: build_subagent_client(cfg, model_id) for alias, model_id in cfg.subagents.models.items()
        },
        role_prompts=load_role_prompts(cfg.subagents.prompts_path),
        semaphore=asyncio.Semaphore(cfg.subagents.max_concurrency),
    )


def build_retriever(cfg: ExperimentConfig) -> Retriever:
    """HTTP-клиент общего сервера поиска (``retrieval.server_url``)."""
    return HttpRetriever(
        cfg.retrieval.server_url,
        timeout_seconds=cfg.retrieval.timeout_seconds,
        max_retries=cfg.retrieval.max_retries,
    )


def build_orchestrator_client(
    cfg: ExperimentConfig,
    *,
    base_url: str,
    model: str,
    api_key: str,
    temperature: float,
) -> ChatClient:
    """Клиент обучаемого оркестратора: прокси Agent Lightning или локальный vLLM."""
    return ChatClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        top_p=cfg.orchestrator.top_p,
        timeout_seconds=cfg.orchestrator.timeout_seconds,
        max_retries=cfg.orchestrator.max_retries,
        extra_body={"chat_template_kwargs": {"enable_thinking": cfg.orchestrator.enable_thinking}},
    )


def build_subagent_client(cfg: ExperimentConfig, model_id: str) -> ChatClient:
    """Клиент модели субагента; его запросы в данные обучения не попадают."""
    return ChatClient(
        base_url=cfg.subagents.base_url,
        model=model_id,
        api_key=api_key_from_env(cfg.subagents.api_key_env),
        temperature=cfg.subagents.temperature,
        top_p=cfg.subagents.top_p,
        timeout_seconds=cfg.subagents.timeout_seconds,
        max_retries=cfg.subagents.max_retries,
        extra_body=dict(cfg.subagents.extra_body),
    )


def build_judge(cfg: ExperimentConfig) -> Judge:
    return OpenAICompatibleJudge(cfg.judge)


def build_strategy(cfg: ExperimentConfig, deps: EpisodeDeps) -> Strategy:
    """Стратегия на один эпизод: новый бюджет и пул субагентов поверх общих клиентов."""
    budget = TaskBudget(cfg.limits)
    pool = SubagentPool(
        clients=deps.subagent_clients,
        role_prompts=deps.role_prompts,
        budget=budget,
        limits=cfg.limits,
        retriever=deps.retriever,
        top_k=cfg.retrieval.top_k,
        return_max_chars=cfg.subagents.return_max_chars,
        semaphore=deps.semaphore,
        context_limit=cfg.subagents.max_context_tokens,
    )
    env = EpisodeEnvironment(
        orchestrator=deps.orchestrator,
        pool=pool,
        retriever=deps.retriever,
        budget=budget,
        limits=cfg.limits,
        top_k=cfg.retrieval.top_k,
        context_limit=cfg.training.max_context_tokens,
    )
    return get_strategy(cfg.experiment.strategy)(env)


async def run_episode(
    cfg: ExperimentConfig,
    question: Question,
    deps: EpisodeDeps,
    *,
    step: int,
    is_validation: bool,
) -> EpisodeRecord:
    """Вопрос целиком: стратегия → судья → награда.

    ``r_perf`` = 1.0 только при вердикте «correct». Любой сбой судьи — 0.0 со статусом в ``grade``,
    чтобы сбой не выглядел правильным ответом.
    """
    result = await build_strategy(cfg, deps).run(question)
    grade = await _grade_fail_closed(deps.judge, question, result.final_answer)
    r_perf = 1.0 if grade.status is GradeStatus.GRADED and grade.correct is True else 0.0
    reward = compose_reward(r_perf, result.stages, step=step, is_validation=is_validation, cfg=cfg.reward)
    return EpisodeRecord(episode=result, grade=grade, reward=reward)


def episode_document(
    record: EpisodeRecord,
    *,
    split: str,
    step: int,
    is_validation: bool,
    **extra: Any,
) -> dict[str, Any]:
    """JSON эпизода — один формат для обучения и оценки. Без ключей и секретов."""
    return {
        "schema_version": SCHEMA_VERSION,
        "question_id": record.episode.question_id,
        "split": split,
        "step": step,
        "is_validation": is_validation,
        **extra,
        "reward": record.reward,
        "grade": record.grade,
        "episode": record.episode,
    }


async def _grade_fail_closed(judge: Judge, question: Question, answer: str | None) -> GradeResult:
    try:
        return await judge.grade(question, answer)
    except Exception as exc:  # сетевые ошибки судья уже возвращает статусом; это — на случай багов
        return GradeResult(
            status=GradeStatus.TRANSPORT_ERROR,
            correct=None,
            error=f"{type(exc).__name__}: {exc}",
        )


__all__ = [
    "EpisodeDeps",
    "EpisodeRecord",
    "build_episode_deps",
    "build_judge",
    "build_orchestrator_client",
    "build_retriever",
    "build_strategy",
    "build_subagent_client",
    "episode_document",
    "run_episode",
]
