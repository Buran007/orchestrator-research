"""Точка входа роллаута для Agent Lightning: вопрос → эпизод → награда.

agl-controller создаёт ``OrchestratorAgent`` без аргументов в новом процессе на каждый роллаут и ждёт ``run()``.
Всё нужное приходит через окружение (``training/overrides.py: ENV_MAP``) и адреса agl:

    AGL_OPENAI_BASE_URL  OpenAI-совместимый прокси; в пути /mode/train/ или /mode/val/
                         и /rollout/<rollout_id>/attempt/<attempt_id>/
    AGL_EVENT_URL        сюда отправляется {"event_type": "reward", "data": {...}}
    AGL_KEY              bearer-токен для обоих; не логируется и не пишется на диск

Через прокси идут только запросы оркестратора — поэтому учится только он; субагенты и судья — со своими
клиентами. Обучение и оценка используют один ``run_episode``, отличается лишь адрес оркестратора.

Отладка одного эпизода против ``orchestrator.eval_base_url`` (печатает запись, награду не шлёт):
``python -m orchestrator_research.agent --config CFG --question-id ID``.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

from . import assembly
from .adapters.chat import api_key_from_env
from .benchmark import HuggingFaceBrowseCompRepository
from .config import ExperimentConfig, load_config
from .domain import Question
from .storage import atomic_write_json, canonical_json

ROLLOUT_RE = re.compile(r"/rollout/(?P<rollout>[^/]+)/attempt/(?P<attempt>[^/]+)")
REWARD_TIMEOUT_SECONDS = 30.0


def parse_rollout_ids(base_url: str) -> tuple[str, str]:
    """``(rollout_id, attempt_id)`` из адреса прокси agl; ошибка, если их там нет."""
    match = ROLLOUT_RE.search(base_url)
    if match is None:
        raise ValueError("AGL_OPENAI_BASE_URL has no /rollout/<rollout_id>/attempt/<attempt_id>/ segment")
    return match.group("rollout"), match.group("attempt")


def is_validation_url(base_url: str) -> bool:
    """Это прокси валидации? Если в адресе нет ни одного режима — ошибка, а не догадка."""
    if "/mode/val/" in base_url:
        return True
    if "/mode/train/" in base_url:
        return False
    raise ValueError("AGL_OPENAI_BASE_URL must contain /mode/train/ or /mode/val/")


def post_reward(event_url: str, agl_key: str, payload: dict[str, Any]) -> None:
    """Отправить награду; любая ошибка — исключение, чтобы роллаут молча не получил 0."""
    request = urllib.request.Request(
        event_url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {agl_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=REWARD_TIMEOUT_SECONDS) as response:
        status = response.status
    if not 200 <= status < 300:
        raise RuntimeError(f"reward event rejected with HTTP {status}")


def _question_from_env(env: dict[str, str]) -> Question:
    answers = json.loads(env["ANSWERS"])
    if not isinstance(answers, list) or not all(isinstance(item, str) for item in answers):
        raise ValueError("ANSWERS must be a JSON list of strings")
    return Question(question_id=env["QUESTION_ID"], text=env["QUESTION"], gold_answers=tuple(answers))


class OrchestratorAgent:
    """Роллаут: эпизод → запись на диск → награда."""

    async def run(self) -> None:
        env = dict(os.environ)
        cfg = load_config(env["ORCH_RESEARCH_CONFIG"])
        run_dir = Path(env["ORCH_RESEARCH_RUN_DIR"])
        base_url = env["AGL_OPENAI_BASE_URL"]
        agl_key = env["AGL_KEY"]
        event_url = env["AGL_EVENT_URL"]
        split = env.get("SPLIT", "")
        step = int(env.get("TRAIN_STEP", "-1"))
        question = _question_from_env(env)
        is_validation = is_validation_url(base_url)
        rollout_id, attempt_id = parse_rollout_ids(base_url)

        deps = assembly.build_episode_deps(
            cfg,
            base_url=base_url,
            model="auto",  # прокси agl подставит имя обучаемой модели
            api_key=agl_key,
            # В режиме обучения прокси всё равно подставит температуру роллаута.
            temperature=cfg.orchestrator.eval_temperature if is_validation else cfg.orchestrator.temperature,
        )
        record = await assembly.run_episode(cfg, question, deps, step=step, is_validation=is_validation)

        mode = "val" if is_validation else "train"
        path = run_dir / "episodes" / mode / str(step) / f"{rollout_id}-{attempt_id}.json"
        document = assembly.episode_document(
            record,
            split=split,
            step=step,
            is_validation=is_validation,
            experiment=cfg.experiment.name,
            rollout_id=rollout_id,
            attempt_id=attempt_id,
        )
        await asyncio.to_thread(atomic_write_json, path, document)

        reason = f"{record.episode.status.value}/{record.grade.status.value}"
        payload = {
            "event_type": "reward",
            "data": {
                "value": record.reward.total,
                "source": "agent",
                "reason": reason,
                "breakdown": dataclasses.asdict(record.reward),
                "split": split,
            },
        }
        await asyncio.to_thread(post_reward, event_url, agl_key, payload)


async def _run_debug_episode(cfg: ExperimentConfig, question_id: str) -> assembly.EpisodeRecord:
    repository = HuggingFaceBrowseCompRepository(
        repo_id=cfg.benchmark.repo_id,
        revision=cfg.benchmark.revision,
        expected_rows=cfg.benchmark.expected_rows,
    )
    matches = [item for item in repository.load() if item.question_id == question_id]
    if not matches:
        raise ValueError(f"question id {question_id!r} is not in the BrowseComp-Plus dataset")
    deps = assembly.build_episode_deps(
        cfg,
        base_url=cfg.orchestrator.eval_base_url,
        model=cfg.orchestrator.eval_model,
        api_key=api_key_from_env(cfg.orchestrator.api_key_env),
        temperature=cfg.orchestrator.eval_temperature,
    )
    return await assembly.run_episode(cfg, matches[0], deps, step=-1, is_validation=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one episode against the eval endpoint. Posts no reward.")
    parser.add_argument("--config", required=True, help="experiment TOML file")
    parser.add_argument("--question-id", required=True, help="BrowseComp-Plus query_id to run")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    record = asyncio.run(_run_debug_episode(cfg, args.question_id))
    document = assembly.episode_document(
        record,
        split="debug",
        step=-1,
        is_validation=True,
        experiment=cfg.experiment.name,
        rollout_id=None,
        attempt_id=None,
    )
    print(json.dumps(json.loads(canonical_json(document)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
