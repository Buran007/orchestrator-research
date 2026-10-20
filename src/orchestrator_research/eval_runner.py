"""Оценка чекпоинта оркестратора на одном сплите: параллельно, с продолжением, только дописыванием.

Папка прогона::

    artifacts/runs/<experiment>/<run_id>/eval/config.json    пишется один раз; другой конфиг — отказ
    artifacts/runs/<experiment>/<run_id>/eval/<split>.jsonl  эпизод на строку, дописывается с fsync

Эпизоды идут через ``assembly.run_episode`` — тот же код, что в обучении, только вместо прокси Agent Lightning
адрес чекпоинта. При продолжении вопросы с годным ответом пропускаются; со сбоем судьи или модели — повторяются.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import assembly
from .adapters.chat import api_key_from_env
from .benchmark import SPLIT_NAMES
from .config import ExperimentConfig
from .domain import Question
from .storage import JsonlResultStore, write_immutable_json


@dataclass(frozen=True, slots=True)
class EvalSummary:
    run_dir: Path
    path: Path
    evaluated: int
    resumed: int
    errors: tuple[str, ...]


def split_path(cfg: ExperimentConfig, split: str) -> Path:
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; allowed: {list(SPLIT_NAMES)}")
    return cfg.benchmark.data_dir / f"{split}.parquet"


def question_from_row(row: dict[str, Any]) -> Question:
    """Строка parquet (``question_id``, ``question``, ``answers``) → ``Question``."""
    return Question(
        question_id=str(row["question_id"]),
        text=str(row["question"]),
        gold_answers=tuple(str(answer) for answer in row["answers"]),
    )


def load_split_questions(path: Path) -> list[Question]:
    """Прочитать подготовленный сплит (``datasets`` импортируется только здесь)."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run prepare-data for this config first")
    from datasets import Dataset

    return [question_from_row(row) for row in Dataset.from_parquet(str(path)).to_list()]


async def run_eval(
    cfg: ExperimentConfig,
    *,
    split: str,
    run_id: str,
    artifact_root: Path,
    checkpoint_url: str | None = None,
    max_questions: int | None = None,
    concurrency: int = 8,
    questions: Iterable[Question] | None = None,
) -> EvalSummary:
    """Оценить ``split`` чекпоинтом по ``checkpoint_url``, продолжая с его JSONL.

    ``questions`` можно передать готовыми; иначе читается parquet сплита из ``benchmark.data_dir``.
    """
    if concurrency <= 0:
        raise ValueError("concurrency must be > 0")
    if max_questions is not None and max_questions <= 0:
        raise ValueError("max_questions must be > 0 when given")
    rows = list(questions) if questions is not None else load_split_questions(split_path(cfg, split))
    ids = [question.question_id for question in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"split {split!r} has duplicate question_id values")
    selected = rows[:max_questions] if max_questions is not None else rows

    run_dir = Path(artifact_root) / "runs" / cfg.experiment.name / run_id
    eval_dir = run_dir / "eval"
    write_immutable_json(eval_dir / "config.json", {"config_hash": cfg.config_hash(), "config": cfg.serializable()})
    store = JsonlResultStore(eval_dir / f"{split}.jsonl")
    completed = store.completed_ids()
    pending = [question for question in selected if question.question_id not in completed]

    deps = assembly.build_episode_deps(
        cfg,
        base_url=checkpoint_url or cfg.orchestrator.eval_base_url,
        model=cfg.orchestrator.eval_model,
        api_key=api_key_from_env(cfg.orchestrator.api_key_env),
        temperature=cfg.orchestrator.eval_temperature,
    )
    question_gate = asyncio.Semaphore(concurrency)

    async def evaluate_one(question: Question) -> None:
        async with question_gate:
            record = await assembly.run_episode(cfg, question, deps, step=-1, is_validation=True)
        store.append(
            assembly.episode_document(record, split=split, step=-1, is_validation=True, experiment=cfg.experiment.name)
        )

    outcomes = await asyncio.gather(*(evaluate_one(question) for question in pending), return_exceptions=True)
    errors = tuple(
        f"{question.question_id}: {type(outcome).__name__}: {outcome}"
        for question, outcome in zip(pending, outcomes, strict=True)
        if isinstance(outcome, BaseException)
    )
    return EvalSummary(
        run_dir=run_dir,
        path=store.path,
        evaluated=len(pending) - len(errors),
        resumed=len(selected) - len(pending),
        errors=errors,
    )


def eval_rows_for_split(rows: Sequence[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Из ``storage.collect_run_rows`` — только строки оценки одного сплита."""
    return [row for row in rows if row["source"] == "eval" and row["split"] == split]
