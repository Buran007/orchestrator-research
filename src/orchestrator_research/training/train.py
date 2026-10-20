"""Запуск RL-обучения оркестратора через Agent Lightning + verl (ADR 0005).

Запуск::

    .venv/bin/python -m orchestrator_research.training.train --config experiments/template/config.toml
    ... --smoke          # 2 шага, батч 2, по 4 строки на сплит
    ... --dry-run        # напечатать переопределения, ничего не писать, verl/agl не импортировать

Окружение (через переменные, а не флаги: argv сохраняется в manifest.json):

    AGL_SERVER_URL       адрес agl-server с точки зрения тренера, например http://127.0.0.1:18080
    AGL_KEY              тот же ключ, что у agl-server и agl-controller

Папка прогона ``artifacts/runs/<experiment.name>/<run_id>/``: ``config.resolved.json``, ``trainer_overrides.json``,
``manifest.json``; если уже существует — отказ.
Строки данных: ``question_id``, ``question``, ``answers``, ``split``, ``train_step`` (-1; для обучающих роллаутов
его перезаписывает ``hooks.StepCounterHooks``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from orchestrator_research.config import ExperimentConfig, find_project_root, load_config
from orchestrator_research.storage import atomic_write_json
from orchestrator_research.training.overrides import (
    ROLLOUTS_PER_STEP_ENV,
    assert_lengths_consistent,
    build_trainer_config,
    with_agl_endpoint,
)

REQUIRED_ROW_FIELDS = ("question_id", "question", "answers", "split", "train_step")
RAY_ENV_VARS = "ray_kwargs.ray_init.runtime_env.env_vars"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the orchestrator with Agent Lightning + verl.")
    parser.add_argument("--config", required=True, help="experiment TOML file")
    parser.add_argument("--run-id", default=None, help="run folder name (default: UTC timestamp)")
    parser.add_argument("--smoke", action="store_true", help="tiny run: 2 steps, batch 2, 4 rows per split")
    parser.add_argument("--dry-run", action="store_true", help="print overrides and exit; write nothing")
    return parser


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def _load_rows(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}; run data prep first")
    from datasets import Dataset  # лениво: тяжёлый, нужен только для настоящего прогона

    dataset = Dataset.from_parquet(str(path))
    if limit is not None:
        dataset = dataset.select(range(min(limit, dataset.num_rows)))
    if dataset.num_rows == 0:
        raise ValueError(f"dataset is empty: {path}")
    missing = [field for field in REQUIRED_ROW_FIELDS if field not in dataset.column_names]
    if missing:
        raise ValueError(f"{path} rows miss fields {missing}; data prep must write {list(REQUIRED_ROW_FIELDS)}")
    return dataset.to_list()


def _compose_base_config(overrides: Mapping[str, Any]) -> Any:
    """Собрать базовый конфиг agl/verl и применить к нему переопределения."""
    import importlib.resources

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    verl_pkg = importlib.resources.files("agentlightning.verl")
    with initialize_config_dir(config_dir=str(verl_pkg), version_base=None):
        base_cfg = compose(config_name="config")
    OmegaConf.set_struct(base_cfg, False)
    for key, value in overrides.items():
        OmegaConf.update(base_cfg, key, value, merge=True)
    return base_cfg


def _prepare_run_dir(
    cfg: ExperimentConfig, cfg_path: Path, root: Path, run_dir: Path, *, smoke: bool, argv: Sequence[str]
) -> None:
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}; choose another --run-id")
    run_dir.mkdir(parents=True)
    atomic_write_json(run_dir / "config.resolved.json", cfg.serializable())
    manifest = {
        "run_id": run_dir.name,
        "experiment": cfg.experiment.name,
        "config_path": str(cfg_path),
        "config_hash": cfg.config_hash(),
        "git": _git_state(root),
        "python": sys.version,
        "argv": list(argv),
        "smoke": smoke,
        "extra_overrides_keys": sorted(cfg.training.extra_overrides),
        "created_utc": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(run_dir / "manifest.json", manifest)


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(args_list)

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_config(cfg_path)
    root = find_project_root(cfg_path.parent)
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = root / "artifacts" / "runs" / cfg.experiment.name / run_id

    overrides = build_trainer_config(cfg, run_dir=run_dir, smoke=args.smoke)
    assert_lengths_consistent(cfg, overrides)

    if args.dry_run:
        for key, value in sorted(overrides.items()):
            print(f"{key} = {json.dumps(value, ensure_ascii=False)}")
        return 0

    # AGL_SERVER_URL и AGL_KEY проверяются до создания папки прогона; ключ — только в копию для тренера, не в файл.
    endpoint = with_agl_endpoint(
        overrides, base_url=os.environ.get("AGL_SERVER_URL", ""), key=os.environ.get("AGL_KEY", "")
    )
    _prepare_run_dir(cfg, cfg_path, root, run_dir, smoke=args.smoke, argv=args_list)
    atomic_write_json(run_dir / "trainer_overrides.json", overrides)

    # StepCounterHooks в процессе тренера читает эту переменную; воркеры Ray получают её через runtime_env.
    os.environ[ROLLOUTS_PER_STEP_ENV] = endpoint[f"{RAY_ENV_VARS}.{ROLLOUTS_PER_STEP_ENV}"]

    from agentlightning.verl.entrypoint import run_ppo
    from verl.trainer.main_ppo import get_ppo_ray_runtime_env

    # agl сливает runtime_env поверхностно ({**verl_default, **ours}): наши env_vars заменили бы
    # переменные verl, а не дополнили их.
    for name, value in get_ppo_ray_runtime_env().get("env_vars", {}).items():
        endpoint.setdefault(f"{RAY_ENV_VARS}.{name}", value)

    limit = 4 if args.smoke else None
    # agl-controller — отдельный процесс: пути к конфигу и папке прогона агент узнаёт из входа роллаута
    # (agentlightning.local.env_map в overrides.py).
    episode_context = {"config_path": str(cfg_path), "run_dir": str(run_dir)}
    train_rows = [
        {**row, **episode_context} for row in _load_rows(cfg.benchmark.data_dir / "train.parquet", limit=limit)
    ]
    val_file = cfg.benchmark.data_dir / f"{cfg.training.val_split}.parquet"
    val_rows = [{**row, **episode_context} for row in _load_rows(val_file, limit=limit)]
    print(f"train rows: {len(train_rows)}, val rows: {len(val_rows)}, run dir: {run_dir}")

    config = _compose_base_config(endpoint)
    run_ppo(config, train_dataset=train_rows, val_dataset=val_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
