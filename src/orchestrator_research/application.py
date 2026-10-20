"""Служебные команды: сводка конфига, подготовка данных, предстартовые проверки ``doctor``.

``doctor`` не печатает секреты — только имена переменных и задана ли каждая. Сетевые и GPU-проверки
подставляются снаружи, поэтому тесты работают без сети.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ExperimentConfig

TRAINING_PYTHON = (3, 12)
REQUIRED_SPLIT_FILES = ("train", "val_test", "val_full")
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost"})
GPU_HEADROOM_MIB = 6 * 1024  # запас под фазу обучения сверх доли vLLM


@dataclass(frozen=True, slots=True)
class Check:
    status: str  # "OK", "WARN" или "FAIL"
    name: str
    detail: str
    hint: str = ""


def config_summary(cfg: ExperimentConfig) -> list[str]:
    """Строки для ``validate-config``."""
    limits = ", ".join(f"{key}={value}" for key, value in dataclasses.asdict(cfg.limits).items())
    pool = ", ".join(f"{alias}={model}" for alias, model in sorted(cfg.subagents.models.items()))
    return [
        f"experiment: {cfg.experiment.name}",
        f"strategy: {cfg.experiment.strategy}",
        f"config_hash: {cfg.config_hash()}",
        f"limits: {limits}",
        f"subagent pool: {pool}",
        f"judge model: {cfg.judge.model}",
        f"retrieval server: {cfg.retrieval.server_url}",
    ]


def prepare_data(cfg: ExperimentConfig) -> list[str]:
    """Записать parquet train / val_test / val_full и скачать индекс поиска."""
    from .adapters.retrieval import download_index
    from .benchmark import write_training_splits

    lines = []
    for name, path in write_training_splits(cfg).items():
        lines.append(f"wrote {name}: {path}")
    shard_dir = download_index(
        cfg.retrieval.index_path.parent,
        variant=cfg.retrieval.index_variant,
        repo_id=cfg.retrieval.index_repo_id,
        revision=cfg.retrieval.index_revision,
        token=os.environ.get("HF_TOKEN") or None,
    )
    lines.append(f"index ready: {shard_dir}")
    return lines


def doctor(
    cfg: ExperimentConfig,
    *,
    http_get: Callable[[str, float], Any] | None = None,
    gpu_query: Callable[[], str | None] | None = None,
    python_version: tuple[int, int] | None = None,
    env: Mapping[str, str] | None = None,
) -> list[Check]:
    """Все предстартовые проверки; хоть один ``FAIL`` — эксперимент не запустится."""
    fetch = http_get or _get_json
    query = gpu_query or _nvidia_smi
    version = tuple(sys.version_info[:2]) if python_version is None else python_version
    environment = os.environ if env is None else env
    return [
        _env_check("subagents", cfg.subagents.api_key_env, environment),
        _env_check("judge", cfg.judge.api_key_env, environment),
        _models_check("subagent models", cfg.subagents.base_url, sorted(set(cfg.subagents.models.values())), fetch),
        _models_check("judge model", cfg.judge.base_url, [cfg.judge.model], fetch),
        _data_check(cfg.benchmark.data_dir),
        _service_check(
            "retrieval server",
            cfg.retrieval.server_url.rstrip("/") + "/healthz",
            fetch,
            hint="start it: python -m orchestrator_research.adapters.retrieval_server --config <config>",
        ),
        *_embedding_checks(cfg, fetch, environment),
        _gpu_check(query, cfg.training.rollout_gpu_memory_utilization),
        _python_check(version),
    ]


def render_checks(checks: Sequence[Check]) -> str:
    lines = []
    for check in checks:
        line = f"{check.status:<4} {check.name}: {check.detail}"
        if check.hint and check.status != "OK":
            line += f"\n     hint: {check.hint}"
        lines.append(line)
    return "\n".join(lines)


def _env_check(label: str, env_name: str, environment: Mapping[str, str]) -> Check:
    name = f"{label} API key"
    if not env_name:
        return Check("OK", name, "no key required (api_key_env is empty)")
    if environment.get(env_name):
        return Check("OK", name, f"{env_name} is set")
    return Check("FAIL", name, f"{env_name} is not set", hint=f"export {env_name} before running")


def _models_check(name: str, base_url: str, model_ids: Sequence[str], fetch: Callable[[str, float], Any]) -> Check:
    url = base_url.rstrip("/") + "/models"
    try:
        payload = fetch(url, 20.0)
        listed = {str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)}
    except (OSError, ValueError, AttributeError) as exc:
        return Check(
            "FAIL", name, f"cannot list models at {url}: {type(exc).__name__}", hint="check base_url and network"
        )
    missing = [model_id for model_id in model_ids if model_id not in listed]
    if missing:
        return Check("FAIL", name, f"not offered by {url}: {', '.join(missing)}", hint="fix the model id in the config")
    return Check("OK", name, f"all {len(model_ids)} model id(s) listed")


def _data_check(data_dir: Path) -> Check:
    missing = [name for name in REQUIRED_SPLIT_FILES if not (data_dir / f"{name}.parquet").is_file()]
    if missing:
        return Check("FAIL", "benchmark data", f"missing in {data_dir}: {', '.join(missing)}", hint="run prepare-data")
    return Check("OK", "benchmark data", f"all split files present in {data_dir}")


def _service_check(name: str, url: str, fetch: Callable[[str, float], Any], *, hint: str) -> Check:
    try:
        fetch(url, 10.0)
    except (OSError, ValueError) as exc:
        return Check("FAIL", name, f"no answer from {url}: {type(exc).__name__}", hint=hint)
    return Check("OK", name, f"responds at {url}")


def _embedding_checks(
    cfg: ExperimentConfig, fetch: Callable[[str, float], Any], environment: Mapping[str, str]
) -> list[Check]:
    """Локальный эмбеддер проверяется по HTTP, удалённый — одним настоящим запросом эмбеддинга."""
    base_url = cfg.embedding.base_url
    if (urllib.parse.urlsplit(base_url).hostname or "") in LOCAL_HOSTS:
        return [
            _service_check(
                "embedding server",
                base_url.rstrip("/") + "/models",
                fetch,
                hint=f"serve {cfg.embedding.model} at {base_url} or change [embedding] base_url",
            )
        ]
    key_check = _remote_key_check(cfg.embedding.api_key_env, environment)
    if key_check.status != "OK":
        return [key_check]
    return [key_check, _embedding_call_check(cfg, environment)]


def _remote_key_check(env_name: str, environment: Mapping[str, str]) -> Check:
    if not env_name:
        return Check(
            "FAIL",
            "embedding API key",
            "[embedding] api_key_env is empty but the embedder is remote",
            hint="set api_key_env to the name of the variable that holds the key, e.g. OPENROUTER_API_KEY",
        )
    return _env_check("embedding", env_name, environment)


def _embedding_call_check(cfg: ExperimentConfig, environment: Mapping[str, str]) -> Check:
    """Один запрос эмбеддинга для "ping"; ключ из окружения, не печатается."""
    import openai  # SDK нужен только этой проверке

    base_url = cfg.embedding.base_url
    try:
        client = openai.OpenAI(
            base_url=base_url,
            api_key=environment[cfg.embedding.api_key_env],
            timeout=cfg.embedding.timeout_seconds,
            max_retries=0,
        )
        response = client.embeddings.create(model=cfg.embedding.model, input="ping")
        dimension = len(response.data[0].embedding)
    except Exception as exc:  # любая ошибка и есть диагноз; показываем только тип и HTTP-статус
        status = getattr(exc, "status_code", None)
        suffix = f" (HTTP {status})" if status else ""
        return Check(
            "FAIL",
            "embedding call",
            f"embeddings request to {base_url} failed: {type(exc).__name__}{suffix}",
            hint=(
                f"check [embedding] base_url, model and {cfg.embedding.api_key_env}; "
                f"{cfg.embedding.model} must be listed at {base_url.rstrip('/')}/models"
            ),
        )
    return Check("OK", "embedding call", f"{cfg.embedding.model} returned a {dimension}-dim vector")


def _gpu_check(query: Callable[[], str | None], rollout_utilization: float) -> Check:
    """Свободной памяти первой видимой GPU должно хватать на долю vLLM плюс запас."""
    output = query()
    if output is None:
        return Check("WARN", "GPU", "nvidia-smi is unavailable", hint="training needs an NVIDIA GPU; ignore on CPU")
    numbers = [int(value) for value in re.findall(r"\d+", output)]
    if len(numbers) < 2:
        return Check(
            "WARN", "GPU", f"cannot parse nvidia-smi output: {output}", hint="expected memory.total,memory.used in MiB"
        )
    total, used = numbers[0], numbers[1]
    free = total - used
    needed = rollout_utilization * total + GPU_HEADROOM_MIB
    detail = (
        f"free {free} MiB of {total} MiB; need {needed:.0f} MiB "
        f"(rollout {rollout_utilization:g} x total + {GPU_HEADROOM_MIB} MiB headroom)"
    )
    if free < needed:
        return Check(
            "FAIL",
            "GPU",
            detail,
            hint=(
                "lower training.rollout_gpu_memory_utilization or training.max_context_tokens; "
                "other processes on a shared card may hold memory (see nvidia-smi)"
            ),
        )
    return Check("OK", "GPU", detail)


def _python_check(version: tuple[int, int]) -> Check:
    text = ".".join(str(part) for part in version)
    if version >= TRAINING_PYTHON:
        return Check("OK", "python", f"{text} (training needs >= 3.12)")
    return Check("WARN", "python", f"{text}; training needs >= 3.12", hint="install Python 3.12 for training")


def _get_json(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - адреса из конфига
        return json.loads(response.read())


def _nvidia_smi() -> str | None:
    """``"<total>, <used>"`` в MiB для карты обучения; None, если nvidia-smi не сработал.

    nvidia-smi не смотрит на CUDA_VISIBLE_DEVICES, поэтому первая видимая карта передаётся явно.
    """
    device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    query = ["nvidia-smi", "-i", device, "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(query, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else None
