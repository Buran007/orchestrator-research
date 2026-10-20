"""Конфиг эксперимента (pydantic v2) из TOML с наследованием через ``include``.

Дочерний файл глубоко сливается поверх родителя, поэтому эксперимент пишет только то, что меняет.
Секции строгие (неизвестный ключ — ошибка) и неизменяемые. Относительные пути — от корня проекта
(каталог с ``pyproject.toml``); при сериализации пути снова относительные, поэтому ``config_hash``
одинаков на любой машине. Вход — ``load_config``.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PrivateAttr,
    SerializationInfo,
    StringConstraints,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .limits import TaskLimits

_NAME = r"^[a-z0-9][a-z0-9_\-]*$"
Alias = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
ModelId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# Индекс построен одной конкретной моделью эмбеддингов; запросы должны кодироваться ею же.
EMBEDDING_MODEL_BY_VARIANT: dict[str, str] = {
    "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-embedding-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-embedding-8b": "Qwen/Qwen3-Embedding-8B",
}


def _model_name(model_id: str) -> str:
    """Имя без префикса провайдера, в нижнем регистре: ``qwen/qwen3-embedding-8b`` == ``Qwen/Qwen3-Embedding-8B``."""
    return model_id.rsplit("/", 1)[-1].lower()


def find_project_root(start: Path) -> Path:
    """Ближайший к ``start`` (включительно) каталог с ``pyproject.toml``."""
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError(f"no pyproject.toml found in or above {start}; cannot resolve relative config paths")


def _resolve_project_path(value: Path, info: ValidationInfo) -> Path:
    """Раскрыть ``~``; относительный путь — от корня проекта из контекста загрузки."""
    context = info.context or {}
    root = Path(context["project_root"]) if "project_root" in context else find_project_root(Path(__file__).parent)
    expanded = value.expanduser()
    return (expanded if expanded.is_absolute() else root / expanded).resolve()


def _portable_path(value: Path, info: SerializationInfo) -> str:
    """Путь относительно корня, иначе относительно ``~``, иначе абсолютный — для одинакового хэша везде."""
    root = (info.context or {}).get("project_root")
    if root is None:
        return str(value)
    if value.is_relative_to(Path(root)):
        return value.relative_to(root).as_posix()
    if value.is_relative_to(Path.home()):
        return "~/" + value.relative_to(Path.home()).as_posix()
    return str(value)


ProjectPath = Annotated[
    Path,
    AfterValidator(_resolve_project_path),
    PlainSerializer(_portable_path, when_used="json"),
]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ExperimentMeta(_Section):
    """[experiment]"""

    name: str = Field(pattern=_NAME)
    strategy: str = "swarm"


class OrchestratorConfig(_Section):
    """[orchestrator] обучаемая модель; при обучении её обслуживает vLLM."""

    model_path: str = "Qwen/Qwen3.5-4B"
    eval_base_url: str = "http://127.0.0.1:18000/v1"
    eval_model: str = "orchestrator"
    api_key_env: str = ""
    temperature: float = 1.0
    eval_temperature: float = 0.0
    top_p: float = 1.0
    enable_thinking: bool = True
    timeout_seconds: float = Field(300.0, gt=0)
    max_retries: int = Field(3, ge=0)


class SubagentsConfig(_Section):
    """[subagents] необучаемые субагенты через OpenAI-совместимый API; ``models``: алиас → id модели."""

    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    prompts_path: ProjectPath = Path("configs/subagent_prompts.toml")
    models: dict[Alias, ModelId] = Field(min_length=1, description="alias -> provider model id")
    temperature: float = 0.0
    top_p: float = 1.0
    timeout_seconds: float = Field(180.0, gt=0)
    max_retries: int = Field(4, ge=0)
    max_concurrency: int = Field(64, gt=0)
    return_max_chars: int = Field(2000, gt=0)
    max_context_tokens: int | None = Field(None, gt=0)  # лимит промпта запроса субагента; None — решает провайдер
    extra_body: dict[str, Any] = Field(default_factory=dict)


class JudgeConfig(_Section):
    """[judge] LLM-судья, оценивает ответ."""

    base_url: str = "https://openrouter.ai/api/v1"
    model: str = Field(min_length=1)
    api_key_env: str = "OPENROUTER_API_KEY"
    temperature: float = 0.0
    max_tokens: int = Field(2048, gt=0)
    timeout_seconds: float = Field(120.0, gt=0)
    max_retries: int = Field(5, ge=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)


class EmbeddingConfig(_Section):
    """[embedding] эмбеддинги запросов; модель должна совпадать с той, которой построен индекс."""

    base_url: str = "http://127.0.0.1:18003/v1"
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    api_key_env: str = ""
    timeout_seconds: float = Field(60.0, gt=0)
    max_retries: int = Field(3, ge=0)


class RetrievalConfig(_Section):
    """[retrieval] готовый индекс BrowseComp-Plus и длина сниппетов."""

    server_url: str = "http://127.0.0.1:18100"
    timeout_seconds: float = Field(30.0, gt=0)
    max_retries: int = Field(3, ge=0)
    index_repo_id: str = "Tevatron/browsecomp-plus-indexes"
    index_revision: str = "main"
    index_variant: str = "qwen3-embedding-0.6b"
    index_path: ProjectPath = Path("~/data/browsecomp_index/qwen3-embedding-0.6b")
    corpus_repo_id: str = "Tevatron/browsecomp-plus-corpus"
    corpus_revision: str = "main"
    top_k: int = Field(5, gt=0)
    snippet_tokens: int = Field(512, gt=0)
    snippet_tokenizer: str = "Qwen/Qwen3-0.6B"

    @field_validator("index_variant")
    @classmethod
    def _check_variant(cls, value: str) -> str:
        if value not in EMBEDDING_MODEL_BY_VARIANT:
            allowed = ", ".join(sorted(EMBEDDING_MODEL_BY_VARIANT))
            raise ValueError(f"unsupported retrieval.index_variant {value!r}; allowed: {allowed}")
        return value


class BenchmarkConfig(_Section):
    """[benchmark] датасет; train и отложенные строки не пересекаются."""

    repo_id: str = "Tevatron/browsecomp-plus"
    revision: str = "main"
    expected_rows: int = Field(830, gt=0)
    data_dir: ProjectPath = Path("~/data/orchestrator_research/browsecomp")


class RewardConfig(_Section):
    """[reward] бонусы к награде: делегирование субагентам и доведение их заданий до ответа (см. reward.py)."""

    lambda_parallel: float = Field(0.1, ge=0)
    lambda_finish: float = Field(0.05, ge=0)
    anneal_steps: int = Field(7, gt=0)
    anneal_mode: Literal["linear", "cosine"] = "linear"
    r_parallel_mode: Literal["fraction_of_stages", "count_normalised", "cohort_size_avg"] = "fraction_of_stages"
    max_subagents_init: int = Field(8, gt=0)


class TrainingConfig(_Section):
    """[training] запуск verl + Agent Lightning; любой ключ verl можно задать напрямую в ``extra_overrides``."""

    total_steps: int = Field(100, gt=0)
    lr: float = Field(5e-6, gt=0)
    rollout_n: int = Field(4, ge=2)
    train_batch_size: int = Field(16, gt=0)
    ppo_mini_batch_size: int = Field(8, gt=0)
    max_prompt_tokens: int = Field(4096, gt=0)
    max_context_tokens: int = Field(24576, gt=0)
    rollout_gpu_memory_utilization: float = Field(0.45, gt=0, lt=1)
    val_split: Literal["val_test", "val_full"] = "val_test"
    val_before_train: bool = True
    test_freq: int = 10
    save_freq: int = 10
    k25_alpha: float = Field(-0.2, lt=0)
    k25_beta: float = Field(0.2, gt=0)
    k25_tau: float = Field(1e-3, ge=0)
    loggers: list[str] = Field(default_factory=lambda: ["console"])
    project: str = "orchestrator-research"
    agent_concurrency: int = Field(64, gt=0)
    rollout_timeout_seconds: int = Field(1800, gt=0)
    checkpoint_dir: ProjectPath = Path("artifacts/checkpoints")
    # GPU для обучения: FSDP делит между ними веса, градиенты и оптимизатор; vLLM — по копии на GPU
    n_gpus: int = Field(1, ge=1)
    # 0 — полное обучение; > 0 — LoRA-адаптеры этого ранга (verl actor_rollout_ref.model.lora_*)
    lora_rank: int = Field(0, ge=0)
    lora_alpha: int = Field(32, gt=0)
    lora_target_modules: str | list[str] = "all-linear"
    lora_exclude_modules: str | None = None
    # > 0: процесс обучения сразу занимает столько GB на карту и не отдаёт их (training/gpu_reserve.py), чтобы
    # соседи по общей карте не заняли память между фазами. Ставить пик обучения + запас.
    gpu_reserve_gb: float = Field(0, ge=0)
    extra_overrides: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _mini_batch_divides_batch(self) -> TrainingConfig:
        if self.train_batch_size % self.ppo_mini_batch_size != 0:
            raise ValueError(
                f"training.ppo_mini_batch_size ({self.ppo_mini_batch_size}) must divide "
                f"training.train_batch_size ({self.train_batch_size}); pick a divisor, e.g. 8 for 16."
            )
        # verl делит каждый (мини-)батч роллаутов поровну между GPU.
        for name, size in (("train_batch_size", self.train_batch_size), ("ppo_mini_batch_size", self.ppo_mini_batch_size)):
            if (size * self.rollout_n) % self.n_gpus != 0:
                raise ValueError(
                    f"training.{name} x rollout_n = {size * self.rollout_n} must be divisible by training.n_gpus ({self.n_gpus})"
                )
        return self


class ExperimentConfig(_Section):
    """Весь файл эксперимента."""

    experiment: ExperimentMeta
    limits: TaskLimits = Field(default_factory=TaskLimits)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    subagents: SubagentsConfig
    judge: JudgeConfig
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    _project_root: Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _cross_field_checks(self) -> ExperimentConfig:
        expected = EMBEDDING_MODEL_BY_VARIANT[self.retrieval.index_variant]
        if _model_name(self.embedding.model) != _model_name(expected):
            raise ValueError(
                f"embedding.model={self.embedding.model!r} does not match retrieval.index_variant="
                f"{self.retrieval.index_variant!r}, whose prebuilt index needs {expected!r}. "
                f"Fix: set [embedding] model = {expected!r}, or change index_variant to match the embedding model."
            )

        budget = self.limits.orchestrator_tokens_per_task
        if budget is None:
            budget = self.limits.orchestrator_tokens_per_turn * self.limits.max_turns
        needed = self.training.max_prompt_tokens + budget
        if self.training.max_context_tokens <= needed:
            raise ValueError(
                f"training.max_context_tokens={self.training.max_context_tokens} must be greater than "
                f"training.max_prompt_tokens + orchestrator generation budget = "
                f"{self.training.max_prompt_tokens} + {budget} = {needed}. "
                f"Fix: raise training.max_context_tokens to at least {needed + 1}, or lower "
                f"training.max_prompt_tokens, or lower limits.orchestrator_tokens_per_task "
                f"(unset, it defaults to orchestrator_tokens_per_turn * max_turns)."
            )
        return self

    def serializable(self) -> dict[str, Any]:
        """Конфиг для JSON; пути по возможности относительно корня проекта."""
        root = self._project_root or find_project_root(Path(__file__).parent)
        return self.model_dump(mode="json", context={"project_root": root})

    def config_hash(self) -> str:
        """SHA-256 канонического JSON ``serializable()``."""
        canonical = json.dumps(self.serializable(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    root = find_project_root(config_path.parent)
    raw = _load_with_include(config_path)
    try:
        config = ExperimentConfig.model_validate(raw, context={"project_root": root})
    except ValidationError as exc:
        raise ValueError(f"invalid experiment config {config_path}:\n{exc}") from exc
    config._project_root = root
    return config


def _load_with_include(path: Path, chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    if path in chain:
        cycle = " -> ".join(str(p) for p in (*chain, path))
        raise ValueError(f"config include cycle: {cycle}")
    with path.open("rb") as stream:
        child = tomllib.load(stream)
    include = child.pop("include", None)
    if include is None:
        return child
    parent = _load_with_include((path.parent / str(include)).resolve(), (*chain, path))
    return _deep_merge(parent, child)


# Таблицы, которые дочерний файл ЗАМЕНЯЕТ, а не сливает: [subagents.models] — это набор вариантов выбора,
# и слияние молча оставило бы алиасы родителя.
_REPLACED_TABLES = {("subagents", "models")}


def _deep_merge(parent: Mapping[str, Any], child: Mapping[str, Any], path: tuple[str, ...] = ()) -> dict[str, Any]:
    merged: dict[str, Any] = dict(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict) and (*path, key) not in _REPLACED_TABLES:
            merged[key] = _deep_merge(merged[key], value, (*path, key))
        else:
            merged[key] = value
    return merged


# --- промпты ролей субагентов (TOML) ---


def load_role_prompts(path: str | Path) -> dict[str, str]:
    """``{роль: промпт}``; набор ролей — какой объявлен в файле, но не пустой."""
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    roles = raw.get("roles")
    if not isinstance(roles, dict) or not roles:
        raise ValueError(f"missing or empty [roles] table in {path}")
    prompts: dict[str, str] = {}
    for name, value in roles.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"role {name!r} in {path} must have a non-empty prompt string")
        prompts[str(name)] = value.strip()
    return prompts
