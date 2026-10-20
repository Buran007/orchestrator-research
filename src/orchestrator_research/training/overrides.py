"""Конфиг experiment → плоский словарь переопределений verl / Agent Lightning (без импорта verl и torch).

``build_trainer_config`` возвращает ``{"dotted.key": value}``; ``train.py`` пишет его в ``trainer_overrides.json``
и применяет к базовому конфигу ``agentlightning.verl``. Значения — из ``ExperimentConfig``, литералы —
фиксированные решения ADR 0005. Имена ключей — по примеру Search-R1 из Agent Lightning v1.0.2 и
``ppo_trainer.yaml`` verl v0.8.0. Здесь же ``StepCounter``: номер шага обучения по номеру роллаута (для ``hooks.py``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from orchestrator_research.algorithm import K25Params
from orchestrator_research.config import ExperimentConfig
from orchestrator_research.training.gpu_reserve import RESERVE_ENV

ENV_MAP: dict[str, str] = {
    "QUESTION_ID": "input.question_id",
    "QUESTION": "input.question",
    "ANSWERS": "input.answers",
    "SPLIT": "input.split",
    "TRAIN_STEP": "input.train_step",
    # train.py дописывает это в каждую строку данных: agl-controller — отдельный процесс и окружения тренера не видит.
    "ORCH_RESEARCH_CONFIG": "input.config_path",
    "ORCH_RESEARCH_RUN_DIR": "input.run_dir",
}
ROLLOUTS_PER_STEP_ENV = "ORCH_RESEARCH_TRAIN_ROLLOUTS_PER_STEP"
_RAY_ENV_VARS = "ray_kwargs.ray_init.runtime_env.env_vars"


def build_trainer_config(cfg: ExperimentConfig, *, run_dir: Path, smoke: bool = False) -> dict[str, Any]:
    """Переопределения для прогона; ``training.extra_overrides`` побеждают всё остальное."""
    from .k25_loss import ENV_VAR, LOSS_MODE, to_env  # лениво: модуль лосса импортирует torch

    t = cfg.training
    o = cfg.orchestrator
    max_context = t.max_context_tokens
    max_response = max_context - t.max_prompt_tokens

    train_batch = t.train_batch_size
    rollout_n = t.rollout_n
    ppo_mini = t.ppo_mini_batch_size
    total_steps = t.total_steps
    test_freq = t.test_freq
    save_freq = t.save_freq
    val_before_train = t.val_before_train
    loggers = list(t.loggers)
    if smoke:
        train_batch = 2
        rollout_n = 2
        ppo_mini = 2
        total_steps = 2
        test_freq = -1
        save_freq = -1
        val_before_train = False
        loggers = ["console"]

    params = K25Params(alpha=t.k25_alpha, beta=t.k25_beta, tau=t.k25_tau)
    overrides: dict[str, Any] = {
        "algorithm.adv_estimator": "grpo",
        "algorithm.norm_adv_by_std_in_grpo": False,
        "algorithm.use_kl_in_reward": False,
        "algorithm.enable_rollout_level_advantage": True,
        "data.train_batch_size": train_batch,
        "data.max_prompt_length": t.max_prompt_tokens,
        "data.max_response_length": max_response,
        "data.truncation": "error",
        "actor_rollout_ref.model.path": o.model_path,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.model.use_fused_kernels": True,
        "actor_rollout_ref.actor.optim.lr": t.lr,
        "actor_rollout_ref.actor.ppo_mini_batch_size": ppo_mini,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": max_context,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.entropy_coeff": 0,
        "actor_rollout_ref.actor.policy_loss.loss_mode": LOSS_MODE,
        "actor_rollout_ref.actor.strategy": "fsdp2",
        # Не проверено по исходникам verl: ключи взяты из примера Search-R1.
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.n": rollout_n,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": t.rollout_gpu_memory_utilization,
        "actor_rollout_ref.rollout.max_model_len": max_context,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": True,
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": max_context,
        "actor_rollout_ref.rollout.free_cache_engine": True,
        "actor_rollout_ref.rollout.temperature": o.temperature,
        "actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice": True,
        "actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser": "qwen3_coder",
        "actor_rollout_ref.rollout.val_kwargs.temperature": o.eval_temperature,
        "trainer.n_gpus_per_node": t.n_gpus,
        "trainer.nnodes": 1,
        "trainer.total_training_steps": total_steps,
        # Прогон ограничивают шаги; эпох ставим с запасом, чтобы они не кончились раньше.
        "trainer.total_epochs": 1000,
        # При продолжении счётчик шагов начнётся с 0, и затухание бонусов награды начнётся заново.
        "trainer.resume_mode": "disable",
        "trainer.test_freq": test_freq,
        "trainer.save_freq": save_freq,
        "trainer.val_before_train": val_before_train,
        "trainer.critic_warmup": 0,
        "trainer.logger": loggers,
        "trainer.project_name": t.project,
        "trainer.experiment_name": f"{cfg.experiment.name}-{run_dir.name}",
        "trainer.default_local_dir": str(t.checkpoint_dir / cfg.experiment.name / run_dir.name),
        "agentlightning.rollout_timeout_seconds": t.rollout_timeout_seconds,
        "agentlightning.reward_fillna_value": 0.0,
        "agentlightning.hooks": str(Path(__file__).resolve().parent / "hooks.py"),
        "agentlightning.local.agent_class": "orchestrator_research.agent:OrchestratorAgent",
        **{f"agentlightning.local.env_map.{name}": path for name, path in ENV_MAP.items()},
        "agentlightning.trace_aggregator.level": "trajectory",
        "agentlightning.trace_aggregator.trajectory_max_prompt_length": t.max_prompt_tokens,
        "agentlightning.trace_aggregator.trajectory_max_response_length": max_response,
        "agentlightning.async_rollout.enabled": False,
        "ray_kwargs.ray_init.runtime_env.worker_process_setup_hook": (
            "orchestrator_research.training.k25_loss.setup_worker"
        ),
        f"{_RAY_ENV_VARS}.{ENV_VAR}": to_env(params),
        f"{_RAY_ENV_VARS}.{ROLLOUTS_PER_STEP_ENV}": str(train_batch * rollout_n),
        # Наш setup-хук импортирует verl, а тот зовёт torch.cuda.is_available() до того, как Ray назначит воркеру
        # GPU. Проверка через cuInit запоминает список карт, и все ранги попадали на одну («Duplicate GPU detected»).
        # Проверка через NVML CUDA не инициализирует.
        f"{_RAY_ENV_VARS}.PYTORCH_NVML_BASED_CUDA_CHECK": "1",
    }
    if t.gpu_reserve_gb > 0:
        overrides[f"{_RAY_ENV_VARS}.{RESERVE_ENV}"] = str(t.gpu_reserve_gb)
    if t.lora_rank > 0:
        # verl v0.8.0 (model.lora_* и гайд по LoRA): vLLM сам грузит базовые веса (load_format=safetensors),
        # синхронизируются только адаптеры. layered_summon выключен: с fsdp2 и Qwen3.5 он отдавал vLLM пустой адаптер.
        overrides |= {
            "actor_rollout_ref.model.lora_rank": t.lora_rank,
            "actor_rollout_ref.model.lora_alpha": t.lora_alpha,
            "actor_rollout_ref.model.target_modules": t.lora_target_modules,
            "actor_rollout_ref.rollout.load_format": "safetensors",
        }
        if t.lora_exclude_modules:
            overrides["actor_rollout_ref.model.exclude_modules"] = t.lora_exclude_modules
    return {**overrides, **cfg.training.extra_overrides}


def with_agl_endpoint(overrides: Mapping[str, Any], *, base_url: str, key: str) -> dict[str, Any]:
    """Копия ``overrides`` с адресом agl-server и ключом.

    Ключ попадает только в эту копию в памяти — в ``trainer_overrides.json`` его нет.
    """
    if not base_url.strip():
        raise ValueError("AGL_SERVER_URL is empty; start agl-server first and export its URL")
    if not key.strip():
        raise ValueError("AGL_KEY is empty; an empty key disables agl-server authentication")
    return {**overrides, "agentlightning.agl_base_url": base_url, "agentlightning.agl_key": key}


def assert_lengths_consistent(cfg: ExperimentConfig, overrides: Mapping[str, Any]) -> None:
    """Проверить: промпт + ответ = длина модели = бюджет токенов на GPU; иначе ``ValueError``.

    Проверяется итоговый словарь (после ``extra_overrides``), чтобы ручные ключи не сломали это незаметно.
    """
    prompt = overrides["data.max_prompt_length"]
    response = overrides["data.max_response_length"]
    model_len = overrides["actor_rollout_ref.rollout.max_model_len"]
    per_gpu = overrides["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"]
    if prompt + response != model_len or model_len != per_gpu:
        raise ValueError(
            "length settings disagree: data.max_prompt_length + data.max_response_length = "
            f"{prompt} + {response} = {prompt + response}; rollout.max_model_len = {model_len}; "
            f"actor.ppo_max_token_len_per_gpu = {per_gpu}. All three must be equal."
        )
    limits = cfg.limits
    budget = limits.orchestrator_tokens_per_task
    if budget is None:
        budget = limits.orchestrator_tokens_per_turn * limits.max_turns + limits.final_answer_reserve
    if response < budget:
        raise ValueError(
            f"data.max_response_length={response} is smaller than the orchestrator token budget {budget} "
            "(limits.orchestrator_tokens_per_task, or orchestrator_tokens_per_turn * max_turns + "
            "final_answer_reserve when unset); the trajectory could not hold it."
        )


class StepCounter:
    """Номер шага по номеру роллаута: каждые ``rollouts_per_step`` обучающих роллаутов — один шаг.

    Роллауты валидации получают -1 и счётчик не двигают.
    """

    def __init__(self, rollouts_per_step: int) -> None:
        if not isinstance(rollouts_per_step, int) or rollouts_per_step <= 0:
            raise ValueError(f"rollouts_per_step must be a positive int; got {rollouts_per_step!r}")
        self._rollouts_per_step = rollouts_per_step
        self._train_count = 0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> StepCounter:
        """Из ``ROLLOUTS_PER_STEP_ENV`` (её ставит train.py); обязательна."""
        raw = environ.get(ROLLOUTS_PER_STEP_ENV)
        if raw is None or not raw.strip():
            raise ValueError(f"{ROLLOUTS_PER_STEP_ENV} is not set; train.py must export it before the trainer starts")
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{ROLLOUTS_PER_STEP_ENV} must be an integer; got {raw!r}") from exc
        return cls(value)

    @property
    def train_rollouts_seen(self) -> int:
        return self._train_count

    def step_for(self, is_train: bool) -> int:
        """Шаг для следующего роллаута; обучающие роллауты двигают счётчик."""
        if not is_train:
            return -1
        step = self._train_count // self._rollouts_per_step
        self._train_count += 1
        return step
