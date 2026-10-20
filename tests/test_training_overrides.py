"""Чистый сборщик оверрайдов verl, счётчик шагов ``StepCounter`` и dry run ``train.py``.

verl, agentlightning и omegaconf здесь не импортируются; константы K25 берутся из k25_loss (только torch).
"""

import dataclasses
import json
from pathlib import Path

import pytest

from orchestrator_research.config import ExperimentConfig, load_config
from orchestrator_research.training.k25_loss import ENV_VAR, LOSS_MODE, from_env
from orchestrator_research.training.overrides import (
    ROLLOUTS_PER_STEP_ENV,
    StepCounter,
    assert_lengths_consistent,
    build_trainer_config,
    with_agl_endpoint,
)
from orchestrator_research.training.train import main

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments" / "template" / "config.toml"
RUN_DIR = ROOT / "artifacts" / "runs" / "template" / "pytest-run"


@pytest.fixture(scope="module")
def cfg() -> ExperimentConfig:
    return load_config(CONFIG)


def _with_extra(cfg: ExperimentConfig, extra: dict) -> ExperimentConfig:
    training = cfg.training.model_copy(update={"extra_overrides": extra})
    return cfg.model_copy(update={"training": training})


def test_lengths_are_consistent(cfg):
    overrides = build_trainer_config(cfg, run_dir=RUN_DIR)
    assert_lengths_consistent(cfg, overrides)
    assert overrides["data.max_prompt_length"] == cfg.training.max_prompt_tokens
    assert overrides["data.max_response_length"] == cfg.training.max_context_tokens - cfg.training.max_prompt_tokens


def test_lengths_mismatch_is_rejected(cfg):
    bad = _with_extra(cfg, {"data.max_response_length": 5})
    overrides = build_trainer_config(bad, run_dir=RUN_DIR)
    with pytest.raises(ValueError, match="length settings disagree"):
        assert_lengths_consistent(bad, overrides)


def test_response_must_hold_orchestrator_budget(cfg):
    small = _with_extra(
        cfg,
        {
            "data.max_response_length": 100,
            "data.max_prompt_length": 24476,
            "actor_rollout_ref.rollout.max_model_len": 24576,
            "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 24576,
        },
    )
    overrides = build_trainer_config(small, run_dir=RUN_DIR)
    with pytest.raises(ValueError, match="orchestrator_tokens_per_task"):
        assert_lengths_consistent(small, overrides)


@pytest.mark.parametrize(("response", "fits"), [(1000, False), (1512, True)])
def test_unset_budget_counts_final_answer_reserve(cfg, response, fits):
    limits = dataclasses.replace(
        cfg.limits,
        orchestrator_tokens_per_task=None,
        orchestrator_tokens_per_turn=1000,
        max_turns=1,
        final_answer_reserve=512,
    )
    prompt = cfg.training.max_prompt_tokens
    extra = {
        "data.max_response_length": response,
        "actor_rollout_ref.rollout.max_model_len": prompt + response,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": prompt + response,
    }
    base = _with_extra(cfg, extra).model_copy(update={"limits": limits})
    overrides = build_trainer_config(base, run_dir=RUN_DIR)
    if fits:
        assert_lengths_consistent(base, overrides)
    else:
        with pytest.raises(ValueError, match="orchestrator_tokens_per_task"):
            assert_lengths_consistent(base, overrides)


def test_algorithm_and_loss_settings(cfg):
    o = build_trainer_config(cfg, run_dir=RUN_DIR)
    assert o["algorithm.adv_estimator"] == "grpo"
    assert o["algorithm.norm_adv_by_std_in_grpo"] is False
    assert o["algorithm.use_kl_in_reward"] is False
    assert o["algorithm.enable_rollout_level_advantage"] is True
    assert o["actor_rollout_ref.actor.policy_loss.loss_mode"] == LOSS_MODE == "k25_token_clip"
    assert o["actor_rollout_ref.actor.use_kl_loss"] is False
    assert not any(key.startswith("actor_rollout_ref.ref.") for key in o)


def test_single_gpu_and_offload(cfg):
    o = build_trainer_config(cfg, run_dir=RUN_DIR)
    assert o["trainer.n_gpus_per_node"] == 1
    assert o["trainer.nnodes"] == 1
    assert o["actor_rollout_ref.actor.fsdp_config.param_offload"] is True
    assert o["actor_rollout_ref.actor.fsdp_config.optimizer_offload"] is True


def test_tool_parser_and_rollout_values(cfg):
    o = build_trainer_config(cfg, run_dir=RUN_DIR)
    assert o["actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice"] is True
    assert o["actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser"] == "qwen3_coder"
    assert o["actor_rollout_ref.rollout.n"] == cfg.training.rollout_n
    assert o["actor_rollout_ref.rollout.val_kwargs.temperature"] == cfg.orchestrator.eval_temperature


def test_env_vars_carry_valid_k25_json_and_rollouts_per_step(cfg):
    o = build_trainer_config(cfg, run_dir=RUN_DIR)
    env_prefix = "ray_kwargs.ray_init.runtime_env.env_vars."
    assert json.loads(o[env_prefix + ENV_VAR]) == {"alpha": -0.2, "beta": 0.2, "tau": 1e-3}
    assert from_env({ENV_VAR: o[env_prefix + ENV_VAR]}).tau == 1e-3
    expected = cfg.training.train_batch_size * cfg.training.rollout_n
    assert o[env_prefix + ROLLOUTS_PER_STEP_ENV] == str(expected)


def test_agentlightning_block(cfg):
    o = build_trainer_config(cfg, run_dir=RUN_DIR)
    assert o["agentlightning.local.agent_class"] == "orchestrator_research.agent:OrchestratorAgent"
    assert o["agentlightning.local.env_map.TRAIN_STEP"] == "input.train_step"
    assert o["agentlightning.hooks"].endswith("hooks.py")
    assert Path(o["agentlightning.hooks"]).is_file()
    assert o["agentlightning.trace_aggregator.level"] == "trajectory"
    assert o["agentlightning.async_rollout.enabled"] is False


def test_extra_overrides_win_over_derived_values(cfg):
    extra = {
        "actor_rollout_ref.actor.optim.lr": 1e-7,
        "trainer.total_epochs": 3,
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
    }
    o = build_trainer_config(_with_extra(cfg, extra), run_dir=RUN_DIR)
    assert o["actor_rollout_ref.actor.optim.lr"] == 1e-7
    assert o["trainer.total_epochs"] == 3
    assert o["actor_rollout_ref.actor.fsdp_config.param_offload"] is False
    assert o["actor_rollout_ref.actor.fsdp_config.optimizer_offload"] is True


def test_resume_is_disabled(cfg):
    assert build_trainer_config(cfg, run_dir=RUN_DIR)["trainer.resume_mode"] == "disable"


def test_smoke_shrinks_run(cfg):
    o = build_trainer_config(cfg, run_dir=RUN_DIR, smoke=True)
    assert o["trainer.total_training_steps"] == 2
    assert o["data.train_batch_size"] == 2
    assert o["actor_rollout_ref.actor.ppo_mini_batch_size"] == 2
    assert o["actor_rollout_ref.rollout.n"] == 2
    assert o["trainer.test_freq"] == -1
    assert o["trainer.save_freq"] == -1
    assert o["trainer.val_before_train"] is False
    assert o["trainer.logger"] == ["console"]
    assert o[f"ray_kwargs.ray_init.runtime_env.env_vars.{ROLLOUTS_PER_STEP_ENV}"] == "4"
    assert_lengths_consistent(cfg, o)


def test_experiment_names_come_from_run_dir(cfg):
    o = build_trainer_config(cfg, run_dir=RUN_DIR)
    assert o["trainer.experiment_name"] == "template-pytest-run"
    assert o["trainer.default_local_dir"].replace("\\", "/").endswith("template/pytest-run")


def test_agl_endpoint_is_added_only_by_the_helper(cfg):
    base = build_trainer_config(cfg, run_dir=RUN_DIR)
    assert "agentlightning.agl_key" not in base
    with_key = with_agl_endpoint(base, base_url="http://127.0.0.1:8080", key="k-test")
    assert with_key["agentlightning.agl_base_url"] == "http://127.0.0.1:8080"
    assert with_key["agentlightning.agl_key"] == "k-test"
    assert "agentlightning.agl_key" not in base, "the helper must not mutate its input"


@pytest.mark.parametrize(
    ("url", "key", "message"), [(" ", "k", "AGL_SERVER_URL"), ("http://127.0.0.1:8080", "", "AGL_KEY")]
)
def test_agl_endpoint_rejects_empty_values(url, key, message):
    with pytest.raises(ValueError, match=message):
        with_agl_endpoint({}, base_url=url, key=key)


def test_step_counter_maps_rollouts_to_steps():
    counter = StepCounter(rollouts_per_step=3)
    assert [counter.step_for(is_train=True) for _ in range(6)] == [0, 0, 0, 1, 1, 1]


def test_step_counter_val_is_minus_one_and_does_not_advance():
    counter = StepCounter(rollouts_per_step=2)
    assert counter.step_for(is_train=False) == -1
    assert counter.step_for(is_train=True) == 0
    assert counter.step_for(is_train=False) == -1
    assert counter.train_rollouts_seen == 1


def test_step_counter_from_env_is_strict():
    with pytest.raises(ValueError, match="not set"):
        StepCounter.from_env({})
    with pytest.raises(ValueError, match="integer"):
        StepCounter.from_env({ROLLOUTS_PER_STEP_ENV: "x"})
    with pytest.raises(ValueError, match="positive"):
        StepCounter(0)
    counter = StepCounter.from_env({ROLLOUTS_PER_STEP_ENV: "2"})
    assert [counter.step_for(is_train=True) for _ in range(3)] == [0, 0, 1]


def test_dry_run_writes_nothing(capsys):
    artifacts = ROOT / "artifacts"
    before = sorted(str(p) for p in artifacts.rglob("*")) if artifacts.exists() else []
    code = main(["--config", str(CONFIG), "--run-id", "pytest-run", "--dry-run"])
    assert code == 0
    assert not RUN_DIR.exists()
    after = sorted(str(p) for p in artifacts.rglob("*")) if artifacts.exists() else []
    assert before == after
    assert 'algorithm.adv_estimator = "grpo"' in capsys.readouterr().out


def test_full_fine_tune_by_default_has_no_lora_keys(cfg):
    assert cfg.training.lora_rank == 0
    overrides = build_trainer_config(cfg, run_dir=RUN_DIR)
    assert not any("lora" in key or key.endswith("load_format") for key in overrides)


def test_lora_rank_switches_on_lora_keys_and_keeps_lengths(cfg):
    training = cfg.training.model_copy(update={"lora_rank": 64, "lora_alpha": 128, "lora_exclude_modules": ".*linear_attn.*", "lora_target_modules": ["q_proj", "v_proj"]})
    lora_cfg = cfg.model_copy(update={"training": training})
    overrides = build_trainer_config(lora_cfg, run_dir=RUN_DIR)
    assert overrides["actor_rollout_ref.model.lora_rank"] == 64
    assert overrides["actor_rollout_ref.model.lora_alpha"] == 128
    assert overrides["actor_rollout_ref.model.target_modules"] == ["q_proj", "v_proj"]
    assert "actor_rollout_ref.rollout.layered_summon" not in overrides
    assert overrides["actor_rollout_ref.model.exclude_modules"] == ".*linear_attn.*"
    assert overrides["actor_rollout_ref.rollout.load_format"] == "safetensors"
    assert_lengths_consistent(lora_cfg, overrides)


def test_n_gpus_reaches_verl_and_batches_must_split_evenly(cfg):
    training = cfg.training.model_copy(update={"n_gpus": 2})
    overrides = build_trainer_config(cfg.model_copy(update={"training": training}), run_dir=RUN_DIR)
    assert overrides["trainer.n_gpus_per_node"] == 2
    assert build_trainer_config(cfg, run_dir=RUN_DIR)["trainer.n_gpus_per_node"] == 1
    with pytest.raises(ValueError, match="divisible by training.n_gpus"):
        type(cfg.training).model_validate({**cfg.training.model_dump(), "n_gpus": 3})


def test_ray_workers_use_nvml_cuda_check(cfg):
    overrides = build_trainer_config(cfg, run_dir=RUN_DIR)
    assert overrides["ray_kwargs.ray_init.runtime_env.env_vars.PYTORCH_NVML_BASED_CUDA_CHECK"] == "1"


def test_gpu_reserve_env_only_when_set(cfg):
    key = "ray_kwargs.ray_init.runtime_env.env_vars.OR_GPU_RESERVE_GB"
    assert key not in build_trainer_config(cfg, run_dir=RUN_DIR)
    training = cfg.training.model_copy(update={"gpu_reserve_gb": 30.0})
    assert build_trainer_config(cfg.model_copy(update={"training": training}), run_dir=RUN_DIR)[key] == "30.0"
