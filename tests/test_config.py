"""Загрузка конфигурации: include с глубоким слиянием, неизвестные ключи, перекрёстные проверки, хеш."""

import shutil
from pathlib import Path

import pytest

from orchestrator_research.config import load_config
from orchestrator_research.limits import TaskLimits

ROOT = Path(__file__).parents[1]
EXPERIMENT = ROOT / "experiments/template/config.toml"


def _make_project(directory: Path, experiment_text: str | None = None, base_text: str | None = None) -> Path:
    """Временный проект с base, промптами и экспериментом; возвращает путь к config.toml эксперимента."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pyproject.toml").write_text('[project]\nname = "tmp"\n', encoding="utf-8")
    (directory / "configs").mkdir()
    (directory / "configs/base.toml").write_text(
        base_text if base_text is not None else (ROOT / "configs/base.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    shutil.copy(ROOT / "configs/subagent_prompts.toml", directory / "configs/subagent_prompts.toml")
    exp_dir = directory / "experiments/template"
    exp_dir.mkdir(parents=True)
    path = exp_dir / "config.toml"
    path.write_text(
        experiment_text if experiment_text is not None else EXPERIMENT.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return path


def _experiment_text() -> str:
    return EXPERIMENT.read_text(encoding="utf-8")


def test_experiment_001_loads():
    assert load_config(EXPERIMENT).experiment.strategy == "swarm"


def test_experiment_000_smoke_loads():
    cfg = load_config(ROOT / "experiments/000_smoke_local/config.toml")
    assert cfg.experiment.name == "000_smoke_local"
    # путь flash-attn, без прохода энтропии
    assert cfg.training.extra_overrides["actor_rollout_ref.actor.calculate_entropy"] is False


def test_include_deep_merges_child_over_parent(tmp_path):
    text = _experiment_text().replace("max_turns = 15", "max_turns = 3")
    cfg = load_config(_make_project(tmp_path, text))
    assert cfg.limits.max_turns == 3  # дочерний перекрывает
    assert cfg.limits.orchestrator_tokens_per_turn == 2048  # из base
    assert cfg.subagents.models["large"] == "qwen/qwen3.5-9b"  # вложенная таблица унаследована


def test_limits_section_is_the_task_limits_dataclass():
    assert isinstance(load_config(EXPERIMENT).limits, TaskLimits)


@pytest.mark.parametrize(
    ("needle", "replacement", "message"),
    [
        ("[limits]\n", "[limits]\ntypo = 1\n", "typo"),
        ("[experiment]\n", "[experiment]\nseed = 1\n", "seed"),
    ],
)
def test_unknown_keys_are_named_at_any_level(tmp_path, needle, replacement, message):
    text = _experiment_text().replace(needle, replacement, 1)
    with pytest.raises(ValueError, match=message):
        load_config(_make_project(tmp_path, text))


def test_unknown_top_level_key_is_named(tmp_path):
    text = "bogus_section_key = 1\n" + _experiment_text()
    with pytest.raises(ValueError, match="bogus_section_key"):
        load_config(_make_project(tmp_path, text))


def test_invalid_limits_value_is_rejected_by_task_limits(tmp_path):
    text = _experiment_text().replace("final_answer_reserve = 512", "final_answer_reserve = 9000")
    with pytest.raises(ValueError, match="final_answer_reserve"):
        load_config(_make_project(tmp_path, text))


def test_bad_subagent_alias_is_rejected(tmp_path):
    text = _experiment_text() + '\n[subagents.models]\n"Bad-Alias" = "x/y"\n'
    with pytest.raises(ValueError, match="Bad-Alias"):
        load_config(_make_project(tmp_path, text))


def test_ppo_mini_batch_must_divide_train_batch(tmp_path):
    text = _experiment_text().replace("ppo_mini_batch_size = 8", "ppo_mini_batch_size = 3")
    with pytest.raises(ValueError, match="ppo_mini_batch_size"):
        load_config(_make_project(tmp_path, text))


def test_context_length_cross_check(tmp_path):
    text = _experiment_text().replace("max_context_tokens = 16384", "max_context_tokens = 12288")
    with pytest.raises(ValueError, match=r"max_context_tokens.*Fix"):
        load_config(_make_project(tmp_path, text))


def test_embedding_model_must_match_index_variant(tmp_path):
    base = (ROOT / "configs/base.toml").read_text(encoding="utf-8")
    bad_base = base.replace('model = "qwen/qwen3-embedding-8b"', 'model = "qwen/qwen3-embedding-4b"')
    with pytest.raises(ValueError, match="does not match retrieval.index_variant"):
        load_config(_make_project(tmp_path, base_text=bad_base))


def test_openrouter_embedding_id_matches_index_variant(tmp_path):
    # В base.toml id OpenRouter "qwen/qwen3-embedding-8b", индекс собран под "Qwen/Qwen3-Embedding-8B".
    cfg = load_config(EXPERIMENT)
    assert cfg.embedding.model == "qwen/qwen3-embedding-8b"
    assert cfg.retrieval.index_variant == "qwen3-embedding-8b"
    text = _experiment_text()
    assert load_config(_make_project(tmp_path, text)).embedding.model == "qwen/qwen3-embedding-8b"


def test_embedding_size_mismatch_is_rejected_even_with_provider_prefix(tmp_path):
    base = (ROOT / "configs/base.toml").read_text(encoding="utf-8")
    # Модель 8B против индекса 4B: префикс провайдера тот же, размер другой.
    bad_base = base.replace('index_variant = "qwen3-embedding-8b"', 'index_variant = "qwen3-embedding-4b"')
    with pytest.raises(ValueError, match="does not match retrieval.index_variant"):
        load_config(_make_project(tmp_path, base_text=bad_base))


def test_paths_resolve_against_project_root():
    cfg = load_config(EXPERIMENT)
    assert cfg.subagents.prompts_path == (ROOT / "configs/subagent_prompts.toml").resolve()
    assert cfg.training.checkpoint_dir == (ROOT / "artifacts/checkpoints").resolve()
    assert cfg.retrieval.index_path == Path.home() / "data/browsecomp_index/qwen3-embedding-8b"
    assert cfg.benchmark.data_dir.is_absolute()


def test_config_hash_is_stable_and_sensitive(tmp_path):
    first = load_config(EXPERIMENT).config_hash()
    assert first == load_config(EXPERIMENT).config_hash()
    assert len(first) == 64
    changed = load_config(_make_project(tmp_path / "changed", _experiment_text().replace("lr = 5e-6", "lr = 1e-5")))
    assert changed.config_hash() != first


def test_config_hash_does_not_depend_on_where_the_project_lives(tmp_path):
    first = load_config(_make_project(tmp_path / "one")).config_hash()
    second = load_config(_make_project(tmp_path / "two" / "nested")).config_hash()
    assert first == second


def test_serialised_paths_are_project_relative():
    payload = load_config(EXPERIMENT).serializable()
    assert payload["subagents"]["prompts_path"] == "configs/subagent_prompts.toml"
    assert payload["training"]["checkpoint_dir"] == "artifacts/checkpoints"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


def test_child_subagent_models_replace_the_parent_pool(tmp_path):
    text = _experiment_text() + '\n[subagents.models]\nlarge = "qwen/qwen3.5-9b"\n'
    cfg = load_config(_make_project(tmp_path, text))
    assert cfg.subagents.models == {"large": "qwen/qwen3.5-9b"}  # small/medium из base не сливаются, а пропадают
