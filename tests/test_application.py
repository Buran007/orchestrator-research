"""``doctor``: проверки окружения (ключи, модели, retrieval, эмбеддер, GPU) без сети; ключ нигде не печатается."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest

from orchestrator_research import application
from orchestrator_research.application import doctor, render_checks
from orchestrator_research.config import load_config

EXPERIMENT = Path(__file__).parents[1] / "experiments/template/config.toml"
SECRET = "sk-live-do-not-print-123"
REMOTE_URL = "https://openrouter.ai/api/v1"
LOCAL_URL = "http://127.0.0.1:8003/v1"


def _fetch_factory(model_ids, *, down=()):
    def fetch(url, timeout):
        if any(marker in url for marker in down):
            raise OSError("connection refused")
        if url.endswith("/healthz"):
            return {"status": "ok", "documents": 830, "vectors": 5}
        return {"data": [{"id": model_id} for model_id in model_ids]}

    return fetch


def _by_name(checks):
    return {check.name: check for check in checks}


def _fake_openai(monkeypatch, *, dimension=4096, error=None):
    """Подменить ``openai.OpenAI`` записывающим фейком; вернуть созданные клиенты и запросы эмбеддингов."""
    seen = {"clients": [], "requests": []}

    class FakeEmbeddings:
        def create(self, *, model, input):
            seen["requests"].append({"model": model, "input": input})
            if error is not None:
                raise error
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0] * dimension)])

    class FakeClient:
        def __init__(self, *, base_url, api_key, timeout, max_retries):
            seen["clients"].append({"base_url": base_url, "timeout": timeout, "max_retries": max_retries})
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    return seen


@pytest.fixture(autouse=True)
def no_network_openai(monkeypatch):
    """Фейковый OpenAI во всех тестах: проверка удалённого эмбеддера не должна уйти в сеть."""
    return _fake_openai(monkeypatch)


def _remote_embedding(cfg, **overrides):
    values = {
        "base_url": REMOTE_URL,
        "model": "qwen/qwen3-embedding-8b",
        "api_key_env": "OPENROUTER_API_KEY",
        **overrides,
    }
    return cfg.model_copy(update={"embedding": cfg.embedding.model_copy(update=values)})


def _local_embedding(cfg):
    return cfg.model_copy(update={"embedding": cfg.embedding.model_copy(update={"base_url": LOCAL_URL})})


def test_healthy_environment_has_no_failures_and_never_prints_the_key():
    cfg = load_config(EXPERIMENT)
    ids = set(cfg.subagents.models.values()) | {cfg.judge.model}
    checks = doctor(
        cfg,
        http_get=_fetch_factory(ids),
        gpu_query=lambda: "81920 MiB, 0 MiB",
        python_version=(3, 12),
        env={"OPENROUTER_API_KEY": SECRET},
    )
    assert not [check for check in checks if check.status == "FAIL" and check.name != "benchmark data"]
    assert _by_name(checks)["subagents API key"].status == "OK"
    assert SECRET not in render_checks(checks)


def test_missing_key_fails_and_names_only_the_variable():
    cfg = load_config(EXPERIMENT)
    checks = doctor(cfg, http_get=_fetch_factory([]), gpu_query=lambda: None, python_version=(3, 12), env={})
    failed = _by_name(checks)["judge API key"]
    assert failed.status == "FAIL"
    assert "OPENROUTER_API_KEY" in failed.detail
    assert "export OPENROUTER_API_KEY" in failed.hint


def test_unlisted_model_id_fails_the_models_check():
    cfg = load_config(EXPERIMENT)
    checks = doctor(
        cfg,
        http_get=_fetch_factory(["mistralai/ministral-3b-2512"]),
        gpu_query=lambda: None,
        python_version=(3, 12),
        env={"OPENROUTER_API_KEY": SECRET},
    )
    models = _by_name(checks)["subagent models"]
    assert models.status == "FAIL"
    assert "qwen/qwen3.5-9b" in models.detail


def test_unreachable_retrieval_server_fails_with_a_start_hint():
    cfg = load_config(EXPERIMENT)
    checks = doctor(
        cfg,
        http_get=_fetch_factory([], down=("healthz",)),
        gpu_query=lambda: None,
        python_version=(3, 12),
        env={"OPENROUTER_API_KEY": SECRET},
    )
    retrieval = _by_name(checks)["retrieval server"]
    assert retrieval.status == "FAIL"
    assert "serve" in retrieval.hint or "retrieval_server" in retrieval.hint


def test_missing_gpu_tool_and_old_python_are_warnings_not_failures():
    cfg = load_config(EXPERIMENT)
    checks = doctor(
        cfg,
        http_get=_fetch_factory([]),
        gpu_query=lambda: None,
        python_version=(3, 11),
        env={"OPENROUTER_API_KEY": SECRET},
    )
    named = _by_name(checks)
    assert named["GPU"].status == "WARN"
    assert named["python"].status == "WARN"


def test_both_api_key_checks_are_reported_when_unset():
    cfg = load_config(EXPERIMENT)
    checks = _by_name(doctor(cfg, http_get=_fetch_factory([]), gpu_query=lambda: None, python_version=(3, 12), env={}))
    assert checks["subagents API key"].status == "FAIL"
    assert checks["judge API key"].status == "FAIL"


def test_trailing_slash_in_retrieval_url_does_not_double_the_health_path():
    cfg = load_config(EXPERIMENT)
    cfg = cfg.model_copy(
        update={"retrieval": cfg.retrieval.model_copy(update={"server_url": "http://127.0.0.1:8100/"})}
    )
    seen = []

    def fetch(url, timeout):
        seen.append(url)
        return {"status": "ok"} if url.endswith("/healthz") else {"data": []}

    doctor(cfg, http_get=fetch, gpu_query=lambda: None, python_version=(3, 12), env={"OPENROUTER_API_KEY": SECRET})
    assert "http://127.0.0.1:8100/healthz" in seen


# --- эмбеддер: удалённый (один настоящий запрос эмбеддинга) ---


def test_remote_embedder_makes_one_ping_call_and_reports_ok(no_network_openai):
    cfg = _remote_embedding(load_config(EXPERIMENT))
    seen = []

    def fetch(url, timeout):
        seen.append(url)
        return {"data": []} if not url.endswith("/healthz") else {"status": "ok"}

    checks = _by_name(
        doctor(cfg, http_get=fetch, gpu_query=lambda: None, python_version=(3, 12), env={"OPENROUTER_API_KEY": SECRET})
    )
    assert checks["embedding API key"].status == "OK"
    assert checks["embedding call"].status == "OK"
    assert "4096-dim" in checks["embedding call"].detail
    assert no_network_openai["requests"] == [{"model": "qwen/qwen3-embedding-8b", "input": "ping"}]
    assert no_network_openai["clients"][0]["max_retries"] == 0
    assert "embedding server" not in checks  # у удалённого эмбеддера нет пробы /models (у субагентов и судьи есть)
    assert SECRET not in render_checks(list(checks.values()))


def test_remote_embedder_call_failure_fails_with_type_status_and_hint(monkeypatch):
    class FakeAPIError(Exception):
        status_code = 401

    _fake_openai(monkeypatch, error=FakeAPIError("Invalid key sk-live-do-not-print-123"))
    cfg = _remote_embedding(load_config(EXPERIMENT))
    checks = doctor(
        cfg,
        http_get=_fetch_factory([]),
        gpu_query=lambda: None,
        python_version=(3, 12),
        env={"OPENROUTER_API_KEY": SECRET},
    )
    call = _by_name(checks)["embedding call"]
    assert call.status == "FAIL"
    assert "FakeAPIError (HTTP 401)" in call.detail
    assert REMOTE_URL in call.hint
    assert "OPENROUTER_API_KEY" in call.hint
    # Сообщение провайдера может содержать ключ — в вывод оно не попадает.
    assert SECRET not in render_checks(checks)


def test_remote_embedder_without_key_fails_and_makes_no_call(no_network_openai):
    cfg = _remote_embedding(load_config(EXPERIMENT))
    checks = _by_name(doctor(cfg, http_get=_fetch_factory([]), gpu_query=lambda: None, python_version=(3, 12), env={}))
    key = checks["embedding API key"]
    assert key.status == "FAIL"
    assert "OPENROUTER_API_KEY is not set" in key.detail
    assert "export OPENROUTER_API_KEY" in key.hint
    assert "embedding call" not in checks
    assert no_network_openai["requests"] == []


def test_remote_embedder_with_empty_key_name_fails_before_any_call(no_network_openai):
    cfg = _remote_embedding(load_config(EXPERIMENT), api_key_env="")
    checks = _by_name(doctor(cfg, http_get=_fetch_factory([]), gpu_query=lambda: None, python_version=(3, 12), env={}))
    assert checks["embedding API key"].status == "FAIL"
    assert no_network_openai["requests"] == []


# --- эмбеддер: локальный (только HTTP-проба, без SDK) ---


def test_local_embedder_keeps_the_models_probe_and_skips_the_sdk(no_network_openai):
    cfg = _local_embedding(load_config(EXPERIMENT))
    seen = []

    def fetch(url, timeout):
        seen.append(url)
        return {"data": []} if url.endswith("/models") else {"status": "ok"}

    checks = _by_name(
        doctor(cfg, http_get=fetch, gpu_query=lambda: None, python_version=(3, 12), env={"OPENROUTER_API_KEY": SECRET})
    )
    assert checks["embedding server"].status == "OK"
    assert f"{LOCAL_URL}/models" in seen
    assert "embedding call" not in checks
    assert no_network_openai["requests"] == []


# --- свободная память GPU ---


def _gpu_check_for(used_mib, *, rollout=0.15, total=81920):
    cfg = load_config(EXPERIMENT)
    cfg = cfg.model_copy(
        update={"training": cfg.training.model_copy(update={"rollout_gpu_memory_utilization": rollout})}
    )
    checks = doctor(
        cfg,
        http_get=_fetch_factory([]),
        gpu_query=lambda: f"{total}, {used_mib}",
        python_version=(3, 12),
        env={"OPENROUTER_API_KEY": SECRET},
    )
    return _by_name(checks)["GPU"]


def test_gpu_with_20_gib_free_passes_the_0_15_rollout_share():
    gpu = _gpu_check_for(used_mib=61440)  # 20 GiB свободно на карте 80 GiB
    assert gpu.status == "OK"
    assert "need 18432 MiB" in gpu.detail


def test_gpu_free_memory_just_below_the_requirement_fails_with_a_hint():
    gpu = _gpu_check_for(used_mib=81920 - 18431)  # нужно 0.15 * 81920 + 6144 = 18432 MiB, свободно на 1 меньше
    assert gpu.status == "FAIL"
    assert "rollout_gpu_memory_utilization" in gpu.hint
    assert "max_context_tokens" in gpu.hint


def test_gpu_free_memory_exactly_at_the_requirement_passes():
    assert _gpu_check_for(used_mib=81920 - 18432).status == "OK"


def test_gpu_requirement_follows_the_configured_rollout_share():
    assert _gpu_check_for(used_mib=61440, rollout=0.45).status == "FAIL"


def test_gpu_output_that_cannot_be_parsed_is_a_warning():
    cfg = load_config(EXPERIMENT)
    checks = doctor(
        cfg,
        http_get=_fetch_factory([]),
        gpu_query=lambda: "garbage",
        python_version=(3, 12),
        env={"OPENROUTER_API_KEY": SECRET},
    )
    assert _by_name(checks)["GPU"].status == "WARN"


def test_nvidia_smi_query_reads_gpu0_memory_in_mib(monkeypatch):
    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        return subprocess.CompletedProcess(args, 0, stdout="81920, 0\n81920, 5\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert application._nvidia_smi() == "81920, 0"
    assert "--query-gpu=memory.total,memory.used" in seen["args"]
    assert "nounits" in " ".join(seen["args"])


def test_nvidia_smi_missing_or_failing_returns_none(monkeypatch):
    def missing(args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", missing)
    assert application._nvidia_smi() is None

    monkeypatch.setattr(
        subprocess, "run", lambda args, **kwargs: subprocess.CompletedProcess(args, 9, stdout="", stderr="")
    )
    assert application._nvidia_smi() is None
