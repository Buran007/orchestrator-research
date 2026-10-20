"""Лосс Kimi K2.5 с отсечением по токенам — плагин policy loss для verl.

Формула (по токену t; маска ответа m_t; r_t = log pi_theta - log pi_old, обрезан до [-20, 20];
полоса [alpha, beta], alpha < 0 < beta)::

    g_t    = 1 if alpha <= r_t <= beta else 0          (градиент через g_t не идёт)
    L_pg   = agg( -A_t * r_t * w_t * g_t * m_t )        (w_t — IS-вес роллаута, необязательный)
    L_tau  = agg( tau * r_t^2 * m_t )                   (регуляризатор по ВСЕЙ маске ответа)
    L      = L_pg + L_tau

Токены вне маски и вне полосы не дают policy-градиента; токены вне полосы при этом остаются в tau-слагаемом.
``agg`` — агрегация (по умолчанию token-mean), передаётся снаружи.

Источник: техотчёт Kimi K2.5, раздел 4.4.2; перенесено из форка verl в MultiAgents-main
(core_algos.py:1404-1508, ``compute_policy_loss_k25_token_clip``). Эталон без зависимостей и ``K25Params`` —
``orchestrator_research.algorithm``. Метрики — как в форке: actor/k25_in_band_frac, actor/k25_log_ratio_mean,
actor/k25_pg_loss, actor/k25_tau_loss, actor/ppo_kl.

Строгий ``PolicyLossConfig`` verl не пропускает свои параметры, поэтому они идут JSON-ом в переменной ``ENV_VAR``.
Без неё ``from_env`` падает: воркер Ray не должен молча взять значения по умолчанию.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, fields
from typing import Any

import torch

from orchestrator_research.algorithm import LOG_RATIO_CLAMP, K25Params

LOSS_MODE = "k25_token_clip"
ENV_VAR = "ORCHESTRATOR_RESEARCH_K25"


def to_env(params: K25Params) -> str:
    """``params`` → JSON для ``ENV_VAR``."""
    return json.dumps({name: float(value) for name, value in asdict(params).items()}, sort_keys=True)


def from_env(environ: Mapping[str, str] = os.environ) -> K25Params:
    """Прочитать ``ENV_VAR``; переменная обязательна и не пустая."""
    raw = environ.get(ENV_VAR)
    if raw is None or not raw.strip():
        raise ValueError(f"{ENV_VAR} is not set; overrides.build_trainer_config must put it in the Ray runtime_env")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{ENV_VAR} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{ENV_VAR} must be a JSON object; got {type(data).__name__}")
    allowed = {field.name for field in fields(K25Params)}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{ENV_VAR} has unknown keys: {sorted(unknown)}")
    return K25Params(**data)  # типы и диапазоны проверяет K25Params


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp(min=1)


def k25_token_clip_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    params: K25Params,
    agg: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """``(total_loss, metrics)`` для тензоров (B, T).

    ``agg(loss_mat, loss_mask)`` сворачивает матрицу по токенам в скаляр (для token-mean — сумма по маске
    делённая на число токенов в маске).
    """
    mask = response_mask.to(dtype=log_prob.dtype)
    log_ratio = torch.clamp(log_prob - old_log_prob, min=-LOG_RATIO_CLAMP, max=LOG_RATIO_CLAMP)

    with torch.no_grad():
        in_band = ((log_ratio >= params.alpha) & (log_ratio <= params.beta)).to(log_prob.dtype)
        in_band_mask = mask * in_band

    pg_losses = -advantages * log_ratio
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights
    if float(in_band_mask.sum()) == 0.0:
        # Обычный token-mean дал бы здесь 0/0; слагаемое равно 0 с нулевым градиентом.
        pg_loss = pg_losses.sum() * 0.0
    else:
        pg_loss = agg(pg_losses, in_band_mask)

    tau_loss = agg(params.tau * log_ratio.pow(2), mask)
    total_loss = pg_loss + tau_loss

    metrics = {
        "actor/k25_in_band_frac": _masked_mean(in_band, mask).detach().item(),
        "actor/k25_log_ratio_mean": _masked_mean(log_ratio, mask).detach().item(),
        "actor/k25_pg_loss": pg_loss.detach().item(),
        "actor/k25_tau_loss": tau_loss.detach().item(),
        "actor/ppo_kl": _masked_mean(-log_ratio, mask).detach().item(),
    }
    return total_loss, metrics


def register() -> None:
    """Зарегистрировать лосс в реестре verl (повторный вызов безопасен); verl импортируется лениво."""
    from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY, agg_loss, register_policy_loss

    if LOSS_MODE in POLICY_LOSS_REGISTRY:
        return

    @register_policy_loss(LOSS_MODE)
    def compute_policy_loss_k25_token_clip(
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        loss_agg_mode: str = "token-mean",
        config: Any = None,
        rollout_is_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if config is None:
            raise ValueError("k25_token_clip requires the actor config")
        params = from_env()
        global_info = dict(getattr(config, "global_batch_info", None) or {})

        def agg(loss_mat: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
            return agg_loss(loss_mat=loss_mat, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode, **global_info)

        return k25_token_clip_loss(
            old_log_prob,
            log_prob,
            advantages,
            response_mask,
            params=params,
            agg=agg,
            rollout_is_weights=rollout_is_weights,
        )


def setup_worker() -> None:
    """``worker_process_setup_hook`` для Ray: в каждом воркере verl регистрирует лосс K25 и ставит резерв памяти.

    Этот ключ заменяет хук agentlightning по умолчанию, поэтому хук agl вызывается здесь же вручную.
    """
    from agentlightning.verl.per_rollout_loss import register_in_worker

    from orchestrator_research.training import gpu_reserve

    register_in_worker()
    register()
    gpu_reserve.install()
