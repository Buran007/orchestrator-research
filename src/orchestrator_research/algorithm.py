"""Эталонная математика RL без зависимостей (без torch и фреймворков).

Преимущество в группе: награда роллаута минус средняя награда роллаутов того же вопроса, без деления на
стандартное отклонение (``norm_adv_by_std_in_grpo = false`` в ``training/overrides.py``).

Лосс K25 (из техотчёта Kimi K2.5): здесь — спецификация; то, что реально считает verl, — ``training/k25_loss.py``,
совпадение чисел проверяет ``tests/test_k25_loss.py``. «25» — от названия отчёта, к размеру группы отношения не имеет.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

LOG_RATIO_CLAMP = 20.0


def group_relative_advantages(rewards: Sequence[float], group_ids: Sequence[Hashable]) -> list[float]:
    """Вычесть из каждой награды среднее по её группе; порядок сохраняется."""
    if len(rewards) != len(group_ids):
        raise ValueError("rewards and group_ids must have the same length")
    grouped: dict[Hashable, list[float]] = defaultdict(list)
    for reward, group_id in zip(rewards, group_ids, strict=True):
        grouped[group_id].append(float(reward))
    if any(len(values) < 2 for values in grouped.values()):
        raise ValueError("every prompt group must contain at least two rollouts")
    means = {group_id: sum(values) / len(values) for group_id, values in grouped.items()}
    return [float(reward) - means[group_id] for reward, group_id in zip(rewards, group_ids, strict=True)]


@dataclass(frozen=True, slots=True)
class K25Params:
    """Полоса ``[alpha, beta]`` для log-ratio и вес ``tau`` квадратичного регуляризатора."""

    alpha: float = -0.2
    beta: float = 0.2
    tau: float = 1e-3

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "tau"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"K25 {name} must be a finite number; got {value!r}")
        if not self.alpha < 0 < self.beta:
            raise ValueError(f"K25 requires alpha < 0 < beta; got alpha={self.alpha}, beta={self.beta}")
        if self.tau < 0:
            raise ValueError(f"K25 requires tau >= 0; got tau={self.tau}")


@dataclass(frozen=True, slots=True)
class K25Result:
    total_loss: float
    policy_loss: float
    regularizer_loss: float
    in_band_fraction: float


def k25_token_clip_reference(
    *,
    old_log_probs: Sequence[float],
    new_log_probs: Sequence[float],
    advantages: Sequence[float],
    trainable_mask: Sequence[int],
    params: K25Params | None = None,
) -> K25Result:
    """Эталонный лосс по обучаемым токенам ответа.

    ``old_log_probs`` / ``new_log_probs`` — log-вероятности токенов у политики, которой генерировали, и у новой.
    Выводы инструментов замаскированы.
    """
    if params is None:
        params = K25Params()
    lengths = {len(old_log_probs), len(new_log_probs), len(advantages), len(trainable_mask)}
    if len(lengths) != 1:
        raise ValueError("all token arrays must have the same length")
    deltas = _clipped_log_probability_deltas(old_log_probs, new_log_probs)
    response_indices = [i for i, value in enumerate(trainable_mask) if int(value)]
    if not response_indices:
        raise ValueError("trainable_mask must contain at least one token")
    # Токены вне полосы не дают policy-градиента, но остаются в регуляризаторе — так же должно быть в k25_loss.py.
    in_band_indices = [index for index in response_indices if params.alpha <= deltas[index] <= params.beta]
    policy_loss = (
        sum(-float(advantages[index]) * deltas[index] for index in in_band_indices) / len(in_band_indices)
        if in_band_indices
        else 0.0
    )
    regularizer_loss = params.tau * sum(deltas[i] ** 2 for i in response_indices) / len(response_indices)
    return K25Result(
        total_loss=policy_loss + regularizer_loss,
        policy_loss=policy_loss,
        regularizer_loss=regularizer_loss,
        in_band_fraction=len(in_band_indices) / len(response_indices),
    )


def _clipped_log_probability_deltas(old_log_probs: Sequence[float], new_log_probs: Sequence[float]) -> list[float]:
    """``new - old`` по токенам, обрезанное до ``±LOG_RATIO_CLAMP``."""
    return [
        max(-LOG_RATIO_CLAMP, min(LOG_RATIO_CLAMP, new - old))
        for old, new in zip(old_log_probs, new_log_probs, strict=True)
    ]
