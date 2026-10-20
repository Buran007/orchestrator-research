"""Резерв видеопамяти процессом обучения с первой секунды прогона — для общих карт.

Без него обучение отдаёт память между фазами (offload на CPU и ``empty_cache``), соседи её занимают,
и следующее обновление весов падает с CUDA OOM.

При ``OR_GPU_RESERVE_GB`` > 0 воркер обучения перед ``init_model`` выделяет столько GB и сразу освобождает:
кэширующий аллокатор PyTorch оставляет блоки за процессом, и дальнейшие выделения берут их. Все пути
к ``empty_cache`` в этом процессе отключены, поэтому verl память драйверу не вернёт. vLLM — отдельный процесс;
его память держит ``free_cache_engine = false``.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable

RESERVE_ENV = "OR_GPU_RESERVE_GB"
_MARK = "_or_gpu_reserve"


def reserve_gb_from_env() -> float:
    """Сколько GB резервировать; 0 (выключено), если переменной нет."""
    value = float(os.environ.get(RESERVE_ENV, "") or 0)
    if value < 0:
        raise ValueError(f"{RESERVE_ENV} must be >= 0, got {value}")
    return value


def hold_cuda_memory(gb: float) -> None:
    """Выделить ``gb`` GB на текущей GPU, вернуть в кэш процесса и отключить ``empty_cache``."""
    import torch

    block = torch.empty(int(gb * 1024**3), dtype=torch.uint8, device="cuda")
    del block

    def keep() -> None:
        return None

    # Все пути: torch.cuda.empty_cache, torch.cuda.memory.empty_cache и C-функция, в которую ведут оба.
    # Подмена одного лишь первого не помогала — verl освобождал память через второй.
    torch.cuda.empty_cache = keep
    torch.cuda.memory.empty_cache = keep
    torch._C._cuda_emptyCache = keep


def wrap_init_model(worker_cls: type, gb: float, hold: Callable[[float], None] = hold_cuda_memory) -> None:
    """Обернуть ``worker_cls.init_model``, чтобы сначала вызывался ``hold(gb)``; повторный вызов безопасен."""
    original = worker_cls.init_model
    if getattr(original, _MARK, False):
        return

    @functools.wraps(original)  # копирует __dict__, где @register из verl хранит режим рассылки вызова
    def init_model(self, *args, **kwargs):
        hold(gb)
        return original(self, *args, **kwargs)

    setattr(init_model, _MARK, True)
    worker_cls.init_model = init_model


def install() -> None:
    """Часть хука воркера Ray: при ``OR_GPU_RESERVE_GB`` > 0 обернуть воркер обучения verl."""
    gb = reserve_gb_from_env()
    if gb <= 0:
        return
    from verl.workers.engine_workers import ActorRolloutRefWorker

    wrap_init_model(ActorRolloutRefWorker, gb)
