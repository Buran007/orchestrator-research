# 000 smoke_local: локальный прогон кода без внешних API

## Назначение

Проверка кода на одной общей GPU: роллаут оркестратора, субагенты, судья, поиск, обучение на 2 шагах
без flash-attn. Внешних API нет: OpenRouter недоступен с сервера (403).

Это не научный результат. Модели 0.8B, 2 шага, батч 2 и 4 эпизода на шаг: метрики качества здесь ничего не
значат. Смотрим, что код проходит шаги без ошибок и что память укладывается в свободные ~20 ГБ.

## Что поднимается локально

| Сервис | Модель | Порт | Доля GPU (от 80 ГБ) | Память |
|---|---|---|---|---|
| frozen (субагенты и судья) | Qwen/Qwen3.5-0.8B | 8001 | `FROZEN_GPU_UTIL`, 0.07 | около 5.6 ГБ |
| embedding (эмбеддинги запроса) | Qwen/Qwen3-Embedding-0.6B | 8003 | `EMBED_GPU_UTIL`, 0.03 | около 2.4 ГБ |
| retrieval (поиск, CPU) | индекс qwen3-embedding-0.6b | 8100 | - | ОЗУ |
| обучение (оркестратор, verl) | Qwen/Qwen3.5-0.8B | - | `rollout_gpu_memory_utilization`, 0.08 | около 6.4 ГБ под vLLM роллаутов |

Оценка памяти GPU: rollout 0.08 x 80 = 6.4 ГБ; frozen 0.07 x 80 = 5.6 ГБ; embedder 0.03 x 80 = 2.4 ГБ; обучение
0.8B с offload около 3-5 ГБ. Итого 17-19 ГБ при ~20 ГБ свободно: запас маленький. Оценка не измерена;
`scripts/smoke_train.sh` пишет пик памяти в `artifacts/smoke/`. Оперативная память хоста около 12 ГБ.

Frozen-сервер и судья - один и тот же сервер, потому что `subagents.base_url` и `judge.base_url` совпадают.
Все три алиаса `[subagents.models]` указывают на одну модель: `services.sh` откажется стартовать, если
значения различаются (один локальный сервер обслуживает одну модель).

## Как запустить

Из корня проекта `orchestrator-research`. Видимость GPU задаётся переменной окружения; `services.sh` и
`doctor` читают номер карты из `CUDA_VISIBLE_DEVICES`.

```bash
export CUDA_VISIBLE_DEVICES=0            # номер общей карты
CFG=experiments/000_smoke_local/config.toml

.venv/bin/python -m orchestrator_research.cli validate-config --config $CFG
bash scripts/services.sh start $CFG      # frozen -> embedding -> retrieval, с проверкой здоровья
.venv/bin/orchestrator-research doctor --config $CFG
bash scripts/smoke_train.sh $CFG         # 2 шага обучения с замером пика GPU
bash scripts/services.sh stop $CFG       # остановить свои сервисы (только по pid-файлам)
```

Не запускайте `services.sh` и обучение, пока на карте работают чужие процессы сверх оценки памяти: vLLM
проверяет свободную память при старте и откажется стартовать, если её не хватает.

## Ожидаемый результат

- `validate-config` печатает `config_hash` и без ошибок.
- `train --dry-run` показывает `actor_rollout_ref.model.path = "Qwen/Qwen3.5-0.8B"`,
  `gpu_memory_utilization = 0.08`, `use_remove_padding = false`,
  `override_config.attn_implementation = "sdpa"`.
- `smoke_train.sh` завершается без traceback; в логе есть `k25_in_band_frac` и `n_unmerged_rollouts`.

Если `attn_implementation` или `use_remove_padding` не применились, проверьте в логе verl, что ключи
приняты: dry-run их не проверяет.
