# ADR 0005: RL-обучение через Agent Lightning + verl (без форка verl)

Статус: принято, 2026-10-11; обновлено 2026-10-12 (модели, общая GPU 80 ГБ, удалённые эмбеддинги).

## Контекст
Проект умел только оценивать. Обучение оркестратора существовало в соседнем `MultiAgents-main`:
это форк verl 0.8.0.dev с правками ядра (`ray_trainer.py`, `main_ppo.py`, `agent_loop.py`, `PARLConfig`).
Почти все эти правки нужны для локального GPU-пула замороженных субагентов. У нас одна GPU 80 ГБ, которую делят
с другими процессами (свободно около 20 ГБ). Поэтому субагенты (ministral-3B, qwen-2.5-7B, qwen3.5-9B), судья
(qwen3.5-9B) и эмбеддинги запросов (qwen3-embedding-8B) работают через OpenRouter, а локальный пул не нужен.
Обучаемый оркестратор — Qwen3.5-2B, его роллаут-vLLM получает `training.rollout_gpu_memory_utilization = 0.15`
от всей карты (см. `docs/deployment.md`).

## Решение
Берём Agent Lightning (`agentlightning==1.0.2`, по образцу примера Search-R1) поверх `verl==0.8.0` из PyPI.
Пины: `vllm==0.22.0` (как в `scripts/setup_verl.sh` agl для verl 0.8.0; документация agl упоминает 0.20.2, расхождение описано в `docs/deployment.md`), CUDA 12.9 или 13.0, Python ≥3.12, Linux. Локальный controller на Windows не работает.

| Что | Как | Где у нас |
|---|---|---|
| Эпизод | обычный async Python; `agl-controller` запускает `Class().run()` в отдельном процессе на каждый rollout | `agent.py` |
| Обучаемая модель | OpenAI API по адресу `AGL_OPENAI_BASE_URL` (прокси agl → vLLM verl); прокси сам добавляет `return_token_ids` и `logprobs` | `agent.py`, `assembly.build_orchestrator_client` |
| Только оркестратор обучается | в обучение попадают только вызовы через прокси; субагенты и судья ходят в OpenRouter отдельным клиентом | `subagents.py` |
| Данные вопроса | `agentlightning.local.env_map`: имя env-переменной → путь `input.<поле>` | `training/overrides.py` |
| train/val | режим зашит в URL прокси (`/mode/train/` или `/mode/val/`) | `agent.py` |
| Награда | POST `{"event_type":"reward","data":{"value":r,...}}` на `AGL_EVENT_URL` | `agent.py`, `reward.py` |
| Advantage | встроенный `grpo` + `algorithm.norm_adv_by_std_in_grpo=false` (= PARL: r − mean группы) + `enable_rollout_level_advantage` от agl | overrides |
| Лосс | `@register_policy_loss("k25_token_clip")`, регистрация в Ray-воркерах через `worker_process_setup_hook` (наш хук вызывает и хук agl) | `training/k25_loss.py` (`setup_worker`) |
| α, β, τ лосса | `PolicyLossConfig` в verl строгий, поэтому значения идут через env `ORCHESTRATOR_RESEARCH_K25` (JSON) в `runtime_env.env_vars` | `training/overrides.py` |
| Номер шага для отжига λ | агенту шаг не передаётся; hook `on_enqueue` считает train-rollout'ы и кладёт `step = count // (batch × n)` в `input` | `training/hooks.py` |
| Метрики по сплитам | agl считает только `val/reward`; агент пишет каждый эпизод в run-папку, а `report` считает метрики по сплитам | `agent.py`, `report.py` |
| Tool calling | `rollout.engine_kwargs.vllm.{enable_auto_tool_choice: true, tool_call_parser: qwen3_coder}` | overrides |

## Следствия
- Эмбеддинги запросов удалённые (OpenRouter, `qwen/qwen3-embedding-8b`), поэтому локальный vLLM эмбеддера
  запускается только для локального `[embedding] base_url`. Индекс FAISS остаётся на CPU. `doctor` проверяет
  удалённый эмбеддер одним запросом `ping`, а не `/models`.
- Память GPU общая: 0.15 — доля всей карты, а не свободной памяти. `doctor` проверяет запас
  (rollout-доля × карта + 6 ГБ) до старта.
- torch, verl и agentlightning импортируются только из `training/`. Домен, стратегии и лимиты остаются на чистом Python.
- Эпизод в eval и в обучении — один и тот же код (`assembly.run_episode`); меняются адрес оркестратора (`base_url`)
  и флаг `is_validation` (в eval: `step=-1`, `is_validation=True`).
- Шаблон Qwen3.5 вырезает thinking прошлых ходов, поэтому префикс токенов между вызовами может не совпасть.
  Тогда agl делит rollout на несколько строк (метрика `training/n_unmerged_rollouts`). На корректность это не влияет:
  advantage считается на уровне rollout.
- Подсчёт шага через `on_enqueue` корректен только для sync-режима без dynamic sampling. Это наш режим по умолчанию;
  при включении `async_rollout` отжиг λ нужно пересмотреть.
- Данные как в прошлом проекте: `train` 600 строк, `val_test` 200 (held-out), `val_full` 800 (пересекается с train).
  По умолчанию валидация идёт на `val_test`. Правило 6 AGENTS.md: строки 0–599 не попадают в `val_test` (600–799);
  `val_full` (0–799) пересекается с train намеренно, для паритета, и никогда не отчитывается как held-out.
