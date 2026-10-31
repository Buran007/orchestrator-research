# research/

Эта ветка — официальный Agent Lightning v1.0.2 от Microsoft (история upstream до тега `v1.0.2`) плюс наши
дополнения в `research/` и `.claude/`. Файлы Microsoft не правим: свои вещи кладём только новыми файлами, поэтому
`git diff v1.0.2 main` всегда показывает только наше.

Первая цель — запустить официальный пример Search-R1 на общем сервере с H200 и убедиться, что награда растёт,
а обученные веса доходят до vLLM. Только после этого — свой функционал (план ниже).

## Что проверено (2026-10-31)

- Пакет `agentlightning==1.0.2` из PyPI побайтно совпадает с исходниками тега `v1.0.2` в этом репозитории.
  Сравнение: `diff -r --strip-trailing-cr` по всем 31 файлу пакета.
- После релиза в `main` у Microsoft уже есть исправления в 3 файлах. Для нас важно одно: в
  `agentlightning/verl/rollout_adapter.py` (v1.0.2, строка 774) награды при сборке батча хранятся в `bfloat16`.
  Около 1 шаг этого формата ~0.004–0.008, поэтому мелкие дробные различия наград теряются. В `main` это исправлено
  на `float32`. Для бинарной награды Search-R1 (0/1, точное совпадение) это неважно; для дробных наград (штрафы,
  бонусы) — важно.

## Порядок чтения

Часть 1 — как пользователь подключается к фреймворку (договор между агентом и AGL):

1. `docs/01-quick-start.md`, `docs/05-basics.md` — три процесса: Trainer, API Gateway, Rollout Controller.
2. `docs/65-example-search-r1.md` — данные, сервис поиска, запуск.
3. `examples/search_r1/run.sh` — порядок запуска agl-server → agl-controller → тренер.
4. `examples/search_r1/train_search_r1_agent.py` — переопределения конфига verl и вызов `run_ppo`.
5. `examples/search_r1/agents/search_r1_agent.py` — агент: читает `AGL_OPENAI_BASE_URL`, `AGL_KEY`,
   `AGL_EVENT_URL` и вопрос из env, ходит к модели через прокси, в конце шлёт событие `reward`.
6. `examples/search_r1/agents/qa_em.py` — награда: точное совпадение ответа (пролистать).
7. `examples/search_r1/retrieval_server.py` — сервис поиска FAISS (пролистать).

Часть 2 — сам фреймворк, по пути одного эпизода (`agentlightning/`):

| # | файл | что понять |
|---|---|---|
| 8 | `schemas.py` | словарь: Rollout, RolloutState, Event, ModelRequestData, RewardData |
| 9 | `server/store.py`, `server/app.py`, `config/server.yaml` | состояние сервера (в памяти), сборка приложения |
| 10 | `server/routes/rollouts.py` | очередь эпизодов и переходы состояний |
| 11 | `server/routes/proxy.py`, `server/proxy.py` | прокси к vLLM: id эпизода в URL, пересылка, повторы, таймаут |
| 12 | `server/routes/events.py` | запрос/ответ модели → триплет (токены промпта, токены ответа, logprobs); награда |
| 13 | `controller/__main__.py`, `controller/local_reconciler.py`, `config/controller.yaml` | запуск агента процессом на каждый эпизод |
| 14 | `hooks.py` | крючки жизненного цикла эпизода |
| 15 | `verl/entrypoint.py`, `verl/config.yaml` | `run_ppo`: как AGL встраивается в verl |
| 16 | `verl/trainer.py` | цикл `fit()`: генерацию verl заменили эпизодами AGL; что отбрасывается |
| 17 | `verl/agl_rollout_manager.py` | постановка батча, ожидание, сбор завершённых эпизодов |
| 18 | `verl/rollout_adapter.py` | **главное:** триплеты → батч verl, склейка ходов по префиксу токенов, `response_mask`, обрезка |
| 19 | `verl/rollout_level_advantage.py` | одно преимущество на эпизод |
| 20 | `verl/per_rollout_loss.py`, `client.py`, `verl/dataset.py`, `verl/vllm_compat.py` | пролистать |

Часть 3 — как свои алгоритмы подключаются к verl без правки его кода: `verl/trainer/ppo/core_algos.py`
(`register_policy_loss`, `register_adv_est`) и тесты verl `tests/trainer/ppo/test_core_algos_on_cpu.py`.

## Запуск Search-R1 на общем H200 (`research/search_r1/`)

Обёртки только передают официальным скриптам переменные окружения и ключи конфига; код примера не меняется.

| шаг | скрипт | что делает |
|---|---|---|
| 1 | `prepare_data.sh` | env `retriever` и данные, как официальный `data_process.sh`, но индекс склеивается на лету |
| 2 | `make_val_subset.py` | 512 случайных вопросов из `test.parquet` (seed 42) для валидации |
| 3 | `retrieval_h200.sh` | официальный `retrieval_launch.sh` на одной GPU, `127.0.0.1` |
| 4 | `run_h200.sh` | официальный `run.sh` на 2 GPU с переопределениями ниже |

### Отклонения от примера и зачем

| что | в примере | у нас | зачем |
|---|---|---|---|
| GPU | 8× A100 40 ГБ | 2× H200 под обучение + 1 под поиск | столько свободно на общем сервере |
| `data.train_batch_size` / `ppo_mini_batch_size` | 512 / 256 | 128 / 64 | на 2 GPU шаг с 512 вопросами шёл бы часами |
| `trainer.total_training_steps` | 300 | 150 | для проверки роста хватает |
| валидация | весь `test.parquet` | 512 вопросов, `val_before_train=True` | полная валидация каждые 10 шагов слишком долгая; шаг 0 нужен как базовая точка |
| `trainer.save_freq` | 10 | -1 | чекпоинт 3B с оптимизатором ~40 ГБ, общий диск почти полон |
| `trainer.logger` | console, wandb | console | W&B не настроен |
| порты | 8000 (поиск), 8080 (AGL) | свободные 18xxx | 8000 и 8080 на сервере заняты чужими процессами |
| `rollout.gpu_memory_utilization` | 0.5 | под свободную память карт | карты общие |
| сборка индекса | `part_aa` + `part_ab` на диск, потом `cat` | склейка при загрузке | официальный путь временно держит ~130 ГБ, на диске столько нет |

## Дальше (задание научрука)

Пример из библиотеки — основа. После того как он покажет рост награды, по одному эксперименту за раз:
лосс Kimi K2.5 (`@register_policy_loss`, тесты на CPU в стиле verl) → свой агент-оркестратор с субагентами
(локальные модели) → adaptive reasoning budget → task planning / shared memory → адаптивный выбор агентов по
сложности и штраф за цену (`@register_adv_est` или награда агента).

Старый проект `orchestrator-research` (эксперименты 001–002) сохранён в бэкапе
`C:\Projects\MultiAgents-main\backups\2026-10-31\` (git bundle всех веток и tar рабочей копии).
