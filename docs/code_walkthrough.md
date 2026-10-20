# Walkthrough кода

Читать после `README.md`. Документ описывает слои и правила зависимостей, путь одного шага обучения и одной
оценки, затем модули. Решения и их причины: `docs/adrs/`. Формулы: `docs/reference_algorithm.md`.
Ключи конфига: комментарии в `configs/base.toml`.

## 1. Слои и правила зависимостей

Зависимости идут сверху вниз: нижний слой не импортирует верхний.

| Слой | Модули | Что внутри |
|---|---|---|
| Точки входа | `cli.py`, `application.py`, `training/train.py`, `agent.py`, `eval_runner.py`, `demo.py` | команды, `doctor`, запуск обучения, эпизод для agl, оценка сплита |
| Сборка | `assembly.py` | единственное место, где соединяются адаптеры, стратегия, судья и награда |
| Адаптеры | `adapters/` | HTTP-клиенты, судья, FAISS-сервер, эмбеддер |
| Стратегии и субагенты | `strategies/`, `subagents.py`, `tools.py` | цикл оркестратора, пул субагентов, схемы инструментов |
| Чистый домен | `domain.py`, `limits.py`, `reward.py`, `algorithm.py` | данные, бюджеты, награда, численные эталоны |

Правила (AGENTS.md и ADR 0005, 0006):

1. Домен и алгоритмы (`domain`, `limits`, `reward`, `algorithm`, `tools`, `strategies/base.py`, `subagents.py`)
   не импортируют Ray, Hydra и SDK провайдеров и не делают файлового или сетевого ввода-вывода.
2. Зависимости описаны маленькими `Protocol` из `domain.py`: `ChatModel`, `Retriever`, `Judge`, `Strategy`.
   Код API, GPU и хранения живёт в `adapters/`.
3. `torch`, `verl` и `agentlightning` импортируются только в `training/`. Проверка:
   `grep -rn "import torch\|from verl\|import agentlightning\|from agentlightning" src | grep -v training/`
   должна быть пустой.
4. Индекс, корпус и токенизатор загружает только `adapters/retrieval_server.py` (`load_backend`). Процессы
   роллаутов создают лишь `HttpRetriever` (`assembly.build_retriever`).
5. Обучение и оценка выполняют один и тот же код: `assembly.run_episode`. Меняются адрес оркестратора и флаг
   `is_validation`. Поэтому оценка сопоставима с наградой обучения.

## 2. Один шаг обучения

Шаг потребляет `training.train_batch_size × training.rollout_n` train-роллаутов (по умолчанию 16 × 4 = 64).

1. **`scripts/train.sh CONFIG`.** Проверяет, что заданы переменные ключей (значения не печатаются), запускает
   `orchestrator-research doctor` (при FAIL останавливается), создаёт `AGL_KEY`, поднимает `agl-server`
   (порт `AGL_PORT`, по умолчанию 8080) и `agl-controller` (`runner_type=local`,
   `local_runner.maximum_size = training.agent_concurrency`), затем вызывает `orchestrator-research train`.
2. **`training/train.py`.** `load_config` → `overrides.build_trainer_config` (плоский словарь
   `{"dotted.key": value}`) → `assert_lengths_consistent`. Затем создаёт run-папку (отказ, если она уже есть) и
   пишет `config.resolved.json`, `manifest.json`, `trainer_overrides.json`. Ключ `agl_key` добавляется
   `with_agl_endpoint` только в память, на диск не попадает. Строки читаются из `train.parquet` и
   `<training.val_split>.parquet`, к каждой добавляются `config_path` и `run_dir`. Затем `run_ppo`.
3. **verl и agl в процессе тренера.** Advantage: `grpo` без нормировки на std (`A_i = r_i − mean(группы)`).
   `hooks.StepCounterHooks.on_enqueue` кладёт в вход train-роллаута `train_step = count // (batch × n)`;
   validation получает `-1` и счётчик не сдвигает. Resume отключён (`trainer.resume_mode = "disable"`):
   счётчик при возобновлении начался бы с нуля (`reference_algorithm.md`, раздел 3).
4. **agl-controller.** Для каждого роллаута запускает отдельный процесс `agent.OrchestratorAgent.run()`.
   Поля строки попадают в переменные окружения по `overrides.ENV_MAP` (`QUESTION_ID`, `QUESTION`, `ANSWERS`,
   `SPLIT`, `TRAIN_STEP`, `ORCH_RESEARCH_CONFIG`, `ORCH_RESEARCH_RUN_DIR`). Вместе с ними приходят
   `AGL_OPENAI_BASE_URL` (прокси с `/mode/train/` или `/mode/val/` и `/rollout/<id>/attempt/<id>/`),
   `AGL_EVENT_URL` и `AGL_KEY`.
5. **`agent.py`.** Строит клиент оркестратора на адрес прокси (`model="auto"`), вызывает `assembly.run_episode`,
   пишет документ эпизода в `episodes/<train|val>/<step>/<rollout>-<attempt>.json` (атомарно) и отправляет
   `POST {"event_type": "reward", "data": {"value": r, ...}}`. Если отправка не удалась, роллаут падает,
   а не получает тихо награду 0.
6. **`assembly.run_episode(cfg, question, deps, *, step, is_validation)`.** `deps` (`EpisodeDeps`) строит один раз
   `build_episode_deps` и переиспользует весь процесс. `build_strategy(cfg, deps)` на каждый эпизод создаёт свежий
   `TaskBudget(limits)`, `SubagentPool` и `EpisodeEnvironment` и берёт стратегию из `STRATEGIES[experiment.strategy]`.
   Затем запускается стратегия, судья выставляет вердикт. Любой его сбой (`_grade_fail_closed`) даёт `r_perf = 0`
   и статус ошибки, а не ложное «правильно». `r_perf = 1` только при `GRADED` и `correct`. Затем
   `reward.compose_reward`. Возвращает `EpisodeRecord`; документ строит `assembly.episode_document`.
7. **`strategies/base.OrchestratorLoop`.** Ход: `budget.reserve_orchestrator_turn()` → вызов оркестратора
   через прокси (этот вызов обучается) → `commit_orchestrator(limit, actual)` в `finally`: `actual` — это `output_tokens` (из `completion_tokens`), а при ошибке вызова он равен 0, и резерв всё равно снимается. Вызовы за ход
   ограничены `max_tool_calls_per_turn`, каждый получает ровно одно наблюдение, по порядку:
   - `browsecomp_search` → `tools.run_search` → `Retriever.search` (HTTP к `retrieval_server`);
   - `create_subagent` → `SubagentPool.create`;
   - `assign_task` → `SubagentPool.assign_many` (не больше `max_parallel` одновременно).
   Ответ без вызовов становится финальным. Иначе при нехватке бюджета вызывается финализатор без инструментов
   внутри `final_answer_reserve`. Сообщения только дописываются, поэтому префикс токенов между ходами общий.
8. **`subagents.py`.** Экземпляр регистрируется в `TaskBudget` (`max_subagents_per_task`). Задание
   выполняет `run_subagent`: до `subagent_max_steps` шагов, каждый шаг резервируется и списывается, вызовы
   модели идут через OpenRouter (не через прокси, поэтому не обучаются). Поиск тот же, что у оркестратора.
   Ответ обрезается до `subagents.return_max_chars`. Сбой провайдера возвращается текстом
   `[SUBAGENT_INFRA_FAILURE]` с `infra_failure = True`, а не исключением.
9. **`reward.py`.** `total = r_perf + λ_par(t)·r_parallel + λ_fin(t)·r_finish`; на валидации `total = r_perf`,
   компоненты всё равно пишутся. Формулы: `docs/reference_algorithm.md`.
10. **agl.** Агрегатор траекторий собирает токены и логпробы вызовов прокси в одну траекторию на роллаут;
    advantage считается на уровне роллаута. Если префикс токенов разорвался (шаблон вырезал thinking прошлых
    ходов), роллаут делится на строки: метрика `training/n_unmerged_rollouts`.
11. **Лосс K2.5.** `training/k25_loss.py` регистрирует `k25_token_clip` в каждом Ray-воркере через
    `setup_worker` (он же вызывает хук agl). Параметры α, β, τ приходят JSON-строкой в переменной
    `ORCHESTRATOR_RESEARCH_K25`, потому что строгий `PolicyLossConfig` verl их не принимает.
12. **Валидация.** Перед первым шагом (`val_before_train`) и каждые `test_freq` шагов на `training.val_split`.
    agl сообщает только `val/reward`; полные метрики по сплитам считает `report` по файлам эпизодов.
    Валидационные роллауты имеют `step = -1` и не влияют на отжиг λ.

## 3. Одна оценка

`orchestrator-research eval --config CFG --split val_test --run-id ID [--checkpoint-url URL]`
(обёртка: `scripts/eval.sh`, которая сначала при необходимости сливает FSDP-шарды и поднимает vLLM).

1. `cli._cmd_eval` → `eval_runner.run_eval`. Вопросы читаются из `<data_dir>/<split>.parquet`;
   повторные `question_id` отклоняются.
2. `artifacts/runs/<experiment>/<run_id>/eval/config.json` пишется один раз (`write_immutable_json`).
   Другой конфиг под тем же `run_id` отклоняется.
3. `JsonlResultStore(eval/<split>.jsonl)` отдаёт последнюю строку по каждому вопросу (`latest_records`). Вопрос
   пропускается, если эта строка проходит `is_usable`. Остальные вопросы идут через `assembly.run_episode` с
   `step=-1, is_validation=True`, с параллелизмом `--concurrency` (по умолчанию 8).
4. Каждый эпизод дописывается строкой (append + fsync). Неудачный вопрос в следующий запуск перезапускается
   (правила resume ниже).
5. CLI печатает `report.format_report` по сплиту. Точность — среднее `r_perf` только по оцениваемым строкам;
   судейские сбои не входят в среднее и выводятся отдельно как `judge_failures`. Сравнение двух файлов: `orchestrator-research compare`
   (McNemar и bootstrap). Команда отказывается работать, если множества успешно оценённых ID не совпадают.

**Resume** (`storage.is_usable`). Вопрос считается сделанным, если судья отработал (`GRADED` или
`SKIPPED_NO_ANSWER`), эпизод не `model_error` и нет `infra_errors`. Перезапускаются сбои судьи, сбой endpoint
оркестратора и инфраструктурные сбои (поиск, провайдер субагента). Неверный ответ, отсутствие ответа, ошибки
аргументов инструментов и исчерпание бюджета перезапуском не считаются, это результаты.

## 4. Модули

| Модуль | Роль | Ключевые имена |
|---|---|---|
| `config.py` | строгие модели конфига, `include` с глубоким слиянием, роли из TOML | `ExperimentConfig`, `load_config`, `load_role_prompts`, `config_hash` |
| `domain.py` | неизменяемые данные эпизода и протоколы | `Question`, `ModelTurn`, `Observation`, `TraceEvent`, `SubagentRun`, `Stage`, `EpisodeResult`, `GradeResult`; `ChatModel`, `Retriever`, `Judge`, `Strategy` |
| `limits.py` | бюджеты токенов и шагов: резерв, затем списание | `TaskLimits` (`None` = без лимита), `TaskBudget`, `StopReason`; формулы в docstring |
| `reward.py` | награда PARL и отжиг | `anneal`, `r_parallel`, `r_finish`, `compose_reward`, `RewardBreakdown` |
| `algorithm.py` | численные эталоны без autograd | `group_relative_advantages`, `K25Params`, `k25_token_clip_reference` |
| `tools.py` | схемы инструментов, системные промпты, поиск | `SEARCH_TOOL_NAME`, `CREATE_SUBAGENT_TOOL`, `ASSIGN_TASK_TOOL`, `orchestrator_tools`, `subagent_tools`, `run_search`, `render_hits`, `executed` |
| `strategies/__init__.py` | таблица стратегий | `STRATEGIES` (имя → фабрика), `get_strategy`, `build_swarm`, `build_single` |
| `strategies/base.py` | общий цикл оркестратора и окружение эпизода | `EpisodeEnvironment`, `OrchestratorLoop` (`run`, `_loop`, `_finalize`, `_call`, `_run_tools`) |
| `subagents.py` | цикл одного субагента и пул экземпляров | `run_subagent`, `SubagentPool` (`create`, `assign_many`), `INFRA_FAILURE_MARKER` |
| `assembly.py` | сборка объектов одного эпизода | `EpisodeDeps`, `build_episode_deps`, `build_strategy`, `run_episode`, `episode_document`, `EpisodeRecord` |
| `agent.py` | эпизод для одного роллаута agl | `OrchestratorAgent.run`, `is_validation_url`, `parse_rollout_ids`, `post_reward`; `--question-id` для отладки |
| `eval_runner.py` | оценка сплита с resume | `run_eval`, `EvalSummary`, `split_path` |
| `storage.py` | файлы: атомарная запись, JSONL, неизменяемые снимки | `atomic_write_json`, `write_immutable_json`, `JsonlResultStore`, `read_jsonl_readonly`, `latest_records`, `is_usable`, `collect_run_rows` |
| `report.py` | сводки и парное сравнение; `judge_failures` вне точности | `summarize_groups`, `format_report`, `paired_comparison`, `_mcnemar_exact` |
| `benchmark.py` | границы сплитов, расшифровка, parquet | `SPLITS` (`train` 0–599, `val_test` 600–799, `val_full` 0–799), `write_training_splits`, `training_split_rows`, `HuggingFaceBrowseCompRepository` |
| `adapters/chat.py` | чат-клиент: повторы, учёт usage | `ChatClient` (`output_tokens` = `completion_tokens`), `MissingUsageError` (ответ без `completion_tokens`), `api_key_from_env` |
| `adapters/judge.py` | судья | `OpenAICompatibleJudge`, `parse_judge_response` |
| `adapters/retrieval.py` | FAISS-поиск (для сервера) и HTTP-клиент (для роллаутов) | `HttpRetriever` (повторы на 429, 5xx и сетевые ошибки), `BrowseCompRetriever.from_shards`, `OpenAICompatibleEmbedder`, `TokenizerSnippet`, `download_index` |
| `adapters/retrieval_server.py` | HTTP-сервер поиска | `make_server(search, stats, host, port)`, `load_backend(cfg)`, `parse_search_request`; `GET /healthz`, `POST /search` |
| `adapters/fake.py` | фейковые модель и поиск для тестов и демо | `ScriptedChatModel`, `StaticRetriever` |
| `training/overrides.py` | словарь настроек verl и agl; счётчик шагов | `build_trainer_config`, `with_agl_endpoint`, `assert_lengths_consistent`, `ENV_MAP`, `StepCounter` |
| `training/hooks.py` | номер шага у train-роллаутов | `StepCounterHooks` |
| `training/k25_loss.py` | лосс K2.5 как плагин verl | `k25_token_clip_loss`, `register`, `setup_worker`, `to_env`, `from_env`, `ENV_VAR` |
| `training/train.py` | точка входа обучения | `main`, `_prepare_run_dir`, `_compose_base_config` |
| `application.py` | сводка конфига, подготовка данных, `doctor` | `config_summary`, `prepare_data`, `doctor`, `render_checks` |
| `cli.py` | разбор аргументов и вызов модулей | `main`, команды `validate-config`, `prepare-data`, `serve-retrieval`, `doctor`, `train`, `eval`, `report`, `compare`, `demo` |

## 5. Порядок чтения

1. `configs/base.toml`, затем `experiments/template/config.toml`.
2. `config.py`: какие ключи существуют и что проверяется.
3. `limits.py`: раздел «Notation» в docstring, резерв и списание.
4. `strategies/base.py`: `OrchestratorLoop.run`, `_loop`, `_run_tools`.
5. `subagents.py`: сначала `run_subagent`, затем `SubagentPool`.
6. `reward.py` (docstring с формулами), затем `assembly.run_episode`.
7. `agent.py`, затем `training/overrides.py`, `training/hooks.py`, `training/train.py`.
8. `training/k25_loss.py` и `algorithm.py`, затем `docs/reference_algorithm.md`.

## 6. Можно пропустить на первом проходе

- `demo.py`: CPU-иллюстрация на маленьких числах (`orchestrator-research demo`).
- `report.py`: `paired_comparison` нужен при сравнении, не при обучении.
- `benchmark.py`: `decrypt_query_record` достаточно знать как «читает датасет».
- `adapters/judge.py`, `adapters/chat.py`: транспорт, повторы и разбор ответов провайдера.
- `adapters/retrieval_server.py`: достаточно контракта `/search` (ADR 0006).
- `storage.py`: форматы; важны только пути `episodes/` и `eval/`.

## 7. Правила эксперимента, которые держит код

- Один эксперимент меняет одну переменную относительно `configs/base.toml` (AGENTS.md, правило 4).
- Конфиг и результаты пишутся один раз; повторный запуск под тем же именем отклоняется.
- Базовая модель оценивается тем же кодом и на том же сплите до обучения. Так выглядит контроль.
- Строки `train` не оцениваются как held-out. `val_full` содержит train и не отчитывается как held-out
  (AGENTS.md, правило 6; ADR 0005).
