# Orchestrator Research

Исследовательская реализация RL-обучения оркестратора с замороженными субагентами. Оркестратор (Qwen3.5-2B)
отвечает на вопросы BrowseComp-Plus, вызывая поиск и пул субагентов (ministral-3B, qwen-2.5-7B, qwen3.5-9B через
OpenRouter) при фиксированных токенных лимитах на задачу. Эмбеддинги запросов тоже идут через OpenRouter
(qwen3-embedding-8B), судья — qwen3.5-9B. Обучение идёт через Agent Lightning и verl (ADR 0005), лосс K2.5
подключён как плагин verl, а поиск вынесен в общий HTTP-сервис (ADR 0006). Эксперимент 001:
`experiments/template/`.

## Быстрый старт на сервере

Сервер: Linux, одна GPU 80 ГБ (общая, свободно около 20 ГБ), CUDA 12.9/13.0, Python 3.12, `uv`. Таблица памяти и порты: `docs/deployment.md`.
Команды выполняются из корня репозитория.

```bash
bash scripts/bootstrap_server.sh                                   # 1. окружение и GPU-стек
export OPENROUTER_API_KEY=...                                      # 2. ключ; значение нигде не печатается
.venv/bin/orchestrator-research prepare-data --config experiments/template/config.toml   # 3. данные и индекс
bash scripts/services.sh start experiments/template/config.toml                         # 4. поиск (и эмбеддер, если он локальный)
bash scripts/smoke_train.sh experiments/template/config.toml                            # 5. проверка на 2 шагах
bash scripts/train.sh experiments/template/config.toml --run-id run1                    # 6. обучение
bash scripts/eval.sh experiments/template/config.toml <checkpoint_dir> val_test run1-final  # 7. оценка
```

- `<checkpoint_dir>` — папка шага вида `artifacts/checkpoints/template/run1/global_step_N`
  (внутри есть `actor/`); для базовой модели вместо пути пишется `base`.
- Сервисы: `bash scripts/services.sh status <cfg>` (состояние, GPU, последние прогоны) и
  `bash scripts/services.sh stop <cfg>`.
- Отчёт по прогону: `.venv/bin/orchestrator-research report artifacts/runs/<experiment>/<run_id>`.
- `val_test` — единственный held-out сплит для отчётности. `val_full` перекрывается с обучением, его не
  сообщают как held-out.

## Статус

- Эксперименты ведутся в отдельных ветках `exp/NNN-имя` от `dev`; реестр — `experiments/registry.md`.
- Ограничения общих серверов (карта и RAM, порты 18xxx, прокси до OpenRouter): `docs/deployment.md`.

## Локальная разработка (Windows)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[benchmark,dev]"
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m ruff check src tests
.venv\Scripts\orchestrator-research demo                          # учебный прогон без GPU и сети
.venv\Scripts\orchestrator-research train --config experiments/template/config.toml --dry-run
```

`train --dry-run` печатает словарь overrides и ничего не записывает. Обучение на Windows не запускается:
агент контроллера `runner_type=local` работает только на POSIX. Тесты и демо на Windows работают.

## Структура репозитория

★ — читать первыми. Подробный порядок чтения: `docs/code_walkthrough.md`.

```
orchestrator-research/
├── .gitattributes  — правила переводов строк git
├── .gitignore  — игнорируемые файлы (.venv, артефакты, кэши)
├── AGENTS.md  — инженерные правила проекта
├── README.md  — ★ этот файл: запуск, структура, новые эксперименты и стратегии
├── pyproject.toml  — зависимости, extras (benchmark, train, dev) и точка входа orchestrator-research
├── artifacts/
│   └── .gitkeep  — папка артефактов (чекпоинты, прогоны, логи); содержимое игнорируется git
├── configs/
│   ├── base.toml  — ★ общая база всех экспериментов: limits, orchestrator, subagents, judge, embedding, retrieval, benchmark, reward, training
│   └── subagent_prompts.toml  — системные промпты ролей субагентов (retriever, fact_explorer, evidence_reader, entity_resolver)
├── docs/
│   ├── adding_hypothesis.md  — как проверить новую гипотезу: таблица настроек конфига и примеры
│   ├── code_walkthrough.md  — ★ поток одного шага обучения и одной оценки; слои и порядок чтения кода
│   ├── deployment.md  — развёртывание на общей GPU 80 ГБ: память, OpenRouter, порты, сервисы, логи, неполадки
│   ├── reference_algorithm.md  — формулы лосса K2.5, группового преимущества и награды PARL с источниками
│   └── adrs/
│       ├── 0001-separate-project.md  — отдельный проект, а не форк MultiAgents
│       ├── 0002-generation-budget.md  — бюджет считается по сгенерированным токенам
│       ├── 0005-agent-lightning.md  — обучение через Agent Lightning и verl без форка verl
│       └── 0006-retrieval-server.md  — поиск как общий HTTP-сервис
├── experiments/
│   ├── registry.md  — реестр экспериментов и их статус
│   └── template/
│       ├── README.md  — гипотеза, фиксированные и меняемые переменные эксперимента 001
│       └── config.toml  — ★ конфиг эксперимента 001: стратегия, лимиты, обучение (наследует base.toml)
├── scripts/
│   ├── bootstrap_server.sh  — установка на сервере: .venv, vLLM, verl, flash-attn, проверка тестами и конфига
│   ├── eval.sh  — оценка базы или чекпоинта на одном сплите: слияние FSDP, vLLM, eval, остановка vLLM
│   ├── lib.sh  — общие функции скриптов (ожидание HTTP, остановка по pid, чтение конфига, проверка локального URL); не запускается
│   ├── services.sh  — start/stop/status для сервера поиска и для эмбеддера, если его URL локальный (pid и логи в artifacts/services/)
│   ├── smoke_train.sh  — прогон train.sh на 2 шагах с замером пика памяти GPU
│   └── train.sh  — обучение: agl-server, agl-controller, затем orchestrator-research train
├── src/
│   └── orchestrator_research/
│       ├── __init__.py  — пакет, версия
│       ├── agent.py  — ★ эпизод одного роллаута Agent Lightning: запуск, файл эпизода, отправка награды
│       ├── algorithm.py  — численные эталоны без autograd: групповое преимущество и лосс K2.5
│       ├── application.py  — validate-config, prepare-data и проверки doctor (без печати секретов)
│       ├── assembly.py  — ★ единственное место сборки адаптеров, стратегии, судьи и награды: run_episode
│       ├── benchmark.py  — данные BrowseComp-Plus: расшифровка, границы сплитов train/val_test/val_full, parquet
│       ├── cli.py  — команды orchestrator-research: validate-config, prepare-data, serve-retrieval, doctor, train, eval, report, compare, demo
│       ├── config.py  — ★ строгие модели конфига (pydantic), наследование include, роли субагентов из TOML
│       ├── demo.py  — учебный прогон без GPU, сети и ключей: бюджет, эпизод swarm, награда, K2.5
│       ├── domain.py  — неизменяемые записи и протоколы зависимостей: ChatModel, Retriever, Judge, Strategy
│       ├── eval_runner.py  — оценка сплита: параллельно, с возобновлением, append-only JSONL
│       ├── limits.py  — ★ бюджеты токенов и шагов: резерв перед вызовом, затем списание по usage
│       ├── report.py  — сводки по сплитам и шагам, парное сравнение (McNemar, bootstrap)
│       ├── reward.py  — ★ награда PARL: r_perf + λ-члены с отжигом; на валидации только r_perf
│       ├── storage.py  — атомарная запись JSON, JSONL с fsync, чтение run-папок
│       ├── subagents.py  — ★ субагенты: цикл одного с собственным бюджетом и пул экземпляров
│       ├── tools.py  — схемы инструментов, сборка промптов из конфига, инструмент browsecomp_search
│       ├── adapters/
│       │   ├── __init__.py  — пакет адаптеров (ядро из них не импортирует)
│       │   ├── chat.py  — OpenAI-совместимый асинхронный клиент: повторы через SDK, учёт usage
│       │   ├── fake.py  — детерминированные фейковые модель и поиск для тестов и smoke
│       │   ├── judge.py  — судья: запрос, разбор JSON-вердикта, явные статусы ошибок
│       │   ├── retrieval.py  — FAISS-индекс и корпус, эмбеддинги запроса, HTTP-клиент поиска
│       │   └── retrieval_server.py  — HTTP-сервер поиска: /healthz и /search, индекс загружается один раз
│       ├── strategies/
│       │   ├── __init__.py  — регистр STRATEGIES (swarm, single) и get_strategy
│       │   └── base.py  — ★ общий цикл оркестратора, финализатор, окружение эпизода
│       └── training/
│           ├── __init__.py  — пакет обучения: единственный, где импортируются torch, verl и agentlightning
│           ├── hooks.py  — хук agl: проставляет номер шага train_step на роллаутах
│           ├── k25_loss.py  — ★ лосс K2.5 как плагин verl и регистрация в Ray-воркерах
│           ├── overrides.py  — ★ словарь настроек verl и agl из конфига; endpoint agl; счётчик шагов
│           └── train.py  — точка входа обучения: конфиг, run-папка, run_ppo, --smoke, --dry-run
└── tests/
    ├── conftest.py  — общие детерминированные фейки и фикстуры
    ├── test_agent.py  — эпизод агента
    ├── test_algorithms.py  — групповое преимущество и эталон K2.5
    ├── test_application.py  — doctor и config-summary
    ├── test_browsecomp.py  — расшифровка вопросов и границы сплитов
    ├── test_cli.py  — команды CLI
    ├── test_config.py  — конфиг и include
    ├── test_demo.py  — демо
    ├── test_eval_runner.py  — оценка и возобновление
    ├── test_judge.py  — разбор вердикта судьи
    ├── test_k25_loss.py  — лосс K2.5 против эталона (нужен torch, иначе пропуск)
    ├── test_limits.py  — инварианты бюджета
    ├── test_openai_adapter.py  — чат-клиент
    ├── test_report.py  — отчёты и парное сравнение
    ├── test_retrieval.py  — контракт запроса и сниппетов поиска
    ├── test_retrieval_server.py  — HTTP-сервер поиска
    ├── test_reward.py  — инварианты награды и отжига
    ├── test_storage.py  — атомарная запись и JSONL
    ├── test_strategies.py  — регистр стратегий и цикл
    ├── test_subagents.py  — пул субагентов
    ├── test_tools.py  — инструмент поиска и аргументы
    └── test_training_overrides.py  — словарь overrides, endpoint agl и dry-run
```

## Порядок чтения

1. `configs/base.toml` и `experiments/template/config.toml`: какие настройки есть и что меняет эксперимент.
2. `docs/code_walkthrough.md`: слои, правила зависимостей, путь одного шага обучения и одной оценки.
3. `src/orchestrator_research/assembly.py` и `strategies/base.py`: как собирается эпизод.
4. `limits.py`, `reward.py`, `subagents.py`: бюджеты, награда и субагенты.
5. `docs/reference_algorithm.md`: формулы с источниками.

## Новый эксперимент

Эксперимент — это папка с одним `config.toml`; база `configs/base.toml` общая. Скопируйте
`experiments/template` в `experiments/002_<короткое_имя>`, поменяйте `[experiment] name` (должен
совпадать с именем папки) и значения, которые проверяете, меняя одну переменную относительно базы. Проверьте
конфиг командой `orchestrator-research validate-config --config experiments/002_.../config.toml`, затем
обновите `README.md` эксперимента и `experiments/registry.md`. Подробности: `docs/adding_hypothesis.md`.

Новая стратегия — это функция вида `factory(env: EpisodeEnvironment) -> Strategy` в
`src/orchestrator_research/strategies/__init__.py` (по образцу `build_swarm` и `build_single`) и одна запись в
словаре `STRATEGIES`. После этого имя стратегии можно указать в `[experiment] strategy`. Стратегия должна
использовать общий `OrchestratorLoop` и не обходить бюджет `TaskBudget`; добавьте тесты в
`tests/test_strategies.py`. Подробности: `docs/adding_hypothesis.md`.
