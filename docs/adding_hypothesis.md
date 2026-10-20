# Как проверить новую гипотезу

Эксперимент — каталог `experiments/NNN_name/` с файлами `config.toml` и `README.md` (соглашение, см. AGENTS.md).
`config.toml` подключает `configs/base.toml` через `include` и меняет только свои значения.

Все ключи, единицы и значения по умолчанию описаны комментариями в `configs/base.toml`. Типы и проверки
— в `src/orchestrator_research/config.py`. Таблицы ключей здесь нет, чтобы не расходиться с кодом.

## Порядок

1. `cp -r experiments/template experiments/002_<name>` и поменяйте `[experiment] name` на имя каталога.
2. Измените одну переменную относительно базы.
3. Проверьте конфиг: `.venv/bin/orchestrator-research validate-config --config experiments/002_<name>/config.toml`.
   Команда печатает имя, стратегию, хэш конфига, лимиты, пул субагентов, модель судьи и адрес retrieval-сервера.
   Имя стратегии она не проверяет: неизвестное имя отвергает `get_strategy` при первом эпизоде.
4. Обновите `experiments/002_<name>/README.md`: гипотеза, что фиксировано, что меняется, результаты.
5. Запустите оценку и обучение, как в быстром старте в `README.md`, с новым ID прогона. У `scripts/train.sh` это
   `--run-id`, у `scripts/eval.sh` — последний позиционный аргумент `RUN_ID`. Прогоны не перезаписываются.

## Примеры

### Лимиты

```toml
[limits]
orchestrator_tokens_per_task = 4096
max_turns = 8
```

`training.max_context_tokens` должен быть больше `max_prompt_tokens + orchestrator_tokens_per_task`, а
`orchestrator_tokens_per_task` больше `final_answer_reserve`. Если `orchestrator_tokens_per_task` не задан, в
проверке используется `orchestrator_tokens_per_turn * max_turns`. При нарушении `validate-config` выводит
ошибку с подсказкой и точным минимальным значением.

### Пул субагентов

```toml
[subagents.models]
medium = "qwen/qwen3.5-9b"
```

Таблица `models` сливается с базой по ключам: алиасы, которых нет в эксперименте, остаются. Алиас
соответствует `^[a-z][a-z0-9_]*$`. Каждый `model_id` должен быть доступен у провайдера: `doctor` проверяет
список моделей. Роли субагентов задаются таблицей `[roles]` в `configs/subagent_prompts.toml`. Схема инструмента
`create_subagent` строится из этого файла, поэтому новая роль добавляется строкой туда.

### Стратегия single

```toml
[experiment]
name = "002_single_baseline"
strategy = "single"
```

`single` не создаёт субагентов: оркестратор только ищет сам. Секция `[subagents]` остаётся обязательной, пул
не используется. Для честного сравнения меняйте только стратегию. Компонента `r_parallel` для `single` равна 0,
поэтому различие видно в `r_perf`.

### Новая стратегия

В `src/orchestrator_research/strategies/__init__.py` добавьте фабрику и строку в словарь `STRATEGIES`:

```python
def build_fanout(env: EpisodeEnvironment) -> Strategy:
    """Fanout: subagents allowed, the same loop as swarm (example)."""
    return OrchestratorLoop(env, name="fanout", subagents=True)


STRATEGIES: dict[str, StrategyFactory] = {
    "swarm": build_swarm,
    "single": build_single,
    "fanout": build_fanout,  # новая строка
}
```

Аргумент `name` должен совпадать с ключом словаря: он попадает в результат эпизода (`strategy=self._name`).
Код это не проверяет. Затем укажите `strategy = "fanout"` в конфиге эксперимента. Если нужен другой цикл,
наследуйте `OrchestratorLoop` и переопределите `run`. Фабрика получает только `EpisodeEnvironment`; бюджет
(`TaskBudget`) и лимиты лежат в его полях `budget` и `limits`, поэтому учёт работает без изменений. Тест `tests/test_strategies.py` прогоняет каждую стратегию из `STRATEGIES`.

### Аварийный люк: training.extra_overrides

```toml
[training.extra_overrides]
"actor_rollout_ref.actor.ppo_epochs" = 2   # пример; ключ сверьте с ppo_trainer.yaml verl 0.8.0
```

Ключи — точечные пути verl или agl, в кавычках. Значения сливаются поверх сгенерированного словаря и
побеждают его. Имена ключей записываются в `manifest.json` (`extra_overrides_keys`). Проверка длин идёт уже
после слияния. Используйте этот люк, только если нужного ключа нет среди полей `[training]`: иначе эксперимент
перестаёт читаться по конфигу.

## Нагрузка на OpenRouter при обучении

Каждый роллаут — отдельный процесс со своим семафором `subagents.max_concurrency`, поэтому ограничение
действует на процесс, а не на весь запуск. Одновременных запросов к OpenRouter при обучении примерно
`training.agent_concurrency × limits.max_parallel` (верхняя оценка; по умолчанию 64 × 8 = 512), плюс запросы судьи. При
ошибках 429 уменьшайте `agent_concurrency` или `max_parallel`; повторы задаёт `subagents.max_retries`.
