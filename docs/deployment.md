# Развёртывание: один сервер, общая GPU 80 ГБ

Эксплуатация: требования, память, порты, сервисы и логи, диагностика. Команды быстрого старта — в `README.md`.

Целевой сервер: Linux, одна NVIDIA GPU 80 ГБ, которую делят с другими процессами (свободно около 20 ГБ),
не меньше 128 ГБ RAM, CUDA 12.9 или 13.0, Python 3.12, `uv`. Оркестратор (Qwen3.5-2B) обучается и оценивается на
этой карте. В эксперименте 001 субагенты, судья и эмбеддинги работают через OpenRouter; в 000 и 002 всё локально
(OpenRouter с сервера заблокирован, см. «Известные ограничения сервера»).

### OpenRouter в трёх строках

- Один OpenAI-совместимый API ко всем моделям из его каталога; меняется только `model` в конфиге.
- Один ключ `OPENROUTER_API_KEY` для субагентов, судьи и эмбеддингов; оплата за токены.
- `doctor` проверяет, что модели субагентов и судьи есть в каталоге, и делает один запрос эмбеддинга (`input "ping"`).

Модели: субагенты `mistralai/ministral-3b-2512` (small), `qwen/qwen-2.5-7b-instruct` (medium), `qwen/qwen3.5-9b`
(large); судья `qwen/qwen3.5-9b`; эмбеддинги запросов `qwen/qwen3-embedding-8b`. FAISS-индекс Tevatron
`qwen3-embedding-8b` работает на CPU внутри retrieval-сервера.

## Скрипты и порядок

| Скрипт | Что делает |
|---|---|
| `scripts/bootstrap_server.sh` | окружение и GPU-стек: vLLM (`VLLM_VERSION`, по умолчанию 0.22.0), verl 0.8.0, flash-attn; `CUDA_VARIANT` cu130 или cu129 |
| `scripts/services.sh start\|stop\|status CONFIG` | retrieval-сервер (CPU) всегда; эмбеддер (vLLM, pooling) только если `[embedding] base_url` указывает на `127.0.0.1` или `localhost`, иначе печатается `embedder: remote (...)` и он пропускается; pid и логи в `artifacts/services/<name>.pid` и `<name>.log`; повторный `start` не трогает живые процессы |
| `scripts/train.sh CONFIG [--smoke] [--run-id ID]` | проверка ключей, `doctor`, `agl-server`, `agl-controller`, `orchestrator-research train` |
| `scripts/smoke_train.sh CONFIG` | `train.sh --smoke` плюс замер пика памяти GPU |
| `scripts/eval.sh CONFIG base\|CHECKPOINT_DIR SPLIT RUN_ID` | при необходимости слияние FSDP, vLLM оценки, `orchestrator-research eval` |
| `scripts/services.sh` | см. выше: ports 18xxx, KV-кэш задан явно |
| `scripts/auto_smoke.sh CONFIG` | очередь: ждёт карту с `NEED_GB` (20) свободных ГБ и `NEED_RAM_GB` (30) свободной RAM до `WAIT_HOURS` (12), затем services → doctor → smoke_train; сервисы останавливаются всегда (`trap EXIT`). Ждёт, пока в tmux не исчезнет сессия `fa` (сборка flash-attn) |
| `scripts/ram_guard.sh SESSION...` | защита RAM на сервере: каждые `INTERVAL` (5) с проверяет доступную RAM; при `< MIN_GB` (6) останавливает только наши tmux-сессии из аргументов и наши процессы пользователя (nvcc, ninja, ray, vllm, agl). Срабатывает один раз и выходит; лог `artifacts/ram_guard.log` |
| `scripts/build_flash_attn.sh` | flash-attn 2.8.3 из исходников, только A100 (`TORCH_CUDA_ARCH_LIST=8.0`), `MAX_JOBS` (по умолчанию 2, ~8 ГБ RAM на задачу), около 3 ч |
| `scripts/gpu_free.sh` | только читает: свободная память по картам и номер самой свободной |

`doctor` запускается из `scripts/train.sh` (шаг 2; при FAIL обучение не стартует) и вручную:
`.venv/bin/orchestrator-research doctor --config CFG`. Из `bootstrap_server.sh` и `services.sh` он не вызывается.
Проверяет: переменные ключей (только имена), список моделей субагентов и судьи у провайдера, наличие данных,
`/healthz` поиска, эмбеддер, GPU, версию Python.

- Эмбеддер удалённый (не `127.0.0.1` и не `localhost`): проверяется, что задана переменная `[embedding] api_key_env`,
  и делается один запрос эмбеддинга с `input "ping"`. Значение ключа не печатается.
- Эмбеддер локальный: проверяется `/models` по `[embedding] base_url`.
- GPU: `nvidia-smi` по GPU 0. FAIL, если свободно меньше, чем `training.rollout_gpu_memory_utilization × всего + 6 ГБ`
  (запас под фазу обучения). WARN, если `nvidia-smi` нет.

## Память на общей карте 80 ГБ (оценка, не измерение)

Карта общая, свободно около 20 ГБ. `gpu_memory_utilization` у vLLM — это доля **всей** карты, а не свободной памяти:
0.15 × 80 ГБ ≈ 12 ГБ. vLLM отказывается стартовать, если столько памяти не свободно.

| Фаза | Память | Откуда |
|---|---|---|
| Роллаут vLLM у verl (оркестратор 2B) | ~12 ГБ | `training.rollout_gpu_memory_utilization = 0.15` × 80 ГБ |
| Фаза обучения FSDP2 с offload (vLLM спит) | ~6–8 ГБ | оценка; offload включён в `training/overrides.py` |
| **Пик** | **≈ 12–14 ГБ** | оценка, не проверено; `smoke_train.sh` печатает фактический пик из `nvidia-smi` |
| Субагенты и эмбеддер, локальные vLLM (только 000 и 002) | ≈ 8 ГБ и ≈ 4 ГБ на карте | `services.sh`: `FROZEN_GPU_UTIL` 0.10, `EMBED_GPU_UTIL` 0.05 (× 80 ГБ); KV-кэш явно 3 ГБ и 1 ГБ. Удалённые (001) на карте 0 ГБ |

`doctor` требует свободно не меньше 0.15 × 80 + 6 ≈ 18 ГБ, при около 20 ГБ свободно проверка проходит.

Если doctor или vLLM жалуется на память, снижайте по порядку:

1. `training.rollout_gpu_memory_utilization` (0.15 → 0.12).
2. `training.max_context_tokens` и связанный `max_prompt_tokens` (меньше KV-кэш; соблюдайте проверку длин).
3. `training.train_batch_size` и `ppo_mini_batch_size` (меньше батч, та же форма шага).

`training.agent_concurrency` на пик GPU не влияет: он задаёт число параллельных процессов роллаутов и давление на RAM и сеть.

Внимание: `scripts/eval.sh` (по тексту скрипта, на сервере с этой версией не проверено) берёт
`--gpu-memory-utilization` из `training.rollout_gpu_memory_utilization` (0.15 ≈ 12 ГБ), без явного
`--kv-cache-memory-bytes` и `--max-num-batched-tokens`. Из-за учёта чужой памяти при профилировании оценка может
не стартовать; см. «Известные ограничения сервера».

## Порты

Порты 8000, 8001, 8003 и 8080 заняты другими пользователями, поэтому все наши сервисы на 18xxx.

| Порт | Сервис | Задаётся в | Кто ходит |
|---|---|---|---|
| 18080 | agl-server (шлюз оркестратора, `/healthz`) | `AGL_PORT` в `train.sh` (по умолчанию 18080) | агент и контроллер |
| 18100 | retrieval-сервер (`/healthz`, `/search`), CPU | `[retrieval] server_url` | агент, субагенты, `doctor` |
| 18001 | замороженные субагенты и судья, vLLM (`/v1/models`), только если `[subagents] base_url` локальный (000, 002) | `[subagents] base_url`, `FROZEN_GPU_UTIL` | субагенты, судья |
| 18003 | эмбеддер vLLM (`/v1/models`), только если `[embedding] base_url` локальный (000, 002) | `[embedding] base_url`, `EMBED_GPU_UTIL` | retrieval-сервер |
| — | эмбеддер и субагенты удалённые (`https://openrouter.ai/api/v1`), локального порта нет (001) | `base_url` в конфиге | retrieval-сервер, `doctor` |
| 18000 | vLLM оценки под именем `orchestrator` | `[orchestrator] eval_base_url` | `orchestrator-research eval` |

Порты обучающего vLLM внутри verl выбирает сам verl; проект их не настраивает.

## Известные ограничения сервера

| Ограничение | Что известно | Что делать |
|---|---|---|
| Общая GPU, vLLM считает чужую память | При профилировании vLLM учитывает аллокации других пользователей, поэтому `--gpu-memory-utilization` ненадёжен | `services.sh`: явный `--kv-cache-memory-bytes` (`FROZEN_KV_BYTES` 3 ГБ, `EMBED_KV_BYTES` 1 ГБ); rollout: `actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes` (в 002 — 2 ГБ) |
| Лимиты vLLM для Qwen3.5 (гибридный Mamba-кэш) | Без лимитов кэш не помещается | `--max-num-batched-tokens 2048`, `--max-num-seqs 64` (в `services.sh` и `extra_overrides`) |
| Общая RAM 196 ГБ на весь сервер | В прогоне 2B node RAM 187 из 197 ГБ занята чужими процессами. Монитор памяти Ray убивает НАШИХ воркеров, когда RAM > 95% | `scripts/ram_guard.sh` (MIN_GB 6) в tmux; Ray: `RAY_ADDRESS=local`, `RAY_TMPDIR=~/.ray_tmp`, `num_cpus=8`, `object_store_memory=4 ГБ` (`configs/base.toml`). Чужой кластер Ray не используем |
| Полное обучение по памяти | ~16 байт на параметр (fp32 веса, градиенты, Adam m и v): 2B ≈ 32 ГБ, 4B ≈ 64 ГБ, на GPU или в RAM (offload) | 0.8B помещается; 2B на грани; полный fine-tuning 4B не помещается, нужен LoRA |
| OpenRouter с сервера | Cloudflare 403 «Access denied by security policy», даже без ключа | 000 и 002 локально; 001 через `HTTPS_PROXY` или другой провайдер (спросить пользователя) |
| Порядок установки | vllm 0.22.0 и verl 0.8.0 ставятся отдельными шагами (конфликт numpy); pyarrow < 22; flash-attn нужен verl даже при `sdpa` | `bootstrap_server.sh`; flash-attn через `build_flash_attn.sh` (`MAX_JOBS=2`, около 3 ч) |
| torch.compile в vLLM | Нужен `ninja` из `.venv` | `lib.sh` добавляет `.venv/bin` в PATH |
| Статус проверки | Конвейер проверен до шага обновления актора; сам шаг обновления и смоук 0.8B (`000`) не подтверждены | см. `README.md`, раздел «Статус» |

## Логи и файлы

- Сервисы: `artifacts/services/<name>.log`, `<name>.pid`. Лог vLLM оценки: `artifacts/services/eval-<run_id>.vllm.log`.
- Обучение: `artifacts/runs/<experiment>/<run_id>.train.log`, `<run_id>.agl-server.log`, `<run_id>.agl-controller.log`.
  Лежат рядом с run-папкой, потому что `train.py` не переиспользует существующую папку.
- Run-папка `artifacts/runs/<experiment>/<run_id>/`: `config.resolved.json`, `manifest.json`,
  `trainer_overrides.json`, `episodes/` (обучение и валидация). Оценка: `eval/config.json`, `eval/<split>.jsonl`.
- Чекпоинты: `artifacts/checkpoints/<experiment>/<run_id>/global_step_N/` (шарды `actor/`, после слияния `hf/`).
- Smoke: `artifacts/smoke/<run_id>.gpu.csv` (сэмплер памяти GPU).
- Ключи в файлы не пишутся: `AGL_KEY` передаётся через окружение, а `trainer_overrides.json` записывается до
  добавления ключа.

## Устранение неполадок

| Симптом | Что сделать |
|---|---|
| `doctor`: API key не задан | экспортируйте переменную из `api_key_env` (по умолчанию `OPENROUTER_API_KEY`) |
| `doctor`: модель не найдена в списке провайдера | исправьте `[subagents.models]` или `[judge] model` |
| `doctor`: retrieval server | `scripts/services.sh start CONFIG`; первый старт грузит индекс несколько минут |
| `doctor`: embedding API key | `[embedding] api_key_env` указывает на незаданную переменную; экспортируйте её (значение не печатается) |
| `doctor`: embedding call | удалённый эмбеддер не ответил на `ping`: проверьте `[embedding] base_url`, `model` и ключ; модель должна быть в `<base_url>/models` |
| `doctor`: embedding server | локальный эмбеддер не поднят или `[embedding] base_url` не тот (`services.sh start CONFIG`) |
| `doctor`: GPU FAIL | свободно меньше, чем нужно; уменьшите `training.rollout_gpu_memory_utilization` или `max_context_tokens`, проверьте `nvidia-smi` на чужие процессы |
| `doctor`: данные | выполните `orchestrator-research prepare-data --config CFG` |
| `run directory already exists` | выберите другой `--run-id`; run-папки не перезаписываются |
| `AGL_KEY is empty` или `AGL_SERVER_URL is empty` | `train.sh` задаёт их сам; при ручном запуске экспортируйте |
| agl: «model not found» | `default_proxy.model_name` у agl-server должен совпадать с `orchestrator.model_path` (так делает `train.sh`) |
| `runner_type=local` не стартует | контроллер работает только на POSIX (Linux), на Windows обучение не запускается |
| ошибки длины контекста в роллаутах | поднимите `training.max_context_tokens` (с учётом `max_prompt_tokens` и бюджета) или уменьшите `subagents.return_max_chars` |
| OpenRouter 429 | уменьшите `training.agent_concurrency` или `limits.max_parallel` (`adding_hypothesis.md`); повторы задаёт `subagents.max_retries` |
| `flash-attn` не собирается | нужен CUDA toolkit того же минора, что у torch; для cu130 `bootstrap_server.sh` ставит toolkit из pip |
| сбой между шагами, остались процессы | `ray stop --force`; `scripts/services.sh stop CONFIG`; `pkill -f agl-controller`, только если других экземпляров нет |
| оценка прервана | повторите команду с тем же `--run-id`: пригодные ответы пропускаются (`code_walkthrough.md`, раздел 3) |
| `compare` отказывается работать | у двух файлов разные множества успешно оценённых ID; доведите оба прогона до полного набора |

## Проверено и не проверено

Проверено по источникам (версии в скобках):

- `agl-server` и `agl-controller` принимают `key`, `port`, `default_proxy.model_name`, `runner_type=local`,
  `agl_server.url`, `agl_server.key`, `local_runner.maximum_size` (agl v1.0.2).
- Пары `agentlightning.agl_base_url` и `agentlightning.agl_key` для трейнера (`train_search_r1_agent.py`, v1.0.2).
- Порядок установки: vLLM до verl, затем flash-attn из исходников (agl `docs/00-installation.md`, `scripts/setup_verl.sh`).
- `python -m verl.model_merger merge --backend fsdp --local_dir … --target_dir …` (verl v0.8.0).
- Эмбеддер: `vllm serve … --runner pooling` (документация vLLM, pooling models).

Решено (ADR 0005):

- **Версия vLLM: 0.22.0.** Документация agl упоминает 0.20.2, но `scripts/setup_verl.sh` agl для verl 0.8.0
  ставит 0.22.0. Переменная `VLLM_VERSION` в `bootstrap_server.sh` позволяет поменять версию.
- **Парсер инструментов: `qwen3_coder`.** Задан в `training/overrides.py` и по умолчанию в `eval.sh`
  (переменная `TOOL_CALL_PARSER`). Если vLLM не распознаёт вызовы, поменяйте парсер в `eval.sh` или ключом
  `actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser` в `training.extra_overrides`.

Не проверено:

1. **Ключи FSDP offload** (`actor_rollout_ref.actor.fsdp_config.param_offload`, `optimizer_offload`) взяты из
   примера Search-R1. Конфиг verl 0.8.0 не сверён. Место отмечено комментарием `Unverified` в `training/overrides.py`.
2. **`verl[mcore]`.** `bootstrap_server.sh` ставит с extra, как agl `setup_verl.sh`. Если mcore не нужен,
   можно поставить `verl==0.8.0` без extra.
3. **Память.** Все цифры выше оценки до первого прогона. Фактический пик покажет `smoke_train.sh`.
4. **Шаблон Qwen3.5 и префиксы.** Если шаблон вырезает thinking прошлых ходов, rollout делится на строки
   (`training/n_unmerged_rollouts`). Частоту этого на реальных данных нужно замерить.

## Второй сервер: mipt-vpn (с 2026-10-17)

- Вход: сначала OpenVPN (профиль `Personal-2.0`, запасной `Personal-2.0-TCP`, файлы в `C:\Projects\VPN`), затем
  `ssh mipt-vpn` (адрес, порт и логин — только в `~/.ssh/config`, ключ `~/.ssh/id_ed25519`).
- Железо: 8×H200 141 ГБ (sm_90), 2 ТБ RAM, 192 ядра, Ubuntu 24.04 в контейнере. Всё общее: карты обычно заняты
  чужими задачами, свободно 7–80 ГБ на карту. Домашняя папка `/data/users/<user>`, на `/data` свободно ~0.5 ТБ на всех.
- Порт 18080 занят чужим процессом: agl-server запускаем с `AGL_PORT=18081`. Наши: 18000/18001/18003/18100, прокси 18090.
- OpenRouter/OpenAI отвечают 403 (выход из РФ). API-вызовы идут через user-space прокси xray (ADR 0007):
  `bash scripts/set_secret.sh ~/.vpn_sub` (один раз, вводит пользователь), `bash scripts/vpn_proxy.sh start`,
  затем `export USE_PROXY=1` перед `services.sh`/`train.sh`/`doctor`.
- Сборка: `TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=8 bash scripts/bootstrap_server.sh` (только sm_90; RAM хватает на 8 задач nvcc).
