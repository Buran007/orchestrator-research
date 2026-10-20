# Шпаргалка: сервер Ubuntu, общий GPU, запуск проекта

Сервер: `ssh brain-lab` (адрес, порт и логин — в `~/.ssh/config`). 4× A100 80 ГБ, карты **общие** с другими людьми.

## Правила, чтобы никому ничего не сломать
- Работаем только в своей папке `~`. Никакого `sudo`, не трогаем системный Python и чужие conda-окружения.
- **Никогда**: `kill`/`pkill` чужих процессов, `nvidia-smi -r`, `nvidia-smi -pm`, `rm -rf` вне своей папки.
- Всегда задаём **одну** карту: `export CUDA_VISIBLE_DEVICES=N`. Без этого verl/vLLM возьмут GPU 0.
- Наш vLLM берёт фиксированную долю карты: `training.rollout_gpu_memory_utilization × 80 ГБ`
  (0.15 ≈ 12 ГБ), а KV-кэш задан явно в байтах. Если столько не свободно, он **не стартует** и никого не вытесняет.
- RAM общая на весь сервер (196 ГБ). Тяжёлые задачи запускаем только вместе с `ram_guard.sh` (ниже).
- Останавливаем только своё: `bash scripts/services.sh stop <config>` и Ctrl+C в своём tmux.
- Порты наших сервисов: 18000 (оценка), 18001 (субагенты и судья), 18003 (эмбеддер), 18080 (agl-server), 18100 (поиск). Порты 8000/8001/8003/8080 заняты другими.

## Базовые команды Ubuntu
| Команда | Что делает |
|---|---|
| `pwd`, `ls -la`, `cd ~/orchestrator-research` | где я / что в папке / перейти |
| `nvidia-smi` | все GPU: память, загрузка, чьи процессы |
| `bash scripts/gpu_free.sh` | свободная память по картам + какая свободнее (только читает) |
| `free -g`, `df -h ~`, `nproc` | RAM, место на диске, число CPU |
| `ps -u $USER -f` | **мои** процессы |
| `tail -f artifacts/runs/<exp>/<run>.train.log` | смотреть лог в реальном времени (выход Ctrl+C) |
| `less файл` | читать файл (выход `q`) |
| `export X=...` | переменная окружения в текущей сессии |

## tmux: чтобы обучение не умерло при обрыве ssh / севшем ноутбуке
| Команда | Что делает |
|---|---|
| `tmux new -s orch` | новая сессия `orch` |
| Ctrl+B, затем D | отсоединиться (всё продолжает работать) |
| `tmux attach -t orch` | вернуться в сессию |
| `tmux ls` | список своих сессий |
| Ctrl+B, затем C / N | новое окно / следующее окно |

## Выбор видеокарты (каждый раз перед запуском)
```bash
bash scripts/gpu_free.sh                 # смотрим, где свободнее (нужно ≥ ~20 ГБ)
export CUDA_VISIBLE_DEVICES=2            # подставь номер из вывода
```
Номер внутри процесса всегда будет 0 — это нормально: процесс видит только выбранную карту.
`doctor` проверяет память именно на этой карте.

## Защита RAM и сборка flash-attn (в своих tmux-сессиях)
```bash
tmux new -d -s guard "MIN_GB=6 bash scripts/ram_guard.sh fa auto"   # при RAM < 6 ГБ останавливает только наши сессии fa/auto и наши процессы
tmux new -d -s fa "MAX_JOBS=2 bash scripts/build_flash_attn.sh > artifacts/flash_attn_build.log 2>&1; echo EXIT=\$? >> artifacts/flash_attn_build.log"   # ~3 ч, ~8 ГБ RAM на задачу
tail -f artifacts/ram_guard.log                        # что сделал guard (выход Ctrl+C)
```
`ram_guard.sh` срабатывает один раз и завершается. Сессия `fa` нужна, потому что `auto_smoke.sh` ждёт её окончания.

## Очередь смоука (без присмотра)
```bash
tmux new -d -s auto "bash scripts/auto_smoke.sh experiments/000_smoke_local/config.toml > artifacts/auto_smoke.log 2>&1"
tail -f artifacts/auto_smoke.log                       # ждёт карту с ≥ 20 ГБ и RAM ≥ 30 ГБ, затем сервисы → doctor → smoke
```
Переменные: `NEED_GB`, `NEED_RAM_GB`, `WAIT_HOURS` (по умолчанию 20, 30, 12), `FROZEN_GPU_UTIL`, `EMBED_GPU_UTIL`, `FROZEN_KV_BYTES`.
Сервисы останавливаются в любом случае (даже после ошибки).

## Первый запуск (по порядку, всё в tmux)
Эксперимент 001 требует прокси (`HTTPS_PROXY`), OpenRouter с сервера заблокирован. Без прокси используйте
`experiments/000_smoke_local`: он полностью локальный.
```bash
# 0. код на сервер (один раз): git clone <repo> ~/orchestrator-research  или  scp архива
cd ~/orchestrator-research
bash scripts/bootstrap_server.sh                       # окружение .venv в папке проекта (~20–30 мин)
source .venv/bin/activate                              # включить окружение (каждая новая сессия)
export OPENROUTER_API_KEY=...                          # ключ; в файлы не пишем (для 001)
export CUDA_VISIBLE_DEVICES=2                          # выбранная карта
CFG=experiments/000_smoke_local/config.toml
orchestrator-research prepare-data --config $CFG       # данные + индекс (~2 ГБ, один раз)
bash scripts/services.sh start $CFG                    # поиск (CPU) и локальные vLLM (18001, 18003)
orchestrator-research doctor --config $CFG             # всё должно быть OK
bash scripts/smoke_train.sh $CFG                       # 2 шага обучения, печатает пик памяти
bash scripts/train.sh $CFG --run-id run1               # полное обучение (AGL на 18080)
```
Остановить сервисы: `bash scripts/services.sh stop $CFG`. Статус: `bash scripts/services.sh status $CFG`.

## Если что-то не так
| Симптом | Что делать |
|---|---|
| `doctor`: GPU FAIL | карта занята → `gpu_free.sh`, другая карта или меньше `training.rollout_gpu_memory_utilization` |
| vLLM: `not enough memory` / CUDA OOM | то же; также можно уменьшить `training.max_context_tokens` |
| OpenRouter 429 или 403 (Cloudflare) | 429: меньше `training.agent_concurrency` или `limits.max_parallel`; 403: сервер заблокирован, нужен прокси (001) или 000/002 |
| Порт занят | наши сервисы на 18xxx; проверьте конфиг, не меняйте чужие процессы |
| Ray убил наших воркеров (RAM > 95%) | запускайте `ram_guard.sh`; снизьте `ray_init` в `configs/base.toml` или `agent_concurrency`; `ray stop --force` |
| Оценка не стартует из-за памяти | см. `docs/deployment.md`, «Известные ограничения сервера» |
| ssh оборвался | `ssh ...`, затем `tmux attach -t orch` |
