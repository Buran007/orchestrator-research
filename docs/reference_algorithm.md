# Справочник алгоритма

Три части: лосс K2.5 (политика), групповое преимущество без нормировки на std и награда PARL с отжигом.
Для каждой указано, где формула в коде и откуда взята. Любое изменение формулы требует источника в этом
документе и тестов на инварианты (AGENTS.md, правило 3).

## 1. Лосс K2.5 (token-clipped policy loss)

Для токена `t` с маской ответа `m_t`, log-ratio `r_t = log π_θ − log π_old`, обрезанным до `[−20, 20]`
(`LOG_RATIO_CLAMP`), и полосой `[α, β]`, `α < 0 < β`:

```
g_t    = 1  если  α ≤ r_t ≤ β,  иначе 0          (через g_t градиент не идёт)
L_pg   = agg( −A_t · r_t · g_t · m_t )            (агрегация по токенам в полосе)
L_tau  = agg( τ · r_t² · m_t )                    (по всей маске ответа)
L      = L_pg + L_tau
```

- Токены вне полосы не входят в `L_pg`, но входят в `L_tau`: дрейф за полосу штрафуется, градиента политики нет.
- Если в полосе нет ни одного токена, `L_pg = 0`.
- В torch-версии есть опциональный IS-вес `w_t` (`rollout_is_weights`), который умножает `−A·r`. Эталон
  `algorithm.py` его не содержит.
- Эталон делит `L_pg` на число токенов в полосе, `L_tau` на число токенов ответа: это token-mean.

Источник: Kimi K2.5 tech report, §4.4.2. Порт из MultiAgents-main, `verl/trainer/ppo/core_algos.py`,
`compute_policy_loss_k25_token_clip`. Имя K25 не означает группу из 25 роллаутов: размер группы задаёт
`training.rollout_n`.

| Что | Где |
|---|---|
| Эталон без autograd (спецификация) | `algorithm.py: k25_token_clip_reference`, `K25Params` |
| Реализация для verl | `training/k25_loss.py: k25_token_clip_loss`, регистрация `register` |
| Параметры α, β, τ | `[training] k25_alpha, k25_beta, k25_tau`; передаются JSON в `ORCHESTRATOR_RESEARCH_K25` (`training/overrides.py`, `runtime_env`) |
| Регистрация в Ray-воркерах | `training/k25_loss.py: setup_worker` |
| Метрики | `actor/k25_in_band_frac`, `actor/k25_log_ratio_mean`, `actor/k25_pg_loss`, `actor/k25_tau_loss`, `actor/ppo_kl` (`ppo_kl` = среднее `−r`) |
| Тесты | `tests/test_k25_loss.py`: `test_matches_reference_on_random_batch`, `test_out_of_band_tokens_get_only_tau_gradient`, `test_masked_tokens_get_zero_gradient`; `tests/test_algorithms.py`: `test_out_of_band_token_only_keeps_regularizer`, `test_masked_observation_token_does_not_affect_loss` |

Полоса ограничивает изменение политики за один шаг. Значения по умолчанию `α = −0.2`, `β = 0.2`, `τ = 1e-3`
стоят в `configs/base.toml`; подбор этих значений — отдельный эксперимент.

## 2. Групповое преимущество без нормировки на std

Для роллаутов одной задачи (группа из `training.rollout_n ≥ 2`) с наградами `r_1 … r_n`:

```
A_i = r_i − mean(r_1 … r_n)
```

Деления на стандартное отклонение нет: `algorithm.norm_adv_by_std_in_grpo = false` (ADR 0005). Преимущество
считается на уровне роллаута (`enable_rollout_level_advantage = true`, agl).

| Что | Где |
|---|---|
| Эталон | `algorithm.py: group_relative_advantages` (группа меньше двух роллаутов отклоняется) |
| Настройка для verl | `training/overrides.py`: `algorithm.adv_estimator = "grpo"`, `norm_adv_by_std_in_grpo = false` |
| Тест | `tests/test_algorithms.py: test_group_advantages_are_centred_per_group` |

## 3. Награда PARL с отжигом

Обучение (шаг `t`, `T = anneal_steps`):

```
r_PARL(t) = r_perf + λ1(t) · r_parallel + λ2(t) · r_finish
```

Валидация: `r_PARL = r_perf`. Компоненты при этом считаются и попадают в поле `reward` JSON-документа эпизода
(`assembly.episode_document`), чтобы видеть, что оптимизировалось бы.

Отжиг, где `init` — `[reward] lambda_parallel` (для `λ1`) или `lambda_finish` (для `λ2`). Шаги `t < 0`
считаются как 0:

```
linear:  λ(t) = init · (1 − t / T)                    при t < T, иначе 0
cosine:  λ(t) = init · ½ · (1 + cos(π t / T))         при t < T, иначе 0
```

Компоненты, все в `[0, 1]`. `S` — стадии (ходы оркестратора с вызовами инструментов), `N` — запущенные субагенты:

```
r_parallel:
  fraction_of_stages   доля стадий, где запущен хотя бы один субагент
  count_normalised     min(1, всего запущено / max_subagents_init)
  cohort_size_avg      min(1, среднее число запущенных на стадию / max_subagents_init)

r_finish = finished_ok / observable,   observable = запущенные − инфраструктурные сбои
  запущенных нет                     → 0
  observable = 0 (все заблокированы)  → 1
  finished_ok — завершённые среди observable, поэтому r_finish ≤ 1
```

Отличия от референса (MultiAgents-main): счётчики берутся из `Stage.subagent_runs` и `SubagentRun`. Сбои
инфраструктуры не входят в числитель и знаменатель `r_finish` и здесь, и в референсе (`#finished / max(1, #spawned − #infra_failed)`).
Различие в вырожденном случае: если все запуски заблокированы инфраструктурой, референс даёт 0, а здесь `r_finish = 1`
(`observable = 0`).

`r_perf` равен 1 только при вердикте судьи `GRADED` и `correct` (`assembly.run_episode`). Сбой судьи даёт 0
и сохраняет статус ошибки, а не ложную правоту.

| Что | Где |
|---|---|
| Формулы и docstring | `reward.py` (модуль) |
| Отжиг | `reward.py: anneal` |
| Компоненты | `reward.py: r_parallel`, `r_finish` |
| Сборка | `reward.py: compose_reward` → `RewardBreakdown`; вызов в `assembly.run_episode` |
| Настройки | `[reward]` в `configs/base.toml` |
| Тесты | `tests/test_reward.py`: `test_anneal_endpoints_midpoint_and_zero_after`, `test_r_parallel_stays_in_unit_interval`, `test_r_parallel_modes_match_their_formulas`, `test_r_finish_excludes_infra_failures_from_denominator`, `test_lambdas_vanish_after_anneal`, `test_validation_total_is_task_outcome_only` |

Источник: Kimi K2.5 PARL, §3 (декомпозиция награды и отжиг весов). Порт из MultiAgents-main:
`verl/workers/reward_manager/parl.py`, `parl_lambda.py`.

Шаг `t` берётся из `training/hooks.py` (`StepCounterHooks`): `count // (train_batch_size × rollout_n)`, где
счётчик считает train-роллауты. Формула верна только при синхронном режиме без dynamic sampling
(`agentlightning.async_rollout.enabled = false`, ADR 0005). Resume отключён (`trainer.resume_mode = "disable"`):
счётчик при возобновлении начинался бы с 0, и отжиг λ стартовал бы заново.
