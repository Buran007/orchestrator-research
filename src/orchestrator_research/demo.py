"""Учебный запуск без GPU, сети и ключей API: бюджет, один эпизод swarm, награда и K2.5.

Ответы и числа токенов заданы вручную, поэтому цифры показывают механику, а не поведение модели.
Все шаги вызывают настоящие limits, strategies, reward и algorithm, а не копируют формулы.
"""

from __future__ import annotations

import asyncio

from .adapters.fake import ScriptedChatModel, StaticRetriever
from .algorithm import group_relative_advantages, k25_token_clip_reference
from .config import RewardConfig
from .domain import EpisodeResult, ModelTurn, Question, SearchResult, ToolCall
from .limits import TaskBudget, TaskLimits
from .reward import anneal, compose_reward
from .strategies import get_strategy
from .strategies.base import EpisodeEnvironment
from .subagents import SubagentPool

ROLE_PROMPTS = {"fact_checker": "Проверьте одно утверждение с помощью поиска и верните вывод."}
QUESTION = Question("demo-001", "What is the capital of France?", ("Paris",))
EPISODE_LIMITS = TaskLimits(
    orchestrator_tokens_per_turn=64,
    subagent_tokens_per_step=64,
    final_answer_reserve=16,
    max_turns=4,
    max_subagents_per_task=2,
    max_parallel=2,
    subagent_max_steps=3,
)


def run_demo() -> int:
    print("Учебный запуск: ответы и токены заданы вручную. GPU, сеть и обучение не используются.")
    _budget_section()
    episode = asyncio.run(_swarm_episode())
    _episode_section(episode)
    _reward_section(episode)
    _k25_section()
    return 0


def _budget_section() -> None:
    limits = TaskLimits(
        orchestrator_tokens_per_task=20,
        orchestrator_tokens_per_turn=12,
        final_answer_reserve=4,
        max_turns=5,
    )
    budget = TaskBudget(limits)
    print("\n1. TaskBudget: резерв хода вычитается из лимита оркестратора")
    print(
        f"  лимит оркестратора на задачу: {limits.orchestrator_tokens_per_task}, "
        f"резерв ответа: {limits.final_answer_reserve}"
    )
    while True:
        reserved = budget.reserve_orchestrator_turn()
        if reserved == 0:
            break
        actual = min(reserved, 9)  # модель сгенерировала 9 токенов или меньше
        budget.commit_orchestrator(reserved, actual)
        print(f"  ход {budget.orchestrator_turns}: зарезервировано {reserved}, фактически {actual}")
    stop = budget.orchestrator_stop_reason()
    final = budget.reserve_final_answer()
    budget.commit_orchestrator(final, min(final, 3))
    print(f"  остановка: {stop.value if stop else None}; для финального ответа зарезервировано: {final}")


async def _swarm_episode() -> EpisodeResult:
    limits = EPISODE_LIMITS
    budget = TaskBudget(limits)
    retriever = StaticRetriever([SearchResult("doc-1", "Paris is the capital of France.", 0.9)])
    orchestrator = ScriptedChatModel(
        [
            ModelTurn(
                text="",
                output_tokens=20,
                finish_reason="tool_calls",
                tool_calls=(
                    ToolCall("c1", "create_subagent", {"name": "checker", "role": "fact_checker", "model": "small"}),
                    ToolCall("c2", "assign_task", {"agent": "checker", "prompt": "Check: Paris is the capital."}),
                ),
            ),
            ModelTurn("Paris", 5, "stop", final_answer="Paris"),
        ]
    )
    subagent = ScriptedChatModel(
        [ModelTurn("Confirmed by the retrieved passage.", 8, "stop", final_answer="Paris is confirmed")]
    )
    pool = SubagentPool(
        clients={"small": subagent},
        role_prompts=ROLE_PROMPTS,
        budget=budget,
        limits=limits,
        retriever=retriever,
        top_k=3,
        return_max_chars=500,
        semaphore=asyncio.Semaphore(limits.max_parallel),
    )
    env = EpisodeEnvironment(
        orchestrator=orchestrator,
        pool=pool,
        retriever=retriever,
        budget=budget,
        limits=limits,
        top_k=3,
    )
    return await get_strategy("swarm")(env).run(QUESTION)


def _episode_section(result: EpisodeResult) -> None:
    subagent_tokens = sum(result.budget["subagent_tokens"].values())
    print("\n2. Один эпизод swarm на фейках")
    print(f"  статус={result.status.value}, ответ={result.final_answer!r}")
    for stage in result.stages:
        print(f"  ход {stage.turn_index}: субагенты {[run.name for run in stage.subagent_runs]}")
    print(f"  токены оркестратора={result.budget['orchestrator_tokens']}, субагентов={subagent_tokens}")


def _reward_section(result: EpisodeResult) -> None:
    cfg = RewardConfig()
    r_perf = 1.0 if result.final_answer == QUESTION.gold_answers[0] else 0.0
    print("\n3. Награда: r_perf + λ·r_parallel + λ·r_finish, λ отжигается к нулю")
    rows = {
        "train_step_0": compose_reward(r_perf, result.stages, step=0, is_validation=False, cfg=cfg),
        "train_step_7": compose_reward(r_perf, result.stages, step=7, is_validation=False, cfg=cfg),
        "validation": compose_reward(r_perf, result.stages, step=0, is_validation=True, cfg=cfg),
    }
    for key, row in rows.items():
        print(
            f"  {key}: total={row.total:.3f} (r_perf={row.r_perf}, r_parallel={row.r_parallel:.2f}, "
            f"r_finish={row.r_finish:.2f}, λ_par={row.lambda_parallel:.4f})"
        )
    by_step = {step: anneal(cfg.lambda_parallel, step, cfg.anneal_steps, cfg.anneal_mode) for step in (0, 3, 7)}
    print(f"  λ_parallel по шагам: {by_step}")


def _k25_section() -> None:
    rewards = [1.0, 1.0, 0.0, 0.0]  # четыре попытки ответить на один вопрос
    advantages = group_relative_advantages(rewards, [QUESTION.question_id] * len(rewards))
    mask = [1, 1, 0, 1]  # третий токен — наблюдение инструмента, не обучается
    token_advantages = [advantages[0] if is_trainable else 0.0 for is_trainable in mask]  # первая попытка
    loss = k25_token_clip_reference(
        old_log_probs=[-1.0] * 4,
        new_log_probs=[-0.9, -0.95, 0.5, -0.7],
        advantages=token_advantages,
        trainable_mask=mask,
    )
    print("\n4. Будущий RL: advantage → маска токенов → K2.5 loss")
    print(f"  rewards={rewards}, advantages={advantages}")
    print(f"  mask={mask}; token_advantages={token_advantages}")
    print(f"  K2.5={loss}")
    print("  K2.5 — имя функции потерь, не число попыток. Веса здесь не обновляются.")
