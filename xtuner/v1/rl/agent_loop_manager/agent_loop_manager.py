import asyncio
import json
import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from xtuner.v1.data_proto.rl_data import RolloutState, Status
from xtuner.v1.rl.agent_loop import AgentLoopConfig, AgentLoopSpec, get_agent_loop_rollout_ctl
from xtuner.v1.rl.judger import ComposedJudgerConfig, JudgerConfig, build_judger
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.rollout import RolloutController
from xtuner.v1.utils import get_logger

from .producer import (
    GROUP_GENERATE_TIME_KEY,
    DisaggProduceContext,
    DisaggProduceProgress,
    DisaggProduceStrategy,
    DisaggProduceStrategyConfig,
    ProduceBatchStatus,
    ProduceContext,
    ProduceProgress,
    ProduceStrategy,
    ProduceStrategyConfig,
    SyncProduceStrategyConfig,
    default_is_valid_sample_fn,
)
from .sampler import Sampler, SamplerConfig


@dataclass
class ProduceBatchResult:
    """Result of a single ``produce_batch`` call.

    Args:
        rollout_states (list[list[RolloutState]]): Completed rollout groups retrieved from the replay buffer for training.
        group_gen_count (int | None): Number of generate-group calls finished in this batch (None if no generations ran).
        group_gen_mean_s (float | None): Mean wall-clock time per generate-group call, in seconds.
        group_gen_p50_s (float | None): Median (p50) generate-group time, in seconds.
        group_gen_p99_s (float | None): 99th percentile generate-group time, in seconds.
        group_gen_p99_p50_ratio (float | None): Ratio of p99 to p50, indicating tail-latency skew.
        group_gen_pause_time_s (float | None): Time spent in pause/cleanup phase (async strategy only), in seconds.
        leftover_init (int): Number of init groups remaining in the replay buffer after this batch.
        leftover_completed (int): Number of completed groups remaining in the replay buffer after this batch.
        leftover_aborted (int): Number of aborted groups remaining in the replay buffer.
        leftover_expired (int): Number of expired groups remaining in the replay buffer.
        leftover_failed (int): Number of failed groups remaining in the replay buffer.
        leftover_filtered (int): Number of filtered groups remaining in the replay buffer.
        raw_rewards_sum (float): Sum of rewards produced before replay-buffer insertion for the current window.
        raw_rewards_count (int): Number of reward-bearing samples included in ``raw_rewards_sum``.
        produced_samples (int): Number of rollout samples produced in the current produce window.
        produced_tokens (int): Number of response tokens produced in the current produce window.
        produce_time_s (float): Wall-clock production time consumed by the current produce window.
    """

    rollout_states: list[list[RolloutState]]
    status: ProduceBatchStatus = ProduceBatchStatus.NORMAL
    # per-group generation timing stats (all None if no generations occurred)
    group_gen_count: int | None = None
    group_gen_mean_s: float | None = None
    group_gen_p50_s: float | None = None
    group_gen_p99_s: float | None = None
    group_gen_p99_p50_ratio: float | None = None
    group_gen_pause_time_s: float | None = None
    # leftover samples remaining in replay buffer after batch retrieval
    leftover_init: int = 0
    leftover_completed: int = 0
    leftover_aborted: int = 0
    leftover_expired: int = 0
    leftover_failed: int = 0
    leftover_filtered: int = 0
    # rewards produced during the current produce window, including completed and filtered groups.
    raw_rewards_sum: float = 0.0
    raw_rewards_count: int = 0
    produced_samples: int = 0
    produced_tokens: int = 0
    produce_time_s: float = 0.0
    task_batch_sizes: dict[str, int] | None = None
    task_results: dict[str, "ProduceBatchResult"] | None = None


@dataclass(frozen=True)
class _TaskRunner:
    task_name: str
    agent_loop: AgentLoopSpec
    produce_strategy: ProduceStrategy | DisaggProduceStrategy
    sampler: Sampler
    weight: float = 1.0
    order: int = 0


class _TaskSamplerView:
    def __init__(self, samplers: list[Sampler]):
        self._samplers = samplers

    def __len__(self) -> int:
        return sum(len(sampler) for sampler in self._samplers)


class AgentLoopManagerStatus(Enum):
    """AgentLoopManager 的全局状态.

    按下面的路径流转：
    - 初始状态是 NORMAL
    - NORMAL -> UPDATE_WEIGHT_AND_ABORT
      - trainer 开始做权重同步前触发
    - UPDATE_WEIGHT_AND_ABORT -> NORMAL
      - 权重同步完成后调用 continue_product()
    - NORMAL -> EXPIRED_BATCH
      - 当前 rollout model 已经过旧
    - EXPIRED_BATCH -> UPDATE_WEIGHT_AND_ABORT
      - trainer 检测到过期后，进入权重同步阶段
    - 任意状态 -> FINISH
      - 训练结束

    这里有一个重要区分：
    - AgentLoopManagerStatus 是“后台 producer 的全局运行状态”
    - ProduceBatchStatus 是“单次调度调用的局部结果”
    """

    NORMAL = auto()
    UPDATE_WEIGHT_AND_ABORT = auto()
    EXPIRED_BATCH = auto()
    FINISH = auto()


def _fill_produce_timing_stats(
    result: ProduceBatchResult, generate_times_s: list[float], pause_time_s: float = 0.0
) -> None:
    if not generate_times_s:
        if pause_time_s > 0:
            result.group_gen_pause_time_s = pause_time_s
        return
    sorted_times = sorted(generate_times_s)
    n = len(sorted_times)
    mean_s = sum(sorted_times) / n
    p50_s = sorted_times[n // 2]
    p99_s = sorted_times[int(n * 0.99)]
    ratio = p99_s / p50_s if p50_s > 0 else float("inf")
    result.group_gen_count = n
    result.group_gen_mean_s = mean_s
    result.group_gen_p50_s = p50_s
    result.group_gen_p99_s = p99_s
    result.group_gen_p99_p50_ratio = ratio
    result.group_gen_pause_time_s = pause_time_s


def _fill_group_timing_stats(
    result: ProduceBatchResult, rollout_states: list[list[RolloutState]], pause_time_s: float = 0.0
) -> None:
    generate_times: list[float] = []
    for group in rollout_states:
        if not group:
            continue
        group_time = getattr(group[0], "extra_fields", {}).get(GROUP_GENERATE_TIME_KEY)
        if group_time is not None:
            generate_times.append(group_time)

    _fill_produce_timing_stats(result, generate_times, pause_time_s=pause_time_s)


def _aggregate_status(statuses: list[ProduceBatchStatus]) -> ProduceBatchStatus:
    if any(status == ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT for status in statuses):
        return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
    if any(status == ProduceBatchStatus.EXPIRED_BATCH for status in statuses):
        return ProduceBatchStatus.EXPIRED_BATCH
    return ProduceBatchStatus.NORMAL


_LEFTOVER_STATUSES = [
    Status.INIT,
    Status.COMPLETED,
    Status.ABORTED,
    Status.EXPIRED,
    Status.FAILED,
    Status.FILTERED,
]
_TASK_CHECKPOINT_DIR = "tasks"
_MANAGER_STATE_PATH = "agent_loop_manager_state.json"
_STATUS_POLL_INTERVAL_S = 1.0


def _fill_leftover_counts(result: ProduceBatchResult, status_counts: dict[Status, int]) -> None:
    result.leftover_init = status_counts.get(Status.INIT, 0)
    result.leftover_completed = status_counts.get(Status.COMPLETED, 0)
    result.leftover_aborted = status_counts.get(Status.ABORTED, 0)
    result.leftover_expired = status_counts.get(Status.EXPIRED, 0)
    result.leftover_failed = status_counts.get(Status.FAILED, 0)
    result.leftover_filtered = status_counts.get(Status.FILTERED, 0)


def _init_manager_fields(
    manager, task_runners: list[_TaskRunner], replay_buffer: ReplayBuffer, logger, name: str
) -> None:
    if not task_runners:
        raise ValueError(f"{name} requires at least one task runner.")
    if sum(task.weight for task in task_runners) <= 0:
        raise ValueError(f"At least one task weight must be positive for {name}.")

    manager.task_runners = task_runners
    manager.replay_buffer = replay_buffer
    manager.data_sampler = (
        task_runners[0].sampler
        if len(task_runners) == 1
        else _TaskSamplerView([task.sampler for task in task_runners])
    )
    manager.name = task_runners[0].task_name if len(task_runners) == 1 else "multi_task"
    manager.logger = get_logger() if logger is None else logger
    manager.task_names = [task.task_name for task in task_runners]


def allocate_task_batch_sizes(
    task_runners: list[_TaskRunner],
    global_batch_size: int,
    train_step: int,
) -> dict[str, int]:
    # 默认按 task weight 静态分配；保留 train_step 参数，和 manager 的可覆盖分配入口保持同一形状。
    if global_batch_size < 0:
        raise ValueError(f"global_batch_size must be non-negative, got {global_batch_size}")

    total_weight = sum(task.weight for task in task_runners)
    if total_weight <= 0:
        raise ValueError("Sum of task weights must be positive.")
    if global_batch_size == 0:
        return {task.task_name: 0 for task in task_runners}

    raw_allocations = [global_batch_size * task.weight / total_weight for task in task_runners]
    floor_allocations = [math.floor(raw) for raw in raw_allocations]
    remaining = global_batch_size - sum(floor_allocations)

    task_batch_sizes = {task.task_name: floor_allocations[idx] for idx, task in enumerate(task_runners)}
    if remaining <= 0:
        return task_batch_sizes

    ranked_tasks = sorted(
        enumerate(task_runners),
        key=lambda item: (
            -(raw_allocations[item[0]] - floor_allocations[item[0]]),
            item[1].order,
        ),
    )
    for idx, task in ranked_tasks[:remaining]:
        task_batch_sizes[task.task_name] += 1
    return task_batch_sizes


def validate_task_batch_sizes(
    task_runners: list[_TaskRunner], task_batch_sizes: dict[str, int], global_batch_size: int
) -> None:
    expected_task_names = {task.task_name for task in task_runners}
    actual_task_names = set(task_batch_sizes.keys())
    if actual_task_names != expected_task_names:
        missing_task_names = expected_task_names - actual_task_names
        extra_task_names = actual_task_names - expected_task_names
        raise ValueError(
            "Invalid task batch sizes returned by get_task_batch_sizes: "
            f"missing={sorted(missing_task_names)}, extra={sorted(extra_task_names)}"
        )

    negative_batch_sizes = {
        task_name: task_batch_size for task_name, task_batch_size in task_batch_sizes.items() if task_batch_size < 0
    }
    if negative_batch_sizes:
        raise ValueError(f"Task batch sizes must be non-negative, got {negative_batch_sizes}")

    total_batch_size = sum(task_batch_sizes.values())
    if total_batch_size != global_batch_size:
        raise ValueError(
            "Task batch sizes must sum to the requested global batch size, "
            f"got total={total_batch_size}, expected={global_batch_size}"
        )


def get_task_batch_sizes_for_step(manager, batch_size: int, train_step: int) -> dict[str, int]:
    if len(manager.task_runners) == 1:
        return {manager.task_runners[0].task_name: batch_size}

    task_batch_sizes = manager.get_task_batch_sizes(batch_size, train_step)
    validate_task_batch_sizes(manager.task_runners, task_batch_sizes, batch_size)
    return task_batch_sizes


async def refresh_for_all_tasks(
    *,
    task_runners: list[_TaskRunner],
    replay_buffer: ReplayBuffer,
    logger,
    manager_name: str,
    train_step: int,
    statuses: list[Status],
) -> None:
    task_stale_thresholds: dict[str, int] = {}
    for task in task_runners:
        # colocate / disagg 都统一刷新 staleness；同步策略没有 stale_threshold 时使用 1。
        stale_threshold = getattr(task.produce_strategy, "stale_threshold", 1)
        task_stale_thresholds[task.task_name] = stale_threshold

    expired_counts = await replay_buffer.refresh_staleness(
        task_stale_thresholds=task_stale_thresholds,
        current_train_step=train_step,
        statuses=statuses,
    )
    for task_name, expired_count in expired_counts.items():
        logger.info(
            f"[AgentLoopManager][{manager_name}] Refresh staleness for task {task_name}: expired_count={expired_count}"
        )


def aggregate_task_results(
    ordered_tasks: list[_TaskRunner], task_results: dict[str, ProduceBatchResult]
) -> ProduceBatchResult:
    rollout_states: list[list[RolloutState]] = []
    leftover_init = 0
    leftover_completed = 0
    leftover_aborted = 0
    leftover_expired = 0
    leftover_failed = 0
    leftover_filtered = 0
    total_group_count = 0
    weighted_group_mean_sum = 0.0
    weighted_group_p50_sum = 0.0
    weighted_group_p99_sum = 0.0
    weighted_group_ratio_sum = 0.0
    total_pause_time_s = 0.0
    raw_rewards_sum = 0.0
    raw_rewards_count = 0
    produced_samples = 0
    produced_tokens = 0
    produce_time_s = 0.0

    for task in ordered_tasks:
        result = task_results[task.task_name]
        rollout_states.extend(result.rollout_states)
        leftover_init += result.leftover_init
        leftover_completed += result.leftover_completed
        leftover_aborted += result.leftover_aborted
        leftover_expired += result.leftover_expired
        leftover_failed += result.leftover_failed
        leftover_filtered += result.leftover_filtered
        raw_rewards_sum += result.raw_rewards_sum
        raw_rewards_count += result.raw_rewards_count
        produced_samples += result.produced_samples
        produced_tokens += result.produced_tokens
        produce_time_s += result.produce_time_s
        if result.group_gen_count is not None and result.group_gen_mean_s is not None:
            total_group_count += result.group_gen_count
            weighted_group_mean_sum += result.group_gen_count * result.group_gen_mean_s
            weighted_group_p50_sum += result.group_gen_count * (result.group_gen_p50_s or 0.0)
            weighted_group_p99_sum += result.group_gen_count * (result.group_gen_p99_s or 0.0)
            weighted_group_ratio_sum += result.group_gen_count * (result.group_gen_p99_p50_ratio or 0.0)
            total_pause_time_s += result.group_gen_pause_time_s or 0.0

    aggregated = ProduceBatchResult(
        rollout_states=rollout_states,
        leftover_init=leftover_init,
        leftover_completed=leftover_completed,
        leftover_aborted=leftover_aborted,
        leftover_expired=leftover_expired,
        leftover_failed=leftover_failed,
        leftover_filtered=leftover_filtered,
        raw_rewards_sum=raw_rewards_sum,
        raw_rewards_count=raw_rewards_count,
        produced_samples=produced_samples,
        produced_tokens=produced_tokens,
        produce_time_s=produce_time_s,
        task_results={task.task_name: task_results[task.task_name] for task in ordered_tasks},
    )
    if total_group_count > 0:
        aggregated.group_gen_count = total_group_count
        aggregated.group_gen_mean_s = weighted_group_mean_sum / total_group_count
        aggregated.group_gen_p50_s = weighted_group_p50_sum / total_group_count
        aggregated.group_gen_p99_s = weighted_group_p99_sum / total_group_count
        aggregated.group_gen_p99_p50_ratio = weighted_group_ratio_sum / total_group_count
        aggregated.group_gen_pause_time_s = total_pause_time_s
    return aggregated


def log_buffer_counts(
    logger,
    *,
    manager_name: str,
    task_runners: list[_TaskRunner],
    task_batch_sizes: dict[str, int],
    batch_by_task: dict[str, list[list[RolloutState]]],
    leftover_counts: dict[str, dict[Status, int]],
) -> None:
    for task in task_runners:
        task_name = task.task_name
        task_counts = leftover_counts.get(task_name, {})
        logger.info(
            f"[AgentLoopManager][{manager_name}] get_batch from buffer for task {task_name}: "
            f"requested={task_batch_sizes[task_name]}, retrieved={len(batch_by_task.get(task_name, []))}, "
            f"leftover_init={task_counts.get(Status.INIT, 0)}, "
            f"leftover_completed={task_counts.get(Status.COMPLETED, 0)}, "
            f"leftover_aborted={task_counts.get(Status.ABORTED, 0)}, "
            f"leftover_expired={task_counts.get(Status.EXPIRED, 0)}, "
            f"leftover_failed={task_counts.get(Status.FAILED, 0)}, "
            f"leftover_filtered={task_counts.get(Status.FILTERED, 0)}"
        )


def build_produce_batch_result(
    *,
    task_runners: list[_TaskRunner],
    task_batch_sizes: dict[str, int],
    batch_by_task: dict[str, list[list[RolloutState]]],
    leftover_counts: dict[str, dict[Status, int]],
    progress: ProduceProgress | DisaggProduceProgress,
    pause_time_s: float,
) -> ProduceBatchResult:
    if len(task_runners) == 1:
        task = task_runners[0]
        raw_rewards_sum, raw_rewards_count = progress.consume_raw_rewards(task.task_name)
        produced_samples, produced_tokens = progress.consume_produced(task.task_name)
        produce_time_s = progress.consume_produce_time()
        result = ProduceBatchResult(
            rollout_states=batch_by_task.get(task.task_name, []),
            raw_rewards_sum=raw_rewards_sum,
            raw_rewards_count=raw_rewards_count,
            produced_samples=produced_samples,
            produced_tokens=produced_tokens,
            produce_time_s=produce_time_s,
        )
        _fill_leftover_counts(result, leftover_counts.get(task.task_name, {}))
        _fill_group_timing_stats(result, result.rollout_states, pause_time_s=pause_time_s)
        return result

    task_results: dict[str, ProduceBatchResult] = {}
    produce_time_s = progress.consume_produce_time()
    for task in task_runners:
        raw_rewards_sum, raw_rewards_count = progress.consume_raw_rewards(task.task_name)
        produced_samples, produced_tokens = progress.consume_produced(task.task_name)
        result = ProduceBatchResult(
            rollout_states=batch_by_task.get(task.task_name, []),
            raw_rewards_sum=raw_rewards_sum,
            raw_rewards_count=raw_rewards_count,
            produced_samples=produced_samples,
            produced_tokens=produced_tokens,
        )
        _fill_leftover_counts(result, leftover_counts.get(task.task_name, {}))
        task_results[task.task_name] = result

    ordered_tasks = sorted(task_runners, key=lambda task: (task.task_name, task.order))
    aggregated = aggregate_task_results(ordered_tasks, task_results)
    aggregated.produce_time_s = produce_time_s
    aggregated.task_batch_sizes = {task.task_name: task_batch_sizes[task.task_name] for task in ordered_tasks}
    _fill_group_timing_stats(aggregated, aggregated.rollout_states, pause_time_s=pause_time_s)
    return aggregated


async def take_train_batch(
    *,
    task_runners: list[_TaskRunner],
    replay_buffer: ReplayBuffer,
    logger,
    manager_name: str,
    batch_size: int,
    task_batch_sizes: dict[str, int],
    progress: ProduceProgress | DisaggProduceProgress,
    pause_time_s: float = 0.0,
) -> ProduceBatchResult:
    validate_task_batch_sizes(task_runners, task_batch_sizes, batch_size)
    batch_by_task, consumed_counts = await replay_buffer.take_batch(task_batch_sizes)
    if isinstance(progress, DisaggProduceProgress):
        progress.mark_consumed(consumed_counts)
    task_names = [task.task_name for task in task_runners]
    leftover_counts = await replay_buffer.count_statuses(task_names, _LEFTOVER_STATUSES)
    log_buffer_counts(
        logger,
        manager_name=manager_name,
        task_runners=task_runners,
        task_batch_sizes=task_batch_sizes,
        batch_by_task=batch_by_task,
        leftover_counts=leftover_counts,
    )
    return build_produce_batch_result(
        task_runners=task_runners,
        task_batch_sizes=task_batch_sizes,
        batch_by_task=batch_by_task,
        leftover_counts=leftover_counts,
        progress=progress,
        pause_time_s=pause_time_s,
    )


async def continue_generation(task_runners: list[_TaskRunner]) -> None:
    rollout_ctl = await get_agent_loop_rollout_ctl(task_runners[0].agent_loop)
    await rollout_ctl.continue_generation.remote()  # type: ignore[attr-defined]


def task_checkpoint_path(checkpoint_path: Path | str, task_name: str) -> Path:
    return Path(checkpoint_path) / _TASK_CHECKPOINT_DIR / task_name


def manager_state_path(checkpoint_path: Path | str) -> Path:
    return Path(checkpoint_path) / _MANAGER_STATE_PATH


def get_pending_task_counts(task_runners: list[_TaskRunner]) -> dict[str, int]:
    pending_task_counts: dict[str, int] = {}
    for task in task_runners:
        pending_count = task.produce_strategy.pending_task_count()
        if pending_count > 0:
            pending_task_counts[task.task_name] = pending_count
    return pending_task_counts


def _build_produce_context(
    task_runner: _TaskRunner,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    train_step: int,
    model_step: int,
    progress: ProduceProgress,
) -> ProduceContext:
    return ProduceContext(
        agent_loop=task_runner.agent_loop,
        sampler=task_runner.sampler,
        replay_buffer=replay_buffer,
        task_batch_size=batch_size,
        task_name=task_runner.task_name,
        train_step=train_step,
        model_step=model_step,
        progress=progress,
        is_valid_sample_fn=getattr(task_runner.produce_strategy, "is_valid_sample_fn", default_is_valid_sample_fn),
        stale_threshold=getattr(task_runner.produce_strategy, "stale_threshold", None),
    )


def _build_disagg_produce_context(
    task_runner: _TaskRunner,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    train_step: int,
    model_step: int,
    update_event: asyncio.Event,
    progress: DisaggProduceProgress,
) -> DisaggProduceContext:
    return DisaggProduceContext(
        agent_loop=task_runner.agent_loop,
        sampler=task_runner.sampler,
        replay_buffer=replay_buffer,
        task_batch_size=batch_size,
        task_name=task_runner.task_name,
        train_step=train_step,
        model_step=model_step,
        progress=progress,
        update_event=update_event,
        is_valid_sample_fn=getattr(task_runner.produce_strategy, "is_valid_sample_fn", default_is_valid_sample_fn),
        stale_threshold=getattr(task_runner.produce_strategy, "stale_threshold", None),
    )


class TaskSpecConfig(BaseModel):
    """Configuration for one task managed by ``AgentLoopManager``.

    A task spec binds together the dataset sampler, agent loop, optional judger,
    production strategy, and sampling weight for one RL data source. Multi-task
    training is represented as a list of ``TaskSpecConfig`` objects.

    Args:
        task_name (str): Unique task name used for logging, replay-buffer
            routing, and checkpoint state.
        weight (float): Relative batch allocation weight for this task in
            multi-task training. Defaults to 1.0.
        agent_loop_config (AgentLoopConfig): Agent loop configuration used to
            generate rollout samples for this task.
        judger_config (JudgerConfig | ComposedJudgerConfig | None): Optional
            judger configuration used to score generated samples. Defaults to
            None.
        produce_strategy_config (ProduceStrategyConfig | DisaggProduceStrategyConfig):
            Strategy used to produce rollout samples. Colocate managers accept
            ``ProduceStrategyConfig`` subclasses; disaggregated managers accept
            ``DisaggProduceStrategyConfig`` subclasses. Defaults to
            ``SyncProduceStrategyConfig``.
        sampler_config (SamplerConfig): Dataset sampler configuration for this
            task.

    **Examples:**

    Example configuration for one task::

        task = TaskSpecConfig(
            task_name="gsm8k",
            weight=1.0,
            agent_loop_config=SingleTurnAgentLoopConfig(
                hf_checkpoint="Qwen/Qwen3-8B",
                sample_params=SampleParams(max_tokens=1024),
            ),
            judger_config=GSM8KJudgerConfig(),
            sampler_config=SamplerConfig(dataloader_cfg=dataloader_cfg, prompt_repeat_k=8),
        )
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task_name: str
    weight: float = Field(default=1.0, ge=0.0)
    agent_loop_config: AgentLoopConfig
    judger_config: JudgerConfig | ComposedJudgerConfig | None = None
    produce_strategy_config: ProduceStrategyConfig | DisaggProduceStrategyConfig = SyncProduceStrategyConfig()
    sampler_config: SamplerConfig


class AgentLoopManagerConfig(BaseModel):
    """Configuration for the agent loop manager.

    ``AgentLoopManagerConfig`` defines the rollout-producing side of RL
    training. It may manage a single task or a weighted list of tasks, and each
    task owns its sampler, agent loop, judger, and production strategy.

    Args:
        tasks (list[TaskSpecConfig] | TaskSpecConfig): One task config or a
            list of task configs. Task names must be unique when a list is
            provided.

    **Examples:**

    Example configuration for a single-task manager::

        config = AgentLoopManagerConfig(
            tasks=TaskSpecConfig(
                task_name="gsm8k",
                agent_loop_config=SingleTurnAgentLoopConfig(
                    hf_checkpoint="Qwen/Qwen3-8B",
                    sample_params=SampleParams(max_tokens=1024),
                ),
                judger_config=GSM8KJudgerConfig(),
                sampler_config=SamplerConfig(dataloader_cfg=dataloader_cfg, prompt_repeat_k=8),
            )
        )
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    tasks: list[TaskSpecConfig] | TaskSpecConfig
    mode: Literal["colocate", "disaggregated"] = "colocate"

    def build(
        self,
        rollout_controller: RolloutController,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        replay_buffer: ReplayBuffer,
        logger=None,
        sync_weights_interval: int = 1,
    ) -> "AgentLoopManager | DisaggAgentLoopManager":
        tasks = self.tasks if isinstance(self.tasks, list) else [self.tasks]
        if not tasks:
            raise ValueError("AgentLoopManagerConfig requires at least one task config.")

        seen_task_names: set[str] = set()
        task_runners: list[_TaskRunner] = []
        for order, task_cfg in enumerate(tasks):
            if task_cfg.task_name in seen_task_names:
                raise ValueError(f"Duplicate task_name found in AgentLoopManagerConfig: {task_cfg.task_name}")
            seen_task_names.add(task_cfg.task_name)

            agent_loop = task_cfg.agent_loop_config.build(
                rollout_controller=rollout_controller,
                judger=build_judger(task_cfg.judger_config) if task_cfg.judger_config is not None else None,
                logger=logger,
            )
            if self.mode == "colocate" and not isinstance(task_cfg.produce_strategy_config, ProduceStrategyConfig):
                raise ValueError(
                    "AgentLoopManagerConfig(mode='colocate') expects ProduceStrategyConfig, "
                    f"got {type(task_cfg.produce_strategy_config).__name__} for task {task_cfg.task_name!r}."
                )
            if self.mode == "disaggregated" and not isinstance(
                task_cfg.produce_strategy_config, DisaggProduceStrategyConfig
            ):
                raise ValueError(
                    "AgentLoopManagerConfig(mode='disaggregated') expects DisaggProduceStrategyConfig, "
                    f"got {type(task_cfg.produce_strategy_config).__name__} for task {task_cfg.task_name!r}."
                )
            produce_strategy = task_cfg.produce_strategy_config.build(
                sync_weights_interval=sync_weights_interval,
                rollout_controller=rollout_controller,
            )
            sampler = task_cfg.sampler_config.build(tokenizer=tokenizer, replay_buffer=replay_buffer)
            task_runners.append(
                _TaskRunner(
                    task_name=task_cfg.task_name,
                    agent_loop=agent_loop,
                    produce_strategy=produce_strategy,
                    sampler=sampler,
                    weight=task_cfg.weight,
                    order=order,
                )
            )

        manager_cls = AgentLoopManager if self.mode == "colocate" else DisaggAgentLoopManager
        return manager_cls(
            task_runners=task_runners,
            replay_buffer=replay_buffer,
            logger=logger,
        )


class AgentLoopManager:
    _TASK_CHECKPOINT_DIR = _TASK_CHECKPOINT_DIR
    _MANAGER_STATE_PATH = _MANAGER_STATE_PATH
    _STATUS_POLL_INTERVAL_S = _STATUS_POLL_INTERVAL_S
    task_runners: list[_TaskRunner]
    replay_buffer: ReplayBuffer
    data_sampler: Sampler | _TaskSamplerView
    name: str
    logger: Any
    task_names: list[str]

    def __init__(
        self,
        task_runners: list[_TaskRunner],
        replay_buffer: ReplayBuffer,
        logger=None,
    ):
        _init_manager_fields(self, task_runners, replay_buffer, logger, "AgentLoopManager")

    def get_task_batch_sizes(self, global_batch_size: int, train_step: int) -> dict[str, int]:
        """Return the per-task batch sizes for the current train step.

        Subclasses may override this method to implement custom dynamic batch allocation policies. Returning 0 for a
        task effectively disables that task for the current produce_batch call.
        """
        return allocate_task_batch_sizes(self.task_runners, global_batch_size, train_step)

    async def _produce_batch_to_buffer(
        self,
        task_batch_sizes: dict[str, int],
        progress: ProduceProgress,
        *,
        model_step: int,
    ) -> ProduceBatchStatus:
        current_future_step = progress.producer_future_step
        active_tasks = [task for task in self.task_runners if progress.target_samples[task.task_name] > 0]
        assert active_tasks, "No active tasks found"

        produce_start = time.perf_counter()
        try:
            produce_futures = []
            for task in active_tasks:
                produce_strategy = cast(ProduceStrategy, task.produce_strategy)
                produce_futures.append(
                    produce_strategy.produce_batch(
                        _build_produce_context(
                            task,
                            self.replay_buffer,
                            task_batch_sizes[task.task_name],
                            current_future_step,
                            model_step,
                            progress,
                        )
                    )
                )
            statuses = await asyncio.gather(*produce_futures)
        finally:
            progress.add_produce_time(time.perf_counter() - produce_start)
        return _aggregate_status(statuses)

    async def _pause_produce_for_progress(
        self,
        *,
        progress: ProduceProgress,
        model_step: int,
    ) -> float:
        # 共卡 pause 只负责让本次 produce_batch 的本地 pending 收尾，不维护非共卡 update_event。
        rollout_ctl = await get_agent_loop_rollout_ctl(self.task_runners[0].agent_loop)
        await rollout_ctl.pause_generation.remote()  # type: ignore[attr-defined]

        pause_time_s = 0.0
        for task in self.task_runners:
            produce_strategy = cast(ProduceStrategy, task.produce_strategy)
            ctx = _build_produce_context(
                task,
                self.replay_buffer,
                0,
                progress.producer_future_step,
                model_step,
                progress,
            )
            pause_time_s += await produce_strategy.pause_produce(
                ctx,
            )
        return pause_time_s

    async def produce_batch(
        self,
        batch_size: int,
        train_step: int,
        *,
        model_step: int,
    ) -> ProduceBatchResult:
        # `produce_batch()` 是保留给 colocate 路径的同步入口。
        #
        # 它虽然名字没变，但内部已经改成三段式：
        # 1. `_produce_batch_to_buffer()` 只负责生产，把结果写入 replay buffer
        # 2. 本地 pause/drain 显式收尾 pending rollout
        # 3. 从 replay buffer 再把训练 batch 取出来
        #
        # 这也是为什么这里要求返回非空 batch：
        # - colocate 语义下，调用它就是为了拿一批可训练 completed groups
        # - 如果需要合法返回空 batch + 特殊状态，那应该走 disagg 的 `get_batch()`
        if batch_size <= 0:
            raise ValueError(f"produce_batch expects batch_size > 0, got {batch_size}")
        start = time.perf_counter()
        self.logger.info(
            f"[AgentLoopManager][{self.name}] Start produce_batch: train_step={train_step} model_step={model_step} batch_size={batch_size}"
        )
        current_sizes = get_task_batch_sizes_for_step(self, batch_size, train_step)
        active_tasks = [task for task in self.task_runners if current_sizes[task.task_name] > 0]
        assert active_tasks, "No active tasks found"

        # 共卡路径下，produce_batch() 对应 rollout worker 当前持有的权重版本。
        # 这里只恢复 rollout controller，不维护非共卡 status/update_event/model_step，也不做 model expired 状态流转。
        await continue_generation(self.task_runners)
        local_progress = ProduceProgress.build(
            task_names=self.task_names,
            target_samples=current_sizes,
            train_step=train_step,
        )
        status = ProduceBatchStatus.NORMAL
        # 共卡 produce_batch 也是消费入口；生产前先刷新 buffer 中已有 completed / aborted。
        await refresh_for_all_tasks(
            task_runners=self.task_runners,
            replay_buffer=self.replay_buffer,
            logger=self.logger,
            manager_name=self.name,
            train_step=train_step,
            statuses=[Status.COMPLETED, Status.ABORTED],
        )
        status = await self._produce_batch_to_buffer(
            task_batch_sizes=current_sizes,
            progress=local_progress,
            model_step=model_step,
        )
        pause_time_s = await self._pause_produce_for_progress(
            progress=local_progress,
            model_step=model_step,
        )
        result = await take_train_batch(
            task_runners=self.task_runners,
            replay_buffer=self.replay_buffer,
            logger=self.logger,
            manager_name=self.name,
            batch_size=batch_size,
            task_batch_sizes=current_sizes,
            progress=local_progress,
            pause_time_s=pause_time_s,
        )
        result.status = status
        assert result.rollout_states, (
            "AgentLoopManager.produce_batch() must return non-empty rollout_states for colocated training. "
            "Use get_batch() for disaggregated empty/expired reads."
        )

        self.logger.info(
            f"[AgentLoopManager][{self.name}] produce_batch done "
            f"elapsed={time.perf_counter() - start:.3f}, completed_groups={len(result.rollout_states)}"
        )
        return result

    async def save(self, checkpoint_path: Path | str, model_step: int) -> None:
        """Save all task sampler states and the shared replay buffer."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        pending_task_counts = get_pending_task_counts(self.task_runners)
        if pending_task_counts:
            raise RuntimeError(
                "Cannot save AgentLoopManager while pending rollout tasks still exist: "
                f"{pending_task_counts}. Finish the current produce_batch before saving."
            )
        for task in self.task_runners:
            checkpoint_dir = task_checkpoint_path(checkpoint_path, task.task_name)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            task.sampler.save(checkpoint_dir)
        # manager 层保持 async 语义；同步入口只允许在 trainer 边界用 asyncio_run 包起来。
        await self.replay_buffer.save(checkpoint_path)
        state_path = manager_state_path(checkpoint_path)
        with state_path.open("w") as f:
            # 共卡 checkpoint 只需要恢复 sampler/replay buffer 和已完成的 model_step；
            # 非共卡 status/event/progress 由 DisaggAgentLoopManager 独占保存。
            json.dump({"model_step": model_step}, f)

    async def resume(self, checkpoint_path: Path | str) -> int:
        """Resume all task sampler states and the shared replay buffer."""
        checkpoint_path = Path(checkpoint_path)
        for task in self.task_runners:
            task.sampler.resume(task_checkpoint_path(checkpoint_path, task.task_name))
        # replay buffer 恢复是 async I/O，不能在已有 event loop 中再次嵌套 asyncio_run。
        await self.replay_buffer.resume(checkpoint_path)

        state_path = manager_state_path(checkpoint_path)
        with state_path.open("r") as f:
            manager_state = json.load(f)
        return manager_state["model_step"]


class DisaggAgentLoopManager:
    """非共卡 producer / consumer manager。

    独占后台 producer / 前台 consumer 的状态机，避免共卡 produce_batch 继续携带非共卡 event/status/progress。
    """

    _TASK_CHECKPOINT_DIR = _TASK_CHECKPOINT_DIR
    _MANAGER_STATE_PATH = _MANAGER_STATE_PATH
    _STATUS_POLL_INTERVAL_S = _STATUS_POLL_INTERVAL_S
    task_runners: list[_TaskRunner]
    replay_buffer: ReplayBuffer
    data_sampler: Sampler | _TaskSamplerView
    name: str
    logger: Any
    task_names: list[str]

    def __init__(
        self,
        task_runners: list[_TaskRunner],
        replay_buffer: ReplayBuffer,
        logger=None,
    ):
        _init_manager_fields(self, task_runners, replay_buffer, logger, "DisaggAgentLoopManager")

        # 非共卡并发控制信号：consumer 在同步权重前置位，producer / strategy 应直接观察
        # event 状态并尽快停止继续发新 rollout；不要用额外布尔快照替代这个 event。
        self._update_event = asyncio.Event()
        self._finish_event = asyncio.Event()

        # 非共卡 producer 读取的 model_step：rollout 侧当前使用的是哪个 train_step 同步后的模型。
        # 权重更新前必须先 pause 并清空 pending task，因此一个 pending 生命周期内只对应一个 model_step。
        self._model_step = 0

        # 非共卡 producer / consumer 共享的控制状态。produce_loop / get_batch 应直接读取
        # self._status，不要跨 await 缓存局部快照，避免错过同步、过期或结束状态变化。
        self._status = AgentLoopManagerStatus.NORMAL

        # pause_produce 写入、下一次 get_batch 读取并清零的耗时指标。
        # 只用于消费侧日志/metrics；读写不构成生产正确性依赖。
        self._pause_time_s = 0.0

        # 非共卡 producer / consumer 共享的绝对累计进度。对象引用必须保持稳定；
        # consumer 原地更新字段，producer / strategy 需要字段值时直接读取 progress.xxx。
        self._produce_progress = DisaggProduceProgress.build(self.task_names)

    def get_task_batch_sizes(self, global_batch_size: int, train_step: int) -> dict[str, int]:
        return allocate_task_batch_sizes(self.task_runners, global_batch_size, train_step)

    def _consume_pause_time(self) -> float:
        pause_time_s = self._pause_time_s
        self._pause_time_s = 0.0
        return pause_time_s

    async def _produce_batch_to_buffer(
        self,
        task_batch_sizes: dict[str, int],
        progress: DisaggProduceProgress,
        *,
        model_step: int,
    ) -> ProduceBatchStatus:
        current_future_step = progress.producer_future_step
        expired_tasks = []
        for task in self.task_runners:
            produce_strategy = cast(DisaggProduceStrategy, task.produce_strategy)
            if produce_strategy.is_model_expired(current_future_step, model_step):
                expired_tasks.append(task.task_name)
        if expired_tasks:
            self.logger.info(
                f"[DisaggAgentLoopManager][{self.name}] EXPIRED_BATCH: "
                f"future_step={current_future_step}, tasks={expired_tasks}"
            )
            return ProduceBatchStatus.EXPIRED_BATCH

        active_tasks = [task for task in self.task_runners if progress.target_samples[task.task_name] > 0]
        assert active_tasks, "No active tasks found"

        produce_start = time.perf_counter()
        try:
            produce_futures = []
            for task in active_tasks:
                produce_strategy = cast(DisaggProduceStrategy, task.produce_strategy)
                produce_futures.append(
                    produce_strategy.produce_batch(
                        _build_disagg_produce_context(
                            task,
                            self.replay_buffer,
                            task_batch_sizes[task.task_name],
                            current_future_step,
                            model_step,
                            self._update_event,
                            progress,
                        )
                    )
                )
            statuses = await asyncio.gather(*produce_futures)
        finally:
            progress.add_produce_time(time.perf_counter() - produce_start)
        return _aggregate_status(statuses)

    async def pause_produce(self) -> float:
        # 非共卡 producer 的显式刹车接口；共卡没有 public pause，也不再有混合模式分支。
        self._status = AgentLoopManagerStatus.UPDATE_WEIGHT_AND_ABORT
        self._update_event.set()
        rollout_ctl = await get_agent_loop_rollout_ctl(self.task_runners[0].agent_loop)
        await rollout_ctl.pause_generation.remote()  # type: ignore[attr-defined]

        pause_time_s = 0.0
        for task in self.task_runners:
            produce_strategy = cast(DisaggProduceStrategy, task.produce_strategy)
            ctx = _build_disagg_produce_context(
                task,
                self.replay_buffer,
                0,
                self._produce_progress.producer_future_step,
                self._model_step,
                self._update_event,
                self._produce_progress,
            )
            pause_time_s += await produce_strategy.pause_produce(ctx)
        self._pause_time_s = pause_time_s
        return pause_time_s

    async def continue_produce(self, model_step: int) -> None:
        #
        # 它和 pause_produce() 是一对：
        # - pause_produce() 负责让后台 producer 停下来；
        # - continue_produce(...) 负责在同步/评测完成后解除暂停。
        #
        # 这里同步更新 `_model_step`，表示 rollout 侧接下来生成样本时，
        # 应把“当前正在使用的是哪一版权重”记录成这个版本号。
        self._model_step = model_step
        await continue_generation(self.task_runners)
        # rollout controller 真正恢复后，再把 manager 暴露成 NORMAL，produce_loop 才能继续生产。
        self._status = AgentLoopManagerStatus.NORMAL
        self._update_event.clear()

    def shutdown(self) -> None:
        # 公开收口后台 producer 的退出信号，避免 trainer 直接写 manager 私有状态。
        self._status = AgentLoopManagerStatus.FINISH
        self._update_event.set()
        self._finish_event.set()

    async def _wait_for_status_exit(self, blocked_status: AgentLoopManagerStatus) -> None:
        while not self._finish_event.is_set() and self._status == blocked_status:
            await asyncio.sleep(self._STATUS_POLL_INTERVAL_S)

    async def produce_loop(self, batch_size: int) -> None:
        # `produce_loop()` 是非共卡后台生产循环：它持续把样本写入 replay buffer，
        # 前台 trainer 再通过 `get_batch()` 异步消费。
        while not self._finish_event.is_set():
            if self._status == AgentLoopManagerStatus.FINISH:
                break
            if self._status in (AgentLoopManagerStatus.UPDATE_WEIGHT_AND_ABORT, AgentLoopManagerStatus.EXPIRED_BATCH):
                # 同步前主动暂停和模型过期都只能由 trainer 调用 continue_produce() 恢复。
                await self._wait_for_status_exit(self._status)
                continue

            task_batch_sizes = self._produce_progress.ensure_target_upto(
                batch_size=batch_size,
                future_step=self._produce_progress.producer_future_step,
                allocate_batch_sizes=lambda current_batch_size, future_step: get_task_batch_sizes_for_step(
                    self,
                    current_batch_size,
                    future_step,
                ),
            )
            produce_status = await self._produce_batch_to_buffer(
                task_batch_sizes=task_batch_sizes,
                progress=self._produce_progress,
                model_step=self._model_step,
            )

            if produce_status == ProduceBatchStatus.EXPIRED_BATCH:
                # EXPIRED_BATCH 是 producer 自己检测出来的“立即停下”信号。
                self._status = AgentLoopManagerStatus.EXPIRED_BATCH
            elif produce_status == ProduceBatchStatus.NORMAL:
                # 只有正常完成一轮生产时，producer 自己维护的 train_step 才前进一步。
                self._produce_progress.advance_future_step()

            # 主动让出事件循环，避免 fake strategy / 极快路径在测试里造成忙等空转。
            await asyncio.sleep(0)

    async def get_batch(self, batch_size: int, train_step: int) -> ProduceBatchResult:
        # `get_batch()` 是非共卡路径给 trainer 的消费接口；允许空 batch 的唯一合法场景是：
        # manager 已进入 EXPIRED_BATCH，且训练侧已经有比 rollout 侧更新的 Model Step。
        progress = self._produce_progress
        progress.begin_consume(train_step)
        await refresh_for_all_tasks(
            task_runners=self.task_runners,
            replay_buffer=self.replay_buffer,
            logger=self.logger,
            manager_name=self.name,
            train_step=train_step,
            statuses=[Status.COMPLETED, Status.ABORTED],
        )
        task_batch_sizes = get_task_batch_sizes_for_step(self, batch_size, train_step)
        current_model_step = train_step - 1

        while not self._finish_event.is_set():
            if self._status == AgentLoopManagerStatus.EXPIRED_BATCH:
                if current_model_step > self._model_step:
                    pause_time_s = self._consume_pause_time()
                    result = ProduceBatchResult(
                        rollout_states=[],
                        status=ProduceBatchStatus.EXPIRED_BATCH,
                    )
                    if pause_time_s > 0:
                        result.group_gen_pause_time_s = pause_time_s
                    return result
                # 没有更新模型且当前 batch 不 ready 时，producer 已停且无法靠同步恢复，必须立即暴露不变量。
                if not await self.replay_buffer.is_ready(task_batch_sizes):
                    leftover_counts = await self.replay_buffer.count_statuses(self.task_names, _LEFTOVER_STATUSES)
                    raise RuntimeError(
                        "AgentLoopManager reached EXPIRED_BATCH without a newer model or a ready batch: "
                        f"train_step={train_step}, current_model_step={current_model_step}, "
                        f"rollout_model_step={self._model_step}, manager_status={self._status.name}, "
                        f"producer_future_step={progress.producer_future_step}, "
                        f"next_consumer_step={progress.next_consumer_step}, "
                        f"target_upto_future_step={progress.target_upto_future_step}, "
                        f"target_samples={progress.target_samples}, "
                        f"consumed_samples={progress.consumed_samples}, "
                        f"task_batch_sizes={task_batch_sizes}, "
                        f"leftover_status_counts={leftover_counts}"
                    )
            if await self.replay_buffer.is_ready(task_batch_sizes):
                result = await take_train_batch(
                    task_runners=self.task_runners,
                    replay_buffer=self.replay_buffer,
                    logger=self.logger,
                    manager_name=self.name,
                    batch_size=batch_size,
                    task_batch_sizes=task_batch_sizes,
                    progress=progress,
                    pause_time_s=self._consume_pause_time(),
                )
                if self._status == AgentLoopManagerStatus.EXPIRED_BATCH:
                    # expired 但带数据表示 trainer 仍需完成本 step，再用新 Model Step 恢复 producer。
                    result.status = ProduceBatchStatus.EXPIRED_BATCH
                if result.rollout_states:
                    progress.finish_consume(train_step)
                    await refresh_for_all_tasks(
                        task_runners=self.task_runners,
                        replay_buffer=self.replay_buffer,
                        logger=self.logger,
                        manager_name=self.name,
                        train_step=train_step + 1,
                        statuses=[Status.COMPLETED, Status.ABORTED],
                    )
                    return result
            await asyncio.sleep(self._STATUS_POLL_INTERVAL_S)

        return ProduceBatchResult(rollout_states=[])

    async def save(self, checkpoint_path: Path | str, model_step: int) -> None:
        """Save all task sampler states, replay buffer, and disaggregated
        progress."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        pending_task_counts = get_pending_task_counts(self.task_runners)
        if pending_task_counts:
            raise RuntimeError(
                "Cannot save AgentLoopManager while pending rollout tasks still exist: "
                f"{pending_task_counts}. Call pause_produce() first."
            )
        self._model_step = model_step
        for task in self.task_runners:
            checkpoint_dir = task_checkpoint_path(checkpoint_path, task.task_name)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            task.sampler.save(checkpoint_dir)
        await self.replay_buffer.save(checkpoint_path)
        state_path = manager_state_path(checkpoint_path)
        progress_state = self._produce_progress.state_dict()
        with state_path.open("w") as f:
            json.dump(
                {
                    "status": self._status.name,
                    "model_step": self._model_step,
                    **progress_state,
                },
                f,
            )

    async def resume(self, checkpoint_path: Path | str) -> int:
        """Resume all task sampler states, replay buffer, and disaggregated
        progress."""
        checkpoint_path = Path(checkpoint_path)
        for task in self.task_runners:
            task.sampler.resume(task_checkpoint_path(checkpoint_path, task.task_name))
        await self.replay_buffer.resume(checkpoint_path)

        state_path = manager_state_path(checkpoint_path)
        with state_path.open("r") as f:
            manager_state = json.load(f)
        saved_model_step = manager_state["model_step"]
        self._produce_progress.load_state_dict(manager_state)

        self._update_event = asyncio.Event()
        self._finish_event = asyncio.Event()
        self._update_event.set()
        self._status = AgentLoopManagerStatus.UPDATE_WEIGHT_AND_ABORT
        self._pause_time_s = 0.0
        self._model_step = saved_model_step
        return saved_model_step
