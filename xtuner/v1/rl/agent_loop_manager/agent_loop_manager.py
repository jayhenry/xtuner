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
from xtuner.v1.rl.agent_loop import AgentLoopConfig, AgentLoopSpec
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
    IsValidSampleFn,
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

    @property
    def is_valid_sample_fn(self) -> IsValidSampleFn:
        return getattr(self.produce_strategy, "is_valid_sample_fn", default_is_valid_sample_fn)

    @property
    def stale_threshold(self) -> int | None:
        return getattr(self.produce_strategy, "stale_threshold", None)


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


def allocate_task_batch_sizes(
    task_runners: list[_TaskRunner],
    global_batch_size: int,
    train_step: int,
) -> dict[str, int]:
    # train_step 只为后台 progress 回调保留同一形状；当前按静态 weight 分配。
    if global_batch_size < 0:
        raise ValueError(f"global_batch_size must be non-negative, got {global_batch_size}")

    total_weight = sum(task.weight for task in task_runners)
    if total_weight <= 0:
        raise ValueError("Sum of task weights must be positive.")
    if global_batch_size == 0:
        task_batch_sizes = {task.task_name: 0 for task in task_runners}
    else:
        raw_allocations = [global_batch_size * task.weight / total_weight for task in task_runners]
        floor_allocations = [math.floor(raw) for raw in raw_allocations]
        remaining = global_batch_size - sum(floor_allocations)

        task_batch_sizes = {task.task_name: floor_allocations[idx] for idx, task in enumerate(task_runners)}
        ranked_tasks = sorted(
            enumerate(task_runners),
            key=lambda item: (
                -(raw_allocations[item[0]] - floor_allocations[item[0]]),
                item[1].order,
            ),
        )
        for idx, task in ranked_tasks[:remaining]:
            task_batch_sizes[task.task_name] += 1

    expected_task_names = {task.task_name for task in task_runners}
    actual_task_names = set(task_batch_sizes.keys())
    if actual_task_names != expected_task_names:
        missing_task_names = expected_task_names - actual_task_names
        extra_task_names = actual_task_names - expected_task_names
        raise ValueError(
            "Invalid task batch sizes allocated: "
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
        # 没有 stale_threshold 的同步策略按 1 处理。
        task_stale_thresholds[task.task_name] = task.stale_threshold or 1

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
    task_batch_sizes: dict[str, int],
    progress: ProduceProgress | DisaggProduceProgress,
    pause_time_s: float = 0.0,
) -> ProduceBatchResult:
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
            rollout_controller=rollout_controller,
            logger=logger,
        )


class AgentLoopManager:
    _TASK_CHECKPOINT_DIR = _TASK_CHECKPOINT_DIR
    _MANAGER_STATE_PATH = _MANAGER_STATE_PATH
    _STATUS_POLL_INTERVAL_S = _STATUS_POLL_INTERVAL_S
    task_runners: list[_TaskRunner]
    replay_buffer: ReplayBuffer
    _rollout_controller: RolloutController
    data_sampler: Sampler | _TaskSamplerView
    name: str
    logger: Any
    task_names: list[str]

    def __init__(
        self,
        task_runners: list[_TaskRunner],
        replay_buffer: ReplayBuffer,
        rollout_controller: RolloutController,
        logger=None,
    ):
        if not task_runners:
            raise ValueError("AgentLoopManager requires at least one task runner.")
        if sum(task.weight for task in task_runners) <= 0:
            raise ValueError("At least one task weight must be positive for AgentLoopManager.")

        self.task_runners = task_runners
        self.replay_buffer = replay_buffer
        self._rollout_controller = rollout_controller
        self.data_sampler = (
            task_runners[0].sampler
            if len(task_runners) == 1
            else _TaskSamplerView([task.sampler for task in task_runners])
        )
        self.name = task_runners[0].task_name if len(task_runners) == 1 else "multi_task"
        self.logger = get_logger() if logger is None else logger
        self.task_names = [task.task_name for task in task_runners]

    async def produce_batch(
        self,
        batch_size: int,
        train_step: int,
        *,
        model_step: int,
    ) -> ProduceBatchResult:
        # 共卡同步入口：生产入 buffer -> pause/drain 本轮 pending -> 取非空训练 batch。
        if batch_size <= 0:
            raise ValueError(f"produce_batch expects batch_size > 0, got {batch_size}")
        start = time.perf_counter()
        self.logger.info(
            f"[AgentLoopManager][{self.name}] Start produce_batch: train_step={train_step} model_step={model_step} batch_size={batch_size}"
        )
        current_sizes = allocate_task_batch_sizes(self.task_runners, batch_size, train_step)
        active_tasks = [task for task in self.task_runners if current_sizes[task.task_name] > 0]
        assert active_tasks, "No active tasks found"

        await self._rollout_controller.continue_generation.remote()  # type: ignore[attr-defined]
        local_progress = ProduceProgress.build(
            task_names=self.task_names,
            target_samples=current_sizes,
        )
        # 生产前刷新已有 completed / aborted 的 staleness。
        await refresh_for_all_tasks(
            task_runners=self.task_runners,
            replay_buffer=self.replay_buffer,
            logger=self.logger,
            manager_name=self.name,
            train_step=train_step,
            statuses=[Status.COMPLETED, Status.ABORTED],
        )
        produce_start = time.perf_counter()
        produce_futures = []
        for task in active_tasks:
            produce_strategy = cast(ProduceStrategy, task.produce_strategy)
            produce_futures.append(
                produce_strategy.produce_batch(
                    ProduceContext(
                        agent_loop=task.agent_loop,
                        sampler=task.sampler,
                        replay_buffer=self.replay_buffer,
                        task_batch_size=current_sizes[task.task_name],
                        task_name=task.task_name,
                        train_step=train_step,
                        model_step=model_step,
                        progress=local_progress,
                        is_valid_sample_fn=task.is_valid_sample_fn,
                        stale_threshold=task.stale_threshold,
                    )
                )
            )
        await asyncio.gather(*produce_futures)
        local_progress.add_produce_time(time.perf_counter() - produce_start)

        # pause 只收尾本轮本地 pending。
        await self._rollout_controller.pause_generation.remote()  # type: ignore[attr-defined]

        pause_time_s = 0.0
        for task in active_tasks:
            produce_strategy = cast(ProduceStrategy, task.produce_strategy)
            pause_time_s += await produce_strategy.pause_produce(
                ProduceContext(
                    agent_loop=task.agent_loop,
                    sampler=task.sampler,
                    replay_buffer=self.replay_buffer,
                    task_batch_size=0,
                    task_name=task.task_name,
                    train_step=train_step,
                    model_step=model_step,
                    progress=local_progress,
                    is_valid_sample_fn=task.is_valid_sample_fn,
                    stale_threshold=task.stale_threshold,
                )
            )
        result = await take_train_batch(
            task_runners=self.task_runners,
            replay_buffer=self.replay_buffer,
            logger=self.logger,
            manager_name=self.name,
            task_batch_sizes=current_sizes,
            progress=local_progress,
            pause_time_s=pause_time_s,
        )
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
        await self.replay_buffer.save(checkpoint_path)
        state_path = manager_state_path(checkpoint_path)
        with state_path.open("w") as f:
            json.dump({"model_step": model_step}, f)

    async def resume(self, checkpoint_path: Path | str) -> int:
        """Resume all task sampler states and the shared replay buffer."""
        checkpoint_path = Path(checkpoint_path)
        for task in self.task_runners:
            task.sampler.resume(task_checkpoint_path(checkpoint_path, task.task_name))
        await self.replay_buffer.resume(checkpoint_path)

        state_path = manager_state_path(checkpoint_path)
        with state_path.open("r") as f:
            manager_state = json.load(f)
        return manager_state["model_step"]


class DisaggAgentLoopManager:
    """非共卡后台 producer / 前台 consumer 状态机。"""

    _TASK_CHECKPOINT_DIR = _TASK_CHECKPOINT_DIR
    _MANAGER_STATE_PATH = _MANAGER_STATE_PATH
    _STATUS_POLL_INTERVAL_S = _STATUS_POLL_INTERVAL_S
    task_runners: list[_TaskRunner]
    replay_buffer: ReplayBuffer
    _rollout_controller: RolloutController
    data_sampler: Sampler | _TaskSamplerView
    name: str
    logger: Any
    task_names: list[str]

    def __init__(
        self,
        task_runners: list[_TaskRunner],
        replay_buffer: ReplayBuffer,
        rollout_controller: RolloutController,
        logger=None,
    ):
        if not task_runners:
            raise ValueError("DisaggAgentLoopManager requires at least one task runner.")
        if sum(task.weight for task in task_runners) <= 0:
            raise ValueError("At least one task weight must be positive for DisaggAgentLoopManager.")

        self.task_runners = task_runners
        self.replay_buffer = replay_buffer
        self._rollout_controller = rollout_controller
        self.data_sampler = (
            task_runners[0].sampler
            if len(task_runners) == 1
            else _TaskSamplerView([task.sampler for task in task_runners])
        )
        self.name = task_runners[0].task_name if len(task_runners) == 1 else "multi_task"
        self.logger = get_logger() if logger is None else logger
        self.task_names = [task.task_name for task in task_runners]

        # consumer 同步权重前置位；producer / strategy 直接观察 event。
        self._update_event = asyncio.Event()
        self._finish_event = asyncio.Event()

        # rollout 侧当前模型版本；pause 清空 pending 后才能更新。
        self._model_step = 0

        # 跨 await 直接读 self._status，避免错过状态变化。
        self._status = AgentLoopManagerStatus.NORMAL

        # pause_produce 写入，下一次 get_batch 消费并清零。
        self._pause_time_s = 0.0

        # producer / consumer 共享绝对进度；对象引用保持稳定。
        self._produce_progress = DisaggProduceProgress.build(self.task_names)

    def _consume_pause_time(self) -> float:
        pause_time_s = self._pause_time_s
        self._pause_time_s = 0.0
        return pause_time_s

    async def _produce_batch_to_buffer(
        self,
        task_batch_sizes: dict[str, int],
        progress: DisaggProduceProgress,
    ) -> ProduceBatchStatus:
        producer_train_step = progress.producer_future_step
        expired_tasks = []
        for task in self.task_runners:
            produce_strategy = cast(DisaggProduceStrategy, task.produce_strategy)
            if produce_strategy.is_model_expired(producer_train_step, self._model_step):
                expired_tasks.append(task.task_name)
        if expired_tasks:
            self.logger.info(
                f"[DisaggAgentLoopManager][{self.name}] EXPIRED_BATCH: "
                f"future_step={producer_train_step}, tasks={expired_tasks}"
            )
            return ProduceBatchStatus.EXPIRED_BATCH

        active_tasks = [task for task in self.task_runners if progress.target_samples[task.task_name] > 0]
        assert active_tasks, "No active tasks found"

        produce_start = time.perf_counter()
        produce_futures = []
        for task in active_tasks:
            produce_strategy = cast(DisaggProduceStrategy, task.produce_strategy)
            produce_futures.append(
                produce_strategy.produce_batch(
                    DisaggProduceContext(
                        agent_loop=task.agent_loop,
                        sampler=task.sampler,
                        replay_buffer=self.replay_buffer,
                        task_batch_size=task_batch_sizes[task.task_name],
                        task_name=task.task_name,
                        train_step=producer_train_step,
                        model_step=self._model_step,
                        progress=progress,
                        update_event=self._update_event,
                        is_valid_sample_fn=task.is_valid_sample_fn,
                        stale_threshold=task.stale_threshold,
                    )
                )
            )
        produce_status = _aggregate_status(await asyncio.gather(*produce_futures))
        progress.add_produce_time(time.perf_counter() - produce_start)
        return produce_status

    async def pause_produce(self) -> float:
        # 非共卡显式刹车；共卡没有 public pause。
        self._status = AgentLoopManagerStatus.UPDATE_WEIGHT_AND_ABORT
        self._update_event.set()
        await self._rollout_controller.pause_generation.remote()  # type: ignore[attr-defined]

        pause_time_s = 0.0
        for task in self.task_runners:
            produce_strategy = cast(DisaggProduceStrategy, task.produce_strategy)
            ctx = DisaggProduceContext(
                agent_loop=task.agent_loop,
                sampler=task.sampler,
                replay_buffer=self.replay_buffer,
                task_batch_size=0,
                task_name=task.task_name,
                train_step=self._produce_progress.producer_future_step,
                model_step=self._model_step,
                progress=self._produce_progress,
                update_event=self._update_event,
                is_valid_sample_fn=task.is_valid_sample_fn,
                stale_threshold=task.stale_threshold,
            )
            pause_time_s += await produce_strategy.pause_produce(ctx)
        self._pause_time_s = pause_time_s
        return pause_time_s

    async def continue_produce(self, model_step: int) -> None:
        # 与 pause_produce 成对：同步/评测完成后，用新 model_step 恢复后台 producer。
        self._model_step = model_step
        await self._rollout_controller.continue_generation.remote()  # type: ignore[attr-defined]
        self._status = AgentLoopManagerStatus.NORMAL
        self._update_event.clear()

    def shutdown(self) -> None:
        self._status = AgentLoopManagerStatus.FINISH
        self._update_event.set()
        self._finish_event.set()

    async def _wait_for_status_exit(self, blocked_status: AgentLoopManagerStatus) -> None:
        while not self._finish_event.is_set() and self._status == blocked_status:
            await asyncio.sleep(self._STATUS_POLL_INTERVAL_S)

    async def produce_loop(self, batch_size: int) -> None:
        # 后台持续生产；前台通过 get_batch 消费。
        while not self._finish_event.is_set():
            if self._status == AgentLoopManagerStatus.FINISH:
                break
            if self._status in (AgentLoopManagerStatus.UPDATE_WEIGHT_AND_ABORT, AgentLoopManagerStatus.EXPIRED_BATCH):
                # 暂停/过期只能由 trainer 调用 continue_produce 恢复。
                await self._wait_for_status_exit(self._status)
                continue

            task_batch_sizes = self._produce_progress.ensure_target_upto(
                batch_size=batch_size,
                future_step=self._produce_progress.producer_future_step,
                allocate_batch_sizes=lambda current_batch_size, future_step: allocate_task_batch_sizes(
                    self.task_runners,
                    current_batch_size,
                    future_step,
                ),
            )
            produce_status = await self._produce_batch_to_buffer(task_batch_sizes, self._produce_progress)

            if produce_status == ProduceBatchStatus.EXPIRED_BATCH:
                # EXPIRED_BATCH 是 producer 自己检测出来的“立即停下”信号。
                self._status = AgentLoopManagerStatus.EXPIRED_BATCH
            elif produce_status == ProduceBatchStatus.NORMAL:
                # 只有正常完成一轮生产时，producer 自己维护的 train_step 才前进一步。
                self._produce_progress.advance_future_step()

            # 极快路径下主动让出事件循环。
            await asyncio.sleep(0)

    async def get_batch(self, batch_size: int, train_step: int) -> ProduceBatchResult:
        # 非共卡消费入口；空 batch 只表示已过期且已有更新模型可同步。
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
        task_batch_sizes = allocate_task_batch_sizes(self.task_runners, batch_size, train_step)
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
                # producer 已停且没有新模型可同步，立即暴露坏状态。
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
                    task_batch_sizes=task_batch_sizes,
                    progress=progress,
                    pause_time_s=self._consume_pause_time(),
                )
                if self._status == AgentLoopManagerStatus.EXPIRED_BATCH:
                    # 有数据的 expired batch 仍需训练本 step。
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
        """保存非共卡 sampler、replay buffer 和后台生产进度。"""
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
        """恢复非共卡 sampler、replay buffer 和后台生产进度。"""
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
