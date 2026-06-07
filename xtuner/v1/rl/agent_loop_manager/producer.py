import asyncio
import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Protocol, runtime_checkable


if TYPE_CHECKING:
    from xtuner.v1.rl.rollout.controller import RolloutControllerProxy

import ray
import tqdm
from mmengine.dist import get_rank
from pydantic import BaseModel, ConfigDict, Field

from xtuner.v1.data_proto.rl_data import (
    RolloutState,
    Status,
    get_group_status,
    reset_rollout_response,
)
from xtuner.v1.rl.agent_loop import AgentLoopSpec
from xtuner.v1.rl.replay_buffer import ReplayBuffer
from xtuner.v1.rl.utils import (
    AGENT_LOOP_PAUSE_REQUEST_TIMEOUT_S,
    PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S,
    calculate_seq_staleness,
    cancel_and_drain,
    create_task,
)
from xtuner.v1.utils import get_logger

from .sampler import Sampler


logger = get_logger()
GROUP_GENERATE_TIME_KEY = "group_generate_time_s"
PERIODIC_ABORT_INTERVAL_S = 5.0


class _ProgressDisplayer:
    def __init__(self, progress_bar: Any | None) -> None:
        self._tqdm = progress_bar

    @classmethod
    def create(cls, *, strategy_name: str, task_name: str, total: int, initial: int) -> "_ProgressDisplayer":
        total = max(0, total)
        initial = min(total, max(0, initial))
        if total <= 0 or get_rank() != 0:
            return cls(None)
        return cls(
            tqdm.tqdm(
                total=total,
                initial=initial,
                desc=f"{strategy_name} {task_name}",
                unit="sample",
                dynamic_ncols=True,
                mininterval=30,
                leave=False,
            )
        )

    def update(self, value: int) -> None:
        if self._tqdm is None:
            return
        total = max(0, int(self._tqdm.total or 0))
        value = min(total, max(0, value))
        delta = value - self._tqdm.n
        if delta > 0:
            self._tqdm.update(delta)
            self._tqdm.n = value

    def close(self) -> None:
        if self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None


@dataclass
class ProduceProgress:
    """共卡单次 produce_batch 的局部进度。

    中文不变量：
    - 只表达本次调用，不进入 checkpoint。
    - 裁剪非共卡需要的 next_consumer_step / consumed_samples / target_upto_future_step / state_dict。
    - 不新增 model_step，model_step 仍由 manager 放进 ProduceContext。
    """

    producer_future_step: int = 1
    target_samples: dict[str, int] = field(default_factory=dict)
    raw_rewards_sum: dict[str, float] = field(default_factory=dict)
    raw_rewards_count: dict[str, int] = field(default_factory=dict)
    produced_samples: dict[str, int] = field(default_factory=dict)
    produced_tokens: dict[str, int] = field(default_factory=dict)
    produce_time_s: float = 0.0

    @classmethod
    def build(
        cls,
        *,
        task_names: list[str],
        target_samples: dict[str, int],
        train_step: int = 1,
    ) -> "ProduceProgress":
        return cls(
            producer_future_step=train_step,
            target_samples=dict(target_samples),
            raw_rewards_sum={task_name: 0.0 for task_name in task_names},
            raw_rewards_count={task_name: 0 for task_name in task_names},
            produced_samples={task_name: 0 for task_name in task_names},
            produced_tokens={task_name: 0 for task_name in task_names},
        )

    def add_raw_rewards(self, task_name: str, rewards_sum: float, rewards_count: int) -> None:
        self.raw_rewards_sum[task_name] += rewards_sum
        self.raw_rewards_count[task_name] += rewards_count

    def add_produced(self, task_name: str, samples: int, tokens: int) -> None:
        self.produced_samples[task_name] += samples
        self.produced_tokens[task_name] += tokens

    def add_produce_time(self, elapsed_s: float) -> None:
        self.produce_time_s += elapsed_s

    def consume_produced(self, task_name: str) -> tuple[int, int]:
        samples = self.produced_samples[task_name]
        tokens = self.produced_tokens[task_name]
        self.produced_samples[task_name] = 0
        self.produced_tokens[task_name] = 0
        return samples, tokens

    def consume_produce_time(self) -> float:
        produce_time_s = self.produce_time_s
        self.produce_time_s = 0.0
        return produce_time_s

    def consume_raw_rewards(self, task_name: str) -> tuple[float, int]:
        rewards_sum = self.raw_rewards_sum[task_name]
        rewards_count = self.raw_rewards_count[task_name]
        self.raw_rewards_sum[task_name] = 0.0
        self.raw_rewards_count[task_name] = 0
        return rewards_sum, rewards_count


@dataclass
class DisaggProduceProgress:
    """非共卡 Background Producer / Training Consumer 共享进度。"""

    task_names: list[str] = field(default_factory=list)
    producer_future_step: int = 1
    next_consumer_step: int = 1
    target_upto_future_step: int = 0
    consumed_samples: dict[str, int] = field(default_factory=dict)
    target_samples: dict[str, int] = field(default_factory=dict)
    raw_rewards_sum: dict[str, float] = field(default_factory=dict)
    raw_rewards_count: dict[str, int] = field(default_factory=dict)
    produced_samples: dict[str, int] = field(default_factory=dict)
    produced_tokens: dict[str, int] = field(default_factory=dict)
    produce_time_s: float = 0.0

    @classmethod
    def build(cls, task_names: list[str]) -> "DisaggProduceProgress":
        return cls(
            task_names=list(task_names),
            consumed_samples={task_name: 0 for task_name in task_names},
            target_samples={task_name: 0 for task_name in task_names},
            raw_rewards_sum={task_name: 0.0 for task_name in task_names},
            raw_rewards_count={task_name: 0 for task_name in task_names},
            produced_samples={task_name: 0 for task_name in task_names},
            produced_tokens={task_name: 0 for task_name in task_names},
        )

    def ensure_target_upto(
        self,
        *,
        batch_size: int,
        future_step: int,
        allocate_batch_sizes: Callable[[int, int], dict[str, int]],
    ) -> dict[str, int]:
        """把累计 target 推进到指定 future step，并返回该 step 的 task batch size。"""

        current_task_batch_sizes: dict[str, int] | None = None
        if future_step > self.target_upto_future_step:
            for step in range(self.target_upto_future_step + 1, future_step + 1):
                current_task_batch_sizes = allocate_batch_sizes(batch_size, step)
                for task_name, task_batch_size in current_task_batch_sizes.items():
                    self.target_samples[task_name] += task_batch_size
            self.target_upto_future_step = future_step

        if current_task_batch_sizes is None:
            current_task_batch_sizes = allocate_batch_sizes(batch_size, future_step)
        return current_task_batch_sizes

    def begin_consume(self, train_step: int) -> None:
        self.next_consumer_step = train_step

    def mark_consumed(self, consumed_counts: dict[str, int]) -> None:
        # consumer 真实取出多少就累计多少，target 不回退，避免 producer 把已消费样本当成缺口。
        for task_name, count in consumed_counts.items():
            self.consumed_samples[task_name] += count

    def finish_consume(self, train_step: int) -> None:
        self.next_consumer_step = train_step + 1

    def advance_future_step(self) -> None:
        self.producer_future_step += 1

    def add_raw_rewards(self, task_name: str, rewards_sum: float, rewards_count: int) -> None:
        self.raw_rewards_sum[task_name] += rewards_sum
        self.raw_rewards_count[task_name] += rewards_count

    def add_produced(self, task_name: str, samples: int, tokens: int) -> None:
        self.produced_samples[task_name] += samples
        self.produced_tokens[task_name] += tokens

    def add_produce_time(self, elapsed_s: float) -> None:
        self.produce_time_s += elapsed_s

    def consume_produced(self, task_name: str) -> tuple[int, int]:
        samples = self.produced_samples[task_name]
        tokens = self.produced_tokens[task_name]
        self.produced_samples[task_name] = 0
        self.produced_tokens[task_name] = 0
        return samples, tokens

    def consume_produce_time(self) -> float:
        produce_time_s = self.produce_time_s
        self.produce_time_s = 0.0
        return produce_time_s

    def consume_raw_rewards(self, task_name: str) -> tuple[float, int]:
        rewards_sum = self.raw_rewards_sum[task_name]
        rewards_count = self.raw_rewards_count[task_name]
        self.raw_rewards_sum[task_name] = 0.0
        self.raw_rewards_count[task_name] = 0
        return rewards_sum, rewards_count

    def state_dict(self) -> dict[str, Any]:
        return {
            "producer_future_step": self.producer_future_step,
            "next_consumer_step": self.next_consumer_step,
            "target_upto_future_step": self.target_upto_future_step,
            "consumed_samples": dict(self.consumed_samples),
            "target_samples": dict(self.target_samples),
            "raw_rewards_sum": dict(self.raw_rewards_sum),
            "raw_rewards_count": dict(self.raw_rewards_count),
            "produced_samples": dict(self.produced_samples),
            "produced_tokens": dict(self.produced_tokens),
            "produce_time_s": self.produce_time_s,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # 原地更新 dict，避免 strategy / context 持有旧引用。
        self.producer_future_step = state["producer_future_step"]
        self.next_consumer_step = state["next_consumer_step"]
        self.target_upto_future_step = state["target_upto_future_step"]
        self.consumed_samples.clear()
        self.consumed_samples.update(state["consumed_samples"])
        self.target_samples.clear()
        self.target_samples.update(state["target_samples"])
        task_names = set(self.consumed_samples) | set(self.target_samples)
        self.raw_rewards_sum.clear()
        self.raw_rewards_sum.update(
            {task_name: float(state.get("raw_rewards_sum", {}).get(task_name, 0.0)) for task_name in task_names}
        )
        self.raw_rewards_count.clear()
        self.raw_rewards_count.update(
            {task_name: int(state.get("raw_rewards_count", {}).get(task_name, 0)) for task_name in task_names}
        )
        produced_samples_state = state.get("produced_samples", {})
        produced_tokens_state = state.get("produced_tokens", {})
        self.produced_samples.clear()
        self.produced_samples.update(
            {task_name: int(produced_samples_state.get(task_name, 0)) for task_name in task_names}
        )
        self.produced_tokens.clear()
        self.produced_tokens.update(
            {task_name: int(produced_tokens_state.get(task_name, 0)) for task_name in task_names}
        )
        self.produce_time_s = float(state.get("produce_time_s", 0.0))


class ProduceBatchStatus(Enum):
    NORMAL = auto()
    UPDATE_WEIGHT_AND_ABORT = auto()
    EXPIRED_BATCH = auto()


def default_is_valid_sample_fn(samples: list[RolloutState]) -> bool:
    return True


def default_should_continue_fn(completed_count: int, batch_size: int, **kwargs) -> bool:
    return completed_count < batch_size


def calculate_stale_threshold(max_staleness: int, sync_weights_interval: int) -> int:
    if max_staleness < 0:
        raise ValueError(f"max_staleness must be non-negative, got {max_staleness}.")
    if sync_weights_interval <= 0:
        raise ValueError(f"sync_weights_interval must be positive, got {sync_weights_interval}.")

    # max_staleness 按同步周期计数；+1 表示训练天然必须接受的当前同步周期滞后。
    return (max_staleness + 1) * sync_weights_interval


@runtime_checkable
class IsValidSampleFn(Protocol):
    def __call__(self, samples: list[RolloutState]) -> bool: ...


@runtime_checkable
class ShouldContinueFn(Protocol):
    def __call__(self, completed_count: int, batch_size: int, **kwargs) -> bool: ...


@dataclass(kw_only=True)
class BaseProduceContext:
    """单 task 生产上下文的共享能力。

    这里保留共卡和非共卡都需要的 sample/generate/put 运行时契约：
    - rollout generate 的 ray/local 差异和 timing 字段写入；
    - 生成结果先按业务有效性过滤，再统一交给 ReplayBuffer 写版本、刷新 staleness、执行过期。
    非共卡独有的 update_event / 绝对进度访问只放在 DisaggProduceContext。
    """

    agent_loop: AgentLoopSpec
    sampler: Sampler
    replay_buffer: ReplayBuffer
    task_batch_size: int
    task_name: str
    train_step: int
    model_step: int
    progress: ProduceProgress | DisaggProduceProgress
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    stale_threshold: int | None = None

    @property
    def consumer_step(self) -> int:
        return self.train_step

    async def expired_count(self) -> int:
        return await self.replay_buffer.count(task_name=self.task_name, group_status=Status.EXPIRED)

    async def sample_group(self, *, from_expired_pool: bool) -> list[RolloutState]:
        group_status = [Status.EXPIRED, Status.ABORTED] if from_expired_pool else [Status.ABORTED]
        return await self.sampler.sample(task_name=self.task_name, group_status=group_status)

    async def generate_group(
        self,
        rollout_state: list[RolloutState],
        *,
        enable_partial_rollout: bool = False,
    ) -> list[RolloutState]:
        # strategy 只表达“要生成”，不关心 agent_loop 是 ray actor 还是本地对象。
        start = time.perf_counter()
        if isinstance(self.agent_loop, ray.actor.ActorHandle):
            result = await self.agent_loop.generate_group.remote(
                rollout_state,
                enable_partial_rollout=enable_partial_rollout,
            )
        else:
            result = await self.agent_loop.generate_group(
                rollout_state,
                enable_partial_rollout=enable_partial_rollout,
            )
        elapsed = time.perf_counter() - start
        for item in result:
            extra_fields = getattr(item, "extra_fields", None)
            if extra_fields is None:
                extra_fields = {}
                setattr(item, "extra_fields", extra_fields)
            extra_fields[GROUP_GENERATE_TIME_KEY] = elapsed
        return result

    async def put_generated_group(self, group: list[RolloutState]) -> bool:
        # 只有完整生成的 group 才需要业务有效性过滤；ABORTED / EXPIRED 保留原状态供重试或统计。
        is_completed = get_group_status(group) == Status.COMPLETED
        produced_tokens = sum(len(item.response_ids) for item in group if item.response_ids is not None)
        if is_completed:
            rewards_sum = 0.0
            rewards_count = 0
            for item in group:
                if item.reward is None or "score" not in item.reward:
                    logger.warning(
                        f"Missing reward score in item (uid: {item.uid}) of completed group for task {self.task_name}. This item will be skipped in reward statistics."
                    )
                    continue
                rewards_sum += float(item.reward["score"])  # type: ignore[index]
                rewards_count += 1
            self.progress.add_raw_rewards(self.task_name, rewards_sum, rewards_count)
            is_valid = self.is_valid_sample_fn(group)
            if not is_valid:
                for item in group:
                    item.status = Status.FILTERED
                    reset_rollout_response(item)
        await self.replay_buffer.put(
            group,
            self.task_name,
            model_step=self.model_step,
            current_train_step=self.consumer_step,
            stale_threshold=self.stale_threshold,
        )
        self.progress.add_produced(self.task_name, samples=len(group), tokens=produced_tokens)
        # replay_buffer.put 可能把 stale group 转为 EXPIRED，返回前重新判断是否仍可训练。
        is_completed = get_group_status(group) == Status.COMPLETED
        return is_completed


@dataclass(kw_only=True)
class ProduceContext(BaseProduceContext):
    """共卡 strategy context。

    共卡只表达一次 produce_batch 的本地生产窗口，不暴露 update_event / should_abort / 绝对 consumed+completed 口径，避免把非共卡后台状态机语义泄漏进同步生产路径。
    """

    @property
    def target_count(self) -> int:
        return self.progress.target_samples[self.task_name]

    async def completed_count(self) -> int:
        return await self.replay_buffer.count(task_name=self.task_name, group_status=Status.COMPLETED)


@dataclass(kw_only=True)
class DisaggProduceContext(BaseProduceContext):
    """非共卡 strategy context。"""

    progress: DisaggProduceProgress
    update_event: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def consumer_step(self) -> int:
        return self.progress.next_consumer_step

    @property
    def target_abs(self) -> int:
        return self.progress.target_samples[self.task_name]

    def should_abort(self) -> bool:
        return self.update_event.is_set()

    async def available_count(self) -> int:
        completed_count = await self.replay_buffer.count(task_name=self.task_name, group_status=Status.COMPLETED)
        return self.progress.consumed_samples[self.task_name] + completed_count


class ProduceStrategyConfig(ABC, BaseModel):
    """Base configuration for rollout production strategies.

    Production strategies decide how the agent loop fills the replay buffer and
    when it should stop producing samples for the current training step.

    Args:
        is_valid_sample_fn (IsValidSampleFn): Function used to decide whether a
            generated rollout group is trainable. Defaults to
            ``default_is_valid_sample_fn``.
        should_continue_fn (ShouldContinueFn): Function used to decide whether
            production should continue after a group is processed. Defaults to
            ``default_should_continue_fn``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    should_continue_fn: ShouldContinueFn = default_should_continue_fn

    @abstractmethod
    def build(
        self,
        *,
        sync_weights_interval: int = 1,
        rollout_controller: "Optional[RolloutControllerProxy]" = None,
    ) -> "ProduceStrategy": ...


class DisaggProduceStrategyConfig(ABC, BaseModel):
    """非共卡后台 producer strategy 配置。

    共卡和非共卡使用不同 Config 类型表达执行环境，避免在 strategy build 里用 mode 分支切换语义。
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    should_continue_fn: ShouldContinueFn = default_should_continue_fn

    @abstractmethod
    def build(
        self,
        *,
        sync_weights_interval: int = 1,
        rollout_controller: "Optional[RolloutControllerProxy]" = None,
    ) -> "DisaggProduceStrategy": ...


class SyncProduceStrategyConfig(ProduceStrategyConfig):
    """Configuration for synchronous rollout production.

    The synchronous strategy produces samples on demand for the current training
    step. It is simpler and is the default choice when rollout and training run
    in a colocated or tightly synchronized workflow.

    Args:
        is_valid_sample_fn (IsValidSampleFn): Function used to decide whether a
            generated rollout group is trainable. Defaults to
            ``default_is_valid_sample_fn``.
        should_continue_fn (ShouldContinueFn): Function used to decide whether
            production should continue after a group is processed. Defaults to
            ``default_should_continue_fn``.

    **Examples:**

    Example synchronous strategy::

        config = SyncProduceStrategyConfig()
    """

    def build(
        self,
        *,
        sync_weights_interval: int = 1,
        rollout_controller: "Optional[RolloutControllerProxy]" = None,
    ) -> "SyncProduceStrategy":
        return SyncProduceStrategy(
            is_valid_sample_fn=self.is_valid_sample_fn, should_continue_fn=self.should_continue_fn
        )


class AsyncProduceStrategyConfig(ProduceStrategyConfig):
    """Configuration for colocated asynchronous rollout production.

    The colocated asynchronous strategy produces rollout samples concurrently
    within one ``AgentLoopManager.produce_batch`` call and stores them in the
    replay buffer. It can oversample, allow partial rollout continuation, and
    discard samples that are too stale relative to the current training step.

    Args:
        is_valid_sample_fn (IsValidSampleFn): Function used to decide whether a
            generated rollout group is trainable. Defaults to
            ``default_is_valid_sample_fn``.
        should_continue_fn (ShouldContinueFn): Function used to decide whether
            production should continue after a group is processed. Defaults to
            ``default_should_continue_fn``.
        over_sample_threshold (float): Extra completed-sample ratio allowed
            before the producer stops. Defaults to 0.0.
        enable_partial_rollout (bool): Whether unfinished rollouts can be
            continued after a weight sync. Defaults to False.
        max_staleness (int): Maximum allowed model-step staleness for replayed
            samples. Defaults to 0.
        tail_batch_trigger_size (int): Minimum pending tail size that can
            trigger a final batch. Defaults to 0.

    **Examples:**

    Example asynchronous strategy::

        config = AsyncProduceStrategyConfig(
            over_sample_threshold=0.2,
            enable_partial_rollout=True,
            max_staleness=1,
        )
    """

    over_sample_threshold: float = 0.0
    enable_partial_rollout: bool = False
    max_staleness: int = Field(default=0, ge=0)
    tail_batch_trigger_size: int = 0

    def build(
        self,
        *,
        sync_weights_interval: int = 1,
        rollout_controller: "Optional[RolloutControllerProxy]" = None,
    ) -> "AsyncProduceStrategy":
        if rollout_controller is not None:
            import ray

            ray.get(rollout_controller.set_enable_partial_rollout.remote(self.enable_partial_rollout))
        return AsyncProduceStrategy(
            over_sample_threshold=self.over_sample_threshold,
            enable_partial_rollout=self.enable_partial_rollout,
            max_staleness=self.max_staleness,
            sync_weights_interval=sync_weights_interval,
            tail_batch_trigger_size=self.tail_batch_trigger_size,
            is_valid_sample_fn=self.is_valid_sample_fn,
            should_continue_fn=self.should_continue_fn,
        )


class DisaggAsyncProduceStrategyConfig(DisaggProduceStrategyConfig):
    """非共卡异步 rollout production 配置。"""

    over_sample_threshold: float = 0.0
    enable_partial_rollout: bool = False
    max_staleness: int = Field(default=0, ge=0)
    tail_batch_trigger_size: int = 0

    def build(
        self,
        *,
        sync_weights_interval: int = 1,
        rollout_controller: "Optional[RolloutControllerProxy]" = None,
    ) -> "DisaggAsyncProduceStrategy":
        if rollout_controller is not None:
            import ray

            ray.get(rollout_controller.set_enable_partial_rollout.remote(self.enable_partial_rollout))
        return DisaggAsyncProduceStrategy(
            over_sample_threshold=self.over_sample_threshold,
            enable_partial_rollout=self.enable_partial_rollout,
            max_staleness=self.max_staleness,
            sync_weights_interval=sync_weights_interval,
            tail_batch_trigger_size=self.tail_batch_trigger_size,
            is_valid_sample_fn=self.is_valid_sample_fn,
            should_continue_fn=self.should_continue_fn,
        )


class ProduceStrategy(ABC):
    def __init__(
        self,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ):
        self.is_valid_sample_fn = is_valid_sample_fn
        self.should_continue_fn = should_continue_fn

    @abstractmethod
    async def produce_batch(self, ctx: ProduceContext) -> ProduceBatchStatus: ...

    async def pause_produce(self, ctx: ProduceContext) -> float:
        return 0.0

    def pending_task_count(self) -> int:
        return 0


class DisaggProduceStrategy(ABC):
    def __init__(
        self,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ):
        self.is_valid_sample_fn = is_valid_sample_fn
        self.should_continue_fn = should_continue_fn

    @abstractmethod
    async def produce_batch(self, ctx: DisaggProduceContext) -> ProduceBatchStatus: ...

    async def pause_produce(self, ctx: DisaggProduceContext) -> float:
        return 0.0

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        return False

    def pending_task_count(self) -> int:
        return 0


class _PendingTasks:
    """AsyncProduceStrategy 的并发 pending task 集合。

    这里只封装 pending set 的并发协议，不理解 sampler / rollout / replay buffer：
    - wait 使用快照，随后必须二次 claim，避免 produce 和 pause 重复处理同一个 done task。
    - cancel 前先原子 claim 并清空集合，避免 cancel 后又被其他路径 claim。
    - schedule one 在锁内同时检查 abort 和 pending 数，避免 pause 已触发后继续新增 task。
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    def count(self) -> int:
        # 只暴露已经纳入 pending 集合的 task 数量。
        return len(self._tasks)

    async def claim_ready(self) -> set[asyncio.Task]:
        async with self._lock:
            ready = {task for task in self._tasks if task.done()}
            self._tasks.difference_update(ready)
            return ready

    async def wait_and_claim(self, *, timeout_s: float) -> set[asyncio.Task]:
        async with self._lock:
            snapshot = set(self._tasks)
        if not snapshot:
            return set()

        done, _ = await asyncio.wait(snapshot, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
        async with self._lock:
            claimed = done & self._tasks
            self._tasks.difference_update(claimed)
            return claimed

    async def schedule_one(
        self,
        *,
        max_pending: int,
        should_abort: Callable[[], bool],
        spawn_one: Callable[[], Awaitable[asyncio.Task]],
    ) -> bool:
        async with self._lock:
            if should_abort() or len(self._tasks) >= max_pending:
                return False
            # 保持“检查 abort / pending 数 / 新增 task”这一组操作原子化。
            self._tasks.add(await spawn_one())
            return True

    async def _claim_all(self) -> set[asyncio.Task]:
        async with self._lock:
            claimed = set(self._tasks)
            self._tasks.clear()
            return claimed

    async def cancel_all(self) -> int:
        tasks = await self._claim_all()
        if not tasks:
            return 0
        logger.warning(f"Cancelling {len(tasks)} pending rollout tasks.")
        await cancel_and_drain(list(tasks))
        return len(tasks)


class _LocalPendingTasks:
    """把共卡本次调用的局部 pending set 适配成统一 drain 协议。

    共卡 pending 不跨 produce_batch 调用；这里原地修改传入的 set，让 pending_task_count() 在 pause 过程中仍能反映剩余本地任务数量。
    """

    def __init__(self, tasks: set[asyncio.Task]) -> None:
        self._tasks = tasks

    def count(self) -> int:
        return len(self._tasks)

    async def wait_and_claim(self, *, timeout_s: float) -> set[asyncio.Task]:
        if not self._tasks:
            return set()
        done, _ = await asyncio.wait(set(self._tasks), timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
        self._tasks.difference_update(done)
        return done

    async def cancel_all(self) -> int:
        tasks = set(self._tasks)
        self._tasks.clear()
        if not tasks:
            return 0
        logger.warning(f"Cancelling {len(tasks)} pending rollout tasks.")
        await cancel_and_drain(list(tasks))
        return len(tasks)


async def request_agent_loop_pause(ctx: BaseProduceContext, *, pending_count: int) -> None:
    """发送一次 agent loop pause 请求，供共卡和非共卡 pending drain 复用。"""

    pause_request_start = time.perf_counter()
    if isinstance(ctx.agent_loop, ray.actor.ActorHandle):
        pause_future = ctx.agent_loop.pause.remote()
    else:
        pause_future = ctx.agent_loop.pause()
    try:
        await asyncio.wait_for(pause_future, timeout=AGENT_LOOP_PAUSE_REQUEST_TIMEOUT_S)
    except asyncio.TimeoutError:
        logger.warning(
            f"Agent loop pause timed out: task={ctx.task_name}, timeout_s={AGENT_LOOP_PAUSE_REQUEST_TIMEOUT_S}, "
            f"elapsed={time.perf_counter() - pause_request_start:.2f}s, pending={pending_count}"
        )
    except Exception:
        logger.exception(
            f"Agent loop pause failed: task={ctx.task_name}, "
            f"elapsed={time.perf_counter() - pause_request_start:.2f}s, pending={pending_count}"
        )


async def pause_pending_tasks(
    *,
    pending_tasks: set[asyncio.Task] | _PendingTasks,
    ctx: BaseProduceContext,
    put_claimed_task: Callable[[asyncio.Task], Awaitable[Any]],
) -> float:
    """复用 pending task pause / drain / cancel 协议。

    中文不变量：
    - 先发 pause，再等待 pending 产出；
    - pending 没清空时周期性补发 pause，兼容后端 pause/abort 信号延迟；
    - 超时后 cancel 剩余 pending，避免 checkpoint/save 前仍有任务写 buffer；
    - 已完成任务必须 claim 后再 put，避免 produce 和 pause 重复入库同一个 done task。
    """

    pending = _LocalPendingTasks(pending_tasks) if isinstance(pending_tasks, set) else pending_tasks
    pause_start = time.perf_counter()
    if pending.count() == 0:
        return 0.0

    initial_pending_count = pending.count()
    logger.info(
        f"Pause signal loop started for task {ctx.task_name}. "
        f"Waiting for {initial_pending_count} pending tasks to complete. "
        f"periodic_abort_interval_s={PERIODIC_ABORT_INTERVAL_S}, "
        f"producer_pause_pending_task_timeout_s={PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S}"
    )

    pending_pause_tasks = {create_task(request_agent_loop_pause(ctx, pending_count=initial_pending_count))}
    cleanup_start_time = time.perf_counter()
    next_periodic_abort_time = cleanup_start_time + PERIODIC_ABORT_INTERVAL_S
    while True:
        elapsed_time = time.perf_counter() - cleanup_start_time
        if elapsed_time > PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S:
            cancelled_count = await pending.cancel_all()
            logger.warning(
                f"Cleanup timeout of {PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S}s reached. "
                f"Forcefully cancelling {cancelled_count} remaining tasks. task={ctx.task_name}"
            )
            break

        if pending.count() == 0:
            break
        current_time = time.perf_counter()
        pending_pause_tasks = {task for task in pending_pause_tasks if not task.done()}

        # 定时发送 pause 信号，避免后端漏掉第一次 pause 后 pending 长时间不结束。
        if PERIODIC_ABORT_INTERVAL_S > 0 and current_time >= next_periodic_abort_time:
            pending_pause_tasks.add(create_task(request_agent_loop_pause(ctx, pending_count=pending.count())))
            next_periodic_abort_time += PERIODIC_ABORT_INTERVAL_S

        claimed_done = await pending.wait_and_claim(timeout_s=1)
        for task in claimed_done:
            await put_claimed_task(task)

    await cancel_and_drain(list(pending_pause_tasks))
    pause_time = time.perf_counter() - pause_start
    logger.info(f"pause_produce completed for task {ctx.task_name} within {pause_time}s.")
    return pause_time


async def _put_claimed_tasks(
    claimed_tasks: set[asyncio.Task],
    ctx: BaseProduceContext,
    *,
    available_base: int | None = None,
    progress_displayer: _ProgressDisplayer | None = None,
) -> None:
    completed_count = 0
    for task in claimed_tasks:
        is_completed = await ctx.put_generated_group(task.result())
        if is_completed:
            completed_count += 1
        if is_completed and available_base is not None and progress_displayer is not None:
            progress_displayer.update(available_base + completed_count)


class SyncProduceStrategy(ProduceStrategy):
    async def produce_batch(self, ctx: ProduceContext) -> ProduceBatchStatus:
        pending_tasks = set()
        completed_sample_count = await ctx.replay_buffer.count(task_name=ctx.task_name, group_status=Status.COMPLETED)
        # TODO: 是否支持 SyncProduceStrategy 在非共卡时使用？如果支持，下面这行注释掉？
        # assert completed_sample_count == 0, "SyncProduceStrategy assumes no completed samples at the start."

        for _ in range(ctx.task_batch_size):
            rollout_state = await ctx.sampler.sample(task_name=ctx.task_name)
            task = create_task(ctx.generate_group(rollout_state))
            pending_tasks.add(task)

        logger.info(f"[SyncProduceStrategy] Started {len(pending_tasks)} initial tasks.")

        progress_displayer = _ProgressDisplayer.create(
            strategy_name=self.__class__.__name__,
            task_name=ctx.task_name,
            total=ctx.target_count,
            initial=completed_sample_count,
        )
        try:
            while self.should_continue_fn(completed_sample_count, ctx.task_batch_size):
                if not pending_tasks:
                    logger.warning("[SyncProduceStrategy] All tasks are done but not enough samples collected.")
                    break
                done_tasks, pending_tasks = await asyncio.wait(
                    pending_tasks, timeout=1, return_when=asyncio.FIRST_COMPLETED
                )
                # 如果要过滤，在这个地方处理，然后加入到 replay buffer
                # 如果被过滤的数据就放到 put_to_filtered pool 中
                for task in done_tasks:
                    items = task.result()

                    is_completed = await ctx.put_generated_group(items)
                    if not is_completed:
                        continue

                    completed_sample_count += 1
                    progress_displayer.update(completed_sample_count)

                while len(pending_tasks) + completed_sample_count < ctx.task_batch_size and self.should_continue_fn(
                    completed_sample_count, ctx.task_batch_size
                ):
                    rollout_state = await ctx.sampler.sample(task_name=ctx.task_name)
                    task = create_task(ctx.generate_group(rollout_state))
                    pending_tasks.add(task)
        finally:
            progress_displayer.close()

        return ProduceBatchStatus.NORMAL


class AsyncProduceStrategy(ProduceStrategy):
    # Local retry interval for re-sending pause/abort while pending tasks drain.
    PERIODIC_ABORT_INTERVAL_S = PERIODIC_ABORT_INTERVAL_S

    def __init__(
        self,
        over_sample_threshold: float,
        enable_partial_rollout: bool,
        tail_batch_trigger_size: int,
        max_staleness: int,
        sync_weights_interval: int,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ):
        super().__init__(is_valid_sample_fn, should_continue_fn)

        # TODO: 需要添加 tail_batch_max_tries
        # 作用是：如果一个样本多次重试，则将它置为特殊状态 MAX_TRIES，这类样本和过期样本一起触发tail batch逻辑
        # 这个依赖：RolloutState 添加并维护一个新的属性 num_tries，每次打断时加1，达到 max_tries 时置为 MAX_TRIES
        # 如果 enable_partial_rollout=True，不会触发这个逻辑，所以不受此影响
        # 如果 enable_partial_rollout=False，分两种情况：
        # 1) staleness = 0，即不允许过期样本，此时过期触发tail batch逻辑已经cover了tail batch逻辑
        # 2) staleness > 0，此时需要 重试tail batch逻辑，否则多次重试的样本会影响rollout 效率
        if not enable_partial_rollout and max_staleness > 0:
            logger.warning(
                "max_staleness > 0, enable_partial_rollout is False, this will affect rollout efficiency because not support tail_batch_max_tries logic now"
            )

        self.over_sample_threshold = over_sample_threshold
        self.enable_partial_rollout = enable_partial_rollout
        self.max_staleness = max_staleness
        self.sync_weights_interval = sync_weights_interval
        self.stale_threshold = calculate_stale_threshold(max_staleness, sync_weights_interval)
        self.tail_batch_trigger_size = tail_batch_trigger_size
        self._local_pending_tasks: set[asyncio.Task] = set()

    def pending_task_count(self) -> int:
        return len(self._local_pending_tasks)

    async def pause_produce(self, ctx: ProduceContext) -> float:
        return await pause_pending_tasks(
            pending_tasks=self._local_pending_tasks,
            ctx=ctx,
            put_claimed_task=lambda task: ctx.put_generated_group(task.result()),
        )

    async def produce_batch(self, ctx: ProduceContext) -> ProduceBatchStatus:
        if ctx.task_name not in ctx.progress.target_samples:
            raise KeyError(f"ProduceProgress.target_samples missing task_name={ctx.task_name!r}")

        # 共卡 async 的 pending 生命周期只属于本次 produce_batch 调用。
        # Manager 会在所有 active task produce_batch 正常返回后统一调用 pause_produce 收尾。
        self._local_pending_tasks = set()

        if ctx.target_count <= 0:
            return ProduceBatchStatus.NORMAL

        expired_count = await ctx.expired_count()
        sample_from_expired = self.tail_batch_trigger_size > 0 and expired_count >= self.tail_batch_trigger_size
        if sample_from_expired:
            logger.info(
                f"Tail batch trigger condition met: {expired_count} expired samples "
                f"(threshold: {self.tail_batch_trigger_size}). Enabling tail batch mode."
            )

        # 本轮 produce_batch 的必要累计目标固定；normal 模式只按当前 task batch 追加固定超发预算。
        # tail-batch 模式只补必要缺口，新增任务固定从 EXPIRED pool 取，不再扩大超发窗口。
        target_abs = ctx.target_count
        oversample_budget = 0 if sample_from_expired else math.ceil(self.over_sample_threshold * ctx.task_batch_size)
        scheduled_target = target_abs + oversample_budget
        logger.info(
            f"Starting produce_batch for task {ctx.task_name} with target_abs={target_abs}, "
            f"oversample_budget={oversample_budget}, scheduled_target={scheduled_target}."
        )

        async def spawn_one() -> asyncio.Task:
            rollout_state = await ctx.sample_group(from_expired_pool=sample_from_expired)
            return create_task(
                ctx.generate_group(
                    rollout_state,
                    enable_partial_rollout=self.enable_partial_rollout,
                )
            )

        initial_available = await ctx.completed_count()
        progress_displayer = _ProgressDisplayer.create(
            strategy_name=self.__class__.__name__,
            task_name=ctx.task_name,
            total=ctx.target_count,
            initial=initial_available,
        )
        try:
            while True:
                available = await ctx.completed_count()
                progress_displayer.update(available)
                if not self.should_continue_fn(available, target_abs):
                    return ProduceBatchStatus.NORMAL

                pending_count = len(self._local_pending_tasks)
                desired_pending = max(0, scheduled_target - available)
                if available + pending_count < scheduled_target:
                    while len(self._local_pending_tasks) < desired_pending:
                        self._local_pending_tasks.add(await spawn_one())

                if not self._local_pending_tasks:
                    logger.warning("All tasks are done but not enough samples collected.")
                    return ProduceBatchStatus.NORMAL

                done_tasks, _ = await asyncio.wait(
                    set(self._local_pending_tasks), timeout=1, return_when=asyncio.FIRST_COMPLETED
                )
                self._local_pending_tasks.difference_update(done_tasks)
                await _put_claimed_tasks(
                    done_tasks,
                    ctx,
                    available_base=available,
                    progress_displayer=progress_displayer,
                )
        finally:
            progress_displayer.close()


class DisaggAsyncProduceStrategy(DisaggProduceStrategy):
    """非共卡后台 producer 使用的 async strategy。

    非共卡 pending 跨后台 produce_batch 调用存在，因此这里直接实现 DisaggProduceStrategy， 不继承共卡 AsyncProduceStrategy，避免把本地 pending / 本地
    progress 语义带进后台 producer。
    """

    PERIODIC_ABORT_INTERVAL_S = PERIODIC_ABORT_INTERVAL_S

    def __init__(
        self,
        over_sample_threshold: float,
        enable_partial_rollout: bool,
        tail_batch_trigger_size: int,
        max_staleness: int,
        sync_weights_interval: int,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ):
        super().__init__(is_valid_sample_fn, should_continue_fn)

        if not enable_partial_rollout and max_staleness > 0:
            logger.warning(
                "max_staleness > 0, enable_partial_rollout is False, this will affect rollout efficiency because not support tail_batch_max_tries logic now"
            )

        self.over_sample_threshold = over_sample_threshold
        self.enable_partial_rollout = enable_partial_rollout
        self.max_staleness = max_staleness
        self.sync_weights_interval = sync_weights_interval
        self.stale_threshold = calculate_stale_threshold(max_staleness, sync_weights_interval)
        self.tail_batch_trigger_size = tail_batch_trigger_size
        self._pending_tasks = _PendingTasks()

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        staleness = calculate_seq_staleness(model_step, train_step)
        return staleness >= self.stale_threshold

    def pending_task_count(self) -> int:
        return self._pending_tasks.count()

    async def pause_produce(self, ctx: DisaggProduceContext) -> float:
        return await pause_pending_tasks(
            pending_tasks=self._pending_tasks,
            ctx=ctx,
            put_claimed_task=lambda task: ctx.put_generated_group(task.result()),
        )

    async def produce_batch(self, ctx: DisaggProduceContext) -> ProduceBatchStatus:
        if ctx.task_name not in ctx.progress.consumed_samples:
            raise KeyError(f"DisaggProduceProgress.consumed_samples missing task_name={ctx.task_name!r}")
        if ctx.task_name not in ctx.progress.target_samples:
            raise KeyError(f"DisaggProduceProgress.target_samples missing task_name={ctx.task_name!r}")

        if ctx.target_abs <= 0:
            return ProduceBatchStatus.NORMAL

        if ctx.should_abort():
            return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
        if self.is_model_expired(ctx.train_step, ctx.model_step):
            return ProduceBatchStatus.EXPIRED_BATCH

        # 非共卡 pending 跨后台 produce_batch 调用存在；进入下一轮时先回收已经完成的旧任务。
        claimed_done = await self._pending_tasks.claim_ready()
        await _put_claimed_tasks(claimed_done, ctx)

        if ctx.should_abort():
            return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
        if self.is_model_expired(ctx.train_step, ctx.model_step):
            return ProduceBatchStatus.EXPIRED_BATCH

        expired_count = await ctx.expired_count()
        sample_from_expired = self.tail_batch_trigger_size > 0 and expired_count >= self.tail_batch_trigger_size
        if sample_from_expired:
            logger.info(
                f"Tail batch trigger condition met: {expired_count} expired samples "
                f"(threshold: {self.tail_batch_trigger_size}). Enabling tail batch mode."
            )

        # 本轮 produce_batch 的必要累计目标固定；normal 模式只按当前 task batch 追加固定超发预算。
        # tail-batch 模式只补必要缺口，新增任务固定从 EXPIRED pool 取，不再扩大超发窗口。
        target_abs = ctx.target_abs
        oversample_budget = 0 if sample_from_expired else math.ceil(self.over_sample_threshold * ctx.task_batch_size)
        scheduled_target = target_abs + oversample_budget
        logger.info(
            f"Starting produce_batch for task {ctx.task_name} with target_abs={target_abs}, "
            f"oversample_budget={oversample_budget}, scheduled_target={scheduled_target}."
        )

        async def spawn_one() -> asyncio.Task:
            rollout_state = await ctx.sample_group(from_expired_pool=sample_from_expired)
            return create_task(
                ctx.generate_group(
                    rollout_state,
                    enable_partial_rollout=self.enable_partial_rollout,
                )
            )

        initial_available = await ctx.available_count()
        progress_displayer = _ProgressDisplayer.create(
            strategy_name=self.__class__.__name__,
            task_name=ctx.task_name,
            total=ctx.target_abs,
            initial=initial_available,
        )
        try:
            while True:
                if ctx.should_abort():
                    return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
                if self.is_model_expired(ctx.train_step, ctx.model_step):
                    return ProduceBatchStatus.EXPIRED_BATCH

                available = await ctx.available_count()
                progress_displayer.update(available)
                if not self.should_continue_fn(available, target_abs):
                    return ProduceBatchStatus.NORMAL

                pending_count = self._pending_tasks.count()
                desired_pending = max(0, scheduled_target - available)
                if available + pending_count < scheduled_target:
                    while await self._pending_tasks.schedule_one(
                        max_pending=desired_pending,
                        should_abort=ctx.should_abort,
                        spawn_one=spawn_one,
                    ):
                        pass
                    if ctx.should_abort():
                        return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT

                if ctx.should_abort():
                    return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
                if self._pending_tasks.count() == 0:
                    logger.warning("All tasks are done but not enough samples collected.")
                    return ProduceBatchStatus.NORMAL

                claimed_done = await self._pending_tasks.wait_and_claim(timeout_s=1)
                await _put_claimed_tasks(
                    claimed_done,
                    ctx,
                    available_base=available,
                    progress_displayer=progress_displayer,
                )
        finally:
            progress_displayer.close()
