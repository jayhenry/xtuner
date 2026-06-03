"""共卡 / 非共卡生产代码拆分伪代码。

说明：
- 这是设计伪代码，用来展示 Module、Interface 和 Adapter 关系，不是可直接运行实现。
- 重点是把共卡同步生产和非共卡 Background Producer / Training Consumer 分开。
- AsyncProduceStrategyConfig 仍可用于共卡；区别是按模式构建不同的具体 ProduceStrategy Adapter。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol


class Status(Enum):
    INIT = auto()
    COMPLETED = auto()
    ABORTED = auto()
    EXPIRED = auto()
    FAILED = auto()
    FILTERED = auto()


class ProduceBatchStatus(Enum):
    NORMAL = auto()
    UPDATE_WEIGHT_AND_ABORT = auto()
    EXPIRED_BATCH = auto()


class ProducerMode(Enum):
    COLOCATE = auto()
    DISAGGREGATED = auto()


class DisaggregatedManagerStatus(Enum):
    NORMAL = auto()
    UPDATE_WEIGHT_AND_ABORT = auto()
    EXPIRED_BATCH = auto()
    FINISH = auto()


def get_group_status(group: list[Any]) -> Status:
    """聚合 rollout group 状态。

    这里只读状态，不修改样本。过滤和过期翻转必须发生在显式业务逻辑里。
    """

    ...


def calculate_seq_staleness(model_step: int, train_step: int) -> int:
    ...


AGENT_LOOP_PAUSE_REQUEST_TIMEOUT_S = 10.0
PERIODIC_ABORT_INTERVAL_S = 5.0
PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S = 60.0


def calculate_stale_threshold(max_staleness: int, sync_weights_interval: int) -> int:
    return (max_staleness + 1) * sync_weights_interval


@dataclass
class ProduceMetrics:
    raw_rewards_sum: float = 0.0
    raw_rewards_count: int = 0
    produced_samples: int = 0
    produced_tokens: int = 0
    produce_time_s: float = 0.0
    group_generate_times_s: list[float] = field(default_factory=list)
    pause_time_s: float = 0.0

    def add_group_time(self, elapsed_s: float) -> None:
        self.group_generate_times_s.append(elapsed_s)


@dataclass
class ProduceBatchResult:
    rollout_states: list[list[Any]]
    status: ProduceBatchStatus = ProduceBatchStatus.NORMAL
    metrics: ProduceMetrics = field(default_factory=ProduceMetrics)
    leftover_counts: dict[Status, int] = field(default_factory=dict)
    task_batch_sizes: dict[str, int] | None = None
    task_results: dict[str, "ProduceBatchResult"] | None = None


class ReplayBuffer(Protocol):
    async def put(
        self,
        group: list[Any],
        task_name: str,
        *,
        model_step: int | None = None,
        current_train_step: int | None = None,
        stale_threshold: int | None = None,
    ) -> None: ...

    async def count(self, task_name: str, group_status: Status) -> int: ...

    async def refresh_staleness(
        self,
        *,
        task_stale_thresholds: dict[str, int],
        current_train_step: int,
        statuses: list[Status],
    ) -> dict[str, int]: ...

    async def is_ready(self, task_batch_sizes: dict[str, int]) -> bool: ...

    async def take_batch(
        self,
        task_batch_sizes: dict[str, int],
    ) -> tuple[dict[str, list[list[Any]]], dict[str, int]]: ...

    async def count_statuses(
        self,
        task_names: list[str],
        statuses: list[Status],
    ) -> dict[str, dict[Status, int]]: ...

    async def save(self, checkpoint_path: Path) -> None: ...

    async def resume(self, checkpoint_path: Path) -> None: ...


class Sampler(Protocol):
    async def sample(
        self,
        *,
        task_name: str,
        group_status: Status | list[Status] | None = None,
    ) -> list[Any]: ...

    def save(self, checkpoint_path: Path) -> None: ...

    def resume(self, checkpoint_path: Path) -> None: ...


class AgentLoop(Protocol):
    async def generate_group(
        self,
        group: list[Any],
        *,
        enable_partial_rollout: bool = False,
    ) -> list[Any]: ...

    async def pause(self) -> None: ...


class RolloutController(Protocol):
    async def continue_generation(self) -> None: ...

    async def pause_generation(self) -> None: ...


class ShouldContinueFn(Protocol):
    def __call__(self, completed_count: int, batch_size: int, **kwargs: Any) -> bool: ...


class IsValidSampleFn(Protocol):
    def __call__(self, samples: list[Any]) -> bool: ...


def default_should_continue_fn(completed_count: int, batch_size: int, **kwargs: Any) -> bool:
    return completed_count < batch_size


def default_is_valid_sample_fn(samples: list[Any]) -> bool:
    return True


@dataclass
class ColocateProduceProgress:
    """共卡单次 produce_batch 的局部进度。

    中文不变量：
    - 只表达本次调用，不进入 checkpoint。
    - pending task 由具体 strategy 在本次调用内持有。
    - 不维护 producer_future_step / consumed_samples 等非共卡绝对累计状态。
    """

    task_batch_sizes: dict[str, int]
    train_step: int
    model_step: int
    metrics_by_task: dict[str, ProduceMetrics] = field(default_factory=dict)

    def target_for(self, task_name: str) -> int:
        return self.task_batch_sizes[task_name]

    def metrics_for(self, task_name: str) -> ProduceMetrics:
        return self.metrics_by_task.setdefault(task_name, ProduceMetrics())


@dataclass
class BackgroundProduceProgress:
    """非共卡 Background Producer / Training Consumer 共享进度。

    中文不变量：
    - target_samples / consumed_samples 使用绝对累计口径。
    - consumer 从 replay buffer 取走样本后只增加 consumed，不回退 target。
    - producer_future_step 只由后台 producer 正常完成生产后推进。
    - 该对象会进入 checkpoint/resume。
    """

    task_names: list[str]
    producer_future_step: int = 1
    next_consumer_step: int = 1
    target_upto_future_step: int = 0
    consumed_samples: dict[str, int] = field(default_factory=dict)
    target_samples: dict[str, int] = field(default_factory=dict)
    metrics_by_task: dict[str, ProduceMetrics] = field(default_factory=dict)

    @classmethod
    def build(cls, task_names: list[str]) -> "BackgroundProduceProgress":
        return cls(
            task_names=task_names,
            consumed_samples={name: 0 for name in task_names},
            target_samples={name: 0 for name in task_names},
        )

    def ensure_target_upto(
        self,
        *,
        batch_size: int,
        future_step: int,
        allocate_batch_sizes: Callable[[int, int], dict[str, int]],
    ) -> dict[str, int]:
        if future_step > self.target_upto_future_step:
            for step in range(self.target_upto_future_step + 1, future_step + 1):
                task_sizes = allocate_batch_sizes(batch_size, step)
                for task_name, task_size in task_sizes.items():
                    self.target_samples[task_name] += task_size
            self.target_upto_future_step = future_step
        return allocate_batch_sizes(batch_size, future_step)

    def begin_consume(self, train_step: int) -> None:
        self.next_consumer_step = train_step

    def mark_consumed(self, consumed_counts: dict[str, int]) -> None:
        for task_name, count in consumed_counts.items():
            self.consumed_samples[task_name] += count

    def finish_consume(self, train_step: int) -> None:
        self.next_consumer_step = train_step + 1

    def advance_future_step(self) -> None:
        self.producer_future_step += 1

    def metrics_for(self, task_name: str) -> ProduceMetrics:
        return self.metrics_by_task.setdefault(task_name, ProduceMetrics())

    def state_dict(self) -> dict[str, Any]:
        return {
            "producer_future_step": self.producer_future_step,
            "next_consumer_step": self.next_consumer_step,
            "target_upto_future_step": self.target_upto_future_step,
            "consumed_samples": dict(self.consumed_samples),
            "target_samples": dict(self.target_samples),
            "metrics_by_task": self.metrics_by_task,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.producer_future_step = state["producer_future_step"]
        self.next_consumer_step = state["next_consumer_step"]
        self.target_upto_future_step = state["target_upto_future_step"]
        self.consumed_samples.clear()
        self.consumed_samples.update(state["consumed_samples"])
        self.target_samples.clear()
        self.target_samples.update(state["target_samples"])
        self.metrics_by_task = state.get("metrics_by_task", {})


@dataclass
class ProduceContext:
    """strategy 生产一个 task 时看到的公共上下文。

    共卡和非共卡共享生成、采样、入库能力；是否允许 abort、target 如何计算由子类表达。
    """

    task_name: str
    agent_loop: AgentLoop
    sampler: Sampler
    replay_buffer: ReplayBuffer
    train_step: int
    model_step: int
    is_valid_sample_fn: IsValidSampleFn
    stale_threshold: int | None

    @property
    def current_train_step_for_staleness(self) -> int:
        return self.train_step

    def should_abort(self) -> bool:
        return False

    async def generate_group(
        self,
        group: list[Any],
        *,
        enable_partial_rollout: bool,
    ) -> tuple[list[Any], float]:
        start = time.perf_counter()
        result = await self.agent_loop.generate_group(
            group,
            enable_partial_rollout=enable_partial_rollout,
        )
        return result, time.perf_counter() - start

    async def put_generated_group(self, group: list[Any], metrics: ProduceMetrics) -> bool:
        """统一处理生成结果过滤、统计和入库。

        中文设计点：
        - 只有 completed group 才执行业务过滤。
        - ReplayBuffer.put 负责写 model_step、刷新 staleness、按阈值转 expired。
        - put 之后重新判断 group 状态，因为 completed 可能在入库前被转成 expired。
        """

        if get_group_status(group) == Status.COMPLETED:
            if not self.is_valid_sample_fn(group):
                for item in group:
                    item.status = Status.FILTERED

        await self.replay_buffer.put(
            group,
            self.task_name,
            model_step=self.model_step,
            current_train_step=self.current_train_step_for_staleness,
            stale_threshold=self.stale_threshold,
        )
        metrics.produced_samples += len(group)
        return get_group_status(group) == Status.COMPLETED


@dataclass
class BackgroundProduceContext(ProduceContext):
    update_event: asyncio.Event
    progress: BackgroundProduceProgress

    @property
    def current_train_step_for_staleness(self) -> int:
        return self.progress.next_consumer_step

    def should_abort(self) -> bool:
        return self.update_event.is_set()

    async def available_count(self) -> int:
        completed = await self.replay_buffer.count(self.task_name, Status.COMPLETED)
        return self.progress.consumed_samples[self.task_name] + completed

    @property
    def target_abs(self) -> int:
        return self.progress.target_samples[self.task_name]


class ProduceStrategy(Protocol):
    async def produce_batch(self, ctx: ProduceContext, progress: Any) -> ProduceBatchStatus: ...

    async def pause_produce(self, ctx: ProduceContext, progress: Any) -> float:
        return 0.0

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        return False

    def pending_task_count(self) -> int:
        return 0


class ProduceStrategyConfig(Protocol):
    def build(
        self,
        *,
        mode: ProducerMode,
        sync_weights_interval: int,
        rollout_controller: RolloutController,
    ) -> ProduceStrategy: ...


@dataclass
class SyncProduceStrategyConfig:
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    should_continue_fn: ShouldContinueFn = default_should_continue_fn

    def build(
        self,
        *,
        mode: ProducerMode,
        sync_weights_interval: int,
        rollout_controller: RolloutController,
    ) -> ProduceStrategy:
        return SyncProduceStrategy(
            is_valid_sample_fn=self.is_valid_sample_fn,
            should_continue_fn=self.should_continue_fn,
        )


@dataclass
class AsyncProduceStrategyConfig:
    over_sample_threshold: float = 0.0
    enable_partial_rollout: bool = False
    max_staleness: int = 0
    tail_batch_trigger_size: int = 0
    is_valid_sample_fn: IsValidSampleFn = default_is_valid_sample_fn
    should_continue_fn: ShouldContinueFn = default_should_continue_fn

    def build(
        self,
        *,
        mode: ProducerMode,
        sync_weights_interval: int,
        rollout_controller: RolloutController,
    ) -> ProduceStrategy:
        # 同一个 AsyncProduceStrategyConfig 按执行模式构建不同 Adapter。
        if mode == ProducerMode.COLOCATE:
            return ColocateAsyncProduceStrategy(
                over_sample_threshold=self.over_sample_threshold,
                enable_partial_rollout=self.enable_partial_rollout,
                max_staleness=self.max_staleness,
                sync_weights_interval=sync_weights_interval,
                tail_batch_trigger_size=self.tail_batch_trigger_size,
                is_valid_sample_fn=self.is_valid_sample_fn,
                should_continue_fn=self.should_continue_fn,
            )

        return BackgroundAsyncProduceStrategy(
            over_sample_threshold=self.over_sample_threshold,
            enable_partial_rollout=self.enable_partial_rollout,
            max_staleness=self.max_staleness,
            sync_weights_interval=sync_weights_interval,
            tail_batch_trigger_size=self.tail_batch_trigger_size,
            is_valid_sample_fn=self.is_valid_sample_fn,
            should_continue_fn=self.should_continue_fn,
        )


class SyncProduceStrategy:
    def __init__(
        self,
        *,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ) -> None:
        self.is_valid_sample_fn = is_valid_sample_fn
        self.should_continue_fn = should_continue_fn

    async def produce_batch(self, ctx: ProduceContext, progress: ColocateProduceProgress) -> ProduceBatchStatus:
        pending: set[asyncio.Task] = set()
        metrics = progress.metrics_for(ctx.task_name)
        target = progress.target_for(ctx.task_name)

        for _ in range(target):
            group = await ctx.sampler.sample(task_name=ctx.task_name)
            pending.add(asyncio.create_task(ctx.generate_group(group, enable_partial_rollout=False)))

        completed = 0
        while self.should_continue_fn(completed, target):
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                group, elapsed_s = task.result()
                metrics.add_group_time(elapsed_s)
                if await ctx.put_generated_group(group, metrics):
                    completed += 1

        return ProduceBatchStatus.NORMAL


async def sample_retry_or_new_group(ctx: ProduceContext, *, from_expired_pool: bool) -> list[Any]:
    statuses = [Status.EXPIRED, Status.ABORTED] if from_expired_pool else [Status.ABORTED]
    return await ctx.sampler.sample(task_name=ctx.task_name, group_status=statuses)


def is_model_expired_for_threshold(*, stale_threshold: int, train_step: int, model_step: int) -> bool:
    return calculate_seq_staleness(model_step, train_step) >= stale_threshold


async def put_finished_task_result(task: asyncio.Task, ctx: ProduceContext, metrics: ProduceMetrics) -> bool:
    group, elapsed_s = task.result()
    metrics.add_group_time(elapsed_s)
    return await ctx.put_generated_group(group, metrics)


class ColocateAsyncProduceStrategy:
    def __init__(
        self,
        *,
        over_sample_threshold: float,
        enable_partial_rollout: bool,
        max_staleness: int,
        sync_weights_interval: int,
        tail_batch_trigger_size: int,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ) -> None:
        self.over_sample_threshold = over_sample_threshold
        self.enable_partial_rollout = enable_partial_rollout
        self.max_staleness = max_staleness
        self.sync_weights_interval = sync_weights_interval
        self.tail_batch_trigger_size = tail_batch_trigger_size
        self.is_valid_sample_fn = is_valid_sample_fn
        self.should_continue_fn = should_continue_fn
        self.stale_threshold = calculate_stale_threshold(max_staleness, sync_weights_interval)

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        return is_model_expired_for_threshold(
            stale_threshold=self.stale_threshold,
            train_step=train_step,
            model_step=model_step,
        )

    async def produce_batch(
        self,
        ctx: ProduceContext,
        progress: ColocateProduceProgress,
    ) -> ProduceBatchStatus:
        """共卡 async 生产。

        中文不变量：
        - pending 是本次调用局部变量。
        - 本函数结束前 drain pending。
        - 不读取 update_event，不返回 UPDATE_WEIGHT_AND_ABORT。
        """

        pending: set[asyncio.Task] = set()
        metrics = progress.metrics_for(ctx.task_name)
        target = progress.target_for(ctx.task_name)
        scheduled_target = target + int(target * self.over_sample_threshold)
        completed = await ctx.replay_buffer.count(ctx.task_name, Status.COMPLETED)

        async def schedule_one() -> None:
            group = await sample_retry_or_new_group(ctx, from_expired_pool=False)
            pending.add(
                asyncio.create_task(
                    ctx.generate_group(
                        group,
                        enable_partial_rollout=self.enable_partial_rollout,
                    )
                )
            )

        while len(pending) + completed < scheduled_target:
            await schedule_one()

        try:
            while self.should_continue_fn(completed, target):
                if not pending:
                    break
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if await put_finished_task_result(task, ctx, metrics):
                        completed += 1

                while len(pending) + completed < scheduled_target and self.should_continue_fn(completed, target):
                    await schedule_one()
        finally:
            metrics.pause_time_s += await self._cleanup_local_pending(pending, ctx, metrics)

        return ProduceBatchStatus.NORMAL

    async def _cleanup_local_pending(
        self,
        pending: set[asyncio.Task],
        ctx: ProduceContext,
        metrics: ProduceMetrics,
    ) -> float:
        return await pause_pending_tasks(
            pending_tasks=_LocalPendingTasks(pending),
            ctx=ctx,
            put_claimed_task=lambda task: put_finished_task_result(task, ctx, metrics),
        )


class _LocalPendingTasks:
    """把共卡本次调用的局部 set 包装成 pause helper 可使用的形状。"""

    def __init__(self, tasks: set[asyncio.Task]) -> None:
        self._tasks = tasks

    def count(self) -> int:
        return len(self._tasks)

    async def wait_and_claim(self, timeout_s: float) -> set[asyncio.Task]:
        if not self._tasks:
            return set()
        done, _ = await asyncio.wait(self._tasks, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
        self._tasks.difference_update(done)
        return done

    async def cancel_all(self) -> int:
        tasks = set(self._tasks)
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        return len(tasks)


class _PendingTasks:
    """非共卡专用 pending 集合。

    共卡不使用它，因为共卡 pending 不跨 produce_batch 调用。
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()
        self._lock = asyncio.Lock()

    def count(self) -> int:
        return len(self._tasks)

    async def claim_ready(self) -> set[asyncio.Task]:
        async with self._lock:
            ready = {task for task in self._tasks if task.done()}
            self._tasks.difference_update(ready)
            return ready

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
            self._tasks.add(await spawn_one())
            return True

    async def wait_and_claim(self, timeout_s: float) -> set[asyncio.Task]:
        async with self._lock:
            snapshot = set(self._tasks)
        if not snapshot:
            return set()
        done, _ = await asyncio.wait(snapshot, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
        async with self._lock:
            claimed = done & self._tasks
            self._tasks.difference_update(claimed)
            return claimed

    async def cancel_all(self) -> int:
        async with self._lock:
            tasks = set(self._tasks)
            self._tasks.clear()
        for task in tasks:
            task.cancel()
        return len(tasks)


async def request_agent_loop_pause(ctx: ProduceContext, *, pending_count: int) -> None:
    """发送一次 agent loop pause 请求。

    最新生产代码里 pause_produce 会周期性调用 agent_loop.pause()，这里把这段协议抽成全局工具函数，
    让共卡本地 pending 收尾和非共卡后台 pending drain 使用同一套超时/日志语义。
    """

    try:
        await asyncio.wait_for(ctx.agent_loop.pause(), timeout=AGENT_LOOP_PAUSE_REQUEST_TIMEOUT_S)
    except asyncio.TimeoutError:
        # 真实实现写 logger.warning，伪代码只保留关键上下文。
        print(
            f"Agent loop pause timed out: task={ctx.task_name}, "
            f"timeout_s={AGENT_LOOP_PAUSE_REQUEST_TIMEOUT_S}, pending={pending_count}"
        )
    except Exception:
        print(f"Agent loop pause failed: task={ctx.task_name}, pending={pending_count}")


async def pause_pending_tasks(
    *,
    pending_tasks: _LocalPendingTasks | _PendingTasks,
    ctx: ProduceContext,
    put_claimed_task: Callable[[asyncio.Task], Awaitable[Any]],
) -> float:
    """复用当前 pause_produce 的 pending drain 协议。

    中文不变量：
    - 先发 pause，再等待 pending 产出。
    - pending 没清空时周期性补发 pause，兼容后端 abort 信号丢失或延迟。
    - 超时后 cancel 剩余 pending，避免 checkpoint/save 前仍有任务写 buffer。
    - 已完成任务必须 claim 后再 put，避免 produce 和 pause 重复入库同一个 done task。
    """

    pause_start = time.perf_counter()
    if pending_tasks.count() == 0:
        return 0.0

    pending_pause_tasks = {
        asyncio.create_task(request_agent_loop_pause(ctx, pending_count=pending_tasks.count()))
    }
    cleanup_start_time = time.perf_counter()
    next_periodic_abort_time = cleanup_start_time + PERIODIC_ABORT_INTERVAL_S

    while True:
        elapsed_time = time.perf_counter() - cleanup_start_time
        if elapsed_time > PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S:
            cancelled_count = await pending_tasks.cancel_all()
            print(
                f"Cleanup timeout reached. Forcefully cancelling {cancelled_count} "
                f"remaining tasks for task={ctx.task_name}."
            )
            break

        if pending_tasks.count() == 0:
            break

        current_time = time.perf_counter()
        pending_pause_tasks = {task for task in pending_pause_tasks if not task.done()}
        if PERIODIC_ABORT_INTERVAL_S > 0 and current_time >= next_periodic_abort_time:
            pending_pause_tasks.add(
                asyncio.create_task(request_agent_loop_pause(ctx, pending_count=pending_tasks.count()))
            )
            next_periodic_abort_time += PERIODIC_ABORT_INTERVAL_S

        claimed_done = await pending_tasks.wait_and_claim(timeout_s=1.0)
        for task in claimed_done:
            await put_claimed_task(task)

    for task in pending_pause_tasks:
        task.cancel()
    if pending_pause_tasks:
        await asyncio.gather(*pending_pause_tasks, return_exceptions=True)

    return time.perf_counter() - pause_start


class BackgroundAsyncProduceStrategy:
    def __init__(
        self,
        *,
        over_sample_threshold: float,
        enable_partial_rollout: bool,
        max_staleness: int,
        sync_weights_interval: int,
        tail_batch_trigger_size: int,
        is_valid_sample_fn: IsValidSampleFn,
        should_continue_fn: ShouldContinueFn,
    ) -> None:
        self.over_sample_threshold = over_sample_threshold
        self.enable_partial_rollout = enable_partial_rollout
        self.max_staleness = max_staleness
        self.sync_weights_interval = sync_weights_interval
        self.tail_batch_trigger_size = tail_batch_trigger_size
        self.is_valid_sample_fn = is_valid_sample_fn
        self.should_continue_fn = should_continue_fn
        self.stale_threshold = calculate_stale_threshold(max_staleness, sync_weights_interval)
        self._pending_tasks = _PendingTasks()

    def is_model_expired(self, train_step: int, model_step: int) -> bool:
        return is_model_expired_for_threshold(
            stale_threshold=self.stale_threshold,
            train_step=train_step,
            model_step=model_step,
        )

    def pending_task_count(self) -> int:
        return self._pending_tasks.count()

    async def produce_batch(
        self,
        ctx: BackgroundProduceContext,
        progress: BackgroundProduceProgress,
    ) -> ProduceBatchStatus:
        """非共卡后台 async 生产。

        中文不变量：
        - pending 可以跨多次 produce_batch 调用存在。
        - 每轮循环都观察 update_event 和 model expired。
        - 只负责生产到 replay buffer，不取训练 batch。
        """

        if ctx.should_abort():
            return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
        if self.is_model_expired(progress.producer_future_step, ctx.model_step):
            return ProduceBatchStatus.EXPIRED_BATCH

        metrics = progress.metrics_for(ctx.task_name)
        await self._put_claimed(await self._pending_tasks.claim_ready(), ctx, metrics)

        target_abs = ctx.target_abs
        oversample_budget = int(ctx.train_step * self.over_sample_threshold)
        scheduled_target = target_abs + oversample_budget

        async def spawn_one() -> asyncio.Task:
            group = await sample_retry_or_new_group(ctx, from_expired_pool=False)
            return asyncio.create_task(
                ctx.generate_group(
                    group,
                    enable_partial_rollout=self.enable_partial_rollout,
                )
            )

        while True:
            if ctx.should_abort():
                return ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT
            if self.is_model_expired(progress.producer_future_step, ctx.model_step):
                return ProduceBatchStatus.EXPIRED_BATCH

            available = await ctx.available_count()
            if not self.should_continue_fn(available, target_abs):
                return ProduceBatchStatus.NORMAL

            desired_pending = max(0, scheduled_target - available)
            while await self._pending_tasks.schedule_one(
                max_pending=desired_pending,
                should_abort=ctx.should_abort,
                spawn_one=spawn_one,
            ):
                pass

            claimed = await self._pending_tasks.wait_and_claim(timeout_s=1.0)
            await self._put_claimed(claimed, ctx, metrics)

    async def pause_produce(
        self,
        ctx: BackgroundProduceContext,
        progress: BackgroundProduceProgress,
    ) -> float:
        metrics = progress.metrics_for(ctx.task_name)
        return await pause_pending_tasks(
            pending_tasks=self._pending_tasks,
            ctx=ctx,
            put_claimed_task=lambda task: put_finished_task_result(task, ctx, metrics),
        )

    async def _put_claimed(
        self,
        claimed: set[asyncio.Task],
        ctx: ProduceContext,
        metrics: ProduceMetrics,
    ) -> None:
        for task in claimed:
            await put_finished_task_result(task, ctx, metrics)


@dataclass(frozen=True)
class TaskRunner:
    task_name: str
    agent_loop: AgentLoop
    sampler: Sampler
    produce_strategy: ProduceStrategy
    weight: float = 1.0
    order: int = 0

    @property
    def stale_threshold(self) -> int | None:
        return getattr(self.produce_strategy, "stale_threshold", None)


class AgentLoopManagerConfig:
    def __init__(self, tasks: list[Any]) -> None:
        self.tasks = tasks

    def build_colocate(
        self,
        *,
        rollout_controller: RolloutController,
        tokenizer: Any,
        replay_buffer: ReplayBuffer,
        logger: Any,
        sync_weights_interval: int,
    ) -> "ColocateAgentLoopManager":
        runners = self._build_task_runners(
            mode=ProducerMode.COLOCATE,
            rollout_controller=rollout_controller,
            tokenizer=tokenizer,
            replay_buffer=replay_buffer,
            logger=logger,
            sync_weights_interval=sync_weights_interval,
        )
        return ColocateAgentLoopManager(runners, replay_buffer, rollout_controller, logger)

    def build_disaggregated(
        self,
        *,
        rollout_controller: RolloutController,
        tokenizer: Any,
        replay_buffer: ReplayBuffer,
        logger: Any,
        sync_weights_interval: int,
    ) -> "DisaggregatedAgentLoopManager":
        runners = self._build_task_runners(
            mode=ProducerMode.DISAGGREGATED,
            rollout_controller=rollout_controller,
            tokenizer=tokenizer,
            replay_buffer=replay_buffer,
            logger=logger,
            sync_weights_interval=sync_weights_interval,
        )
        return DisaggregatedAgentLoopManager(runners, replay_buffer, rollout_controller, logger)

    def _build_task_runners(
        self,
        *,
        mode: ProducerMode,
        rollout_controller: RolloutController,
        tokenizer: Any,
        replay_buffer: ReplayBuffer,
        logger: Any,
        sync_weights_interval: int,
    ) -> list[TaskRunner]:
        runners: list[TaskRunner] = []
        for task_cfg in self.tasks:
            strategy = task_cfg.produce_strategy_config.build(
                mode=mode,
                sync_weights_interval=sync_weights_interval,
                rollout_controller=rollout_controller,
            )
            runners.append(
                TaskRunner(
                    task_name=task_cfg.task_name,
                    agent_loop=task_cfg.agent_loop_config.build(rollout_controller, logger),
                    sampler=task_cfg.sampler_config.build(tokenizer, replay_buffer),
                    produce_strategy=strategy,
                    weight=task_cfg.weight,
                    order=len(runners),
                )
            )
        return runners


def task_names_of(task_runners: list[TaskRunner]) -> list[str]:
    return [task.task_name for task in task_runners]


def allocate_task_batch_sizes(
    task_runners: list[TaskRunner],
    global_batch_size: int,
    train_step: int,
) -> dict[str, int]:
    # 真实实现沿用当前按 task weight 分配的逻辑；保持为全局 helper，避免两个 manager 继承公共父类。
    ...


def validate_task_batch_sizes(
    task_runners: list[TaskRunner],
    task_sizes: dict[str, int],
    global_batch_size: int,
) -> None:
    ...


async def refresh_for_all_tasks(
    *,
    task_runners: list[TaskRunner],
    replay_buffer: ReplayBuffer,
    train_step: int,
) -> None:
    thresholds = {
        task.task_name: task.stale_threshold or 1
        for task in task_runners
    }
    await replay_buffer.refresh_staleness(
        task_stale_thresholds=thresholds,
        current_train_step=train_step,
        statuses=[Status.COMPLETED, Status.ABORTED],
    )


async def take_train_batch(
    *,
    task_runners: list[TaskRunner],
    replay_buffer: ReplayBuffer,
    task_sizes: dict[str, int],
    progress: ColocateProduceProgress | BackgroundProduceProgress,
) -> ProduceBatchResult:
    batch_by_task, consumed_counts = await replay_buffer.take_batch(task_sizes)
    if isinstance(progress, BackgroundProduceProgress):
        progress.mark_consumed(consumed_counts)

    counts = await replay_buffer.count_statuses(
        task_names_of(task_runners),
        [Status.INIT, Status.COMPLETED, Status.ABORTED, Status.EXPIRED, Status.FAILED, Status.FILTERED],
    )
    return build_produce_batch_result(
        task_runners=task_runners,
        batch_by_task=batch_by_task,
        leftover_counts=counts,
        progress=progress,
    )


def build_produce_batch_result(
    *,
    task_runners: list[TaskRunner],
    batch_by_task: dict[str, list[list[Any]]],
    leftover_counts: dict[str, dict[Status, int]],
    progress: ColocateProduceProgress | BackgroundProduceProgress,
) -> ProduceBatchResult:
    # 真实实现负责 task result 聚合、timing 聚合、leftover 聚合。
    ...


class ColocateAgentLoopManager:
    def __init__(
        self,
        task_runners: list[TaskRunner],
        replay_buffer: ReplayBuffer,
        rollout_controller: RolloutController,
        logger: Any,
    ) -> None:
        self.task_runners = task_runners
        self.replay_buffer = replay_buffer
        self.rollout_controller = rollout_controller
        self.logger = logger
        self.task_names = task_names_of(task_runners)

    def get_task_batch_sizes(self, global_batch_size: int, train_step: int) -> dict[str, int]:
        return allocate_task_batch_sizes(self.task_runners, global_batch_size, train_step)

    async def produce_batch(
        self,
        batch_size: int,
        train_step: int,
        *,
        model_step: int,
    ) -> ProduceBatchResult:
        """共卡训练唯一生产入口。

        中文不变量：
        - 不触碰非共卡 status/update_event。
        - Async strategy 也必须在本次调用内收尾 pending。
        - 返回必须是非空训练 batch。
        """

        task_sizes = self.get_task_batch_sizes(batch_size, train_step)
        validate_task_batch_sizes(self.task_runners, task_sizes, batch_size)
        progress = ColocateProduceProgress(task_sizes, train_step, model_step)

        await self.rollout_controller.continue_generation()
        try:
            await refresh_for_all_tasks(
                task_runners=self.task_runners,
                replay_buffer=self.replay_buffer,
                train_step=train_step,
            )
            await asyncio.gather(
                *[
                    self._produce_colocate_task(task, task_sizes[task.task_name], progress)
                    for task in self.task_runners
                    if task_sizes[task.task_name] > 0
                ]
            )
            result = await take_train_batch(
                task_runners=self.task_runners,
                replay_buffer=self.replay_buffer,
                task_sizes=task_sizes,
                progress=progress,
            )
        finally:
            await self.rollout_controller.pause_generation()

        assert result.rollout_states, "共卡 produce_batch 必须返回非空训练 batch。"
        return result

    async def _produce_colocate_task(
        self,
        task: TaskRunner,
        task_batch_size: int,
        progress: ColocateProduceProgress,
    ) -> ProduceBatchStatus:
        ctx = ProduceContext(
            task_name=task.task_name,
            agent_loop=task.agent_loop,
            sampler=task.sampler,
            replay_buffer=self.replay_buffer,
            train_step=progress.train_step,
            model_step=progress.model_step,
            is_valid_sample_fn=getattr(task.produce_strategy, "is_valid_sample_fn", default_is_valid_sample_fn),
            stale_threshold=task.stale_threshold,
        )
        return await task.produce_strategy.produce_batch(ctx, progress)

    async def save(self, checkpoint_path: Path, model_step: int) -> None:
        # 共卡 checkpoint 不保存 BackgroundProduceProgress。
        for task in self.task_runners:
            task.sampler.save(checkpoint_path / "tasks" / task.task_name)
        await self.replay_buffer.save(checkpoint_path)

    async def resume(self, checkpoint_path: Path) -> int:
        for task in self.task_runners:
            task.sampler.resume(checkpoint_path / "tasks" / task.task_name)
        await self.replay_buffer.resume(checkpoint_path)
        return 0


class DisaggregatedAgentLoopManager:
    def __init__(
        self,
        task_runners: list[TaskRunner],
        replay_buffer: ReplayBuffer,
        rollout_controller: RolloutController,
        logger: Any,
    ) -> None:
        self.task_runners = task_runners
        self.replay_buffer = replay_buffer
        self.rollout_controller = rollout_controller
        self.logger = logger
        self.task_names = task_names_of(task_runners)
        self.status = DisaggregatedManagerStatus.NORMAL
        self.update_event = asyncio.Event()
        self.finish_event = asyncio.Event()
        self.model_step = 0
        self.pause_time_s = 0.0
        self.progress = BackgroundProduceProgress.build(self.task_names)

    def get_task_batch_sizes(self, global_batch_size: int, train_step: int) -> dict[str, int]:
        return allocate_task_batch_sizes(self.task_runners, global_batch_size, train_step)

    async def produce_loop(self, batch_size: int) -> None:
        """非共卡 Background Producer。"""

        while not self.finish_event.is_set():
            if self.status == DisaggregatedManagerStatus.FINISH:
                break
            if self.status in (
                DisaggregatedManagerStatus.UPDATE_WEIGHT_AND_ABORT,
                DisaggregatedManagerStatus.EXPIRED_BATCH,
            ):
                await self._wait_for_status_exit(self.status)
                continue

            task_sizes = self.progress.ensure_target_upto(
                batch_size=batch_size,
                future_step=self.progress.producer_future_step,
                allocate_batch_sizes=self.get_task_batch_sizes,
            )
            validate_task_batch_sizes(self.task_runners, task_sizes, batch_size)
            statuses = await asyncio.gather(
                *[
                    self._produce_background_task(task, task_sizes[task.task_name])
                    for task in self.task_runners
                    if task_sizes[task.task_name] > 0
                ]
            )

            if ProduceBatchStatus.EXPIRED_BATCH in statuses:
                self.status = DisaggregatedManagerStatus.EXPIRED_BATCH
            elif ProduceBatchStatus.UPDATE_WEIGHT_AND_ABORT in statuses:
                self.status = DisaggregatedManagerStatus.UPDATE_WEIGHT_AND_ABORT
            else:
                self.progress.advance_future_step()

            await asyncio.sleep(0)

    async def get_batch(self, batch_size: int, train_step: int) -> ProduceBatchResult:
        """非共卡 Training Consumer。"""

        self.progress.begin_consume(train_step)
        await refresh_for_all_tasks(
            task_runners=self.task_runners,
            replay_buffer=self.replay_buffer,
            train_step=train_step,
        )
        task_sizes = self.get_task_batch_sizes(batch_size, train_step)
        validate_task_batch_sizes(self.task_runners, task_sizes, batch_size)
        current_model_step = train_step - 1

        while not self.finish_event.is_set():
            if self.status == DisaggregatedManagerStatus.EXPIRED_BATCH:
                if current_model_step > self.model_step:
                    return ProduceBatchResult([], status=ProduceBatchStatus.EXPIRED_BATCH)
                if not await self.replay_buffer.is_ready(task_sizes):
                    raise RuntimeError("Expired Produce Batch 不能跳过，且当前训练 batch 未 ready。")

            if await self.replay_buffer.is_ready(task_sizes):
                result = await take_train_batch(
                    task_runners=self.task_runners,
                    replay_buffer=self.replay_buffer,
                    task_sizes=task_sizes,
                    progress=self.progress,
                )
                if self.status == DisaggregatedManagerStatus.EXPIRED_BATCH:
                    result.status = ProduceBatchStatus.EXPIRED_BATCH
                if result.rollout_states:
                    self.progress.finish_consume(train_step)
                    await refresh_for_all_tasks(
                        task_runners=self.task_runners,
                        replay_buffer=self.replay_buffer,
                        train_step=train_step + 1,
                    )
                    return result

            await asyncio.sleep(1.0)

        return ProduceBatchResult([])

    async def pause_produce(self) -> float:
        """非共卡权重同步前的显式暂停入口。"""

        self.update_event.set()
        self.status = DisaggregatedManagerStatus.UPDATE_WEIGHT_AND_ABORT
        await self.rollout_controller.pause_generation()

        pause_time_s = 0.0
        for task in self.task_runners:
            ctx = self._build_background_context(task, task_batch_size=0)
            pause_time_s += await task.produce_strategy.pause_produce(ctx, self.progress)
        self.pause_time_s = pause_time_s
        return pause_time_s

    async def continue_produce(self, model_step: int) -> None:
        self.model_step = model_step
        await self.rollout_controller.continue_generation()
        self.status = DisaggregatedManagerStatus.NORMAL
        self.update_event.clear()

    def shutdown(self) -> None:
        self.status = DisaggregatedManagerStatus.FINISH
        self.update_event.set()
        self.finish_event.set()

    async def save(self, checkpoint_path: Path, model_step: int) -> None:
        pending = {
            task.task_name: task.produce_strategy.pending_task_count()
            for task in self.task_runners
            if task.produce_strategy.pending_task_count() > 0
        }
        if pending:
            raise RuntimeError(f"保存 checkpoint 前必须先 pause producer: {pending}")

        for task in self.task_runners:
            task.sampler.save(checkpoint_path / "tasks" / task.task_name)
        await self.replay_buffer.save(checkpoint_path)
        self._save_manager_state(checkpoint_path, model_step)

    async def resume(self, checkpoint_path: Path) -> int:
        for task in self.task_runners:
            task.sampler.resume(checkpoint_path / "tasks" / task.task_name)
        await self.replay_buffer.resume(checkpoint_path)
        saved_model_step = self._load_manager_state(checkpoint_path)

        self.update_event = asyncio.Event()
        self.finish_event = asyncio.Event()
        self.update_event.set()
        self.status = DisaggregatedManagerStatus.UPDATE_WEIGHT_AND_ABORT
        self.model_step = saved_model_step
        return saved_model_step

    async def _produce_background_task(self, task: TaskRunner, task_batch_size: int) -> ProduceBatchStatus:
        ctx = self._build_background_context(task, task_batch_size)
        return await task.produce_strategy.produce_batch(ctx, self.progress)

    def _build_background_context(self, task: TaskRunner, task_batch_size: int) -> BackgroundProduceContext:
        return BackgroundProduceContext(
            task_name=task.task_name,
            agent_loop=task.agent_loop,
            sampler=task.sampler,
            replay_buffer=self.replay_buffer,
            train_step=self.progress.producer_future_step,
            model_step=self.model_step,
            is_valid_sample_fn=getattr(task.produce_strategy, "is_valid_sample_fn", default_is_valid_sample_fn),
            stale_threshold=task.stale_threshold,
            update_event=self.update_event,
            progress=self.progress,
        )

    async def _wait_for_status_exit(self, blocked_status: DisaggregatedManagerStatus) -> None:
        while not self.finish_event.is_set() and self.status == blocked_status:
            await asyncio.sleep(1.0)

    def _save_manager_state(self, checkpoint_path: Path, model_step: int) -> None:
        ...

    def _load_manager_state(self, checkpoint_path: Path) -> int:
        ...


class RLColocateTrainer:
    def __init__(self, cfg: Any) -> None:
        self.agent_loop_manager = cfg.agent_loop_manager_cfg.build_colocate(
            rollout_controller=cfg.rollout_controller,
            tokenizer=cfg.tokenizer,
            replay_buffer=cfg.replay_buffer_config.build(),
            logger=cfg.logger,
            sync_weights_interval=cfg.sync_weights_interval,
        )
        if cfg.eval_agent_loop_manager_cfg is not None:
            self.eval_agent_loop_manager = cfg.eval_agent_loop_manager_cfg.build_colocate(...)

    def fit(self) -> None:
        for train_step in range(1, self.total_train_steps + 1):
            produce_result = asyncio.run(
                self.agent_loop_manager.produce_batch(
                    self.train_batch_size,
                    train_step=train_step,
                    model_step=self._current_rollout_model_step(train_step),
                )
            )
            self._train_one_batch(produce_result.rollout_states, train_step)


class RLDisaggregatedTrainer:
    def __init__(self, cfg: Any) -> None:
        train_replay_buffer = cfg.replay_buffer_config.build()
        self.agent_loop_manager = cfg.agent_loop_manager_cfg.build_disaggregated(
            rollout_controller=cfg.rollout_controller,
            tokenizer=cfg.tokenizer,
            replay_buffer=train_replay_buffer,
            logger=cfg.logger,
            sync_weights_interval=cfg.sync_weights_interval,
        )
        # eval 是一次性同步 produce_batch，不应构建成后台 manager。
        if cfg.eval_agent_loop_manager_cfg is not None:
            self.eval_agent_loop_manager = cfg.eval_agent_loop_manager_cfg.build_colocate(
                rollout_controller=cfg.rollout_controller,
                tokenizer=cfg.tokenizer,
                replay_buffer=train_replay_buffer,
                logger=cfg.logger,
                sync_weights_interval=cfg.sync_weights_interval,
            )

    async def _fit(self) -> None:
        producer_task = asyncio.create_task(
            self.agent_loop_manager.produce_loop(batch_size=self.train_batch_size)
        )
        try:
            train_step = self.cur_step + 1
            while train_step <= self.total_train_steps:
                produce_result = await self.agent_loop_manager.get_batch(
                    self.train_batch_size,
                    train_step=train_step,
                )

                empty_expired = (
                    produce_result.status == ProduceBatchStatus.EXPIRED_BATCH
                    and not produce_result.rollout_states
                )
                if not empty_expired:
                    self._train_one_batch(produce_result.rollout_states, train_step)
                    sync_model_step = train_step
                else:
                    sync_model_step = train_step - 1

                if self._need_sync(sync_model_step, produce_result):
                    await self.agent_loop_manager.pause_produce()
                    await self._sync_weights_and_save(sync_model_step)
                    await self.agent_loop_manager.continue_produce(model_step=sync_model_step)

                if empty_expired:
                    continue
                self.cur_step = train_step
                train_step += 1
        finally:
            self.agent_loop_manager.shutdown()
            await producer_task
