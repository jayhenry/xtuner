# 共卡 / 非共卡生产代码拆分设计

## 1. 背景

当前 `AgentLoopManager` 同时承担两套生产模式：

- 共卡训练：一次 `produce_batch()` 内完成 rollout 生产、pending 收尾、从 replay buffer 取训练 batch。
- 非共卡训练：后台 **Background Producer** 持续写 replay buffer，前台 **Training Consumer** 通过 `get_batch()` 消费，并在 **Expired Produce Batch**、权重同步、评测、checkpoint 之间切换状态。

这两套模式共享同一个 `AgentLoopManager`、同一个 `ProduceProgress`、同一个 `AsyncProduceStrategy` 实现。结果是：

- 共卡路径需要理解 `_status / _update_event / _model_step / _produce_progress` 等非共卡状态。
- 非共卡路径修改容易改变共卡 `produce_batch()` 的同步行为。
- `AsyncProduceStrategy` 的 pending task 既被当作单次调用局部状态，又被当作非共卡跨调用后台状态。

本设计目标是拆开生产侧代码，让共卡生产和非共卡生产各自有独立 **Module**、独立 **Interface** 和独立状态；同时保留 `AsyncProduceStrategyConfig` 在共卡训练中的异步生产能力。

## 2. 目标

1. 共卡生产修改不影响非共卡生产。
2. 非共卡 **Background Producer** / **Training Consumer** 状态机修改不影响共卡 `produce_batch()`。
3. `AsyncProduceStrategyConfig` 保持可用于共卡训练和非共卡训练，但构建出不同的具体 strategy **Adapter**。
4. 共卡 async 生产保持简单：pending task 是单次 `produce_batch()` 的局部变量，不跨调用保存。
5. 非共卡 async 生产保留后台 pending、pause/continue、Expired Produce Batch、checkpoint/resume 等能力。

## 3. 非目标

- 不改变 trainer 配置文件中的 `AsyncProduceStrategyConfig(...)` 用法。
- 不改变 replay buffer 的领域语义。
- 不在共卡路径引入非共卡状态机。
- 不把所有共享 helper 都拆成公开接口；共享逻辑可以作为私有 Implementation 留在 manager 包内部。

## 4. 总体方案

把现在一个宽 `AgentLoopManager` 拆成两个 manager **Module**：

- `ColocateAgentLoopManager`
- `DisaggregatedAgentLoopManager`

把现在一个 `ProduceProgress` 拆成两个进度 **Module**：

- `ColocateProduceProgress`
- `BackgroundProduceProgress`

把现在一个完整 `AsyncProduceStrategy` 拆成两个具体 strategy **Adapter**：

- `ColocateAsyncProduceStrategy`
- `BackgroundAsyncProduceStrategy`

`AsyncProduceStrategyConfig` 保留，按 manager 模式构建具体 Adapter：

```python
AsyncProduceStrategyConfig(...).build(mode=ProducerMode.COLOCATE)
# -> ColocateAsyncProduceStrategy

AsyncProduceStrategyConfig(...).build(mode=ProducerMode.DISAGGREGATED)
# -> BackgroundAsyncProduceStrategy
```

也就是说，拆分的是执行模式，不是删除共卡 async。

设计约束：

- `ColocateAsyncProduceStrategy` 和 `BackgroundAsyncProduceStrategy` 不继承公共父类。两者各自显式持有配置字段，少量共享算法用 module-level helper 函数表达。
- `ColocateAgentLoopManager` 和 `DisaggregatedAgentLoopManager` 不继承公共父类。task batch 分配、staleness refresh、take batch、result 聚合等共享逻辑用 module-level helper 函数表达。
- `pause_produce` 的关键顺序和 pending drain 协议必须复用当前生产代码语义，抽成独立 helper，而不是藏在某个 manager 父类或 async strategy 父类里。

## 5. Module 职责

| Module | Interface | Implementation |
| --- | --- | --- |
| `AgentLoopManagerConfig` | `build_colocate(...)`, `build_disaggregated(...)` | 构建 task runner、sampler、agent loop、mode-specific strategy |
| `ColocateAgentLoopManager` | `produce_batch(batch_size, train_step, model_step)` | 共卡单次生产、局部收尾、取训练 batch |
| `DisaggregatedAgentLoopManager` | `produce_loop`, `get_batch`, `pause_produce`, `continue_produce`, `shutdown` | 非共卡后台生产和消费状态机 |
| `ColocateProduceProgress` | `target_for`, `record_metrics`, `cleanup` | 单次共卡生产窗口，不进 checkpoint |
| `BackgroundProduceProgress` | `ensure_target_upto`, `begin_consume`, `mark_consumed`, `state_dict` | 非共卡绝对累计 target/consumed 和 resume 状态 |
| `ColocateAsyncProduceStrategy` | `produce_batch(ctx)` | 局部 pending set，结束时 drain 当前 pending |
| `BackgroundAsyncProduceStrategy` | `produce_batch(ctx)`, `pause_produce(ctx)` | `_PendingTasks` 跨调用保存，处理 update event 和 model expired |

建议的共享 helper：

| Helper | 用途 |
| --- | --- |
| `allocate_task_batch_sizes(...)` | 复用当前按 task weight 分配 batch 的逻辑 |
| `validate_task_batch_sizes(...)` | 复用 batch size 校验 |
| `refresh_for_all_tasks(...)` | 复用 completed / aborted staleness refresh |
| `take_train_batch(...)` | 复用 replay buffer take、consumed 记账、leftover 统计、result 聚合 |
| `sample_retry_or_new_group(...)` | 复用 async strategy 从 retry pool / dataloader 抽样的选择 |
| `is_model_expired_for_threshold(...)` | 复用 staleness threshold 判定 |
| `pause_pending_tasks(...)` | 复用 pending task pause / drain / cancel 协议 |

这些 helper 是 Implementation 复用，不是新的业务 **Interface**。调用方仍只看到 mode-specific manager 和 strategy。

## 6. Config 构建规则

`TaskSpecConfig.produce_strategy_config` 继续接受 `SyncProduceStrategyConfig` 或 `AsyncProduceStrategyConfig`。差异只发生在 build 阶段：

```python
class AgentLoopManagerConfig:
    def build_colocate(...):
        task_runners = self._build_task_runners(mode=ProducerMode.COLOCATE, ...)
        return ColocateAgentLoopManager(task_runners, replay_buffer, logger)

    def build_disaggregated(...):
        task_runners = self._build_task_runners(mode=ProducerMode.DISAGGREGATED, ...)
        return DisaggregatedAgentLoopManager(task_runners, replay_buffer, logger)
```

`SyncProduceStrategyConfig` 可以在两种模式下都构建 `SyncProduceStrategy`，但非共卡 trainer 可继续保持现有约束：如果非共卡不允许 early stopping，则检查 `should_continue_fn` 必须是默认函数。

`AsyncProduceStrategyConfig` 按模式分派：

```python
class AsyncProduceStrategyConfig:
    def build(self, *, mode, sync_weights_interval, rollout_controller):
        if mode == ProducerMode.COLOCATE:
            return ColocateAsyncProduceStrategy(...)
        if mode == ProducerMode.DISAGGREGATED:
            return BackgroundAsyncProduceStrategy(...)
```

这个 **Seam** 的价值是：调用方仍只关心 `ProduceStrategy` **Interface**，但共卡和非共卡拿到的是不同 **Adapter**。

## 7. 共卡生产流程

共卡路径只允许一个 public 入口：

```python
await manager.produce_batch(batch_size, train_step, model_step=model_step)
```

流程：

1. 根据 `train_step` 计算 task batch sizes。
2. 创建 `ColocateProduceProgress`。
3. `continue_generation()`，切到 rollout 阶段。
4. 各 task 调用 mode-specific strategy 生产到 replay buffer。
5. strategy 收尾本次调用的 pending task。
6. 从 replay buffer 取 completed rollout groups。
7. `pause_generation()`，切回静止态。
8. 返回非空 `ProduceBatchResult`。

共卡路径不出现：

- `_status`
- `_update_event`
- `_finish_event`
- `BackgroundProduceProgress`
- `_PendingTasks`
- `produce_loop`
- `get_batch`
- `pause_produce`
- `continue_produce`

## 8. 非共卡生产流程

非共卡路径由两个 public 入口协作：

```python
producer_task = create_task(manager.produce_loop(batch_size))
produce_result = await manager.get_batch(batch_size, train_step=train_step)
```

`DisaggregatedAgentLoopManager` 独占以下状态：

- `status`
- `update_event`
- `finish_event`
- `model_step`
- `pause_time_s`
- `BackgroundProduceProgress`

核心不变量：

- **Background Producer** 只在 `NORMAL` 状态下推进 `producer_future_step`。
- **Training Consumer** 成功取出非空 batch 后推进 `consumed_samples` 和 `next_consumer_step`。
- **Expired Produce Batch** 只有在训练侧已有更新 **Model Step** 时，才允许返回空 batch 跳过训练。
- 权重同步前必须 `pause_produce()`，同步/评测后必须 `continue_produce(model_step=...)`。

## 9. Async 策略拆分

`AsyncProduceStrategyConfig` 不变，但旧 `AsyncProduceStrategy` 的完整实现拆成两个具体 Adapter。

### 9.1 `ColocateAsyncProduceStrategy`

职责：

- 本次 `produce_batch()` 内创建 `pending_tasks = set()`。
- 按 `over_sample_threshold`、tail batch、partial rollout 规则调度 rollout group。
- 收到完成结果后过滤、写 replay buffer、更新本次 metrics。
- 达到本次 batch target 后，暂停 agent loop 并 drain 本次 pending。

它不负责：

- 跨调用保存 pending。
- 观察 `update_event`。
- 返回 `UPDATE_WEIGHT_AND_ABORT`。
- 维护 `model_step` 状态机。
- checkpoint pending task。
- 继承公共 async 父类。

### 9.2 `BackgroundAsyncProduceStrategy`

职责：

- 持有 `_PendingTasks`，允许 pending task 跨多次 `produce_batch()` 调用存在。
- 观察 `ctx.should_abort()`。
- 根据 `model_step / producer_future_step` 判断 **Expired Produce Batch**。
- `pause_produce()` drain 或 cancel pending。
- 为 checkpoint 提供 `pending_task_count()`。

它不负责：

- 从 replay buffer 取训练 batch。
- 推进 `BackgroundProduceProgress` 的 consumer step。
- 触发权重同步。
- 继承公共 async 父类。

### 9.3 pause pending helper

当前最新 `pause_produce` 有两个层次：

1. manager 层：先设置暂停信号，切换 manager 状态，再暂停 rollout controller。
2. strategy 层：如果还有 pending task，周期性发送 agent loop pause，claim 已完成任务并入库，超过 timeout 后 cancel 剩余 pending。

拆分后保留这个顺序，但把 strategy 层 pending drain 抽成全局 helper：

```python
async def pause_pending_tasks(
    *,
    pending_tasks,
    ctx,
    put_claimed_task,
) -> float:
    if pending_tasks.count() == 0:
        return 0.0

    pending_pause_tasks = {create_task(request_agent_loop_pause(ctx))}
    deadline = now() + PRODUCER_PAUSE_PENDING_TASK_TIMEOUT_S
    next_periodic_pause = now() + PERIODIC_ABORT_INTERVAL_S

    while pending_tasks.count() > 0:
        if now() > deadline:
            await pending_tasks.cancel_all()
            break

        if now() >= next_periodic_pause:
            pending_pause_tasks.add(create_task(request_agent_loop_pause(ctx)))
            next_periodic_pause += PERIODIC_ABORT_INTERVAL_S

        claimed = await pending_tasks.wait_and_claim(timeout_s=1)
        for task in claimed:
            await put_claimed_task(task)

    await cancel_and_drain(pending_pause_tasks)
    return elapsed()
```

共卡路径把本次调用的局部 `set[Task]` 包成 `_LocalPendingTasks` 后调用这个 helper；非共卡路径直接把 `_PendingTasks` 传给它。这样 pause 协议复用，但 pending 的生命周期仍然独立：

- 共卡：pending 生命周期等于一次 `produce_batch()`。
- 非共卡：pending 生命周期跨多次后台 `produce_batch()`。

## 10. Progress 拆分

### 10.1 `ColocateProduceProgress`

字段：

- `task_batch_sizes`
- `train_step`
- `model_step`
- 本次 raw reward / produced samples / produced tokens / produce time metrics

特点：

- 不保存到 checkpoint。
- 不维护绝对累计 target。
- 不维护 consumer step。
- 不暴露 `state_dict()`。

### 10.2 `BackgroundProduceProgress`

字段：

- `producer_future_step`
- `next_consumer_step`
- `target_samples`
- `consumed_samples`
- `target_upto_future_step`
- 后台 producer metrics

特点：

- `target_samples / consumed_samples` 使用绝对累计口径。
- `state_dict / load_state_dict` 是非共卡 checkpoint 的一部分。
- producer 和 consumer 共享同一个对象引用。

## 11. ReplayBuffer 保持共享

Replay buffer 是真正共享的深 **Module**，不按共卡/非共卡拆。它提供：

- `put(...)`
- `refresh_staleness(...)`
- `is_ready(...)`
- `take_batch(...)`
- `count_statuses(...)`

共享理由：

- 共卡和非共卡都需要落库和取 completed rollout groups。
- Replay buffer 不理解 manager 状态机。
- Replay buffer 的 **Interface** 已经足够表达 storage / replay ordering 行为。

## 12. Trainer 集成

共卡 trainer：

```python
self.agent_loop_manager = cfg.agent_loop_manager_cfg.build_colocate(...)
```

非共卡 trainer：

```python
self.agent_loop_manager = cfg.agent_loop_manager_cfg.build_disaggregated(...)
```

评测 manager 建议始终用共卡 manager：

```python
self.eval_agent_loop_manager = cfg.eval_agent_loop_manager_cfg.build_colocate(...)
```

原因：evaluation 是一次性 `produce_batch()`，不是后台 **Background Producer**。

## 13. 迁移步骤

1. 新增 `ProducerMode` 和 mode-specific `build_colocate / build_disaggregated`。
2. 新增 `ColocateAgentLoopManager`，把当前 `produce_batch()` 的共卡逻辑迁移过去。
3. 新增 `DisaggregatedAgentLoopManager`，把 `produce_loop/get_batch/pause/continue/shutdown/save/resume` 迁移过去。
4. 拆出 `ColocateProduceProgress` 和 `BackgroundProduceProgress`。
5. 把当前 `AsyncProduceStrategy` 拆成 `ColocateAsyncProduceStrategy` 和 `BackgroundAsyncProduceStrategy`。
6. 把 batch allocation、refresh、take batch、pause pending drain 抽成 module-level helper。
7. trainer 按类型调用不同 build 方法。
8. 保留必要兼容导出，减少配置文件修改。

## 14. 测试建议

共卡 manager：

- `AsyncProduceStrategyConfig` 在共卡下构建 `ColocateAsyncProduceStrategy`。
- 共卡 async `produce_batch()` 每次调用后不保留 pending。
- 共卡 `produce_batch()` 不访问 `_status/_update_event/BackgroundProduceProgress`。
- 共卡 multi-task batch allocation 仍稳定。
- 共卡 local pending cleanup 复用 `pause_pending_tasks(...)`，但不创建 `_PendingTasks`。

非共卡 manager：

- `AsyncProduceStrategyConfig` 在非共卡下构建 `BackgroundAsyncProduceStrategy`。
- `produce_loop/get_batch` 仍处理空/非空 **Expired Produce Batch**。
- `pause_produce/continue_produce` 顺序不变。
- `pause_produce` 先设置 update event / manager status，再暂停 rollout controller，然后调用 strategy pending drain。
- checkpoint/resume 恢复 `BackgroundProduceProgress` 和 `model_step`。

策略：

- `ColocateAsyncProduceStrategy` 覆盖 oversample、partial rollout、tail batch、local pending drain。
- `BackgroundAsyncProduceStrategy` 覆盖 `_PendingTasks` claim/schedule/cancel、abort、expired。

trainer：

- 共卡 trainer 只依赖 `produce_batch()`。
- 非共卡 trainer 只依赖 `produce_loop/get_batch/pause/continue/shutdown`。
- 非共卡 trainer 的 eval manager 是 colocate manager，initial evaluate 后按非共卡训练需求恢复 producer。

## 15. 关键判断

`AsyncProduceStrategy` 的领域含义不是“非共卡策略”，而是“异步 rollout 生产策略”。因此它应继续支持共卡训练。

真正需要隔离的是执行环境：

- 共卡执行环境：局部 pending，单次调用完成。
- 非共卡执行环境：后台 pending，跨调用状态机。

所以最终代码形状应是：

```python
AsyncProduceStrategyConfig
    -> ColocateAsyncProduceStrategy
    -> BackgroundAsyncProduceStrategy
```

而不是：

```python
AsyncProduceStrategy
    .colocate_runtime
    .background_runtime
```

后者会让一个 Module 继续知道两套执行协议，复杂度只是换位置，不能提供足够的 **Locality**。
