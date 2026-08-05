# 2026-08-05_03-20-01_MoonEP前向集成API与VMM异步链路

完成 Issue 01：MoonEP-mod 现在提供带版本号的 XTuner integration capability、`ExpertVMMWorkspace` 前向工作区，以及 `Buffer` 的 projection sequence 与 route-scaled combine 接口。实现位于 `MoonEP-mod/moonep/{workspace.py,api.py,__init__.py}`，公开行为测试位于 `MoonEP-mod/tests/test_xtuner_integration.py`。

## 关键单测

- 纯 metadata `validate()`：验证 BF16、EP2/4/8、expert divisibility、top-k、projection alignment、双 generation；测试前后均未初始化 CUDA。
- 真实 VMM 行为：landing、global `[E+B]` 和 local `[2B]` 映射同一 physical storage；两个 home generation 相互隔离，duplicated pool 跨 generation 共享。
- 真实 8-GPU topology matrix：分别显式构造 EP2、EP4、EP8 subgroup，覆盖 local、remote duplicated、empty、skew route，并跑通 dispatch、双 projection prefetch、route-scaled combine。
- 非默认 caller stream：仅用 dispatch/prefetch/combine 返回的 CUDA event 串联消费者；profiler 未发现 `.item()`、local scalar 或 CUDA host synchronization。
- 生命周期：`destroy()` 可重复调用；遗漏显式销毁时只发 `ResourceWarning` 并保留资源，不在析构器执行 collective teardown。
- 回归：MoonEP 原有 8-GPU E2E 通过；combine/prefetch 为 `24 passed`；planning/dispatch/grad-reduce 为 `42 passed, 1 skipped`。

## 类/接口设计和主要改动

```mermaid
classDiagram
    class ExpertVMMWorkspace {
        +validate(metadata) None
        +allocate(ep_group) ExpertVMMWorkspace
        +landing(generation) ProjectionPair
        +local_token_counts(cu_seqlens) Tensor
        +materialize(buffer, plan, generation) ProjectionPair, Event
        +destroy() None
        -global_weights : E+B aliases
        -local_weights : 2B aliases
        -home_generations : 2
        -shared_duplicate_pool : B
    }
    class Buffer {
        +dispatch(..., async_finish)
        +prefetch_weight(projections, ...)
        +combine(hidden_scales_nvs, ...)
    }
    class MoonEPCommPlan {
        +experts_to_copy
        +dst
    }
    ExpertVMMWorkspace --> Buffer : materialize
    Buffer --> MoonEPCommPlan : consumes
```

`ExpertVMMWorkspace` 是唯一新增的 Deep Class。它隐藏 FD 交换、VMM VA、global/local aliases、generation 与资源释放顺序；XTuner 客户端只接触 landing、local counts、materialize 和 event。`Buffer` 保留原 gate/up/down 三投影兼容接口，并新增任意 projection sequence；`hidden_scales_nvs` 只允许 non-zero-copy combine。

## 类交互和主要改动

```mermaid
flowchart LR
    A["Caller stream: dispatch"] --> B["MoonEP comm stream"]
    B --> C["GPU plan + activation dispatch"]
    C --> D["dispatch event"]
    D --> E["Workspace.materialize"]
    E --> F["global E+B VMM prefetch"]
    F --> G["local 2B alias + prefetch event"]
    G --> H["Grouped compute consumer"]
    H --> I["combine staging: BF16 row scaling"]
    I --> J["route-scaled combine event"]
    J --> K["Caller-stream consumer"]
```

初始化期仅在调用方传入的 `ep_group` 内完成 hostname/device 收集和 FD 交换。热路径中的 plan、counts 与 routing 均留在 device；每段通信由 caller event 入队到 comm stream，再向消费者返回 completion event。

## 其他重要细节

- MoonEP-mod worktree：`/mnt/shared-storage-user/zhaopenghao/github/MoonEP-mod`。
- MoonEP-mod commit：`0be637d feat: add XTuner MoonEP forward integration API`。
- 正式 backward API 与 BF16 duplicated-gradient return 留给 Issue 02；本 issue 不提前混入训练梯度语义。
- 结论：Issue 01 的 forward capability、显式 EP topology、VMM alias 与无 host sync 异步契约均已由真实多 GPU public tests 固化。

# 2026-08-05_03-38-56_MoonEP_BF16梯度环与Duplicate_Return

完成 Issue 02：`ExpertVMMWorkspace` 按调用方给定的 Domino width 分配两块 fused projection 的 BF16 gradient-slot ring，`Buffer.reduce_grad()` 新增正式 BF16 路径，并由 CUDA owner kernel 将所有 duplicated partials 执行 FP32 累加、一次 BF16 舍入的 EP-local SUM。旧 FP32 reduce-grad 接口保持兼容。

## 关键单测

- 两个 gradient slots：验证 BF16、连续 `[2B,O,I]`、storage 独立、slot 越界及错配 slot 拒绝；输出 target 可被生产者直接覆盖，无完整 dW copy。
- BF16 owner SUM：真实远端 VMM reads 使用可区分 partials，逐位比较 rank-major/slot-major FP32 累加后的一次 BF16 舍入，并能检测 BF16 逐项舍入和错误 averaging。
- 真实 8-GPU EP2/EP4/EP8：覆盖 local、remote、empty、skew、同一 expert 多 rank partials；两个 slots 先后生产、逆序完成，并跨 step 复用 slot 0。
- non-default caller stream：gradient target 写入、owner SUM、suffix 清零和消费者只通过 returned CUDA event 排序；profiler 未发现 host scalar 或 CUDA synchronize。
- 回归：2-GPU integration suite 为 `12 passed, 3 skipped`；原有 8-GPU E2E 通过；旧 FP32 grad-reduce suite 为 `12 passed`。

## 类/接口设计和主要改动

```mermaid
classDiagram
    class ExpertVMMWorkspace {
        +materialize(buffer, plan, generation, grad_slot) Weights, GradTargets, Event
        +complete_gradients(buffer, plan, local_grads, grad_slot) HomeGrads, Event
        -local_grad_outputs : N x 2 projections x 2B
        -distributed_duplicate_grads : N x 2 projections x R x B
    }
    class Buffer {
        +reduce_grad(local_grads, distributed_duplicate_grads, accumulation_dtype)
        +reduce_grad(legacy_fp32_args)
    }
    class bf16_grad_reduce_home {
        +launch(local_2B, distributed_RB, plan, rank, num_sms)
        -accumulator : FP32 registers
        -input_output : BF16
    }
    ExpertVMMWorkspace --> Buffer : complete_gradients
    Buffer --> bf16_grad_reduce_home : two projections
```

每个 slot 的 home `[B]` 与 duplicate `[B]` 分别拥有 physical allocation，再映射为 grouped-GEMM 可直接写入的 contiguous local `[2B]`；所有 ranks 的 duplicate chunks 另映射为 owner 可读的 distributed `[R,B]`。ring 大小只等于 `gradient_slots`，不随 layer、MTP depth 或 route 动态扩容。

## 类交互和主要改动

```mermaid
flowchart LR
    A["Caller stream: grouped dW writes local 2B"] --> B["input-ready event"]
    B --> C["MoonEP comm stream"]
    C --> D["device EP barrier: publish all duplicate writes"]
    D --> E["BF16 remote reads + FP32 ordered SUM"]
    E --> F["single BF16 home write"]
    F --> G["device EP barrier: fence all readers"]
    G --> H["fixed duplicate suffix clear"]
    H --> I["completion event"]
    I --> J["FSDP-facing home BF16 consumer"]
```

两个 barrier 都是 GPU-side EP synchronization：第一个保证 owner 不会读取尚未完成的 remote dW，第二个保证任何 rank 清理本地 suffix 前，其他 owners 已完成 remote reads。清理固定 `[B:]`，不查询 plan scalar、不按 route 决定 host launch。

## 其他重要细节

- MoonEP-mod commit：`df548b0 feat: return duplicated expert gradients in BF16`。
- 新 binding 位于 `csrc/bf16_grad_reduce.cuh`；plan 先固定搬入 block shared memory，数据存储和 NVLink traffic 全程 BF16，只有寄存器 accumulator 为 FP32。
- 当前机器默认 `/usr/local/cuda` 为 12.8；重建与 PyTorch 13.2 匹配的 extension 时显式使用环境内 `nvidia/cu13` 作为 `CUDA_HOME`，editable module 与 `_C` 均确认来自 `MoonEP-mod`。
- `M_g=0` 行由后续 XTuner direct-output grouped-GEMM 对所有固定 groups 的覆盖写负责；workspace 不增加整块 memset 或 completion-time clone。
- 结论：Issue 02 已建立可直接交给后续 XTuner expert autograd/FSDP handoff 的 BF16 gradient completion boundary，同时保留旧 FP32 reference path。

# 2026-08-05_05-01-57_XTuner_Staging_Forward端到端链路

完成 Issue 03：XTuner 新增可选的 `dispatcher="moonep"`、model-scoped `MoonEPRuntime`、per-forward `_MoonEPInvocation` 和六阶段 `MoonEPDispatcher`。Router 统一传递 device-resident `tokens_per_expert`；原生 FSDP 安装后才分配 VMM workspace，并通过显式 `moonep_staging_reference` 将完整 BF16 home expert 权重复制到 landing。主要改动位于 `xtuner/v1/module/dispatcher/moonep.py`、`xtuner/v1/model/moe/moe.py`、`xtuner/v1/module/decoder_layer/moe_decoder_layer.py`、`xtuner/v1/module/grouped_linear/moe_group_linear.py` 和对应测试。

## 关键单测

- backend contract：验证 lazy optional import、integration API/capability、module source、固定 PyTorch 版本和 meta-only validation。
- dispatcher public API：验证 staging 显式开关与 warning、Fixed-S、双 generation、六阶段结果以及 completion event 顺序。
- 原有 grouped GEMM public API：验证 `weight_override` 接收 local `[2B]` fused weights，默认 Parameter 路径不变。
- 真实 8-GPU compiled forward：Qwen-compatible tiny 模型在 BF16 FSDP2×EP4 下与 DeepEP 对齐，并连续运行四次 MoonEP no-grad forward；GLM5.2 tiny 模型与 All2All 对齐。
- 非 MoonEP 回归：NoEP、DeepEP、All2All、AGRS 的统一 `tokens_per_expert` seam 通过，合计 `14 passed, 1 skipped`。
- MoonEP-mod peer-skew 回归：在一个 rank 的 landing copy 前注入 GPU-only 延迟，确认 prefetch 不会读取上一 generation 的远端旧权重；profiler 继续排除 `.item()` 和 CUDA host synchronization。

## 类/接口设计和主要改动

```mermaid
classDiagram
    class MoonEPRuntime {
        +dispatcher_for(layer_fqn, experts) MoonEPDispatcher
        +install_fsdp(model, fsdp_config, staging_reference) None
        +destroy() None
        -buffer : Buffer
        -workspace : ExpertVMMWorkspace
        -fixed_tokens_per_rank : int
    }
    class MoonEPDispatcher {
        +dispatch_preprocess(...) MoonEPPreDispatchResult
        +dispatch(...) MoonEPDispatchResult
        +dispatch_postprocess(...) MoonEPPostDispatchResult
        +combine_preprocess(...) MoonEPPreCombineResult
        +combine(...) MoonEPCombineResult
        +combine_postprocess(...) MoonEPPostCombineResult
    }
    class _MoonEPInvocation {
        +dispatch(...) Result
        +prepare_experts(...) ExpertTensorBundle
        +combine(...) Tensor
        +wait_combined() None
        +finish_forward_only() None
    }
    class GroupedLinear {
        +forward(x, tokens_per_expert, weight_override) Tensor
    }
    MoonEPRuntime --> MoonEPDispatcher : registers physical routed layers
    MoonEPDispatcher --> _MoonEPInvocation : creates per forward
    _MoonEPInvocation --> GroupedLinear : supplies local 2B tensors
```

`MoonEPRuntime` 隐藏 model/EP-group 资源和 Fixed-S 生命周期；`_MoonEPInvocation` 隐藏一次 dispatch/combine 的 plan 与 device events；decoder 仅消费六阶段 TypedDict 和显式 tensor bundle，不按 backend 分支执行专家计算。

## 类交互和主要改动

```mermaid
sequenceDiagram
    participant R as Router
    participant D as MoonEPDispatcher
    participant W as ExpertVMMWorkspace
    participant G as GroupedLinear
    participant B as MoonEP Buffer
    R->>D: hidden, topk ids/weights, tokens_per_expert
    D->>B: dispatch(async event)
    D->>W: stage BF16 FSDP views into home landing
    W->>W: GPU EP barrier, prefetch remote duplicates
    W-->>D: local 2B weights, direct grad targets, event
    D->>G: ExpertTensorBundle
    G-->>D: local expert output
    D->>B: route-scaled combine
    B-->>D: completion event
```

staging、remote prefetch 和 dispatch 的依赖全部通过 CUDA event/device barrier 排序。真实 peer-skew 复现定位到远端 rank landing 尚未发布的竞态，因此 MoonEP-mod 在 prefetch 前增加已有的 GPU inter-rank barrier；热路径未增加 host sync。

## 其他重要细节

- MoonEP-mod 修复提交：`4b3aefe fix: synchronize staged weights before prefetch`。
- `intra_layer_micro_batch` 由 Trainer 在 model build 前只传入一个 scalar；未传递整个 TrainerConfig。
- dense prefix 不注册 dispatcher、不占 generation；shared experts/gate 保持原路径；全 dense 配置在 workspace allocation 前失败。
- staging 是数值参考路径并明确 warning；Issue 05 将用 direct FSDP landing 替换逐 forward 的完整权重 copy。
- 结论：Issue 03 已跑通 staging forward tracer，保留 BF16 FSDP 参数身份、原 grouped-GEMM 计算和无 host sync 契约。
