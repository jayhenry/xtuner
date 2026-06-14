# LoadSpec Refactor — Progress Log

配套设计文档：[`load_spec_refactor.md`](load_spec_refactor.md)。后续 TP 设计 [`dense_tp.md`](dense_tp.md) 依赖本 refactor 完成。

迁移路径（设计文档 §6）：PR 1 additive schema → PR 2 load 合一 → PR 3 save 合一 → PR 4 删除 legacy。

---

## 状态总览

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| PR 1 | additive schema + `_init_load_spec` 同时填新旧字段 + `__init__` 契约断言 | ✅ 已落地，回归绿 |
| PR 2a | 统一 `_load_hf_param` + 引入私有 `_plan_hf_keys`；保留 `safetensors_to_params` 旧签名 | ✅ 代码已落地；MoE 回归因 GPU 0 OOM 阻塞未跑完 |
| PR 2b | 切 `safetensors_to_params` 签名为 `(safetensors, local_tensor, load_spec)`，迁移三个子类 override | ⏳ 推后（小步子） |
| PR 3 | 统一 `_get_hf_param`（save path），保留 `_fsdp_foreach_allgather` fp8 分支 | ⏳ 未开始 |
| PR 4 | 删除 `LoadEnum` 与 `LoadSpec` legacy 字段 | ⏳ 未开始 |
| 后续 | 精简 `dense_tp.md` §4.5，改为引用 refactor 文档 | ⏳ 未开始 |

---

## 已完成内容

### PR 1 — additive schema + `_init_load_spec`

**改动文件**
- `xtuner/v1/utils/load_spec.py`
  - 新增 `ShardDescriptor(BaseModel)`：`dim` / `start` / `end` / `group`。语义：应用于"fused full tensor"的一次分区；`shards` 里后一个 descriptor 的 offset 相对于前一个的 sub-tensor（对齐 DTensor placements）。
  - `LoadSpec` 新增字段：`global_hf_keys` / `global_shape` / `fused_dim` / `shards`；legacy 字段（`hf_keys` / `shape` / `dim` / `shard_start` / `shard_end` / `group` / `load_enum`）保留。
  - `@computed_field` `is_fused` / `is_sharded`：前者基于 `global_hf_keys`（fallback `hf_keys`），后者严格基于 `shards` 非空（**不**依赖 `load_enum`，这样 DTensor 的 Replicate 占位也能正确归类）。
  - `model_post_init`：补默认值 + 检查契约（SAME 单 key / FUSED dim==0 / SHARD 必须带 dim+offset / fused_dim 必填 / shard offset 在 global_shape 范围内）。
- `xtuner/v1/model/base.py` 的 `_init_load_spec`：统一出口同时填新旧字段；非 MoE 走 full key list，MoE EP 走本 rank 的 local subset（legacy）+ 完整 global key list（新）。

**回归（全部绿）**
- `tests/model/test_qwen3_dense.py::TestQwen3Dense::test_save_hf`（tp=1），101 s。
- `tests/model/test_qwen3_moe.py::TestQwen3MoE::test_save_hf`（ep∈{1,4,8}），169 s，3 个参数化全绿，safetensors bit-equal。

**单测**
- 新增 `tests/utils/test_load_spec.py`，10 个 test，3 个 class：
  - `TestLoadSpecLegacyDefaults`：legacy-only 构造；验证 `_init_load_spec` 自动补新字段后 `is_fused` / `is_sharded` 语义正确。
  - `TestLoadSpecNewSchema`：显式填新字段；包括 `is_sharded` 仅由 `shards` 决定（即使 `load_enum=SAME`）以及多轴 `shards` 保序。
  - `TestLoadSpecInvariants`：`model_post_init` 抛异常时走 `pytest.raises(ValidationError, match=...)`（Pydantic 把 AssertionError 包成 ValidationError，**不能**用 AssertionError 接）。
  - 用 module-scoped 单 rank gloo PG fixture（`dist.ProcessGroup` Pydantic 强校验）；避开 multiprocess 基础设施。

---

### PR 2a — 统一 `_load_hf_param` + `_plan_hf_keys`

**改动文件**
- `xtuner/v1/model/base.py`
  - `_load_params` 中 `LoadEnum` 三路 dispatch（`_load_same_hf_param` / `_load_fused_hf_param` / `_load_shard_hf_param`）替换为单一 `self._load_hf_param(param, load_spec, checkpoint_loader)`。
  - 删除上述三个私有方法（约 180 行），引入：
    - `_load_hf_param`：拆 DTensor → 调 `_plan_hf_keys` → `None` 分支（fp8 pad-only：要求 `config.float8_cfg is not None`，`local_tensor.zero_()` 后 return `[]`）→ 否则依次 load hf_keys（`is_float8_weight` 走 `_load_fp8` dequant 分支）→ 处理 missing → 转给 `safetensors_to_params`。
    - `_plan_hf_keys`（私有）：按 `load_enum` 分支返回 `(hf_keys, start, end) | None`：
      - `SAME`：`[hf_keys[0]]` + FSDP 切片（仅当本 tensor 是 DTensor 且 mesh 里有 `fsdp_mesh` 且 placement 为 Shard）。
      - `FUSED`：EP-subset hf_keys；fp8 时用 `local_tensor._ori_shape[FSDP_SHARD_DIM] / (ep_world_size * len(hf_keys))` 算每个 hf_key 的 pre-pad 大小；`hf_keys_start == hf_keys_end` 时返回 `None`（fp8 pad-only）。
      - `SHARD`：`[hf_keys[0]]`；当同时有 FSDP 时 `start = fsdp_start + shard_start`，否则回退到 legacy `shard_start/shard_end`。
  - **`safetensors_to_params` 签名保持不变**（仍为 `(safetensors, local_tensor, param_name, start, end, dim)`）。这是 PR 2a 小步子的核心：结构统一与签名切分两件事。

**fp8 语义决策（记忆归档）**
- 设计文档 §4/§5 曾探讨把 `is_fp8` / `padded_shape` / `_ori_shape` 放进 LoadSpec schema（"fp8 感知抽到 schema 层"）。
- 用户 2026-04-20 明确："fp8 在 load 和 save 的时候一定是会感知到的，涉及到 quant 和 dequant，所以 fp8 这一层和 load_spec 一定是耦合的" → 选 Y：fp8 的感知**只**在 load/save 路径里通过 `is_float8_weight(local_tensor)` / `local_tensor._ori_shape` 就地处理；LoadSpec schema 保持纯"布局描述"。
- 对应实现：`_plan_hf_keys` 里现场 fp8 分支；`global_shape` 就是 `param.shape`（= padded），不为 fp8 额外加字段。
- 记忆：`memory/project_fp8_load_spec_coupling.md`。

**PR 拆分决策（记忆归档）**
- 原计划把"结构合一（三入口合一）+ 签名改动（`safetensors_to_params` 换参数 + 迁移子类 override）"塞一个 PR。
- 用户 2026-04-20 选小步子：PR 2a 只做结构合一保留旧签名（这步能跑 `test_save_hf` 做 bit-equal regression 兜底），PR 2b 单独做签名变更 + 子类迁移。
- 记忆：`memory/feedback_refactor_small_steps.md`。

**回归状态**
- ✅ `test_qwen3_dense::test_save_hf`（tp=1），66 s 通过。
- ⛔ `test_qwen3_moe::test_save_hf` 被 GPU 0 OOM 阻塞（`CUDA_VISIBLE_DEVICES` 重映射也不行——测试硬编码 `world_size=8`）。报错位置是 `loaded_tensor.index_select` 内的 `OutOfMemoryError`，在走到任何 refactor 新逻辑之前就触发，属**环境问题**（GPU 0 被 PID 1569589 leaked worker 占 83 GiB + 另一用户 39 GiB），**不是**代码缺陷。
- 待用户确认后重跑（kill 自己的 leaked worker / 换时段）。

---

## 未完成内容（按执行顺序）

### 【阻塞确认】PR 2a MoE 回归
- 需要用户对 GPU 0 上 leaked 的 `PID 1569589` 作决策（kill or wait），重跑 `test_qwen3_moe::test_save_hf` 三个参数化（ep∈{1,4,8}）+ 补上 `test_qwen3_moe::test_save_hf_fope` 与 GPTOSS MoE。
- 验收标准：safetensors bit-equal，对齐 PR 1 的 baseline。

### PR 2b — `safetensors_to_params` 签名切换 + 子类迁移
- 目标签名：`safetensors_to_params(safetensors: dict[str, Tensor], local_tensor: Tensor, load_spec: LoadSpec)`（对应设计文档 §5.3）。
- `_load_hf_param` 调用点同步更新（去掉 `param_name` / `start` / `end` / `dim` 显式传参）。
- 子类 override 迁移（三个）：
  - `xtuner/v1/model/moe/gpt_oss.py`
  - `xtuner/v1/model/moe/qwen3_5_text.py`
  - `xtuner/v1/model/moe/qwen3vl_text.py`
  - 共性：cat → reshape/transpose（按 `load_spec.name` 如 `"fused_w1w3.weight"` / `"fused_w2.weight"` 分支）→ 通用 FSDP_SHARD_DIM narrow（fp8 走 pad 分支）。迁移时把原本靠参数拿到的 `param_name` / `start` / `end` / `dim` 改为从 `load_spec` 字段读。
- 回归：同 PR 2a 的 `test_save_hf` 矩阵。

### PR 3 — save path 合一（`_get_hf_param`）
- 对照 `_load_*_hf_param`，把 `_get_same_hf_param` / `_get_fused_hf_param` / `_get_shard_hf_param` 合并到 `_get_hf_param`。
- `_fsdp_foreach_allgather`（fp8 per-block 场景）保留为 fp8 专属分支——与 PR 2a 的 fp8 决策一致，不抽到 LoadSpec schema。
- 回归：`test_save_hf`（dense + moe 全矩阵）bit-equal。

### PR 4 — 删除 legacy
- 删除 `LoadEnum`（`xtuner/v1/utils/load_spec.py`）。
- 删除 `LoadSpec.hf_keys` / `shape` / `load_enum` / `dim` / `shard_start` / `shard_end` / `group` 等 legacy 字段。
- 所有 consumer（`_load_hf_param` / `_get_hf_param` / `_group_param_by_load_spec` / RL worker weight sync 等）切到新字段。
- 回归：`test_save_hf` + `test_load_spec` 全绿。

### 文档收尾
- `docs/design/dense_tp.md` §4.5 中对老 LoadSpec 的讨论精简，改为 "详见 `load_spec_refactor.md`" 引用。

---

## 代码改动 footprint（当前未 commit）

```
xtuner/v1/model/base.py      | 308 +++++++++++++++++++++-----------
xtuner/v1/utils/load_spec.py | 125 +++++++++++++-
tests/utils/test_load_spec.py | (新增文件)
```

分支：`main`（尚未切 feature 分支 / 尚未 commit）。

---

## 决策 & 记忆索引

- `memory/project_fp8_load_spec_coupling.md` — fp8 的感知只存在于 load/save 路径；LoadSpec schema 不加 `is_fp8` / `padded_shape` / `ori_shape`。
- `memory/feedback_refactor_small_steps.md` — 大 refactor 拆小步；签名变更要单独 PR。
