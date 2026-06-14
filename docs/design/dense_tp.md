# XTuner Dense Tensor Parallel 设计

> Draft v3 · 当前实现文档。本文只描述 Dense 模型 TP，不描述 MoE EP 的内部策略。

## 1. 目标

本轮实现 Dense 模型的 Tensor Parallel，并配套 Megatron 风格 Sequence Parallel
（以下简称 TP-SP）：

- Dense 模型从 `TransformerConfig.tp_size` 读取 TP size。
- TP mesh 由 Dense 模型自己创建，engine 不感知 `tp_mesh`。
- 开启 TP 时，attention / MLP 内部按需在序列维做 `all_gather` / `reduce_scatter`。
- KV head 少于 TP size 时，MHA 内部使用 KV replication，而不是限制只能开很小的 TP。
- 本阶段暂时禁止 Ulysses SP 与 TP-SP 同时开启，直到两套 SP 的组合语义单独设计清楚。

非目标：

- 不支持 Dense TP + Ulysses SP 同时开启。
- 不支持 Dense TP + MLA / GatedDeltaNet / linear attention。
- 不支持 Dense TP + float8。
- 不改 KV cache、prefilling、decoding。
- 不把 attention 专属的 KV replication 语义放进通用 `_Linear`。

## 2. 配置与入口

Dense TP 的配置入口是 `TransformerConfig.tp_size`：

```python
class TransformerConfig(XTunerBaseModelConfig):
    tp_size: int = 1
```

`FSDPConfig.tp_size` 是旧路径，不再作为 Dense TP 的行为来源。旧字段过去没有真正启用模型
TP；现在如果配置为 `> 1` 会直接报错，要求用户迁移到 `model.tp_size`。

当前行为边界：

- `Trainer` 通过 model config 推导 `model_tp_size`。
- `TrainEngine.data_replicate_size` 通过 model config 推导 TP 副本数。
- `TrainEngine.build_model()` 仍然只做 `model_cfg.build()` 和 `model.fully_shard(fsdp_cfg)`。
- Dense 在 `__init__` 阶段创建 TP mesh，并在模型构建过程中调用 `parallelize()`。

这样保持了 Dense 和 MoE 的风格一致：模型构建完成时，模型自身的并行切分已经完成。

## 3. Mesh 设计

Dense 自己创建 TP mesh：

```python
init_device_mesh(
    DEVICE,
    (world_size // tp_size, tp_size),
    mesh_dim_names=(f"{mesh_prefix}.dp", f"{mesh_prefix}.tp"),
)[f"{mesh_prefix}.tp"]
```

FSDP / HSDP 后续会创建包含 TP 维的 composite mesh，并校验其中的 TP submesh 与 Dense
初始化阶段创建的 `self.tp_mesh` 一致：

- FSDP + TP：`(fsdp, tp)`
- HSDP + TP：`(hsdp_replicate, hsdp_shard, tp)`

这个设计刻意让 engine 不传递 `tp_mesh`。engine 只负责训练生命周期；模型并行拓扑由模型配置和
模型实现自己维护。

Dense 和 MoE 的 mesh 不放在 `BaseModel` 上统一管理。Dense TP mesh 与 MoE EP mesh
不一定具备相同语义，也不要求完全一致。

## 4. SP 策略

当前全局只允许一种 active SP：

```python
if model_tp_size > 1 and sp_size > 1:
    raise ParallelConfigException("Ulysses SP and TP-SP cannot be enabled together yet.")

self.sp_mesh = self.data_mesh["tp"] if model_tp_size > 1 else self.data_mesh["sp"]
```

含义：

- `tp_size == 1` 时，`sp_mesh` 是 Ulysses SP mesh，保持旧行为。
- `tp_size > 1` 时，`sp_mesh` 是 TP mesh，用同一个 `SequenceContext.split()` 表达 TP-SP。
- 不存在“先 Ulysses split，再 TP-SP split”的双重切分路径。

`SequenceContext.split()` 会切 `input_ids`、`position_ids`、loss kwargs 等序列维张量。
在 TP-SP 下，进入 Dense 的 `hidden_states` 是局部序列：

```text
[b, s / tp, h]
```

loss 也按同一个 TP-SP mesh 切分，所以 Dense 最后的 `norm + lm_head + loss` 也保持局部序列。
Dense 末尾不会再把 hidden gather 成 full sequence，否则 logits 与 labels 的序列长度会不一致。

## 5. RoPE 与 raw_position_ids

TP attention 会在 QKV 投影前 gather hidden：

```text
[b, s / tp, h] -> tp all_gather -> [b, s, h]
```

因此 RoPE metadata 也必须覆盖 gathered 后的 full sequence。这个责任放在 Dense 层，而不是 MHA
层：

```python
position_ids = seq_ctx.position_ids
if self.tp_mesh is not None:
    position_ids = seq_ctx.raw_position_ids
position_embeddings = self.rotary_emb(hidden_states, position_ids)
```

`SequenceContext` 负责记录 `raw_position_ids`：

- 构造默认 `position_ids` 时，先 pad 并记录 full tensor，再按 SP mesh split。
- 显式调用 `split()` 时，在切 `position_ids` 前记录 padded full tensor。
- 如果用户显式传入 `position_ids` 且没有同步传入 raw tensor，`raw_position_ids` property 会在
  首次访问时 fallback gather。
- `raw_position_ids` property 优先返回已记录的 full tensor；缺失时才 fallback gather。

MHA 永远不 gather `cos/sin`。MHA 只校验传入的 RoPE embedding 与 gathered hidden 的序列长度一致。
这样 `position_ids` / RoPE 的布局选择集中在 Dense，attention 不需要理解 `SequenceContext` 的 raw
metadata 语义。

## 6. Linear TP 边界

通用 `_Linear` 只支持普通 1-D tensor parallel：

- `parallelize(tp_mesh, dim=0)`：切 PyTorch weight dim 0，即 col-parallel。
- `parallelize(tp_mesh, dim=1)`：切 PyTorch weight dim 1，即 row-parallel。
- row-parallel bias 保持 replicated，`forward()` 不提前加 bias；调用方在 TP reduce-scatter 后调用
  `add_replicated_bias()`。

`_Linear` 不接受通用 placements，也不理解 attention head、KV replication、GQA 等语义。
attention 特殊策略只放在 MHA 内部。

## 7. DenseMLP

Dense MLP 的 TP 规则：

```python
gate_proj.parallelize(tp_mesh, dim=0)
up_proj.parallelize(tp_mesh, dim=0)
down_proj.parallelize(tp_mesh, dim=1)
```

forward 布局：

```text
x:          [b, s / tp, h]
tp gather: [b, s, h]
gate/up:   [b, s, intermediate / tp]
down:      [b, s, h]
tp scatter:[b, s / tp, h]
```

row-parallel `down_proj` 的 replicated bias 在 scatter 之后添加。

## 8. MHA

MHA 的 TP 规则：

- Q projection：col-parallel，切 out dim。
- O projection：row-parallel，切 in dim。
- sinks：沿 head dim 切。
- Q/K norm 不切。
- K/V projection 根据 KV head 数选择 normal shard 或 KV replication。

forward 布局：

```text
hidden:        [b, s / tp, h]
tp gather:     [b, s, h]
q_proj:        [b, s, q_heads / tp, d]
k/v_proj:      [b, s, local_kv_heads, d]
RoPE:          使用 Dense 传入的 full-seq cos/sin
attention:     本地 attention
o_proj:        [b, s, h]
tp scatter:    [b, s / tp, h]
```

MHA 内部保留 Ulysses all-to-all 代码路径，但会区分 `seq_ctx.sequence_parallel_mesh` 是否就是
`self._tp_mesh`。如果是 TP-SP mesh，则不能把它当作 Ulysses mesh 做 all-to-all。
当前 Trainer 已经禁止 TP-SP 与 Ulysses SP 同时开启，这个判断只是避免误用 mesh 语义。

## 9. KV Replication

KV head 不够 TP 切时，使用 contiguous KV replication。

约束：

- `tp_size <= num_attention_heads`
- `num_attention_heads % tp_size == 0`
- K/V 若 `num_key_value_heads % tp_size == 0`，直接普通 col-parallel shard。
- K/V 若 `tp_size % num_key_value_heads == 0`，使用 KV replication。
- 其他组合报错。

KV replication 的 mesh：

```python
kv_mesh = DeviceMesh(
    tp_mesh.device_type,
    tp_mesh.mesh.reshape(num_key_value_heads, tp_size // num_key_value_heads),
    mesh_dim_names=("kv_shard", "kv_replica"),
)
placements = [Shard(0), Replicate()]
```

以 `tp_size=8, num_key_value_heads=2` 为例：

- 先按 KV head shard 成 2 份。
- 每个 KV head 在 4 个 contiguous TP ranks 上 replicate。
- 每个 rank 只持有自己对应的 KV head shard。
- forward 得到的 K/V state 已经是本 rank 应该使用的 local K/V，不需要再手动 narrow。

不使用 interleaved replicate。contiguous layout 与 grouped-query attention 的 Q head 分组一致，
更容易保持 Q/K/V 的局部 head 对齐关系。

## 10. Dense.forward 端到端布局

TP-SP 模式下：

```text
seq_ctx.input_ids:     [b, s / tp]
seq_ctx.position_ids:  [b, s / tp]
seq_ctx.raw_position_ids: [b, s_padded]

embed:                 [b, s / tp, h]
rotary_emb(raw ids):   cos/sin [b, s_padded, rope_dim]

for each layer:
  input_layernorm:     local seq
  MHA:                 gather -> compute -> scatter
  residual:            local seq
  post_attn_layernorm: local seq
  MLP:                 gather -> compute -> scatter
  residual:            local seq

final norm:            local seq
lm_head/loss:          local seq
```

注意：Dense 不在最后做 TP gather。TP 训练语义与现有 SP 训练语义一致，loss 在各 rank 的 local
sequence shard 上计算，再通过 loss calibration / distributed reduction 聚合。

## 11. Load Spec 与权重 I/O

`refactor-load-spec` 已经把参数切分描述从具体并行名中解耦出来。Dense TP 只依赖当前参数
DTensor layout：

- Dense `parallelize()` 后调用 `_init_load_spec()`，记录 TP-only layout。
- `BaseModel.fully_shard()` 后重新调用 `_init_load_spec()`，记录 FSDP/HSDP + TP 的当前 layout。
- HF load / save 不应该硬编码“这是 TP”或“这是 FSDP”，只按参数实际 placements 计算本 rank
  的 local tensor 范围。

KV replication 也是参数 layout 的一部分：K/V weight 可能是 2-D DTensor
`[Shard(0), Replicate()]`。加载逻辑应按 DTensor placement 还原 local shard，不在 attention
forward 里补 narrow。

## 12. Unsupported Checks

入口需要显式报错，不能静默降级：

- `tp_size < 1`
- `fsdp_cfg.tp_size > 1`
- `world_size % tp_size != 0`
- `tp_size > 1 and sp_size > 1`
- `tp_size > num_attention_heads`
- `num_attention_heads % tp_size != 0`
- `num_key_value_heads` 既不能被 `tp_size` 整除，也不能整除 `tp_size`
- Dense TP + float8
- Dense TP + MLA
- Dense TP + GatedDeltaNet / linear attention

HSDP + TP 还要求：

```text
world_size % (hsdp_sharding_size * tp_size) == 0
```

## 13. 修改文件清单

核心实现：

- `xtuner/v1/model/base.py`：`TransformerConfig.tp_size`
- `xtuner/v1/model/dense/dense.py`：Dense 创建 TP mesh，初始化期 parallelize，TP-aware FSDP/HSDP mesh
- `xtuner/v1/module/linear/linear.py`：普通 col/row `_Linear.parallelize`
- `xtuner/v1/module/decoder_layer/dense_decoder_layer.py`：DenseMLP / DenseDecoderLayer TP
- `xtuner/v1/module/attention/mha.py`：MHA TP、KV replication、TP-SP forward
- `xtuner/v1/data_proto/sequence_context.py`：`raw_position_ids`
- `xtuner/v1/ops/tensor_parallel/sp_collectives.py`：TP-SP gather/scatter
- `xtuner/v1/train/trainer.py`：从 model config 读取 TP size，禁止 TP-SP + Ulysses SP
- `xtuner/v1/engine/train_engine.py`：`data_replicate_size` 从 model config 读取 TP size

## 14. 验证计划

必须覆盖：

- `tp_size=1` 行为回归。
- `tp_size>1` Dense forward / loss 对齐。
- `num_key_value_heads < tp_size` 的 KV replication。
- `tp_size>1 and sp_size>1` 抛 `ParallelConfigException`。
- `Dense TP + MLA / GatedDeltaNet / float8` 抛 `NotImplementedError`。
- HF load / save 在 TP-only 和 FSDP+TP 下的 local shard 语义。

当前已做的轻量检查不替代分布式验证：

- 相关 Python 文件 `compile()`
- `git diff --check`
