# MHA + torch.compile + FSDP2 Determinism Root Cause

## 结论

`torch.compile(mha, fullgraph=True)` + FSDP2 + 4 GPU 在
`Qwen3_5_VLTextMoE35BA3BConfig` 的 MHA backward 上出现跨 run 不确定性，第一
个已定位分歧点是 k-path RMSNorm backward 的 Triton reduction kernel：

```python
triton_red_fused__to_copy_add_clone_div_mul_pow_sum_5
```

真正根因不是 FSDP reduce-scatter race，也不是同一个 Triton launcher 对同一份
输入随机算错。更精确的机制是：

1. `torch.use_deterministic_algorithms(True)` 会让 Inductor 常规 reduction
   autotune 退化成单个候选 config：`XBLOCK=32, R0_BLOCK=128`。
2. 但 `torch._inductor.config.dynamic_scale_rblock=True` 会在编译后根据寄存器占用
   再追加一个缩小版 config：`XBLOCK=32, R0_BLOCK=64`。
3. `CachingAutotuner.run()` 看到多个 launcher 后仍会 runtime benchmark，然后用
   timing 最小的 launcher；这个选择跨 rank / run 可变。
4. `R0_BLOCK=64` 和 `R0_BLOCK=128` 对同一个 reduction 使用不同分块和累加顺序。
   fp32 reduction 后再转 bf16，因此少量元素会出现 1 ULP 级别差异。
5. `buf24_after` 随后进入 `k_proj.weight.grad` GEMM，差异被放大，再由 FSDP
   reduce-scatter 传播到各 rank 的 sharded grad。

所以这是一个 Inductor deterministic 语义层面的 bug：deterministic 模式下仍允许
`dynamic_scale_rblock` 引入多个 reduction launcher 并 runtime autotune。差异本身
是浮点 reduction 顺序导致的数值差异，不是同一 launcher 的随机计算错误。

## 最小修复

更直接的 workaround 是在 `torch.compile(...)` 前关闭 dynamic rblock scaling：

```python
import torch._inductor.config as inductor_cfg

inductor_cfg.dynamic_scale_rblock = False
```

也可以在 Python 进程启动前设置：

```bash
TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0
```

之前验证过的 workaround 仍然有效：

```python
inductor_cfg.inplace_buffers = False
```

但它是间接规避：`inplace_buffers=False` 下候选 config 仍然是
`[R0_BLOCK=128, R0_BLOCK=64]`，只是当前实验中 runtime benchmark 稳定选了
`R0_BLOCK=128`。`dynamic_scale_rblock=False` 会直接把候选 config 收敛成单个
`R0_BLOCK=128`。

## 为什么之前看起来是“同输入不同输出”

旧 probe 已确认以下显式输入跨 run 完全一致：

- `buf11_input`
- `buf24_before` / saved `mm_1`
- `norm_weight` / `primals_6`
- `rsqrt_1`

但当时没有同时记录目标 Triton launcher 的实际 config，因此现象被误读成“同一个
kernel 对同一份输入产生不同输出”。

补上 `--print-target-config` 后，分歧完全对应 config 变化：

```text
Run A:
rank1 R0_BLOCK=128
rank3 R0_BLOCK=64
rank0 R0_BLOCK=128
rank2 R0_BLOCK=128

Run B:
rank1 R0_BLOCK=64
rank2 R0_BLOCK=64
rank3 R0_BLOCK=64
rank0 R0_BLOCK=128
```

比较结果：

```text
rank 1 inputs:       IDENTICAL
rank 1 buf24_after:  n_differ=935/33554432 max_abs_diff=1.5625e-02

rank 2 inputs:       IDENTICAL
rank 2 buf24_after:  n_differ=949/33554432 max_abs_diff=7.8125e-03

rank 0 config 128 -> 128, buf24_after IDENTICAL
rank 3 config 64  -> 64,  buf24_after IDENTICAL
```

因此，“同输入不同输出”的缺失变量是 `R0_BLOCK` config；显式 tensor 输入相同，但实际
执行的 compiled launcher 不同。

## 关键 Inductor 机制

本地 PyTorch 代码路径：

```text
/mnt/shared-storage-user/zhaopenghao/miniconda3/envs/fla/lib/python3.12/site-packages/torch/_inductor/runtime/triton_heuristics.py
```

相关逻辑：

```python
def disable_pointwise_autotuning(inductor_meta):
    if inductor_meta.get("are_deterministic_algorithms_enabled"):
        return True
    return not inductor_meta.get("autotune_pointwise", True)
```

这会让 deterministic 模式下的 reduction 初始候选退化为：

```python
if disable_pointwise_autotuning(inductor_meta):
    return [make_config(32, 128)]
```

但随后：

```python
self._make_launchers()
self._dynamic_scale_rblock()
```

`_dynamic_scale_rblock()` 会把 largest `R*_BLOCK` 除以 2，追加新 config：

```text
compile_configs=[
  {'XBLOCK': 32, 'R0_BLOCK': 128},
  {'XBLOCK': 32, 'R0_BLOCK': 64},
]
```

`CachingAutotuner.run()` 发现 `len(self.launchers) > 1` 后会：

```python
self.autotune_to_one_config(*args, **kwargs)
```

也就是在 deterministic 模式下仍然按 benchmark timing 选 launcher。

## 证据

下面命令均使用最后 4 张 GPU 和 `fla` 环境对应的 Python：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=. XTUNER_DETERMINISTIC=true \
OMP_NUM_THREADS=1 XTUNER_USE_FA3=0 NCCL_ALGO=Ring NCCL_PROTO=Simple \
NCCL_NUM_CHANNELS=1 CUBLAS_WORKSPACE_CONFIG=:16:8 \
/mnt/shared-storage-user/zhaopenghao/miniconda3/envs/fla/bin/torchrun \
  --standalone --nproc-per-node 4 ...
```

### 1. 默认 dynamic_scale_rblock=ON 会产生两个候选 config

```bash
tests/profiler/mha_kpath_triton_probe.py \
  --seq-len 65536 \
  --sync-after-target \
  --print-target-config \
  --save-dir /tmp/codex_kpath_dynamic_on_latest_a
```

输出：

```text
target_config rank=1 kwargs={'XBLOCK': 32, 'R0_BLOCK': 128}
  compile_configs=[{'XBLOCK': 32, 'R0_BLOCK': 128}, {'XBLOCK': 32, 'R0_BLOCK': 64}]
target_config rank=3 kwargs={'XBLOCK': 32, 'R0_BLOCK': 64}
  compile_configs=[{'XBLOCK': 32, 'R0_BLOCK': 128}, {'XBLOCK': 32, 'R0_BLOCK': 64}]
```

第二次 run 和第一次比较：

```text
target_config rank=1 kwargs={'XBLOCK': 32, 'R0_BLOCK': 64}
target_config rank=2 kwargs={'XBLOCK': 32, 'R0_BLOCK': 64}
target_config rank=3 kwargs={'XBLOCK': 32, 'R0_BLOCK': 64}
target_config rank=0 kwargs={'XBLOCK': 32, 'R0_BLOCK': 128}

rank 1 buf24_after: n_differ=935/33554432 max_abs_diff=1.5625e-02 DIFFERS
rank 2 buf24_after: n_differ=949/33554432 max_abs_diff=7.8125e-03 DIFFERS
rank 0 buf24_after: IDENTICAL
rank 3 buf24_after: IDENTICAL
```

变化 config 的 rank 发生输出差异；config 不变的 rank 输出一致。

### 2. dynamic_scale_rblock=OFF 后只剩一个 config，输出稳定

```bash
tests/profiler/mha_kpath_triton_probe.py \
  --seq-len 65536 \
  --sync-after-target \
  --print-target-config \
  --no-dynamic-scale-rblock \
  --save-dir /tmp/codex_kpath_no_dynamic_arg_a

tests/profiler/mha_kpath_triton_probe.py \
  --seq-len 65536 \
  --sync-after-target \
  --print-target-config \
  --no-dynamic-scale-rblock \
  --save-dir /tmp/codex_kpath_no_dynamic_arg_b \
  --compare /tmp/codex_kpath_no_dynamic_arg_a
```

输出：

```text
target_config rank=0 kwargs={'XBLOCK': 32, 'R0_BLOCK': 128}
  compile_configs=[{'XBLOCK': 32, 'R0_BLOCK': 128}]
target_config rank=1 kwargs={'XBLOCK': 32, 'R0_BLOCK': 128}
  compile_configs=[{'XBLOCK': 32, 'R0_BLOCK': 128}]
target_config rank=2 kwargs={'XBLOCK': 32, 'R0_BLOCK': 128}
  compile_configs=[{'XBLOCK': 32, 'R0_BLOCK': 128}]
target_config rank=3 kwargs={'XBLOCK': 32, 'R0_BLOCK': 128}
  compile_configs=[{'XBLOCK': 32, 'R0_BLOCK': 128}]

rank 0/1/2/3 buf11_input:   IDENTICAL
rank 0/1/2/3 buf24_before:  IDENTICAL
rank 0/1/2/3 norm_weight:   IDENTICAL
rank 0/1/2/3 rsqrt_1:       IDENTICAL
rank 0/1/2/3 buf24_after:   IDENTICAL
```

### 3. 完整 numerics_test 也被 dynamic_scale_rblock=OFF 稳住

为了排除“只在 kernel probe 里稳定”的可能性，完整跑了两次
`tests/profiler/numerics_test.py`，并显式保持 `inplace_buffers=True`：

```bash
TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0 \
XTUNER_COMPILE_NO_INPLACE_BUFFERS=0 \
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=. XTUNER_DETERMINISTIC=true \
OMP_NUM_THREADS=1 XTUNER_USE_FA3=0 NCCL_ALGO=Ring NCCL_PROTO=Simple \
NCCL_NUM_CHANNELS=1 CUBLAS_WORKSPACE_CONFIG=:16:8 \
QWEN35_MOE_PATH=/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B \
/mnt/shared-storage-user/zhaopenghao/miniconda3/envs/fla/bin/torchrun \
  --standalone --nproc-per-node 4 \
  tests/profiler/numerics_test.py \
  --record-path /tmp/codex_numerics_dynamic_off_full_a \
  --seq-len 65536 \
  --deterministic

TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0 \
XTUNER_COMPILE_NO_INPLACE_BUFFERS=0 \
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=. XTUNER_DETERMINISTIC=true \
OMP_NUM_THREADS=1 XTUNER_USE_FA3=0 NCCL_ALGO=Ring NCCL_PROTO=Simple \
NCCL_NUM_CHANNELS=1 CUBLAS_WORKSPACE_CONFIG=:16:8 \
QWEN35_MOE_PATH=/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B \
/mnt/shared-storage-user/zhaopenghao/miniconda3/envs/fla/bin/torchrun \
  --standalone --nproc-per-node 4 \
  tests/profiler/numerics_test.py \
  --record-path /tmp/codex_numerics_dynamic_off_full_b \
  --compare /tmp/codex_numerics_dynamic_off_full_a \
  --seq-len 65536 \
  --deterministic
```

输出：

```text
[Rank 0] all 72 grad shard sums identical to run 1
[Rank 1] all 72 grad shard sums identical to run 1
[Rank 2] all 72 grad shard sums identical to run 1
[Rank 3] all 72 grad shard sums identical to run 1

RESULT: FULLY DETERMINISTIC — all gradient shard sums identical
across both process invocations on every rank.
```

这说明不需要关闭 `inplace_buffers` 也能稳定最终梯度；直接关闭
`dynamic_scale_rblock` 足够覆盖当前 repro。

### 4. 直接 replay 两个 launcher 可复现相同差异

新增 replay 参数：

```bash
tests/profiler/mha_kpath_triton_replay.py --all-configs
```

用同一份 rank1 输入直接跑两个 precompiled launcher：

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. \
/mnt/shared-storage-user/zhaopenghao/miniconda3/envs/fla/bin/python \
  tests/profiler/mha_kpath_triton_replay.py \
  --output-code /tmp/codex_check_kpath_keep_trace_a/inductor_trace/rank0/torchinductor/model__0_backward_3.1/output_code.py \
  --save-dir /tmp/codex_kpath_dynamic_on_latest_a \
  --rank 1 \
  --mode inplace \
  --device cuda:0 \
  --all-configs
```

输出：

```text
launcher 0: kwargs={'XBLOCK': 32, 'R0_BLOCK': 128}
launcher 1: kwargs={'XBLOCK': 32, 'R0_BLOCK': 64}
launcher 1: vs_launcher0 n_differ=935/33554432 max_abs_diff=1.5625e-02
```

这个差异量和跨 run 里 rank1 从 `128 -> 64` 时的 `buf24_after` 差异完全一致。

### 5. inplace_buffers=False 为什么也稳定

`--no-inplace-buffers` 下目标 kernel 改为 out-of-place：

```python
triton_red_fused__to_copy_add_clone_div_mul_pow_sum_5.run(
    buf11, primals_6, mm_1, rsqrt_1, buf24, 131072, 256, stream=stream0
)
```

而默认 in-place 是：

```python
buf24 = reinterpret_tensor(mm_1, ...)
triton_red_fused__to_copy_add_clone_div_mul_pow_sum_5.run(
    buf24, buf11, primals_6, rsqrt_1, 131072, 256, stream=stream0
)
```

out-of-place 模式下候选 config 仍然是：

```text
compile_configs=[
  {'XBLOCK': 32, 'R0_BLOCK': 128},
  {'XBLOCK': 32, 'R0_BLOCK': 64},
]
```

但两次 A/B 里 4 个 rank 都稳定选了 `R0_BLOCK=128`，因此：

```text
rank 0/1/2/3 buf24_after: IDENTICAL
```

所以 `inplace_buffers=False` 是有效 workaround，但它不是最小根因开关；它改变了
generated code 和 benchmark 环境，使当前 case 不再跨 run 选择不同 reduction
config。

## 数值精度还是计算错误？

结论：

- 对同一个固定 launcher，standalone replay 多次运行是确定的。
- `R0_BLOCK=64` 和 `R0_BLOCK=128` 都在算同一个数学表达式，但 reduction 分块不同，
  浮点累加顺序不同，最终 bf16 rounding 后会有少量元素不同。
- 因此差异本身属于浮点 reduction order 的数值差异。
- 但在 `torch.use_deterministic_algorithms(True)` 下，runtime autotune 仍跨 run 选择
  不同 launcher，是 Inductor deterministic 语义 bug。

换句话说：不是“同一份 Triton 代码随机算错”；是“确定性模式下不该发生的 launcher
选择不确定性”，它把一个本来可解释的数值差异暴露成训练结果不确定。

## 更新过的脚本

| 文件 | 用途 |
|------|------|
| `tests/profiler/mha_kpath_triton_probe.py` | 捕获目标 Triton kernel 输入/输出；支持 `--keep-trace`、`--print-target-config`、`--no-dynamic-scale-rblock`。 |
| `tests/profiler/mha_kpath_triton_replay.py` | 导入 `output_code.py` replay 目标 kernel；支持 `--all-configs` 直接比较所有 precompiled launcher。 |
| `tests/profiler/mha_kpath_triton_probe.sh` | 一键跑 dynamic-scale ON/OFF 的 A/B 对比。 |

## 仍被排除的旧假设

- 不是 FSDP2 reduce-scatter 在 COMM stream 上读到了半写 grad：分歧在 pre-RS 的
  `k_proj.weight.grad` 输入之前已经出现。
- 不是 `k_proj.weight.grad` GEMM 第一次引入差异：GEMM 输入 `buf24` 已经不同。
- 不是 saved `mm_1`、`buf11`、`norm_weight`、`rsqrt_1` 的上游不确定：这些输入跨 run
  完全一致。
- 不是同一个 Triton launcher 自身随机：固定 launcher replay 是确定的。
