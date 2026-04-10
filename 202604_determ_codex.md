# MHA + torch.compile + FSDP2 Non-Determinism: Root Cause Update

## Summary

`torch.compile(mha, fullgraph=True)` + FSDP2 + multi-GPU 在
`Qwen3_5_VLTextMoE35BA3BConfig` 的 MHA 路径上出现跨 launch 梯度非确定性，
漂移集中在 `k_proj.weight`。

当前最强结论：

**根因不是 FSDP2 reduce-scatter 在 COMM stream 上读到了半写的 flat grad。**
真正的第一分歧点在 compiled backward 内部：

```python
buf24 = reinterpret_tensor(mm_1, (1, 65536, 2, 256), (33554432, 512, 256, 1), 0)
triton_red_fused__to_copy_add_clone_div_mul_pow_sum_5.run(
    buf24, buf11, primals_6, rsqrt_1, ...
)
extern_kernels.mm(reinterpret_tensor(buf24, (512, 65536), (1, 512), 0), view, out=buf25)
```

在 `inductor_cfg.inplace_buffers=True` 时，Inductor 把 saved K activation
`mm_1` 原地复用为 k-norm backward 输出 `buf24`。新 probe 显示：

- 目标 Triton kernel 的上游输入 `buf11_input` 跨 launch 完全一致；
- 目标 Triton kernel 的 in-place 源 buffer `buf24_before` / saved `mm_1`
  跨 launch 完全一致；
- `rsqrt_1` 跨 launch 完全一致；
- 但这个 in-place Triton kernel 写出的 `buf24_after` 会跨 launch 漂；
- `buf24` 随后进入 `k_proj.weight.grad` GEMM，GEMM 输出漂移；
- FSDP2 的 reduce-scatter 只是把某个 rank-local 的错误 `k_proj` full grad 传播到各 rank 的 sharded grad。

实用 workaround 仍然是：

```python
import torch._inductor.config as inductor_cfg
inductor_cfg.inplace_buffers = False
```

这必须在 `torch.compile(...)` 前设置。

## Updated Root Cause

### What Changed

旧文档的 best guess 是：

> compiled backward 和 FSDP2 reduce-scatter/COMM stream 之间存在 cross-stream race，
> RS 读取 `reduce_scatter_input` 时可能和 stream0 的 grad 写入重叠。

新证据推翻了这个解释的核心部分：

- 真正的 `FSDPParam.unsharded_param` AccumulateGrad 全部发生在
  `FSDPParamGroup.post_backward()` / `reduce_scatter_tensor()` 之前；
- 旧的 `mha_accgrad_order_probe.py` 看到的 `ACCUM_DONE(...)` 是
  `sharded_param` 的 post-accumulate hook，FSDP2 在 reduce-scatter 后手动触发它，
  不能代表 unsharded grad 产生得比 RS 晚；
- `sync after RS` 在重新测试中没有稳定恢复确定性，所以它不应再作为根因证据。

新的根因描述：

**Inductor 的 `inplace_buffers=True` 生成了一个不安全的 in-place reuse 计划：
把 saved K activation `mm_1` 当作 `buf24` 给 k-norm backward Triton kernel 原地写。
即使该 kernel 的显式输入跨 launch 一致，它写出的 `buf24_after` 仍可能不同。
这个差异随后被 k_proj weight-gradient GEMM 放大，并由 FSDP2 reduce-scatter 传播。**

## Key Codegen Difference

Buggy mode (`inplace_buffers=True`):

```python
# saved k_proj activation: mm_1
buf24 = reinterpret_tensor(mm_1, (1, 65536, 2, 256), (33554432, 512, 256, 1), 0)
del mm_1

# in-place: in_out_ptr0 == buf24 == mm_1 storage
triton_red_fused__to_copy_add_clone_div_mul_pow_sum_5.run(
    buf24, buf11, primals_6, rsqrt_1, 131072, 256, stream=stream0
)

# k_proj.weight.grad
extern_kernels.mm(reinterpret_tensor(buf24, (512, 65536), (1, 512), 0), view, out=buf25)
```

Fixed mode (`inplace_buffers=False`):

```python
# buf24 uses another buffer; mm_1 remains read-only input
triton_red_fused__to_copy_add_clone_div_mul_pow_sum_5.run(
    buf11, primals_6, mm_1, rsqrt_1, buf24, ...
)

extern_kernels.mm(reinterpret_tensor(buf24, (512, 65536), (1, 512), 0), view, out=buf25)
```

The important change is not FSDP communication. It is whether k-norm backward writes
the output by mutating `mm_1` in-place.

## Evidence

除特别说明外，下面的 probe 都用最后 4 张 GPU 和 `fla` 环境执行：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=. XTUNER_DETERMINISTIC=true \
OMP_NUM_THREADS=1 XTUNER_USE_FA3=0 NCCL_ALGO=Ring NCCL_PROTO=Simple \
NCCL_NUM_CHANNELS=1 CUBLAS_WORKSPACE_CONFIG=:16:8 \
/mnt/shared-storage-user/zhaopenghao/miniconda3/envs/fla/bin/torchrun \
  --standalone --nproc-per-node 4 ...
```

### 1. FSDP unsharded-grad ordering is not the race

New probe:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=. XTUNER_DETERMINISTIC=true \
OMP_NUM_THREADS=1 XTUNER_USE_FA3=0 NCCL_ALGO=Ring NCCL_PROTO=Simple \
NCCL_NUM_CHANNELS=1 CUBLAS_WORKSPACE_CONFIG=:16:8 \
/mnt/shared-storage-user/zhaopenghao/miniconda3/envs/fla/bin/torchrun \
  --standalone --nproc-per-node 4 \
  tests/profiler/mha_fsdp_unsharded_order_probe.py --seq-len 65536
```

Buggy mode output order:

```text
[00] UNSHARDED_ACCUM_DONE(q_proj.weight, ...)
[01] UNSHARDED_ACCUM_DONE(v_proj.weight, ...)
[02] UNSHARDED_ACCUM_DONE(o_proj.weight, ...)
[03] UNSHARDED_ACCUM_DONE(k_norm.weight, ...)
[04] UNSHARDED_ACCUM_DONE(k_proj.weight, ...)
[05] UNSHARDED_ACCUM_DONE(q_norm.weight, ...)
[06] POST_BWD_ENTRY(present=[all 6 params], missing=[], stream=0)
[07] RS_FIRE(...)
[08] SHARDED_POST_ACCUM_HOOK(q_proj.weight)
[09] SHARDED_POST_ACCUM_HOOK(k_proj.weight)
...
```

Fixed mode showed the same ordering.

Interpretation:

- FSDP sees all unsharded grads present before reduce-scatter.
- The old `ACCUM_DONE` ordering was observing sharded-param hooks after RS, not the true
  autograd AccumulateGrad that creates the unsharded grads.

### 2. Pre-RS input already differs, and only in k_proj

Probe:

```bash
torchrun --standalone --nproc-per-node 4 \
  tests/profiler/mha_pre_rs_grad_probe.py \
  --seq-len 65536 --deterministic --save-dir /tmp/codex_pre_rs_a

torchrun --standalone --nproc-per-node 4 \
  tests/profiler/mha_pre_rs_grad_probe.py \
  --seq-len 65536 --deterministic \
  --save-dir /tmp/codex_pre_rs_b --compare /tmp/codex_pre_rs_a
```

Observed in one buggy comparison:

```text
rank 0: rs0 IDENTICAL
rank 1: rs0 n_differ=3108/27263488 max_abs_diff=2.0000e+00 DIFFERS
rank 2: rs0 IDENTICAL
rank 3: rs0 IDENTICAL
```

Mapping the flat reduce input back to FSDP param slices:

```text
rank1: n=3108 max=2.0
  k_proj.weight: count=3108
```

All differing elements were in `k_proj.weight`; `q_proj`, `v_proj`, `o_proj`,
`q_norm`, and `k_norm` were clean.

Fixed mode (`--no-inplace-buffers`) cross-launch:

```text
rank 0/1/2/3: rs0 IDENTICAL
```

Interpretation:

- Non-determinism is already present before NCCL reduce-scatter.
- The local full `k_proj.weight.grad` is the corrupted object.
- FSDP communication is not the first source of the drift.

### 3. `sync after RS` is not a reliable fix

Re-tested:

```bash
torchrun --standalone --nproc-per-node 4 \
  tests/profiler/test_rs_sync.py --seq-len 65536 \
  --sync-mode rs --save-dir /tmp/codex_rs_sync_b \
  --compare /tmp/codex_rs_sync_a
```

Observed:

```text
rank 0: n_differ=155/262144 max_abs_diff=5.0000e-01 DIFFERS
rank 1: n_differ=127/262144 max_abs_diff=1.0000e+00 DIFFERS
rank 2: n_differ=145/262144 max_abs_diff=1.0000e+00 DIFFERS
rank 3: n_differ=120/262144 max_abs_diff=1.0000e+00 DIFFERS
```

Interpretation:

- The old statement “sync after RS restores determinism” is not stable in this environment.
- Since the pre-RS input can already differ, syncing after RS cannot be the root fix.

### 4. k_proj GEMM input already differs

New probe:

```bash
torchrun --standalone --nproc-per-node 4 \
  tests/profiler/mha_kproj_mm_probe.py \
  --seq-len 65536 --save-dir /tmp/codex_kproj_mm_a

torchrun --standalone --nproc-per-node 4 \
  tests/profiler/mha_kproj_mm_probe.py \
  --seq-len 65536 \
  --save-dir /tmp/codex_kproj_mm_b --compare /tmp/codex_kproj_mm_a
```

Observed in buggy mode:

```text
rank 1 input:  n_differ=935/33554432  max_abs_diff=1.5625e-02 DIFFERS
rank 1 output: n_differ=3108/1048576  max_abs_diff=2.0000e+00 DIFFERS

rank 0/2/3 input:  IDENTICAL
rank 0/2/3 output: IDENTICAL
```

The final `kgrad` can differ on all ranks because reduce-scatter reduces and shards
the bad rank-local full grad into every rank's local shard.

Interpretation:

- The `k_proj.weight.grad` GEMM is not the first divergence point.
- Its input `buf24` is already different.
- The GEMM amplifies the small `buf24` differences into larger gradient differences.

### 5. First observed divergence: in-place k-path Triton kernel output

New probe:

```bash
torchrun --standalone --nproc-per-node 4 \
  tests/profiler/mha_kpath_triton_probe.py \
  --seq-len 65536 --save-dir /tmp/codex_kpath_before_20260410_a

torchrun --standalone --nproc-per-node 4 \
  tests/profiler/mha_kpath_triton_probe.py \
  --seq-len 65536 \
  --save-dir /tmp/codex_kpath_before_20260410_b \
  --compare /tmp/codex_kpath_before_20260410_a
```

Buggy mode, after updating the probe to snapshot `buf24_before` before the
target Triton launch:

```text
rank 0/1/2/3 buf11_input: IDENTICAL
rank 0/1/2/3 buf24_before: IDENTICAL
rank 0/1/2/3 rsqrt_1:     IDENTICAL

rank 0 buf24_after: n_differ=926/33554432 max_abs_diff=7.8125e-03 DIFFERS
rank 1 buf24_after: n_differ=935/33554432 max_abs_diff=1.5625e-02 DIFFERS
rank 2 buf24_after: n_differ=949/33554432 max_abs_diff=7.8125e-03 DIFFERS
rank 3 buf24_after: IDENTICAL
```

Adding a sync immediately before the target Triton kernel did not eliminate the issue:

```bash
torchrun --standalone --nproc-per-node 4 \
  tests/profiler/mha_kpath_triton_probe.py \
  --seq-len 65536 --sync-before-target \
  --save-dir /tmp/codex_kpath_sync_b --compare /tmp/codex_kpath_sync_a
```

Observed:

```text
buf11_input: IDENTICAL on all ranks
rsqrt_1:     IDENTICAL on all ranks
buf24_after: still DIFFERS on rank 0 and rank 3 in that run
```

Fixed mode:

```bash
torchrun --standalone --nproc-per-node 4 \
  tests/profiler/mha_kpath_triton_probe.py \
  --seq-len 65536 --no-inplace-buffers \
  --save-dir /tmp/codex_kpath_fixed_b --compare /tmp/codex_kpath_fixed_a
```

Observed:

```text
rank 0/1/2/3 buf11_input: IDENTICAL
rank 0/1/2/3 rsqrt_1:     IDENTICAL
rank 0/1/2/3 buf24_after: IDENTICAL
```

Interpretation:

- The in-place source buffer `buf24_before` / saved `mm_1`, the upstream
  dK-like input, and RMSNorm metadata are deterministic.
- The in-place Triton norm-backward kernel is the first observed divergence point.
- Out-of-place codegen (`inplace_buffers=False`) removes that divergence.

## Final Mechanism

The failure path is:

```text
forward k_proj output
  -> saved as mm_1
  -> Inductor reuses mm_1 storage as in-place buf24
  -> k-path norm backward Triton kernel writes buf24 non-deterministically
     even though buf24_before, buf11_input, and rsqrt_1 match
  -> k_proj.weight.grad GEMM consumes buf24 and amplifies the drift
  -> FSDP2 chunk_cat packs k_proj full grad into reduce_scatter_input
  -> reduce_scatter propagates the bad local full grad into sharded grads
```

This explains the observed fingerprint:

- `--no-compile`: no Inductor in-place compiled backward plan;
- `world_size=1`: no multi-rank FSDP propagation and the existing repro is deterministic;
- `seq_len=65535`: did not hit the same generated kernel/reuse trigger in the existing repro;
- `seq_len=65536`: hits the in-place `mm_1 -> buf24` plan and trigger shape;
- `--no-inplace-buffers`: out-of-place `buf24` plan, deterministic.

## Practical Fix

Use the minimal confirmed workaround:

```python
import torch._inductor.config as inductor_cfg

# Work around a TorchInductor non-determinism bug in compiled MHA backward:
# with inplace_buffers=True, Inductor reuses saved k_proj activation `mm_1`
# as an in-place output buffer (`buf24`) for the k-path norm-backward Triton
# kernel. That kernel can produce cross-launch-different `buf24`, which then
# corrupts k_proj.weight.grad and is propagated by FSDP2 reduce-scatter.
inductor_cfg.inplace_buffers = False
```

More conservative:

```python
inductor_cfg.allow_buffer_reuse = False
inductor_cfg.inplace_buffers = False
```

So far, `inplace_buffers=False` alone is sufficient.

## Investigation Scripts Added

| File | Purpose |
|------|---------|
| `tests/profiler/mha_fsdp_unsharded_order_probe.py` | Distinguish true `unsharded_param` AccumulateGrad from FSDP sharded post-accumulate hooks; rules out the “RS reads before unsharded grad exists” explanation. |
| `tests/profiler/mha_kproj_mm_probe.py` | Snapshots the input/output of the `k_proj.weight.grad` GEMM inside compiled backward; shows GEMM input `buf24` can already differ. |
| `tests/profiler/mha_kpath_triton_probe.py` | Snapshots k-path norm-backward Triton kernel inputs, `buf24_before`, and output; identifies `buf24_after` as the first observed divergence point and verifies fixed out-of-place mode. |

## Notes

- Rank IDs that show the first local divergence vary across launches/probes. That is expected for this class of nondeterminism and does not change the mechanism.
- Existing allocator/alias probes remain useful as negative evidence: RS output did not alias `mm_1`, and AG output did not overlap `mm_1`.
- The old “cross-stream FSDP reduce-scatter race” hypothesis should be treated as superseded unless a future probe shows a more precise alias/race below the in-place Triton kernel itself.
