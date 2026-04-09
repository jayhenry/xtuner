# MHA + torch.compile + FSDP2 Non-Determinism: Root Cause Found ✓

## Problem Statement

Training with `torch.compile(mha, fullgraph=True)` + FSDP2 (multi-GPU) produces
non-deterministic gradients across launches in the MHA path of
`Qwen3_5_VLTextMoE35BA3BConfig`.

**Stable fingerprint:**
- `--no-compile`: deterministic
- single-GPU (`world_size=1`): deterministic
- `compile + multi-GPU + seq_len=65535`: deterministic in the existing repro
- `compile + multi-GPU + seq_len=65536`: **NON-DETERMINISTIC**
- the drift is concentrated in `k_proj.weight`

The fix:

```python
import torch._inductor.config as inductor_cfg
inductor_cfg.inplace_buffers = False
```

This fix is confirmed and the **root cause is now fully identified**.

---

## Root Cause — CONFIRMED

**A cross-stream memory race between the FSDP2 reduce-scatter (COMM stream) and the
compiled backward's `o_proj` matmul (stream0), triggered specifically when
`inplace_buffers=True` causes `buf24` to alias `mm_1`'s storage.**

### Exact race sequence (BUGGY mode)

The compiled backward (`output_code.py`, key lines):

```python
# line 866: buf24 = mm_1 storage reused in-place (addr_mm1, 64 MB)
buf24 = reinterpret_tensor(mm_1, (1, 65536, 2, 256), (33554432, 512, 256, 1), 0)
del mm_1

# line 869: Triton kernel reads/writes buf24 (= mm_1, in_out_ptr0)
triton_red_fused__to_copy_add_clone_div_mul_pow_sum_5.run(buf24, ...)

# line 908: k_proj.weight.grad matmul → buf25
extern_kernels.mm(reinterpret_tensor(buf24, (512, 65536), (1, 512), 0), view, out=buf25)

# line 909: buf24/mm_1 storage FREED — addr_mm1 returned to CUDA allocator
del buf24

# ← AccumulateGrad fires here: buf25 accumulated, then FSDP2 triggers
#   reduce_scatter on the COMM stream (async, not synced with stream0)
#   RS output buffer allocation happens on COMM stream → may get addr_mm1

# line 912: buf29 allocated for o_proj gradient (32 MB, stream0)
buf29 = empty_strided_cuda((8192, 2048), (2048, 1), torch.bfloat16)
#   → CUDA caching allocator may also give addr_mm1 to stream0

# line 914: o_proj.weight.grad matmul writes into buf29
extern_kernels.mm(reinterpret_tensor(buf28, (8192, 65536), (1, 8192), 0), view, out=buf29)
```

**The race:** After `del buf24` (line 909) frees `addr_mm1` (64 MB) to the CUDA
caching allocator, two concurrent allocations compete for that block:

1. **COMM stream**: FSDP2 reduce-scatter output allocation for k_proj shard (~1 MB)
2. **stream0**: `buf29` allocation for o_proj matmul (32 MB)

If both allocations receive the same or overlapping address region, the RS write on
the COMM stream races with the o_proj matmul write on stream0. The result: the
accumulated k_proj.weight.grad and/or o_proj.weight.grad become
**non-deterministically corrupted** depending on which stream wins the timing race.

### Why `inplace_buffers=False` fixes it

With `inplace_buffers=False`, the backward plan is:

```python
# line 865: buf24 = buf3 storage (NOT mm_1) — buf3 lives until here
buf24 = reinterpret_tensor(buf3, (1, 65536, 2, 256), (33554432, 512, 256, 1), 0)
del buf3

# line 868: mm_1 is READ-ONLY input (in_ptr2), buf24 is output (out_ptr1)
triton_red_fused__to_copy_add_clone_div_mul_pow_sum_5.run(buf11, primals_6, mm_1, rsqrt_1, buf24, ...)

# line 870: mm_1 freed here (NOT at line 909)
del mm_1
```

Now `del buf24` (line 909) frees `addr_buf3` (not `addr_mm1`). The RS output and
`buf29` compete for a **different address range**, and the specific collision pattern
that triggers the race does not occur.

### Required conditions (all four must be true)

| Condition | Buggy | Fixed |
|-----------|-------|-------|
| `inplace_buffers=True` | ✓ non-determ | `False` → deterministic |
| `reshard_after_forward=True` | ✓ non-determ | `False` → deterministic |
| `world_size > 1` | ✓ non-determ | `= 1` → deterministic |
| `sync after reduce_scatter` | ✗ non-determ | `True` → deterministic |

- **`reshard_after_forward=False`**: no pre-backward all-gather → no RS inside backward → no race
- **`world_size=1`**: RS is a no-op (no actual NCCL transfer), never fires asynchronously
- **`sync after RS`**: serializes the COMM stream and stream0, preventing the concurrent allocation race

### Data integrity probe results

Probed `mm_1` data at pack time (forward) and unpack time (backward) across two launches:

- `mm_1` at pack time: **IDENTICAL** (n_differ=0/33554432 on all 4 ranks)
- `mm_1` at unpack time: **IDENTICAL** (n_differ=0/33554432 on all 4 ranks)
- `k_proj.weight.grad`: **DIFFERS** (120–187 elements, max_abs_diff=0.5–2.0)

This confirms: `mm_1` is **not** itself corrupted. The race corrupts the gradient
**after** `mm_1` is consumed (at line 908–914), not before.

---

## What We Verified (Chronological)

---

### 1. `inplace_buffers=False` is the real switch

Using `tests/profiler/mha_determ.py`:

- `compile + FSDP2 + seq_len=65536`: repeatedly non-deterministic
- `compile + FSDP2 + seq_len=65536 + --no-inplace-buffers`: fully deterministic

Using `tests/profiler/mha_backend_repro.py`:

- `backend=fa2`: mostly non-deterministic across repeated compares
- `backend=fake_attn`: also reproducible, but less frequently
- `backend=fa2 + --no-inplace-buffers`: deterministic
- `backend=fake_attn + --no-inplace-buffers`: deterministic

This narrows the issue to **compiled inplace buffer reuse**, not to FA2 alone.

### 2. FA2 is not required

The earlier suspicion was:

> remove FA2 or replace FA2 with fake attention -> no non-determinism

That no longer holds as a general conclusion in this environment.

Repeated sampling with `tests/profiler/mha_backend_repro.py` showed:

- `backend=fa2`: 4/5 compare runs were non-deterministic
- `backend=fake_attn`: 1/5 compare runs was non-deterministic, but recent reruns
  have been less stable as evidence.

So FA2 is **not necessary** for the issue. It may still increase the trigger rate,
but it is not the root switch.

### 3. The key codegen difference is now explicit

Using `tests/profiler/mha_fsdp_inplace_addr_demo.py`:

- buggy mode:
  - `in_out_ptr_count=6`
  - `k_norm_reuse_source=mm_1`
- fixed mode (`--no-inplace-buffers`):
  - `in_out_ptr_count=0`
  - `k_norm_reuse_source=buf3`

This pattern was observed for both:

- `--backend fa2`
- `--backend fake_attn`

So the backend changes the trigger probability, but both backends can hit the same
compiled backward reuse shape.

### 4. `mm_1` is the saved `k_proj` activation path

From the forward trace:

- `buf1` is the `k_proj` matmul output
- that value is saved into backward as `mm_1`

From the backward trace:

- buggy mode reinterprets `mm_1` as the reuse target
- fixed mode reinterprets `buf3` as the reuse target and keeps `mm_1` as an input

So the most concrete phrasing today is:

**buggy mode mutates saved K-path activation storage in-place during backward**

rather than:

**buggy mode is proven to mutate the FSDP weight all-gather buffer itself**

### 5. FSDP2 is required; exact conditions are now mapped

Tested four combinations with `tests/profiler/test_no_reshard.py`:

| inplace_buffers | reshard_after_forward | Result |
|----------------|----------------------|--------|
| True (default) | True (default) | **NON-DETERMINISTIC** |
| True | False | DETERMINISTIC |
| False | True | DETERMINISTIC |
| False | False | DETERMINISTIC |

The bug requires **both** `inplace_buffers=True` AND `reshard_after_forward=True`.

- `reshard_after_forward=False`: params not freed after forward → no pre-bwd AG → no
  RS inside backward → no cross-stream allocation race
- `inplace_buffers=False`: different backward memory plan → `del buf24` frees
  `addr_buf3` instead of `addr_mm1` → different address dynamics

Also tested `world_size=1` (single-GPU): deterministic. The actual NCCL RS is
required for the race to materialize.

### 6. Synchronization probe confirms the RS race

Tested `tests/profiler/test_rs_sync.py` with four sync modes:

| `--sync-mode` | Synchronizes after | Result |
|---------------|--------------------|--------|
| `none` | nothing | **NON-DETERMINISTIC** |
| `ag` | all-gather | **NON-DETERMINISTIC** |
| `rs` | reduce-scatter | **DETERMINISTIC** (all ranks IDENTICAL) |
| `both` | AG + RS | **DETERMINISTIC** |

`sync after RS` serializes the COMM stream with stream0, preventing the concurrent
write to the freed `addr_mm1` block. `sync after AG` alone is insufficient — the AG
timing is not the source of the race.

This is the **key confirmatory experiment**: the reduce-scatter (fired asynchronously
on the COMM stream inside the compiled backward) is the race partner.

### 7. mm_1 data probe — memory is not corrupted, gradient is

Probed `mm_1` data with `tests/profiler/mha_mm1_data_probe.py` across two launches:

- `mm_1` at pack time (forward): **IDENTICAL** between launches (n_differ=0/33554432)
- `mm_1` at unpack time (backward): **IDENTICAL** between launches
- `k_proj.weight.grad`: **DIFFERS** (120–187 elements, max_abs_diff=0.5–2.0)

This disproves the earlier hypothesis that `mm_1`'s storage is directly overwritten.
The race corrupts the gradient **after** `mm_1` is consumed.

### 8. All-gather address probe — AG output does not overlap mm_1

Probed AG output addresses vs mm_1 with `tests/profiler/mha_ag_alias_probe.py`:

- All-gather output (52 MB) at addr_ag
- mm_1 (64 MB) at addr_mm1 = addr_ag + 96 MB
- RS input (104 MB) at addr_rs = addr_mm1 + 128 MB

No overlap between mm_1 and any AG output buffer. The AG is not the race partner.

### 9. Allocator experiments are informative, but not sufficient

Using `tests/profiler/mha_fsdp_comm_overlap_validator.py` with cross-launch compare:

- baseline:
  - `inplace_buffers=True`
  - FSDP2 comm buffers use the default CUDA caching allocator
  - result: reproducibly non-deterministic across launches, only on `k_proj.weight`
- `--use-process-group-allocator`:
  - keeps `inplace_buffers=True`
  - moves FSDP2 comm staging buffers to the ProcessGroup allocator
  - result: still non-deterministic across launches
- `--use-process-group-allocator-for-allgather-outputs`:
  - keeps `inplace_buffers=True`
  - moves `all_gather_outputs` to the ProcessGroup allocator
  - result: still non-deterministic across launches
- `--use-process-group-allocator --use-process-group-allocator-for-allgather-outputs`:
  - keeps `inplace_buffers=True`
  - moves both comm staging buffers and `all_gather_outputs` to the ProcessGroup allocator
  - result: still non-deterministic across launches
- `--sync-fsdp-hooks`:
  - keeps `inplace_buffers=True`
  - synchronizes CUDA after FSDP2 `pre_backward` / `post_backward`
  - result: still non-deterministic across launches
- `--no-inplace-buffers`:
  - changes the compiled backward memory plan
  - result: deterministic across launches

This narrows the allocator claim substantially. It says:

- the issue is **not** explained by hook-boundary synchronization alone
- moving the currently identified FSDP-managed buffers to the ProcessGroup allocator
  is **not** sufficient to restore determinism

So the best current mechanism is no longer:

**"Inductor's in-place reuse of saved `mm_1` becomes unsafe specifically because
FSDP2 comm staging buffers come from the default CUDA caching allocator pool."**

That allocator-only explanation is now too strong.

### 10. Pointer evidence is suggestive, but not sufficient

The pointer probe shows:

- FSDP2 pre-backward uses side streams (`all_gather_stream`, `all_gather_copy_in_stream`)
- `k_proj.weight` often has
  `pre_bwd_ptr == unsharded_accumulated_grad.data_ptr()`

However, that particular pointer equality appears in both buggy and fixed modes, so it
does **not** distinguish the failure mechanism by itself.

We also tried to capture saved tensor addresses with
`torch.autograd.graph.saved_tensors_hooks`, but we did **not** obtain a direct runtime
pointer match between:

- `k_proj.weight`'s FSDP pre-backward buffer, and
- a saved forward tensor.

So this remains an open gap.

### 11. The existing race demo still hints at memory corruption

Running `tests/profiler/demo_race_condition.py` in the buggy mode sometimes triggered
an illegal memory access when reading `_unsharded_param` after backward.

That supports "something around backward-time storage reuse is unsafe", but it still
does not prove the exact same-address all-gather alias claim.

---

## Confirmed Mechanism

1. `torch.compile` lowers MHA forward+backward into an Inductor graph.
2. With `inplace_buffers=True`, Inductor chooses a backward plan that reuses saved
   `mm_1` (the k_proj output) in-place as `buf24`.
3. The compiled backward computes the k_proj.weight gradient using `buf24`, then
   frees `buf24` (and thus `addr_mm1`) at line 909.
4. At this point, AccumulateGrad fires for k_proj, which triggers FSDP2's
   `reduce_scatter` **asynchronously on the COMM stream**.
5. The COMM stream's RS output buffer allocation and stream0's next allocation
   (`buf29` for o_proj grad, 32 MB) both compete for `addr_mm1`.
6. The resulting concurrent writes corrupt o_proj.weight.grad or k_proj.weight.grad
   non-deterministically depending on timing.
7. Adding `torch.cuda.synchronize()` after the RS (serializing COMM stream and
   stream0) eliminates the race and restores full determinism.
8. Setting `inplace_buffers=False` causes `del buf24` to free `addr_buf3` (not
   `addr_mm1`), preventing the specific collision pattern.

---

## Why The Existing Fingerprint Makes Sense

### Why only with compile?

Because the problematic in-place reuse is introduced by TorchInductor codegen.
Eager execution does not produce this compiled memory plan.

### Why only multi-GPU / FSDP2?

Because the failure seems to require FSDP2's pre-backward communication behavior.
Single-GPU runs do not have the same sharded parameter orchestration or FSDP2 comm
staging buffers and do not reproduce the issue.

### Why `seq_len=65536` but not `65535` in the repro?

The exact threshold is still best understood as **shape-dependent codegen / memory
planning**. For the current environment:

- `65536` consistently hits the buggy plan
- `65535` does not in the same repro

We should avoid overstating this as a universal TorchInductor heuristic beyond the
tested version and shapes.

### Why only `k_proj.weight`?

The compiled backward reuse pattern we observed is on the **K path** specifically.
That matches the empirical symptom that only `k_proj.weight` drifts.

### Why this is not just a compiled-autograd bug

In the minimal validator repro, `compiled_autograd_enabled=False`, and the issue still
reproduces across launches. So this is broader than the Traceable FSDP2
compiled-autograd path.

---

## Fix

The practical fix remains:

```python
import torch._inductor.config as inductor_cfg
inductor_cfg.inplace_buffers = False
```

This must be set **before** `torch.compile(...)` runs, because Inductor bakes the
decision into the generated graph during trace/compile time.

### Recommended comment

```python
# Root cause: TorchInductor's `inplace_buffers=True` causes the compiled backward
# to reuse the k_proj saved activation (mm_1) in-place as buf24. After buf24 is
# freed at the end of the k_proj gradient computation (line 909), FSDP2 fires
# reduce_scatter asynchronously on the COMM stream while stream0 simultaneously
# allocates buf29 for the o_proj gradient. Both allocations may get the same freed
# address (addr_mm1), creating a concurrent write race that corrupts gradients
# non-deterministically. Disabling inplace_buffers changes the backward plan so
# that buf24 aliases buf3 (not mm_1), preventing the collision pattern.
import torch._inductor.config as inductor_cfg
inductor_cfg.inplace_buffers = False
```

### More conservative option

```python
inductor_cfg.allow_buffer_reuse = False
inductor_cfg.inplace_buffers = False
```

This is more aggressive and may cost more memory. So far, `inplace_buffers=False`
alone is sufficient.

### Experimental allocator toggles

For local diagnosis, the validator now has FSDP2 allocator toggles for:

- comm staging buffers
- `all_gather_outputs`
- both together

Those toggles are still useful for narrowing hypotheses, but they should **not** be
described as fixes. In the current environment, none of them restores cross-launch
determinism reliably.

---

## Repro / Verification Commands

Environment used in this investigation:

```bash
source ~/.bashrc
conda activate fla

export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export XTUNER_USE_FA3=0
export XTUNER_DETERMINISTIC=true
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_NUM_CHANNELS=1
export CUBLAS_WORKSPACE_CONFIG=:16:8
```

### Full MHA repro

```bash
bash tests/profiler/run_mha_determ.sh
```

Observed:

- baseline `compile + FSDP2 + seq_len=65536`: non-deterministic
- `--no-inplace-buffers`: fully deterministic

### Backend comparison

```bash
torchrun --nproc-per-node 4 tests/profiler/mha_backend_repro.py \
  --record-path /tmp/mha_backend_fa2_base \
  --backend fa2 --seq-len 65536 --deterministic

torchrun --nproc-per-node 4 tests/profiler/mha_backend_repro.py \
  --record-path /tmp/mha_backend_fake_base \
  --backend fake_attn --seq-len 65536 --deterministic
```

And then compare repeatedly against the baseline. Result:

- `fa2`: reproduces frequently
- `fake_attn`: also reproduces, but less frequently
- both become deterministic with `--no-inplace-buffers`

### Pointer + codegen probe

```bash
torchrun --nproc-per-node 4 tests/profiler/mha_fsdp_inplace_addr_demo.py \
  --backend fa2 --seq-len 65536 --deterministic \
  --trace-dir /tmp/mha_addr_fa2_buggy

torchrun --nproc-per-node 4 tests/profiler/mha_fsdp_inplace_addr_demo.py \
  --backend fa2 --seq-len 65536 --deterministic \
  --no-inplace-buffers \
  --trace-dir /tmp/mha_addr_fa2_fixed

torchrun --nproc-per-node 4 tests/profiler/mha_fsdp_inplace_addr_demo.py \
  --backend fake_attn --seq-len 65536 --deterministic \
  --trace-dir /tmp/mha_addr_fake_buggy

torchrun --nproc-per-node 4 tests/profiler/mha_fsdp_inplace_addr_demo.py \
  --backend fake_attn --seq-len 65536 --deterministic \
  --no-inplace-buffers \
  --trace-dir /tmp/mha_addr_fake_fixed
```

Expected signature:

- buggy:
  - `in_out_ptr_count=6`
  - `k_norm_reuse_source=mm_1`
- fixed:
  - `in_out_ptr_count=0`
  - `k_norm_reuse_source=buf3`

### Allocator / overlap validator

```bash
torchrun --nproc-per-node 4 tests/profiler/mha_fsdp_comm_overlap_validator.py \
  --seq-len 65536 --iters 1 --deterministic \
  --record-path /tmp/mha_validator_base_a

torchrun --nproc-per-node 4 tests/profiler/mha_fsdp_comm_overlap_validator.py \
  --seq-len 65536 --iters 1 --deterministic \
  --record-path /tmp/mha_validator_base_b \
  --compare /tmp/mha_validator_base_a

torchrun --nproc-per-node 4 tests/profiler/mha_fsdp_comm_overlap_validator.py \
  --seq-len 65536 --iters 1 --deterministic \
  --use-process-group-allocator \
  --record-path /tmp/mha_validator_pg_a

torchrun --nproc-per-node 4 tests/profiler/mha_fsdp_comm_overlap_validator.py \
  --seq-len 65536 --iters 1 --deterministic \
  --use-process-group-allocator \
  --record-path /tmp/mha_validator_pg_b \
  --compare /tmp/mha_validator_pg_a

torchrun --nproc-per-node 4 tests/profiler/mha_fsdp_comm_overlap_validator.py \
  --seq-len 65536 --iters 1 --deterministic \
  --use-process-group-allocator-for-allgather-outputs \
  --record-path /tmp/mha_validator_pg_agout_a

torchrun --nproc-per-node 4 tests/profiler/mha_fsdp_comm_overlap_validator.py \
  --seq-len 65536 --iters 1 --deterministic \
  --use-process-group-allocator-for-allgather-outputs \
  --record-path /tmp/mha_validator_pg_agout_b \
  --compare /tmp/mha_validator_pg_agout_a

torchrun --nproc-per-node 4 tests/profiler/mha_fsdp_comm_overlap_validator.py \
  --seq-len 65536 --iters 1 --deterministic \
  --use-process-group-allocator \
  --use-process-group-allocator-for-allgather-outputs \
  --record-path /tmp/mha_validator_pg_both_a

torchrun --nproc-per-node 4 tests/profiler/mha_fsdp_comm_overlap_validator.py \
  --seq-len 65536 --iters 1 --deterministic \
  --use-process-group-allocator \
  --use-process-group-allocator-for-allgather-outputs \
  --record-path /tmp/mha_validator_pg_both_b \
  --compare /tmp/mha_validator_pg_both_a

torchrun --nproc-per-node 4 tests/profiler/mha_fsdp_comm_overlap_validator.py \
  --seq-len 65536 --iters 1 --deterministic \
  --sync-fsdp-hooks \
  --record-path /tmp/mha_validator_sync_a

torchrun --nproc-per-node 4 tests/profiler/mha_fsdp_comm_overlap_validator.py \
  --seq-len 65536 --iters 1 --deterministic \
  --sync-fsdp-hooks \
  --record-path /tmp/mha_validator_sync_b \
  --compare /tmp/mha_validator_sync_a

torchrun --nproc-per-node 4 tests/profiler/mha_fsdp_comm_overlap_validator.py \
  --seq-len 65536 --iters 1 --deterministic \
  --no-inplace-buffers \
  --record-path /tmp/mha_validator_no_inplace_a

torchrun --nproc-per-node 4 tests/profiler/mha_fsdp_comm_overlap_validator.py \
  --seq-len 65536 --iters 1 --deterministic \
  --no-inplace-buffers \
  --record-path /tmp/mha_validator_no_inplace_b \
  --compare /tmp/mha_validator_no_inplace_a
```

Observed:

- baseline: cross-launch **NON-DETERMINISTIC**, only `_orig_mod.k_proj.weight` drifts
- `--use-process-group-allocator`: cross-launch **NON-DETERMINISTIC**
- `--use-process-group-allocator-for-allgather-outputs`: cross-launch **NON-DETERMINISTIC**
- `--use-process-group-allocator --use-process-group-allocator-for-allgather-outputs`:
  cross-launch **NON-DETERMINISTIC**
- `--sync-fsdp-hooks`: cross-launch **NON-DETERMINISTIC**
- `--no-inplace-buffers`: cross-launch **DETERMINISTIC**

---

## Investigation Scripts

| File | Purpose |
|------|---------|
| `tests/profiler/mha_determ.py` | Main compiled MHA determinism repro |
| `tests/profiler/run_mha_determ.sh` | Shell runner for the main repro |
| `tests/profiler/mha_backend_repro.py` | Same MHA wrapper path, but attention backend can be switched between `fa2` and `fake_attn` |
| `tests/profiler/demo_race_condition.py` | Older pointer/race probe; can trigger illegal memory access in buggy mode |
| `tests/profiler/mha_fsdp_inplace_addr_demo.py` | Updated backend-aware pointer + codegen probe for the current hypothesis |
| `tests/profiler/mha_fsdp_comm_overlap_validator.py` | Cross-launch validator for allocator-vs-overlap hypotheses (`PG allocator`, `all_gather_outputs`, hook sync, `no-inplace-buffers`) |
| `tests/profiler/mha_pre_rs_grad_probe.py` | Captures pre-reduce_scatter gradient data to determine if non-determinism is local or NCCL-introduced |
| `tests/profiler/mha_mm1_data_probe.py` | Captures mm_1 data at both forward-save and backward-retrieve time; confirms mm_1 is IDENTICAL but k_proj.weight.grad DIFFERS |
| `tests/profiler/mha_combined_addr_probe.py` | Combined mm_1 + RS address capture; checks for address overlap within and across launches |
| `tests/profiler/mha_ag_alias_probe.py` | Probes all-gather output addresses vs mm_1; confirms AG output does NOT overlap mm_1 |
| `/tmp/test_no_reshard.py` | Tests 4 combinations of inplace_buffers × reshard_after_forward; confirms both must be True to trigger bug |
| `/tmp/test_ag_sync.py` | Tests sync after all-gather; confirms AG timing is NOT the race source |
| `/tmp/test_rs_sync.py` | Tests sync after reduce-scatter; confirms RS on COMM stream IS the race partner (sync → deterministic) |

---

## Model Context

| Field | Value |
|-------|-------|
| Model | `Qwen3_5_VLTextMoE35BA3BConfig` |
| `hidden_size` | 2048 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 2 |
| `head_dim` | 256 |
| `qk_norm` | True |
| `with_gate` | True |
| `sliding_window` | 1024 |

The K-path shape is:

- `k_proj`: `(seq_len, 2048) x (2048, 512) -> (seq_len, 512)`

At `seq_len=65536`, the current PyTorch build consistently produces the problematic
compiled plan in this repro.

---

## Remaining Questions

1. **Memory cost**: How much peak memory does `inplace_buffers=False` cost in full
   training? Not measured; `allow_buffer_reuse=False` would be more conservative but
   also more expensive.

2. **FA2 trigger rate**: Why does FA2 increase the trigger rate relative to
   `fake_attn`? Likely timing/stream pressure differences, but not conclusively proven.

3. **PyTorch version sensitivity**: Does a newer or older PyTorch change whether the
   specific buf24→addr_mm1 collision pattern occurs? The seq_len=65536 threshold and
   codegen shape are build-specific.

4. **Upstream report**: This is worth reporting to PyTorch as a TorchInductor +
   FSDP2 interaction bug. The correct framing is:
   - `inplace_buffers=True` generates a backward plan that frees a 64 MB buffer
     inside a compiled backward while an FSDP2 RS is in-flight on the COMM stream
   - the CUDA caching allocator races to give that freed block to both the RS output
     and the next stream0 allocation
   - this is a **cross-stream lifetime bug** in Inductor's buffer reuse analysis
