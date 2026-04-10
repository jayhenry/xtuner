# MHA + torch.compile + FSDP2 Non-Determinism: Investigation In Progress

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

The fix (empirically validated):

```python
import torch._inductor.config as inductor_cfg
inductor_cfg.inplace_buffers = False
```

The fix is confirmed to work. **The exact root cause mechanism is still under investigation.**

---

## Current Best Guess for Root Cause

**A cross-stream race involving FSDP2 reduce-scatter (COMM stream) and the compiled
backward (stream0), triggered when `inplace_buffers=True` causes unsafe buffer reuse
in the compiled backward plan.**

### What we know for certain

1. `inplace_buffers=True` causes the compiled backward to reuse `mm_1`'s storage
   as `buf24` (the k-norm computation buffer).
2. `del buf24` at line 909 of the compiled backward frees that storage block.
3. After `del buf24`, FSDP2 fires `reduce_scatter` asynchronously on the COMM stream.
4. The RS INPUT (`reduce_scatter_input`) is **non-deterministic between launches**
   (confirmed by `mha_flatgrad_nosync_probe.py`: 3418/27M elements differ, max_diff=2.0).
5. Without `inplace_buffers=True`, this non-determinism disappears.
6. `sync after RS` (serializing COMM stream and stream0) restores determinism.

### What remains uncertain

The **exact memory aliasing pattern** that creates the race has not been confirmed:

- The original hypothesis was: RS output ↔ mm_1's freed address → race with stream0.
  **This was disproved** by CUDA memory history probe (`mha_buf29_rs_overlap_probe.py`):
  RS output (26 MB float32) is NEVER allocated at `addr_mm1`. Instead, the 32 MB alloc
  at `addr_mm1` is `q_proj.weight.grad` (bf16), not the RS output.
- The RS INPUT (`reduce_scatter_input`) is non-deterministic. The mechanism by which
  `inplace_buffers=True` causes this non-determinism in the RS input is not yet pinned.

### Compiled backward structure (key lines)

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

# ← FSDP2 triggers reduce_scatter on COMM stream (async)
# ← stream0 continues with o_proj gradient computation

# line 912: buf29 for o_proj gradient (32 MB, stream0)
buf29 = empty_strided_cuda((8192, 2048), (2048, 1), torch.bfloat16)

# line 914: o_proj.weight.grad matmul writes into buf29
extern_kernels.mm(reinterpret_tensor(buf28, (8192, 65536), (1, 8192), 0), view, out=buf29)
```

**Observed address reuse (from CUDA memory history probe):**
- `addr_mm1` (64 MB) freed at line 909
- `q_proj.weight.grad` (32 MB bf16) allocated at `addr_mm1` on stream0
- RS output (26 MB float32) allocated from a **different** CUDA segment — no alias with `addr_mm1`

**Current best guess for the race:** The non-determinism in the RS INPUT may arise
from a race between stream0 (writing the unsharded param grads that `chunk_cat` reads
to build `reduce_scatter_input`) and some COMM stream operation that writes to an
overlapping address. The exact address pair has not been captured.

### Why `inplace_buffers=False` fixes it

With `inplace_buffers=False`, the backward plan changes:

```python
# line 865: buf24 = buf3 storage (NOT mm_1) — buf3 lives until here
buf24 = reinterpret_tensor(buf3, (1, 65536, 2, 256), (33554432, 512, 256, 1), 0)
del buf3

# line 868: mm_1 is READ-ONLY input (in_ptr2), buf24 is output (out_ptr1)
triton_red_fused__to_copy_add_clone_div_mul_pow_sum_5.run(buf11, primals_6, mm_1, rsqrt_1, buf24, ...)

# line 870: mm_1 freed here (NOT at line 909)
del mm_1
```

Now `del buf24` frees `addr_buf3` (not `addr_mm1`), changing the allocator dynamics
enough to prevent the race.

### Required conditions (all four must be true)

| Condition | Buggy | Fixed |
|-----------|-------|-------|
| `inplace_buffers=True` | ✓ non-determ | `False` → deterministic |
| `reshard_after_forward=True` | ✓ non-determ | `False` → deterministic |
| `world_size > 1` | ✓ non-determ | `= 1` → deterministic |
| `sync after reduce_scatter` | ✗ non-determ | `True` → deterministic |

- **`reshard_after_forward=False`**: no pre-backward all-gather → no RS inside backward → no race
- **`world_size=1`**: RS is a no-op (no actual NCCL transfer), never fires asynchronously
- **`sync after RS`**: serializes the COMM stream and stream0, preventing the concurrent race

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
race. `sync after AG` alone is insufficient — the AG timing is not the source of the
race.

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

### 12. RS output does NOT alias mm_1 — original hypothesis disproved

Probed RS output and buf29 (o_proj gradient) addresses with CUDA memory history
(`tests/profiler/mha_buf29_rs_overlap_probe.py`, `tests/profiler/mha_rs_buf29_addr_probe.py`):

- RS output (26 MB float32) comes from a **fresh CUDA segment**, never allocated at `addr_mm1`
- The 32 MB allocation at `addr_mm1` (after it is freed at line 909) is
  **q_proj.weight.grad** (bf16), not o_proj or the RS output
- RS output and all param grad buffers are in completely separate address ranges

This rules out the original specific mechanism:
> "COMM stream RS output and stream0 buf29 both race to get addr_mm1"

The address aliasing is partial (q_proj.weight.grad lands at addr_mm1), but the
previously hypothesized RS-output ↔ addr_mm1 collision was not observed.

### 13. RS INPUT is non-deterministic — flat_grad race confirmed

Probed the RS input (`reduce_scatter_input`) at the moment RS fires, WITHOUT
synchronizing stream0 first (`tests/profiler/mha_flatgrad_nosync_probe.py`):

- Cross-launch: **n_differ=3418/27M, max_diff=2.0** → RS INPUT DIFFERS between launches
- This is the `reduce_scatter_input` buffer built by `chunk_cat` from unsharded param grads

Control with synchronize() before clone
(`tests/profiler/mha_flatgrad_at_rs_probe.py`):

- Cross-launch flat_grad: **IDENTICAL** (sync forces stream0 to complete → race masked)
- Cross-launch k_proj.weight.grad: still DIFFERS by ~287 elements (NCCL float32 non-determinism)

This confirms: **the RS input is non-deterministic due to a cross-stream race with
stream0 at RS fire time**. Synchronizing stream0 before RS reads the RS input
eliminates the non-determinism.

### 14. FSDP2 backward workflow — RS triggered by RegisterPostBackwardFunction

Examined FSDP2 source with `compiled_autograd=False` and `skip_fsdp_hooks=True` (default):

**Code path:**

`_fsdp_param_group.py:585` `_register_post_backward_hook()`:
```python
# (not skip_fsdp_hooks) or compiled_autograd_enabled() = False or False = False
# → RegisterPostBackwardFunction IS registered at module input
inp_tensors = RegisterPostBackwardFunction.apply(self, *inp_tensors)  # line 605
```

`_fsdp_param_group.py:766` `RegisterPostBackwardFunction.backward`:
```python
def backward(ctx, *grads):
    ctx.param_group.post_backward()   # triggers RS
    return grads
```

`_fsdp_param_group.py:391` `post_backward()` → `_fsdp_collectives.py:376` `foreach_reduce()`:
```python
foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)  # line 445
# chunk_cat on stream0: packs unsharded_grads into reduce_scatter_input

current_stream = device_handle.current_stream()  # = stream0
reduce_scatter_stream.wait_stream(current_stream)  # line 449: GPU sync — RS waits for stream0

with device_handle.stream(reduce_scatter_stream):
    dist.reduce_scatter_tensor(output=reduce_output, input=reduce_scatter_input, ...)  # line 461
```

**CPU event order (from `mha_accgrad_order_probe.py`):**
```
[0] RS_FIRE
[1] ACCUM_DONE(q_proj)
[2] ACCUM_DONE(k_proj)
...
```

RS fires on CPU at [0], BEFORE AccumulateGrad Python post-hooks at [1–6]. This is
because `RegisterPostBackwardFunction` is at module input level in the autograd graph,
and fires before AccumulateGrad Python hooks in the CPU-side event order.

**Implication for the race:** `reduce_scatter_stream.wait_stream(stream0)` ensures RS
starts AFTER `chunk_cat` finishes on stream0. But if the unsharded param grads being
read by `chunk_cat` are themselves being written concurrently by stream0 operations
enqueued AFTER `chunk_cat` (due to the inplace buffer aliasing changing the GPU memory
layout), the `wait_stream` guarantee is insufficient to prevent a race on the actual
data.

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
# Workaround for a non-determinism bug with FSDP2 + torch.compile(fullgraph=True):
# TorchInductor's `inplace_buffers=True` generates a backward plan that reuses the
# k_proj saved activation (mm_1) in-place as buf24. After buf24 is freed mid-backward,
# FSDP2's reduce_scatter fires asynchronously on the COMM stream while stream0
# continues writing param gradients. This creates a cross-stream race that corrupts
# the RS input (reduce_scatter_input) non-deterministically.
# Setting inplace_buffers=False changes the backward memory plan (buf24 aliases buf3
# instead of mm_1), preventing the race. The exact aliasing collision is still under
# investigation; the fix is empirically validated.
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

### RS input / flat_grad race probes

```bash
# Probe A: capture RS input WITHOUT syncing stream0 (shows the actual race)
PYTHONPATH=. XTUNER_DETERMINISTIC=true torchrun --nproc-per-node 4 \
  tests/profiler/mha_flatgrad_nosync_probe.py --save-dir /tmp/flatgrad_nosync_a

PYTHONPATH=. XTUNER_DETERMINISTIC=true torchrun --nproc-per-node 4 \
  tests/profiler/mha_flatgrad_nosync_probe.py \
  --save-dir /tmp/flatgrad_nosync_b --compare /tmp/flatgrad_nosync_a

# Expected: n_differ > 0 (non-deterministic RS input)

# Probe B: capture RS input WITH sync (masks the race)
PYTHONPATH=. XTUNER_DETERMINISTIC=true torchrun --nproc-per-node 4 \
  tests/profiler/mha_flatgrad_at_rs_probe.py --save-dir /tmp/flatgrad_sync_a

PYTHONPATH=. XTUNER_DETERMINISTIC=true torchrun --nproc-per-node 4 \
  tests/profiler/mha_flatgrad_at_rs_probe.py \
  --save-dir /tmp/flatgrad_sync_b --compare /tmp/flatgrad_sync_a

# Expected: flat_grad IDENTICAL (sync hides race); k_proj.weight.grad still DIFFERS
```

### AccumulateGrad order probe

```bash
# Show CPU event order: RS_FIRE vs ACCUM_DONE per param
PYTHONPATH=. XTUNER_DETERMINISTIC=true torchrun --nproc-per-node 4 \
  tests/profiler/mha_accgrad_order_probe.py

# Expected: RS_FIRE at [0], ACCUM_DONE(q/k/v/o_proj) at [1]-[6]
```

### Allocator / overlap validator

```bash
torchrun --nproc-per-node 4 tests/profiler/mha_fsdp_comm_overlap_validator.py \
  --seq-len 65536 --iters 1 --deterministic \
  --record-path /tmp/mha_validator_base_a

torchrun --nproc-per-node 4 tests/profiler/mha_fsdp_comm_overlap_validator.py \
  --seq-len 65536 --iters 1 --deterministic \
  --record-path /tmp/mha_validator_base_b \
  --compare /tmp/mha_validator_base_a

# (and variants with --use-process-group-allocator, --sync-fsdp-hooks, --no-inplace-buffers)
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
| `tests/profiler/mha_backend_repro.py` | Same MHA wrapper path; attention backend switchable between `fa2` and `fake_attn` |
| `tests/profiler/demo_race_condition.py` | Older pointer/race probe; can trigger illegal memory access in buggy mode |
| `tests/profiler/mha_fsdp_inplace_addr_demo.py` | Backend-aware pointer + codegen probe; shows `k_norm_reuse_source=mm_1` vs `buf3` |
| `tests/profiler/mha_fsdp_comm_overlap_validator.py` | Cross-launch validator for allocator-vs-overlap hypotheses |
| `tests/profiler/mha_pre_rs_grad_probe.py` | Captures pre-reduce_scatter gradient data to determine if non-determinism is local or NCCL-introduced |
| `tests/profiler/mha_mm1_data_probe.py` | Captures mm_1 data at forward-save and backward-retrieve time; confirms mm_1 IDENTICAL, k_proj.weight.grad DIFFERS |
| `tests/profiler/mha_combined_addr_probe.py` | Combined mm_1 + RS address capture; checks for address overlap within and across launches |
| `tests/profiler/mha_ag_alias_probe.py` | Probes AG output addresses vs mm_1; confirms AG output does NOT overlap mm_1 |
| `tests/profiler/mha_mm1_alias_probe.py` | Additional mm_1 aliasing probe |
| `tests/profiler/mha_buf29_rs_overlap_probe.py` | CUDA memory history probe; confirms RS output does NOT alias mm_1; identifies q_proj.weight.grad at addr_mm1 |
| `tests/profiler/mha_rs_buf29_addr_probe.py` | Probes RS output address vs buf29 (o_proj grad); confirms no overlap |
| `tests/profiler/mha_flatgrad_nosync_probe.py` | Captures RS input WITHOUT stream0 sync; confirms RS input is non-deterministic (3418/27M differ) |
| `tests/profiler/mha_flatgrad_at_rs_probe.py` | Captures RS input WITH stream0 sync; RS input becomes identical (sync masks race) |
| `tests/profiler/mha_accgrad_order_probe.py` | Records CPU-side AccumulateGrad + RS fire order; confirms RS_FIRE[0] before ACCUM_DONE[1-6] |
| `tests/profiler/test_no_reshard.py` | 4-combination matrix: inplace_buffers × reshard_after_forward |
| `tests/profiler/test_ag_sync.py` | Sync after all-gather; confirms AG timing is NOT the race source |
| `tests/profiler/test_rs_sync.py` | Sync after reduce-scatter; confirms RS on COMM stream IS the race partner |
| `tests/profiler/test_accgrad_order.py` | Earlier AccumulateGrad order probe (superseded by mha_accgrad_order_probe.py) |

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

1. **Exact race mechanism**: Why does the RS input (`reduce_scatter_input`) become
   non-deterministic? `reduce_scatter_stream.wait_stream(stream0)` should ensure RS
   reads `reduce_scatter_input` only after stream0's `chunk_cat` finishes. The missing
   piece: what writes to `reduce_scatter_input` (or the unsharded grads it was copied
   from) concurrently with `chunk_cat` on a different stream?

2. **Address pair for the actual race**: The q_proj.weight.grad alloc at `addr_mm1` is
   suspicious. If COMM stream also gets `addr_mm1` for some buffer (not the RS output
   we measured), there could be a q_proj.weight.grad ↔ COMM buffer race. This needs a
   probe that captures ALL COMM stream allocations, not just the RS output.

3. **Memory cost**: How much peak memory does `inplace_buffers=False` cost in full
   training? Not measured; `allow_buffer_reuse=False` would be more conservative but
   also more expensive.

4. **FA2 trigger rate**: Why does FA2 increase the trigger rate relative to
   `fake_attn`? Likely timing/stream pressure differences, but not conclusively proven.

5. **PyTorch version sensitivity**: Does a newer or older PyTorch change whether the
   specific buf24→addr_mm1 collision pattern occurs? The seq_len=65536 threshold and
   codegen shape are build-specific.

6. **Upstream report**: This is worth reporting to PyTorch as a TorchInductor +
   FSDP2 interaction bug. The correct framing is:
   - `inplace_buffers=True` generates a backward plan that frees a large buffer
     mid-backward while an FSDP2 RS is in-flight on the COMM stream
   - this creates a cross-stream lifetime hazard in Inductor's buffer reuse analysis
   - the exact GPU-level race mechanism is still being characterized
