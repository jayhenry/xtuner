# MHA + torch.compile + FSDP2 Non-Determinism: Updated Findings

## Problem Statement

Training with `torch.compile(mha, fullgraph=True)` + FSDP2 (multi-GPU) produces
non-deterministic gradients across launches in the MHA path of
`Qwen3_5_VLTextMoE35BA3BConfig`.

**Stable fingerprint so far:**
- `--no-compile`: deterministic
- single-GPU: deterministic
- `compile + multi-GPU + seq_len=65535`: deterministic in the existing repro
- `compile + multi-GPU + seq_len=65536`: can become **NON-DETERMINISTIC**
- the drift is concentrated in `k_proj.weight`

The minimal mitigation is still:

```python
import torch._inductor.config as inductor_cfg
inductor_cfg.inplace_buffers = False
```

That fix is confirmed. What changed is the confidence level of the mechanism below.

---

## Current Best Explanation

The strongest evidence now points to:

**TorchInductor `inplace_buffers=True` causes the compiled backward to mutate saved
K-path activation storage in-place; this is the only switch that reliably removes
the cross-launch non-determinism in the current repro. The remaining evidence says
the failure still depends on multi-GPU FSDP2 execution, but the allocator-only
explanation is not sufficient.**

What is solidly established:

1. `torch.compile` generates a buggy backward plan only in the problematic mode.
2. In that buggy plan, the K-path backward reuses saved `mm_1` in-place.
3. In the forward trace, that saved value corresponds to the `k_proj` matmul output.
4. Setting `inplace_buffers=False` changes the reuse target from `mm_1` to `buf3`
   and removes all backward `in_out_ptr*` kernels.
5. With that change, the repro becomes deterministic again.
6. Leaving `inplace_buffers=True` and merely synchronizing around FSDP2 hook
   boundaries does **not** make the repro deterministic.
7. Leaving `inplace_buffers=True` and switching only FSDP2 comm staging buffers to
   the ProcessGroup allocator does **not** reliably make the repro deterministic.
8. Leaving `inplace_buffers=True` and switching `all_gather_outputs` to the
   ProcessGroup allocator also does **not** make the repro deterministic.
9. Even switching both FSDP2 comm staging buffers and `all_gather_outputs` to the
   ProcessGroup allocator together still does **not** make the repro deterministic.
10. In the minimal validator repro, `compiled_autograd_enabled=False`, so this is
   **not** specific to Traceable FSDP2 compiled-autograd.

What is **not** yet proven:

- We have **not** directly captured a runtime same-address alias between:
  - an FSDP2 communication staging buffer, and
  - the compiled backward buffer that is reused in-place.

So the older, stronger claim

> "FSDP2 communication writes the exact same GPU address that the compiled
> backward reuses in-place"

should currently be treated as an **unconfirmed hypothesis**, not as proven fact.

The other statement that should now also be treated as **incorrect / withdrawn** is:

> "Moving FSDP2 buffers to the ProcessGroup allocator restores determinism"

That was observed in one earlier run, but it does not hold up after rerunning the
same cross-launch validator and after extending the validator to cover
`all_gather_outputs`.

---

## What We Verified

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

### 5. FSDP2 is still part of the picture

The issue still depends on FSDP2 multi-GPU execution:

- single-GPU does not reproduce
- FSDP2 allocates comm staging buffers from the default CUDA caching allocator by
  default (`torch.empty(...)`) and can optionally switch to the ProcessGroup allocator
- FSDP2 pre-backward / post-backward use separate communication streams
- the nondeterminism only shows up in multi-rank compiled runs

That strongly suggests the unsafe interaction is between the compiled backward reuse
and FSDP2's communication-buffer allocation/lifetime behavior. But the exact storage
alias between an FSDP-managed buffer and the reused backward buffer has not been
directly demonstrated yet.

### 6. Allocator experiments are informative, but not sufficient

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

### 7. Pointer evidence is suggestive, but not sufficient

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

### 8. The existing race demo still hints at memory corruption

Running `tests/profiler/demo_race_condition.py` in the buggy mode sometimes triggered
an illegal memory access when reading `_unsharded_param` after backward.

That supports "something around backward-time storage reuse is unsafe", but it still
does not prove the exact same-address all-gather alias claim.

---

## Refined Mechanism

The current best-supported mechanism is:

1. `torch.compile` lowers MHA forward+backward into an Inductor graph.
2. With `inplace_buffers=True`, Inductor chooses a backward plan that reuses saved
   `mm_1` in-place on the K path.
3. In the forward graph, that saved value comes from the `k_proj` matmul output.
4. In multi-GPU FSDP2 runs, that backward plan becomes cross-launch
   non-deterministic in the current repro.
5. Setting `inplace_buffers=False` forces a different plan:
   - no backward `in_out_ptr*` kernels
   - reuse source changes from `mm_1` to `buf3`
   - determinism is restored
6. Simply synchronizing around FSDP2 hook boundaries is **not** sufficient to remove
   the nondeterminism, so this is not just a coarse "pre-backward overlap" issue.
7. Moving FSDP2 comm staging buffers, `all_gather_outputs`, or both to the
   ProcessGroup allocator is also **not** sufficient to remove the nondeterminism.

This is strong evidence for an **Inductor memory-planning / alias-lifetime bug whose
trigger requires the multi-GPU FSDP2 execution context**, but the exact offending
buffer interaction is still unresolved. The data no longer supports the narrower
claim that the bug is fully explained by default-allocator ownership of the known
FSDP communication buffers.

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
# When compiled modules run under FSDP2, TorchInductor's default
# `inplace_buffers=True` can choose a backward memory plan that mutates saved
# K-path activation storage in-place. In multi-GPU runs this has been observed to
# cause non-deterministic `k_proj.weight` gradients. Disabling `inplace_buffers`
# changes the generated backward plan and restores determinism.
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

## Open Questions

1. Can we capture a **direct same-address runtime alias** between the in-place
   reused backward buffer and any FSDP-managed storage that is live in the failing run?

2. Is the exact unsafe interaction:
   - true storage aliasing with some FSDP-managed buffer,
   - an allocator event / lifetime bug around saved activation reuse,
   - or a different cross-stream lifetime bug entirely?

3. Why does FA2 increase the trigger rate relative to `fake_attn` in this env?
   Is it only timing / stream pressure, or does FA2 also change layout/lifetime in a
   way that makes the bad plan more likely to surface?

4. How much peak memory does `inplace_buffers=False` cost in full training?

5. Does a newer or older PyTorch version change:
   - whether the bug reproduces,
   - which parameter drifts,
   - or the exact seq-len threshold that triggers the bad plan?

6. Would moving additional FSDP-owned buffers, or changing how their lifetime is
   modeled across streams, alter the result? Current partial allocator experiments
   say "not enough yet", but they do not rule out a broader FSDP-side mitigation.

7. This still looks worth reporting upstream as a PyTorch / Inductor + FSDP2 issue,
   but the bug report should use the refined claim above rather than either of these
   stronger statements:
   - "the exact same GPU address is proven to collide"
   - "moving known FSDP buffers to the ProcessGroup allocator fixes it"
