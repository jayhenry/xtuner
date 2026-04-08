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
K-path activation storage in-place; under FSDP2 multi-GPU execution, that reuse
becomes non-deterministic.**

What is solidly established:

1. `torch.compile` generates a buggy backward plan only in the problematic mode.
2. In that buggy plan, the K-path backward reuses saved `mm_1` in-place.
3. In the forward trace, that saved value corresponds to the `k_proj` matmul output.
4. Setting `inplace_buffers=False` changes the reuse target from `mm_1` to `buf3`
   and removes all backward `in_out_ptr*` kernels.
5. With that change, the repro becomes deterministic again.

What is **not** yet proven:

- We have **not** directly captured a runtime same-address alias between:
  - the FSDP2 pre-backward all-gather buffer for `k_proj.weight`, and
  - the compiled backward buffer that is reused in-place.

So the older, stronger claim

> "FSDP2 all-gather writes the exact same GPU address that the compiled backward
> reuses in-place"

should currently be treated as an **unconfirmed hypothesis**, not as proven fact.

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
- `backend=fake_attn`: 1/5 compare runs was non-deterministic. But I cannot reproduce it now for some reason.

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
- FSDP2 pre-backward runs with separate communication streams
- the nondeterminism only shows up in multi-rank compiled runs

That strongly suggests FSDP2 concurrent activity is what makes the buggy reuse unsafe.
But the exact storage alias between FSDP-managed buffers and the reused backward buffer
has not been directly demonstrated yet.

### 6. Pointer evidence is suggestive, but not sufficient

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

### 7. The existing race demo still hints at memory corruption

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
4. In multi-GPU FSDP2 runs, pre-backward communication happens on side CUDA streams.
5. That combination makes the in-place backward reuse non-deterministic.
6. Setting `inplace_buffers=False` forces a different plan:
   - no backward `in_out_ptr*` kernels
   - reuse source changes from `mm_1` to `buf3`
   - determinism is restored

This is strong evidence for an **Inductor memory-planning bug or unsupported aliasing
assumption in the presence of FSDP2**, even though the exact storage collision is still
not fully pinned down.

---

## Why The Existing Fingerprint Makes Sense

### Why only with compile?

Because the problematic in-place reuse is introduced by TorchInductor codegen.
Eager execution does not produce this compiled memory plan.

### Why only multi-GPU / FSDP2?

Because the failure seems to require FSDP2's pre-backward communication behavior.
Single-GPU runs do not have the same sharded parameter orchestration and do not
reproduce the issue.

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

---

## Investigation Scripts

| File | Purpose |
|------|---------|
| `tests/profiler/mha_determ.py` | Main compiled MHA determinism repro |
| `tests/profiler/run_mha_determ.sh` | Shell runner for the main repro |
| `tests/profiler/mha_backend_repro.py` | Same MHA wrapper path, but attention backend can be switched between `fa2` and `fake_attn` |
| `tests/profiler/demo_race_condition.py` | Older pointer/race probe; can trigger illegal memory access in buggy mode |
| `tests/profiler/mha_fsdp_inplace_addr_demo.py` | Updated backend-aware pointer + codegen probe for the current hypothesis |

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

1. Can we capture a **direct same-address runtime alias** between the FSDP2
   pre-backward buffer and the in-place reused backward buffer?

2. Is the exact unsafe interaction:
   - true storage aliasing with an FSDP-managed buffer, or
   - an ordering / lifetime bug around saved activation reuse under FSDP2?

3. Why does FA2 increase the trigger rate relative to `fake_attn` in this env?
   Is it only timing / stream pressure, or does FA2 also change layout/lifetime in a
   way that makes the bad plan more likely to surface?

4. How much peak memory does `inplace_buffers=False` cost in full training?

5. Does a newer or older PyTorch version change:
   - whether the bug reproduces,
   - which parameter drifts,
   - or the exact seq-len threshold that triggers the bad plan?

6. This still looks worth reporting upstream as a PyTorch / Inductor + FSDP2 issue,
   but the bug report should use the refined claim above rather than the stronger
   unproven same-address statement.
