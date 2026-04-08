# MHA + torch.compile + FSDP2 Non-Determinism: Root Cause & Fix

## Problem Statement

Training with `torch.compile(mha, fullgraph=True)` + FSDP2 (multi-GPU) produces
non-deterministic gradients across runs. Observed in Qwen3.5-35B-A3B MoE model.

**Empirical fingerprint (original observations):**
- `--no-compile`: always deterministic
- `NPROC=1`: always deterministic (compile or not)
- `compile + NPROC>1 + seq_len=65535`: deterministic
- `compile + NPROC>1 + seq_len=65536`: **NON-DETERMINISTIC**
- Only `k_proj.weight.grad` affected — not `q_proj`, `v_proj`, `o_proj`, `q_norm`, `k_norm`

---

## Root Cause

**TorchInductor `inplace_buffers=True` (default) + FSDP2 pre-backward all-gather = race condition.**

### Mechanism

1. `torch.compile` compiles the MHA forward+backward into a TorchInductor graph.
2. TorchInductor's memory planner (`inplace_buffers=True`) decides to **reuse the forward
   input buffer for `k_proj.weight`** (the FSDP2 all-gather workspace) as a backward
   intermediate tensor — e.g. the buffer holding `dk` or some reshape thereof.
3. FSDP2's pre-backward hook simultaneously writes the **next all-gather of `k_proj.weight`**
   into that same buffer, on a **separate CUDA stream**.
4. The compiled backward reads/writes that buffer concurrently → race → non-deterministic
   `k_proj.weight.grad`.

### Why only k_proj?

TorchInductor's memory plan specifically aliases the k_proj buffer with a backward
intermediate in the k-path. v_proj, q_proj, o_proj use a different plan (different
tensor sizes or lifetimes in the compiled graph).

### Why seq_len=65536 but not 65535?

At exactly `65536` (2^16), TorchInductor's heuristics pick a memory plan that aliases
the k_proj all-gather buffer. At `65535` the sizes differ slightly, leading to a
different plan that avoids the alias.

### Why only NPROC>1?

With a single rank, FSDP2 does not shard parameters → no all-gather → no concurrent
write to the buffer.

### Why only with compile?

Eager PyTorch never reuses forward input buffers as backward intermediates.

### Investigation path (how we got here)

1. Verified FA2 `deterministic` flag IS passed correctly (monkey-patch probe).
2. Verified FA2's dK/dV CUDA writes are non-atomic → FA2 kernel itself is deterministic.
3. Verified `at::sum_out` for GQA group reduction is deterministic.
4. Ruled out NCCL reduce-scatter (non-det persists with `--no-reduce-scatter`).
5. Ruled out Triton RMSNorm autotuning (simple k_proj+k_norm without FA2 is deterministic
   with proper seeding; the `@triton.autotune` result is persistent within a run).
6. **Key experiment**: setting `torch._inductor.config.allow_buffer_reuse = False` AND
   `inplace_buffers = False` → FULLY DETERMINISTIC.
7. **Minimal fix confirmed**: `inplace_buffers = False` alone is sufficient.

---

## Fix

**File**: `xtuner/v1/model/base.py`, method `_maybe_enable_compile` (around line 1789).

```python
def _maybe_enable_compile(self, compile_cfg: dict[str, TorchCompileOption]):
    if compile_cfg:
        torch._dynamo.config.cache_size_limit = 256
        # When the compiled forward/backward runs inside FSDP2-sharded modules,
        # TorchInductor's `inplace_buffers` optimization causes a race condition:
        # the compiled backward reuses a forward input buffer (the FSDP2 all-gather
        # workspace for a parameter such as k_proj.weight) for a backward
        # intermediate, while FSDP2's pre-backward all-gather simultaneously writes
        # into that same buffer on a side CUDA stream.  The result is non-deterministic
        # gradients (empirically observed in k_proj.weight.grad).
        # Disabling inplace_buffers removes the aliasing and restores determinism.
        import torch._inductor.config as inductor_cfg
        inductor_cfg.inplace_buffers = False

    for target, option in compile_cfg.items():
        self._compile_overwrite(target, option)
```

**The setting must be applied before `torch.compile` is called**, because TorchInductor
bakes it into the compiled graph at first trace time.

### Alternative (more conservative, higher memory cost)

```python
inductor_cfg.allow_buffer_reuse = False
inductor_cfg.inplace_buffers = False
```

`allow_buffer_reuse=False` additionally prevents reuse of intermediate buffers between
ops, which is more aggressive but higher memory overhead. `inplace_buffers=False` alone
is sufficient.

---

## Verification

```bash
export PYTHONPATH="./"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 XTUNER_USE_FA3=0
export XTUNER_DETERMINISTIC=true
export NCCL_ALGO=Ring NCCL_PROTO=Simple NCCL_NUM_CHANNELS=1
export CUBLAS_WORKSPACE_CONFIG=:16:8

TORCHRUN="torchrun --nproc-per-node 8 --master-port 29732"
SCRIPT="tests/profiler/debug_nondeterminism.py"

# Baseline: non-deterministic
$TORCHRUN $SCRIPT --record-path /tmp/r1 --seq-len 65536 --deterministic
$TORCHRUN $SCRIPT --record-path /tmp/r2 --seq-len 65536 --deterministic --compare /tmp/r1
# → RESULT: NON-DETERMINISTIC — 7-8 param shards differ

# Fixed (inplace_buffers=False):
$TORCHRUN $SCRIPT --record-path /tmp/f1 --seq-len 65536 --deterministic --no-inplace-buffers
$TORCHRUN $SCRIPT --record-path /tmp/f2 --seq-len 65536 --deterministic --no-inplace-buffers --compare /tmp/f1
# → RESULT: FULLY DETERMINISTIC
```

---

## Test Scripts

| File | Purpose |
|------|---------|
| `tests/profiler/debug_nondeterminism.py` | Full MHA (FA2 + all projections) non-determinism test. New flags: `--no-inplace-buffers`, `--no-buffer-reuse` |
| `tests/profiler/run_debug_nondeterminism.sh` | Shell runner for the full MHA test |
| `tests/profiler/debug_kproj_nondeterminism.py` | **New**: minimal reproducer — `k_proj + k_norm` only (no FA2). Confirms simple path is deterministic; FA2 interaction is needed to trigger the bug |
| `tests/profiler/run_kproj_nondeterminism.sh` | **New**: shell runner for the minimal reproducer |

### Key flag added to `debug_nondeterminism.py`

```
--no-inplace-buffers   Sets torch._inductor.config.inplace_buffers = False (minimal fix)
--no-buffer-reuse      Sets both inplace_buffers=False and allow_buffer_reuse=False
```

---

## Model Config Context

| Field | Value |
|-------|-------|
| Model | `Qwen3_5_VLTextMoE35BA3BConfig` |
| `hidden_size` | 2048 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 2 (GQA 8:1) |
| `head_dim` | 256 |
| `qk_norm` | True (Triton RMSNorm on Q and K) |
| `with_gate` | True |
| `sliding_window` | 1024 |

k_proj weight gradient matmul: `(512, seq_len) @ (seq_len, 2048)`.
At seq_len=65536 the K dimension (65536 = 2^16) triggers the specific TorchInductor
memory plan that aliases the buffer.

---

## Follow-up / Open Questions

1. **Upstream bug report**: This is a genuine PyTorch bug — TorchInductor should not
   alias an FSDP2-managed all-gather buffer with a backward intermediate. Worth reporting
   to PyTorch with a minimal repro.

2. **Memory overhead of fix**: `inplace_buffers=False` increases peak activation memory
   during the compiled backward. Measure the overhead on large runs.

3. **Other models affected**: Any model using `torch.compile + FSDP2` with per-layer
   compilation (e.g. `DenseDecoderLayer.forward`, `MoEBlock.forward`) could hit the same
   race. The specific parameter affected depends on TorchInductor's memory plan.

4. **PyTorch version sensitivity**: Tested on the current env. The memory planning
   heuristics may change across PyTorch versions, potentially making the bug appear/
   disappear at different seq_len thresholds.
