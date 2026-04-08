"""demo_race_condition.py

Explicitly demonstrate that the TorchInductor compiled backward and the FSDP2
pre-backward all-gather write to the **same GPU address**.

Background
----------
With ``inplace_buffers=True`` (TorchInductor default), the memory planner
declares that a backward intermediate should *reuse* the storage of a forward
input tensor.  For FSDP2-sharded modules the forward input is the all-gathered
parameter buffer.  Concretely:

  • FSDP2 pre-backward hook: allocates a buffer at address P, starts an NCCL
    all-gather to fill P with k_proj.weight on the NCCL stream.
  • TorchInductor compiled backward: sees k_proj.weight's buffer P as
    "free to reuse" after the forward mm, assigns the grad_weight computation
    output to live at address P (on the compute stream).
  • → Both streams write to P simultaneously → non-deterministic gradient.

Evidence gathered by this script
---------------------------------
[A] Runtime address comparison:
    Register a gradient hook on the MHA forward output.  The hook fires during
    backward AFTER FSDP2's pre-backward hook (same tensor, hooks fire in
    registration order, and FSDP2's hook registered during forward → ours
    registered after → ours fires second).  At this point _unsharded_param is
    live; we record its data_ptr() == P.

    After backward (no_reduce_scatter=True keeps unsharded grads accessible),
    read the grad buffer address from FSDP2 internals.  If grad_ptr == P:
    TorchInductor computed the gradient IN-PLACE into the weight buffer P.
    FSDP2's NCCL all-gather is concurrently writing to P → RACE.

[B] TorchInductor output code:
    With ``trace.output_code = True``, TorchInductor writes the compiled
    Python/Triton code to disk.  We parse it for patterns where a parameter
    argument (``arg*_1``) buffer is reused as an output buffer in the backward
    section (``reinterpret_tensor(arg*_1, ...)``, ``out=arg*_1``, etc.).

Usage
-----
    # with default inplace_buffers=True (race present):
    CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    XTUNER_DETERMINISTIC=true torchrun --nproc-per-node 4 \\
        tests/profiler/demo_race_condition.py --seq-len 65536

    # with inplace_buffers=False (fix applied):
    CUDA_VISIBLE_DEVICES=0,1,2,3 \\
    XTUNER_DETERMINISTIC=true torchrun --nproc-per-node 4 \\
        tests/profiler/demo_race_condition.py --seq-len 65536 --no-inplace-buffers

    # Or use the shell wrapper:
    CUDA_VISIBLE_DEVICES=0,1,2,3 bash tests/profiler/run_demo_race.sh
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

import torch
import torch.distributed as dist
from mmengine.runner import set_random_seed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

import torch._inductor.config as inductor_cfg

from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.module.rope.rope import get_rope_embedding


# ---------------------------------------------------------------------------
# FSDP2 helpers
# ---------------------------------------------------------------------------

def _get_fsdp_state(module):
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
        return _get_module_fsdp_state(module)
    except Exception:
        return None


def _iter_fsdp_params(module):
    state = _get_fsdp_state(module)
    if state and state._fsdp_param_group:
        yield from state._fsdp_param_group.fsdp_params


# ---------------------------------------------------------------------------
# Build FSDP2-sharded + compiled MHA
# ---------------------------------------------------------------------------

def _build_fsdp_mha(*, cfg, layer_idx, device, dtype, mesh, fsdp_cfg):
    mha = cfg.attention.build(
        hidden_size=cfg.hidden_size,
        layer_type=None,
        layer_idx=layer_idx,
        rope_scaling_cfg=cfg.rope_scaling_cfg,
    )
    mha = mha.to(device=device, dtype=dtype)
    torch._dynamo.reset()
    mha = torch.compile(mha, fullgraph=True)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=fsdp_cfg.param_dtype, reduce_dtype=fsdp_cfg.reduce_dtype
    )
    offload = CPUOffloadPolicy() if fsdp_cfg.cpu_offload else None
    fully_shard(
        mha, mesh=mesh, mp_policy=mp_policy,
        reshard_after_forward=fsdp_cfg.reshard_after_forward,
        offload_policy=offload,
    )
    return mha


# ---------------------------------------------------------------------------
# Evidence [A]: runtime address comparison
# ---------------------------------------------------------------------------

def run_address_check(mha, rotary, hidden_states, seq_ctx, *, rank: int):
    """
    Return per-parameter (param_fqn, pre_bwd_ptr, post_bwd_grad_ptrs) tuples.

    pre_bwd_ptr  : data_ptr() of _unsharded_param at FSDP2 pre-backward time
                   (the GPU address NCCL is writing k_proj.weight into)
    post_bwd_grad_ptrs : dict of {location_name: data_ptr()} for the gradient
                   accumulated after backward (no reduce-scatter)

    When inplace_buffers=True:
        pre_bwd_ptr == one of the post_bwd_grad_ptrs
        → compiled backward wrote grad INTO the weight buffer → RACE
    When inplace_buffers=False:
        pre_bwd_ptr != any post_bwd_grad_ptrs
        → separate buffers → no race
    """

    pre_bwd_ptrs: dict[str, int] = {}          # param_fqn → data_ptr
    pre_bwd_vals: dict[str, float] = {}        # param_fqn → weight sum (for corruption check)

    # ---- gradient hook on MHA output ----
    # Registered AFTER fully_shard (so FSDP2's hook on this tensor was registered
    # during the forward pass, before ours).  Hooks fire in registration order:
    # FSDP2 fires first (starts all-gather, allocates _unsharded_param),
    # then our hook fires (reads _unsharded_param.data_ptr()).
    def _capture_pre_bwd(grad):
        for fp in _iter_fsdp_params(mha):
            if fp._unsharded_param is not None:
                pre_bwd_ptrs[fp._param_fqn] = fp._unsharded_param.data_ptr()
                # Also snapshot the first 4 elements for a sanity-check
                pre_bwd_vals[fp._param_fqn] = fp._unsharded_param.float().sum().item()
        return grad

    # ---- forward ----
    position_embeddings = rotary(hidden_states, seq_ctx.position_ids)
    mha.zero_grad(set_to_none=True)
    out = mha(hidden_states, position_embeddings, seq_ctx)
    projected = out["projected_output"]

    # Register hook on `projected` (NOT on projected.float(), to stay on the
    # same tensor that FSDP2 registered its hook on during the forward).
    projected.register_hook(_capture_pre_bwd)

    # ---- backward without reduce-scatter (keeps unsharded grads accessible) ----
    mha.set_requires_gradient_sync(False)
    projected.float().sum().backward()

    # ---- collect post-backward grad buffer addresses ----
    records = []
    for fp in _iter_fsdp_params(mha):
        name = fp._param_fqn
        pre_ptr = pre_bwd_ptrs.get(name)
        if pre_ptr is None:
            continue

        post_grad_ptrs: dict[str, int] = {}

        # Location 1: unsharded_accumulated_grad (primary FSDP2 storage)
        g = fp.unsharded_accumulated_grad
        if g is not None:
            post_grad_ptrs["unsharded_accumulated_grad"] = g.data_ptr()

        # Location 2: _unsharded_param.grad (fallback)
        if fp._unsharded_param is not None:
            g2 = fp._unsharded_param.grad
            if g2 is not None:
                post_grad_ptrs["_unsharded_param.grad"] = g2.data_ptr()

        # Corruption check: if the weight buffer was aliased, its sum should now
        # equal the gradient sum, not the original weight sum
        post_weight_sum = None
        if fp._unsharded_param is not None:
            post_weight_sum = fp._unsharded_param.float().sum().item()

        records.append((name, pre_ptr, post_grad_ptrs,
                        pre_bwd_vals.get(name), post_weight_sum))

    mha.set_requires_gradient_sync(True)
    return records


# ---------------------------------------------------------------------------
# Evidence [B]: inductor output code analysis
# ---------------------------------------------------------------------------

def _find_output_code(trace_dir: str) -> str | None:
    """Find the most recently written output_code.py under trace_dir."""
    pattern = os.path.join(trace_dir, "**", "output_code.py")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def analyze_inductor_code(code_path: str) -> list[str]:
    """
    Search the TorchInductor generated code for buffer aliasing patterns where
    a parameter argument (arg*_1 — the unsharded weight passed in from FSDP2)
    is reused as an output buffer in the backward computation.

    Patterns we look for:
      • reinterpret_tensor(argN_M, ...) → arg buffer given a new shape for reuse
      • extern_kernels.mm(..., out=reinterpret_tensor(argN_M, ...)) → grad written in-place
      • buf = argN_M; del argN_M → buffer ownership transfer
    """
    try:
        with open(code_path) as f:
            code = f.read()
    except Exception as e:
        return [f"[error reading {code_path}]: {e}"]

    findings = []
    lines = code.split("\n")

    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Pattern 1: reinterpret_tensor / as_strided on an arg variable
        m = re.search(r'(buf[\w]+)\s*=\s*(?:reinterpret_tensor|torch\.as_strided)\s*\(\s*(arg[\w_]+)\s*,', stripped)
        if m:
            buf_name, arg_name = m.group(1), m.group(2)
            findings.append(
                f"  line {lineno:5d}: {buf_name} = reinterpret_tensor({arg_name}, ...)"
                f"  ← backward buffer aliases forward arg"
            )
            continue

        # Pattern 2: out= targeting an arg (inplace mm / addmm)
        if "out=" in stripped and re.search(r"out\s*=\s*(?:reinterpret_tensor\s*\()?\s*arg[\w_]+", stripped):
            findings.append(
                f"  line {lineno:5d}: {stripped[:160]}"
                f"  ← kernel output written directly into arg buffer"
            )
            continue

        # Pattern 3: buf = arg (direct alias assignment)
        m = re.search(r'^\s*(buf[\w]*)\s*=\s*(arg[\w_]+)\s*$', line)
        if m:
            buf_name, arg_name = m.group(1), m.group(2)
            findings.append(
                f"  line {lineno:5d}: {buf_name} = {arg_name}"
                f"  ← direct buffer ownership transfer from forward arg"
            )

    # Also extract arg shape annotations (comments in inductor code like "# arg2_1: ...")
    arg_shapes = {}
    for line in lines:
        m = re.match(r"\s*#\s*(arg[\w_]+)\s*:\s*(\S+)", line)
        if m:
            arg_shapes[m.group(1)] = m.group(2)

    if arg_shapes and findings:
        findings.insert(0, "  Argument shapes (from code comments):")
        for arg, shape in sorted(arg_shapes.items()):
            findings.insert(1, f"    {arg}: {shape}")
        findings.insert(len(arg_shapes) + 1, "")

    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Demo: FSDP2 all-gather + compiled backward write to same GPU addr"
    )
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--layer-idx", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--no-inplace-buffers", action="store_true",
        help="Apply the fix: set inductor_cfg.inplace_buffers = False",
    )
    parser.add_argument(
        "--trace-dir", default="/tmp/demo_race_trace",
        help="Base dir for TorchInductor output_code.py trace",
    )
    args = parser.parse_args()

    # Apply fix (or not) BEFORE any torch.compile call
    if args.no_inplace_buffers:
        inductor_cfg.inplace_buffers = False

    if args.deterministic:
        os.environ.setdefault("NCCL_ALGO", "Ring")
        os.environ.setdefault("NCCL_PROTO", "Simple")
        os.environ.setdefault("NCCL_NUM_CHANNELS", "1")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Per-rank trace directory so ranks don't stomp on each other
    local_rank_str = os.environ.get("LOCAL_RANK", "0")
    trace_dir = f"{args.trace_dir}_r{local_rank_str}"
    os.makedirs(trace_dir, exist_ok=True)

    # Enable TorchInductor output code trace
    inductor_cfg.trace.enabled = True
    inductor_cfg.trace.debug_dir = trace_dir
    inductor_cfg.trace.output_code = True
    inductor_cfg.trace.fx_graph = False
    inductor_cfg.trace.fx_graph_transformed = False
    inductor_cfg.trace.ir_pre_fusion = False
    inductor_cfg.trace.ir_post_fusion = False
    inductor_cfg.force_disable_caches = True   # always recompile to get fresh trace

    set_random_seed(args.seed, deterministic=args.deterministic)

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(local_rank_str) % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    torch.accelerator.set_device_index(local_rank)

    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    reduce_dtype = torch.float32 if args.deterministic else torch.bfloat16
    fsdp_cfg = FSDPConfig(cpu_offload=False, ep_size=1, reduce_dtype=reduce_dtype)
    mesh = init_device_mesh("cuda", (world_size,))
    cfg = Qwen3_5_VLTextMoE35BA3BConfig()

    mode_str = ("FIX: inplace_buffers=False" if args.no_inplace_buffers
                else "BUGGY: inplace_buffers=True (default)")

    if rank == 0:
        print(f"\n{'='*72}")
        print(f"  RACE CONDITION DEMO — {mode_str}")
        print(f"  world_size={world_size}  seq_len={args.seq_len}  "
              f"layer_idx={args.layer_idx}")
        print(f"{'='*72}\n")

    rotary = get_rope_embedding(cfg, device=None).to(device=device)
    mha = _build_fsdp_mha(
        cfg=cfg, layer_idx=args.layer_idx, device=device, dtype=dtype,
        mesh=mesh, fsdp_cfg=fsdp_cfg,
    )

    g = torch.Generator(device=device)
    g.manual_seed(args.seed + rank)
    hidden_states = torch.randn(
        1, args.seq_len, cfg.hidden_size, dtype=dtype, device=device, generator=g
    )
    dummy_ids = torch.zeros(1, args.seq_len, dtype=torch.long, device=device)
    seq_ctx = SequenceContext.from_input_ids((dummy_ids,), device=str(device))

    # -----------------------------------------------------------------------
    # Evidence [A]: runtime address comparison
    # -----------------------------------------------------------------------
    if rank == 0:
        print("[Evidence A] Runtime address comparison\n")
        print("  Running forward+backward (first call triggers compilation) ...")

    t0 = time.time()
    records = run_address_check(mha, rotary, hidden_states, seq_ctx, rank=rank)
    dist.barrier()

    if rank == 0:
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s\n")

        # ---- Print address table ----
        col_name = max((len(r[0]) for r in records), default=20)
        col_name = max(col_name, 20)

        hdr = (f"  {'Parameter':<{col_name}}  "
               f"{'pre-bwd weight ptr':>20}  "
               f"{'post-bwd grad ptr':>20}  "
               f"{'grad storage location':<30}  Result")
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        any_race = False
        for name, pre_ptr, post_ptrs, pre_sum, post_sum in records:
            if not post_ptrs:
                print(f"  {name:<{col_name}}  0x{pre_ptr:018x}  "
                      f"{'(no grad found)':>20}  {'':30}  —")
                continue
            for loc, gptr in post_ptrs.items():
                is_race = (gptr == pre_ptr)
                if is_race:
                    any_race = True
                marker = "*** SAME ADDR → RACE! ***" if is_race else "different addr → safe"
                print(f"  {name:<{col_name}}  0x{pre_ptr:018x}  "
                      f"0x{gptr:018x}  {loc:<30}  {marker}")

        print()

        # ---- Corruption check (weight buffer value changed?) ----
        print("  Buffer corruption check (pre-bwd weight sum vs post-bwd weight buffer sum):\n")
        print(f"  {'Parameter':<{col_name}}  {'pre-bwd weight sum':>22}  "
              f"{'post-bwd buf sum':>22}  Diff")
        print("  " + "─" * (col_name + 50))
        for name, pre_ptr, post_ptrs, pre_sum, post_sum in records:
            if pre_sum is None or post_sum is None:
                continue
            diff = abs(pre_sum - post_sum)
            rel = diff / (abs(pre_sum) + 1e-12)
            corrupted = "*** CORRUPTED (buf overwritten!) ***" if rel > 0.01 else "unchanged"
            print(f"  {name:<{col_name}}  {pre_sum:>22.4f}  {post_sum:>22.4f}  {corrupted}")

        print()

        # ---- Verdict ----
        print("─" * 72)
        if any_race:
            print(f"  [A] VERDICT: RACE CONDITION PRESENT")
            print(f"      → grad buffer address == FSDP2 all-gather weight buffer address")
            print(f"      → TorchInductor wrote gradient IN-PLACE into the weight buffer")
            print(f"      → NCCL all-gather was concurrently writing to that buffer → RACE")
        else:
            print(f"  [A] VERDICT: NO RACE — param and grad buffers are at different addresses")
        print("─" * 72)
        print()

    # -----------------------------------------------------------------------
    # Evidence [B]: TorchInductor output code analysis (rank 0 only)
    # -----------------------------------------------------------------------
    dist.barrier()

    if rank == 0:
        print("[Evidence B] TorchInductor compiled code analysis\n")
        code_path = _find_output_code(trace_dir)
        if code_path is None:
            print(f"  [warn] output_code.py not found under {trace_dir}")
            print(f"         Check: {trace_dir}")
        else:
            print(f"  Parsed: {code_path}\n")
            findings = analyze_inductor_code(code_path)
            if findings:
                print("  Buffer aliasing patterns found:")
                for line in findings:
                    print(line)
                print()
                print("  Each 'reinterpret_tensor(argN_M, ...)' or 'out=argN_M' means")
                print("  TorchInductor declared a backward output buffer to live at the")
                print("  SAME GPU ADDRESS as a forward parameter input.")
                print()
                print("  At runtime, that parameter address is the FSDP2 all-gather buffer")
                print("  that NCCL is actively writing into on the NCCL stream → RACE.")
            else:
                print("  No explicit aliasing patterns matched.")
                print("  The race may operate at the CUDA allocator level: FSDP2's all-gather")
                print("  buffer and the compiled backward's intermediate happen to get the same")
                print("  physical address from the caching allocator.")
                print()
                print(f"  Inspect the code manually: {code_path}")

        print()
        print("─" * 72)
        print(f"  [B] Trace dir: {trace_dir}")
        print(f"  [B] Inductor code: {code_path or '(not found)'}")
        print("─" * 72)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
