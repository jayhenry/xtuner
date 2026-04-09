"""mha_combined_addr_probe.py — Capture mm_1 address AND reduce_scatter input
address in the SAME process to check for direct address overlap.

Also compares RS input data between overlap iterations (mm_1 inside RS buffer)
and no-overlap iterations, to determine if the overlap causes gradient corruption.

Launch:
  PYTHONPATH=. XTUNER_DETERMINISTIC=true torchrun --nproc-per-node 4 \\
    tests/profiler/mha_combined_addr_probe.py --seq-len 65536 --deterministic \\
    [--save-dir /tmp/rs_combined_a]  [--compare /tmp/rs_combined_a]
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch
import torch._inductor.config as inductor_cfg
import torch.distributed as dist
from mmengine.runner import set_random_seed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.module.rope.rope import get_rope_embedding


# ── RS interceptor ─────────────────────────────────────────────────────────────
_orig_rs = dist.reduce_scatter_tensor
_rs_list = []
_rs_recording = False

def _patched_rs(*args, **kwargs):
    output_t = args[0] if args else kwargs.get("output", kwargs.get("output_tensor"))
    input_t  = args[1] if len(args) > 1 else kwargs.get("input", kwargs.get("input_tensor"))
    if _rs_recording and input_t is not None and isinstance(input_t, torch.Tensor):
        # Save reference — copy to CPU after synchronize
        _rs_list.append({
            "input_ptr": input_t.data_ptr(),
            "input_nbytes": input_t.numel() * input_t.element_size(),
            "input_shape": tuple(input_t.shape),
            "input_dtype": str(input_t.dtype),
            "input_tensor_ref": input_t.detach().clone(),
        })
    return _orig_rs(*args, **kwargs)

def start_rs(): global _rs_recording, _rs_list; _rs_list = []; _rs_recording = True; dist.reduce_scatter_tensor = _patched_rs
def stop_rs():
    global _rs_recording; _rs_recording = False; dist.reduce_scatter_tensor = _orig_rs
    return list(_rs_list)


# ── Build helper ───────────────────────────────────────────────────────────────
def _build_fsdp_mha(*, cfg, layer_idx, device, dtype, mesh, fsdp_cfg):
    mha = cfg.attention.build(
        hidden_size=cfg.hidden_size, layer_type=None,
        layer_idx=layer_idx, rope_scaling_cfg=cfg.rope_scaling_cfg,
    )
    mha = mha.to(device=device, dtype=dtype)
    torch._dynamo.reset()
    mha = torch.compile(mha, fullgraph=True)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=fsdp_cfg.param_dtype, reduce_dtype=fsdp_cfg.reduce_dtype,
    )
    offload = CPUOffloadPolicy() if fsdp_cfg.cpu_offload else None
    fully_shard(mha, mesh=mesh, mp_policy=mp_policy,
                reshard_after_forward=fsdp_cfg.reshard_after_forward,
                offload_policy=offload)
    return mha


def _run_iter(mha, rotary, hidden_states, seq_ctx, capture_rs=False, capture_mm1=False):
    """Run one forward+backward. Returns (mm1_addr, mm1_nbytes, rs_events)."""
    mm1_candidates = []

    def _pack(t):
        if capture_mm1 and t.dtype == torch.bfloat16 and t.shape[-1] == 512 and t.ndim == 2:
            mm1_candidates.append({"ptr": t.data_ptr(), "nbytes": t.numel()*t.element_size()})
        return t

    if capture_rs:
        start_rs()

    with torch.autograd.graph.saved_tensors_hooks(_pack, lambda t: t):
        position_embeddings = rotary(hidden_states, seq_ctx.position_ids)
        mha.zero_grad(set_to_none=True)
        out = mha(hidden_states, position_embeddings, seq_ctx)
        out["projected_output"].float().sum().backward()
    torch.cuda.synchronize()

    rs_evts = stop_rs() if capture_rs else []
    # Convert GPU tensors to numpy
    for ev in rs_evts:
        ref = ev.pop("input_tensor_ref", None)
        if ref is not None:
            ev["input_data"] = ref.float().cpu().numpy()

    mm1 = mm1_candidates[-1] if mm1_candidates else None
    return mm1, rs_evts


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--layer-idx", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-inplace-buffers", action="store_true")
    parser.add_argument("--iters", type=int, default=6,
                        help="Iterations to look for overlap/no-overlap examples")
    parser.add_argument("--save-dir", default=None,
                        help="Save RS inputs (iter 0) here for cross-launch compare")
    parser.add_argument("--compare", default=None,
                        help="Compare RS inputs (iter 0) against this save-dir")
    args = parser.parse_args()

    if args.no_inplace_buffers:
        inductor_cfg.inplace_buffers = False
    if args.deterministic:
        os.environ.setdefault("NCCL_ALGO", "Ring")
        os.environ.setdefault("NCCL_PROTO", "Simple")
        os.environ.setdefault("NCCL_NUM_CHANNELS", "1")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True, warn_only=True)

    set_random_seed(args.seed, deterministic=args.deterministic)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    torch.accelerator.set_device_index(local_rank)

    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    reduce_dtype = torch.float32 if args.deterministic else torch.bfloat16
    fsdp_cfg = FSDPConfig(cpu_offload=False, ep_size=1, reduce_dtype=reduce_dtype)
    mesh = init_device_mesh("cuda", (world_size,))
    cfg = Qwen3_5_VLTextMoE35BA3BConfig()

    mode = "FIXED" if args.no_inplace_buffers else "BUGGY"
    if rank == 0:
        print(f"\n{'='*70}\n  mha_combined_addr_probe — {mode}\n{'='*70}")

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

    # Warm-up
    if rank == 0: print("\n[Warm-up]")
    _run_iter(mha, rotary, hidden_states, seq_ctx)
    dist.barrier()

    # ── Collect overlap/no-overlap examples and their RS inputs ───────────────
    overlap_rs_data = None      # RS input from first iter with overlap
    no_overlap_rs_data = None   # RS input from first iter without overlap
    iter0_rs_data = None        # RS input from iter 0 (for cross-launch compare)
    iter0_addr_info = None

    all_iter_info = []

    for it in range(args.iters):
        mm1, rs_evts = _run_iter(mha, rotary, hidden_states, seq_ctx,
                                 capture_rs=True, capture_mm1=True)
        dist.barrier()

        mm1_ptr = mm1["ptr"] if mm1 else None
        mm1_nb  = mm1["nbytes"] if mm1 else None
        rs_data = rs_evts[0]["input_data"] if rs_evts else None
        rs_ptr  = rs_evts[0]["input_ptr"] if rs_evts else None
        rs_nb   = rs_evts[0]["input_nbytes"] if rs_evts else None

        # Check overlap
        has_overlap = False
        overlap_info = {}
        if mm1_ptr is not None and rs_ptr is not None:
            mm1_end = mm1_ptr + mm1_nb
            rs_end  = rs_ptr + rs_nb
            if mm1_ptr < rs_end and rs_ptr < mm1_end:
                has_overlap = True
                overlap_start = max(mm1_ptr, rs_ptr)
                overlap_bytes = min(mm1_end, rs_end) - overlap_start
                overlap_info = {
                    "byte_offset_in_rs": (overlap_start - rs_ptr),
                    "overlap_bytes": overlap_bytes,
                }

        all_iter_info.append({
            "iter": it, "mm1_ptr": mm1_ptr, "rs_ptr": rs_ptr,
            "has_overlap": has_overlap, "overlap_info": overlap_info,
        })

        if it == 0:
            iter0_rs_data = rs_data
            iter0_addr_info = {"mm1_ptr": mm1_ptr, "rs_ptr": rs_ptr, "has_overlap": has_overlap}

        if has_overlap and overlap_rs_data is None and rs_data is not None:
            overlap_rs_data = (it, rs_data, rs_ptr, mm1_ptr)
        if not has_overlap and no_overlap_rs_data is None and rs_data is not None:
            no_overlap_rs_data = (it, rs_data, rs_ptr, mm1_ptr)

    # ── Gather and report ──────────────────────────────────────────────────────
    all_gathered = [None] * world_size
    payload = {
        "rank": rank,
        "all_iter_info": all_iter_info,
        "overlap_rs_data": (overlap_rs_data[0], overlap_rs_data[2], overlap_rs_data[3])
                           if overlap_rs_data else None,
        "no_overlap_rs_data": (no_overlap_rs_data[0], no_overlap_rs_data[2], no_overlap_rs_data[3])
                               if no_overlap_rs_data else None,
    }
    dist.all_gather_object(all_gathered, payload)

    if rank == 0:
        print("\n[Per-Iteration Address Summary — rank 0]")
        for item in all_gathered[:1]:  # just rank 0
            for info in item["all_iter_info"]:
                it = info["iter"]
                overlap = "*** OVERLAP ***" if info["has_overlap"] else "no overlap"
                ovl_detail = f" ({info['overlap_info'].get('overlap_bytes',0)//1024//1024}MB)" if info["has_overlap"] else ""
                print(f"  iter {it}: mm1=0x{info['mm1_ptr']:016x}  rs=0x{info['rs_ptr']:016x}  {overlap}{ovl_detail}")

        # Compare RS inputs: overlap vs no-overlap (rank 0 only)
        if overlap_rs_data is not None and no_overlap_rs_data is not None:
            ov_it, ov_data, _, _ = overlap_rs_data
            no_it, no_data, _, _ = no_overlap_rs_data
            diff = np.abs(ov_data - no_data)
            n_diff = (diff > 0).sum()
            max_diff = diff.max()
            print(f"\n[Within-launch RS comparison — rank 0]")
            print(f"  overlap_iter={ov_it} vs no_overlap_iter={no_it}")
            print(f"  n_differ={n_diff}/{len(ov_data)}  max_abs_diff={max_diff:.4e}")
            if n_diff > 0:
                print(f"  *** OVERLAP ITER RS INPUT DIFFERS FROM NO-OVERLAP ITER! ***")
                print(f"  → The address overlap DOES cause gradient corruption")
            else:
                print(f"  Overlap and no-overlap iters produce IDENTICAL RS inputs")
                print(f"  → The address overlap itself does NOT corrupt the gradient")
                print(f"  → Cross-launch non-determinism must have a different cause")

        # Save iter 0 RS data for cross-launch compare
        if args.save_dir is not None and iter0_rs_data is not None:
            os.makedirs(args.save_dir, exist_ok=True)

    # Save iter 0 RS data per-rank (only rank 0 prints, but all ranks save)
    if args.save_dir is not None and iter0_rs_data is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        np.save(os.path.join(args.save_dir, f"rank{rank}_iter0_rs.npy"), iter0_rs_data)
        with open(os.path.join(args.save_dir, f"rank{rank}_meta.txt"), "w") as f:
            f.write(f"mm1_ptr=0x{iter0_addr_info['mm1_ptr']:016x}\n")
            f.write(f"rs_ptr=0x{iter0_addr_info['rs_ptr']:016x}\n")
            f.write(f"has_overlap={iter0_addr_info['has_overlap']}\n")
    dist.barrier()
    if rank == 0 and args.save_dir:
        print(f"\n  Saved iter 0 RS input to {args.save_dir}/")

    # Cross-launch comparison
    if args.compare is not None:
        if rank == 0:
            print(f"\n[Cross-launch comparison vs {args.compare}/]")
        cur_path = os.path.join(args.save_dir or "/tmp/rs_tmp", f"rank{rank}_iter0_rs.npy")
        ref_path = os.path.join(args.compare, f"rank{rank}_iter0_rs.npy")
        if iter0_rs_data is not None and os.path.exists(ref_path):
            ref_data = np.load(ref_path)
            diff = np.abs(iter0_rs_data - ref_data)
            n_diff = (diff > 0).sum()
            max_diff = diff.max()
            print(f"  rank {rank}: n_differ={n_diff}  max_abs_diff={max_diff:.4e}  "
                  f"{'DIFFERS' if n_diff > 0 else 'IDENTICAL'}")

            # Read meta
            ref_meta_path = os.path.join(args.compare, f"rank{rank}_meta.txt")
            if os.path.exists(ref_meta_path):
                with open(ref_meta_path) as f:
                    ref_meta = f.read().strip()
                cur_meta_str = (f"mm1_ptr=0x{iter0_addr_info['mm1_ptr']:016x}"
                                if iter0_addr_info else "?")
                print(f"    current: mm1=0x{iter0_addr_info['mm1_ptr']:016x} rs=0x{iter0_addr_info['rs_ptr']:016x} overlap={iter0_addr_info['has_overlap']}")
                print(f"    ref:     {ref_meta.replace(chr(10), ' ')}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
