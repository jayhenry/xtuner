"""Capture the k_proj weight-gradient GEMM input/output inside compiled backward.

The compiled MHA backward has two consecutive GEMMs with output shape
``(512, 2048)``: v_proj.weight.grad first, then k_proj.weight.grad.  This probe
wraps Inductor's ``extern_kernels.mm`` and snapshots only the second one so we
can tell whether cross-launch drift is already present in the GEMM input
(``buf24``) or is introduced by the GEMM itself.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch._inductor.config as inductor_cfg
import torch.distributed as dist
from mmengine.runner import set_random_seed
from torch._inductor.select_algorithm import extern_kernels
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.module.rope.rope import get_rope_embedding


def _tensor_diff(a: torch.Tensor, b: torch.Tensor) -> tuple[int, float]:
    a32 = a.float()
    b32 = b.float()
    diff = (a32 - b32).abs()
    return int((diff > 0).sum().item()), float(diff.max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", default="/tmp/kproj_mm_probe")
    parser.add_argument("--compare", default=None)
    parser.add_argument("--no-inplace-buffers", action="store_true")
    args = parser.parse_args()

    if args.no_inplace_buffers:
        inductor_cfg.inplace_buffers = False

    os.environ.setdefault("NCCL_ALGO", "Ring")
    os.environ.setdefault("NCCL_PROTO", "Simple")
    os.environ.setdefault("NCCL_NUM_CHANNELS", "1")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)

    set_random_seed(args.seed, deterministic=True)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    torch.accelerator.set_device_index(local_rank)

    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    cfg = Qwen3_5_VLTextMoE35BA3BConfig()
    fsdp_cfg = FSDPConfig(cpu_offload=False, ep_size=1, reduce_dtype=torch.float32)
    mesh = init_device_mesh("cuda", (world_size,))

    snapshots: dict[str, torch.Tensor] = {}
    recording = False
    grad512_call_idx = 0
    orig_mm = extern_kernels.mm

    def wrapped_mm(*mm_args, **mm_kwargs):
        nonlocal grad512_call_idx
        out = mm_kwargs.get("out", None)
        capture = (
            recording
            and isinstance(out, torch.Tensor)
            and tuple(out.shape) == (512, 2048)
            and grad512_call_idx == 1
        )
        result = orig_mm(*mm_args, **mm_kwargs)
        if recording and isinstance(out, torch.Tensor) and tuple(out.shape) == (512, 2048):
            if capture:
                snapshots["input"] = mm_args[0].detach().clone()
                snapshots["output"] = out.detach().clone()
            grad512_call_idx += 1
        return result

    extern_kernels.mm = wrapped_mm

    mha = cfg.attention.build(
        hidden_size=cfg.hidden_size,
        layer_type=None,
        layer_idx=3,
        rope_scaling_cfg=cfg.rope_scaling_cfg,
    )
    mha = mha.to(device=device, dtype=dtype)
    torch._dynamo.reset()
    mha = torch.compile(mha, fullgraph=True)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=fsdp_cfg.param_dtype,
        reduce_dtype=fsdp_cfg.reduce_dtype,
    )
    fully_shard(
        mha,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=True,
        offload_policy=None,
    )

    rotary = get_rope_embedding(cfg, device=None).to(device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + rank)
    hidden_states = torch.randn(
        1,
        args.seq_len,
        cfg.hidden_size,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    dummy_ids = torch.zeros(1, args.seq_len, dtype=torch.long, device=device)
    seq_ctx = SequenceContext.from_input_ids((dummy_ids,), device=str(device))

    def run_one() -> None:
        pos_emb = rotary(hidden_states, seq_ctx.position_ids)
        mha.zero_grad(set_to_none=True)
        out = mha(hidden_states, pos_emb, seq_ctx)
        out["projected_output"].float().sum().backward()
        torch.cuda.synchronize()
        dist.barrier()

    run_one()

    snapshots.clear()
    grad512_call_idx = 0
    recording = True
    run_one()
    recording = False
    extern_kernels.mm = orig_mm

    os.makedirs(args.save_dir, exist_ok=True)
    for key, tensor in snapshots.items():
        torch.save(tensor.cpu(), os.path.join(args.save_dir, f"rank{rank}_{key}.pt"))

    k_proj_grad = None
    for name, param in mha.named_parameters():
        if "k_proj" in name and "weight" in name and param.grad is not None:
            grad = param.grad
            if hasattr(grad, "_local_tensor"):
                grad = grad._local_tensor
            k_proj_grad = grad.detach().contiguous().cpu()
            torch.save(k_proj_grad, os.path.join(args.save_dir, f"rank{rank}_kgrad.pt"))
            break

    dist.barrier()

    if rank == 0:
        mode = "inplace=OFF" if args.no_inplace_buffers else "inplace=ON"
        print(f"\n[mha_kproj_mm_probe] mode={mode} world_size={world_size}")
        print(f"  captured keys on rank0: {sorted(snapshots)}")

    if args.compare is not None:
        for key in ("input", "output", "kgrad"):
            cur_path = os.path.join(args.save_dir, f"rank{rank}_{key}.pt")
            ref_path = os.path.join(args.compare, f"rank{rank}_{key}.pt")
            if not os.path.exists(cur_path) or not os.path.exists(ref_path):
                continue
            cur = torch.load(cur_path, map_location="cpu")
            ref = torch.load(ref_path, map_location="cpu")
            n_diff, max_diff = _tensor_diff(cur, ref)
            print(
                f"  rank {rank} {key}: n_differ={n_diff}/{cur.numel()} "
                f"max_abs_diff={max_diff:.4e} "
                f"{'DIFFERS' if n_diff else 'IDENTICAL'}"
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
