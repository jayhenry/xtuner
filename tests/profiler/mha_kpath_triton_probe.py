"""Capture the k-path Triton norm-backward kernel inputs/output.

In the Inductor in-place-buffer plan, the k_proj saved activation ``mm_1`` is
reused in-place as ``buf24``:

    triton_red_fused__to_copy_add_clone_div_mul_pow_sum_5.run(
        buf24, buf11, k_norm_weight, rsqrt_1, ...
    )

``buf24`` is then fed into the k_proj.weight.grad GEMM.  This probe snapshots
``buf11`` (the upstream dK-like input), the in-place ``buf24`` value before the
Triton kernel, the RMSNorm weight/rstd inputs, and ``buf24`` after the Triton
kernel.  With ``--print-target-config`` it also prints the selected Triton
launcher, which lets us distinguish value drift caused by different reduction
block sizes from a same-launcher computation bug.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch._inductor.config as inductor_cfg
import torch.distributed as dist
from mmengine.runner import set_random_seed
from torch._inductor.runtime import triton_heuristics
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.module.rope.rope import get_rope_embedding


def _tensor_diff(a: torch.Tensor, b: torch.Tensor) -> tuple[int, float]:
    diff = (a.float() - b.float()).abs()
    return int((diff > 0).sum().item()), float(diff.max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", default="/tmp/kpath_triton_probe")
    parser.add_argument("--compare", default=None)
    parser.add_argument("--no-inplace-buffers", action="store_true")
    parser.add_argument("--sync-before-target", action="store_true")
    parser.add_argument("--sync-after-target", action="store_true")
    parser.add_argument("--print-target-config", action="store_true")
    parser.add_argument(
        "--no-dynamic-scale-rblock",
        action="store_true",
        help="Set torch._inductor.config.dynamic_scale_rblock=False before compile.",
    )
    parser.add_argument(
        "--keep-trace",
        action="store_true",
        help="Save TorchInductor debug output_code.py files under save-dir.",
    )
    args = parser.parse_args()

    if args.no_inplace_buffers:
        inductor_cfg.inplace_buffers = False
    if args.no_dynamic_scale_rblock:
        inductor_cfg.dynamic_scale_rblock = False

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

    trace_dir = None
    if args.keep_trace:
        trace_dir = Path(args.save_dir).resolve() / "inductor_trace" / f"rank{rank}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        inductor_cfg.fx_graph_cache = False
        inductor_cfg.trace.enabled = True
        inductor_cfg.trace.debug_dir = str(trace_dir)
        inductor_cfg.trace.output_code = True

    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    cfg = Qwen3_5_VLTextMoE35BA3BConfig()
    fsdp_cfg = FSDPConfig(cpu_offload=False, ep_size=1, reduce_dtype=torch.float32)
    mesh = init_device_mesh("cuda", (world_size,))

    snapshots: dict[str, torch.Tensor] = {}
    recording = False
    orig_run = triton_heuristics.CachingAutotuner.run

    def wrapped_run(self, *run_args, stream, benchmark_run=False, **run_kwargs):
        meta = getattr(self, "inductor_meta", {}) or {}
        kernel_name = meta.get("kernel_name", "")
        is_inplace_target = (
            recording
            and "clone_div_mul_pow_sum" in kernel_name
            and len(run_args) >= 4
            and isinstance(run_args[0], torch.Tensor)
            and isinstance(run_args[1], torch.Tensor)
            and tuple(run_args[0].shape) == (1, args.seq_len, 2, 256)
            and run_args[0].dtype == torch.bfloat16
            and run_args[1].dtype == torch.float32
        )
        is_outplace_target = (
            recording
            and "clone_div_mul_pow_sum" in kernel_name
            and len(run_args) >= 5
            and isinstance(run_args[0], torch.Tensor)
            and isinstance(run_args[2], torch.Tensor)
            and isinstance(run_args[4], torch.Tensor)
            and tuple(run_args[0].shape) == (1, args.seq_len, 2, 256)
            and tuple(run_args[4].shape) == (1, args.seq_len, 2, 256)
            and run_args[0].dtype == torch.float32
            and run_args[2].dtype == torch.bfloat16
            and run_args[4].dtype == torch.bfloat16
        )
        is_target = is_inplace_target or is_outplace_target
        if is_target and args.sync_before_target:
            torch.cuda.synchronize()
        should_capture = is_target and "buf24_after" not in snapshots
        if should_capture:
            if is_inplace_target:
                snapshots["buf11_input"] = run_args[1].detach().clone()
                snapshots["buf24_before"] = run_args[0].detach().clone()
                if isinstance(run_args[2], torch.Tensor):
                    snapshots["norm_weight"] = run_args[2].detach().clone()
                if isinstance(run_args[3], torch.Tensor):
                    snapshots["rsqrt_1"] = run_args[3].detach().clone()
            else:
                snapshots["buf11_input"] = run_args[0].detach().clone()
                if isinstance(run_args[1], torch.Tensor):
                    snapshots["norm_weight"] = run_args[1].detach().clone()
                snapshots["mm1_input"] = run_args[2].detach().clone()
                if isinstance(run_args[3], torch.Tensor):
                    snapshots["rsqrt_1"] = run_args[3].detach().clone()

        result = orig_run(
            self,
            *run_args,
            stream=stream,
            benchmark_run=benchmark_run,
            **run_kwargs,
        )
        if should_capture:
            if args.sync_after_target:
                torch.cuda.synchronize()
            if args.print_target_config:
                launcher = self.launchers[0] if getattr(self, "launchers", None) else None
                config = getattr(launcher, "config", None)
                compile_configs = [
                    getattr(result, "config", None)
                    for result in getattr(self, "compile_results", [])
                ]
                print(
                    f"  target_config rank={rank} "
                    f"kwargs={getattr(config, 'kwargs', None)} "
                    f"num_warps={getattr(config, 'num_warps', None)} "
                    f"num_stages={getattr(config, 'num_stages', None)} "
                    f"compile_configs={[getattr(c, 'kwargs', None) for c in compile_configs]}"
                )
            if is_inplace_target:
                snapshots["buf24_after"] = run_args[0].detach().clone()
            else:
                snapshots["buf24_after"] = run_args[4].detach().clone()
        return result

    triton_heuristics.CachingAutotuner.run = wrapped_run

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
    recording = True
    run_one()
    recording = False
    triton_heuristics.CachingAutotuner.run = orig_run

    os.makedirs(args.save_dir, exist_ok=True)
    for key, tensor in snapshots.items():
        torch.save(tensor.cpu(), os.path.join(args.save_dir, f"rank{rank}_{key}.pt"))

    dist.barrier()

    if rank == 0:
        mode = "inplace=OFF" if args.no_inplace_buffers else "inplace=ON"
        rblock_mode = (
            "dynamic_scale_rblock=OFF"
            if args.no_dynamic_scale_rblock
            else "dynamic_scale_rblock=ON"
        )
        print(
            f"\n[mha_kpath_triton_probe] mode={mode} {rblock_mode} "
            f"world_size={world_size}"
        )
        print(f"  captured keys on rank0: {sorted(snapshots)}")
        if args.keep_trace:
            print(
                "  output_code traces: "
                f"{Path(args.save_dir).resolve() / 'inductor_trace' / 'rank*'}"
            )
    if args.keep_trace and trace_dir is not None:
        output_codes = sorted(trace_dir.rglob("output_code.py"))
        print(
            f"  rank {rank} output_code files: "
            f"{len(output_codes)} under {trace_dir}"
        )

    if args.compare is not None:
        for key in (
            "buf11_input",
            "buf24_before",
            "mm1_input",
            "norm_weight",
            "rsqrt_1",
            "buf24_after",
        ):
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
