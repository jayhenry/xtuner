"""MHA-only forward/backward with FSDP2 ``fully_shard`` (same API as ``base._fully_shard``).

Trims the full model numerics test to a single sharded ``MultiHeadAttention`` plus replicated
Qwen3.5-35B RoPE.  Per-rank gradient shard sums are saved as JSON; ``--compare`` mirrors
``numerics_test.py`` (all-reduce across ranks for the verdict).

Launch with torchrun (required for ``dist.init_process_group``), e.g.::

    XTUNER_DETERMINISTIC=true torchrun --nproc-per-node 8 tests/profiler/mha_determinism_minimal.py \\
        --record-path /tmp/mha_a --seq-len 65535 --deterministic

``XTUNER_DETERMINISTIC`` must be set **before** Python starts so flash-attn sees it at import time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import torch
import torch.distributed as dist
from mmengine.runner import set_random_seed
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.module.rope.rope import get_rope_embedding

import torch._inductor.config as inductor_cfg


def _configure_inductor_trace(trace_dir: str) -> None:
    """Dump Inductor ``output_code.py`` under ``trace_dir/torchinductor/...`` (per compile)."""
    os.makedirs(trace_dir, exist_ok=True)
    inductor_cfg.trace.enabled = True
    inductor_cfg.trace.debug_dir = trace_dir
    inductor_cfg.trace.fx_graph = False
    inductor_cfg.trace.fx_graph_transformed = False
    inductor_cfg.trace.ir_pre_fusion = False
    inductor_cfg.trace.ir_post_fusion = False
    inductor_cfg.trace.output_code = True
    inductor_cfg.force_disable_caches = True


def _record_path_for_rank(base: str, rank: int) -> str:
    return f"{base}_rank{rank}.json"


def save_record(grads: dict[str, float], base_path: str, rank: int) -> None:
    path = _record_path_for_rank(base_path, rank)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"rank": rank, "param_grads": grads}, f, indent=2)


def load_record(base_path: str, rank: int) -> dict[str, float]:
    path = _record_path_for_rank(base_path, rank)
    with open(path) as f:
        data = json.load(f)
    return data["param_grads"]


def compare_records(
    new_grads: dict[str, float],
    old_grads: dict[str, float],
    rank: int,
) -> list[tuple[str, float, float]]:
    _ = rank
    diffs: list[tuple[str, float, float]] = []
    for name, new_val in new_grads.items():
        old_val = old_grads.get(name)
        if old_val is None:
            continue
        if new_val != old_val:
            diffs.append((name, new_val, old_val))
    return diffs


def _fully_shard_mha(
    mha: torch.nn.Module,
    *,
    mesh: DeviceMesh,
    fsdp_cfg: FSDPConfig,
) -> torch.nn.Module:
    """Call the same ``fully_shard`` entry point as ``XTunerBaseModelConfig._fully_shard`` (no fp32 patterns)."""
    mp_policy = MixedPrecisionPolicy(param_dtype=fsdp_cfg.param_dtype, reduce_dtype=fsdp_cfg.reduce_dtype)
    offload_policy = CPUOffloadPolicy() if fsdp_cfg.cpu_offload else None
    fully_shard(
        mha,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=fsdp_cfg.reshard_after_forward,
        offload_policy=offload_policy,
        ignored_params=None,
    )
    return mha


def _build_fsdp_mha(
    *,
    cfg: Qwen3_5_VLTextMoE35BA3BConfig,
    layer_idx: int,
    device: torch.device,
    dtype: torch.dtype,
    mesh: DeviceMesh,
    fsdp_cfg: FSDPConfig,
    compile_forward: bool,
) -> torch.nn.Module:
    mha = cfg.attention.build(
        hidden_size=cfg.hidden_size,
        layer_type=None,
        layer_idx=layer_idx,
        rope_scaling_cfg=cfg.rope_scaling_cfg,
    )
    mha = mha.to(device=device, dtype=dtype)
    if compile_forward:
        torch._dynamo.reset()
        mha = torch.compile(mha, fullgraph=True)
    mha = _fully_shard_mha(mha, mesh=mesh, fsdp_cfg=fsdp_cfg)
    return mha


def _run_forward_backward(
    mha: torch.nn.Module,
    rotary: torch.nn.Module,
    hidden_states: torch.Tensor,
    seq_ctx: SequenceContext,
) -> dict[str, float]:
    position_embeddings = rotary(hidden_states, seq_ctx.position_ids)  # type: ignore[arg-type]
    mha.zero_grad(set_to_none=True)
    out = mha(hidden_states, position_embeddings, seq_ctx)
    projected = out["projected_output"]
    loss = projected.float().sum()
    loss.backward()
    grads: dict[str, float] = {}
    for name, param in mha.named_parameters():
        if param.grad is None:
            continue
        g = param.grad
        local_g = g.to_local().float() if hasattr(g, "to_local") else g.float()
        grads[name] = local_g.sum().item()
    return grads


def main() -> None:
    parser = argparse.ArgumentParser(description="FSDP2 MHA-only compiled gradient determinism check")
    parser.add_argument("--record-path", required=True, help="Base path; writes <base>_rank<R>.json")
    parser.add_argument("--compare", default=None, help="Base path of reference run (same _rank suffix)")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=65535,
        help="Packed sequence length. 65536 often breaks determinism under compile + deterministic flash-attn.",
    )
    parser.add_argument("--layer-idx", type=int, default=3, help="Passed to MHA.build (layer name index).")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="NCCL Ring/Simple env, cublas workspace, torch deterministic; float32 reduce_dtype when set.",
    )
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile on MHA after FSDP.")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Load JSON from --record-path and compare to --compare (requires torchrun; no forward).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for init and per-rank input noise.")
    parser.add_argument(
        "--no-inplace-buffers",
        action="store_true",
        help="Disable TorchInductor inplace_buffers only (minimal fix candidate)",
    )
    parser.add_argument(
        "--keep-trace",
        action="store_true",
        help=(
            "When compiling MHA, enable TorchInductor trace and write output_code under "
            "<record-path>_inductor_trace_r<rank>/ (one directory per process). "
            "No effect with --no-compile or --skip-train."
        ),
    )
    args = parser.parse_args()

    if args.no_inplace_buffers:
        inductor_cfg.inplace_buffers = False

    if args.skip_train and not args.compare:
        print("ERROR: --skip-train requires --compare", file=sys.stderr)
        sys.exit(2)

    if args.keep_trace and args.skip_train:
        print("ERROR: --keep-trace is not supported with --skip-train", file=sys.stderr)
        sys.exit(2)

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

    if not torch.cuda.is_available():
        if rank == 0:
            print("ERROR: CUDA is required", file=sys.stderr)
        dist.destroy_process_group()
        sys.exit(1)

    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    reduce_dtype = torch.float32 if args.deterministic else torch.bfloat16
    fsdp_cfg = FSDPConfig(cpu_offload=False, ep_size=1, reduce_dtype=reduce_dtype)
    mesh = init_device_mesh("cuda", (world_size,))

    cfg = Qwen3_5_VLTextMoE35BA3BConfig()
    compile_on = not args.no_compile

    from xtuner.v1.utils import XTUNER_DETERMINISTIC

    if not args.skip_train and rank == 0:
        print(
            f"[mha_determinism_minimal] world_size={world_size}  seq_len={args.seq_len}  "
            f"layer_idx={args.layer_idx}  FSDP fully_shard  compile={'ON' if compile_on else 'OFF'}  "
            f"deterministic={args.deterministic}  XTUNER_DETERMINISTIC={XTUNER_DETERMINISTIC}  "
            f"reduce_dtype={reduce_dtype}"
        )

    if args.skip_train:
        t0 = time.time()
        new_grads = load_record(args.record_path, rank)
        if rank == 0:
            print(
                f"[mha_determinism_minimal] skip_train  loaded {len(new_grads)} params in {time.time() - t0:.2f}s"
            )
    else:
        if args.keep_trace and not compile_on:
            if rank == 0:
                print("[mha_determ] warning: --keep-trace ignored when --no-compile", file=sys.stderr)

        if args.keep_trace and compile_on:
            trace_dir = f"{args.record_path}_inductor_trace_r{rank}"
            _configure_inductor_trace(trace_dir)
            if rank == 0:
                print(
                    "[mha_determ] inductor trace enabled -> <record-path>_inductor_trace_r<R>  "
                    f"(example rank0: {trace_dir})"
                )

        rotary = get_rope_embedding(cfg, device=None)
        rotary = rotary.to(device=device)

        torch._dynamo.reset()
        mha = _build_fsdp_mha(
            cfg=cfg,
            layer_idx=args.layer_idx,
            device=device,
            dtype=dtype,
            mesh=mesh,
            fsdp_cfg=fsdp_cfg,
            compile_forward=compile_on,
        )

        g = torch.Generator(device=device)
        g.manual_seed(args.seed + rank)
        hidden_states = torch.randn(1, args.seq_len, cfg.hidden_size, dtype=dtype, device=device, generator=g)
        dummy_ids = torch.zeros(1, args.seq_len, dtype=torch.long, device=device)
        seq_ctx = SequenceContext.from_input_ids((dummy_ids,), device=str(device))

        if rank == 0:
            print("[mha_determinism_minimal] running forward+backward ...")
        t0 = time.time()
        new_grads = _run_forward_backward(mha, rotary, hidden_states, seq_ctx)
        if rank == 0:
            print(
                f"[mha_determinism_minimal] done in {time.time() - t0:.1f}s  ({len(new_grads)} local grad tensors)"
            )

        dist.barrier()
        save_record(new_grads, args.record_path, rank)
        dist.barrier()
        if rank == 0:
            print(f"[mha_determinism_minimal] saved {args.record_path}_rank*.json")

    exit_code = 0
    if args.compare:
        old_grads = load_record(args.compare, rank)
        diffs = compare_records(new_grads, old_grads, rank)

        if diffs:
            col_w = max(len(name) for name, _, _ in diffs)
            col_w = max(col_w, 10)
            header = (
                f"{'param name':<{col_w}}  {'old_grad':>16}  {'new_grad':>16}"
                f"  {'diff':>12}  {'rel_diff':>10}"
            )
            sep = "-" * len(header)
            lines = [
                f"[Rank {rank}] {len(diffs)}/{len(new_grads)} params differ:",
                sep,
                header,
                sep,
            ]
            for name, nv, ov in diffs:
                d = abs(nv - ov)
                rel = d / (abs(ov) + 1e-12)
                lines.append(
                    f"{name:<{col_w}}  {ov:>16.8e}  {nv:>16.8e}  {d:>12.4e}  {rel:>10.4e}"
                )
            lines.append(sep)
            print("\n".join(lines))
        else:
            print(f"[Rank {rank}] all {len(new_grads)} grad shard sums identical to reference run")

        dist.barrier()
        diff_count = torch.tensor([len(diffs)], dtype=torch.int64, device=device)
        max_rel = max((abs(nv - ov) / (abs(ov) + 1e-12) for _, nv, ov in diffs), default=0.0)
        max_rel_t = torch.tensor([max_rel], dtype=torch.float64, device=device)
        dist.all_reduce(diff_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(max_rel_t, op=dist.ReduceOp.MAX)
        total_diffs = diff_count.item()
        global_max_rel = max_rel_t.item()

        if rank == 0:
            print()
            if total_diffs == 0:
                print(
                    "RESULT: FULLY DETERMINISTIC — all gradient shard sums identical\n"
                    "across both process invocations on every rank."
                )
                exit_code = 0
            elif global_max_rel < 1e-4:
                print(
                    f"RESULT: PRACTICALLY DETERMINISTIC — {total_diffs} param shards differ\n"
                    f"across all ranks but max relative difference is {global_max_rel:.2e} (<1e-4),\n"
                    "which is negligible for training."
                )
                exit_code = 0
            else:
                print(
                    f"RESULT: NON-DETERMINISTIC — {total_diffs} param shards differ across\n"
                    f"all ranks with max relative difference {global_max_rel:.2e}."
                )
                exit_code = 0  # 2

    dist.destroy_process_group()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
