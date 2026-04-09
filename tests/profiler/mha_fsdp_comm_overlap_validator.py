"""Validate whether FSDP2 comm buffering/overlap is the real trigger.

This keeps the existing minimal repro shape:
  torch.compile(MHA, fullgraph=True) + FSDP2 + seq_len=65536

The new part is a pair of targeted toggles that do not change the compiled MHA
graph itself:

1. ``--use-process-group-allocator``
   Ask FSDP2 to allocate all-gather / reduce-scatter staging buffers via the
   ProcessGroup backend allocator instead of the default CUDA caching allocator.
   If this restores determinism while ``inplace_buffers=True`` stays enabled,
   that strongly suggests the failure needs allocator-level reuse between
   compiled activations and FSDP communication buffers.

2. ``--sync-fsdp-hooks``
   Synchronize CUDA after FSDP2 pre-backward / post-backward hooks. This is a
   heavier control that removes communication/computation overlap without
   changing the math.

Launch example:

    PYTHONPATH=. XTUNER_DETERMINISTIC=true \\
      /mnt/shared-storage-user/zhaopenghao/miniconda3/envs/fla/bin/torchrun \\
      --nproc-per-node 4 tests/profiler/mha_fsdp_comm_overlap_validator.py \\
      --seq-len 65536 --iters 3 --deterministic
"""

from __future__ import annotations

import argparse
import json
import os
import types
from collections.abc import Iterable
from typing import Any

import torch
import torch._inductor.config as inductor_cfg
import torch.distributed as dist
from mmengine.runner import set_random_seed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard import _fsdp_param as fsdp_param_mod
from torch.distributed.fsdp._fully_shard._fsdp_common import compiled_autograd_enabled
from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state

from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.module.rope.rope import get_rope_embedding


def _fully_shard_mha(
    mha: torch.nn.Module,
    *,
    mesh,
    fsdp_cfg: FSDPConfig,
) -> torch.nn.Module:
    mp_policy = MixedPrecisionPolicy(
        param_dtype=fsdp_cfg.param_dtype,
        reduce_dtype=fsdp_cfg.reduce_dtype,
    )
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
    mesh,
    fsdp_cfg: FSDPConfig,
) -> torch.nn.Module:
    mha = cfg.attention.build(
        hidden_size=cfg.hidden_size,
        layer_type=None,
        layer_idx=layer_idx,
        rope_scaling_cfg=cfg.rope_scaling_cfg,
    )
    mha = mha.to(device=device, dtype=dtype)
    torch._dynamo.reset()
    mha = torch.compile(mha, fullgraph=True)
    mha = _fully_shard_mha(mha, mesh=mesh, fsdp_cfg=fsdp_cfg)
    return mha


def _iter_fsdp_param_groups(module: torch.nn.Module) -> Iterable[Any]:
    state = _get_module_fsdp_state(module)
    if state is not None and state._fsdp_param_group is not None:
        yield state._fsdp_param_group


def _patch_fsdp_behavior(
    module: torch.nn.Module,
    *,
    use_process_group_allocator: bool,
    use_process_group_allocator_for_allgather_outputs: bool,
    sync_fsdp_hooks: bool,
) -> list[dict[str, Any]]:
    patch_info: list[dict[str, Any]] = []
    for param_group in _iter_fsdp_param_groups(module):
        rs_backend = param_group._reduce_scatter_process_group._get_backend(param_group.device)
        supports_tensor_alloc = rs_backend.supports_tensor_alloc(param_group.device)
        if use_process_group_allocator and supports_tensor_alloc:
            param_group.allocate_memory_from_process_group = True
        ag_backend = param_group._all_gather_process_group._get_backend(param_group.device)
        ag_supports_tensor_alloc = ag_backend.supports_tensor_alloc(param_group.device)

        if use_process_group_allocator_for_allgather_outputs and ag_supports_tensor_alloc:
            for fsdp_param in param_group.fsdp_params:
                orig_init_all_gather_outputs = fsdp_param.init_all_gather_outputs
                orig_alloc_all_gather_outputs = fsdp_param.alloc_all_gather_outputs
                orig_free_unsharded_param = fsdp_param.free_unsharded_param

                def wrapped_init_all_gather_outputs(
                    self,
                    all_gather_input_numels,
                    all_gather_input_dtypes,
                    world_size,
                    device,
                    force_recreate=False,
                    _backend=ag_backend,
                    _orig=orig_init_all_gather_outputs,
                ):
                    if compiled_autograd_enabled():
                        return _orig(
                            all_gather_input_numels,
                            all_gather_input_dtypes,
                            world_size,
                            device,
                            force_recreate=force_recreate,
                        )

                    need_recreate = force_recreate or len(self.all_gather_outputs) != len(
                        all_gather_input_numels
                    )
                    if not need_recreate:
                        for tensor, numel, dtype in zip(
                            self.all_gather_outputs,
                            all_gather_input_numels,
                            all_gather_input_dtypes,
                        ):
                            expected_numel = numel * world_size
                            if (
                                tensor.numel() != expected_numel
                                or tensor.dtype != dtype
                                or tensor.device != device
                            ):
                                need_recreate = True
                                break
                    if need_recreate:
                        self.all_gather_outputs = [
                            _backend.allocate_tensor(
                                numel * world_size,
                                dtype=dtype,
                                device=device,
                            )
                            for numel, dtype in zip(
                                all_gather_input_numels,
                                all_gather_input_dtypes,
                            )
                        ]

                def wrapped_alloc_all_gather_outputs(self):
                    if compiled_autograd_enabled():
                        return orig_alloc_all_gather_outputs()
                    return None

                def wrapped_free_unsharded_param(self, _orig=orig_free_unsharded_param):
                    if compiled_autograd_enabled():
                        return _orig()
                    for tensor in self._unsharded_inner_tensors:
                        fsdp_param_mod.free_storage(tensor)

                fsdp_param.init_all_gather_outputs = types.MethodType(
                    wrapped_init_all_gather_outputs,
                    fsdp_param,
                )
                fsdp_param.alloc_all_gather_outputs = types.MethodType(
                    wrapped_alloc_all_gather_outputs,
                    fsdp_param,
                )
                fsdp_param.free_unsharded_param = types.MethodType(
                    wrapped_free_unsharded_param,
                    fsdp_param,
                )

        if sync_fsdp_hooks:
            orig_pre_backward = param_group.pre_backward
            orig_post_backward = param_group.post_backward

            def wrapped_pre_backward(self, *args, _orig=orig_pre_backward):
                result = _orig(*args)
                torch.cuda.synchronize(self.device)
                return result

            def wrapped_post_backward(self, *args, _orig=orig_post_backward):
                result = _orig(*args)
                torch.cuda.synchronize(self.device)
                return result

            param_group.pre_backward = types.MethodType(wrapped_pre_backward, param_group)
            param_group.post_backward = types.MethodType(wrapped_post_backward, param_group)

        patch_info.append(
            {
                "module_fqn": getattr(param_group, "_module_fqn", None),
                "supports_tensor_alloc": supports_tensor_alloc,
                "using_process_group_allocator": bool(
                    getattr(param_group, "allocate_memory_from_process_group", False)
                ),
                "supports_allgather_output_tensor_alloc": ag_supports_tensor_alloc,
                "using_process_group_allocator_for_allgather_outputs": (
                    use_process_group_allocator_for_allgather_outputs and ag_supports_tensor_alloc
                ),
                "sync_fsdp_hooks": sync_fsdp_hooks,
            }
        )
    return patch_info


def _run_forward_backward(
    mha: torch.nn.Module,
    rotary: torch.nn.Module,
    hidden_states: torch.Tensor,
    seq_ctx: SequenceContext,
) -> dict[str, float]:
    position_embeddings = rotary(hidden_states, seq_ctx.position_ids)  # type: ignore[arg-type]
    mha.zero_grad(set_to_none=True)
    out = mha(hidden_states, position_embeddings, seq_ctx)
    out["projected_output"].float().sum().backward()
    torch.cuda.synchronize()

    grads: dict[str, float] = {}
    for name, param in mha.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad
        local_grad = grad.to_local().float() if hasattr(grad, "to_local") else grad.float()
        grads[name] = local_grad.sum().item()
    return grads


def _compare_grads(
    baseline: dict[str, float],
    current: dict[str, float],
) -> list[tuple[str, float, float]]:
    diffs: list[tuple[str, float, float]] = []
    for name in sorted(set(baseline) | set(current)):
        b = baseline.get(name)
        c = current.get(name)
        if b != c:
            diffs.append((name, b, c))
    return diffs


def _record_path_for_rank(base: str, rank: int) -> str:
    return f"{base}_rank{rank}.json"


def _save_record(base: str, rank: int, grads: dict[str, float]) -> None:
    path = _record_path_for_rank(base, rank)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"rank": rank, "param_grads": grads}, f, indent=2)


def _load_record(base: str, rank: int) -> dict[str, float]:
    with open(_record_path_for_rank(base, rank)) as f:
        return json.load(f)["param_grads"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate FSDP2 comm/allocator overlap hypothesis")
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--layer-idx", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--record-path", default=None, help="Base path for <base>_rank<R>.json")
    parser.add_argument("--compare", default=None, help="Compare first measured grads to this record base")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-inplace-buffers", action="store_true")
    parser.add_argument("--use-process-group-allocator", action="store_true")
    parser.add_argument("--use-process-group-allocator-for-allgather-outputs", action="store_true")
    parser.add_argument("--sync-fsdp-hooks", action="store_true")
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

    rotary = get_rope_embedding(cfg, device=None).to(device=device)
    mha = _build_fsdp_mha(
        cfg=cfg,
        layer_idx=args.layer_idx,
        device=device,
        dtype=dtype,
        mesh=mesh,
        fsdp_cfg=fsdp_cfg,
    )
    patch_info = _patch_fsdp_behavior(
        mha,
        use_process_group_allocator=args.use_process_group_allocator,
        use_process_group_allocator_for_allgather_outputs=(
            args.use_process_group_allocator_for_allgather_outputs
        ),
        sync_fsdp_hooks=args.sync_fsdp_hooks,
    )

    g = torch.Generator(device=device)
    g.manual_seed(args.seed + rank)
    hidden_states = torch.randn(
        1, args.seq_len, cfg.hidden_size, dtype=dtype, device=device, generator=g
    )
    dummy_ids = torch.zeros(1, args.seq_len, dtype=torch.long, device=device)
    seq_ctx = SequenceContext.from_input_ids((dummy_ids,), device=str(device))

    mode_bits = [
        "inplace=OFF" if args.no_inplace_buffers else "inplace=ON",
        "pg_allocator=ON" if args.use_process_group_allocator else "pg_allocator=OFF",
        (
            "pg_ag_outputs=ON"
            if args.use_process_group_allocator_for_allgather_outputs
            else "pg_ag_outputs=OFF"
        ),
        "sync_hooks=ON" if args.sync_fsdp_hooks else "sync_hooks=OFF",
    ]
    if rank == 0:
        print("=" * 80)
        print("mha_fsdp_comm_overlap_validator")
        print(f"world_size={world_size} seq_len={args.seq_len} layer_idx={args.layer_idx}")
        print(f"mode: {'  '.join(mode_bits)}")
        print(f"compiled_autograd_enabled={compiled_autograd_enabled()}")
        print("=" * 80)

    gathered_patch_info: list[Any] = [None] * world_size
    dist.all_gather_object(
        gathered_patch_info,
        {
            "rank": rank,
            "patch_info": patch_info,
            "compiled_autograd_enabled": compiled_autograd_enabled(),
        },
    )
    if rank == 0:
        print("[Patch Info]")
        for item in gathered_patch_info:
            print(
                f"  rank {item['rank']}: compiled_autograd_enabled="
                f"{item['compiled_autograd_enabled']} patch_info={item['patch_info']}"
            )

    if rank == 0:
        print("[Step 1] warm-up compile")
    _run_forward_backward(mha, rotary, hidden_states, seq_ctx)
    dist.barrier()

    if rank == 0:
        print(f"[Step 2] repeated backward check ({args.iters} iterations)")
    grad_history: list[dict[str, float]] = []
    for _ in range(args.iters):
        grad_history.append(_run_forward_backward(mha, rotary, hidden_states, seq_ctx))
        dist.barrier()

    gathered_history: list[Any] = [None] * world_size
    dist.all_gather_object(gathered_history, {"rank": rank, "grads": grad_history})

    if rank == 0:
        total_kproj_diffs = 0
        total_other_diffs = 0
        for item in gathered_history:
            r = item["rank"]
            grads = item["grads"]
            baseline = grads[0]
            for idx, current in enumerate(grads[1:], start=1):
                diffs = _compare_grads(baseline, current)
                kproj_diffs = [d for d in diffs if "k_proj" in d[0]]
                other_diffs = [d for d in diffs if "k_proj" not in d[0]]
                for name, b, c in kproj_diffs:
                    total_kproj_diffs += 1
                    rel = abs(c - b) / (abs(b) + 1e-12)
                    print(
                        f"  rank {r} iter {idx}: {name} differs "
                        f"baseline={b:.6e} current={c:.6e} rel={rel:.2e}"
                    )
                for name, b, c in other_diffs:
                    total_other_diffs += 1
                    rel = abs(c - b) / (abs(b) + 1e-12)
                    print(
                        f"  rank {r} iter {idx}: {name} differs "
                        f"baseline={b:.6e} current={c:.6e} rel={rel:.2e}"
                    )

        if total_kproj_diffs == 0 and total_other_diffs == 0:
            print("[VERDICT] DETERMINISTIC")
        else:
            print(
                "[VERDICT] NON-DETERMINISTIC "
                f"(k_proj_diffs={total_kproj_diffs}, other_diffs={total_other_diffs})"
            )

    measured_grads = grad_history[0]
    if args.record_path is not None:
        _save_record(args.record_path, rank, measured_grads)
    dist.barrier()

    if args.compare is not None:
        reference_grads = _load_record(args.compare, rank)
        launch_diffs = _compare_grads(reference_grads, measured_grads)
        gathered_launch_diffs: list[Any] = [None] * world_size
        dist.all_gather_object(
            gathered_launch_diffs,
            {"rank": rank, "diffs": launch_diffs},
        )
        if rank == 0:
            total_kproj_diffs = 0
            total_other_diffs = 0
            print("[Cross-Launch Compare]")
            for item in gathered_launch_diffs:
                r = item["rank"]
                for name, ref_v, cur_v in item["diffs"]:
                    rel = abs(cur_v - ref_v) / (abs(ref_v) + 1e-12)
                    if "k_proj" in name:
                        total_kproj_diffs += 1
                    else:
                        total_other_diffs += 1
                    print(
                        f"  rank {r}: {name} differs "
                        f"reference={ref_v:.6e} current={cur_v:.6e} rel={rel:.2e}"
                    )
            if total_kproj_diffs == 0 and total_other_diffs == 0:
                print("[CROSS-LAUNCH VERDICT] DETERMINISTIC")
            else:
                print(
                    "[CROSS-LAUNCH VERDICT] NON-DETERMINISTIC "
                    f"(k_proj_diffs={total_kproj_diffs}, other_diffs={total_other_diffs})"
                )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
