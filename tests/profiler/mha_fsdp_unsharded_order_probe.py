"""Probe actual FSDP2 unsharded-grad vs reduce-scatter ordering.

This is meant to disambiguate two different events:

* autograd AccumulateGrad on FSDPParam.unsharded_param, which produces the
  unsharded gradients consumed by FSDP post_backward();
* post-accumulate hooks on FSDPParam.sharded_param, which FSDP2 manually invokes
  after reduce-scatter when it installs the sharded gradient.

The older `mha_accgrad_order_probe.py` observes the second event when registering
hooks through `mha.named_parameters()`, so `RS_FIRE` before those hooks does not
prove that RS raced with unsharded-gradient production.
"""
from __future__ import annotations

import argparse
import os
import types
from typing import Any

import torch
import torch._inductor.config as inductor_cfg
import torch.distributed as dist
from mmengine.runner import set_random_seed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state

from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.module.rope.rope import get_rope_embedding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
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

    state = _get_module_fsdp_state(mha)
    assert state is not None and state._fsdp_param_group is not None
    param_group = state._fsdp_param_group
    fsdp_params = list(param_group.fsdp_params)

    event_log: list[str] = []

    def pname(index: int, fsdp_param: Any) -> str:
        fqn = fsdp_param._param_fqn
        if fqn is not None:
            return fqn.replace("_orig_mod.", "")
        return f"param{index}:{tuple(fsdp_param.sharded_param.shape)}"

    def log(msg: str) -> None:
        if rank == 0:
            event_log.append(f"[{len(event_log):02d}] {msg}")

    for idx, fsdp_param in enumerate(fsdp_params):
        fsdp_param.unsharded_param.register_post_accumulate_grad_hook(
            lambda param, i=idx, p=fsdp_param: log(
                "UNSHARDED_ACCUM_DONE("
                f"{pname(i, p)}, grad_ptr=0x{param.grad.data_ptr():016x})"
                if param.grad is not None
                else f"UNSHARDED_ACCUM_DONE({pname(i, p)}, grad=None)"
            )
        )
        fsdp_param.sharded_param.register_post_accumulate_grad_hook(
            lambda _param, i=idx, p=fsdp_param: log(
                f"SHARDED_POST_ACCUM_HOOK({pname(i, p)})"
            )
        )

    orig_post_backward = param_group.post_backward

    def wrapped_post_backward(self, *unused: Any, _orig=orig_post_backward):
        present = []
        missing = []
        for idx, fsdp_param in enumerate(fsdp_params):
            target = present if fsdp_param.unsharded_param.grad is not None else missing
            target.append(pname(idx, fsdp_param))
        log(
            "POST_BWD_ENTRY("
            f"present={present}, missing={missing}, "
            f"stream={torch.cuda.current_stream().cuda_stream})"
        )
        result = _orig(*unused)
        log("POST_BWD_EXIT")
        return result

    param_group.post_backward = types.MethodType(wrapped_post_backward, param_group)

    orig_rs = dist.reduce_scatter_tensor

    def patched_rs(*rs_args: Any, **rs_kwargs: Any):
        input_t = (
            rs_args[1]
            if len(rs_args) > 1
            else rs_kwargs.get("input", rs_kwargs.get("input_tensor"))
        )
        ptr = input_t.data_ptr() if isinstance(input_t, torch.Tensor) else 0
        log(
            "RS_FIRE("
            f"input_ptr=0x{ptr:016x}, "
            f"stream={torch.cuda.current_stream().cuda_stream})"
        )
        return orig_rs(*rs_args, **rs_kwargs)

    dist.reduce_scatter_tensor = patched_rs

    event_log.clear()
    run_one()

    dist.reduce_scatter_tensor = orig_rs

    if rank == 0:
        mode = "inplace=OFF" if args.no_inplace_buffers else "inplace=ON"
        print(f"\n[unsharded-order-probe] mode={mode} world_size={world_size}")
        for line in event_log:
            print(line)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
