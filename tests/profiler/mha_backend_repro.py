"""MHA-only FSDP2 + torch.compile determinism repro with interchangeable attention backend.

This keeps the xtuner ``MultiHeadAttention`` wrapper path intact:
  q_proj / k_proj / v_proj
  q_norm / k_norm
  rotary embedding
  raw_output reshape + o_proj

The only thing we swap is the attention backend:
  - ``fa2``: xtuner's normal FlashAttention path
  - ``fake_attn``: deterministic custom op with FA2-like backward allocation

If ``fake_attn`` reproduces the same cross-launch non-determinism as ``fa2``,
then FA2 itself is not required. If only ``fa2`` reproduces, then FA2's exact
wrapper / layout / autograd path still seems necessary.
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
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

import torch._inductor.config as inductor_cfg

from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.module.rope.rope import get_rope_embedding


@torch.library.custom_op("mha_backend_repro::fake_attn_fwd", mutates_args=(), device_types="cuda")
def _fake_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    groups = q.shape[1] // k.shape[1]
    k_rep = k.repeat_interleave(groups, dim=1)
    v_rep = v.repeat_interleave(groups, dim=1)
    return q * softmax_scale + k_rep * softmax_scale + v_rep * softmax_scale


@torch.library.register_fake("mha_backend_repro::fake_attn_fwd")
def _fake_attn_fwd_fake(q, k, v, softmax_scale):
    return torch.empty_like(q)


@torch.library.custom_op(
    "mha_backend_repro::fake_attn_bwd",
    mutates_args=("dq", "dk", "dv"),
    device_types="cuda",
)
def _fake_attn_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    softmax_scale: float,
) -> None:
    del q
    del v
    dq.copy_(dout * softmax_scale)
    groups = dq.shape[1] // dk.shape[1]
    dout_kv = dout.view(dout.shape[0], dk.shape[1], groups, dout.shape[2]).sum(dim=2)
    dk.copy_(dout_kv * softmax_scale)
    dv.copy_(dout_kv * softmax_scale)


@torch.library.register_fake("mha_backend_repro::fake_attn_bwd")
def _fake_attn_bwd_fake(dout, q, k, v, dq, dk, dv, softmax_scale):
    return


class FakeAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, softmax_scale):
        out = _fake_attn_fwd(q, k, v, softmax_scale)
        ctx.save_for_backward(q, k, v)
        ctx.softmax_scale = softmax_scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        _fake_attn_bwd(dout.contiguous(), q, k, v, dq, dk, dv, ctx.softmax_scale)
        return dq, dk, dv, None


def fake_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    del cu_seqlens_q
    del cu_seqlens_k
    del max_seqlen_q
    del max_seqlen_k
    del dropout_p
    del causal
    del window_size
    del softcap
    del alibi_slopes
    del deterministic
    del block_table
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    raw_output = FakeAttnFunc.apply(q, k, v, float(softmax_scale))
    if return_attn_probs:
        softmax_lse = torch.zeros((q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device)
        return raw_output, softmax_lse
    return raw_output


def _configure_inductor_trace(trace_dir: str) -> None:
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


def _save_record(base: str, rank: int, grads: dict[str, float]) -> None:
    path = _record_path_for_rank(base, rank)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"rank": rank, "param_grads": grads}, f, indent=2)


def _load_record(base: str, rank: int) -> dict[str, float]:
    with open(_record_path_for_rank(base, rank)) as f:
        return json.load(f)["param_grads"]


def _compare_records(
    new_grads: dict[str, float],
    old_grads: dict[str, float],
) -> list[tuple[str, float, float]]:
    diffs: list[tuple[str, float, float]] = []
    for name in sorted(set(new_grads) | set(old_grads)):
        new_val = new_grads.get(name)
        old_val = old_grads.get(name)
        if new_val != old_val:
            diffs.append((name, new_val, old_val))
    return diffs


def _fully_shard_mha(
    mha: torch.nn.Module,
    *,
    mesh,
    fsdp_cfg: FSDPConfig,
) -> torch.nn.Module:
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
    mesh,
    fsdp_cfg: FSDPConfig,
    compile_forward: bool,
    backend: str,
) -> torch.nn.Module:
    if backend == "fake_attn":
        import xtuner.v1.ops.attn_imp as attn_imp

        attn_imp.flash_attn_varlen_func = fake_flash_attn_varlen_func

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
        grad = param.grad
        local_grad = grad.to_local().float() if hasattr(grad, "to_local") else grad.float()
        grads[name] = local_grad.sum().item()
    return grads


def main() -> None:
    parser = argparse.ArgumentParser(description="xtuner MHA backend determinism repro")
    parser.add_argument("--record-path", required=True, help="Base path; writes <base>_rank<R>.json")
    parser.add_argument("--compare", default=None)
    parser.add_argument("--backend", choices=("fa2", "fake_attn"), required=True)
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--layer-idx", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-inplace-buffers", action="store_true")
    parser.add_argument("--keep-trace", action="store_true")
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
    compile_on = not args.no_compile

    if rank == 0:
        print(
            f"\n[mha_backend_repro] world={world_size}  seq_len={args.seq_len}  "
            f"backend={args.backend}  compile={'ON' if compile_on else 'OFF'}  "
            f"inplace_buffers={'OFF' if args.no_inplace_buffers else 'ON(default)'}"
        )

    if args.keep_trace and compile_on:
        _configure_inductor_trace(f"{args.record_path}_inductor_trace_r{rank}")

    rotary = get_rope_embedding(cfg, device=None).to(device=device)
    mha = _build_fsdp_mha(
        cfg=cfg,
        layer_idx=args.layer_idx,
        device=device,
        dtype=dtype,
        mesh=mesh,
        fsdp_cfg=fsdp_cfg,
        compile_forward=compile_on,
        backend=args.backend,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + rank)
    hidden_states = torch.randn(
        1, args.seq_len, cfg.hidden_size, dtype=dtype, device=device, generator=generator
    )
    dummy_ids = torch.zeros(1, args.seq_len, dtype=torch.long, device=device)
    seq_ctx = SequenceContext.from_input_ids((dummy_ids,), device=str(device))

    t0 = time.time()
    grads = _run_forward_backward(mha, rotary, hidden_states, seq_ctx)
    if rank == 0:
        print(f"[mha_backend_repro] forward+backward done in {time.time() - t0:.1f}s")

    dist.barrier()
    _save_record(args.record_path, rank, grads)
    dist.barrier()

    if args.compare:
        old_grads = _load_record(args.compare, rank)
        diffs = _compare_records(grads, old_grads)
        if diffs:
            max_name = max(len(name) for name, _, _ in diffs)
            print(f"[Rank {rank}] {len(diffs)}/{len(grads)} params differ")
            for name, new_val, old_val in diffs:
                diff = abs(new_val - old_val)
                rel = diff / (abs(old_val) + 1e-12)
                print(
                    f"  {name:<{max_name}}  old={old_val:>16.8e}  "
                    f"new={new_val:>16.8e}  diff={diff:>12.4e}  rel={rel:>10.4e}"
                )
        else:
            print(f"[Rank {rank}] all {len(grads)} grad shard sums identical")

        dist.barrier()
        diff_count = torch.tensor([len(diffs)], dtype=torch.int64, device=device)
        max_rel = max((abs(new_val - old_val) / (abs(old_val) + 1e-12) for _, new_val, old_val in diffs), default=0.0)
        max_rel_tensor = torch.tensor([max_rel], dtype=torch.float64, device=device)
        dist.all_reduce(diff_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(max_rel_tensor, op=dist.ReduceOp.MAX)

        if rank == 0:
            if diff_count.item() == 0:
                print("\nRESULT: FULLY DETERMINISTIC")
            else:
                print(
                    f"\nRESULT: NON-DETERMINISTIC — {diff_count.item()} param shards differ "
                    f"(max relative diff {max_rel_tensor.item():.2e})"
                )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
