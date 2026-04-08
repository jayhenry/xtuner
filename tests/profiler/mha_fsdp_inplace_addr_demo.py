"""Probe the FSDP2 + torch.compile nondeterminism at pointer and codegen level.

This script started as a strict pointer-alias probe for the original hypothesis:

    compiled backward buffer == FSDP2 pre-backward all-gather buffer

That direct same-address claim is still not proven. To make the root cause
clearer, this probe now reports two kinds of evidence in one place:

1. Runtime pointer / stream evidence from FSDP2 pre-backward.
2. TorchInductor codegen evidence for whether backward reuses saved ``mm_1``
   in-place on the K path or switches to a fresh buffer instead.

It also supports ``backend=fa2`` and ``backend=fake_attn`` so we can check
whether the same compiled backward reuse pattern appears without FA2 itself.

Compared with ``demo_race_condition.py``, this script avoids reading tensor
contents after backward. That makes it safer when the buggy configuration has
already corrupted a buffer or triggered an asynchronous illegal access.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch._inductor.config as inductor_cfg
from mmengine.runner import set_random_seed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.module.rope.rope import get_rope_embedding


@torch.library.custom_op(
    "mha_fsdp_inplace_addr_demo::fake_attn_fwd",
    mutates_args=(),
    device_types="cuda",
)
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


@torch.library.register_fake("mha_fsdp_inplace_addr_demo::fake_attn_fwd")
def _fake_attn_fwd_fake(q, k, v, softmax_scale):
    return torch.empty_like(q)


@torch.library.custom_op(
    "mha_fsdp_inplace_addr_demo::fake_attn_bwd",
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


@torch.library.register_fake("mha_fsdp_inplace_addr_demo::fake_attn_bwd")
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


def _get_module_fsdp_state(module: torch.nn.Module) -> Any:
    from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state

    return _get_module_fsdp_state(module)


def _iter_fsdp_params(module: torch.nn.Module):
    state = _get_module_fsdp_state(module)
    if state is None or state._fsdp_param_group is None:
        return
    yield from state._fsdp_param_group.fsdp_params


def _find_output_code(trace_dir: str, kind: str | None = None) -> str | None:
    files = sorted(glob.glob(os.path.join(trace_dir, "**", "output_code.py"), recursive=True))
    if kind is not None:
        files = [path for path in files if kind in path]
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _summarize_output_code(trace_dir: str) -> dict[str, Any]:
    backward_path = _find_output_code(trace_dir, "backward")
    forward_path = _find_output_code(trace_dir, "forward")
    if backward_path is None:
        return {
            "backward_path": None,
            "forward_path": forward_path,
            "in_out_ptr_count": 0,
            "k_norm_reuse_source": None,
            "forward_has_k_proj_mm": None,
        }
    with open(backward_path) as f:
        backward_code = f.read()
    reuse_source = None
    match = re.search(
        r"reinterpret_tensor\((mm_1|buf3),\s*\(1,\s*\d+,\s*2,\s*256\).+?# reuse",
        backward_code,
    )
    if match:
        reuse_source = match.group(1)
    forward_has_k_proj_mm = None
    if forward_path is not None:
        with open(forward_path) as f:
            forward_code = f.read()
        forward_has_k_proj_mm = bool(
            re.search(
                r"buf1 = empty_strided_cuda\(\(\d+,\s*512\).+?"
                r"extern_kernels\.mm\(.+?primals_3.+?out=buf1",
                forward_code,
                re.S,
            )
        )
    return {
        "backward_path": backward_path,
        "forward_path": forward_path,
        "in_out_ptr_count": backward_code.count("in_out_ptr"),
        "k_norm_reuse_source": reuse_source,
        "forward_has_k_proj_mm": forward_has_k_proj_mm,
    }


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


def _stream_label(stream: object | None) -> str:
    if stream is None:
        return "None"
    cuda_stream = getattr(stream, "cuda_stream", None)
    if cuda_stream is None:
        return repr(stream)
    return f"cuda_stream=0x{int(cuda_stream):x}"


@dataclass
class PreBackwardRecorder:
    module: torch.nn.Module
    pre_bwd_ptrs: dict[str, int]
    stream_info: dict[str, str] | None
    pre_bwd_calls: int
    _orig_pre_backward: Any = None
    _target_state: Any = None

    @classmethod
    def install(cls, module: torch.nn.Module) -> "PreBackwardRecorder":
        from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

        recorder = cls(module=module, pre_bwd_ptrs={}, stream_info=None, pre_bwd_calls=0)
        recorder._orig_pre_backward = FSDPState._pre_backward
        recorder._target_state = _get_module_fsdp_state(module)
        orig_pre_backward = recorder._orig_pre_backward
        target_state = recorder._target_state

        def _wrapped_pre_backward(state, grad: torch.Tensor) -> torch.Tensor:
            grad = orig_pre_backward(state, grad)
            if state is target_state and state._fsdp_param_group is not None:
                recorder.pre_bwd_calls += 1
                comm = getattr(state, "_comm_ctx", None)
                recorder.stream_info = {
                    "current_stream": _stream_label(torch.cuda.current_stream()),
                    "all_gather_stream": _stream_label(getattr(comm, "all_gather_stream", None)),
                    "all_gather_copy_in_stream": _stream_label(getattr(comm, "all_gather_copy_in_stream", None)),
                }
                for fsdp_param in state._fsdp_param_group.fsdp_params:
                    tensor = getattr(fsdp_param, "_unsharded_param", None)
                    if tensor is not None:
                        recorder.pre_bwd_ptrs[fsdp_param._param_fqn] = tensor.data_ptr()
            return grad

        FSDPState._pre_backward = _wrapped_pre_backward
        return recorder

    def uninstall(self) -> None:
        from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

        if self._orig_pre_backward is not None:
            FSDPState._pre_backward = self._orig_pre_backward

    def capture_from_live_params(self) -> None:
        for fsdp_param in _iter_fsdp_params(self.module):
            tensor = getattr(fsdp_param, "_unsharded_param", None)
            if tensor is not None:
                self.pre_bwd_ptrs.setdefault(fsdp_param._param_fqn, tensor.data_ptr())


def _collect_pointer_records(
    mha: torch.nn.Module,
    rotary: torch.nn.Module,
    hidden_states: torch.Tensor,
    seq_ctx: SequenceContext,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str] | None, int, str | None]:
    recorder = PreBackwardRecorder.install(mha)
    sync_error: str | None = None
    saved_tensors: list[dict[str, Any]] = []
    try:
        position_embeddings = rotary(hidden_states, seq_ctx.position_ids)  # type: ignore[arg-type]
        def _pack_saved_tensor(tensor: torch.Tensor) -> torch.Tensor:
            saved_tensors.append(
                {
                    "shape": tuple(tensor.shape),
                    "stride": tuple(tensor.stride()),
                    "dtype": str(tensor.dtype),
                    "data_ptr": tensor.data_ptr(),
                }
            )
            return tensor

        def _unpack_saved_tensor(tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(_pack_saved_tensor, _unpack_saved_tensor):
            mha.zero_grad(set_to_none=True)
            out = mha(hidden_states, position_embeddings, seq_ctx)
            projected = out["projected_output"]

            def _fallback_capture(grad: torch.Tensor) -> torch.Tensor:
                recorder.capture_from_live_params()
                return grad

            projected.register_hook(_fallback_capture)
            mha.set_requires_gradient_sync(False)
            projected.float().sum().backward()

        records: list[dict[str, Any]] = []
        for fsdp_param in _iter_fsdp_params(mha):
            name = fsdp_param._param_fqn
            pre_ptr = recorder.pre_bwd_ptrs.get(name)
            if pre_ptr is None:
                continue

            grad_ptrs: dict[str, int] = {}
            unsharded_accum_grad = getattr(fsdp_param, "unsharded_accumulated_grad", None)
            if unsharded_accum_grad is not None:
                grad_ptrs["unsharded_accumulated_grad"] = unsharded_accum_grad.data_ptr()

            unsharded_param = getattr(fsdp_param, "_unsharded_param", None)
            unsharded_param_grad = None if unsharded_param is None else unsharded_param.grad
            if unsharded_param_grad is not None:
                grad_ptrs["_unsharded_param.grad"] = unsharded_param_grad.data_ptr()

            records.append(
                {
                    "param_fqn": name,
                    "pre_bwd_ptr": pre_ptr,
                    "post_bwd_grad_ptrs": grad_ptrs,
                    "same_addr_locations": [loc for loc, ptr in grad_ptrs.items() if ptr == pre_ptr],
                }
            )

        try:
            torch.cuda.synchronize()
        except Exception as exc:  # pragma: no cover - debugging path
            sync_error = f"{type(exc).__name__}: {exc}"

        return records, saved_tensors, recorder.stream_info, recorder.pre_bwd_calls, sync_error
    finally:
        mha.set_requires_gradient_sync(True)
        recorder.uninstall()


def _format_hex(ptr: int | None) -> str:
    if ptr is None:
        return "None"
    return f"0x{ptr:018x}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pointer-level demo for the FSDP2 + compile inplace-buffer race")
    parser.add_argument("--backend", choices=("fa2", "fake_attn"), default="fa2")
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--layer-idx", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-inplace-buffers", action="store_true")
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="Base dir for per-rank TorchInductor traces; if omitted, no trace is collected.",
    )
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

    if not torch.cuda.is_available():
        if rank == 0:
            print("ERROR: CUDA is required", file=sys.stderr)
        dist.destroy_process_group()
        sys.exit(1)

    torch.cuda.set_device(local_rank)
    torch.accelerator.set_device_index(local_rank)

    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    reduce_dtype = torch.float32 if args.deterministic else torch.bfloat16
    fsdp_cfg = FSDPConfig(cpu_offload=False, ep_size=1, reduce_dtype=reduce_dtype)
    mesh = init_device_mesh("cuda", (world_size,))
    cfg = Qwen3_5_VLTextMoE35BA3BConfig()
    compile_on = not args.no_compile

    trace_summary: dict[str, Any] | None = None
    if args.trace_dir and compile_on:
        trace_dir = f"{args.trace_dir}_r{rank}"
        _configure_inductor_trace(trace_dir)
    else:
        trace_dir = None

    if rank == 0:
        mode = "inplace_buffers=False" if args.no_inplace_buffers else "inplace_buffers=True"
        print(
            f"\n[mha_fsdp_inplace_addr_demo] world={world_size} seq_len={args.seq_len} "
            f"backend={args.backend} compile={'ON' if compile_on else 'OFF'} {mode}"
        )

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
    hidden_states = torch.randn(1, args.seq_len, cfg.hidden_size, dtype=dtype, device=device, generator=generator)
    dummy_ids = torch.zeros(1, args.seq_len, dtype=torch.long, device=device)
    seq_ctx = SequenceContext.from_input_ids((dummy_ids,), device=str(device))

    t0 = time.time()
    records, saved_tensors, stream_info, pre_bwd_calls, sync_error = _collect_pointer_records(
        mha, rotary, hidden_states, seq_ctx
    )
    elapsed = time.time() - t0

    if trace_dir is not None:
        trace_summary = _summarize_output_code(trace_dir)

    payload = {
        "rank": rank,
        "elapsed_s": elapsed,
        "pre_bwd_calls": pre_bwd_calls,
        "stream_info": stream_info,
        "records": records,
        "saved_tensors": saved_tensors,
        "sync_error": sync_error,
        "trace_summary": trace_summary,
    }

    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, payload)

    if rank == 0:
        print("\n[Runtime Evidence]")
        any_kproj_saved_match = False
        any_buggy_mm1_reuse = False
        for item in gathered:
            print(f"\nRank {item['rank']}  elapsed={item['elapsed_s']:.1f}s  pre_bwd_calls={item['pre_bwd_calls']}")
            if item["stream_info"] is not None:
                info = item["stream_info"]
                print(
                    "  streams: "
                    f"current={info['current_stream']}  "
                    f"all_gather={info['all_gather_stream']}  "
                    f"copy_in={info['all_gather_copy_in_stream']}"
                )
            if item["sync_error"]:
                print(f"  torch.cuda.synchronize(): {item['sync_error']}")

            if not item["records"]:
                print("  no FSDP parameter pointers captured")
                continue

            for record in item["records"]:
                name = record["param_fqn"]
                pre_ptr = record["pre_bwd_ptr"]
                grad_ptrs = record["post_bwd_grad_ptrs"]
                same_locs = record["same_addr_locations"]
                saved_matches = [t for t in item["saved_tensors"] if t["data_ptr"] == pre_ptr]
                if not grad_ptrs:
                    print(f"  {name}: pre_bwd_ptr={_format_hex(pre_ptr)}  no post-bwd grad ptrs")
                    continue
                for loc, grad_ptr in grad_ptrs.items():
                    marker = "SAME ADDR" if grad_ptr == pre_ptr else "different addr"
                    print(
                        f"  {name}: pre_bwd_ptr={_format_hex(pre_ptr)}  "
                        f"{loc}={_format_hex(grad_ptr)}  -> {marker}"
                    )
                if same_locs:
                    print(f"    alias locations: {', '.join(same_locs)}")
                if name.endswith("k_proj.weight") and saved_matches:
                    any_kproj_saved_match = True
                    print(f"    saved-tensor matches ({len(saved_matches)}):")
                    for saved in saved_matches[:4]:
                        print(
                            f"      shape={saved['shape']} stride={saved['stride']} "
                            f"dtype={saved['dtype']} ptr={_format_hex(saved['data_ptr'])}"
                        )

        print("\n[Inductor Trace]")
        for item in gathered:
            summary = item["trace_summary"]
            if summary is None:
                print(f"  rank {item['rank']}: trace disabled")
                continue
            if summary["k_norm_reuse_source"] == "mm_1" and summary["in_out_ptr_count"] > 0:
                any_buggy_mm1_reuse = True
            print(
                f"  rank {item['rank']}: in_out_ptr_count={summary['in_out_ptr_count']}  "
                f"k_norm_reuse_source={summary['k_norm_reuse_source']}  "
                f"forward_has_k_proj_mm={summary['forward_has_k_proj_mm']}  "
                f"backward_path={summary['backward_path']}"
            )

        print("\n[Verdict]")
        if any_kproj_saved_match:
            print(
                "  SAME GPU ADDR observed between `k_proj.weight`'s FSDP2 pre-backward "
                "buffer and a saved forward tensor."
            )
            print(
                "  Combine that with `k_norm_reuse_source=mm_1` and nonzero `in_out_ptr_count`: "
                "the compiled backward is reusing that saved tensor storage in-place."
            )
        elif any_buggy_mm1_reuse:
            print(
                "  No direct runtime pointer match to `k_proj.weight` was captured in this run, "
                "so the original 'weight all-gather buffer == backward in-place buffer' claim is "
                "not proven here."
            )
            print(
                "  The strongest evidence is codegen-level: buggy mode reuses saved `mm_1` "
                "in-place on the K path, while fixed mode switches that reuse source to "
                "`buf3` and removes all `in_out_ptr*` backward kernels."
            )
            print(
                "  In the forward trace, `buf1` is the saved `k_proj` matmul output, so "
                "the practical issue is compiled backward mutation of saved K-path activation "
                "storage rather than a proven direct alias to the FSDP weight all-gather buffer."
            )
        else:
            print("  No direct same-address evidence was captured in this run.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
