"""Same as test_flatgrad_at_rs.py but WITHOUT torch.cuda.synchronize() before snapshot.
This shows the ACTUAL flat_grad state when RS fires (with in-flight stream0 writes).
"""
import argparse, os
import numpy as np
import torch
import torch.distributed as dist
from mmengine.runner import set_random_seed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.module.rope.rope import get_rope_embedding

parser = argparse.ArgumentParser()
parser.add_argument("--save-dir",  default="/tmp/flatgrad_nosync_a")
parser.add_argument("--compare",   default=None)
args = parser.parse_args()

os.environ.setdefault("NCCL_ALGO",        "Ring")
os.environ.setdefault("NCCL_PROTO",       "Simple")
os.environ.setdefault("NCCL_NUM_CHANNELS","1")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True, warn_only=True)

set_random_seed(0, deterministic=True)
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
torch.cuda.set_device(local_rank)

device = torch.device("cuda", local_rank)
dtype  = torch.bfloat16
cfg    = Qwen3_5_VLTextMoE35BA3BConfig()
fsdp_cfg = FSDPConfig(cpu_offload=False, ep_size=1, reduce_dtype=torch.float32)
mesh = init_device_mesh("cuda", (world_size,))

mha = cfg.attention.build(hidden_size=cfg.hidden_size, layer_type=None,
                           layer_idx=3, rope_scaling_cfg=cfg.rope_scaling_cfg)
mha = mha.to(device=device, dtype=dtype)
torch._dynamo.reset()
mha = torch.compile(mha, fullgraph=True)
mp_policy = MixedPrecisionPolicy(param_dtype=fsdp_cfg.param_dtype,
                                  reduce_dtype=fsdp_cfg.reduce_dtype)
fully_shard(mha, mesh=mesh, mp_policy=mp_policy,
            reshard_after_forward=True, offload_policy=None)

rotary = get_rope_embedding(cfg, device=None).to(device=device)
g = torch.Generator(device=device)
g.manual_seed(rank)
hidden_states = torch.randn(1, 65536, cfg.hidden_size,
                            dtype=dtype, device=device, generator=g)
dummy_ids = torch.zeros(1, 65536, dtype=torch.long, device=device)
seq_ctx = SequenceContext.from_input_ids((dummy_ids,), device=str(device))

_orig_rs = dist.reduce_scatter_tensor
_rs_snap = {}

def _patched_rs(*a, **kw):
    input_t = a[1] if len(a) > 1 else kw.get("input", kw.get("input_tensor"))
    if input_t is not None and isinstance(input_t, torch.Tensor):
        # NO synchronize() here — capture flat_grad with in-flight stream0 writes
        _rs_snap[len(_rs_snap)] = input_t.detach().clone().cpu()
    return _orig_rs(*a, **kw)

# Warm-up
if rank == 0: print("\n[Warm-up]")
pos_emb = rotary(hidden_states, seq_ctx.position_ids)
mha.zero_grad(set_to_none=True)
out = mha(hidden_states, pos_emb, seq_ctx)
out["projected_output"].float().sum().backward()
torch.cuda.synchronize(); dist.barrier()

# Probe pass (NO sync before RS snapshot)
if rank == 0: print("[Probe pass — NO sync before RS snapshot]")
_rs_snap.clear()
dist.reduce_scatter_tensor = _patched_rs

pos_emb = rotary(hidden_states, seq_ctx.position_ids)
mha.zero_grad(set_to_none=True)
out = mha(hidden_states, pos_emb, seq_ctx)
out["projected_output"].float().sum().backward()
torch.cuda.synchronize()

dist.reduce_scatter_tensor = _orig_rs
dist.barrier()

flat_grad_at_rs = _rs_snap.get(0)

# Collect k_proj grad after backward
k_proj_grad_after = None
for name, param in mha.named_parameters():
    if "k_proj" in name and "weight" in name and param.grad is not None:
        g2 = param.grad
        if hasattr(g2, "_local_tensor"): g2 = g2._local_tensor
        k_proj_grad_after = g2.detach().contiguous().float().cpu()
        break

if rank == 0 and flat_grad_at_rs is not None:
    print(f"  flat_grad at RS time (NO sync): {flat_grad_at_rs.numel()} float32 elements, "
          f"{flat_grad_at_rs.numel()*4//1024//1024}MB")
    # Are ALL values zero? (would mean AccumulateGrad hasn't run yet)
    n_nonzero = (flat_grad_at_rs != 0).sum().item()
    print(f"  non-zero elements: {n_nonzero}/{flat_grad_at_rs.numel()} "
          f"({'all zero' if n_nonzero==0 else 'some nonzero'})")

os.makedirs(args.save_dir, exist_ok=True)
if flat_grad_at_rs is not None:
    np.save(f"{args.save_dir}/rank{rank}_fg.npy", flat_grad_at_rs.numpy())
if k_proj_grad_after is not None:
    np.save(f"{args.save_dir}/rank{rank}_kg.npy", k_proj_grad_after.numpy())
dist.barrier()

if args.compare is not None and rank == 0:
    print(f"\n[Cross-launch compare vs {args.compare}/]")
    fg_ref = np.load(f"{args.compare}/rank0_fg.npy")
    fg_cur = flat_grad_at_rs.numpy() if flat_grad_at_rs is not None else None
    if fg_cur is not None:
        diff = np.abs(fg_cur.astype(np.float32) - fg_ref.astype(np.float32))
        n_diff = (diff > 0).sum()
        print(f"  flat_grad (NO sync): n_differ={n_diff}/{diff.size}  "
              f"max_diff={diff.max():.4e}  {'DIFFERS' if n_diff else 'IDENTICAL'}")
        if n_diff > 0:
            print(f"  *** flat_grad ALREADY non-deterministic at RS time (with in-flight stream0) ***")
            print(f"  → The race IS on flat_grad (concurrent AccumulateGrad write vs RS read)")
        else:
            print(f"  flat_grad IDENTICAL (even with in-flight stream0) → race not on flat_grad")
    kg_ref = np.load(f"{args.compare}/rank0_kg.npy")
    kg_cur = k_proj_grad_after.numpy() if k_proj_grad_after is not None else None
    if kg_cur is not None:
        diff2 = np.abs(kg_cur.astype(np.float32) - kg_ref.astype(np.float32))
        n_diff2 = (diff2 > 0).sum()
        print(f"  k_proj.weight.grad: n_differ={n_diff2}/{diff2.size}  "
              f"max_diff={diff2.max():.4e}  {'DIFFERS' if n_diff2 else 'IDENTICAL'}")

dist.barrier()
dist.destroy_process_group()
