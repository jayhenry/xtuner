"""Test: does disabling reshard_after_forward (no pre-bwd all-gather) fix the non-determinism?"""
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

parser = argparse.ArgumentParser()
parser.add_argument("--seq-len", type=int, default=65536)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no-inplace-buffers", action="store_true")
parser.add_argument("--no-reshard", action="store_true", help="Disable reshard_after_forward")
parser.add_argument("--save-dir", default="/tmp/no_reshard_probe")
parser.add_argument("--compare", default=None)
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

mha = cfg.attention.build(hidden_size=cfg.hidden_size, layer_type=None, layer_idx=3, rope_scaling_cfg=cfg.rope_scaling_cfg)
mha = mha.to(device=device, dtype=dtype)
torch._dynamo.reset()
mha = torch.compile(mha, fullgraph=True)
mp_policy = MixedPrecisionPolicy(param_dtype=fsdp_cfg.param_dtype, reduce_dtype=fsdp_cfg.reduce_dtype)
fully_shard(mha, mesh=mesh, mp_policy=mp_policy,
            reshard_after_forward=not args.no_reshard,
            offload_policy=None)

if rank == 0:
    mode = "no_inplace" if args.no_inplace_buffers else "inplace"
    reshard = "no_reshard" if args.no_reshard else "reshard"
    print(f"\n{'='*60}\n  {mode} + {reshard}\n{'='*60}")

rotary = get_rope_embedding(cfg, device=None).to(device=device)
g = torch.Generator(device=device)
g.manual_seed(args.seed + rank)
hidden_states = torch.randn(1, args.seq_len, cfg.hidden_size, dtype=dtype, device=device, generator=g)
dummy_ids = torch.zeros(1, args.seq_len, dtype=torch.long, device=device)
seq_ctx = SequenceContext.from_input_ids((dummy_ids,), device=str(device))

# Warm-up
pos_emb = rotary(hidden_states, seq_ctx.position_ids)
mha.zero_grad(set_to_none=True)
out = mha(hidden_states, pos_emb, seq_ctx)
out["projected_output"].float().sum().backward()
torch.cuda.synchronize(); dist.barrier()

# Probe pass
pos_emb = rotary(hidden_states, seq_ctx.position_ids)
mha.zero_grad(set_to_none=True)
out = mha(hidden_states, pos_emb, seq_ctx)
out["projected_output"].float().sum().backward()
torch.cuda.synchronize(); dist.barrier()

# Get k_proj gradient
k_proj_grad = None
for name, param in mha.named_parameters():
    if "k_proj" in name and "weight" in name and param.grad is not None:
        grad_t = param.grad
        if hasattr(grad_t, "_local_tensor"):
            grad_t = grad_t._local_tensor
        k_proj_grad = grad_t.detach().contiguous().float().cpu().numpy()
        break

os.makedirs(args.save_dir, exist_ok=True)
if k_proj_grad is not None:
    np.save(os.path.join(args.save_dir, f"rank{rank}_kgrad.npy"), k_proj_grad)
dist.barrier()

if args.compare is not None:
    ref_path = os.path.join(args.compare, f"rank{rank}_kgrad.npy")
    if k_proj_grad is not None and os.path.exists(ref_path):
        ref = np.load(ref_path)
        diff = np.abs(k_proj_grad - ref)
        n_diff = (diff > 0).sum()
        max_diff = diff.max()
        print(f"  rank {rank}: n_differ={n_diff}/{k_proj_grad.size}  max_abs_diff={max_diff:.4e}  {'DIFFERS' if n_diff else 'IDENTICAL'}")

dist.barrier()
dist.destroy_process_group()
