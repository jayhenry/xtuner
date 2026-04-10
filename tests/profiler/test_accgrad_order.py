"""test_accgrad_order.py — Capture the order in which AccumulateGrad hooks fire
for each MHA parameter, and when the RS fires, to confirm:

  With inplace_buffers=True:  k_proj's AccumulateGrad fires LAST → RS fires
     immediately after, while k_proj's accum kernel is still writing to flat_grad.
  With inplace_buffers=False: k_proj's AccumulateGrad fires EARLIER → by the time
     RS fires, k_proj's flat_grad slice is already fully written.

The key: FSDP2 fires RS right after the LAST AccumulateGrad hook fires on the CPU.
But the GPU accumulation kernel for that last param is still in-flight on stream0.
COMM stream reads flat_grad (incl. the last param's slice) while stream0 writes it.

Launch:
  PYTHONPATH=. XTUNER_DETERMINISTIC=true torchrun --nproc-per-node 4 \\
    /tmp/test_accgrad_order.py [--no-inplace-buffers]
"""
import argparse, os
import torch
import torch._inductor.config as inductor_cfg
import torch.distributed as dist
from mmengine.runner import set_random_seed
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from xtuner.v1.config import FSDPConfig
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.module.rope.rope import get_rope_embedding

parser = argparse.ArgumentParser()
parser.add_argument("--seq-len", type=int, default=65536)
parser.add_argument("--seed",    type=int, default=0)
parser.add_argument("--no-inplace-buffers", action="store_true")
args = parser.parse_args()

if args.no_inplace_buffers:
    inductor_cfg.inplace_buffers = False

os.environ.setdefault("NCCL_ALGO", "Ring")
os.environ.setdefault("NCCL_PROTO", "Simple")
os.environ.setdefault("NCCL_NUM_CHANNELS", "1")
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
hidden_states = torch.randn(1, args.seq_len, cfg.hidden_size,
                            dtype=dtype, device=device, generator=g)
dummy_ids = torch.zeros(1, args.seq_len, dtype=torch.long, device=device)
seq_ctx = SequenceContext.from_input_ids((dummy_ids,), device=str(device))

mode = "FIXED" if args.no_inplace_buffers else "BUGGY"
if rank == 0:
    print(f"\n{'='*60}\n  AccumulateGrad order probe — {mode}\n{'='*60}")

# Warm-up
pos_emb = rotary(hidden_states, seq_ctx.position_ids)
mha.zero_grad(set_to_none=True)
out = mha(hidden_states, pos_emb, seq_ctx)
out["projected_output"].float().sum().backward()
torch.cuda.synchronize(); dist.barrier()

# ── Patch RS to record when it fires ──────────────────────────────────────────
_orig_rs = dist.reduce_scatter_tensor
_event_log = []   # list of (order_idx, event_name)

def _patched_rs(*a, **kw):
    if rank == 0:
        _event_log.append((len(_event_log), "RS_FIRE"))
    return _orig_rs(*a, **kw)

dist.reduce_scatter_tensor = _patched_rs

# ── Register post-accumulate hooks on each param ──────────────────────────────
try:
    register_fn = torch.nn.Parameter.register_post_accumulate_grad_hook
    hook_available = True
except AttributeError:
    hook_available = False
    if rank == 0:
        print("  register_post_accumulate_grad_hook not available on this PyTorch version")
        print("  Falling back to register_hook (fires before AccumulateGrad)")

if hook_available:
    for name, param in mha.named_parameters():
        short = name.replace("_orig_mod.", "")
        def make_hook(n):
            def hook(p):
                if rank == 0:
                    _event_log.append((len(_event_log), f"ACCUM_DONE({n})"))
            return hook
        param.register_post_accumulate_grad_hook(make_hook(short))
else:
    for name, param in mha.named_parameters():
        short = name.replace("_orig_mod.", "")
        def make_hook(n):
            def hook(g):
                if rank == 0:
                    _event_log.append((len(_event_log), f"GRAD_HOOK({n})"))
                return g
            return hook
        param.register_hook(make_hook(short))

# ── Probe pass ─────────────────────────────────────────────────────────────────
if rank == 0:
    print("\n[Probe pass]")
_event_log.clear()

pos_emb = rotary(hidden_states, seq_ctx.position_ids)
mha.zero_grad(set_to_none=True)
out = mha(hidden_states, pos_emb, seq_ctx)
out["projected_output"].float().sum().backward()
torch.cuda.synchronize()

dist.reduce_scatter_tensor = _orig_rs

if rank == 0:
    print(f"\n  CPU-side AccumulateGrad + RS fire order (mode={mode}):")
    for idx, name in _event_log:
        marker = "  <-- RS fires here" if "RS_FIRE" in name else ""
        print(f"    [{idx:2d}] {name}{marker}")

    # Find which ACCUM fires just before RS
    rs_pos = next((i for i, (idx, n) in enumerate(_event_log) if "RS_FIRE" in n), None)
    if rs_pos is not None and rs_pos > 0:
        last_before_rs = _event_log[rs_pos - 1][1]
        print(f"\n  Event right before RS: '{last_before_rs}'")
        if "ACCUM_DONE(k_proj" in last_before_rs or "GRAD_HOOK(k_proj" in last_before_rs:
            print(f"  *** k_proj is LAST before RS — CONFIRMS the race on flat_grad[k_proj_slice] ***")
        else:
            print(f"  '{last_before_rs}' is last before RS — the race is on that param's flat_grad slice")

dist.barrier()
dist.destroy_process_group()
