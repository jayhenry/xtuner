"""Standalone torchrun script to record and compare FSDP+compiled gradient checksums.

Usage (called by run_test.sh):
    # Run 1 — record grad sums to files:
    torchrun --nproc-per-node 8 numerics_test.py --record-path /tmp/grads/run1

    # Run 2 — record again, then compare against run 1:
    torchrun --nproc-per-node 8 numerics_test.py --record-path /tmp/grads/run2 \
        --compare /tmp/grads/run1

Each rank writes its own JSON file:  <record-path>_rank<N>.json
The comparison is done per-rank (same rank sees its own shard of the reduce-scattered
gradient), then the per-rank "any_diff" flags are reduced across all ranks so the
overall result is printed exactly once on rank 0.

Hypothesis being verified:
    FSDP2 gradient reduce-scatter is non-deterministic *across separate process
    invocations* (different OS scheduling of NCCL ring operations → different
    floating-point accumulation order) even though it is perfectly deterministic
    within a single process.  Two runs of this script with identical model weights
    and inputs should produce slightly different gradient shard sums on at least
    one rank.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist

from xtuner.v1.utils.misc import monkey_patch_hf_modules_cache


# ---------------------------------------------------------------------------
# Model + run helpers (mirroring TestFSDPCompiledMHAGradNumerics)
# ---------------------------------------------------------------------------

def _make_model_config(num_hidden_layers: int = 4):
    from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
    cfg = Qwen3_5_VLTextMoE35BA3BConfig(num_hidden_layers=num_hidden_layers)
    cfg.compile_cfg = True  # match production: TORCH_COMPILE=1
    return cfg


def _build_fsdp_model(hf_path: str, num_hidden_layers: int = 4):
    from xtuner.v1.config import FSDPConfig

    cfg = _make_model_config(num_hidden_layers)
    with torch.device("meta"):
        model = cfg.build()
    fsdp_cfg = FSDPConfig(cpu_offload=False, ep_size=1)
    model.fully_shard(fsdp_config=fsdp_cfg)
    model.from_hf(hf_path, strict=False)
    return model


def _run_forward_backward(model) -> dict[str, float]:
    """Run 2 micro-batches of seq_len=65536, return rank-local grad shard sums."""
    from xtuner.v1.data_proto import SequenceContext
    from xtuner.v1.loss.ce_loss import CELossConfig

    SEQ_LEN = 65536
    GRAD_ACCUM_STEPS = 2

    loss_cfg = CELossConfig(mode="chunk")
    LossCtx = loss_cfg.loss_ctx_cls

    rank = dist.get_rank()
    vocab_size = model.config.vocab_size

    model.zero_grad()
    for micro in range(GRAD_ACCUM_STEPS):
        torch.manual_seed(rank * GRAD_ACCUM_STEPS + micro)
        input_ids = torch.randint(0, vocab_size, (1, SEQ_LEN), device="cuda")
        shifted_labels = input_ids[:, 1:].clone()
        shift_input_ids = input_ids[:, :-1]
        seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids,))
        loss_ctx = loss_cfg.build(shifted_labels=shifted_labels, sp_mesh=None)
        loss_ctx = LossCtx.build_batches([loss_ctx])[0]
        outputs = model(seq_ctx=seq_ctx, loss_ctx=loss_ctx)
        loss = outputs.loss
        if hasattr(outputs, "balancing_loss") and outputs.balancing_loss is not None:
            loss = loss + outputs.balancing_loss
        loss.backward()

    grads: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad
            local_g = g.to_local().float() if hasattr(g, "to_local") else g.float()
            grads[name] = local_g.sum().item()
    return grads


# ---------------------------------------------------------------------------
# Record / compare helpers
# ---------------------------------------------------------------------------

def _record_path_for_rank(base: str, rank: int) -> str:
    return f"{base}_rank{rank}.json"


def save_record(grads: dict, base_path: str, rank: int) -> None:
    path = _record_path_for_rank(base_path, rank)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
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
    """Return list of (name, new_val, old_val) for params that differ."""
    diffs = []
    for name, new_val in new_grads.items():
        old_val = old_grads.get(name)
        if old_val is None:
            continue
        if new_val != old_val:
            diffs.append((name, new_val, old_val))
    return diffs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="FSDP+compiled gradient numerics recorder")
    parser.add_argument(
        "--record-path", required=True,
        help="Base path for output JSON files (per-rank suffix added automatically)",
    )
    parser.add_argument(
        "--compare", default=None,
        help="Base path of a previous recording to compare against",
    )
    parser.add_argument(
        "--hf-path", default=None,
        help="HF model path (overrides QWEN35_MOE_PATH env var)",
    )
    parser.add_argument(
        "--num-hidden-layers", type=int, default=4,
        help="Number of transformer layers in the small model (default: 4)",
    )
    args = parser.parse_args()

    # --- HF module cache patch (required for from_hf()) ---
    monkey_patch_hf_modules_cache()

    # --- Distributed init ---
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    torch.accelerator.set_device_index(local_rank)

    hf_path = args.hf_path or os.environ.get("QWEN35_MOE_PATH")
    if not hf_path:
        if rank == 0:
            print("ERROR: QWEN35_MOE_PATH env var or --hf-path must be set", file=sys.stderr)
        dist.destroy_process_group()
        sys.exit(1)

    if rank == 0:
        print(f"[numerics_test] world_size={world_size}  record_path={args.record_path}")
        if args.compare:
            print(f"[numerics_test] comparing against: {args.compare}")

    # --- Build model and run ---
    t0 = time.time()
    if rank == 0:
        print("[numerics_test] building FSDP+compiled model ...")
    torch._dynamo.reset()
    model = _build_fsdp_model(hf_path, num_hidden_layers=args.num_hidden_layers)

    if rank == 0:
        print("[numerics_test] running forward+backward ...")
    new_grads = _run_forward_backward(model)
    elapsed = time.time() - t0
    if rank == 0:
        print(f"[numerics_test] done in {elapsed:.1f}s  ({len(new_grads)} params with grad)")

    # --- Save this run's records ---
    dist.barrier()
    save_record(new_grads, args.record_path, rank)
    dist.barrier()
    if rank == 0:
        print(f"[numerics_test] saved records to {args.record_path}_rank*.json")

    # --- Compare against previous run (if requested) ---
    if args.compare:
        old_grads = load_record(args.compare, rank)
        diffs = compare_records(new_grads, old_grads, rank)

        # Print per-rank diff details (only if there are diffs on this rank)
        if diffs:
            lines = [f"[Rank {rank}] {len(diffs)}/{len(new_grads)} params differ:"]
            for name, nv, ov in diffs[:10]:
                lines.append(
                    f"  {name}: run1={ov:.12f}  run2={nv:.12f}  "
                    f"diff={abs(nv - ov):.4e}  rel={abs(nv-ov)/(abs(ov)+1e-12):.4e}"
                )
            print("\n".join(lines))
        else:
            print(f"[Rank {rank}] all {len(new_grads)} grad shard sums identical to run 1")

        # Reduce "any diff" flag across all ranks
        dist.barrier()
        diff_flag = torch.tensor([1 if diffs else 0], dtype=torch.int64, device="cuda")
        dist.all_reduce(diff_flag, op=dist.ReduceOp.MAX)
        any_rank_differs = diff_flag.item() > 0

        if rank == 0:
            print()
            if any_rank_differs:
                print(
                    "RESULT: CONFIRMED — cross-process NCCL non-determinism reproduced.\n"
                    "At least one rank produced different gradient shard sums between the\n"
                    "two separate torchrun invocations (different NCCL ring timing)."
                )
                sys.exit(0)
            else:
                print(
                    "RESULT: NOT REPRODUCED — all gradient shard sums are identical across\n"
                    "both runs on every rank.  The hypothesis is not confirmed by this run."
                )
                sys.exit(2)  # distinct from error (1) to signal "not reproduced"

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
