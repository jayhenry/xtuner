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
from mmengine.runner import set_random_seed

from xtuner.v1.utils.misc import monkey_patch_hf_modules_cache


# ---------------------------------------------------------------------------
# Model + run helpers (mirroring TestFSDPCompiledMHAGradNumerics)
# ---------------------------------------------------------------------------

def _make_model_config(num_hidden_layers: int = 4, compile: bool = True):
    from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
    cfg = Qwen3_5_VLTextMoE35BA3BConfig(num_hidden_layers=num_hidden_layers)
    cfg.compile_cfg = compile  # match production: TORCH_COMPILE=1
    return cfg


def _build_fsdp_model(hf_path: str, num_hidden_layers: int = 4, reduce_dtype=torch.bfloat16,
                      compile: bool = True):
    from xtuner.v1.config import FSDPConfig

    cfg = _make_model_config(num_hidden_layers, compile=compile)
    with torch.device("meta"):
        model = cfg.build()
    fsdp_cfg = FSDPConfig(cpu_offload=False, ep_size=1, reduce_dtype=reduce_dtype)
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
    parser.add_argument(
        "--deterministic", action="store_true",
        help=(
            "Force NCCL to use Ring algorithm + Simple protocol so that "
            "reduce-scatter produces the same result across process invocations. "
            "Must be set before dist.init_process_group."
        ),
    )
    parser.add_argument(
        "--no-compile", action="store_true",
        help="Disable torch.compile (use eager mode). Default: compile is ON.",
    )
    args = parser.parse_args()

    # --- HF module cache patch (required for from_hf()) ---
    monkey_patch_hf_modules_cache()

    # --- Force deterministic NCCL *before* init_process_group (env vars are
    #     read by NCCL at init time; setting them afterwards has no effect).
    # NOTE: XTUNER_DETERMINISTIC is a module-level constant evaluated at import
    # time in xtuner.v1.utils.misc.  It must be set in the *shell environment*
    # before torchrun launches — setting os.environ here is too late.
    # run_test.sh passes `XTUNER_DETERMINISTIC=true` for the deterministic runs.
    if args.deterministic:
        # Belt-and-suspenders: also pin NCCL to a single ring channel.
        os.environ.setdefault("NCCL_ALGO", "Ring")
        os.environ.setdefault("NCCL_PROTO", "Simple")
        os.environ.setdefault("NCCL_NUM_CHANNELS", "1")

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True, warn_only=True)

    set_random_seed(0, deterministic=args.deterministic)

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

    from xtuner.v1.utils import XTUNER_DETERMINISTIC
    compile_mode = not args.no_compile
    if rank == 0:
        if args.deterministic:
            det = (
                f"  deterministic=True  XTUNER_DETERMINISTIC={XTUNER_DETERMINISTIC}"
                f"  NCCL_ALGO={os.environ.get('NCCL_ALGO','(default)')}"
                f"  NCCL_NUM_CHANNELS={os.environ.get('NCCL_NUM_CHANNELS','(default)')}"
                f"  reduce_dtype=float32"
            )
        else:
            det = f"  deterministic=False  XTUNER_DETERMINISTIC={XTUNER_DETERMINISTIC}"
        compile_str = "compile=ON" if compile_mode else "compile=OFF (eager)"
        print(f"[numerics_test] world_size={world_size}  {compile_str}  record_path={args.record_path}{det}")
        if args.compare:
            print(f"[numerics_test] comparing against: {args.compare}")

    # --- Build model and run ---
    t0 = time.time()
    if rank == 0:
        print("[numerics_test] building FSDP+compiled model ...")
    reduce_dtype = torch.float32 if args.deterministic else torch.bfloat16
    torch._dynamo.reset()
    model = _build_fsdp_model(
        hf_path,
        num_hidden_layers=args.num_hidden_layers,
        reduce_dtype=reduce_dtype,
        compile=compile_mode,
    )

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
            col_w = max(len(name) for name, _, _ in diffs)
            col_w = max(col_w, 10)
            header = (
                f"{'param name':<{col_w}}  {'old_grad':>16}  {'new_grad':>16}"
                f"  {'diff':>12}  {'rel_diff':>10}"
            )
            sep = "-" * len(header)
            lines = [
                f"[Rank {rank}] {len(diffs)}/{len(new_grads)} params differ:",
                sep, header, sep,
            ]
            for name, nv, ov in diffs:
                d = abs(nv - ov)
                rel = d / (abs(ov) + 1e-12)
                lines.append(
                    f"{name:<{col_w}}  {ov:>16.8e}  {nv:>16.8e}"
                    f"  {d:>12.4e}  {rel:>10.4e}"
                )
            lines.append(sep)
            print("\n".join(lines))
        else:
            print(f"[Rank {rank}] all {len(new_grads)} grad shard sums identical to run 1")

        # Reduce summary stats across all ranks
        dist.barrier()
        diff_count = torch.tensor([len(diffs)], dtype=torch.int64, device="cuda")
        max_rel = max((abs(nv - ov) / (abs(ov) + 1e-12) for _, nv, ov in diffs), default=0.0)
        max_rel_t = torch.tensor([max_rel], dtype=torch.float64, device="cuda")
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
                sys.exit(0)
            elif global_max_rel < 1e-4:
                print(
                    f"RESULT: PRACTICALLY DETERMINISTIC — {total_diffs} param shards differ\n"
                    f"across all ranks but max relative difference is {global_max_rel:.2e} (<1e-4),\n"
                    "which is negligible for training."
                )
                sys.exit(0)
            else:
                print(
                    f"RESULT: NON-DETERMINISTIC — {total_diffs} param shards differ across\n"
                    f"all ranks with max relative difference {global_max_rel:.2e}."
                )
                sys.exit(2)  # exit 2 = non-determinism still present

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
