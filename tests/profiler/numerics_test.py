"""Standalone torchrun script to record and compare FSDP+compiled gradient checksums.

Usage (called by run_test.sh):
    # Run 1 — record grad sums to files:
    torchrun --nproc-per-node 8 numerics_test.py --record-path /tmp/grads/run1

    # Run 2 — record again, then compare against run 1:
    torchrun --nproc-per-node 8 numerics_test.py --record-path /tmp/grads/run2 \
        --compare /tmp/grads/run1

    # Compare two existing recordings only (no forward/backward):
    torchrun --nproc-per-node 8 numerics_test.py --record-path /tmp/grads/run2 \
        --compare /tmp/grads/run1 --skip-train

    # Dense random sequence instead of packed collator-style batch:
    torchrun ... numerics_test.py ... --batch-style simple

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

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from mmengine.runner import set_random_seed

from xtuner.v1.profiler.prober_utils import setup_prober_list
from xtuner.v1.utils import IGNORE_INDEX
from xtuner.v1.utils.misc import monkey_patch_hf_modules_cache


# ---------------------------------------------------------------------------
# Model + run helpers (mirroring TestFSDPCompiledMHAGradNumerics)
# ---------------------------------------------------------------------------

NUMERICS_SEQ_LEN = 65536
NUMERICS_GRAD_ACCUM_STEPS = 2

# Set from `--batch-style` in main(); used when `_precompute_micro_batches` omits `realistic`.
_NUMERICS_BATCH_STYLE_REALISTIC: bool | None = None


def _randint(g: torch.Generator, low: int, high: int) -> int:
    """Inclusive low, exclusive high (same as torch.randint)."""
    return int(torch.randint(low, high, (1,), generator=g).item())


def _random_partition(g: torch.Generator, total: int, k: int, min_part: int) -> list[int]:
    assert total >= k * min_part
    parts = [min_part] * k
    for _ in range(total - k * min_part):
        parts[_randint(g, 0, k)] += 1
    return parts


def _build_realistic_packed_batch(
    *,
    g: torch.Generator,
    device: torch.device,
    vocab_size: int,
    pack_max_length: int,
    padding_token_idx: int,
    pad_chunk_size: int = 256,
) -> tuple[Any, torch.Tensor]:
    """Mimic `build_text_ctx_labels` + SFT masking (ftdp-style).

    - Several logical samples concatenated; `num_tokens` / `cu_seq_lens` follow collator.
    - Padded to `pack_max_length` (labels padded with IGNORE_INDEX).
    - Within each sample: first label -100; a contiguous \"user\" span also -100.
    """
    from xtuner.v1.datasets.collator import build_text_ctx_labels

    min_seg = 8
    k = _randint(g, 3, 9)
    t_lower = k * min_seg + 2
    if t_lower > pack_max_length:
        k = max(2, pack_max_length // (2 * min_seg))
        t_lower = k * min_seg + 2
    # t_total <= pack_max_length => shifted len <= pack_max_length - 1 => always some pad.
    t_total = _randint(g, t_lower, pack_max_length + 1)

    seg_lens = _random_partition(g, t_total, k, min_seg)

    instances: list[dict] = []
    for L in seg_lens:
        tok = torch.randint(0, vocab_size, (L,), generator=g)
        labels = tok.tolist()
        labels[0] = IGNORE_INDEX
        # Contiguous prefix (after BOS slot) masked like template \"user\" / no-loss regions in ftdp.
        max_user = max(1, L - 2)
        user_len = _randint(g, 1, max_user + 1)
        for j in range(1, 1 + user_len):
            labels[j] = IGNORE_INDEX
        instances.append(
            {
                "input_ids": tok.tolist(),
                "labels": labels,
                "num_tokens": L,
            }
        )

    seq_ctx, shifted_labels, _ = build_text_ctx_labels(
        instances,
        pack_max_length,
        padding_token_idx,
        pack_to_max_length=True,
        pad_chunk_size=pad_chunk_size,
    )
    seq_ctx = seq_ctx.to(device)
    shifted_labels = shifted_labels.to(device)
    assert seq_ctx.input_ids.shape[-1] == pack_max_length
    assert shifted_labels.shape == seq_ctx.input_ids.shape
    return seq_ctx, shifted_labels


def _build_simple_packed_batch(
    *,
    g: torch.Generator,
    device: torch.device,
    vocab_size: int,
    seq_len: int,
) -> tuple[Any, torch.Tensor]:
    """Dense random sequence + `from_input_ids` (single segment), full next-token loss."""
    from xtuner.v1.data_proto import SequenceContext

    full = torch.randint(0, vocab_size, (1, seq_len + 1), generator=g, dtype=torch.long)
    full = full.to(device)
    shifted_labels = full[:, 1:].clone()
    shift_input_ids = full[:, :-1]
    seq_ctx = SequenceContext.from_input_ids(input_ids=(shift_input_ids,), device=str(device))
    return seq_ctx, shifted_labels


def _precompute_micro_batches(
    *,
    rank: int,
    device: torch.device,
    vocab_size: int,
    padding_token_idx: int,
    seq_len: int,
    num_steps: int,
    realistic: bool | None = None,
) -> list[tuple[Any, torch.Tensor]]:
    """Build all `(seq_ctx, shifted_labels)` for this rank before any forward."""
    use_realistic = _NUMERICS_BATCH_STYLE_REALISTIC if realistic is None else realistic
    assert use_realistic is not None, "--batch-style must be set"
    batches: list[tuple[Any, torch.Tensor]] = []
    for micro in range(num_steps):
        g = torch.Generator(device="cpu")
        g.manual_seed(rank * num_steps + micro)
        if use_realistic:
            print("Building realistic packed batch")
            seq_ctx, shifted_labels = _build_realistic_packed_batch(
                g=g,
                device=device,
                vocab_size=vocab_size,
                pack_max_length=seq_len,
                padding_token_idx=padding_token_idx,
            )
        else:
            print("Building simple dense batch")
            seq_ctx, shifted_labels = _build_simple_packed_batch(
                g=g,
                device=device,
                vocab_size=vocab_size,
                seq_len=seq_len,
            )
        batches.append((seq_ctx, shifted_labels))
    return batches


def _make_model_config(num_hidden_layers: int = 4, compile: bool = True):
    # from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
    # cfg = Qwen3_5_VLTextMoE35BA3BConfig(num_hidden_layers=num_hidden_layers)
    # cfg.compile_cfg = compile  # match production: TORCH_COMPILE=1

    # xTODO: create qwen 3.5 VL whole model
    from xtuner.v1.model.compose.qwen3_5 import Qwen3_5_VLMoE35BA3Config
    moe_cfg = Qwen3_5_VLMoE35BA3Config()
    moe_cfg.text_config.num_hidden_layers = num_hidden_layers
    moe_cfg.text_config.ep_size = 1
    moe_cfg.only_llm_forward = True
    moe_cfg.compile_cfg = compile  # match production: TORCH_COMPILE=1
    return moe_cfg


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


def _run_forward_backward(
    model,
    micro_batches: list[tuple[Any, torch.Tensor]],
) -> dict[str, float]:
    """Run precomputed micro-batches (packed length NUMERICS_SEQ_LEN), return grad shard sums."""
    from xtuner.v1.loss.ce_loss import CELossConfig

    loss_cfg = CELossConfig(mode="chunk")
    LossCtx = loss_cfg.loss_ctx_cls

    rank = dist.get_rank()
    GRAD_ACCUM_STEPS = len(micro_batches)
    model.zero_grad()
    for micro, (seq_ctx, shifted_labels) in enumerate(micro_batches):
        torch.manual_seed(rank * GRAD_ACCUM_STEPS + micro)
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
    parser.add_argument(
        "--skip-train", action="store_true",
        help=(
            "Skip model build and forward/backward; load JSON from --record-path "
            "and compare against --compare (requires --compare). No writes."
        ),
    )
    parser.add_argument(
        "--batch-style",
        choices=("simple", "realistic"),
        default="simple",
        help=(
            "Training micro-batch layout: 'simple' is dense random ids + from_input_ids; "
            "'realistic' follows collator packing, padding, and partial label masking (SFT-like)."
        ),
    )
    args = parser.parse_args()

    global _NUMERICS_BATCH_STYLE_REALISTIC
    _NUMERICS_BATCH_STYLE_REALISTIC = args.batch_style == "realistic"

    if args.skip_train and not args.compare:
        print("ERROR: --skip-train requires --compare", file=sys.stderr)
        sys.exit(2)

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

    vocab_size = 248320
    padding_token_idx = 0
    micro_batches = _precompute_micro_batches(
        rank=rank,
        device=torch.device("cuda"),
        vocab_size=vocab_size,
        padding_token_idx=padding_token_idx,
        seq_len=NUMERICS_SEQ_LEN,
        num_steps=NUMERICS_GRAD_ACCUM_STEPS,
    )

    set_random_seed(0, deterministic=args.deterministic)

    from xtuner.v1.utils import XTUNER_DETERMINISTIC
    compile_mode = not args.no_compile

    hf_path = None
    if args.skip_train:
        print(
            f"[numerics_test] skip_train=True  world_size={world_size}  "
            f"record_path={args.record_path}  compare={args.compare}"
        )
        t0 = time.time()
        new_grads = load_record(args.record_path, rank)
        elapsed = time.time() - t0
        if rank == 0:
            print(
                f"[numerics_test] loaded {len(new_grads)} params from {args.record_path} "
                f"in {elapsed:.2f}s"
            )
    else:
        hf_path = args.hf_path or os.environ.get("QWEN35_MOE_PATH")
        if not hf_path:
            if rank == 0:
                print("ERROR: QWEN35_MOE_PATH env var or --hf-path must be set", file=sys.stderr)
            dist.destroy_process_group()
            sys.exit(1)

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
            print(
                f"[numerics_test] world_size={world_size}  {compile_str}  "
                f"batch_style={args.batch_style}  record_path={args.record_path}{det}"
            )
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

        setup_prober_list(Path(args.record_path)/'prober', [0, 1], model, ['AccProber'])

        if rank == 0:
            print("[numerics_test] running forward+backward ...")
        new_grads = _run_forward_backward(model, micro_batches)
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
