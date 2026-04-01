#!/bin/bash
# Two-part determinism verification for FSDP+compiled Qwen3.5-35B-A3B.
#
# Part A — Non-compiled (eager) + deterministic:
#   Expected: FULLY DETERMINISTIC (total_diffs=0)
#   Confirms that XTUNER_DETERMINISTIC + float32 reduce + NCCL Ring/Simple
#   eliminates all non-determinism when torch.compile is NOT involved.
#
# Part B — Compiled + deterministic:
#   Expected: PRACTICALLY DETERMINISTIC (global_max_rel < 1e-4)
#   Confirms that the same settings keep compiled training "good enough";
#   residual differences come from causal_conv1d backward (no deterministic mode).

set -e

cd "$(dirname "$0")/../.."   # repo root

# ── environment ────────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fla

export QWEN35_MOE_PATH="/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="./"
unset XTUNER_USE_CUTLASS_GROUP_GEMM

TORCHRUN="torchrun --nproc-per-node 8 --master-port 29710"
SCRIPT="tests/profiler/numerics_test.py"

GRAD_DIR="/tmp/fsdp_nccl_test_$$"
mkdir -p "$GRAD_DIR"
echo "Grad records dir: $GRAD_DIR"

# # ══════════════════════════════════════════════════════════════════════════════
# # Part A: Eager (non-compiled) + deterministic → expect total_diffs=0
# # ══════════════════════════════════════════════════════════════════════════════
# echo ""
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo " [A] Run 1  (eager + deterministic, recording)"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# XTUNER_DETERMINISTIC=true $TORCHRUN $SCRIPT \
#     --record-path "$GRAD_DIR/eager_run1" \
#     --deterministic \
#     --no-compile
# 
# echo ""
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# echo " [A] Run 2  (eager + deterministic, comparing)"
# echo " Expected: FULLY DETERMINISTIC (total_diffs=0)"
# echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# XTUNER_DETERMINISTIC=true $TORCHRUN $SCRIPT \
#     --record-path "$GRAD_DIR/eager_run2" \
#     --compare    "$GRAD_DIR/eager_run1" \
#     --deterministic \
#     --no-compile

# ══════════════════════════════════════════════════════════════════════════════
# Part B: Compiled + deterministic + seq_len = 65535 → expect global_max_rel < 1e-4
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [B] Run 3  (compiled + deterministic, recording)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# XTUNER_DETERMINISTIC=true must be in the environment *before* torchrun so that
# flash-attn is called with deterministic=True (the constant is evaluated at
# Python module-import time, not at runtime).
XTUNER_DETERMINISTIC=true $TORCHRUN $SCRIPT \
    --record-path "$GRAD_DIR/compiled_run1" \
    --seq-len 65535 \
    --deterministic

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [B] Run 4  (compiled + deterministic, comparing)"
echo " Expected: PRACTICALLY DETERMINISTIC (global_max_rel < 1e-4)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
XTUNER_DETERMINISTIC=true $TORCHRUN $SCRIPT \
    --record-path "$GRAD_DIR/compiled_run2" \
    --compare    "$GRAD_DIR/compiled_run1" \
    --seq-len 65535 \
    --deterministic

# ══════════════════════════════════════════════════════════════════════════════
# Part C: Compiled + deterministic + seq_len = 65536 → expect failure: RESULT: NON-DETERMINISTIC
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [C] Run 5  (compiled + deterministic, recording)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# XTUNER_DETERMINISTIC=true must be in the environment *before* torchrun so that
# flash-attn is called with deterministic=True (the constant is evaluated at
# Python module-import time, not at runtime).
XTUNER_DETERMINISTIC=true $TORCHRUN $SCRIPT \
    --record-path "$GRAD_DIR/compiled_run5" \
    --seq-len 65536 \
    --deterministic

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [C] Run 6  (compiled + deterministic, comparing)"
echo " Expected: PRACTICALLY DETERMINISTIC (global_max_rel < 1e-4)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
XTUNER_DETERMINISTIC=true $TORCHRUN $SCRIPT \
    --record-path "$GRAD_DIR/compiled_run6" \
    --compare    "$GRAD_DIR/compiled_run5" \
    --seq-len 65536 \
    --deterministic

echo ""
echo "Records saved in: $GRAD_DIR"
