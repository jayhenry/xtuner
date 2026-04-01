#!/bin/bash
# Run numerics_test.py twice (non-deterministic) then twice (deterministic)
# to verify that NCCL_ALGO=Ring + NCCL_PROTO=Simple eliminates the cross-process
# reduce-scatter non-determinism observed in 8-GPU FSDP training.
#
# Expected outcome:
#   Non-deterministic pair: some params differ between run1 and run2
#   Deterministic pair    : all params identical between run3 and run4

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

# ══════════════════════════════════════════════════════════════════════════════
# Part A: Non-deterministic baseline (default NCCL settings)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [A] Run 1  (non-deterministic, recording)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$TORCHRUN $SCRIPT --record-path "$GRAD_DIR/nd_run1"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [A] Run 2  (non-deterministic, comparing)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$TORCHRUN $SCRIPT --record-path "$GRAD_DIR/nd_run2" \
                  --compare    "$GRAD_DIR/nd_run1" || true   # exit 2 = not reproduced, still continue

# ══════════════════════════════════════════════════════════════════════════════
# Part B: Deterministic (NCCL_ALGO=Ring + NCCL_PROTO=Simple)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [B] Run 3  (deterministic, recording)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# XTUNER_DETERMINISTIC=true must be in the environment *before* torchrun so that
# flash-attn is called with deterministic=True (the constant is evaluated at
# Python module-import time, not at runtime).
XTUNER_DETERMINISTIC=true $TORCHRUN $SCRIPT --record-path "$GRAD_DIR/det_run1" --deterministic

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [B] Run 4  (deterministic, comparing)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
XTUNER_DETERMINISTIC=true $TORCHRUN $SCRIPT --record-path "$GRAD_DIR/det_run2" \
                                            --compare    "$GRAD_DIR/det_run1" \
                                            --deterministic

echo ""
echo "Records saved in: $GRAD_DIR"
