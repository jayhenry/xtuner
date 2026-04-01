#!/bin/bash
# Run numerics_test.py twice with separate torchrun invocations to verify
# whether FSDP reduce-scatter is non-deterministic across process boundaries.
#
# Usage:
#   bash tests/profiler/run_test.sh
#
# Each torchrun call creates a fresh NCCL process group.  If NCCL's ring-allreduce
# timing varies between the two OS-level process groups, at least one rank will
# accumulate gradient partial sums in a different order → different final values.

set -e

cd "$(dirname "$0")/../.."   # repo root

# ── environment ────────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fla

export QWEN35_MOE_PATH="/mnt/shared-storage-user/llmit/user/maningsheng/data/models/models--Qwen--Qwen3.5-35B-A3B"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="./"

# Disable CUTLASS grouped GEMM (matches accuracy-regression CI config)
unset XTUNER_USE_CUTLASS_GROUP_GEMM

# ── temp directory for grad records ───────────────────────────────────────────
GRAD_DIR="/tmp/fsdp_nccl_test_$$"
mkdir -p "$GRAD_DIR"
echo "Grad records dir: $GRAD_DIR"
echo ""

# ── Run 1: record grad shard sums ────────────────────────────────────────────
echo "══════════════════════════════════════════"
echo " Run 1 — recording grad shard sums"
echo "══════════════════════════════════════════"
torchrun --nproc-per-node 8 --master-port 29710 \
    tests/profiler/numerics_test.py \
    --record-path "$GRAD_DIR/run1"

echo ""

# ── Run 2: record again, compare against Run 1 ───────────────────────────────
echo "══════════════════════════════════════════"
echo " Run 2 — recording and comparing vs Run 1"
echo "══════════════════════════════════════════"
torchrun --nproc-per-node 8 --master-port 29710 \
    tests/profiler/numerics_test.py \
    --record-path "$GRAD_DIR/run2" \
    --compare    "$GRAD_DIR/run1"

echo ""
echo "Records saved in: $GRAD_DIR"
