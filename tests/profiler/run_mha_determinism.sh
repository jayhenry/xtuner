#!/bin/bash
# FSDP2 + MHA-only determinism (``fully_shard`` on ``MultiHeadAttention``, same spirit as base.py).
#
# Requires torchrun. Set XTUNER_DETERMINISTIC before launch so flash-attn reads it at import time.

set -e

cd "$(dirname "$0")/../.."   # repo root

export PYTHONPATH="./"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
# FA3 raises RuntimeError: Deterministic backward not supported for hdim 256.
export XTUNER_USE_FA3=0

NPROC="${NPROC:-4}"
TORCHRUN="torchrun --nproc-per-node ${NPROC} --master-port 29711"
SCRIPT="tests/profiler/mha_determinism_minimal.py"

OUT="/tmp/mha_determinism_fsdp_$$"
mkdir -p "$OUT"
echo "Output dir: $OUT  (NPROC=${NPROC})"

SEQ_LEN1=8191  # 65535
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [A] Record ($SEQ_LEN1, FSDP + compile + deterministic)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
XTUNER_DETERMINISTIC=true $TORCHRUN "$SCRIPT" \
    --record-path "$OUT/run_${SEQ_LEN1}_v1" \
    --seq-len $SEQ_LEN1 \
    --deterministic

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [A] Compare (expect: deterministic)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
XTUNER_DETERMINISTIC=true $TORCHRUN "$SCRIPT" \
    --record-path "$OUT/run_${SEQ_LEN1}_v2" \
    --compare "$OUT/run_${SEQ_LEN1}_v1" \
    --seq-len $SEQ_LEN1 \
    --deterministic

SEQ_LEN2=8192  # 65536
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [B] Record ($SEQ_LEN2, FSDP + compile + deterministic)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
XTUNER_DETERMINISTIC=true $TORCHRUN "$SCRIPT" \
    --record-path "$OUT/run_${SEQ_LEN2}_v1" \
    --seq-len $SEQ_LEN2 \
    --deterministic

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [B] Compare (often NON-DETERMINISTIC)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
XTUNER_DETERMINISTIC=true $TORCHRUN "$SCRIPT" \
    --record-path "$OUT/run_${SEQ_LEN2}_v2" \
    --compare "$OUT/run_${SEQ_LEN2}_v1" \
    --seq-len $SEQ_LEN2 \
    --deterministic

echo ""
echo "Done. Per-rank JSON under: $OUT"
