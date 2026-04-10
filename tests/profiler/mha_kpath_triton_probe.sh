#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_ENV=(
  env
  CUDA_VISIBLE_DEVICES=4,5,6,7
  PYTHONPATH=.
  XTUNER_DETERMINISTIC=true
  OMP_NUM_THREADS=1
  XTUNER_USE_FA3=0
  NCCL_ALGO=Ring
  NCCL_PROTO=Simple
  NCCL_NUM_CHANNELS=1
  CUBLAS_WORKSPACE_CONFIG=:16:8
)
TORCHRUN=torchrun
PROBE=tests/profiler/mha_kpath_triton_probe.py

COMMON_ARGS=(
  --standalone
  --nproc-per-node 4
  "$PROBE"
  --seq-len 65536
  --sync-after-target
  --print-target-config
)

echo "[A] dynamic_scale_rblock=ON, run 1"
"${BASE_ENV[@]}" "$TORCHRUN" "${COMMON_ARGS[@]}" \
  --save-dir /tmp/codex_kpath_dynamic_on_a

echo "[B] dynamic_scale_rblock=ON, run 2 compare: expect non-deterministic"
"${BASE_ENV[@]}" "$TORCHRUN" "${COMMON_ARGS[@]}" \
  --save-dir /tmp/codex_kpath_dynamic_on_b \
  --compare /tmp/codex_kpath_dynamic_on_a

echo "[C] dynamic_scale_rblock=OFF, run 1"
"${BASE_ENV[@]}" "$TORCHRUN" "${COMMON_ARGS[@]}" \
  --no-dynamic-scale-rblock \
  --save-dir /tmp/codex_kpath_dynamic_off_a

echo "[D] dynamic_scale_rblock=OFF, run 2 compare: expect deterministic"
"${BASE_ENV[@]}" "$TORCHRUN" "${COMMON_ARGS[@]}" \
  --no-dynamic-scale-rblock \
  --save-dir /tmp/codex_kpath_dynamic_off_b \
  --compare /tmp/codex_kpath_dynamic_off_a
