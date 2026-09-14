#!/usr/bin/env bash

set -euo pipefail

[[ $# -eq 3 ]] || { echo "Usage: $0 POLICY CONFIG_NAME JOB_NAME" >&2; exit 2; }
POLICY="$1"
CONFIG_NAME="$2"
JOB_NAME="$3"
REPO_DIR="/home/yuan/bowang/LeTools-Learning"
TASK_DIR="/home/yuan/bowang/icra_task1"
GPU_ID="${GPU_ID:-0}"
CONFIG_ROOT="$REPO_DIR/configs/train/$CONFIG_NAME"
LOG_FILE="$TASK_DIR/logs/train_${JOB_NAME}.log"
LOCK_FILE="$TASK_DIR/locks/gpu_${GPU_ID}.lock"

[[ -d "$CONFIG_ROOT" ]] || { echo "Missing: $CONFIG_ROOT" >&2; exit 1; }
mkdir -p "$TASK_DIR/logs" "$TASK_DIR/training" "$TASK_DIR/locks" \
  "$TASK_DIR/cache/huggingface" "$TASK_DIR/cache/torch" "$TASK_DIR/tmp"

source /home/yuan/miniforge3/etc/profile.d/conda.sh
conda activate letools_bo
source /opt/ros/noetic/setup.bash
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$TASK_DIR/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$TASK_DIR/cache/torch}"
export TMPDIR="${TMPDIR:-$TASK_DIR/tmp}"

exec 9>"$LOCK_FILE"
echo "[$(date '+%F %T')] $JOB_NAME waiting for GPU $GPU_ID"
flock 9
echo "[$(date '+%F %T')] $JOB_NAME acquired GPU $GPU_ID"

cd "$REPO_DIR"
set +e
python kuavo_model/train.py --policy "$POLICY" --mode total \
  --config-root "$CONFIG_ROOT" 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
echo "[$(date '+%F %T')] $JOB_NAME finished with exit code $status"
exit "$status"
