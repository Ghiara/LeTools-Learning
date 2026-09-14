#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="/home/yuan/bowang/LeTools-Learning"
TASK_DIR="/home/yuan/bowang/icra_task1"
CONDA_SH="/home/yuan/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV="letools_bo"
ROS_SETUP="/opt/ros/noetic/setup.bash"
GPU_ID="${GPU_ID:-0}"
WORKER="${REPO_DIR}/scripts/train_icra_task1_job.sh"

JOBS=(
  "icra_train_dp100_bo|diffusion|icratask1_dp_100rosbags|dp100"
  "icra_train_dp300_bo|diffusion|icratask1_dp_300rosbags|dp300"
  "icra_train_act100_bo|act|icratask1_act_100rosbags|act100"
  "icra_train_act300_bo|act|icratask1_act_300rosbags|act300"
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_icra_task1_tmux.sh --dry-run
  GPU_ID=0 bash scripts/train_icra_task1_tmux.sh

Creates four independent tmux sessions. A GPU lock makes them train one at a
time, while keeping failures isolated between sessions.
EOF
}

config_root_for() {
  local config_name="$1"
  printf '%s/configs/train/%s\n' "$REPO_DIR" "$config_name"
}

validate_inputs() {
  local required_paths=(
    "$CONDA_SH"
    "$ROS_SETUP"
    "$TASK_DIR/datasets/rgb_100rosbags/lerobot/meta/info.json"
    "$TASK_DIR/datasets/rgb_300rosbags/lerobot/meta/info.json"
    "$REPO_DIR/configs/train/icratask1_dp_100rosbags/total/diffusion_total.yaml"
    "$REPO_DIR/configs/train/icratask1_dp_300rosbags/total/diffusion_total.yaml"
    "$REPO_DIR/configs/train/icratask1_act_100rosbags/total/act_total.yaml"
    "$REPO_DIR/configs/train/icratask1_act_300rosbags/total/act_total.yaml"
  )

  local path
  for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
      echo "Missing required path: $path" >&2
      return 1
    fi
  done
}

dry_run_jobs() {
  local job session policy config_name job_name
  for job in "${JOBS[@]}"; do
    IFS='|' read -r session policy config_name job_name <<< "$job"
    echo "$session: GPU_ID=$GPU_ID bash $WORKER $policy $config_name $job_name"
  done
}

main() {
  validate_inputs

  case "${1:-}" in
    --dry-run)
      dry_run_jobs
      ;;
    -h|--help)
      usage
      ;;
    "")
      command -v tmux >/dev/null
      local job session policy config_name job_name command
      for job in "${JOBS[@]}"; do
        IFS='|' read -r session policy config_name job_name <<< "$job"
        if tmux has-session -t "$session" 2>/dev/null; then
          echo "tmux session already exists: $session" >&2
          exit 1
        fi
      done
      for job in "${JOBS[@]}"; do
        IFS='|' read -r session policy config_name job_name <<< "$job"
        printf -v command 'GPU_ID=%q bash %q %q %q %q' \
          "$GPU_ID" "$WORKER" "$policy" "$config_name" "$job_name"
        tmux new-session -d -s "$session" -c "$REPO_DIR"
        tmux send-keys -t "$session:0.0" -l "$command"
        tmux send-keys -t "$session:0.0" Enter
        echo "Started tmux session: $session"
        sleep 1
      done
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
