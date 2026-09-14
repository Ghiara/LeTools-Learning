#!/usr/bin/env bash

set -euo pipefail

SOURCE_DIR="/home/yuan/bowang/icra_task1/datasets/"
REMOTE_HOST="pro6000"
REMOTE_DIR="/media/data/bowang/icra_task1/icra_task1_converted/"
SESSION_NAME="icra_datasets_to_pro6000"
LOG_DIR="/home/yuan/bowang/icra_task1/logs"
LOG_FILE="${LOG_DIR}/transfer_datasets_to_pro6000.log"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

usage() {
  cat <<EOF
Usage:
  bash ${SCRIPT_PATH}              Start the transfer in tmux
  bash ${SCRIPT_PATH} --dry-run    Preview rsync changes without transferring
  bash ${SCRIPT_PATH} --status     Show the tmux session and recent log output

tmux session: ${SESSION_NAME}
log file:     ${LOG_FILE}
source:       ${SOURCE_DIR}
destination:  ${REMOTE_HOST}:${REMOTE_DIR}
EOF
}

check_local_requirements() {
  command -v rsync >/dev/null
  command -v ssh >/dev/null
  [[ -d "$SOURCE_DIR" ]] || {
    echo "Source directory does not exist: $SOURCE_DIR" >&2
    exit 1
  }
}

run_transfer() {
  mkdir -p "$LOG_DIR"
  exec > >(tee -a "$LOG_FILE") 2>&1

  echo "[$(date --iso-8601=seconds)] Starting dataset transfer"
  echo "Source:      $SOURCE_DIR"
  echo "Destination: ${REMOTE_HOST}:${REMOTE_DIR}"

  ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"
  rsync -aP --info=progress2 --human-readable \
    "$SOURCE_DIR" \
    "${REMOTE_HOST}:${REMOTE_DIR}"

  echo "[$(date --iso-8601=seconds)] Transfer completed; running checksum verification"
  local differences
  differences="$(rsync -acn --delete --itemize-changes \
    "$SOURCE_DIR" \
    "${REMOTE_HOST}:${REMOTE_DIR}")"

  if [[ -n "$differences" ]]; then
    echo "Verification found differences:" >&2
    printf '%s\n' "$differences" >&2
    exit 1
  fi

  echo "[$(date --iso-8601=seconds)] Verification passed: source and destination match"
}

dry_run() {
  check_local_requirements
  ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"
  rsync -aPvn --info=progress2 --human-readable \
    "$SOURCE_DIR" \
    "${REMOTE_HOST}:${REMOTE_DIR}"
}

show_status() {
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session is running: $SESSION_NAME"
  else
    echo "tmux session is not running: $SESSION_NAME"
  fi

  if [[ -f "$LOG_FILE" ]]; then
    echo
    echo "Recent log output ($LOG_FILE):"
    tail -n 30 "$LOG_FILE"
  else
    echo "Log file has not been created yet: $LOG_FILE"
  fi
}

start_tmux() {
  check_local_requirements
  command -v tmux >/dev/null

  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session already exists: $SESSION_NAME" >&2
    echo "Attach with: tmux attach -t $SESSION_NAME" >&2
    exit 1
  fi

  mkdir -p "$LOG_DIR"
  tmux new-session -d -s "$SESSION_NAME" \
    "bash '$SCRIPT_PATH' --worker"

  echo "Started background transfer in tmux session: $SESSION_NAME"
  echo "Attach: tmux attach -t $SESSION_NAME"
  echo "Status: bash $SCRIPT_PATH --status"
  echo "Log:    tail -f $LOG_FILE"
}

case "${1:-}" in
  "")
    start_tmux
    ;;
  --worker)
    check_local_requirements
    run_transfer
    ;;
  --dry-run)
    dry_run
    ;;
  --status)
    show_status
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
