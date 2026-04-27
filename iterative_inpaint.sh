#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

SOURCE_PATH=""
MODEL_PATH=""
WORKFLOW_CONFIG=""
OBJECT_NUM=""
RENDER_INTERMEDIATE=0

usage() {
  cat <<'EOF'
Usage:
  bash iterative_inpaint.sh \
    -s <source_path> \
    -m <model_path> \
    --workflow_config <workflow_config> \
    --object_num <round_count> \
    [--render_intermediate]

Environment:
  PYTHON_BIN    Python executable to use. Default: python
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--source_path)
      SOURCE_PATH="$2"
      shift 2
      ;;
    -m|--model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --workflow_config)
      WORKFLOW_CONFIG="$2"
      shift 2
      ;;
    --object_num)
      OBJECT_NUM="$2"
      shift 2
      ;;
    --render_intermediate)
      RENDER_INTERMEDIATE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SOURCE_PATH" || -z "$MODEL_PATH" || -z "$WORKFLOW_CONFIG" || -z "$OBJECT_NUM" ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 1
fi

if ! [[ "$OBJECT_NUM" =~ ^[0-9]+$ ]] || (( OBJECT_NUM <= 0 )); then
  echo "--object_num must be a positive integer: $OBJECT_NUM" >&2
  exit 1
fi

run_stage() {
  local command="$1"
  local round_index="$2"
  local extra_args=()
  if (( RENDER_INTERMEDIATE )); then
    extra_args+=(--render_intermediate)
  fi
  echo "[$(date '+%F %T')] round_index=${round_index} command=${command}"
  "$PYTHON_BIN" "$SCRIPT_DIR/iterative_inpaint.py" "$command" \
    -s "$SOURCE_PATH" \
    -m "$MODEL_PATH" \
    --workflow_config "$WORKFLOW_CONFIG" \
    --round_index "$round_index" \
    --python_bin "$PYTHON_BIN" \
    "${extra_args[@]}"
}

for (( round_index=0; round_index<OBJECT_NUM; round_index++ )); do
  run_stage prepare-round "$round_index"
  run_stage run-leftrefill "$round_index"
  run_stage finalize-round "$round_index"
done
