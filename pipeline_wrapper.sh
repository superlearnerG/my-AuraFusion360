#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_SCRIPT="$SCRIPT_DIR/pipeline.sh"

DATA_ROOT="${DATA_ROOT:-../../siga26/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-../../siga26/output}"

COMMON_ARGS=(
  --removal_thresh 0.7
  --fit_mask_iterations 2000
  --finetune_iteration 5000
)

SCENE_CONFIGS=(
  "bear|128"
  "bonsai|128"
  "fruits|[48,30,67,86,105,123,142],[161,180,217,198,236]"
  "doppelherz|[30,105,180]"
)

SELECTED_SCENES=()
PASSTHROUGH_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash pipeline_wrapper.sh [--scene NAME ...] [pipeline.sh options...]

Behavior:
  - By default, runs all predefined scenes in this file.
  - Repeat --scene to run only a subset.
  - Any other arguments are forwarded to pipeline.sh and can override defaults.

Examples:
  bash pipeline_wrapper.sh
  bash pipeline_wrapper.sh --scene bear --scene bonsai --skip_base_train
  bash pipeline_wrapper.sh --scene fruits --port 6021 --base_iteration 40000

Environment:
  DATA_ROOT     Scene root. Default: ../../siga26/data
  OUTPUT_ROOT   Output root. Default: ../../siga26/output
  PYTHON_BIN    Forwarded through pipeline.sh
EOF
}

find_scene_config() {
  local scene_name="$1"
  local config=""
  for config in "${SCENE_CONFIGS[@]}"; do
    if [[ "${config%%|*}" == "$scene_name" ]]; then
      printf '%s\n' "$config"
      return 0
    fi
  done
  return 1
}

list_scene_names() {
  local config=""
  local names=()
  local IFS=' '
  for config in "${SCENE_CONFIGS[@]}"; do
    names+=("${config%%|*}")
  done
  printf '%s\n' "${names[*]}"
}

run_scene() {
  local scene_name="$1"
  local target_ids="$2"
  local source_path="$DATA_ROOT/$scene_name"
  local model_path="$OUTPUT_ROOT/$scene_name/aurafusion360"
  local raw_mask_dir="$source_path/object_mask"

  echo "============================================================"
  echo "scene        : $scene_name"
  echo "source_path  : $source_path"
  echo "model_path   : $model_path"
  echo "raw_mask_dir : $raw_mask_dir"
  echo "target_ids   : $target_ids"
  echo "============================================================"

  bash "$PIPELINE_SCRIPT" \
    -s "$source_path" \
    -m "$model_path" \
    --raw_mask_dir "$raw_mask_dir" \
    --target_ids "$target_ids" \
    "${COMMON_ARGS[@]}" \
    "${PASSTHROUGH_ARGS[@]}"
}

if [[ ! -f "$PIPELINE_SCRIPT" ]]; then
  echo "Missing pipeline script: $PIPELINE_SCRIPT" >&2
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --scene" >&2
        usage >&2
        exit 1
      fi
      SELECTED_SCENES+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

cd "$SCRIPT_DIR"

if [[ ${#SELECTED_SCENES[@]} -eq 0 ]]; then
  for config in "${SCENE_CONFIGS[@]}"; do
    IFS='|' read -r scene_name target_ids <<< "$config"
    run_scene "$scene_name" "$target_ids"
  done
else
  for scene_name in "${SELECTED_SCENES[@]}"; do
    config="$(find_scene_config "$scene_name")" || {
      echo "Unknown scene: $scene_name" >&2
      echo "Available scenes: $(list_scene_names)" >&2
      exit 1
    }
    IFS='|' read -r _ target_ids <<< "$config"
    run_scene "$scene_name" "$target_ids"
  done
fi
