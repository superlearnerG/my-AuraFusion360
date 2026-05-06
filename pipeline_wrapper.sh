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
  # Format: "scene|target_ids|lama_reference_stem"
  # Leave lama_reference_stem empty to use pipeline.sh default max_mask selection.
  # "bear|128|00001"
  # "bonsai|128|{lama_reference_stem}"
  # "fruits|[48,30,67,86,105,123,142],[161,180,217,198,236]|{lama_reference_stem}"
  # "doppelherz|[30,105,180]|{lama_reference_stem}"
  # "toys|[30,105,180]|{lama_reference_stem}"
  # "figurines|[30,83,72,51,40,94,115,158,222,233],[62,105,169,180,126,142,201,212,244,190]|{lama_reference_stem}"
  # "scene_1_colmap|[12,25,38,51,63,76]|{lama_reference_stem}"
  # "scene_5_colmap|[12,25,38,51,63,76]|{lama_reference_stem}"
  # "scene_6_colmap|[12,25,38,51,63,76]|{lama_reference_stem}"
  # "bag|30|{lama_reference_stem}"
  "office_desk|[119,102,17],[34,51,68,85]|"
  "dining_table|[170,136,102,204],[119,187,85,153]|"
  "bedroom|[85,17,34],[68,153,102,136,51]|"
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
  - SCENE_CONFIGS entries are scene|target_ids|lama_reference_stem. Leave the
    final field empty to use pipeline.sh default max_mask reference selection.

Examples:
  bash pipeline_wrapper.sh
  bash pipeline_wrapper.sh --scene bear --scene bonsai --skip_base_train
  bash pipeline_wrapper.sh --scene fruits --port 6021 --base_iteration 40000
  bash pipeline_wrapper.sh --scene bear --lama_reference_stem 00067 --skip_base_train

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
  local lama_reference_stem="${3:-}"
  local source_path="$DATA_ROOT/$scene_name"
  local model_path="$OUTPUT_ROOT/$scene_name/aurafusion360"
  local raw_mask_dir="$source_path/object_mask"
  local reference_args=()
  if [[ -n "$lama_reference_stem" ]]; then
    reference_args+=(--lama_reference_stem "$lama_reference_stem")
  fi

  echo "============================================================"
  echo "scene        : $scene_name"
  echo "source_path  : $source_path"
  echo "model_path   : $model_path"
  echo "raw_mask_dir : $raw_mask_dir"
  echo "target_ids   : $target_ids"
  echo "lama_ref     : ${lama_reference_stem:-<max_mask>}"
  echo "============================================================"

  bash "$PIPELINE_SCRIPT" \
    -s "$source_path" \
    -m "$model_path" \
    --raw_mask_dir "$raw_mask_dir" \
    --target_ids "$target_ids" \
    "${COMMON_ARGS[@]}" \
    "${reference_args[@]}" \
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
    IFS='|' read -r scene_name target_ids lama_reference_stem <<< "$config"
    run_scene "$scene_name" "$target_ids" "$lama_reference_stem"
  done
else
  for scene_name in "${SELECTED_SCENES[@]}"; do
    config="$(find_scene_config "$scene_name")" || {
      echo "Unknown scene: $scene_name" >&2
      echo "Available scenes: $(list_scene_names)" >&2
      exit 1
    }
    IFS='|' read -r _ target_ids lama_reference_stem <<< "$config"
    run_scene "$scene_name" "$target_ids" "$lama_reference_stem"
  done
fi

# 我直接在 wrapper 后边加 --use_depth_loss 就行了，pipeline.sh 里会覆盖默认值，没必要在这里加一个环境变量了
