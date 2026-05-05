#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

SOURCE_PATH="../../siga26/data/figurines"
MODEL_PATH="../../siga26/output/figurines/aurafusion360"
RAW_MASK_DIR=""
REFERENCE_DIR=""
WORKFLOW_CONFIG=""
TARGET_IDS=""
BASE_ITERATION=30000
TRAIN_ITERATIONS=30000
FIT_MASK_ITERATIONS=2000
FINETUNE_ITERATION=5000
REMOVAL_THRESH=0.7
REFERENCE_INDEX=0
DILATE_MASK_KERNEL_SIZE=5
DILATE_MASK_ITER=3
PORT=6017
SKIP_BASE_TRAIN=0
RENDER_INTERMEDIATE=0
USE_DEPTH_LOSS=0
DEPTHS=""
DEPTH_SCALE=0.0
DEPTH_L1_WEIGHT_INIT=1.0
DEPTH_L1_WEIGHT_FINAL=0.01
DEPTH_ALIGN_MIN_VAL=-1.0
DEPTH_ALIGN_MAX_VAL=0.0
DEPTH_ALIGN_PERCENTILE=99.5

write_time_consuming() {
  local end_time end_text elapsed hours minutes seconds elapsed_hms output_path
  end_time="$(date +%s)"
  end_text="$(date '+%F %T')"
  elapsed=$((end_time - PIPELINE_START_TIME))
  hours=$((elapsed / 3600))
  minutes=$(((elapsed % 3600) / 60))
  seconds=$((elapsed % 60))
  printf -v elapsed_hms "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
  mkdir -p "$MODEL_PATH"
  output_path="$MODEL_PATH/time_consuming.txt"
  {
    printf 'source_path: %s\n' "$SOURCE_PATH"
    printf 'model_path: %s\n' "$MODEL_PATH"
    printf 'start_time: %s\n' "$PIPELINE_START_TEXT"
    printf 'end_time: %s\n' "$end_text"
    printf 'elapsed_seconds: %d\n' "$elapsed"
    printf 'elapsed_hms: %s\n' "$elapsed_hms"
  } > "$output_path"
  echo "[$end_text] Total pipeline time: $elapsed_hms ($elapsed seconds). Wrote $output_path"
}

usage() {
  cat <<'EOF'
Usage:
  bash pipeline.sh [options]

Default paths:
  --source_path ../../siga26/data/figurines
  --model_path  ../../siga26/output/figurines/aurafusion360
  --raw_mask_dir <source_path>/object_mask

Options:
  -s, --source_path PATH          COLMAP scene root with images/ and sparse/
  -m, --model_path PATH           AuraFusion360 output/model root
  --raw_mask_dir PATH             Multi-gray masks. Default: <source_path>/object_mask
  --reference_dir PATH            Deprecated for iterative inpaint; removal renders are used instead
  --workflow_config PATH          Output workflow JSON. Default: <model_path>/aura_iterative/workflow_config.json
  --target_ids IDS                Optional round ids, e.g. "1,[2,3,4],5". Default: all non-zero ids as separate rounds.
  --base_iteration N              Base checkpoint iteration used by iterative workflow. Default: 30000
  --train_iterations N            Initial Aura/3DGS training iterations. Default: 30000
  --fit_mask_iterations N         Per-round _is_masked fitting iterations. Default: 2000
  --finetune_iteration N          Per-round final inpaint finetune iterations. Default: 5000
  --removal_thresh V              Gaussian removal threshold. Default: 0.7
  --reference_index N             Removal-render index used as inpaint/LeftRefill reference. Default: 0
  --dilate_mask_kernel_size N     Unseen mask dilation kernel. Default: 5
  --dilate_mask_iter N            Unseen mask dilation iterations. Default: 3
  --port N                        train.py GUI/network port. Default: 6017
  --skip_base_train               Skip initial train.py and reuse <model_path>/point_cloud/iteration_<base_iteration>
  --render_intermediate           Keep non-final round train/test renders
  --use_depth_loss                Enable raw depth supervision during base training
  --depths PATH                   Raw .npy depth folder. Default when enabled: <source_path>/depth
  --depth_scale V                 Raw-depth to COLMAP/Aura scale. Default: 0.0, estimate from COLMAP tracks
  --depth_l1_weight_init V        Initial inverse-depth loss weight. Default: 1.0
  --depth_l1_weight_final V       Final inverse-depth loss weight. Default: 0.01
  --depth_align_min_val V         AGDD normalization min. Default: -1.0, use observed finite non-zero min
  --depth_align_max_val V         AGDD normalization max. Default: 0.0, use percentile; set 30 for issue #11 behavior
  --depth_align_percentile V      AGDD max percentile when max is unset. Default: 99.5; use 100 for raw max
  -h, --help                      Show this help

Environment:
  PYTHON_BIN                      Python executable to use. Default: python
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
    --raw_mask_dir)
      RAW_MASK_DIR="$2"
      shift 2
      ;;
    --reference_dir)
      REFERENCE_DIR="$2"
      shift 2
      ;;
    --workflow_config)
      WORKFLOW_CONFIG="$2"
      shift 2
      ;;
    --target_ids)
      TARGET_IDS="$2"
      shift 2
      ;;
    --base_iteration)
      BASE_ITERATION="$2"
      shift 2
      ;;
    --train_iterations)
      TRAIN_ITERATIONS="$2"
      shift 2
      ;;
    --fit_mask_iterations)
      FIT_MASK_ITERATIONS="$2"
      shift 2
      ;;
    --finetune_iteration)
      FINETUNE_ITERATION="$2"
      shift 2
      ;;
    --removal_thresh)
      REMOVAL_THRESH="$2"
      shift 2
      ;;
    --reference_index)
      REFERENCE_INDEX="$2"
      shift 2
      ;;
    --dilate_mask_kernel_size)
      DILATE_MASK_KERNEL_SIZE="$2"
      shift 2
      ;;
    --dilate_mask_iter)
      DILATE_MASK_ITER="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --skip_base_train)
      SKIP_BASE_TRAIN=1
      shift
      ;;
    --render_intermediate)
      RENDER_INTERMEDIATE=1
      shift
      ;;
    --use_depth_loss)
      USE_DEPTH_LOSS=1
      shift
      ;;
    --depths)
      DEPTHS="$2"
      shift 2
      ;;
    --depth_scale)
      DEPTH_SCALE="$2"
      shift 2
      ;;
    --depth_l1_weight_init)
      DEPTH_L1_WEIGHT_INIT="$2"
      shift 2
      ;;
    --depth_l1_weight_final)
      DEPTH_L1_WEIGHT_FINAL="$2"
      shift 2
      ;;
    --depth_align_min_val)
      DEPTH_ALIGN_MIN_VAL="$2"
      shift 2
      ;;
    --depth_align_max_val)
      DEPTH_ALIGN_MAX_VAL="$2"
      shift 2
      ;;
    --depth_align_percentile)
      DEPTH_ALIGN_PERCENTILE="$2"
      shift 2
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

cd "$SCRIPT_DIR"
export TORCH_HOME="$SCRIPT_DIR/../pretrained_models/torch"

if [[ -z "$RAW_MASK_DIR" ]]; then
  RAW_MASK_DIR="$SOURCE_PATH/object_mask"
fi
if [[ -z "$WORKFLOW_CONFIG" ]]; then
  WORKFLOW_CONFIG="$MODEL_PATH/aura_iterative/workflow_config.json"
fi

if [[ ! -d "$SOURCE_PATH" ]]; then
  echo "source_path not found: $SOURCE_PATH" >&2
  exit 1
fi
if [[ ! -d "$RAW_MASK_DIR" ]]; then
  echo "raw_mask_dir not found: $RAW_MASK_DIR" >&2
  exit 1
fi
if [[ -n "$REFERENCE_DIR" ]]; then
  echo "[$(date '+%F %T')] Warning: --reference_dir is ignored by iterative inpaint; per-round removal renders are used instead."
fi

mkdir -p "$MODEL_PATH"
mkdir -p "$(dirname "$WORKFLOW_CONFIG")"

PIPELINE_START_TIME="$(date +%s)"
PIPELINE_START_TEXT="$(date '+%F %T')"

TRAIN_DEPTH_ARGS=()
if (( USE_DEPTH_LOSS )); then
  if [[ -z "$DEPTHS" ]]; then
    DEPTHS="$SOURCE_PATH/depth"
  fi
  TRAIN_DEPTH_ARGS+=(
    --use_depth_loss
    --depths "$DEPTHS"
    --depth_scale "$DEPTH_SCALE"
    --depth_l1_weight_init "$DEPTH_L1_WEIGHT_INIT"
    --depth_l1_weight_final "$DEPTH_L1_WEIGHT_FINAL"
  )
fi

echo "[$(date '+%F %T')] Generating workflow config: $WORKFLOW_CONFIG"
SOURCE_PATH="$SOURCE_PATH" \
RAW_MASK_DIR="$RAW_MASK_DIR" \
WORKFLOW_CONFIG="$WORKFLOW_CONFIG" \
TARGET_IDS="$TARGET_IDS" \
BASE_ITERATION="$BASE_ITERATION" \
FIT_MASK_ITERATIONS="$FIT_MASK_ITERATIONS" \
FINETUNE_ITERATION="$FINETUNE_ITERATION" \
REMOVAL_THRESH="$REMOVAL_THRESH" \
REFERENCE_INDEX="$REFERENCE_INDEX" \
DILATE_MASK_KERNEL_SIZE="$DILATE_MASK_KERNEL_SIZE" \
DILATE_MASK_ITER="$DILATE_MASK_ITER" \
DEPTH_ALIGN_MIN_VAL="$DEPTH_ALIGN_MIN_VAL" \
DEPTH_ALIGN_MAX_VAL="$DEPTH_ALIGN_MAX_VAL" \
DEPTH_ALIGN_PERCENTILE="$DEPTH_ALIGN_PERCENTILE" \
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def parse_target_specs(text):
    text = text.strip()
    if not text:
        return None
    specs = []
    index = 0
    length = len(text)

    def skip_separators(pos):
        while pos < length and (text[pos].isspace() or text[pos] == ","):
            pos += 1
        return pos

    def parse_int(pos):
        start = pos
        if pos < length and text[pos] in "+-":
            pos += 1
        while pos < length and text[pos].isdigit():
            pos += 1
        if start == pos or (text[start] in "+-" and start + 1 == pos):
            raise ValueError(f"Expected integer near: {text[start:start + 20]!r}")
        return int(text[start:pos]), pos

    index = skip_separators(index)
    while index < length:
        if text[index] == "[":
            end = text.find("]", index + 1)
            if end < 0:
                raise ValueError(f"Missing closing bracket in --target_ids: {text!r}")
            body = text[index + 1:end].strip()
            if not body:
                raise ValueError("Empty bracket group in --target_ids")
            group = []
            body_index = 0
            body_len = len(body)
            while body_index < body_len:
                while body_index < body_len and (body[body_index].isspace() or body[body_index] == ","):
                    body_index += 1
                if body_index >= body_len:
                    break
                start = body_index
                if body[body_index] in "+-":
                    body_index += 1
                while body_index < body_len and body[body_index].isdigit():
                    body_index += 1
                if start == body_index or (body[start] in "+-" and start + 1 == body_index):
                    raise ValueError(f"Expected integer inside bracket group: {body!r}")
                group.append(int(body[start:body_index]))
            if not group:
                raise ValueError("Empty bracket group in --target_ids")
            specs.append(group)
            index = end + 1
        else:
            value, index = parse_int(index)
            specs.append(value)
        index = skip_separators(index)
    return specs


def scan_ids(mask_dir):
    ids = set()
    for path in sorted(mask_dir.iterdir()):
        if path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue
        mask = np.array(Image.open(path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        ids.update(int(value) for value in np.unique(mask) if int(value) != 0)
    return sorted(ids)


source_path = Path(os.environ["SOURCE_PATH"]).expanduser().resolve()
raw_mask_dir = Path(os.environ["RAW_MASK_DIR"]).expanduser().resolve()
workflow_config = Path(os.environ["WORKFLOW_CONFIG"]).expanduser().resolve()

target_specs = parse_target_specs(os.environ["TARGET_IDS"])
if target_specs is None:
    target_specs = scan_ids(raw_mask_dir)
if not target_specs:
    raise RuntimeError(f"No non-zero target ids found in {raw_mask_dir}")

workflow = {
    "base_iteration": int(os.environ["BASE_ITERATION"]),
    "raw_object_mask_dir": str(raw_mask_dir),
    "source_path": str(source_path),
    "defaults": {
        "removal_thresh": float(os.environ["REMOVAL_THRESH"]),
        "fit_mask_iterations": int(os.environ["FIT_MASK_ITERATIONS"]),
        "finetune_iteration": int(os.environ["FINETUNE_ITERATION"]),
        "reference_index": int(os.environ["REFERENCE_INDEX"]),
        "dilate_mask_kernel_size": int(os.environ["DILATE_MASK_KERNEL_SIZE"]),
        "dilate_mask_iter": int(os.environ["DILATE_MASK_ITER"]),
        "depth_align_min_val": float(os.environ["DEPTH_ALIGN_MIN_VAL"]),
        "depth_align_max_val": float(os.environ["DEPTH_ALIGN_MAX_VAL"]),
        "depth_align_percentile": float(os.environ["DEPTH_ALIGN_PERCENTILE"]),
        "skip_eval": True,
    },
    "rounds": [
        {"target_id": [int(item) for item in target_spec]}
        if isinstance(target_spec, list)
        else {"target_id": int(target_spec)}
        for target_spec in target_specs
    ],
}

workflow_config.parent.mkdir(parents=True, exist_ok=True)
workflow_config.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
print("target_specs:", target_specs)
print("round_count:", len(target_specs))
PY

OBJECT_NUM="$("$PYTHON_BIN" - <<'PY' "$WORKFLOW_CONFIG"
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(len(json.load(handle)["rounds"]))
PY
)"

if (( SKIP_BASE_TRAIN )); then
  BASE_PLY="$MODEL_PATH/point_cloud/iteration_${BASE_ITERATION}/point_cloud.ply"
  if [[ ! -f "$BASE_PLY" ]]; then
    echo "--skip_base_train was set, but base checkpoint is missing: $BASE_PLY" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] Skipping base training; using $BASE_PLY"
else
  echo "[$(date '+%F %T')] Training base AuraFusion360 checkpoint"
  "$PYTHON_BIN" train.py \
    -s "$SOURCE_PATH" \
    -m "$MODEL_PATH" \
    --object_mask_dir "$RAW_MASK_DIR" \
    --iterations "$TRAIN_ITERATIONS" \
    --save_iterations "$BASE_ITERATION" "$TRAIN_ITERATIONS" \
    --test_iterations "$BASE_ITERATION" "$TRAIN_ITERATIONS" \
    --optimize_is_masked_iter "$TRAIN_ITERATIONS" \
    --optimize_is_seen_iter "$TRAIN_ITERATIONS" \
    "${TRAIN_DEPTH_ARGS[@]}" \
    --port "$PORT"
fi

echo "[$(date '+%F %T')] Rendering vanilla 3DGS test split for evaluation"
"$PYTHON_BIN" render.py \
  -s "$SOURCE_PATH" \
  -m "$MODEL_PATH" \
  --object_mask_dir "$RAW_MASK_DIR" \
  --iteration "$BASE_ITERATION" \
  --skip_train \
  --skip_mesh

echo "[$(date '+%F %T')] Evaluating vanilla 3DGS test split"
"$PYTHON_BIN" scripts/evaluate_vanilla_3dgs.py \
  -m "$MODEL_PATH" \
  --iteration "$BASE_ITERATION" \
  --output "$MODEL_PATH/evaluation_results.txt"

ITERATIVE_ARGS=()
if (( RENDER_INTERMEDIATE )); then
  ITERATIVE_ARGS+=(--render_intermediate)
fi

echo "[$(date '+%F %T')] Running iterative multi-object inpaint, rounds=$OBJECT_NUM"
PYTHON_BIN="$PYTHON_BIN" bash "$SCRIPT_DIR/iterative_inpaint.sh" \
  -s "$SOURCE_PATH" \
  -m "$MODEL_PATH" \
  --workflow_config "$WORKFLOW_CONFIG" \
  --object_num "$OBJECT_NUM" \
  "${ITERATIVE_ARGS[@]}"

echo "[$(date '+%F %T')] Iterative workflow status"
"$PYTHON_BIN" iterative_inpaint.py status \
  -s "$SOURCE_PATH" \
  -m "$MODEL_PATH" \
  --workflow_config "$WORKFLOW_CONFIG"

echo "Workflow config: $WORKFLOW_CONFIG"
echo "Outputs root: $MODEL_PATH/aura_iterative"

write_time_consuming


# 运行指令：

# bash pipeline.sh \
#   -s ../../siga26/data/figurines \
#   -m ../../siga26/output/figurines/aurafusion360 \
#   --raw_mask_dir ../../siga26/data/figurines/object_mask \
#   --target_ids "[30,83,72,51],[40,62,105,94],[115,158,222,233],[169,180,126,142],[201,212,244,190]" \
#   --removal_thresh 0.7 \
#   --fit_mask_iterations 2000 \
#   --finetune_iteration 5000 \
#   --skip_base_train

# bash pipeline.sh \
#   -s ../../siga26/data/bear \
#   -m ../../siga26/output/bear/aurafusion360 \
#   --raw_mask_dir ../../siga26/data/bear/object_mask \
#   --target_ids "128" \
#   --removal_thresh 0.7 \
#   --fit_mask_iterations 2000 \
#   --finetune_iteration 5000

# bash pipeline.sh \
#   -s ../../siga26/data/bonsai \
#   -m ../../siga26/output/bonsai/aurafusion360 \
#   --raw_mask_dir ../../siga26/data/bonsai/object_mask \
#   --target_ids "128" \
#   --removal_thresh 0.7 \
#   --fit_mask_iterations 2000 \
#   --finetune_iteration 5000

# bash pipeline.sh \
#   -s ../../siga26/data/fruits \
#   -m ../../siga26/output/fruits/aurafusion360 \
#   --raw_mask_dir ../../siga26/data/fruits/object_mask \
#   --target_ids "[48,30,67,86,105,123,142],[161,180,217,198,236]" \
#   --removal_thresh 0.7 \
#   --fit_mask_iterations 2000 \
#   --finetune_iteration 5000

# bash pipeline.sh \
#   -s ../../siga26/data/doppelherz \
#   -m ../../siga26/output/doppelherz/aurafusion360 \
#   --raw_mask_dir ../../siga26/data/doppelherz/object_mask \
#   --target_ids "[30,105,180]" \
#   --removal_thresh 0.7 \
#   --fit_mask_iterations 2000 \
#   --finetune_iteration 5000
