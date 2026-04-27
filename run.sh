#!/bin/bash
dataset_name=$1
scene_name=$2
port=$3

SECONDS=0

python train.py --config configs/$dataset_name/$scene_name/train.config --port $port &&
python render.py -s data/$dataset_name/$scene_name -m output/$dataset_name/$scene_name --skip_mesh --render_path --iteration 30000 &&
python remove.py --config configs/$dataset_name/$scene_name/remove.config &&
python utils/sam2_utils.py --dataset $dataset_name --scene $scene_name &&
# python scripts/visualize_mask.py --dataset $dataset_name --scene $scene_name --type mask && # (optional) 
# python scripts/visualize_mask.py --dataset $dataset_name --scene $scene_name --type contour && # (optional) 
python inpaint.py --config configs/$dataset_name/$scene_name/inpaint.config &&
python utils/LeftRefill/sdedit_utils.py --config configs/$dataset_name/$scene_name/sdedit.config && 
python inpaint.py --config configs/$dataset_name/$scene_name/inpaint.config --images inpaint --finetune_iteration 10000 


minutes=$((SECONDS / 60))
seconds=$((SECONDS % 60))
echo "Total time elapsed: ${minutes} minutes and ${seconds} seconds"
