import os
import re
import shutil
from argparse import ArgumentParser
from glob import glob

import cv2
import numpy as np
from natsort import natsorted
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm

from utils.pretrained_paths import require_pretrained_file

IMAGE_EXTENSIONS = {".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"}


predictor = build_sam2_video_predictor(
    config_file="configs/sam2/sam2_hiera_l.yaml",
    ckpt_path=str(require_pretrained_file("sam2-hiera-large", "sam2_hiera_large.pt", min_bytes=1024 * 1024)),
)


def _image_paths(directory):
    return sorted(
        [
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name)) and os.path.splitext(name)[1] in IMAGE_EXTENSIONS
        ],
        key=lambda path: os.path.splitext(os.path.basename(path))[0],
    )


def _split_entry_keys(text):
    name = os.path.basename(text.strip())
    if not name:
        return set()
    stem = os.path.splitext(name)[0]
    return {name, stem}


def _load_split_list(list_path):
    entries = set()
    with open(list_path, "r", encoding="utf-8") as handle:
        for line in handle:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            entries.update(_split_entry_keys(item))
    return entries


def _load_dataset_split(source_path):
    train_list_path = os.path.join(source_path, "train_list.txt")
    test_list_path = os.path.join(source_path, "test_list.txt")
    has_train_list = os.path.exists(train_list_path)
    has_test_list = os.path.exists(test_list_path)

    if has_train_list and has_test_list:
        train_entries = _load_split_list(train_list_path)
        test_entries = _load_split_list(test_list_path)
        overlap = train_entries.intersection(test_entries)
        if overlap:
            sample = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"train_list.txt and test_list.txt overlap: {sample}")
        return {
            "mode": "list",
            "train_entries": train_entries,
            "test_entries": test_entries,
        }

    return {
        "mode": "holdout",
        "train_entries": set(),
        "test_entries": set(),
    }


def _basename_sequence_number(image_name):
    stem = os.path.splitext(os.path.basename(image_name))[0]
    match = re.search(r"\d+$", stem)
    if match is None:
        return None
    return int(match.group(0))


def _split_for_image(image_name, split_config, llffhold=8):
    keys = _split_entry_keys(image_name)
    if split_config["mode"] == "list":
        in_train = bool(keys.intersection(split_config["train_entries"]))
        in_test = bool(keys.intersection(split_config["test_entries"]))
        if in_train and in_test:
            raise ValueError(f"Image appears in both train and test splits: {image_name}")
        if in_train:
            return "train"
        if in_test:
            return "test"
        return None

    sequence_number = _basename_sequence_number(image_name)
    if sequence_number is not None and sequence_number % llffhold == 0:
        return "test"
    return "train"


def _train_name_paths(source_path, name_dir, frame_count):
    split_config = _load_dataset_split(source_path)
    image_paths = [
        path
        for path in _image_paths(name_dir)
        if _split_for_image(os.path.basename(path), split_config) == "train"
    ]

    if len(image_paths) != frame_count:
        raise ValueError(
            "Train image/mask frame count mismatch: "
            f"{len(image_paths)} train image names from {name_dir}, "
            f"but {frame_count} removal frames were generated. "
            "Check train/test split files or holdout naming."
        )
    return image_paths


def export_unseen_mask(source_path, output_path, unseen_mask_output_dir=None, name_dir=None):
    """
    Export unseen mask by using unseen contour as bbox prompt to SAM2.
    Save the mask as png image with the same name as the image in the source_path.
    
    source_path: path to the source directory
    output_path: path to the output directory
    """
    removal_image_dir = os.path.join(output_path, "renders")
    unseen_contour_dir = os.path.join(output_path, "unseen_contour")
    unseen_mask_dir = unseen_mask_output_dir or os.path.join(source_path, "unseen_masks")
    name_dir = name_dir or os.path.join(source_path, "images")
    os.makedirs(unseen_mask_dir, exist_ok=True)
    # 1. 'init_state' only support .jpg, so save all removal images in .png to another dir but as .jpg.
    removal_image_dir_jpg = os.path.join(os.path.dirname(removal_image_dir), "renders_jpg")
    os.makedirs(removal_image_dir_jpg, exist_ok=True)
    for file in os.listdir(removal_image_dir):
        if file.endswith(".png"):
            shutil.copy(os.path.join(removal_image_dir, file), os.path.join(removal_image_dir_jpg, file.replace(".png", ".jpg")))
    removal_frame_paths = natsorted(glob(os.path.join(removal_image_dir_jpg, "*.jpg")))
    
    # 2. init_state
    state = predictor.init_state(video_path=removal_image_dir_jpg)
    predictor.reset_state(state)
    
    # 3. add new prompts and instantly get the output on the same frame
    unseen_contour_paths = natsorted(glob(os.path.join(unseen_contour_dir, "*.png")))
    for frame_idx, unseen_contour_path in tqdm(enumerate(unseen_contour_paths), desc="Add unseen contour as a bbox prompt"):
        unseen_contour = Image.open(unseen_contour_path)
        unseen_contour_np = np.array(unseen_contour) # shape (H, W)
        
        # TODO: Opt. Some mask operations
        # unseen_contour_np = cv2.morphologyEx(unseen_contour_np, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        # unseen_contour_np = cv2.morphologyEx(unseen_contour_np, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        
        unseen_contour_np = np.where(unseen_contour_np > 0, 1, 0)
        
        # get the bbox of the unseen contour
        y_indices, x_indices = np.nonzero(unseen_contour_np)
        if len(y_indices) == 0 or len(x_indices) == 0:
            print(f"No unseen contour found at frame {frame_idx}")
            continue
        
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        bbox = np.array([[x_min, y_min], [x_max, y_max]])
        # add new prompts and instantly get the output on the same frame
        frame_idx, object_ids, masks = predictor.add_new_points_or_box(
            state, 
            frame_idx=frame_idx, 
            obj_id=1,
            box=bbox
        )
        
    # 4. propagate the prompts to get masklets throughout the video
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }   
        
    # 5. visualize the masks with red & save to tmp
    vis_frame_stride = 30
    for out_frame_idx in range(0, len(removal_frame_paths), vis_frame_stride):
        img = Image.open(removal_frame_paths[out_frame_idx])
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            img = np.array(img)
            img[out_mask[0]] = (255, 0, 0)
            img = Image.fromarray(img)
            os.makedirs("tmp/unseen_masks", exist_ok=True)
            img.save(os.path.join("tmp/unseen_masks", f"{out_frame_idx}_{out_obj_id}.jpg"))

    # 6. save the output unseenmasks to unseen_masks dir, and in the same name as the image in the source_path
    name_paths = _train_name_paths(source_path, name_dir, len(removal_frame_paths))
    for out_frame_idx in range(0, len(removal_frame_paths)):
        file_name = os.path.basename(name_paths[out_frame_idx])
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            out_mask = np.where(out_mask > 0, 255, 0)
            out_mask = out_mask.astype(np.uint8)[0]
            out_mask = Image.fromarray(out_mask)
            out_mask.save(os.path.join(unseen_mask_dir, file_name))
            
    # cleanup
    shutil.rmtree(removal_image_dir_jpg)
            
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default=None, choices=["360-USID", "Other-360"])
    parser.add_argument("--scene", "-s", type=str, default=None)
    parser.add_argument("--source_path", type=str, default=None)
    parser.add_argument("--removal_output_path", type=str, default=None)
    parser.add_argument("--unseen_mask_output_dir", type=str, default=None)
    parser.add_argument("--name_dir", type=str, default=None, help="Image directory whose filenames are used for output masks")
    args = parser.parse_args()

    if args.source_path or args.removal_output_path:
        if not args.source_path or not args.removal_output_path:
            raise ValueError("--source_path and --removal_output_path must be provided together")
        export_unseen_mask(
            source_path=args.source_path,
            output_path=args.removal_output_path,
            unseen_mask_output_dir=args.unseen_mask_output_dir,
            name_dir=args.name_dir,
        )
    else:
        if not args.dataset or not args.scene:
            raise ValueError("Either explicit paths or --dataset/--scene are required")
        dataset_name = args.dataset
        scene_name = args.scene
        export_unseen_mask(
            source_path=f"data/{dataset_name}/{scene_name}",
            output_path=f"output/{dataset_name}/{scene_name}/train/ours_30000_object_removal"
        )
