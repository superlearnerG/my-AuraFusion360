import argparse
import os
import re

import cv2
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
from natsort import natsorted
from PIL import Image
from tqdm import tqdm

# Usage Example:
# 1. contour:
# python scripts/visualize_mask.py -i output/Other-360/pinecone/train/ours_30000_object_removal/renders/ -m output/Other-360/pinecone/train/ours_30000_object_removal/unseen_contour/ -o visualize_results/unseen_contours/Other-360/pinecone.mp4
# 2. unseen mask:
# python scripts/visualize_mask.py -i output/Other-360/pinecone/train/ours_30000_object_removal/renders/ -m data/Other-360/pinecone/unseen_masks/ -o visualize_results/unseen_masks/Other-360/pinecone.mp4


def blend_image_with_mask(image, mask, alpha=0.5, obj_id=None):
    """
    Blends an image with a colormap color where the mask is 255 and keeps the original image where the mask is 0.

    Parameters:
    - image: The original image as a numpy array (H, W, 3).
    - mask: The binary mask image (H, W) with values 0 and 255.
    - alpha: The opacity level of the color in the masked area (0 = fully transparent, 1 = fully opaque).
    - obj_id: Index for the colormap color.

    Returns:
    - Blended image as a numpy array.
    """
    # Ensure the mask is a 2D array
    mask = mask.astype(np.uint8)

    # Get the color from the colormap based on the obj_id
    cmap = plt.get_cmap("tab10")
    cmap_idx = 0 if obj_id is None else obj_id
    red = np.array([1, 0, 0, 1])
    color = np.array(red[:3])  # Extract the RGB values from the colormap

    # Normalize the mask to range [0, 1] for blending
    mask_normalized = mask / 255.0

    # Create a color overlay of the same shape as the image
    color_overlay = (
        np.ones_like(image, dtype=np.float32) * color * 255
    )  # Scale to 0-255

    # Blend the color overlay with the original image based on the mask
    blended_image = image * (1 - mask_normalized[..., None] * alpha) + color_overlay * (
        mask_normalized[..., None] * alpha
    )

    return blended_image


def render_video_with_mask(
    img_dir, mask_dir, output_video, alpha=0.5, fps=30, obj_id=None
):
    """
    Renders a video by blending images with corresponding masks using a colormap color in masked areas.

    Parameters:
    - img_dir: Directory containing the input images.
    - mask_dir: Directory containing the corresponding masks.
    - output_video: Path to the output video file.
    - alpha: Opacity of the color in the masked areas.
    - fps: Frames per second for the output video.
    - obj_id: Index for the colormap color.
    """
    # Get sorted list of images and masks
    img_files = natsorted(
        [
            f
            for f in os.listdir(img_dir)
            if f.endswith((".png", ".jpg", ".jpeg", "PNG", "JPG", "JPEG"))
        ]
    )
    mask_files = natsorted(
        [
            f
            for f in os.listdir(mask_dir)
            if f.endswith((".png", ".jpg", ".jpeg", "PNG", "JPG", "JPEG"))
        ]
    )

    print(f"Found {len(img_files)} images and {len(mask_files)} masks.")

    # Check if the number of images matches the number of masks
    if len(img_files) != len(mask_files):
        print("The number of images and masks do not match.")

        print("Cropping to the shorter one: ", min(len(img_files), len(mask_files)))
        img_files = img_files[: min(len(img_files), len(mask_files))]
        mask_files = mask_files[: min(len(img_files), len(mask_files))]
        # return
        
    # make output directory
    output_dir = os.path.dirname(output_video)
    os.makedirs(output_dir, exist_ok=True)
    

    # Read the first image to get the frame size
    first_image_path = os.path.join(img_dir, img_files[0])
    first_image = np.array(Image.open(first_image_path).convert("RGB"))
    height, width, _ = first_image.shape


    video_kwargs = {
        'shape': (height, width),
        'codec': 'h264',
        'fps': 60,
        'crf': 18,
    }

    with media.VideoWriter(output_video, **video_kwargs, input_format='rgb') as video_writer:

        # Create progress bar
        pbar = tqdm(zip(img_files, mask_files), total=len(img_files), desc="Processing frames")
        
        # Iterate over each image and mask, blend them, and write to the video
        for img_file, mask_file in pbar:
            # Update postfix with current file names
            pbar.set_postfix_str(f"Processed {img_file} with {mask_file}")
            
            img_path = os.path.join(img_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)

            # Read the image and mask
            image = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(
                Image.open(mask_path).convert("L")
            )  # Convert mask to grayscale (0 and 255)
            # mask = (mask > 0).astype(np.uint8) * 255  # Binarize the mask

            # Blend the image with the mask using the colormap color
            blended_image = blend_image_with_mask(image, mask, alpha, obj_id=obj_id)

            # Convert to the right format for OpenCV
            blended_image_bgr = cv2.cvtColor(
                blended_image.astype(np.uint8), cv2.COLOR_RGB2BGR
            )

            # Also turn the mask region into zero
            # image[mask > 127] = [0, 0, 0]

            # Write the frame to the video
            blended_image_rgb = cv2.cvtColor(blended_image_bgr, cv2.COLOR_BGR2RGB)
            
            # concatenated_image = np.concatenate((image, blended_image_rgb), axis=1)
            
            video_writer.add_image(blended_image_rgb)
            # video_writer.write(blended_image_bgr)
            # video_writer.add_image(concatenated_image)

        # Release the video writer
        # video_writer.release()
        print(f"Video saved to {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Either (provide dataset_name and scene_name) or (img_dir and mask_dir)")
    parser.add_argument("--img_dir", "-i", type=str)
    parser.add_argument("--mask_dir", "-m", type=str)
    parser.add_argument("--type", "-t", type=str, default="mask", choices=["contour", "mask"])
    parser.add_argument("--dataset_name", "-d", type=str)
    parser.add_argument("--scene_name", "-s", type=str)
    args = parser.parse_args()
    
    # Either (provide dataset_name and scene_name) or (img_dir and mask_dir)
    img_dir = None
    mask_dir = None
    output_file_path = None
    if args.dataset_name and args.scene_name:
        img_dir = os.path.join("output", args.dataset_name, args.scene_name, "train/ours_30000_object_removal/renders")
        if args.type == "contour":
            mask_dir = os.path.join("output", args.dataset_name, args.scene_name, "train/ours_30000_object_removal/unseen_contour")
        elif args.type == "mask":
            mask_dir = os.path.join("data", args.dataset_name, args.scene_name, "unseen_masks")
        
    else:
        img_dir = args.img_dir
        mask_dir = args.mask_dir
    output_file_path = os.path.join("visualize_results", "unseen_contour" if args.type == "contour" else "unseen_masks", args.dataset_name, f"{args.scene_name}.mp4")
    
    
    render_video_with_mask(
        img_dir, mask_dir, output_file_path
    )
