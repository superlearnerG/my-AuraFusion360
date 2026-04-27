#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

import cv2
import numpy as np
import torch
from PIL import Image

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, PILtoTorch_depth
from utils.graphics_utils import fov2focal


WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if len(cam_info.image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    # load mask if available
    gt_image_mask = None
    if cam_info.image_mask is not None:
        resized_image_mask_rgb = cam_info.image_mask.resize(resolution, resample=Image.Resampling.NEAREST)
        gt_image_mask = np.array(resized_image_mask_rgb)
        gt_image_mask = np.where(gt_image_mask > 127, 1, 0).astype(np.uint8)
        
        # Process mask
        if args.stage == "train" or args.stage == "removal":
            if args.dilate_mask_iter > 0:
                try:
                    dilate_kernel_size = args.dilate_mask_kernel_size
                    dilate_iter = args.dilate_mask_iter
                    gt_image_mask = cv2.dilate(gt_image_mask, np.ones((dilate_kernel_size, dilate_kernel_size), dtype=np.uint8), iterations=dilate_iter)
                except Exception as e:
                    print(f"Error processing mask {cam_info.image_name} at inpainting stage: {e}")
                    gt_image_mask = gt_image_mask
        elif args.stage == "inpaint":
            if args.dilate_mask_iter > 0:
                try:
                    dilate_kernel_size = args.dilate_mask_kernel_size
                    dilate_iter = args.dilate_mask_iter
                    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
                    cleaned_mask = cv2.morphologyEx(gt_image_mask, cv2.MORPH_OPEN, kernel)
                    if not getattr(args, "keep_all_unseen_components", False):
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask)
                        if num_labels > 1:
                            largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                            cleaned_component = np.zeros_like(labels)
                            cleaned_component[labels == largest_component] = 1
                            cleaned_mask = cleaned_component.astype(np.uint8)
                    cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=dilate_iter)
                    gt_image_mask = cleaned_mask
                except Exception as e:
                    print(f"Error processing mask {cam_info.image_name} at inpaint stage: {e}")
                    gt_image_mask = gt_image_mask
                
                tmp_dir = getattr(args, "unseen_mask_dilated_dir", "") or os.path.join(args.source_path, "unseen_masks_dilated")
                os.makedirs(tmp_dir, exist_ok=True)
                Image.fromarray(gt_image_mask * 255).save(os.path.join(tmp_dir, os.path.basename(cam_info.image_path)))
        
        gt_image_mask = torch.tensor(gt_image_mask, dtype=torch.float).cuda()
    
    # load depth if available
    image_depth = None
    if cam_info.depth is not None:
        image_depth = PILtoTorch_depth(cam_info.depth, resolution)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, image_mask=gt_image_mask, depth=image_depth, is_forward_facing=args.is_forward_facing)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
