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

import numpy as np
import torch
from torch import nn

from utils.graphics_utils import (
    getProjectionMatrix,
    getProjectionMatrixForwardFacing,
    getWorld2View2,
)


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", image_mask=None, depth=None, is_forward_facing=False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        self.original_image_mask = image_mask
        self.original_image_depth = None
        self.depth_mask = None
        self.depth_reliable = False
        if depth is not None:
            depth = depth.to(self.data_device).float()
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            if depth.shape[0] != 1:
                depth = depth[:1]
            valid_depth = torch.isfinite(depth) & (depth > 0.0)
            depth = torch.where(valid_depth, depth, torch.zeros_like(depth))
            self.original_image_depth = depth
            self.depth_mask = valid_depth.float()
            self.depth_reliable = bool(valid_depth.any().item())
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
            if self.depth_mask is not None:
                self.depth_mask *= self.gt_alpha_mask
                self.depth_reliable = bool((self.depth_mask > 0).any().item())
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        if is_forward_facing:
            # TODO: set the spin-nerf dataset cx = 0, cy = 0. This is a temporary fix
            self.projection_matrix = getProjectionMatrixForwardFacing(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=0, cy=0).transpose(0,1).cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
