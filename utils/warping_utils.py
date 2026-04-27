import os
import subprocess

import cv2
import PIL
import torch
import torch.nn.functional as F
import torchvision
from utils.general_utils import vis_depth
from utils.graphics_utils import fov2focal, geom_transform_points

# from softmax_splatting.softsplat import softsplat
# from softmax_splatting import run

# import utils.debug_utils as debug_utils

@torch.no_grad()
def unproject(viewpoint_cam, depth, rgb, device, mask=None):
    """
    Unproject the depth and rgb to xyz and rgb
    Args:
        viewpoint_cam: the camera pose
        depth: the depth values of the image (torch.tensor)
        rgb: the rgb values of the image (torch.tensor)
        device: the device to run the unprojection on
        mask: the mask to apply to the unprojection (torch.tensor)
    Returns:
        xyz: the coordinates of the unprojected points in the world coordinate system
        rgb: the rgb values of the unprojected points
    """
    # # Sol.1
    # H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
    # fovx, fovy = viewpoint_cam.FoVx, viewpoint_cam.FoVy
    # fx, fy = fov2focal(fovx, W), fov2focal(fovy, H)
    # v, u = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    # z = depth
    
    # x = (u - W * 0.5) * z / fx
    # y = (v - H * 0.5) * z / fy
    
    # xyz = torch.stack([x, y, z], dim=0).reshape(3, -1).T
    
    # xyz = geom_transform_points(xyz, viewpoint_cam.world_view_transform.inverse())
    
    # Sol.2
    import math
    c2w = viewpoint_cam.world_view_transform.T.inverse()
    W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ viewpoint_cam.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    xyz = depth.reshape(-1, 1) * rays_d + rays_o
    
    
    if mask is not None:
        mask_flat = mask.reshape(-1)
        masked_xyz = xyz[mask_flat == 1]
        masked_rgb = rgb.reshape(3, -1).T[mask_flat == 1]
        return masked_xyz, masked_rgb
    else:
        return xyz, rgb.reshape(3, -1).T

@torch.no_grad()
def voxel_downsample(points, rgb, voxel_size):
    voxel_indices = torch.floor(points / voxel_size).int()
    
    # Create a unique key for each voxel
    unique_indices, inverse_indices, counts = torch.unique(voxel_indices, dim=0, return_inverse=True, return_counts=True)
    
    
    # Compute the centroid of points in each voxel
    voxel_centroids = torch.zeros((unique_indices.size(0), points.size(1)), device=points.device)
    voxel_centroids.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, points.size(1)), points)
    
    # Normalize by the count of points in each voxel
    voxel_centroids /= counts.float().unsqueeze(1)
    
    
    
    voxel_rgb = torch.zeros((unique_indices.size(0), rgb.size(1)), device=rgb.device)
    
    # Loop over each voxel to find the most common color
    for i in range(unique_indices.size(0)):
        voxel_rgb_points = rgb[inverse_indices == i]  # RGB values for points in the voxel
        unique_colors, color_counts = voxel_rgb_points.unique(dim=0, return_counts=True)  # Find unique colors and counts
        most_common_color = unique_colors[color_counts.argmax()]  # Find the color that occurs the most
        voxel_rgb[i] = most_common_color
    # Return the centroid of each voxel, and corresponding color
    return voxel_centroids, voxel_rgb
    


@torch.no_grad()
def warping(viewpoint_cam, viewpoint_cam2, seen01, depth, device):
    H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
    fovx, fovy = viewpoint_cam.FoVx, viewpoint_cam.FoVy
    fx, fy = fov2focal(fovx, W), fov2focal(fovy, H)
    projection_image = torch.zeros((3, H, W), device=device)
    projection_image_mask = torch.zeros((H, W), device=device)
    projection_depth = torch.full((1, H, W), float('inf'), device=device)
    v, u = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    
    # lift to camera i coordinates
    z = depth[0]
    x = (u - W * 0.5) * z / fx
    y = (v - H * 0.5) * z / fy
    xyz = torch.stack([x, y, z], dim=0).reshape(3, -1).T
    
    # camera i coordinates to world coordinates to camera j coordinates
    xyz = geom_transform_points(xyz, viewpoint_cam.world_view_transform.inverse())
    xyz = geom_transform_points(xyz, viewpoint_cam2.world_view_transform).T
    xyz = xyz.reshape(3, H, W)
    
    # camera j coordinates to plane coordinates
    x = xyz[0, :, :] / xyz[2, :, :] * fx + W * 0.5
    y = xyz[1, :, :] / xyz[2, :, :] * fy + H * 0.5
    x, y = x.round().long(), y.round().long()
    
    # render on plane
    valid_mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    
    original_indices = torch.stack([v[valid_mask], u[valid_mask]], dim=-1)
    projected_indices = torch.stack([y[valid_mask], x[valid_mask]], dim=-1)
    
    projection_image[:, y[valid_mask], x[valid_mask]] = seen01[:, v[valid_mask], u[valid_mask]]
    projection_image_mask[y[valid_mask], x[valid_mask]] = 1
    # breakpoint()
    return projection_image, original_indices, projected_indices
    
    
    # softmax splatting
    tenFlow = torch.stack([x - u, y - v], dim=0).float().unsqueeze(0)
    tenMetric = torch.exp(-projection_depth).unsqueeze(0) # use projection depth as tenMetric
    projection_image_softmax = softsplat(tenIn=viewpoint_cam.original_image.unsqueeze(0), tenFlow=tenFlow * 1.0, tenMetric=(600.0 * tenMetric).clip(-80.0, 80.0), strMode='soft') # -20.0 is a hyperparameter, called 'alpha' in the paper, that could be learned using a torch.Parameter
    
    # combine viewpoint_cam2 background and projection_image
    combined_image = viewpoint_cam2.original_image.clone()
    combined_mask = (viewpoint_cam2_removal_mask == 1) & (projection_image_mask == 1)
    combined_image[:, combined_mask == True] = projection_image_softmax[0][:, combined_mask == True]
    reserved_mask = ((viewpoint_cam2_removal_mask == 1) & (projection_image_mask == 0)).to(torch.float32)
    
    
    if debug_utils.DEBUG:
        torchvision.utils.save_image(combined_image, "tmp/combined_img_j.png")
        torchvision.utils.save_image(combined_image, "tmp/combined_img_j.png")
        torchvision.utils.save_image(projection_image, "tmp/proj_img_j.png")
        torchvision.utils.save_image(projection_image_softmax, "tmp/proj_img_j_softmax.png")
   
        # save images
        os.makedirs("tmp", exist_ok=True)
        # view i
        torchvision.utils.save_image(viewpoint_cam.original_image, "tmp/render_img_i.png")
        torchvision.utils.save_image(viewpoint_cam.original_image, "tmp/original_img_i.png")
        # depth_map = vis_depth(depth[0].detach().cpu().numpy())
        # cv2.imwrite("tmp/depth_map_i.png", depth_map)
        # view j
        torchvision.utils.save_image(viewpoint_cam2.original_image, "tmp/original_img_j.png")
        torchvision.utils.save_image(projection_image, "tmp/proj_img_j.png")
        torchvision.utils.save_image(viewpoint_cam2.original_image - projection_image, "tmp/diff_img_j.png")
        
        # misc
        torchvision.utils.save_image(viewpoint_cam2.original_image - viewpoint_cam.original_image, "tmp/diff.png")
        # normalized_projection_depth = (projection_depth - projection_depth.min()) / (projection_depth.max() - projection_depth.min())
        # torchvision.utils.save_image(normalized_projection_depth, "tmp/proj_depth_j.png")
        normalized_projection_depth = vis_depth(projection_depth[0].detach().cpu().numpy())
        cv2.imwrite("tmp/proj_depth_j.png", normalized_projection_depth)
    return combined_image, reserved_mask



@torch.no_grad()
def warping_backward(viewpoint_cam, viewpoint_cam2, seen01, depth, device):
    """
    With bug
    """
    H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
    fovx, fovy = viewpoint_cam.FoVx, viewpoint_cam.FoVy
    fx, fy = fov2focal(fovx, W), fov2focal(fovy, H)

    # Create a grid for the target view
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device=device).float(), 
                                     torch.arange(H, device=device).float(), 
                                     indexing='xy')

    grid_x = grid_x * 2 / (W - 1) - 1  # Normalize to [-1, 1]
    grid_y = grid_y * 2 / (H - 1) - 1  # Normalize to [-1, 1]
    
    # Create a grid of shape (H, W, 2)
    grid = torch.stack((grid_x, grid_y), dim=-1)  # Shape: (H, W, 2)

    # Lift to camera coordinates
    z = depth[0]
    x = (grid_x * z) / fx
    y = (grid_y * z) / fy
    xyz = torch.stack([x, y, z], dim=-1)  # Shape: (H, W, 3)

    # Transform to world coordinates
    xyz = geom_transform_points(xyz.reshape(-1, 3), viewpoint_cam.world_view_transform.inverse()).reshape(H, W, 3)

    # Transform from world coordinates to camera 2 coordinates
    xyz_cam2 = geom_transform_points(xyz.view(-1, 3), viewpoint_cam2.world_view_transform).reshape(H, W, 3)

    # Create grid for grid_sample
    grid_cam2 = torch.zeros((H, W, 2), device=device)
    grid_cam2[..., 0] = xyz_cam2[..., 0] / xyz_cam2[..., 2] * fx + W * 0.5  # x coordinate
    grid_cam2[..., 1] = xyz_cam2[..., 1] / xyz_cam2[..., 2] * fy + H * 0.5  # y coordinate
    
    # Normalize the grid for grid_sample
    grid_cam2 = grid_cam2 / torch.tensor([W, H], device=device) * 2 - 1  # Normalize to [-1, 1]

    # Perform backward warping
    projection_image = F.grid_sample(seen01.unsqueeze(0), grid_cam2.unsqueeze(0), mode='bilinear', padding_mode='zeros', align_corners=True)

    # Create a mask for valid projections
    projection_mask = (grid_cam2[..., 0] >= -1) & (grid_cam2[..., 0] <= 1) & (grid_cam2[..., 1] >= -1) & (grid_cam2[..., 1] <= 1)

    return projection_image.squeeze(0), projection_mask.float()