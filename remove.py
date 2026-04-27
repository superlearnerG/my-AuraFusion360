import os

import configargparse
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from scipy.spatial import Delaunay

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from utils.render_utils import create_videos, generate_path



def points_inside_convex_hull(point_cloud, mask, remove_outliers=True, outlier_factor=1.0):
    """
    Given a point cloud and a mask indicating a subset of points, this function computes the convex hull of the 
    subset of points and then identifies all points from the original point cloud that are inside this convex hull.
    
    Parameters:
    - point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the point cloud.
    - mask (torch.Tensor): A tensor of shape (N,) indicating the subset of points to be used for constructing the convex hull.
    - remove_outliers (bool): Whether to remove outliers from the masked points before computing the convex hull. Default is True.
    - outlier_factor (float): The factor used to determine outliers based on the IQR method. Larger values will classify more points as outliers.
    
    Returns:
    - inside_hull_tensor_mask (torch.Tensor): A mask of shape (N,) with values set to True for the points inside the convex hull 
                                              and False otherwise.
    """

    # Extract the masked points from the point cloud
    masked_points = point_cloud[mask].cpu().numpy()

    # Remove outliers if the option is selected
    if remove_outliers:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR))
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    else:
        filtered_masked_points = masked_points

    # Compute the Delaunay triangulation of the filtered masked points
    delaunay = Delaunay(filtered_masked_points)

    # Determine which points from the original point cloud are inside the convex hull
    points_inside_hull_mask = delaunay.find_simplex(point_cloud.cpu().numpy()) >= 0

    # Convert the numpy mask back to a torch tensor and return
    inside_hull_tensor_mask = torch.tensor(points_inside_hull_mask, device='cuda')

    return inside_hull_tensor_mask

def count_mask_pixels(mask_dir):
    if not mask_dir:
        return None
    if not os.path.isdir(mask_dir):
        return None
    total = 0
    for file_name in os.listdir(mask_dir):
        if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        mask = np.array(Image.open(os.path.join(mask_dir, file_name)).convert("L"))
        total += int((mask > 0).sum())
    return total


def removal_setup(opt, model_path, iteration, views, gaussians, pipeline, background, classifier, selected_obj_ids, cameras_extent, removal_thresh, outlier_factor=1.0, target_tag="", mask_pixel_count=None):
    selected_obj_ids = torch.tensor(selected_obj_ids).cuda()
    with torch.no_grad():
        prob_obj3d = gaussians.get_is_masked[..., :1]
        
        mask = prob_obj3d > removal_thresh # reserve the non-masked region
        selected_count = int(mask.sum().item())
        max_prob = float(prob_obj3d.max().item()) if prob_obj3d.numel() else 0.0
        mean_prob = float(prob_obj3d.mean().item()) if prob_obj3d.numel() else 0.0
        if selected_count == 0:
            target_text = target_tag or "unknown"
            pixel_text = "unknown" if mask_pixel_count is None else str(mask_pixel_count)
            raise RuntimeError(
                "Removal selected zero Gaussians "
                f"(target={target_text}, threshold={removal_thresh}, max_prob={max_prob:.6f}, "
                f"mean_prob={mean_prob:.6f}, mask_pixel_count={pixel_text}). "
                "Run fit_target_mask.py for this target or lower removal_thresh."
            )
        mask3d = mask
        if selected_count >= 4:
            mask3d_convex = points_inside_convex_hull(gaussians._xyz.detach(), mask3d.squeeze(), outlier_factor=outlier_factor)
            mask3d = torch.logical_or(mask3d, mask3d_convex.unsqueeze(1))
        else:
            print(f"Warning: only {selected_count} Gaussians selected; skipping convex-hull expansion")
    
    # remove & fix gaussians that outside the mask   
    gaussians.removal_setup(opt, mask3d)
    
    # save gaussians
    point_cloud_path = os.path.join(model_path, "point_cloud/iteration_{}_object_removal".format(iteration))
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    return gaussians

def removal(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, opt : OptimizationParams, select_obj_id : int, removal_thresh : float):
    # 1. load gaussian model
    dataset.stage = 'removal'
    
    gaussians_before_remove = GaussianModel(dataset.sh_degree)
    scene_before_remove = Scene(dataset, gaussians_before_remove, load_iteration=iteration, shuffle=False)
    
    gaussians = GaussianModel(dataset.sh_degree)
    
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 2. remove selected object
    mask_pixel_count = count_mask_pixels(getattr(args, "object_mask_dir", ""))
    gaussians = removal_setup(
        opt,
        dataset.model_path,
        scene.loaded_iter,
        scene.getTrainCameras(),
        gaussians,
        pipeline,
        background,
        None,
        select_obj_id,
        scene.cameras_extent,
        removal_thresh,
        outlier_factor=args.outlier_factor,
        target_tag=getattr(args, "target_tag", ""),
        mask_pixel_count=mask_pixel_count,
    )
    print("Number of gaussians: ", gaussians._xyz.shape[0])
    
    scene = Scene(dataset, gaussians, load_iteration=str(scene.loaded_iter)+'_object_removal', shuffle=False)
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)
    
    train_dir = os.path.join(args.model_path, '{}'.format(pipe.exp), 'train',  "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, '{}'.format(pipe.exp), 'test', "ours_{}".format(scene.loaded_iter))
    
    with torch.no_grad():
        if not skip_train:
            print("export removal training images ...")
            os.makedirs(train_dir, exist_ok=True)
            # gaussExtractor.gen_unseen_mask(scene.getTrainCameras(), unseen_thr=args.unseen_thresh, gaussians_before_remove=gaussians_before_remove, args=args)
            gaussExtractor.depth_aware_unseen_mask_generation(scene.getTrainCameras(), unseen_thr=args.unseen_thresh, gaussians_before_remove=gaussians_before_remove, args=args)
            gaussExtractor.export_image(train_dir)

        if not skip_test and (len(scene.getTestCameras()) > 0):
            print("export removal rendered testing images ...")
            os.makedirs(test_dir, exist_ok=True)
            gaussExtractor.reconstruction(scene.getTestCameras())
            gaussExtractor.export_image(test_dir)
            
        if args.render_path:
            print("render videos ...")
            traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
            os.makedirs(traj_dir, exist_ok=True)
            n_fames = 240
            cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
            gaussExtractor.reconstruction(cam_traj)
            gaussExtractor.export_image(traj_dir)
            create_videos(base_dir=traj_dir,
                        input_dir=traj_dir, 
                        out_name='render_traj', 
                        num_frames=n_fames)

        if not args.skip_mesh:
            print("export mesh ...")
            os.makedirs(train_dir, exist_ok=True)
            # set the active_sh to 0 to export only diffuse texture
            gaussExtractor.gaussians.active_sh_degree = 0
            gaussExtractor.reconstruction(scene.getTrainCameras())
            # extract the mesh and save
            if args.unbounded:
                name = 'fuse_unbounded.ply'
                mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
            else:
                name = 'fuse.ply'
                depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
                voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
                sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
                mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
            
            o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
            print("mesh saved at {}".format(os.path.join(train_dir, name)))
            # post-process the mesh and save, saving the largest N clusters
            mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
            o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
            print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))



if __name__ == "__main__":
    # Set up command line argument parser
    parser = configargparse.ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    # args for removal & unseen mask generation
    parser.add_argument("--removal_thresh", default=0.3, type=float, help='Removal: threshold for object removal')
    parser.add_argument("--outlier_factor", default=1.0, type=float, help='Removal: outlier factor for convex hull')
    parser.add_argument("--unseen_thresh", default=0.0, type=float, help='Removal: threshold for unseen mask')
    parser.add_argument("--debug_unseen", action="store_true")
    parser.add_argument("--removal_region", type=str, choices=['depth_diff', 'object_mask'], default='depth_diff')
    parser.add_argument("--aggreagte_threshold", default=0.6, type=float)
    parser.add_argument("--output_root", default="", type=str, help="Optional model/output root override for iterative workspaces")
    parser.add_argument("--target_tag", default="", type=str, help="Human-readable target id tag for diagnostics")
    args = get_combined_args(parser)
    if args.output_root:
        args.model_path = os.path.abspath(args.output_root)
    print("Removing: " + args.model_path)


    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    
    # remove the masked area
    removal(dataset, iteration, pipe, args.skip_train, args.skip_test, opt.extract(args), select_obj_id=0, removal_thresh=args.removal_thresh)
 
