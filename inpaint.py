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

import glob
import os
from datetime import datetime
from os import makedirs
from pathlib import Path
from random import randint

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parent.parent / "pretrained_models" / "torch"))

import configargparse
import cv2
import lpips
import numpy as np
import open3d as o3d
import torch
import torchvision
import torchvision.transforms.functional as tf
from diffusers import DDIMScheduler
from natsort import natsorted
from PIL import Image
from pytorch_fid import fid_score
from scipy.spatial import ConvexHull, Delaunay
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.regression import pearson_corrcoef
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.depth_utils import estimate_depth_marigold
from utils.general_utils import colormap
from utils.image_utils import masked_psnr, psnr
from utils.loss_utils import l1_loss, ssim
from utils.mesh_utils import GaussianExtractor, post_process_mesh
from utils.render_utils import create_videos, generate_path
from utils.warping_utils import unproject, voxel_downsample


def evaluation(scene_name, finetune_iteration, pipeline):
    spatial_lpips = lpips.LPIPS(net='vgg', spatial=True).cuda()
    original_lpips = lpips.LPIPS(net='vgg').cuda()
    
    gt_imgs = natsorted(glob.glob(f"data/360-USID/{scene_name}/test_images/*jpg"))
    if len(gt_imgs) == 0: 
        print("No test images found, skip evaluation")
        return
    
    if finetune_iteration == -1:
        render_type = "ours_object_inpaint_init"
    else:
        render_type = f"ours_{finetune_iteration}_object_inpaint"
    ours_imgs = natsorted(glob.glob(os.path.join(dataset.model_path, '{}'.format(pipeline.exp), f"test/{render_type}/renders/*png")))
    object_masks = natsorted(glob.glob(os.path.join(dataset.model_path, "test/ours_30000/object_mask/*png")))
    assert len(gt_imgs) == len(ours_imgs) == len(object_masks)
    
    ssims = []
    psnrs = []
    lpipss = []
    ssims_object = []
    psnrs_object = []
    lpipss_object = []
    for i in range(len(gt_imgs)):
        # prepocess gt, img, object_mask
        gt_img = Image.open(gt_imgs[i]).convert("RGB")
        img_name = img_name = os.path.basename(gt_imgs[i])
        object_mask = Image.open(object_masks[i])
        gt_tensor = tf.to_tensor(gt_img).unsqueeze(0)[:, :3, :, :].cuda()
        object_mask_tensor = tf.to_tensor(object_mask).unsqueeze(0)[:, :3, :, :].cuda()
        object_mask_tensor[object_mask_tensor > 0.5] = 1
        img = Image.open(ours_imgs[i]).convert("RGB")
        img_tensor = tf.to_tensor(img).unsqueeze(0)[:, :3, :, :].cuda()
        
        ###### Eval w/ object_mask
        # psnr w/ object_mask
        psnrs_object.append(masked_psnr(img_tensor, gt_tensor, object_mask_tensor).item())
        # lpips w/ object_mask
        lpips_map = spatial_lpips(img_tensor, gt_tensor)
        lpips_score = torch.sum(lpips_map * object_mask_tensor) / torch.sum(object_mask_tensor)
        lpipss_object.append(lpips_score.item())
        # ssim w/ object_mask
        img_tensor_object = img_tensor * object_mask_tensor.repeat(1, 3, 1, 1)
        gt_tensor_object = gt_tensor * object_mask_tensor.repeat(1, 3, 1, 1)
        ssims_object.append(ssim(img_tensor_object, gt_tensor_object).item())
            
        ##### Eval w/o object_mask
        ssims.append(ssim(img_tensor, gt_tensor).item())
        psnrs.append(psnr(img_tensor, gt_tensor).item())
        lpipss.append(original_lpips(img_tensor, gt_tensor).item())
        
    fid = fid_score.calculate_fid_given_paths([os.path.dirname(gt_imgs[0]), os.path.dirname(ours_imgs[0])], 50, 'cuda', 2048, 0)
    print(f"PSNR: {np.mean(psnrs):.4f}, SSIM: {np.mean(ssims):.4f}, LPIPS: {np.mean(lpipss):.4f}, FID: {fid:.4f}")
    print(f"PSNR OBJECT: {np.mean(psnrs_object):.4f}, SSIM OBJECT: {np.mean(ssims_object):.4f}, LPIPS OBJECT: {np.mean(lpipss_object):.4f}")

    
    print(f"{np.mean(psnrs):.3f}, {np.mean(ssims):.3f}, {np.mean(lpipss):.3f}, {fid:.3f}")
    print(f"{np.mean(psnrs_object):.3f}, {np.mean(ssims_object):.3f}, {np.mean(lpipss_object):.3f}")
    print("============================================================================================================")
    

def mask_to_bbox(mask):
        # Find the rows and columns where the mask is non-zero
        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        xmin, xmax = torch.where(cols)[0][[0, -1]]
    
        return xmin, ymin, xmax, ymax

def crop_using_bbox(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    return image[:, ymin:ymax+1, xmin:xmax+1]

def divide_into_patches(image, K):
    B, C, H, W = image.shape
    patch_h, patch_w = H // K, W // K
    patches = torch.nn.functional.unfold(image, (patch_h, patch_w), stride=(patch_h, patch_w))
    patches = patches.view(B, C, patch_h, patch_w, -1)
    return patches.permute(0, 4, 1, 2, 3)

def get_intrinsics2(H, W, fovx, fovy):
    fx = 0.5 * W / np.tan(0.5 * fovx)
    fy = 0.5 * H / np.tan(0.5 * fovy)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[fx,  0, cx],
                     [ 0, fy, cy],
                     [ 0,  0,  1]])

def inpaint_init(scene, opt, model_path, iteration, views, gaussians, pipeline, background, classifier, selected_obj_ids, cameras_extent, voxel_down_size, finetune_iteration, reference_index=None):
    
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_writer = SummaryWriter(os.path.join(model_path, "tensorboard", "inpaint_init", run_name))
    

    with torch.no_grad():
        if reference_index is None:
            raise ValueError("Shuold provide a reference index")
        reference_view = views[reference_index]
        
        print("reference view: ", reference_view.image_name)
        render_pkg = render(reference_view, gaussians, pipeline, background)
        rgb, depth = render_pkg["render"], render_pkg["surf_depth"]
        inpaint_rgb = reference_view.original_image
        unseen_mask = torch.tensor(reference_view.original_image_mask).cuda()
        
        masked_rgb = rgb.clone()
        masked_rgb[0, unseen_mask > 0.5] = 1.0
        
        
        estimated_depth = estimate_depth_marigold(inpaint_rgb)
    depth_ref = None
    if opt.dc_method == 'agddv2':
        from utils.depth_utils import align_depth_agdd_v2
        depth_ref = align_depth_agdd_v2(depth, inpaint_rgb, unseen_mask, opt, tb_writer=tb_writer)[0]
    else:
        raise ValueError("Invalid depth alignment method")
    
    
    with torch.no_grad():
        reference_view_aligned_depth = depth_ref * unseen_mask + depth[0] * (1 - unseen_mask)
        # reference_view_aligned_depth = depth_ref
        
        initial_gs_xyz, initial_gs_rgb = unproject(reference_view, reference_view_aligned_depth, inpaint_rgb, device="cuda", mask=unseen_mask)
        if voxel_down_size > 0:
            initial_gs_xyz, initial_gs_rgb = voxel_downsample(initial_gs_xyz, initial_gs_rgb, voxel_down_size)
        
    
    # fix some gaussians
    # gaussians.inpaint_setup(opt) # random init gaussians for inpainting
    gaussians.inpaint_setup(opt, initial_gs_xyz, initial_gs_rgb)
    

    
    # save composed gaussians
    with torch.no_grad():        
        # apply the unseen mask region to the rgb with color 'red'
    
        tb_writer.add_image('depth_alignment/00_rgb', rgb, global_step=0)
        tb_writer.add_image('depth_alignment/00_masked_rgb', masked_rgb, global_step=0)
        tb_writer.add_images('depth_alignment/01_inpaint_rgb', inpaint_rgb[None], global_step=0)
        
        depth_map = colormap((depth / depth.max()).cpu().numpy()[0], cmap='turbo')
        tb_writer.add_images('depth_alignment/02_depth', depth_map[None], global_step=0)
        
        disparity_map = colormap(1 / (depth / depth.max()).cpu().numpy()[0], cmap='turbo')
        tb_writer.add_images('depth_alignment/03_disparity', disparity_map[None], global_step=0)
        
        
        
        estiamted_disparity_map = colormap((estimated_depth).cpu().numpy(), cmap='turbo')
        tb_writer.add_images('depth_alignment/04_estimated_disparity', estiamted_disparity_map[None], global_step=0)
        
        depth_ref_map = colormap((depth_ref / depth_ref.max()).cpu().numpy(), cmap='turbo')
        tb_writer.add_images('depth_alignment/05_depth_ref', depth_ref_map[None], global_step=0)
    
        # estimated_real_depth = 1 / (estimated_depth + 1e-6)    
        # estiamted_depth_map = colormap((estimated_real_depth).cpu().numpy(), cmap='turbo')
        # tb_writer.add_images('depth_alignment/estimated_depth', estiamted_depth_map[None], global_step=0)
        
        aligned_depth_map = colormap(1 / (reference_view_aligned_depth / reference_view_aligned_depth.max()).cpu().numpy(), cmap='turbo')
        tb_writer.add_images('depth_alignment/06_aligned_disparity', aligned_depth_map[None], global_step=0)
        
        aligned_depth_map = colormap((reference_view_aligned_depth / reference_view_aligned_depth.max()).cpu().numpy(), cmap='turbo')
        tb_writer.add_images('depth_alignment/07_aligned_depth', aligned_depth_map[None], global_step=0)
        
        # render the initial pts
        render_pkg = render(reference_view, gaussians, pipe, background)
        rgb, depth = render_pkg["render"], render_pkg["surf_depth"]
        tb_writer.add_images('depth_alignment/08_initial_render_rgb', rgb[None], global_step=0)
        depth_map = colormap((depth / depth.max()).cpu().numpy()[0], cmap='turbo')
        tb_writer.add_images('depth_alignment/09_initial_render_depth', depth_map[None], global_step=0)

        if len(scene.getTestCameras()) > 0:
            test_cameras =  [scene.getTestCameras()[idx * len(scene.getTestCameras()) // 8] for idx in range(8)]
            for idx, viewpoint in enumerate(test_cameras):
                render_pkg = render(viewpoint, gaussians, pipe, background)
                image = render_pkg["render"]
                tb_writer.add_image('depth_alignment/{}_render'.format(viewpoint.image_name), image, global_step=0)
                depth = render_pkg["surf_depth"]    
                disparity_map = colormap(1 / (depth / depth.max()).cpu().numpy()[0], cmap='turbo')
                tb_writer.add_images('depth_alignment/{}_disparity'.format(viewpoint.image_name), disparity_map[None], global_step=0)
                
        elif len(scene.getTrainCameras()) > 0:
            train_cameras = [scene.getTrainCameras()[idx * len(scene.getTrainCameras()) // 8] for idx in range(8)]          
            for idx, viewpoint in enumerate(train_cameras):
                render_pkg = render(viewpoint, gaussians, pipe, background)
                image = render_pkg["render"]
                tb_writer.add_image('depth_alignment/{}_render'.format(viewpoint.image_name), image, global_step=0)
                depth = render_pkg["surf_depth"]    
                disparity_map = colormap(1 / (depth / depth.max()).cpu().numpy()[0], cmap='turbo')
                tb_writer.add_images('depth_alignment/{}_disparity'.format(viewpoint.image_name), disparity_map[None], global_step=0)

    if finetune_iteration == -1: # render the initial gs only
        print("Unproject Only")
        point_cloud_path = os.path.join(model_path, '{}'.format(pipeline.exp), "point_cloud/iteration_object_inpaint_init")
        gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        return gaussians, True


def inpaint_finetune(scene, opt, model_path, iteration, views, gaussians_removal, pipeline, background, classifier, selected_obj_ids, cameras_extent, voxel_down_size, finetune_iteration, reference_index=None):
    
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tb_writer = SummaryWriter(os.path.join(model_path, "tensorboard", "inpaint_finetune", run_name))
    
    # fix the gradient of removal gaussians
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration="object_inpaint_init", shuffle=False)
    num_fixed_gaussians = gaussians_removal.get_xyz.shape[0]
    gaussians.stop_grad(opt, num_fixed_gaussians)
    # clean up the gaussians_removal from cuda memory
    del gaussians_removal
    torch.cuda.empty_cache()
    
    
    # finetune the initial gaussians only
    iterations = finetune_iteration
    progress_bar = tqdm(range(iterations), desc="Finetuning progress")
    LPIPS = lpips.LPIPS(net='vgg')
    for param in LPIPS.parameters():
        param.requires_grad = False
    LPIPS.cuda()
    

    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    
    
    view_selection_cnt = 0
    viewpoint_stack = None
    views_copy = None
    for iteration in range(1, iterations + 1):
 
        if not viewpoint_stack:
            viewpoint_stack = views.copy()       
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))        
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()

        
        # finetune only masked region: 1. LPIPS loss
        loss = torch.tensor(0.0).cuda()
        
        
        # Get original mask and dilate it using torch max_pool2d
        image_mask_tensor = viewpoint_cam.original_image_mask.unsqueeze(0)
        
        image_m = image * image_mask_tensor
        gt_image_m = gt_image * image_mask_tensor
        Ll1 = torch.tensor(0.0).cuda()
        loss += (1.0 - opt.lambda_dssim) * l1_loss(image_m, gt_image_m) + opt.lambda_dssim * (1 - ssim(image_m, gt_image_m))
        
        try: # To catch the error when the mask is all zeros
            bbox = mask_to_bbox(image_mask_tensor[0])
            cropped_image = crop_using_bbox(image, bbox)
            cropped_gt_image = crop_using_bbox(gt_image, bbox)
            K = 2
            rendering_patches = divide_into_patches(cropped_image[None, ...], K)
            gt_patches = divide_into_patches(cropped_gt_image[None, ...], K)
            lpips_loss = LPIPS((rendering_patches.squeeze()*2-1), (gt_patches.squeeze()*2-1)).mean()
            loss += opt.lambda_lpips * lpips_loss
        except Exception as e:
            print(f"Error in LPIPS loss: {e}")
            pass
        
        # unmasked region
        image_um = image * (1 - image_mask_tensor)
        gt_image_um = gt_image * (1 - image_mask_tensor)
        loss += 0.8 * l1_loss(image_um, gt_image_um) + 0.2 * (1 - ssim(image_um, gt_image_um))
        
        
        
        # fintune only masked region: 2. normal/dist regularization
        lambda_normal = opt.lambda_normal if iteration > 700 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 300 else 0.0
        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal'] * image_mask_tensor
        surf_normal = render_pkg['surf_normal'] * image_mask_tensor
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()
        
        total_loss = loss + dist_loss + normal_loss

        total_loss.backward()
        
        
        
        
        with torch.no_grad():
            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % 300 == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)

                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()
                    
            # Optimizer step
            if iteration < iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
            # if (iteration == iterations):
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + "_inpaint.pth")
            
            # Logging to tensorboard
            if iteration % 1000 == 0:
                tb_writer.add_scalar('inpaint/total_loss', total_loss, global_step=iteration)
                if len(scene.getTestCameras()) > 0:
                    # 取test cameras中8個viewpoint
                    test_cameras = [scene.getTestCameras()[idx * len(scene.getTestCameras()) // 8] for idx in range(8)]
                    
                    for idx, viewpoint in enumerate(test_cameras):
                        render_pkg = render(viewpoint, gaussians, pipe, background)
                        image = render_pkg["render"]
                        tb_writer.add_image('inpaint/{}_render'.format(viewpoint.image_name), image, global_step=iteration)
                        depth = render_pkg["surf_depth"]    
                        depth_map = colormap((depth / depth.max()).cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images('inpaint/{}_depth'.format(viewpoint.image_name), depth_map[None], global_step=iteration)     
                elif len(scene.getTrainCameras()) > 0:
                    train_cameras = [scene.getTrainCameras()[idx * len(scene.getTrainCameras()) // 8] for idx in range(8)]          
                    for idx, viewpoint in enumerate(train_cameras):
                        render_pkg = render(viewpoint, gaussians, pipe, background)
                        image = render_pkg["render"]
                        tb_writer.add_image('inpaint/{}_render'.format(viewpoint.image_name), image, global_step=iteration)
                        depth = render_pkg["surf_depth"]    
                        depth_map = colormap((depth / depth.max()).cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images('inpaint/{}_depth'.format(viewpoint.image_name), depth_map[None], global_step=iteration)     


        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss:.{7}f}"})
            progress_bar.update(10)
    progress_bar.close()
    
    # save gaussians
    point_cloud_path = os.path.join(model_path, '{}'.format(pipeline.exp), "point_cloud/iteration_{}_object_inpaint".format(iteration))
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    return gaussians, False
        
    

def inpaint(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, opt : OptimizationParams, select_obj_id : int, voxel_down_size : float,  finetune_iteration: int, reference_index: int):
    # 1. load gaussian checkpoint
    dataset.stage = 'inpaint'
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 2. unproject or finetuning
    is_initial_gs = None
    if finetune_iteration == -1:
        gaussians, is_initial_gs = inpaint_init(scene, opt, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, None, select_obj_id, scene.cameras_extent, voxel_down_size, finetune_iteration, reference_index)
    else:
        gaussians, is_initial_gs = inpaint_finetune(scene, opt, dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, None, select_obj_id, scene.cameras_extent, voxel_down_size, finetune_iteration, reference_index)
    
    
    # 3. render new result
    if is_initial_gs:
        scene = Scene(dataset, gaussians, load_iteration='object_inpaint_init', shuffle=False, exp_name=pipeline.exp)
    else:
        scene = Scene(dataset, gaussians, load_iteration=str(finetune_iteration)+'_object_inpaint', shuffle=False, exp_name=pipeline.exp)
    

    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)    
    
    train_dir = os.path.join(args.model_path, '{}'.format(pipeline.exp), 'train',  "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, '{}'.format(pipeline.exp), 'test', "ours_{}".format(scene.loaded_iter))
    
    with torch.no_grad():
        if not skip_train:
            print("export removal training images ...")
            os.makedirs(train_dir, exist_ok=True)
            gaussExtractor.reconstruction(scene.getTrainCameras())
            gaussExtractor.export_image(train_dir)
             
            # render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier)

        if not skip_test and (len(scene.getTestCameras()) > 0):
            print("export removal rendered testing images ...")
            os.makedirs(test_dir, exist_ok=True)
            gaussExtractor.reconstruction(scene.getTestCameras())
            gaussExtractor.export_image(test_dir)
            
        if args.render_path:
            print("render videos ...")
            traj_dir = os.path.join(args.model_path, '{}'.format(pipeline.exp), 'traj', "ours_{}".format(scene.loaded_iter))
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
    # parser = ArgumentParser(description="Testing script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
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
    parser.add_argument("--voxel_down_size", default=-1, type=float, help='voxel size for voxel downsampling')
    parser.add_argument("--finetune_iteration", default=-1, type=int, help='Inpaint: number of finetune iterations')
    parser.add_argument("--reference_index", default=0, type=int, help='Inpaint: reference view index. AURA daataset: -1, 360 dataset: 0')
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation after inpainting")
    parser.add_argument("--output_root", default="", type=str, help="Optional model/output root override for iterative workspaces")
    args = get_combined_args(parser)
    if args.output_root:
        args.model_path = os.path.abspath(args.output_root)
    print("Rendering " + args.model_path)

    dataset, iteration, pipe, opt = lp.extract(args), args.iteration, pp.extract(args), op.extract(args)
    
    reference_index = args.reference_index
    if "our_dataset" in dataset.source_path or "360-USID" in dataset.source_path:
        print("Warning: reference index must be the last (-1) image in the training set for 360-USID dataset")
        reference_index = -1
    elif "spinnerf_dataset_processed" in dataset.source_path:
        reference_index = 29
    
    iteration = str(iteration) + "_object_removal"
    # inpaint
    inpaint(dataset, iteration, pipe, args.skip_train, args.skip_test, opt, 0, args.voxel_down_size, args.finetune_iteration, reference_index)
        
    # Evaluate the inpainted result
    if not args.skip_eval:
        evaluation(scene_name=os.path.basename(dataset.source_path), finetune_iteration=args.finetune_iteration, pipeline=pipe)
    
    
