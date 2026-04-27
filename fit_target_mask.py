from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from random import randint
from pathlib import Path

import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render_mask
from scene import Scene
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss


def freeze_scene_except_is_masked(gaussians: GaussianModel) -> None:
    for attr_name in ["_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"]:
        tensor = getattr(gaussians, attr_name, None)
        if tensor is not None:
            tensor.requires_grad_(False)
    gaussians._is_masked.requires_grad_(True)


def collect_mask_pixel_counts(views) -> dict[str, int]:
    counts = {}
    for view in views:
        if view.original_image_mask is None:
            counts[view.image_name] = 0
        else:
            counts[view.image_name] = int((view.original_image_mask > 0.5).sum().item())
    return counts


def fit_target_mask(dataset, opt, pipe, iteration: int, fit_iterations: int, removal_thresh: float, stats_path: str | None) -> None:
    dataset.stage = "train"
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    freeze_scene_except_is_masked(gaussians)
    optimizer = torch.optim.Adam([gaussians._is_masked], lr=opt.is_masked_lr, eps=1e-15)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    views = scene.getTrainCameras()
    if not views:
        raise RuntimeError("No train cameras found for target-mask fitting")

    mask_pixel_counts = collect_mask_pixel_counts(views)
    if sum(mask_pixel_counts.values()) == 0:
        raise RuntimeError("All target masks are empty; aborting target-mask fitting")

    progress_bar = tqdm(range(fit_iterations), desc="Fitting target mask")
    viewpoint_stack = None
    ema_loss = 0.0
    for fit_iter in range(1, fit_iterations + 1):
        if not viewpoint_stack:
            viewpoint_stack = views.copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if viewpoint_cam.original_image_mask is None:
            continue
        target_mask = viewpoint_cam.original_image_mask.cuda().float()
        render_pkg = render_mask(viewpoint_cam, gaussians, pipe, background)
        pred_mask = render_pkg["is_masked"][0]
        loss = l1_loss(pred_mask, target_mask)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
        if fit_iter % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss:.6f}"})
            progress_bar.update(10)
    progress_bar.close()

    with torch.no_grad():
        prob_obj3d = gaussians.get_is_masked[..., 0]
        max_prob = float(prob_obj3d.max().item()) if prob_obj3d.numel() else 0.0
        mean_prob = float(prob_obj3d.mean().item()) if prob_obj3d.numel() else 0.0
        selected_gaussians = int((prob_obj3d > removal_thresh).sum().item())
        stats = {
            "model_path": dataset.model_path,
            "iteration": iteration,
            "fit_iterations": fit_iterations,
            "removal_thresh": removal_thresh,
            "mask_pixel_counts": mask_pixel_counts,
            "is_masked_max": max_prob,
            "is_masked_mean": mean_prob,
            "selected_gaussians": selected_gaussians,
        }
        if stats_path:
            stats_path = Path(stats_path).expanduser().resolve()
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with stats_path.open("w", encoding="utf-8") as handle:
                json.dump(stats, handle, indent=2)
        if selected_gaussians == 0:
            raise RuntimeError(
                "Target mask fitting selected zero Gaussians "
                f"(threshold={removal_thresh}, max_prob={max_prob:.6f}, mask_pixels={sum(mask_pixel_counts.values())})"
            )

    point_cloud_path = os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    gaussians.save_ply(point_cloud_path)
    print(f"Saved fitted target mask attributes to {point_cloud_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Fit AuraFusion360 binary _is_masked attributes for one iterative target.")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=0, type=int)
    parser.add_argument("--fit_mask_iterations", default=2000, type=int)
    parser.add_argument("--removal_thresh", default=0.6, type=float)
    parser.add_argument("--stats_path", default="", type=str)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)
    dataset, pipe, opt_params = model.extract(args), pipeline.extract(args), opt.extract(args)
    fit_target_mask(dataset, opt_params, pipe, args.iteration, args.fit_mask_iterations, args.removal_thresh, args.stats_path or None)
