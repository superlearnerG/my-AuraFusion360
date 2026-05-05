import torch
from diffusers import EulerDiscreteScheduler
from diffusers.schedulers import DDIMScheduler, DDPMScheduler

from utils.autoencoder_utils import AutoencoderKL
from utils.marigold_di_utils import AGDDv2
from utils.pretrained_paths import require_pretrained_dir


def dilate_mask(mask, iterations=1, kernel_size=3):
    # Dilate the mask
    dilated_mask = mask.clone()
    for _ in range(iterations):
        dilated_mask = torch.nn.functional.max_pool2d(dilated_mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    return dilated_mask[0]

def estimate_depth_marigold(rgb: torch.Tensor):
    """
    Args:
        rgb (torch.Tensor): (3, H, W)

    Returns:
        torch.Tensor: (1, H, W)
    """
    marigold_depth_path = require_pretrained_dir("marigold-depth-v1-0")
    pipe = AGDDv2.from_pretrained(str(marigold_depth_path), prediction_type="depth").to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    depth = pipe(rgb, is_latent_optimizing=False)[0]
    return depth

# ==============================
# AGDD: Adaptive Guided Depth Diffusion
# ==============================

def _optional_depth_min(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().item()
    value = float(value)
    return None if value < 0 else value


def _optional_depth_max(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().item()
    value = float(value)
    return None if value <= 0 else value


def _optional_percentile(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().float().cpu().item()
    value = float(value)
    return value if 0 < value < 100 else None


def _valid_depth_values(depth_tensor):
    depth_tensor = depth_tensor.float()
    valid_mask = (depth_tensor != 0) & torch.isfinite(depth_tensor)
    if not torch.any(valid_mask):
        raise ValueError("No finite non-zero depth values are available for depth alignment")
    return depth_tensor[valid_mask], valid_mask


def depth_stats_ignore_zeros(depth_tensor):
    valid_values, _ = _valid_depth_values(depth_tensor)
    quantiles = torch.quantile(
        valid_values,
        torch.tensor([0.5, 0.95, 0.99], device=valid_values.device, dtype=valid_values.dtype),
    )
    return {
        "count": int(valid_values.numel()),
        "min": float(torch.min(valid_values).detach().cpu().item()),
        "p50": float(quantiles[0].detach().cpu().item()),
        "p95": float(quantiles[1].detach().cpu().item()),
        "p99": float(quantiles[2].detach().cpu().item()),
        "max": float(torch.max(valid_values).detach().cpu().item()),
    }


def resolve_depth_range_ignore_zeros(depth_tensor, min_val=None, max_val=None, max_percentile=None):
    valid_values, _ = _valid_depth_values(depth_tensor)
    resolved_min = _optional_depth_min(min_val)
    if resolved_min is None:
        resolved_min = float(torch.min(valid_values).detach().cpu().item())

    resolved_max = _optional_depth_max(max_val)
    percentile = _optional_percentile(max_percentile)
    if resolved_max is None and percentile is not None:
        resolved_max = float(torch.quantile(valid_values, percentile / 100.0).detach().cpu().item())
    if resolved_max is None:
        resolved_max = float(torch.max(valid_values).detach().cpu().item())

    if resolved_max <= resolved_min:
        raise ValueError(
            f"Invalid depth alignment range: min={resolved_min}, max={resolved_max}. "
            "Adjust --depth_align_min_val/--depth_align_max_val or disable percentile clamping."
        )
    return resolved_min, resolved_max


def print_depth_alignment_stats(stats, min_val, max_val, range_source="manual_or_raw", label="depth_alignment"):
    print(
        f"[{label}] valid_depth count={stats['count']} "
        f"min={stats['min']:.6f} p50={stats['p50']:.6f} "
        f"p95={stats['p95']:.6f} p99={stats['p99']:.6f} max={stats['max']:.6f}"
    )
    print(
        f"[{label}] normalization_range min={min_val:.6f} max={max_val:.6f} "
        f"source={range_source}"
    )


def write_depth_alignment_stats(tb_writer, stats, min_val, max_val):
    if tb_writer is None:
        return
    for key, value in stats.items():
        tb_writer.add_scalar(f"depth_alignment/input_depth_{key}", value, global_step=0)
    tb_writer.add_scalar("depth_alignment/normalization_min", min_val, global_step=0)
    tb_writer.add_scalar("depth_alignment/normalization_max", max_val, global_step=0)


def normalize_depth_ignore_zeros(depth_tensor, min_val=None, max_val=None):
    """
    將 depth tensor 正規化到 0~1 範圍，忽略值為 0 的區域
    
    參數:
    depth_tensor (torch.Tensor): 輸入的 depth tensor
    min_val (float): 自定義的最小值，預設為 None (使用非零區域的最小值)
    max_val (float): 自定義的最大值，預設為 None (使用非零區域的最大值)
    
    回傳:
    torch.Tensor: 正規化後的 tensor，原本為 0 的區域保持為 0
    """
    normalized, _, _ = normalize_depth_ignore_zeros_with_range(
        depth_tensor,
        min_val=min_val,
        max_val=max_val,
    )
    return normalized


def normalize_depth_ignore_zeros_with_range(depth_tensor, min_val=None, max_val=None, max_percentile=None):
    # 將輸入轉換為 float 類型
    depth_tensor = depth_tensor.float()
    
    # 創建有限且非零區域的 mask
    valid_values, valid_mask = _valid_depth_values(depth_tensor)
    min_val, max_val = resolve_depth_range_ignore_zeros(
        depth_tensor,
        min_val=min_val,
        max_val=max_val,
        max_percentile=max_percentile,
    )
    
    # 避免除以零
    if max_val == min_val:
        return torch.zeros_like(depth_tensor), min_val, max_val
    
    # 創建輸出 tensor
    normalized = torch.zeros_like(depth_tensor)
    
    # 只正規化非零區域
    clamped_depth = torch.clamp(valid_values, min=min_val, max=max_val)
    normalized[valid_mask] = (clamped_depth - min_val) / (max_val - min_val)
    
    return normalized, min_val, max_val


def unnormalize_depth_ignore_zeros(depth_tensor, ref_depth_tensor=None, min_val=None, max_val=None, max_percentile=None):
    if min_val is None or max_val is None:
        if ref_depth_tensor is None:
            raise ValueError("ref_depth_tensor is required when min_val/max_val are not provided")
        min_val, max_val = resolve_depth_range_ignore_zeros(
            ref_depth_tensor,
            min_val=min_val,
            max_val=max_val,
            max_percentile=max_percentile,
        )
    # unnormalize the align_depth from original gt depth, ignore the zero values
    unnormalized = torch.clamp(depth_tensor.float(), 0.0, 1.0) * (max_val - min_val) + min_val
    return unnormalized


def align_depth_agdd_v2(depth, rgb, mask, opt, seed=7777, tb_writer=None):
    """_summary_

    Args:
        depth (_type_): _description_
        rgb (_type_): _description_
        mask (_type_): _description_
        opt (_type_): _description_
        seed (int, optional): _description_. Defaults to 7777.

    Returns:
        align_depth: (1, H, W)
    """
    
    # IMPORTANT NOTE: bfloat16 tends to predict unsmooth align depth, so we use float16 instead
    marigold_path = require_pretrained_dir("marigold-v1-0")
    pipe = AGDDv2.from_pretrained(
        str(marigold_path), variant="fp16", torch_dtype=torch.float16
    ).to("cuda")
    vae = AutoencoderKL.from_pretrained(str(marigold_path), subfolder="vae").to(dtype=torch.float16).to("cuda")
    pipe.register_modules(vae=vae)
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    depth = depth.clone()
    depth[mask[None] == 1] = 0
    max_percentile = getattr(opt, "depth_align_percentile", 99.5)
    depth_min = getattr(opt, "depth_align_min_val", -1.0)
    depth_max = getattr(opt, "depth_align_max_val", 0.0)
    percentile = _optional_percentile(max_percentile)
    range_source = (
        "manual_max"
        if _optional_depth_max(depth_max) is not None
        else f"p{percentile:g}"
        if percentile is not None
        else "raw_max"
    )
    depth_stats = depth_stats_ignore_zeros(depth)
    gt_depth, norm_min, norm_max = normalize_depth_ignore_zeros_with_range(
        depth,
        min_val=depth_min,
        max_val=depth_max,
        max_percentile=max_percentile,
    )
    print_depth_alignment_stats(depth_stats, norm_min, norm_max, range_source=range_source, label="AGDD")
    write_depth_alignment_stats(tb_writer, depth_stats, norm_min, norm_max)

    rgb = rgb.to(torch.float16)
    gt_depth = gt_depth.to(torch.float16)
    mask = (mask == 1).to(torch.float16)[None]
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    inpaint_depth = pipe(image=rgb, incomplete_depth=gt_depth, unseen_mask=mask, num_inference_steps=50, generator=generator, is_latent_optimizing=True, opt=opt, tb_writer=tb_writer)
    align_depth = inpaint_depth.to(torch.float32)
    
    # unnormalize the align_depth from original gt depth, ignore the zero values
    unnormalize_align_depth = unnormalize_depth_ignore_zeros(align_depth, min_val=norm_min, max_val=norm_max)
    
    # get the error of align_depth and gt_depth
    print(f"""
        \033[93m#########final alignment error#########\033[0m
        {torch.abs(unnormalize_align_depth[mask == 0] - depth[mask == 0]).mean().item()}
        \033[93m#########final alignment error#########\033[0m
    """)
    
    return unnormalize_align_depth


# ==============================
# Wonder World Guided Depth Diffusion
# https://github.com/KovenYu/WonderWorld.git
# ==============================

def align_depth_marigold_ww(depth, rgb, mask, opt, seed=7777):
    """Wonder World Guided Depth Diffusion

    Args:
        depth (torch.Tensor): (1, H, W)
        rgb (torch.Tensor): (3, H, W)
        mask (torch.Tensor): (1, H, W)
        opt (OptimizationParams): arguments for optimization
        seed (int, optional): random seed. Defaults to 7777.
        
    Returns:
        torch.Tensor: (1, H, W)
    """
    from utils.marigold_ww_utils import MarigoldPipeline

    marigold_path = require_pretrained_dir("marigold-v1-0")
    pipe = MarigoldPipeline.from_pretrained(
        str(marigold_path), variant="fp16", torch_dtype=torch.bfloat16 
    ).to("cuda")
    # vae = AutoencoderKL.from_pretrained(str(marigold_path), subfolder="vae").to(dtype=torch.float16).to("cuda")
    # pipe.register_modules(vae=vae)
    # pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    
    depth = depth.clone()
    depth[mask[None] == 1] = 0
    max_percentile = getattr(opt, "depth_align_percentile", 99.5)
    depth_min = getattr(opt, "depth_align_min_val", -1.0)
    depth_max = getattr(opt, "depth_align_max_val", 0.0)
    gt_depth, norm_min, norm_max = normalize_depth_ignore_zeros_with_range(
        depth,
        min_val=depth_min,
        max_val=depth_max,
        max_percentile=max_percentile,
    )

    rgb = rgb.to(torch.bfloat16)
    gt_depth = gt_depth.to(torch.bfloat16)
    mask_align = (mask == 0)
    
    align_depth = pipe(
        rgb,
        denoising_steps=30,     # optional
        ensemble_size=1,       # optional
        processing_res=0,     # optional
        match_input_res=True,   # optional
        batch_size=0,           # optional
        color_map=None,   # optional
        show_progress_bar=True, # optional
        depth_conditioning=True,
        target_depth=gt_depth,
        mask_align=mask_align,
        mask_farther=None,
        guidance_steps=8,
        # guidance_steps=20,
        logger=None,
    )[None].to(torch.float32)
    
    unnormalize_align_depth = unnormalize_depth_ignore_zeros(align_depth, min_val=norm_min, max_val=norm_max)
    return unnormalize_align_depth
