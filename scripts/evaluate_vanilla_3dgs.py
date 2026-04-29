import argparse
import contextlib
import os
import ssl
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
os.environ.setdefault("TORCH_HOME", str(REPO_ROOT.parent / "pretrained_models" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import lpips
import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from pytorch_fid import fid_score
from pytorch_fid.inception import FID_WEIGHTS_URL

from utils.image_utils import psnr
from utils.loss_utils import ssim

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FID_LOAD_EXCEPTIONS = (URLError, ssl.SSLError, TimeoutError, ConnectionError)


def list_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        [path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name,
    )


def paired_images(gt_dir, render_dir):
    gt_images = {path.stem: path for path in list_images(gt_dir)}
    render_images = {path.stem: path for path in list_images(render_dir)}
    common_stems = sorted(gt_images.keys() & render_images.keys())
    if not common_stems:
        raise RuntimeError(f"No paired images found under {gt_dir} and {render_dir}")
    if len(common_stems) != len(gt_images) or len(common_stems) != len(render_images):
        missing_render = sorted(gt_images.keys() - render_images.keys())
        missing_gt = sorted(render_images.keys() - gt_images.keys())
        details = []
        if missing_render:
            details.append(f"missing renders for {missing_render[:5]}")
        if missing_gt:
            details.append(f"missing gt for {missing_gt[:5]}")
        raise RuntimeError("Test render/gt image mismatch: " + "; ".join(details))
    return [(gt_images[stem], render_images[stem]) for stem in common_stems]


def load_rgb_tensor(path, device):
    image = Image.open(path).convert("RGB")
    return tf.to_tensor(image).unsqueeze(0).to(device)


def configure_fid_cache():
    weights_name = Path(urlparse(FID_WEIGHTS_URL).path).name
    active_weights = Path(torch.hub.get_dir()) / "checkpoints" / weights_name
    if active_weights.exists():
        return

    default_weights = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / weights_name
    if default_weights.exists():
        torch.hub.set_dir(str(default_weights.parent.parent))


def compute_metrics(gt_dir, render_dir, device, batch_size, fid_dims, fid_workers):
    pairs = paired_images(gt_dir, render_dir)
    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()

    psnr_values = []
    ssim_values = []
    lpips_values = []

    with torch.no_grad():
        for gt_path, render_path in pairs:
            gt_tensor = load_rgb_tensor(gt_path, device)
            render_tensor = load_rgb_tensor(render_path, device)
            if gt_tensor.shape != render_tensor.shape:
                raise RuntimeError(f"Image shape mismatch: {gt_path} {gt_tensor.shape} vs {render_path} {render_tensor.shape}")
            psnr_values.append(psnr(render_tensor, gt_tensor).mean().item())
            ssim_values.append(ssim(render_tensor, gt_tensor).item())
            lpips_values.append(lpips_model(render_tensor, gt_tensor).item())

    fid = None
    fid_error = None
    try:
        configure_fid_cache()
        with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
            fid = fid_score.calculate_fid_given_paths(
                [str(gt_dir), str(render_dir)],
                batch_size,
                str(device),
                fid_dims,
                fid_workers,
            )
    except FID_LOAD_EXCEPTIONS as exc:
        fid_error = f"{type(exc).__name__}: {exc}"
        print(f"Skipping FID: unable to load/download InceptionV3 weights ({fid_error})", file=sys.stderr)

    return {
        "num_images": len(pairs),
        "PSNR": float(np.mean(psnr_values)),
        "SSIM": float(np.mean(ssim_values)),
        "LPIPS": float(np.mean(lpips_values)),
        "FID": None if fid is None else float(fid),
        "FID_error": fid_error,
    }


def write_results(output_path, model_path, iteration, gt_dir, render_dir, metrics):
    lines = [
        "Vanilla 3DGS test-set evaluation",
        f"model_path: {model_path}",
        f"iteration: {iteration}",
        f"gt_dir: {gt_dir}",
        f"render_dir: {render_dir}",
        f"num_images: {metrics['num_images']}",
        f"PSNR: {metrics['PSNR']:.6f}",
        f"SSIM: {metrics['SSIM']:.6f}",
        f"LPIPS: {metrics['LPIPS']:.6f}",
    ]
    if metrics["FID"] is None:
        lines.append("FID: skipped")
        if metrics.get("FID_error"):
            lines.append(f"FID_error: {metrics['FID_error']}")
    else:
        lines.append(f"FID: {metrics['FID']:.6f}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate vanilla 3DGS test renders.")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", default="30000")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--fid_dims", type=int, default=2048)
    parser.add_argument("--fid_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    iteration = str(args.iteration)
    test_root = model_path / "test" / f"ours_{iteration}"
    gt_dir = test_root / "gt"
    render_dir = test_root / "renders"
    output_path = Path(args.output) if args.output else model_path / "evaluation_results.txt"

    device = torch.device(args.device)
    metrics = compute_metrics(gt_dir, render_dir, device, args.batch_size, args.fid_dims, args.fid_workers)
    write_results(output_path, model_path, iteration, gt_dir, render_dir, metrics)


if __name__ == "__main__":
    main()
