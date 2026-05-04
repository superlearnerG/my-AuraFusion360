from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import ssl
import sys
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SHARED_TORCH_HOME = REPO_ROOT.parent / "pretrained_models" / "torch"
SHARED_TORCH_HUB = SHARED_TORCH_HOME / "hub"
os.environ["TORCH_HOME"] = str(SHARED_TORCH_HOME)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import lpips
import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from utils.image_utils import psnr
from utils.loss_utils import ssim


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FID_LOAD_EXCEPTIONS = (ImportError, URLError, ssl.SSLError, TimeoutError, ConnectionError)
METHOD_DIR_NAME = "aurafusion360"


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        return []
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name,
    )


def resolve_model_path(model_path: str | Path) -> Path:
    model_path = Path(model_path).expanduser().resolve()
    direct_rounds = model_path / "aura_iterative" / "rounds"
    if direct_rounds.is_dir() or model_path.name == METHOD_DIR_NAME:
        return model_path

    nested_model_path = model_path / METHOD_DIR_NAME
    nested_rounds = nested_model_path / "aura_iterative" / "rounds"
    if nested_rounds.is_dir():
        return nested_model_path

    return nested_model_path


def latest_round_dir(model_path: str | Path) -> Path:
    rounds_root = Path(model_path).expanduser().resolve() / "aura_iterative" / "rounds"
    if not rounds_root.is_dir():
        raise FileNotFoundError(f"Aura iterative rounds directory not found: {rounds_root}")

    round_dirs: list[tuple[int, str, Path]] = []
    for path in rounds_root.iterdir():
        if not path.is_dir():
            continue
        match = re.match(r"^(\d+)_", path.name)
        if match is None:
            continue
        round_dirs.append((int(match.group(1)), path.name, path))

    if not round_dirs:
        raise FileNotFoundError(f"No numeric round directories found under: {rounds_root}")
    return sorted(round_dirs, key=lambda item: (item[0], item[1]))[-1][2]


def default_render_dir(model_path: str | Path, iteration: int, explicit_round_dir: str | Path | None) -> Path:
    round_dir = Path(explicit_round_dir).expanduser().resolve() if explicit_round_dir else latest_round_dir(model_path)
    render_dir = round_dir / "workspace" / "train" / f"ours_{iteration}_object_inpaint" / "renders"
    if render_dir.is_dir():
        return render_dir

    candidates = sorted(round_dir.glob("workspace/train/ours_*_object_inpaint/renders"))
    candidate_text = "\n".join(f"  {path}" for path in candidates[:20]) or "  <none>"
    raise FileNotFoundError(
        f"After-inpaint render directory not found: {render_dir}\n"
        f"Available after-inpaint render directories under {round_dir}:\n{candidate_text}"
    )


def resolve_gt_path(gt_dir: Path, render_path: Path) -> Path:
    exact = gt_dir / render_path.name
    if exact.is_file():
        return exact

    matches = sorted(
        path
        for path in gt_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.stem == render_path.stem
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous removal_GT match for {render_path.name}: {matches}")
    raise FileNotFoundError(f"removal_GT image not found for render {render_path.name} under {gt_dir}")


def paired_images(render_dir: str | Path, gt_dir: str | Path) -> list[tuple[Path, Path]]:
    render_dir = Path(render_dir).expanduser().resolve()
    gt_dir = Path(gt_dir).expanduser().resolve()
    if not render_dir.is_dir():
        raise FileNotFoundError(f"Render directory not found: {render_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"removal_GT directory not found: {gt_dir}")

    render_paths = list_images(render_dir)
    if not render_paths:
        raise RuntimeError(f"No rendered images found under: {render_dir}")
    return [(render_path, resolve_gt_path(gt_dir, render_path)) for render_path in render_paths]


def load_rgb_tensor(path: str | Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        return tf.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)


def stage_images_for_fid(paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, source_path in enumerate(paths):
        suffix = source_path.suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            suffix = ".png"
        target_path = output_dir / f"{idx:05d}{suffix}"
        try:
            os.symlink(source_path.resolve(), target_path)
        except OSError:
            shutil.copy2(source_path, target_path)


def configure_fid_cache(fid_weights_url: str) -> Path:
    weights_name = Path(urlparse(fid_weights_url).path).name
    candidate_hubs = [
        SHARED_TORCH_HUB,
        Path(torch.hub.get_dir()).expanduser(),
        Path.home() / ".cache" / "torch" / "hub",
    ]

    seen: set[Path] = set()
    checked: list[Path] = []
    for hub_dir in candidate_hubs:
        hub_dir = hub_dir.expanduser().resolve()
        if hub_dir in seen:
            continue
        seen.add(hub_dir)
        checkpoint_path = hub_dir / "checkpoints" / weights_name
        checked.append(checkpoint_path)
        if checkpoint_path.is_file():
            torch.hub.set_dir(str(hub_dir))
            return checkpoint_path

    torch.hub.set_dir(str(SHARED_TORCH_HUB))
    raise FileNotFoundError(
        "pytorch-fid Inception checkpoint not found. Checked:\n"
        + "\n".join(f"  {path}" for path in checked)
    )


def calculate_fid_for_pairs(
    render_paths: list[Path],
    gt_paths: list[Path],
    device: torch.device,
    batch_size: int,
    fid_dims: int,
    fid_workers: int,
) -> tuple[float | None, str | None]:
    try:
        from pytorch_fid import fid_score
        from pytorch_fid.inception import FID_WEIGHTS_URL
    except FID_LOAD_EXCEPTIONS as exc:
        return None, f"{type(exc).__name__}: {exc}"

    try:
        fid_checkpoint = configure_fid_cache(FID_WEIGHTS_URL)
        print(f"Using FID checkpoint: {fid_checkpoint}")
        with tempfile.TemporaryDirectory(prefix="aurafusion_after_inpaint_fid_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            render_subset = tmp_root / "renders"
            gt_subset = tmp_root / "gt"
            stage_images_for_fid(render_paths, render_subset)
            stage_images_for_fid(gt_paths, gt_subset)
            with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
                fid = fid_score.calculate_fid_given_paths(
                    [str(gt_subset), str(render_subset)],
                    batch_size,
                    str(device),
                    fid_dims,
                    fid_workers,
                )
        return float(fid), None
    except FID_LOAD_EXCEPTIONS as exc:
        return None, f"{type(exc).__name__}: {exc}"


def compute_metrics(
    pairs: list[tuple[Path, Path]],
    device: torch.device,
    compute_fid: bool,
    batch_size: int,
    fid_dims: int,
    fid_workers: int,
) -> tuple[dict, dict]:
    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    per_view = {
        "PSNR": {},
        "SSIM": {},
        "LPIPS": {},
        "render_path": {},
        "gt_path": {},
    }

    with torch.no_grad():
        for render_path, gt_path in tqdm(pairs, desc="Metric evaluation progress"):
            render_tensor = load_rgb_tensor(render_path, device)
            gt_tensor = load_rgb_tensor(gt_path, device)
            if tuple(render_tensor.shape) != tuple(gt_tensor.shape):
                raise RuntimeError(
                    f"Image shape mismatch for {render_path.name}: "
                    f"render={tuple(render_tensor.shape)} gt={tuple(gt_tensor.shape)}"
                )

            image_psnr = psnr(render_tensor, gt_tensor).mean().item()
            image_ssim = ssim(render_tensor, gt_tensor).item()
            image_lpips = lpips_model(render_tensor, gt_tensor).item()
            psnr_values.append(image_psnr)
            ssim_values.append(image_ssim)
            lpips_values.append(image_lpips)

            key = render_path.name
            per_view["PSNR"][key] = image_psnr
            per_view["SSIM"][key] = image_ssim
            per_view["LPIPS"][key] = image_lpips
            per_view["render_path"][key] = str(render_path)
            per_view["gt_path"][key] = str(gt_path)

    render_paths = [render_path for render_path, _ in pairs]
    gt_paths = [gt_path for _, gt_path in pairs]
    fid = None
    fid_error = None
    if compute_fid:
        fid, fid_error = calculate_fid_for_pairs(render_paths, gt_paths, device, batch_size, fid_dims, fid_workers)
        if fid_error:
            print(f"Skipping FID: {fid_error}", file=sys.stderr)

    results = {
        "num_images": len(pairs),
        "PSNR": float(np.mean(psnr_values)),
        "SSIM": float(np.mean(ssim_values)),
        "LPIPS": float(np.mean(lpips_values)),
        "FID": fid,
        "FID_error": fid_error,
    }
    return results, per_view


def write_outputs(
    output_dir: str | Path,
    model_path: Path,
    source_path: Path,
    render_dir: Path,
    gt_dir: Path,
    round_dir: Path | None,
    results: dict,
    per_view: dict,
) -> None:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_path": str(model_path),
        "source_path": str(source_path),
        "round_dir": None if round_dir is None else str(round_dir),
        "render_dir": str(render_dir),
        "gt_dir": str(gt_dir),
    }
    json_results = {"after_inpaint": {**metadata, **results}}
    (output_dir / "results_after_inpaint.json").write_text(
        json.dumps(json_results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "per_view_after_inpaint.json").write_text(
        json.dumps({"after_inpaint": per_view}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fid_text = "N/A" if results["FID"] is None else f"{float(results['FID']):.7f}"
    lines = [
        "method\tPSNR\tSSIM\tLPIPS\tFID",
        (
            "after_inpaint"
            f"\t{results['PSNR']:.7f}"
            f"\t{results['SSIM']:.7f}"
            f"\t{results['LPIPS']:.7f}"
            f"\t{fid_text}"
        ),
        "",
        f"model_path: {model_path}",
        f"source_path: {source_path}",
        f"render_dir: {render_dir}",
        f"gt_dir: {gt_dir}",
        f"num_images: {results['num_images']}",
    ]
    if results.get("FID_error"):
        lines.append(f"FID_error: {results['FID_error']}")
    (output_dir / "qualitative_comparison_after_inpaint.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the latest AuraFusion360 iterative after-inpaint train renders "
            "against source_path/removal_GT."
        )
    )
    parser.add_argument("-m", "--model_path", required=True, help="AuraFusion360 model/output root.")
    parser.add_argument("-s", "--source_path", required=True, help="Scene source path containing removal_GT.")
    parser.add_argument("--iteration", type=int, default=5000, help="Finetune iteration in ours_<iter>_object_inpaint.")
    parser.add_argument("--round_dir", default=None, help="Optional explicit aura_iterative/rounds/<round> directory.")
    parser.add_argument("--render_dir", default=None, help="Optional explicit render directory override.")
    parser.add_argument("--gt_dir", default=None, help="Optional GT directory override. Default: <source_path>/removal_GT.")
    parser.add_argument("--output_dir", default=None, help="Output directory. Default: <model_path>/aura_iterative.")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--fid_dims", type=int, default=2048)
    parser.add_argument("--fid_workers", type=int, default=0)
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_model_path = Path(args.model_path).expanduser().resolve()
    model_path = resolve_model_path(requested_model_path)
    source_path = Path(args.source_path).expanduser().resolve()
    explicit_round_dir = Path(args.round_dir).expanduser().resolve() if args.round_dir else None

    if args.render_dir:
        render_dir = Path(args.render_dir).expanduser().resolve()
        round_dir = explicit_round_dir
    else:
        round_dir = explicit_round_dir or latest_round_dir(model_path)
        render_dir = default_render_dir(model_path, args.iteration, round_dir)

    gt_dir = Path(args.gt_dir).expanduser().resolve() if args.gt_dir else source_path / "removal_GT"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else model_path / "aura_iterative"

    pairs = paired_images(render_dir, gt_dir)
    device = torch.device(args.device)
    results, per_view = compute_metrics(
        pairs,
        device,
        compute_fid=not args.skip_fid,
        batch_size=args.batch_size,
        fid_dims=args.fid_dims,
        fid_workers=args.fid_workers,
    )
    write_outputs(output_dir, model_path, source_path, render_dir, gt_dir, round_dir, results, per_view)

    print(f"After-inpaint metrics for {len(pairs)} paired images")
    if model_path != requested_model_path:
        print(f"Resolved model path: {model_path}")
    print(f"Render dir: {render_dir}")
    print(f"GT dir: {gt_dir}")
    print(f"PSNR : {results['PSNR']:.7f}")
    print(f"SSIM : {results['SSIM']:.7f}")
    print(f"LPIPS: {results['LPIPS']:.7f}")
    if results["FID"] is None:
        print("FID  : skipped")
    else:
        print(f"FID  : {results['FID']:.7f}")
    print(f"Wrote: {output_dir / 'qualitative_comparison_after_inpaint.txt'}")


if __name__ == "__main__":
    main()
