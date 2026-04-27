from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARE_METHODS_ROOT = PROJECT_ROOT.parent
PRETRAINED_MODELS_ROOT = COMPARE_METHODS_ROOT / "pretrained_models"


def pretrained_path(*parts: str) -> Path:
    return PRETRAINED_MODELS_ROOT.joinpath(*parts)


def require_pretrained_file(*parts: str, min_bytes: int = 1) -> Path:
    path = pretrained_path(*parts)
    if not path.is_file():
        raise FileNotFoundError(f"Required pretrained model file not found: {path}")
    if path.stat().st_size < min_bytes:
        raise RuntimeError(
            f"Pretrained model file is invalid or incomplete: {path} "
            f"({path.stat().st_size} bytes)"
        )
    return path


def require_pretrained_file_any(candidates: tuple[tuple[str, ...], ...], min_bytes: int = 1) -> Path:
    attempted = []
    invalid = []
    for parts in candidates:
        path = pretrained_path(*parts)
        attempted.append(str(path))
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size < min_bytes:
            invalid.append(f"{path} ({size} bytes)")
            continue
        return path

    message = "Required pretrained model file not found. Tried:\n  " + "\n  ".join(attempted)
    if invalid:
        message += "\nInvalid or incomplete files:\n  " + "\n  ".join(invalid)
    raise FileNotFoundError(message)


def require_pretrained_dir(*parts: str) -> Path:
    path = pretrained_path(*parts)
    if not path.is_dir():
        raise FileNotFoundError(f"Required pretrained model directory not found: {path}")
    return path


def require_stable_diffusion_inpaint_checkpoint(min_bytes: int = 1024 * 1024) -> Path:
    return require_pretrained_file_any(
        (
            ("stable-diffusion-2-inpainting", "512-inpainting-ema.ckpt"),
            ("stable-diffusion-2-inpainting", "512-inpainting-ema.safetensors"),
        ),
        min_bytes=min_bytes,
    )


def use_local_torch_home() -> None:
    os.environ.setdefault("TORCH_HOME", str(pretrained_path("torch")))
