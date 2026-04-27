from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")


def normalize_target_id(target_id) -> str:
    if target_id is None:
        raise ValueError("target_id must not be None")
    if isinstance(target_id, str):
        text = target_id.strip()
        if not text:
            raise ValueError("target_id must not be empty")
        if text.startswith("[") and text.endswith("]"):
            return normalize_target_id(json.loads(text))
        return "_".join(text.replace(",", " ").split())
    if isinstance(target_id, Sequence) and not isinstance(target_id, (bytes, bytearray)):
        if len(target_id) == 0:
            raise ValueError("target_id sequence must not be empty")
        return "_".join(str(int(item)) for item in target_id)
    return str(int(target_id))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_target_inpaint_root(model_path: str | Path, target_id) -> Path:
    return Path(model_path).expanduser().resolve() / "target_inpaint" / normalize_target_id(target_id)


def get_object_masks_dir(model_path: str | Path, target_id) -> Path:
    return get_target_inpaint_root(model_path, target_id) / "object_masks"


def get_unseen_masks_dir(model_path: str | Path, target_id) -> Path:
    return get_target_inpaint_root(model_path, target_id) / "unseen_masks"


def get_unseen_masks_dilated_dir(model_path: str | Path, target_id) -> Path:
    return get_target_inpaint_root(model_path, target_id) / "unseen_masks_dilated"


def get_inpaint_images_dir(model_path: str | Path, target_id) -> Path:
    return get_target_inpaint_root(model_path, target_id) / "inpaint"


def find_image_for_stem(directory: str | Path, stem: str, extensions=DEFAULT_IMAGE_EXTENSIONS) -> Path:
    directory = Path(directory).expanduser().resolve()
    for ext in extensions:
        candidate = directory / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image not found for stem '{stem}' in {directory}")
