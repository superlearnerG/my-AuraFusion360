from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")


def parse_target_ids(value: str) -> list[int]:
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        parsed = json.loads(text)
        if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes, bytearray)):
            return [int(item) for item in parsed]
        return [int(parsed)]
    return [int(item) for item in text.replace(",", " ").split()]


def iter_image_files(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix in IMAGE_EXTENSIONS)


def find_mask_for_stem(raw_mask_dir: Path, stem: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        candidate = raw_mask_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Mask for image stem '{stem}' not found in {raw_mask_dir}")


def read_mask_first_channel(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract per-round binary masks from multi-gray object masks.")
    parser.add_argument("--raw_mask_dir", required=True, help="Directory with multi-gray object masks.")
    parser.add_argument("--output_dir", required=True, help="Directory where binary target masks are written.")
    parser.add_argument("--target_id", required=True, help="Target id or id list, e.g. 70 or 40,42.")
    parser.add_argument("--image_dir", default=None, help="Optional image directory used to enforce basename alignment.")
    args = parser.parse_args()

    raw_mask_dir = Path(args.raw_mask_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not raw_mask_dir.is_dir():
        raise FileNotFoundError(f"raw_mask_dir not found: {raw_mask_dir}")

    target_ids = parse_target_ids(args.target_id)
    if not target_ids:
        raise ValueError("--target_id resolved to an empty id list")

    if args.image_dir:
        image_dir = Path(args.image_dir).expanduser().resolve()
        if not image_dir.is_dir():
            raise FileNotFoundError(f"image_dir not found: {image_dir}")
        input_pairs = [(image_path.stem, find_mask_for_stem(raw_mask_dir, image_path.stem)) for image_path in iter_image_files(image_dir)]
    else:
        input_pairs = [(mask_path.stem, mask_path) for mask_path in iter_image_files(raw_mask_dir)]

    if not input_pairs:
        raise RuntimeError(f"No input masks found in {raw_mask_dir}")

    nonempty_count = 0
    pixel_counts = {}
    for stem, mask_path in input_pairs:
        raw_mask = read_mask_first_channel(mask_path)
        binary = np.isin(raw_mask, target_ids).astype(np.uint8) * 255
        count = int((binary > 0).sum())
        pixel_counts[stem] = count
        if count > 0:
            nonempty_count += 1
        Image.fromarray(binary).save(output_dir / f"{stem}.png")

    if nonempty_count == 0:
        raise RuntimeError(f"All extracted masks are empty for target ids {target_ids}")

    manifest = {
        "raw_mask_dir": str(raw_mask_dir),
        "output_dir": str(output_dir),
        "target_id": target_ids,
        "mask_count": len(input_pairs),
        "nonempty_mask_count": nonempty_count,
        "pixel_counts": pixel_counts,
    }
    with (output_dir / "target_mask_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Prepared {len(input_pairs)} masks for target ids {target_ids}: {output_dir}")


if __name__ == "__main__":
    main()
