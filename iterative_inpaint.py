from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.inpaint_target_paths import (
    ensure_dir,
    find_image_for_stem,
    get_inpaint_images_dir,
    get_object_masks_dir,
    get_unseen_masks_dilated_dir,
    get_unseen_masks_dir,
    normalize_target_id,
)
from utils.iterative_workflow import (
    bootstrap_workspace_from_base_model,
    bootstrap_workspace_from_snapshot,
    get_iterative_root,
    get_round_dir,
    get_round_meta_path,
    get_round_scene_in_dir,
    get_round_scene_out_dir,
    get_round_workspace,
    normalize_id_list,
    read_json,
    remove_path,
    resolve_path,
    save_scene_snapshot,
    write_json,
)
from utils.pretrained_paths import configure_pretrained_env


IMAGE_EXTENSIONS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round-based iterative inpaint workflow for my-AuraFusion360.")
    parser.add_argument("command", choices=["init", "prepare-round", "run-leftrefill", "initialize-round", "finalize-round", "status"])
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--workflow_config", required=True)
    parser.add_argument("--round_index", type=int, default=None)
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--render_intermediate", action="store_true")
    return parser.parse_args()


def load_workflow_config(path: str | Path) -> dict[str, Any]:
    workflow = read_json(path)
    workflow.setdefault("defaults", {})
    workflow.setdefault("rounds", [])
    if not workflow["rounds"]:
        raise ValueError(f"'rounds' must be a non-empty list in workflow config: {path}")
    workflow["_config_dir"] = str(Path(path).expanduser().resolve().parent)
    return workflow


def workflow_manifest_path(model_path: str | Path) -> Path:
    return get_iterative_root(model_path) / "workflow_manifest.json"


def ensure_workflow_initialized(args: argparse.Namespace, workflow: dict[str, Any]) -> Path:
    iterative_root = ensure_dir(get_iterative_root(args.model_path))
    manifest = {
        "source_path": str(Path(args.source_path).expanduser().resolve()),
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "workflow_config": str(Path(args.workflow_config).expanduser().resolve()),
        "round_count": len(workflow["rounds"]),
    }
    write_json(workflow_manifest_path(args.model_path), manifest)
    return iterative_root


def resolve_round_spec(workflow: dict[str, Any], round_index: int) -> dict[str, Any]:
    if round_index < 0 or round_index >= len(workflow["rounds"]):
        raise IndexError(f"round_index out of range: {round_index}")
    spec = dict(workflow.get("defaults", {}))
    spec.update(workflow["rounds"][round_index])
    spec["target_id"] = normalize_id_list(spec.get("target_id"))
    if not spec["target_id"]:
        raise ValueError(f"Round {round_index} has empty target_id")
    spec.setdefault("removal_thresh", 0.6)
    spec.setdefault("fit_mask_iterations", 2000)
    spec.setdefault("finetune_iteration", 10000)
    spec.setdefault("reference_index", -1)
    spec.setdefault("dilate_mask_kernel_size", 5)
    spec.setdefault("dilate_mask_iter", 3)
    spec.setdefault("depth_align_min_val", -1.0)
    spec.setdefault("depth_align_max_val", 0.0)
    spec.setdefault("depth_align_percentile", 99.5)
    spec.setdefault("skip_eval", True)
    spec.setdefault("images", "images")
    spec.setdefault("lama_reference_strategy", workflow.get("lama_reference", {}).get("strategy", "max_mask"))
    spec.setdefault("lama_reference_stem", workflow.get("lama_reference", {}).get("stem", ""))
    spec.setdefault("simple_lama_device", workflow.get("lama_reference", {}).get("device", "cuda"))
    spec.setdefault("sdedit_strength", workflow.get("sdedit", {}).get("strength", 0.85))
    spec.setdefault("sdedit_eta", workflow.get("sdedit", {}).get("eta", 1.0))
    spec.setdefault("sdedit_scale", workflow.get("sdedit", {}).get("scale", 2.5))
    spec.setdefault("sdedit_use_ddim_inversion", workflow.get("sdedit", {}).get("use_ddim_inversion", False))
    return spec


def round_paths(args: argparse.Namespace, workflow: dict[str, Any], spec: dict[str, Any], round_index: int) -> dict[str, Path]:
    iterative_root = ensure_workflow_initialized(args, workflow)
    round_dir = get_round_dir(iterative_root, round_index, spec["target_id"])
    workspace = get_round_workspace(round_dir)
    target_tag = normalize_target_id(spec["target_id"])
    return {
        "iterative_root": iterative_root,
        "round_dir": round_dir,
        "workspace": workspace,
        "scene_in": get_round_scene_in_dir(round_dir),
        "scene_out": get_round_scene_out_dir(round_dir),
        "meta_path": get_round_meta_path(round_dir),
        "object_masks": get_object_masks_dir(workspace, target_tag),
        "unseen_masks": get_unseen_masks_dir(workspace, target_tag),
        "unseen_masks_dilated": get_unseen_masks_dilated_dir(workspace, target_tag),
        "leftrefill_reference": get_inpaint_images_dir(workspace, target_tag).parent / "leftrefill_reference",
        "inpaint_images": get_inpaint_images_dir(workspace, target_tag),
    }


def load_round_meta(paths: dict[str, Path], base: dict[str, Any]) -> dict[str, Any]:
    if paths["meta_path"].is_file():
        return read_json(paths["meta_path"])
    return dict(base)


def save_round_meta(paths: dict[str, Path], meta: dict[str, Any]) -> None:
    write_json(paths["meta_path"], meta)


def base_meta(args: argparse.Namespace, round_index: int, spec: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    return {
        "round_index": round_index,
        "target_id": spec["target_id"],
        "target_tag": normalize_target_id(spec["target_id"]),
        "workspace_model_path": str(paths["workspace"]),
        "scene_in_dir": str(paths["scene_in"]),
        "scene_out_dir": str(paths["scene_out"]),
        "object_masks_dir": str(paths["object_masks"]),
        "unseen_masks_dir": str(paths["unseen_masks"]),
        "unseen_masks_dilated_dir": str(paths["unseen_masks_dilated"]),
        "leftrefill_reference_dir": str(paths["leftrefill_reference"]),
        "inpaint_images_dir": str(paths["inpaint_images"]),
    }


def run_cmd(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command))
    env = os.environ.copy()
    project_root_str = str(PROJECT_ROOT)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = project_root_str if not existing else project_root_str + os.pathsep + existing
    env.setdefault("TORCH_HOME", str(PROJECT_ROOT.parent / "pretrained_models" / "torch"))
    subprocess.run(command, cwd=str(cwd or PROJECT_ROOT), env=env, check=True)


def source_image_dir(source_path: str | Path, spec: dict[str, Any]) -> Path:
    images = Path(str(spec.get("images", "images"))).expanduser()
    if images.is_absolute():
        return images.resolve()
    return (Path(source_path).expanduser().resolve() / images).resolve()


def removal_render_dir(paths: dict[str, Path]) -> Path:
    return paths["workspace"] / "train" / "ours_0_object_removal" / "renders"


def list_images(directory: str | Path) -> list[Path]:
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    images = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images found in {directory}")
    return images


def assert_not_source_images(source_path: str | Path, path: str | Path, label: str) -> None:
    source_images = (Path(source_path).expanduser().resolve() / "images").resolve()
    candidate = Path(path).expanduser().resolve()
    if candidate == source_images or source_images in candidate.parents:
        raise RuntimeError(
            f"{label} points to source/images, which would leak original object pixels into iterative inpaint: {candidate}"
        )


def find_reference_image_from_removal(paths: dict[str, Path], spec: dict[str, Any], args: argparse.Namespace) -> Path:
    render_root = removal_render_dir(paths)
    refs = list_images(render_root)
    reference_index = int(spec.get("reference_index", -1))
    try:
        ref_img_path = refs[reference_index]
    except IndexError as exc:
        raise IndexError(
            f"reference_index {reference_index} is out of range for {len(refs)} removal renders in {render_root}"
        ) from exc
    assert_not_source_images(args.source_path, ref_img_path, "LeftRefill reference image")
    return ref_img_path


def _image_map_by_stem(paths: list[Path], label: str) -> dict[str, Path]:
    by_stem = {}
    for path in paths:
        stem = path.stem
        if stem in by_stem:
            raise ValueError(f"Duplicate {label} stem '{stem}': {by_stem[stem]} and {path}")
        by_stem[stem] = path
    return by_stem


def normalize_reference_stem(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("lama_reference_stem must not be empty when using strategy=stem")
    return Path(text).stem


def load_binary_mask(mask_path: Path) -> Image.Image:
    with Image.open(mask_path) as mask_image:
        mask = np.array(mask_image.convert("L"))
    binary = np.where(mask > 127, 255, 0).astype(np.uint8)
    return Image.fromarray(binary, mode="L")


def crop_output_to_input_size(output: Image.Image, input_size: tuple[int, int], input_path: Path) -> Image.Image:
    if output.size == input_size:
        return output

    input_width, input_height = input_size
    output_width, output_height = output.size
    if output_width < input_width or output_height < input_height:
        raise RuntimeError(
            "SimpleLaMa output is smaller than input: "
            f"{input_path} input={input_size}, output={output.size}"
        )
    return output.crop((0, 0, input_width, input_height))


def init_simple_lama(device: str):
    configure_pretrained_env(include_simple_lama=True)
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError as exc:
        raise ImportError(
            "simple_lama_inpainting is required for LaMa reference generation. "
            "Install it in the Python environment used by pipeline.sh."
        ) from exc
    return SimpleLama(device=device)


def select_lama_reference(paths: dict[str, Path], spec: dict[str, Any]) -> dict[str, Any]:
    source_root = removal_render_dir(paths)
    source_images = list_images(source_root)
    mask_paths = list_images(paths["unseen_masks_dilated"])
    source_by_stem = _image_map_by_stem(source_images, "removal render")
    mask_by_stem = _image_map_by_stem(mask_paths, "unseen mask")
    matched_stems = [path.stem for path in source_images if path.stem in mask_by_stem]
    if not matched_stems:
        raise FileNotFoundError(
            f"No matching stems between removal renders in {source_root} "
            f"and unseen masks in {paths['unseen_masks_dilated']}"
        )

    strategy = str(spec.get("lama_reference_strategy", "max_mask")).strip().lower()
    if strategy == "first":
        selected_stem = matched_stems[0]
        selected_area = int(np.array(load_binary_mask(mask_by_stem[selected_stem])).sum() // 255)
    elif strategy == "stem":
        selected_stem = normalize_reference_stem(str(spec.get("lama_reference_stem", "")))
        if selected_stem not in source_by_stem:
            raise FileNotFoundError(
                f"Selected reference stem '{selected_stem}' has no train-split removal render in "
                f"{source_root}. Pass a basename from the train split, not a held-out test view."
            )
        if selected_stem not in mask_by_stem:
            raise FileNotFoundError(
                f"Selected reference stem '{selected_stem}' has no unseen mask in "
                f"{paths['unseen_masks_dilated']}"
            )
        selected_area = int(np.array(load_binary_mask(mask_by_stem[selected_stem])).sum() // 255)
    elif strategy == "index":
        reference_index = int(spec.get("reference_index", -1))
        try:
            selected_image = source_images[reference_index]
        except IndexError as exc:
            raise IndexError(
                f"reference_index {reference_index} is out of range for "
                f"{len(source_images)} removal renders in {source_root}"
            ) from exc
        selected_stem = selected_image.stem
        if selected_stem not in mask_by_stem:
            raise FileNotFoundError(
                f"Selected reference stem '{selected_stem}' has no unseen mask in "
                f"{paths['unseen_masks_dilated']}"
            )
        selected_area = int(np.array(load_binary_mask(mask_by_stem[selected_stem])).sum() // 255)
    elif strategy == "max_mask":
        selected_stem = ""
        selected_area = -1
        for stem in matched_stems:
            area = int(np.array(load_binary_mask(mask_by_stem[stem])).sum() // 255)
            if area > selected_area:
                selected_stem = stem
                selected_area = area
        if not selected_stem:
            raise RuntimeError(f"Failed to select LaMa reference from {paths['unseen_masks_dilated']}")
    else:
        raise ValueError(
            f"Unsupported lama_reference_strategy '{strategy}'. "
            "Expected one of: max_mask, first, index, stem."
        )

    return {
        "strategy": strategy,
        "stem": selected_stem,
        "source_image": source_by_stem[selected_stem],
        "mask_image": mask_by_stem[selected_stem],
        "mask_pixels": selected_area,
    }


def run_lama_reference(args: argparse.Namespace, spec: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    selected = select_lama_reference(paths, spec)
    source_image = Path(selected["source_image"])
    mask_image = Path(selected["mask_image"])
    output_dir = ensure_dir(paths["leftrefill_reference"])
    output_path = output_dir / f"{selected['stem']}.png"

    assert_not_source_images(args.source_path, source_image, "SimpleLaMa reference source image")
    with Image.open(source_image) as image:
        image = image.convert("RGB")
    mask = load_binary_mask(mask_image)
    if image.size != mask.size:
        raise RuntimeError(
            "SimpleLaMa reference image/mask size mismatch: "
            f"{source_image} size={image.size}, {mask_image} size={mask.size}"
        )

    simple_lama = init_simple_lama(str(spec["simple_lama_device"]))
    output = simple_lama(image, mask).convert("RGB")
    output = crop_output_to_input_size(output, image.size, source_image)
    output.save(output_path)

    print(
        "[LaMa reference] "
        f"strategy={selected['strategy']} "
        f"ref_view={selected['stem']} "
        f"mask_pixels={selected['mask_pixels']} "
        f"source={source_image} "
        f"mask={mask_image} "
        f"output={output_path}"
    )

    return {
        "strategy": selected["strategy"],
        "reference_stem": selected["stem"],
        "reference_source_image": str(source_image),
        "reference_mask_image": str(mask_image),
        "reference_mask_pixels": selected["mask_pixels"],
        "reference_image": str(output_path),
    }


def copy_reference_to_inpaint_output(reference_info: dict[str, Any], paths: dict[str, Path]) -> None:
    ref_img_path = Path(reference_info["reference_image"])
    ref_stem = str(reference_info["reference_stem"])
    ensure_dir(paths["inpaint_images"])
    try:
        output_path = find_image_for_stem(paths["inpaint_images"], ref_stem)
    except FileNotFoundError:
        output_path = paths["inpaint_images"] / f"{ref_stem}.png"
    with Image.open(ref_img_path) as reference:
        reference.convert("RGB").save(output_path)
    print(f"[LaMa reference] copied selected ref view into LeftRefill outputs: {output_path}")


def workflow_path(workflow: dict[str, Any], key: str, default: str | None = None) -> Path | None:
    value = workflow.get(key, default)
    if value is None:
        return None
    return resolve_path(value, PROJECT_ROOT)


def ensure_round_workspace(args: argparse.Namespace, workflow: dict[str, Any], spec: dict[str, Any], paths: dict[str, Path], round_index: int) -> dict[str, Any]:
    if args.force and paths["round_dir"].exists():
        remove_path(paths["round_dir"])
    ensure_dir(paths["round_dir"])
    if paths["workspace"].exists():
        bootstrap_manifest = read_json(paths["workspace"] / "scene_state_bootstrap.json", default={"source_type": "existing_workspace"})
    elif round_index == 0:
        bootstrap_manifest = bootstrap_workspace_from_base_model(
            args.model_path,
            paths["workspace"],
            iteration=int(workflow.get("base_iteration", 30000)),
        )
    else:
        prev_spec = resolve_round_spec(workflow, round_index - 1)
        prev_round_dir = get_round_dir(paths["iterative_root"], round_index - 1, prev_spec["target_id"])
        prev_scene_out = get_round_scene_out_dir(prev_round_dir)
        bootstrap_manifest = bootstrap_workspace_from_snapshot(prev_scene_out, paths["workspace"])
    save_scene_snapshot(
        paths["scene_in"],
        paths["workspace"] / "point_cloud" / "iteration_0" / "point_cloud.ply",
        cfg_args_path=paths["workspace"] / "cfg_args",
        state=bootstrap_manifest,
    )
    return bootstrap_manifest


def is_last_round(workflow: dict[str, Any], round_index: int) -> bool:
    return round_index == len(workflow["rounds"]) - 1


def prepare_unseen_masks_dilated(paths: dict[str, Path], spec: dict[str, Any]) -> None:
    mask_paths = list_images(paths["unseen_masks"])
    output_dir = ensure_dir(paths["unseen_masks_dilated"])
    kernel_size = max(1, int(spec["dilate_mask_kernel_size"]))
    dilate_iter = int(spec["dilate_mask_iter"])
    keep_all_components = len(spec["target_id"]) > 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = np.where(mask > 127, 1, 0).astype(np.uint8)
        if dilate_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            if not keep_all_components:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
                if num_labels > 1:
                    largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
                    cleaned_mask = np.zeros_like(mask)
                    cleaned_mask[labels == largest_component] = 1
                    mask = cleaned_mask.astype(np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
        Image.fromarray((mask * 255).astype(np.uint8)).save(output_dir / mask_path.name)


def run_initial_inpaint(args: argparse.Namespace, spec: dict[str, Any], paths: dict[str, Path], reference_stem: str | None = None) -> None:
    inpaint_images = list_images(paths["inpaint_images"])
    if not inpaint_images:
        raise FileNotFoundError(f"No LeftRefill inpaint images found in {paths['inpaint_images']}")
    assert_not_source_images(args.source_path, paths["inpaint_images"], "Initial inpaint image root")
    init_cmd = [
        args.python_bin,
        "inpaint.py",
        "-s",
        str(Path(args.source_path).expanduser().resolve()),
        "-m",
        str(paths["workspace"]),
        "--iteration",
        "0",
        "--images",
        str(paths["inpaint_images"]),
        "--unseen_mask_dir",
        str(paths["unseen_masks"]),
        "--unseen_mask_dilated_dir",
        str(paths["unseen_masks_dilated"]),
        "--reference_index",
        str(int(spec["reference_index"])),
        "--dilate_mask_kernel_size",
        str(int(spec["dilate_mask_kernel_size"])),
        "--dilate_mask_iter",
        str(int(spec["dilate_mask_iter"])),
        "--depth_align_min_val",
        str(float(spec["depth_align_min_val"])),
        "--depth_align_max_val",
        str(float(spec["depth_align_max_val"])),
        "--depth_align_percentile",
        str(float(spec["depth_align_percentile"])),
        "--finetune_iteration",
        "-1",
        "--skip_test",
        "--skip_mesh",
        "--skip_eval",
    ]
    if reference_stem:
        init_cmd.extend(["--reference_stem", reference_stem])
    if len(spec["target_id"]) > 1:
        init_cmd.append("--keep_all_unseen_components")
    run_cmd(init_cmd)


def run_prepare_round(args: argparse.Namespace, workflow: dict[str, Any], round_index: int) -> None:
    spec = resolve_round_spec(workflow, round_index)
    paths = round_paths(args, workflow, spec, round_index)
    bootstrap_manifest = ensure_round_workspace(args, workflow, spec, paths, round_index)
    meta = load_round_meta(paths, base_meta(args, round_index, spec, paths))
    meta.update(base_meta(args, round_index, spec, paths))
    meta["status"] = "workspace_ready"
    meta["bootstrap"] = bootstrap_manifest
    save_round_meta(paths, meta)

    raw_mask_dir = workflow_path(workflow, "raw_object_mask_dir")
    if raw_mask_dir is None:
        raise ValueError("workflow_config must define raw_object_mask_dir")
    image_dir = source_image_dir(args.source_path, spec)
    target_tag = normalize_target_id(spec["target_id"])
    target_id_arg = ",".join(str(item) for item in spec["target_id"])
    run_cmd([
        args.python_bin,
        "tools/prepare_target_masks.py",
        "--raw_mask_dir",
        str(raw_mask_dir),
        "--output_dir",
        str(paths["object_masks"]),
        "--target_id",
        target_id_arg,
        "--image_dir",
        str(image_dir),
    ])

    stats_path = paths["meta_path"].parent / "fit_target_mask_stats.json"
    run_cmd([
        args.python_bin,
        "fit_target_mask.py",
        "-s",
        str(Path(args.source_path).expanduser().resolve()),
        "-m",
        str(paths["workspace"]),
        "--iteration",
        "0",
        "--object_mask_dir",
        str(paths["object_masks"]),
        "--fit_mask_iterations",
        str(int(spec["fit_mask_iterations"])),
        "--removal_thresh",
        str(float(spec["removal_thresh"])),
        "--stats_path",
        str(stats_path),
        "--dilate_mask_iter",
        "0",
    ])

    remove_cmd = [
        args.python_bin,
        "remove.py",
        "-s",
        str(Path(args.source_path).expanduser().resolve()),
        "-m",
        str(paths["workspace"]),
        "--iteration",
        "0",
        "--object_mask_dir",
        str(paths["object_masks"]),
        "--removal_thresh",
        str(float(spec["removal_thresh"])),
        "--target_tag",
        target_tag,
        "--skip_test",
        "--skip_mesh",
        "--dilate_mask_iter",
        "0",
    ]
    run_cmd(remove_cmd)

    removal_output_path = paths["workspace"] / "train" / "ours_0_object_removal"
    removal_renders = removal_render_dir(paths)
    list_images(removal_renders)
    assert_not_source_images(args.source_path, removal_renders, "Removal render root")
    run_cmd([
        args.python_bin,
        "utils/sam2_utils.py",
        "--source_path",
        str(Path(args.source_path).expanduser().resolve()),
        "--removal_output_path",
        str(removal_output_path),
        "--unseen_mask_output_dir",
        str(paths["unseen_masks"]),
        "--name_dir",
        str(image_dir),
    ])
    prepare_unseen_masks_dilated(paths, spec)

    meta["status"] = "unseen_masks_ready"
    save_round_meta(paths, meta)


def run_leftrefill(args: argparse.Namespace, workflow: dict[str, Any], round_index: int) -> None:
    spec = resolve_round_spec(workflow, round_index)
    paths = round_paths(args, workflow, spec, round_index)
    meta = load_round_meta(paths, base_meta(args, round_index, spec, paths))
    if paths["leftrefill_reference"].exists():
        remove_path(paths["leftrefill_reference"])
    if paths["inpaint_images"].exists():
        remove_path(paths["inpaint_images"])
    reference_info = run_lama_reference(args, spec, paths)
    ref_img_path = Path(reference_info["reference_image"])
    source_root = removal_render_dir(paths)
    ref_root = removal_render_dir(paths)
    list_images(source_root)
    assert_not_source_images(args.source_path, source_root, "LeftRefill source root")
    assert_not_source_images(args.source_path, ref_root, "LeftRefill name-matching root")
    cmd = [
        args.python_bin,
        "utils/LeftRefill/sdedit_utils.py",
        "--script",
        "sdedit",
        "--ref_img_path",
        str(ref_img_path),
        "--source_root",
        str(source_root),
        "--ref_root",
        str(ref_root),
        "--mask_root",
        str(paths["unseen_masks_dilated"]),
        "--output_root",
        str(paths["inpaint_images"]),
        "--strength",
        str(float(spec["sdedit_strength"])),
        "--eta",
        str(float(spec["sdedit_eta"])),
        "--scale",
        str(float(spec["sdedit_scale"])),
    ]
    if bool(spec.get("sdedit_use_ddim_inversion", False)):
        cmd.append("--use_ddim_inversion")
    run_cmd(cmd)
    copy_reference_to_inpaint_output(reference_info, paths)
    meta["status"] = "leftrefill_ready"
    meta["leftrefill_reference"] = reference_info
    meta["reference_image"] = str(ref_img_path)
    meta["reference_stem"] = reference_info["reference_stem"]
    save_round_meta(paths, meta)


def run_initialize_round(args: argparse.Namespace, workflow: dict[str, Any], round_index: int) -> None:
    spec = resolve_round_spec(workflow, round_index)
    paths = round_paths(args, workflow, spec, round_index)
    meta = load_round_meta(paths, base_meta(args, round_index, spec, paths))
    reference_stem = meta.get("reference_stem")
    run_initial_inpaint(args, spec, paths, reference_stem=reference_stem)
    meta["status"] = "initial_inpaint_ready"
    meta["initial_inpaint_image_root"] = str(paths["inpaint_images"])
    if reference_stem:
        meta["initial_reference_stem"] = reference_stem
    meta["initial_point_cloud_ply"] = str(paths["workspace"] / "point_cloud" / "iteration_object_inpaint_init" / "point_cloud.ply")
    save_round_meta(paths, meta)


def run_finalize_round(args: argparse.Namespace, workflow: dict[str, Any], round_index: int) -> None:
    spec = resolve_round_spec(workflow, round_index)
    paths = round_paths(args, workflow, spec, round_index)
    meta = load_round_meta(paths, base_meta(args, round_index, spec, paths))
    init_ply = paths["workspace"] / "point_cloud" / "iteration_object_inpaint_init" / "point_cloud.ply"
    if not init_ply.is_file():
        raise FileNotFoundError(
            f"Initial inpaint point cloud not found: {init_ply}. "
            "Run the initialize-round stage after run-leftrefill before finalizing."
        )
    final_cmd = [
        args.python_bin,
        "inpaint.py",
        "-s",
        str(Path(args.source_path).expanduser().resolve()),
        "-m",
        str(paths["workspace"]),
        "--iteration",
        "0",
        "--images",
        str(paths["inpaint_images"]),
        "--unseen_mask_dir",
        str(paths["unseen_masks"]),
        "--unseen_mask_dilated_dir",
        str(paths["unseen_masks_dilated"]),
        "--dilate_mask_kernel_size",
        str(int(spec["dilate_mask_kernel_size"])),
        "--dilate_mask_iter",
        str(int(spec["dilate_mask_iter"])),
        "--finetune_iteration",
        str(int(spec["finetune_iteration"])),
        "--skip_mesh",
        "--skip_eval",
    ]
    if len(spec["target_id"]) > 1:
        final_cmd.append("--keep_all_unseen_components")
    if not args.render_intermediate and not is_last_round(workflow, round_index):
        final_cmd.extend(["--skip_train", "--skip_test"])
    run_cmd(final_cmd)

    final_ply = paths["workspace"] / "point_cloud" / f"iteration_{int(spec['finetune_iteration'])}_object_inpaint" / "point_cloud.ply"
    save_scene_snapshot(
        paths["scene_out"],
        final_ply,
        cfg_args_path=paths["workspace"] / "cfg_args",
        state={
            "round_index": round_index,
            "target_id": spec["target_id"],
            "finetune_iteration": int(spec["finetune_iteration"]),
            "workspace_model_path": str(paths["workspace"]),
        },
    )
    meta["status"] = "completed"
    meta["scene_out_snapshot"] = str(paths["scene_out"])
    meta["final_point_cloud_ply"] = str(paths["scene_out"] / "point_cloud.ply")
    save_round_meta(paths, meta)


def run_status(args: argparse.Namespace, workflow: dict[str, Any]) -> None:
    iterative_root = ensure_workflow_initialized(args, workflow)
    print(f"Iterative root: {iterative_root}")
    for round_index in range(len(workflow["rounds"])):
        spec = resolve_round_spec(workflow, round_index)
        round_dir = get_round_dir(iterative_root, round_index, spec["target_id"])
        meta_path = get_round_meta_path(round_dir)
        status = read_json(meta_path, default={}).get("status", "not_initialized") if meta_path.is_file() else "not_initialized"
        print(f"[round {round_index:03d}] target={normalize_target_id(spec['target_id'])} status={status}")


def main() -> None:
    args = parse_args()
    workflow = load_workflow_config(args.workflow_config)
    if args.command == "init":
        iterative_root = ensure_workflow_initialized(args, workflow)
        print(f"Initialized workflow root: {iterative_root}")
        return
    if args.command == "status":
        run_status(args, workflow)
        return
    if args.round_index is None:
        raise ValueError(f"--round_index is required for command: {args.command}")
    if args.command == "prepare-round":
        run_prepare_round(args, workflow, args.round_index)
    elif args.command == "run-leftrefill":
        run_leftrefill(args, workflow, args.round_index)
    elif args.command == "initialize-round":
        run_initialize_round(args, workflow, args.round_index)
    elif args.command == "finalize-round":
        run_finalize_round(args, workflow, args.round_index)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
