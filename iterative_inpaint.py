from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.inpaint_target_paths import (
    ensure_dir,
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


IMAGE_EXTENSIONS = (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Round-based iterative inpaint workflow for my-AuraFusion360.")
    parser.add_argument("command", choices=["init", "prepare-round", "run-leftrefill", "finalize-round", "status"])
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
    assert_not_source_images(args.source_path, removal_renders, "Initial inpaint image root")
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
        str(removal_renders),
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
    if len(spec["target_id"]) > 1:
        init_cmd.append("--keep_all_unseen_components")
    run_cmd(init_cmd)

    meta["status"] = "initial_inpaint_ready"
    save_round_meta(paths, meta)


def run_leftrefill(args: argparse.Namespace, workflow: dict[str, Any], round_index: int) -> None:
    spec = resolve_round_spec(workflow, round_index)
    paths = round_paths(args, workflow, spec, round_index)
    meta = load_round_meta(paths, base_meta(args, round_index, spec, paths))
    ref_img_path = find_reference_image_from_removal(paths, spec, args)
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
    meta["status"] = "leftrefill_ready"
    meta["reference_image"] = str(ref_img_path)
    save_round_meta(paths, meta)


def run_finalize_round(args: argparse.Namespace, workflow: dict[str, Any], round_index: int) -> None:
    spec = resolve_round_spec(workflow, round_index)
    paths = round_paths(args, workflow, spec, round_index)
    meta = load_round_meta(paths, base_meta(args, round_index, spec, paths))
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
    elif args.command == "finalize-round":
        run_finalize_round(args, workflow, args.round_index)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
