from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

from utils.inpaint_target_paths import ensure_dir, normalize_target_id


def read_json(path: str | Path, default: Any | None = None) -> Any:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        if default is not None:
            return default
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> Path:
    path = Path(path).expanduser().resolve()
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return path


def normalize_id_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            return normalize_id_list(json.loads(text))
        return [int(item) for item in text.replace(",", " ").split()]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [int(item) for item in value]
    return [int(value)]


def resolve_path(path_value: str | Path, base_dir: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(base_dir).expanduser().resolve() / path).resolve()


def get_iterative_root(model_path: str | Path) -> Path:
    return Path(model_path).expanduser().resolve() / "aura_iterative"


def get_rounds_root(iterative_root: str | Path) -> Path:
    return Path(iterative_root).expanduser().resolve() / "rounds"


def format_round_name(round_index: int, target_id) -> str:
    return f"{round_index:03d}_obj_{normalize_target_id(target_id)}"


def get_round_dir(iterative_root: str | Path, round_index: int, target_id) -> Path:
    return get_rounds_root(iterative_root) / format_round_name(round_index, target_id)


def get_round_workspace(round_dir: str | Path) -> Path:
    return Path(round_dir).expanduser().resolve() / "workspace"


def get_round_meta_dir(round_dir: str | Path) -> Path:
    return Path(round_dir).expanduser().resolve() / "meta"


def get_round_meta_path(round_dir: str | Path) -> Path:
    return get_round_meta_dir(round_dir) / "round_meta.json"


def get_round_scene_in_dir(round_dir: str | Path) -> Path:
    return Path(round_dir).expanduser().resolve() / "scene_in"


def get_round_scene_out_dir(round_dir: str | Path) -> Path:
    return Path(round_dir).expanduser().resolve() / "scene_out"


def remove_path(path: str | Path) -> None:
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def copy_file(src_path: str | Path, dst_path: str | Path, prefer_hardlink: bool = False) -> Path:
    src_path = Path(src_path).expanduser().resolve()
    dst_path = Path(dst_path).expanduser().resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    ensure_dir(dst_path.parent)
    if src_path == dst_path:
        return dst_path
    if prefer_hardlink:
        try:
            if dst_path.exists() or dst_path.is_symlink():
                dst_path.unlink()
            os.link(src_path, dst_path)
            return dst_path
        except OSError:
            pass
    shutil.copy2(src_path, dst_path)
    return dst_path


def bootstrap_workspace_from_base_model(
    base_model_path: str | Path,
    workspace_model_path: str | Path,
    iteration: int,
    prefer_hardlink: bool = False,
) -> dict[str, Any]:
    base_model_path = Path(base_model_path).expanduser().resolve()
    workspace_model_path = ensure_dir(workspace_model_path)
    source_iteration_dir = base_model_path / "point_cloud" / f"iteration_{int(iteration)}"
    target_iteration_dir = ensure_dir(workspace_model_path / "point_cloud" / "iteration_0")
    copy_file(source_iteration_dir / "point_cloud.ply", target_iteration_dir / "point_cloud.ply", prefer_hardlink)
    cfg_args = base_model_path / "cfg_args"
    if cfg_args.is_file():
        copy_file(cfg_args, workspace_model_path / "cfg_args")
    manifest = {
        "source_type": "base_model",
        "source_model_path": str(base_model_path),
        "source_iteration": int(iteration),
        "workspace_iteration": 0,
    }
    write_json(workspace_model_path / "scene_state_bootstrap.json", manifest)
    return manifest


def bootstrap_workspace_from_snapshot(
    snapshot_dir: str | Path,
    workspace_model_path: str | Path,
    prefer_hardlink: bool = False,
) -> dict[str, Any]:
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    workspace_model_path = ensure_dir(workspace_model_path)
    target_iteration_dir = ensure_dir(workspace_model_path / "point_cloud" / "iteration_0")
    copy_file(snapshot_dir / "point_cloud.ply", target_iteration_dir / "point_cloud.ply", prefer_hardlink)
    cfg_args = snapshot_dir / "cfg_args"
    if cfg_args.is_file():
        copy_file(cfg_args, workspace_model_path / "cfg_args")
    manifest = {
        "source_type": "previous_round_snapshot",
        "source_snapshot_dir": str(snapshot_dir),
        "workspace_iteration": 0,
    }
    write_json(workspace_model_path / "scene_state_bootstrap.json", manifest)
    return manifest


def save_scene_snapshot(
    snapshot_dir: str | Path,
    point_cloud_ply: str | Path,
    cfg_args_path: str | Path | None = None,
    state: dict[str, Any] | None = None,
    prefer_hardlink: bool = False,
) -> Path:
    snapshot_dir = ensure_dir(snapshot_dir)
    copy_file(point_cloud_ply, snapshot_dir / "point_cloud.ply", prefer_hardlink)
    if cfg_args_path is not None and Path(cfg_args_path).is_file():
        copy_file(cfg_args_path, snapshot_dir / "cfg_args")
    write_json(snapshot_dir / "state.json", state or {})
    return snapshot_dir
