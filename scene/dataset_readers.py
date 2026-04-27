#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

IMAGE_EXTENSIONS = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_mask: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly_modi(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    colors = np.zeros_like(colors) # zeros
    normals = np.zeros_like(colors) # zeros
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def read_pfm(file_path, process=True):
    with open(file_path, 'rb') as f:
        # read file head
        header = f.readline().rstrip().decode('utf-8')
        if header != 'Pf':
            raise Exception('Invalid PFM file.'+header)

        # read w h
        width, height = map(int, f.readline().rstrip().split())

        # read scale: positive-small，negative-big
        scale = float(f.readline().rstrip())

        # read binary data
        data = np.fromfile(f, '<f')

    # transfer to 2d array
    image = np.reshape(data, (height, width))

    # modify
    image = np.flipud(image) * scale

    if process:
        mi, ma = np.percentile(image, 0.01), np.percentile(image, 99.9)
        image[image<mi] = mi
        image[image>ma] = ma
        image += -mi
        image = image / image.max()
    return image

def _resolve_optional_dir(path_value):
    if path_value is None:
        return None
    path_value = str(path_value)
    if path_value == "":
        return None
    return os.path.abspath(os.path.expanduser(path_value))

def _find_image_for_stem(directory, stem):
    if directory is None:
        return None
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(directory, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None

def _mask_dir_for_stage(images_folder, stage, object_mask_dir=None, unseen_mask_dir=None, unseen_mask_dilated_dir=None):
    if stage in ["train", "removal"]:
        return _resolve_optional_dir(object_mask_dir) or os.path.join(os.path.dirname(images_folder), "object_masks")
    if stage == "inpaint":
        return _resolve_optional_dir(unseen_mask_dir) or _resolve_optional_dir(unseen_mask_dilated_dir) or os.path.join(os.path.dirname(images_folder), "unseen_masks")
    raise ValueError(f"stage {stage} not supported")

def _load_binary_mask(mask_path, stage):
    if mask_path is None or not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    image_mask = np.array(Image.open(mask_path).convert("L"))
    threshold = 10 if stage == "removal" else 127
    mask_array = np.where(image_mask > threshold, 1, 0)
    return Image.fromarray((mask_array * 255).astype(np.uint8))

def _maybe_use_reference_image(image_path, images_folder, reference_dir=None, use_reference_images=False):
    if not use_reference_images:
        return image_path
    image_stem = os.path.basename(image_path).split(".")[0]
    reference_dir = _resolve_optional_dir(reference_dir)
    if reference_dir is None:
        reference_dir = os.path.join(os.path.dirname(image_path.replace(f"{os.path.basename(images_folder)}", "reference")))
    reference_path = _find_image_for_stem(reference_dir, image_stem)
    return reference_path or image_path

def readColmapCameras(
    cam_extrinsics, cam_intrinsics, images_folder, stage, test_images_folder, object_mask_dir=None, unseen_mask_dir=None, unseen_mask_dilated_dir=None, reference_dir=None, use_reference_images=False
):
    train_cam_infos = []
    test_cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {} - extrinsics {}".format(idx + 1, len(cam_extrinsics))
        )
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        train_image_path = os.path.join(images_folder, os.path.basename(extr.name))
        test_image_path = os.path.join(test_images_folder, os.path.basename(extr.name))
        # handle for inpaint_images
        if stage == "inpaint":
            for ext in IMAGE_EXTENSIONS:
                train_image_path = os.path.join(images_folder, os.path.basename(extr.name).split(".")[0] + ext)
                if os.path.exists(train_image_path):
                    break
            for ext in IMAGE_EXTENSIONS:
                test_image_path = os.path.join(test_images_folder, os.path.basename(extr.name).split(".")[0] + ext)
                if os.path.exists(test_image_path):
                    break
        # check if image exists in training image folder
        if os.path.exists(train_image_path):
            image_path = train_image_path
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path)

            mask_dir = _mask_dir_for_stage(images_folder, stage, object_mask_dir, unseen_mask_dir, unseen_mask_dilated_dir)
            mask_stem = os.path.splitext(os.path.basename(extr.name))[0]
            image_mask_path = _find_image_for_stem(mask_dir, mask_stem)
            image_mask = _load_binary_mask(image_mask_path, stage)

            # handle reference image for inpaint stage
            if stage == "inpaint":
                image_path = _maybe_use_reference_image(image_path, images_folder, reference_dir, use_reference_images)
                image_name = os.path.basename(image_path).split(".")[0]
                image = Image.open(image_path)

            cam_info = CameraInfo(
                uid=uid,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_mask=image_mask,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
            )
            train_cam_infos.append(cam_info)
        # check if image exists in testing image folder
        elif os.path.exists(test_image_path):
            image_path = test_image_path
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path)

            cam_info = CameraInfo(
                uid=uid,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_mask=None,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
            )
            test_cam_infos.append(cam_info)
        else:
            # raise ValueError(f"Image: {image_name} not found in train / test")
            print(f"Image: {extr.name} not found in train / test")
            continue

    sys.stdout.write('\n')
    return train_cam_infos, test_cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8, stage="train", args=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # load training images from {images}, and testing images from {test_images}
    reading_dir = "images" if images == None else images
    train_cam_infos_unsorted, test_cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        stage=stage,
        test_images_folder=os.path.join(path, "test_images"),
        object_mask_dir=getattr(args, "object_mask_dir", ""),
        unseen_mask_dir=getattr(args, "unseen_mask_dir", ""),
        unseen_mask_dilated_dir=getattr(args, "unseen_mask_dilated_dir", ""),
        reference_dir=getattr(args, "reference_dir", ""),
        use_reference_images=getattr(args, "use_reference_images", False),
    )

    train_cam_infos = sorted(
        train_cam_infos_unsorted.copy(), key=lambda x: x.image_name
    )
    test_cam_infos = sorted(test_cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    print(f"Train cameras: {len(train_cam_infos)}, Test cameras: {len(test_cam_infos)}")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readColmapCamerasAura(
    cam_extrinsics, cam_intrinsics, images_folder, stage, test_images_folder, object_mask_dir=None, unseen_mask_dir=None, unseen_mask_dilated_dir=None, reference_dir=None, use_reference_images=False
):
    train_cam_infos = []
    test_cam_infos = []

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {} - extrinsics {}".format(idx + 1, len(cam_extrinsics))
        )
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        train_image_path = os.path.join(images_folder, os.path.basename(extr.name))
        test_image_path = os.path.join(test_images_folder, os.path.basename(extr.name))
        # handle inpaint_images for inpaint stage
        if stage == "inpaint":
            for ext in IMAGE_EXTENSIONS:
                train_image_path = os.path.join(images_folder, os.path.basename(extr.name).split(".")[0] + ext)
                if os.path.exists(train_image_path):
                    break
            for ext in IMAGE_EXTENSIONS:
                test_image_path = os.path.join(test_images_folder, os.path.basename(extr.name).split(".")[0] + ext)
                if os.path.exists(test_image_path):
                    break
        # check if image exists in training image folder
        if os.path.exists(train_image_path):
            image_path = train_image_path
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path)

            mask_dir = _mask_dir_for_stage(images_folder, stage, object_mask_dir, unseen_mask_dir, unseen_mask_dilated_dir)
            mask_stem = os.path.splitext(os.path.basename(extr.name))[0]
            image_mask_path = _find_image_for_stem(mask_dir, mask_stem)
            image_mask = _load_binary_mask(image_mask_path, stage)

            # handle reference image for inpaint stage
            if stage == "inpaint":
                image_path = _maybe_use_reference_image(image_path, images_folder, reference_dir, use_reference_images)
                image_name = os.path.basename(image_path).split(".")[0]
                image = Image.open(image_path)
            
            cam_info = CameraInfo(
                uid=uid,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_mask=image_mask,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
            )
            train_cam_infos.append(cam_info)
        # check if image exists in testing image folder
        elif os.path.exists(test_image_path):
            image_path = test_image_path
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path)

            cam_info = CameraInfo(
                uid=uid,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_mask=None,
                image_path=image_path,
                image_name=image_name,
                width=width,
                height=height,
            )
            test_cam_infos.append(cam_info)
        else:
            # raise ValueError(f"Image: {image_name} not found in train / test")
            print(f"Image: {extr.name} not found in train / test")
            continue

    sys.stdout.write('\n')
    return train_cam_infos, test_cam_infos


def readColmapSceneInfoAura(path, images, eval, llffhold=8, stage="train", args=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # load training images from {images}, and testing images from {test_images}
    reading_dir = "images" if images == None else images
    train_cam_infos_unsorted, test_cam_infos_unsorted = readColmapCamerasAura(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        stage=stage,
        test_images_folder=os.path.join(path, "test_images"),
        object_mask_dir=getattr(args, "object_mask_dir", ""),
        unseen_mask_dir=getattr(args, "unseen_mask_dir", ""),
        unseen_mask_dilated_dir=getattr(args, "unseen_mask_dilated_dir", ""),
        reference_dir=getattr(args, "reference_dir", ""),
        use_reference_images=getattr(args, "use_reference_images", False),
    )

    train_cam_infos = sorted(
        train_cam_infos_unsorted.copy(), key=lambda x: x.image_name
    )
    test_cam_infos = sorted(test_cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    print(f"Train cameras: {len(train_cam_infos)}, Test cameras: {len(test_cam_infos)}")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readColmapCamerasSpin(cam_extrinsics, cam_intrinsics, images_folder, load_mask=False, load_depth=False, load_norm=False, load_midas=False, stage="train"):

    print('readColmapCameras: load_norm ', load_norm) # True
    print('readColmapCameras: load_midas ', load_midas) # False
    all_names = [i[:-4]+'.png' for i in sorted(os.listdir(images_folder))[40:] if i.endswith('jpg')] # all training names actually
    # all_names = sorted(os.listdir(images_folder))

    images_folder_test = '/'.join(images_folder.split('/')[:-1]) + '/' + 'images_4'
    all_names_test = sorted(os.listdir(images_folder_test))
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        # if intr.model=="SIMPLE_PINHOLE":
        print(' intr.model: ', intr.model)
        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            cx = intr.params[1]
            cy = intr.params[2]

        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        # print(f'FovX: {FovX}, FovY: {FovY}')

        # image_path = os.path.join(images_folder, os.path.basename(extr.name))
        if os.path.basename(extr.name) in all_names or os.path.basename(extr.name)[:-4]+'.png' in all_names:
            # image_path = os.path.join(images_folder, os.path.basename(extr.name)[:-4]+'.png')
            image_path = os.path.join(images_folder_test, os.path.basename(extr.name)[:-4]+'.png')
        elif os.path.basename(extr.name) in all_names_test or os.path.basename(extr.name)[:-4]+'.png' in all_names_test:
            image_path = os.path.join(images_folder_test, os.path.basename(extr.name)[:-4]+'.png')
        else:
            print('\nskip img %d, %s, %s '%(idx, key, os.path.basename(extr.name)))
            # continue
        
        # handle for reference image
        if stage == "inpaint": # and reference_img_path is not None:
            reference_img_path = image_path.replace(f"{os.path.basename(images_folder_test)}", "reference")
            for ext in [".jpg", ".JPG", ".png", ".PNG"]:
                reference_img_path = os.path.join(os.path.dirname(reference_img_path), os.path.basename(reference_img_path).split(".")[0] + ext)
                if os.path.exists(reference_img_path):
                    image_path = reference_img_path
                    break
        
        # added
        cx = (cx - width / 2) / width * 2
        cy = (cy - height / 2) / height * 2
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        # image = Image.open(image_path[:-4]+'.jpg')

        # load mask
        mask_train_path = os.path.join(images_folder_test, '../lama_input', os.path.basename(extr.name)[:-4]+'_mask.png')
        mask_test_path = os.path.join(images_folder_test, 'mask_test', os.path.basename(extr.name)[:-4]+'.png')
        if os.path.exists(mask_train_path):
            mask = Image.open(mask_train_path)
        else:
            mask = Image.open(mask_test_path)


        # TODO: load mono depth for all views because we need it for spin-nerf dataset 
        if not load_midas: # here. 
            # # marigold depth loader
            midas_depth_path = os.path.join(images_folder_test, '../marigold_output/depth_npy/', os.path.basename(extr.name)[:-4]+'_pred.npy')
            assert os.path.exists(midas_depth_path), midas_depth_path
            midas_depth = np.load(midas_depth_path) # # (567, 1008) float32 -11021.697 285.07892
            
            midas_depth = Image.fromarray(midas_depth)
        else:
            # midas depth loader
            midas_depth_path = os.path.join(images_folder_test, '../midas_output', os.path.basename(extr.name)[:-4]+'-dpt_beit_large_512.pfm')
            assert os.path.exists(midas_depth_path), midas_depth_path
            midas_depth = read_pfm(midas_depth_path) # # (567, 1008) float32 -11021.

            midas_depth = Image.fromarray(midas_depth)


        # load normal
        # pass
        # omni_normal_path = os.path.join(images_folder_test, '../normal_output', os.path.basename(extr.name)[:-4]+'_normal.npy')
        # assert os.path.exists(omni_normal_path), omni_normal_path
        # omni_normal = np.load(omni_normal_path) # # (3, 384, 384) float32 0 1
        omni_normal = np.float32(np.random.rand(3,384,384)) # fake one normal

        # cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, cx=cx, cy=cy, image=image, mask=mask, depth=midas_depth, normal=omni_normal,
        #                     image_path=image_path, image_name=image_name, width=width, height=height)
        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_mask=mask,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            depth=midas_depth
        )

        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


# for spinnerf dataset with ply from Gscream
def readColmapSceneInfoSpin(path, images, eval, specified_ply_path, load_mask, load_depth, load_norm, load_midas, is_spin, llffhold=8, stage="train"):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    print('Reading cameras ', os.path.join(path, reading_dir))
    cam_infos_unsorted = readColmapCamerasSpin(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), load_mask=load_mask, load_depth=load_depth, load_norm=load_norm, load_midas=load_midas, stage=stage)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    print('self.spin, self.eval: ', is_spin, eval)
    
    if is_spin:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx >= 40] 
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx < 40] 

    else:
        if eval:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "sparse/0/points3D.ply") if specified_ply_path is None else specified_ply_path
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    # try:
    #     # pcd = fetchPly(ply_path)
    #     print('Using fetchPly_modi. ', ply_path)
    #     pcd = fetchPly_modi(ply_path)
    #     print('Finishing fetchPly_modi. ', ply_path)
    # except:
    #     pcd = None
    
    print('Using fetchPly_modi. ', ply_path)
    pcd = fetchPly_modi(ply_path)
    print('Finishing fetchPly_modi. ', ply_path)


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Aura": readColmapSceneInfoAura,
    "Spin": readColmapSceneInfoSpin,
}
