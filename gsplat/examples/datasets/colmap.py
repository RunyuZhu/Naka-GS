# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import pycolmap
import torch
from PIL import Image
from tqdm import tqdm
from typing_extensions import assert_never

from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

OPENGL_TO_OPENCV = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

try:
    # Legacy pycolmap fork API used by original gsplat examples.
    from pycolmap import SceneManager  # type: ignore
except Exception:
    SceneManager = None


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


def _resolve_path(data_dir: str, path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(data_dir, path)


def _extract_image_index(path: str) -> Optional[int]:
    stem = Path(path).stem
    tokens = re.findall(r"\d+", stem)
    if not tokens:
        return None
    return int(tokens[-1])


def _stem_variants(path: str) -> List[str]:
    stem = Path(path).stem.lower()
    variants = [stem]
    for suffix in ["_enhanced", "_image", "_img"]:
        if stem.endswith(suffix):
            variants.append(stem[: -len(suffix)])
    return variants


def _build_unique_map(entries: List[Tuple[Any, int]]) -> Dict[Any, int]:
    counts: Dict[Any, int] = {}
    for key, _ in entries:
        counts[key] = counts.get(key, 0) + 1
    return {key: idx for key, idx in entries if counts[key] == 1}


def _load_transforms_json(
    transforms_path: str, transforms_coord: str
) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
    with open(transforms_path) as f:
        transforms_data = json.load(f)
    if "frames" not in transforms_data:
        raise ValueError(f"`frames` is missing in transforms file: {transforms_path}")

    poses = []
    frame_paths: List[str] = []
    for frame in transforms_data["frames"]:
        if "transform_matrix" not in frame or "file_path" not in frame:
            continue
        pose = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(
                f"Expected transform_matrix to be 4x4, got {pose.shape} in {transforms_path}"
            )
        if transforms_coord == "opengl":
            pose = pose @ OPENGL_TO_OPENCV
        elif transforms_coord != "opencv":
            raise ValueError(
                f"Unsupported transforms_coord={transforms_coord}, choose from ['opengl', 'opencv']"
            )
        poses.append(pose)
        frame_paths.append(frame["file_path"])

    if len(poses) == 0:
        raise ValueError(f"No valid frames found in transforms file: {transforms_path}")

    intrinsics = {}
    for key in ["fl_x", "fl_y", "cx", "cy", "w", "h"]:
        if key in transforms_data:
            intrinsics[key] = float(transforms_data[key])

    return np.stack(poses, axis=0), frame_paths, intrinsics


def _match_image_names_to_frames(
    image_names: List[str], frame_paths: List[str]
) -> Dict[int, int]:
    by_name = _build_unique_map(
        [(Path(frame_path).name.lower(), i) for i, frame_path in enumerate(frame_paths)]
    )

    stem_entries: List[Tuple[str, int]] = []
    for i, frame_path in enumerate(frame_paths):
        for stem in _stem_variants(frame_path):
            stem_entries.append((stem, i))
    by_stem = _build_unique_map(stem_entries)

    index_entries: List[Tuple[int, int]] = []
    for i, frame_path in enumerate(frame_paths):
        index = _extract_image_index(frame_path)
        if index is not None:
            index_entries.append((index, i))
    by_index = _build_unique_map(index_entries)

    matches: Dict[int, int] = {}
    for image_idx, image_name in enumerate(image_names):
        name_key = Path(image_name).name.lower()
        frame_idx = by_name.get(name_key)
        if frame_idx is None:
            for stem_key in _stem_variants(image_name):
                frame_idx = by_stem.get(stem_key)
                if frame_idx is not None:
                    break
        if frame_idx is None:
            image_index = _extract_image_index(image_name)
            if image_index is not None:
                frame_idx = by_index.get(image_index)
        if frame_idx is not None:
            matches[image_idx] = frame_idx
    return matches


def _estimate_similarity_transform(
    src_points: np.ndarray, dst_points: np.ndarray
) -> np.ndarray:
    """Estimate similarity transform T such that dst ~= T(src)."""
    if src_points.shape != dst_points.shape:
        raise ValueError(
            f"src/dst shape mismatch: {src_points.shape} vs {dst_points.shape}"
        )
    if src_points.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 correspondences to estimate similarity, got {src_points.shape[0]}"
        )

    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)
    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    covariance = (dst_centered.T @ src_centered) / src_points.shape[0]
    U, S, Vt = np.linalg.svd(covariance)

    det_fix = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        det_fix[-1, -1] = -1.0
    rotation = U @ det_fix @ Vt

    src_var = (src_centered**2).sum() / src_points.shape[0]
    if src_var < 1e-12:
        raise ValueError("Degenerate source points: variance too small.")
    scale = float((S * np.diag(det_fix)).sum() / src_var)
    translation = dst_mean - scale * (rotation @ src_mean)

    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform


def _camera_model_name(camera: Any) -> str:
    model = None
    if hasattr(camera, "camera_type"):
        model = camera.camera_type
    elif hasattr(camera, "model"):
        model = camera.model
    if isinstance(model, str):
        return model
    name = getattr(model, "name", None)
    if isinstance(name, str):
        return name
    if model is None:
        return "UNKNOWN"
    model_str = str(model)
    if "." in model_str:
        return model_str.split(".")[-1]
    return model_str


def _camera_param_map(camera: Any) -> Dict[str, float]:
    params: Dict[str, float] = {}
    if hasattr(camera, "params") and hasattr(camera, "params_info"):
        info = str(camera.params_info)
        names = [name.strip() for name in info.split(",") if name.strip()]
        values = np.asarray(camera.params).reshape(-1).tolist()
        for name, value in zip(names, values):
            params[name] = float(value)
    for key in ["f", "fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4", "p1", "p2"]:
        if hasattr(camera, key):
            try:
                params[key] = float(getattr(camera, key))
            except Exception:
                pass
    if "fx" not in params and hasattr(camera, "focal_length_x"):
        params["fx"] = float(camera.focal_length_x)
    if "fy" not in params and hasattr(camera, "focal_length_y"):
        params["fy"] = float(camera.focal_length_y)
    if "cx" not in params and hasattr(camera, "principal_point_x"):
        params["cx"] = float(camera.principal_point_x)
    if "cy" not in params and hasattr(camera, "principal_point_y"):
        params["cy"] = float(camera.principal_point_y)
    return params


def _parse_camera(camera: Any, factor: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
    model_name = _camera_model_name(camera)
    param_map = _camera_param_map(camera)

    if "f" in param_map:
        fx = param_map.get("fx", param_map["f"])
        fy = param_map.get("fy", param_map["f"])
    else:
        fx = param_map.get("fx")
        fy = param_map.get("fy")
    if fx is None or fy is None:
        raise ValueError(
            f"Cannot parse focal length from camera model {model_name} with params {param_map}"
        )
    cx = param_map.get("cx")
    cy = param_map.get("cy")
    if cx is None or cy is None:
        raise ValueError(
            f"Cannot parse principal point from camera model {model_name} with params {param_map}"
        )

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    K[:2, :] /= factor

    if model_name in ["SIMPLE_PINHOLE", "PINHOLE"]:
        params = np.empty(0, dtype=np.float32)
        camtype = "perspective"
    elif model_name == "SIMPLE_RADIAL":
        params = np.array([param_map.get("k1", 0.0), 0.0, 0.0, 0.0], dtype=np.float32)
        camtype = "perspective"
    elif model_name == "RADIAL":
        params = np.array(
            [param_map.get("k1", 0.0), param_map.get("k2", 0.0), 0.0, 0.0],
            dtype=np.float32,
        )
        camtype = "perspective"
    elif model_name == "OPENCV":
        params = np.array(
            [
                param_map.get("k1", 0.0),
                param_map.get("k2", 0.0),
                param_map.get("p1", 0.0),
                param_map.get("p2", 0.0),
            ],
            dtype=np.float32,
        )
        camtype = "perspective"
    elif model_name == "OPENCV_FISHEYE":
        params = np.array(
            [
                param_map.get("k1", 0.0),
                param_map.get("k2", 0.0),
                param_map.get("k3", 0.0),
                param_map.get("k4", 0.0),
            ],
            dtype=np.float32,
        )
        camtype = "fisheye"
    else:
        # Fallback for newer/unhandled models: treat as pinhole-like without distortion.
        params = np.empty(0, dtype=np.float32)
        camtype = "perspective"
        print(
            f"Warning: camera model {model_name} is not explicitly handled, "
            "falling back to no-distortion perspective."
        )
    return K, params, camtype, model_name


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        load_exposure: bool = False,
        colmap_path: Optional[str] = None,
        transforms_train_path: Optional[str] = None,
        transforms_test_path: Optional[str] = None,
        pose_source: str = "colmap",
        transforms_coord: str = "opengl",
        use_transforms_intrinsics: bool = False,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.load_exposure = load_exposure
        self.pose_source = pose_source

        if pose_source not in ["colmap", "align", "replace"]:
            raise ValueError(
                f"Unsupported pose_source={pose_source}, choose from ['colmap', 'align', 'replace']"
            )

        colmap_dir = _resolve_path(data_dir, colmap_path)
        if colmap_dir is None:
            colmap_dir = os.path.join(data_dir, "sparse/0/")
            if not os.path.exists(colmap_dir):
                colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        legacy_manager = None
        reconstruction = None
        if SceneManager is not None:
            legacy_manager = SceneManager(colmap_dir)
            legacy_manager.load_cameras()
            legacy_manager.load_images()
            legacy_manager.load_points3D()
            imdata = legacy_manager.images
            cameras_data = legacy_manager.cameras
        else:
            reconstruction = pycolmap.Reconstruction(colmap_dir)
            imdata = reconstruction.images
            cameras_data = reconstruction.cameras

        # Extract extrinsic matrices in world-to-camera format.
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        camtypes_dict: Dict[int, str] = {}
        camera_models = set()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            if hasattr(im, "R") and hasattr(im, "tvec"):
                rot = np.asarray(im.R(), dtype=np.float64)
                trans = np.asarray(im.tvec, dtype=np.float64).reshape(3, 1)
                w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            else:
                cam_from_world = (
                    im.cam_from_world()
                    if callable(im.cam_from_world)
                    else im.cam_from_world
                )
                w2c_34 = np.asarray(cam_from_world.matrix(), dtype=np.float64)
                if w2c_34.shape == (3, 4):
                    w2c = np.concatenate([w2c_34, bottom], axis=0)
                elif w2c_34.shape == (4, 4):
                    w2c = w2c_34
                else:
                    raise ValueError(
                        f"Unexpected cam_from_world matrix shape: {w2c_34.shape}"
                    )
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = cameras_data[camera_id]
            K, params, camtype, model_name = _parse_camera(cam, factor=factor)
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {model_name}"

            Ks_dict[camera_id] = K
            params_dict[camera_id] = params
            imsize_dict[camera_id] = (int(cam.width) // factor, int(cam.height) // factor)
            mask_dict[camera_id] = None
            camtypes_dict[camera_id] = camtype
            camera_models.add(model_name)
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if any(
            model_name not in ["SIMPLE_PINHOLE", "PINHOLE"]
            for model_name in camera_models
        ):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Optional pose/intrinsics from transforms_*.json files.
        transforms_train_path = _resolve_path(data_dir, transforms_train_path)
        transforms_test_path = _resolve_path(data_dir, transforms_test_path)

        self.pose_match_count = 0
        pose_alignment_transform = np.eye(4)
        train_poses: Optional[np.ndarray] = None
        image_to_train_idx: Dict[int, int] = {}

        render_camtoworlds: Optional[np.ndarray] = None
        render_K: Optional[np.ndarray] = None
        render_imsize: Optional[Tuple[int, int]] = None
        render_frame_paths: Optional[List[str]] = None

        if transforms_train_path is not None:
            (
                train_poses,
                train_frame_paths,
                train_intrinsics,
            ) = _load_transforms_json(transforms_train_path, transforms_coord)
            image_to_train_idx = _match_image_names_to_frames(image_names, train_frame_paths)
            self.pose_match_count = len(image_to_train_idx)
            print(
                f"[Parser] Matched {self.pose_match_count}/{len(image_names)} COLMAP images "
                f"with {transforms_train_path}."
            )

            if use_transforms_intrinsics:
                intrinsics_keys = ["fl_x", "fl_y", "cx", "cy", "w", "h"]
                missing = [key for key in intrinsics_keys if key not in train_intrinsics]
                if missing:
                    raise ValueError(
                        f"Missing intrinsics keys {missing} in {transforms_train_path}"
                    )
                K_json = np.array(
                    [
                        [train_intrinsics["fl_x"], 0.0, train_intrinsics["cx"]],
                        [0.0, train_intrinsics["fl_y"], train_intrinsics["cy"]],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                K_json[:2, :] /= factor
                w_json = int(round(train_intrinsics["w"] / factor))
                h_json = int(round(train_intrinsics["h"] / factor))
                for camera_id in Ks_dict:
                    Ks_dict[camera_id] = K_json.copy()
                    # Use no distortion when overriding with transforms intrinsics.
                    params_dict[camera_id] = np.empty(0, dtype=np.float32)
                    imsize_dict[camera_id] = (w_json, h_json)

        if transforms_test_path is not None:
            (
                render_camtoworlds,
                render_frame_paths,
                test_intrinsics,
            ) = _load_transforms_json(transforms_test_path, transforms_coord)
            if all(key in test_intrinsics for key in ["fl_x", "fl_y", "cx", "cy"]):
                render_K = np.array(
                    [
                        [test_intrinsics["fl_x"], 0.0, test_intrinsics["cx"]],
                        [0.0, test_intrinsics["fl_y"], test_intrinsics["cy"]],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                render_K[:2, :] /= factor
            if all(key in test_intrinsics for key in ["w", "h"]):
                render_imsize = (
                    int(round(test_intrinsics["w"] / factor)),
                    int(round(test_intrinsics["h"] / factor)),
                )

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        if factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg":
            image_dir = _resize_image_folder(
                colmap_image_dir, image_dir + "_png", factor=factor
            )
            image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        point_indices = {}
        if legacy_manager is not None:
            points = legacy_manager.points3D.astype(np.float32)
            points_err = legacy_manager.point3D_errors.astype(np.float32)
            points_rgb = legacy_manager.point3D_colors.astype(np.uint8)

            image_id_to_name = {v: k for k, v in legacy_manager.name_to_image_id.items()}
            for point_id, data in legacy_manager.point3D_id_to_images.items():
                for image_id, _ in data:
                    image_name = image_id_to_name[image_id]
                    point_idx = legacy_manager.point3D_id_to_point3D_idx[point_id]
                    point_indices.setdefault(image_name, []).append(point_idx)
        else:
            assert reconstruction is not None
            point3d_ids = list(reconstruction.points3D.keys())
            point3d_list = [reconstruction.points3D[pid] for pid in point3d_ids]
            points = np.stack([p.xyz for p in point3d_list], axis=0).astype(np.float32)
            points_err = np.array([p.error for p in point3d_list], dtype=np.float32)
            points_rgb = np.stack([p.color for p in point3d_list], axis=0).astype(
                np.uint8
            )

            point3d_id_to_idx = {pid: idx for idx, pid in enumerate(point3d_ids)}
            for point3d_id, point3d in reconstruction.points3D.items():
                point_idx = point3d_id_to_idx[point3d_id]
                for elem in point3d.track.elements:
                    image_name = reconstruction.images[elem.image_id].name
                    point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Align or replace COLMAP poses with transforms_train.json poses if requested.
        if pose_source in ["align", "replace"]:
            if transforms_train_path is None or train_poses is None:
                raise ValueError(
                    f"pose_source={pose_source} requires transforms_train_path."
                )
            if len(image_to_train_idx) < 3:
                raise ValueError(
                    f"pose_source={pose_source} requires at least 3 matched images, got {len(image_to_train_idx)}."
                )

            matched_image_ids = sorted(image_to_train_idx.keys())
            src_centers = np.stack(
                [camtoworlds[idx, :3, 3] for idx in matched_image_ids], axis=0
            )
            dst_centers = np.stack(
                [train_poses[image_to_train_idx[idx], :3, 3] for idx in matched_image_ids],
                axis=0,
            )
            pose_alignment_transform = _estimate_similarity_transform(
                src_centers, dst_centers
            )
            camtoworlds = transform_cameras(pose_alignment_transform, camtoworlds)
            points = transform_points(pose_alignment_transform, points)

            aligned_centers = np.stack(
                [camtoworlds[idx, :3, 3] for idx in matched_image_ids], axis=0
            )
            center_err = np.linalg.norm(aligned_centers - dst_centers, axis=1)
            print(
                f"[Parser] Pose alignment center error mean={center_err.mean():.6f}, "
                f"max={center_err.max():.6f}"
            )
            print(f"[Parser] pose_alignment_transform:\n{pose_alignment_transform}")

            if pose_source == "replace":
                for image_idx, frame_idx in image_to_train_idx.items():
                    camtoworlds[image_idx] = train_poses[frame_idx]
                print(
                    f"[Parser] Replaced {len(image_to_train_idx)} train camera poses "
                    "with transforms_train poses."
                )
        elif transforms_test_path is not None:
            print(
                "[Parser] Warning: transforms_test_path is provided while pose_source='colmap'. "
                "If transforms_test is in a different world frame, render views may be misaligned."
            )

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)
            if render_camtoworlds is not None:
                render_camtoworlds = transform_cameras(T1, render_camtoworlds)

            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)
            if render_camtoworlds is not None:
                render_camtoworlds = transform_cameras(T2, render_camtoworlds)

            transform = T2 @ T1

            # Fix for up side down. We assume more points towards
            # the bottom of the scene which is true when ground floor is
            # present in the images.
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                # rotate 180 degrees around x axis such that z is flipped
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                if render_camtoworlds is not None:
                    render_camtoworlds = transform_cameras(T3, render_camtoworlds)
                transform = T3 @ transform
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)
        self.pose_alignment_transform = pose_alignment_transform  # np.ndarray, (4, 4)
        self.render_camtoworlds = render_camtoworlds  # Optional[np.ndarray], (N, 4, 4)
        self.render_K = render_K  # Optional[np.ndarray], (3, 3)
        self.render_imsize = render_imsize  # Optional[(width, height)]
        self.render_frame_paths = render_frame_paths  # Optional[List[str]]

        # Create 0-based contiguous camera indices from COLMAP camera_ids.
        # This is useful for camera-based embeddings/modules.
        unique_camera_ids = sorted(set(camera_ids))
        self.camera_id_to_idx = {cid: idx for idx, cid in enumerate(unique_camera_ids)}
        self.camera_indices = np.asarray(
            [self.camera_id_to_idx[cid] for cid in camera_ids], dtype=np.int64
        )
        self.num_cameras = len(unique_camera_ids)

        # Load EXIF exposure data if requested.
        # Always read from original (non-downscaled) images since PNG doesn't support EXIF.
        if load_exposure:
            from exif import compute_exposure_from_exif

            exposure_values: List[Optional[float]] = []
            for image_name in tqdm(image_names, desc="Loading EXIF exposure"):
                original_path = Path(colmap_image_dir) / image_name
                exposure_values.append(compute_exposure_from_exif(original_path))

            # Compute mean across all valid exposures and subtract
            valid_exposures = [e for e in exposure_values if e is not None]
            if valid_exposures:
                exposure_mean = sum(valid_exposures) / len(valid_exposures)
                self.exposure_values: List[Optional[float]] = [
                    (e - exposure_mean) if e is not None else None
                    for e in exposure_values
                ]
                print(
                    f"[Parser] Loaded exposure for {len(valid_exposures)}/{len(exposure_values)} images "
                    f"(mean={exposure_mean:.3f} EV)"
                )
            else:
                self.exposure_values = [None] * len(exposure_values)
                print("[Parser] No valid EXIF exposure data found in any image.")
        else:
            self.exposure_values = [None] * len(image_paths)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            camtype = camtypes_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
            "camera_idx": self.parser.camera_indices[
                index
            ],  # 0-based contiguous camera index
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        # Add exposure if available for this image
        exposure = self.parser.exposure_values[index]
        if exposure is not None:
            data["exposure"] = torch.tensor(exposure, dtype=torch.float32)

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
