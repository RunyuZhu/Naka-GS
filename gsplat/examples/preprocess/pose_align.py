import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pycolmap

try:
    from pycolmap import SceneManager  # type: ignore
except ImportError:  # pragma: no cover - depends on local pycolmap install
    SceneManager = None


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


def _build_unique_map(entries: List[Tuple[object, str]]) -> Dict[object, str]:
    counts: Dict[object, int] = {}
    for key, _ in entries:
        counts[key] = counts.get(key, 0) + 1
    return {key: value for key, value in entries if counts[key] == 1}


def _match_camera_names(
    src_names: List[str], dst_names: List[str]
) -> List[Tuple[str, str]]:
    dst_by_name = _build_unique_map([(Path(name).name.lower(), name) for name in dst_names])

    dst_stems: List[Tuple[str, str]] = []
    for name in dst_names:
        for stem in _stem_variants(name):
            dst_stems.append((stem, name))
    dst_by_stem = _build_unique_map(dst_stems)

    dst_indices: List[Tuple[int, str]] = []
    for name in dst_names:
        index = _extract_image_index(name)
        if index is not None:
            dst_indices.append((index, name))
    dst_by_index = _build_unique_map(dst_indices)

    matches: List[Tuple[str, str]] = []
    used_dst = set()
    for src_name in src_names:
        dst_name = dst_by_name.get(Path(src_name).name.lower())
        if dst_name is None:
            for stem in _stem_variants(src_name):
                dst_name = dst_by_stem.get(stem)
                if dst_name is not None:
                    break
        if dst_name is None:
            index = _extract_image_index(src_name)
            if index is not None:
                dst_name = dst_by_index.get(index)
        if dst_name is not None and dst_name not in used_dst:
            matches.append((src_name, dst_name))
            used_dst.add(dst_name)
    return matches


def load_camera_centers_from_colmap(sparse_dir: str) -> Dict[str, np.ndarray]:
    if not os.path.isdir(sparse_dir):
        raise FileNotFoundError(f"Sparse dir does not exist: {sparse_dir}")

    centers: Dict[str, np.ndarray] = {}
    if SceneManager is not None:
        manager = SceneManager(sparse_dir)
        manager.load_cameras()
        manager.load_images()
        for image in manager.images.values():
            centers[image.name] = np.asarray(image.C(), dtype=np.float64)
        return centers

    reconstruction = pycolmap.Reconstruction(sparse_dir)
    bottom = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64).reshape(1, 4)
    for image in reconstruction.images.values():
        cam_from_world = (
            image.cam_from_world()
            if callable(image.cam_from_world)
            else image.cam_from_world
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
        centers[image.name] = np.linalg.inv(w2c)[:3, 3]
    return centers


def _estimate_similarity_transform(
    src_points: np.ndarray, dst_points: np.ndarray, with_scale: bool
) -> Tuple[float, np.ndarray, np.ndarray]:
    if src_points.shape != dst_points.shape:
        raise ValueError(
            f"src/dst shape mismatch: {src_points.shape} vs {dst_points.shape}"
        )
    if src_points.shape[0] < 3:
        raise ValueError(
            f"Need at least 3 correspondences to estimate transform, got {src_points.shape[0]}"
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

    scale = 1.0
    if with_scale:
        src_var = (src_centered**2).sum() / src_points.shape[0]
        if src_var < 1e-12:
            raise ValueError("Degenerate source points: variance too small.")
        scale = float((S * np.diag(det_fix)).sum() / src_var)

    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def _rmse(src_points: np.ndarray, dst_points: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((src_points - dst_points) ** 2, axis=-1))))


def compose_sim3_transform(scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}")
    return points @ transform[:3, :3].T + transform[:3, 3]


def estimate_camera_center_transform(
    mvs_centers: Dict[str, np.ndarray],
    gt_centers: Dict[str, np.ndarray],
    align_mode: str = "sim3",
) -> Dict[str, object]:
    if align_mode not in {"sim3", "rigid", "none"}:
        raise ValueError(f"Unsupported align_mode={align_mode}")

    matches = _match_camera_names(list(mvs_centers.keys()), list(gt_centers.keys()))
    if align_mode != "none" and len(matches) < 3:
        raise ValueError(
            f"Need at least 3 matched cameras for alignment, got {len(matches)}"
        )

    if len(matches) == 0:
        identity = np.eye(4, dtype=np.float64)
        return {
            "transform": identity,
            "scale": 1.0,
            "rotation": np.eye(3, dtype=np.float64),
            "translation": np.zeros(3, dtype=np.float64),
            "matches": 0,
            "rmse_before": float("nan"),
            "rmse_after": float("nan"),
            "matched_names": [],
        }

    src = np.stack([mvs_centers[src_name] for src_name, _ in matches], axis=0)
    dst = np.stack([gt_centers[dst_name] for _, dst_name in matches], axis=0)
    rmse_before = _rmse(src, dst)

    if align_mode == "none":
        scale = 1.0
        rotation = np.eye(3, dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)
    else:
        scale, rotation, translation = _estimate_similarity_transform(
            src, dst, with_scale=(align_mode == "sim3")
        )
    transform = compose_sim3_transform(scale, rotation, translation)
    aligned = transform_points(src, transform)
    rmse_after = _rmse(aligned, dst)

    return {
        "transform": transform,
        "scale": scale,
        "rotation": rotation,
        "translation": translation,
        "matches": len(matches),
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "matched_names": matches,
    }
