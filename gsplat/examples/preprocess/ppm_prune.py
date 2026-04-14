import json
import os
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .pose_align import (
    estimate_camera_center_transform,
    load_camera_centers_from_colmap,
    transform_points,
)


_PLY_DTYPE_MAP = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def _resolve_path(data_dir: str, path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(data_dir, path)


def _coerce_colors(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"Expected colors with shape [N, 3], got {colors.shape}")
    if colors.dtype.kind == "f":
        if colors.max() <= 1.0 + 1e-6:
            colors = colors * 255.0
        colors = np.clip(np.round(colors), 0.0, 255.0).astype(np.uint8)
    else:
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    return colors


def _coerce_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}")
    return points


def _load_ply(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header is incomplete: {path}")
            header.append(line.decode("utf-8", "ignore").strip())
            if line.strip() == b"end_header":
                break

        fmt = None
        vertex_count = None
        in_vertex = False
        vertex_props = []
        for line in header:
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element "):
                tokens = line.split()
                in_vertex = len(tokens) >= 3 and tokens[1] == "vertex"
                if in_vertex:
                    vertex_count = int(tokens[2])
            elif in_vertex and line.startswith("property "):
                tokens = line.split()
                if len(tokens) != 3 or tokens[1] == "list":
                    raise ValueError(f"Unsupported PLY property line in {path}: {line}")
                vertex_props.append((tokens[2], tokens[1]))

        if fmt is None or vertex_count is None or len(vertex_props) == 0:
            raise ValueError(f"Failed to parse PLY header: {path}")

        dtype = np.dtype(
            [(name, _PLY_DTYPE_MAP[prop_type]) for name, prop_type in vertex_props]
        )

        if fmt == "binary_little_endian":
            data = np.fromfile(f, dtype=dtype, count=vertex_count)
        elif fmt == "ascii":
            data = np.loadtxt(f, dtype=dtype, max_rows=vertex_count)
        else:
            raise ValueError(f"Unsupported PLY format {fmt} in {path}")

    xyz_names = [name for name in ("x", "y", "z") if name in data.dtype.names]
    if len(xyz_names) != 3:
        raise ValueError(f"PLY is missing xyz properties: {path}")
    points = np.stack([data[name] for name in xyz_names], axis=-1).astype(np.float32)

    color_candidates = [
        ("red", "green", "blue"),
        ("r", "g", "b"),
    ]
    colors = None
    for names in color_candidates:
        if all(name in data.dtype.names for name in names):
            colors = np.stack([data[name] for name in names], axis=-1)
            break
    if colors is not None:
        colors = _coerce_colors(colors)
    return points, colors


def _load_npy_or_npz(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError(f"Unsupported ndarray shape in {path}: {data.shape}")
        if data.shape[1] == 3:
            return _coerce_points(data), None
        if data.shape[1] >= 6:
            return _coerce_points(data[:, :3]), _coerce_colors(data[:, 3:6])
        raise ValueError(f"Unsupported ndarray shape in {path}: {data.shape}")

    keys = set(data.keys())
    point_keys = ["points", "xyz", "vertices"]
    color_keys = ["colors", "rgb"]
    points = None
    colors = None
    for key in point_keys:
        if key in keys:
            points = data[key]
            break
    for key in color_keys:
        if key in keys:
            colors = data[key]
            break
    if points is None:
        raise ValueError(f"Could not find points array in {path}; keys={sorted(keys)}")
    points = _coerce_points(points)
    if colors is not None:
        colors = _coerce_colors(colors)
    return points, colors


def load_dense_points(
    path: str, colors_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    suffix = Path(path).suffix.lower()
    if suffix == ".ply":
        points, colors = _load_ply(path)
    elif suffix in {".npy", ".npz"}:
        points, colors = _load_npy_or_npz(path)
    else:
        raise ValueError(f"Unsupported dense point file format: {path}")

    if colors is None and colors_path is not None:
        loaded = np.load(colors_path, allow_pickle=False)
        colors = _coerce_colors(loaded)

    if colors is None:
        raise ValueError(
            f"Dense point cloud {path} does not contain RGB, and no external RGB path was provided."
        )
    if len(points) != len(colors):
        raise ValueError(
            f"Point/color count mismatch in dense point cloud: {len(points)} vs {len(colors)}"
        )
    return points.astype(np.float32), colors.astype(np.uint8)


def save_points_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    arr = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    arr["x"] = points[:, 0]
    arr["y"] = points[:, 1]
    arr["z"] = points[:, 2]
    arr["red"] = colors[:, 0]
    arr["green"] = colors[:, 1]
    arr["blue"] = colors[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        arr.tofile(f)


def voxel_pool_points(
    points: np.ndarray, colors: np.ndarray, voxel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    if len(points) == 0:
        raise ValueError("Cannot voxel-pool an empty point cloud.")

    voxel_coords = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
    num_voxels = int(inverse.max()) + 1

    point_sums = np.stack(
        [np.bincount(inverse, weights=points[:, dim], minlength=num_voxels) for dim in range(3)],
        axis=-1,
    )
    color_sums = np.stack(
        [np.bincount(inverse, weights=colors[:, dim], minlength=num_voxels) for dim in range(3)],
        axis=-1,
    )
    counts = np.bincount(inverse, minlength=num_voxels).astype(np.float32)
    pooled_points = (point_sums / counts[:, None]).astype(np.float32)
    pooled_colors = np.clip(np.round(color_sums / counts[:, None]), 0, 255).astype(np.uint8)
    return pooled_points, pooled_colors


def _query_nearest_neighbor_distance(points: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError("scipy is required for distance-adaptive pruning.") from exc

    tree = cKDTree(points)
    try:
        distances, _ = tree.query(points, k=2, workers=-1)
    except TypeError:  # pragma: no cover - older scipy
        distances, _ = tree.query(points, k=2)
    return distances[:, 1].astype(np.float32)


def distance_adaptive_prune(
    points: np.ndarray,
    colors: np.ndarray,
    tau0: float,
    beta: float,
    iters: int,
    eps: float,
    min_points_after_prune: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    rng = np.random.default_rng(seed)
    total_init = len(points)
    current_points = points
    current_colors = colors
    tau = float(tau0)
    debug_iters = []

    for iter_idx in range(iters):
        num_before = len(current_points)
        if num_before <= 1:
            break

        nn_dist = _query_nearest_neighbor_distance(current_points)
        keep_prob = np.minimum(1.0, nn_dist / (tau + eps))
        keep_mask = rng.random(num_before) < keep_prob
        kept = int(keep_mask.sum())

        iter_stats = {
            "iter": iter_idx,
            "tau": tau,
            "points_before": num_before,
            "points_after": kept,
            "nn_dist_mean": float(nn_dist.mean()),
            "nn_dist_median": float(np.median(nn_dist)),
            "keep_prob_mean": float(keep_prob.mean()),
        }
        debug_iters.append(iter_stats)
        print(
            f"[PPM] Iter {iter_idx}: points {num_before} -> {kept}, "
            f"tau={tau:.6f}, mean_keep_prob={keep_prob.mean():.4f}"
        )

        if kept < min_points_after_prune:
            print(
                f"[PPM] Iter {iter_idx}: would fall below min_points_after_prune="
                f"{min_points_after_prune}, keeping previous result."
            )
            iter_stats["rolled_back"] = True
            break

        current_points = current_points[keep_mask]
        current_colors = current_colors[keep_mask]
        tau = tau * float(np.exp(beta * num_before / max(total_init, 1)))

    return current_points, current_colors, {"iters": debug_iters}


def run_ppm_preprocess(
    *,
    data_dir: str,
    dense_points_path: Optional[str],
    dense_points_rgb_path: Optional[str],
    gt_sparse_dir: Optional[str],
    mvs_sparse_dir: Optional[str],
    align_to_gt: bool,
    align_mode: str,
    pose_alignment_transform: np.ndarray,
    normalize_transform: np.ndarray,
    voxel_size: float,
    tau0: float,
    beta: float,
    iters: int,
    eps: float,
    min_points_after_prune: int,
    seed: int,
    save_debug: bool,
    save_pruned_ply: bool,
    result_dir: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    dense_points_path = _resolve_path(data_dir, dense_points_path)
    dense_points_rgb_path = _resolve_path(data_dir, dense_points_rgb_path)
    gt_sparse_dir = _resolve_path(data_dir, gt_sparse_dir)
    mvs_sparse_dir = _resolve_path(data_dir, mvs_sparse_dir)

    if dense_points_path is None:
        raise ValueError("ppm_enable=True requires ppm_dense_points_path.")

    points, colors = load_dense_points(dense_points_path, dense_points_rgb_path)
    print(f"[PPM] Loaded dense points: {len(points)} from {dense_points_path}")

    align_stats: Dict[str, object] = {
        "align_mode": align_mode,
        "sim3_scale": 1.0,
        "pose_rmse_before": float("nan"),
        "pose_rmse_after": float("nan"),
        "pose_match_count": 0,
    }

    if align_to_gt and align_mode != "none":
        if mvs_sparse_dir is None:
            warnings.warn(
                "ppm_align_to_gt=True but ppm_mvs_sparse_dir is missing; "
                "assuming dense points are already in GT coordinates."
            )
        elif gt_sparse_dir is None:
            raise ValueError("ppm_align_to_gt=True requires ppm_gt_sparse_dir.")
        else:
            gt_centers = load_camera_centers_from_colmap(gt_sparse_dir)
            mvs_centers = load_camera_centers_from_colmap(mvs_sparse_dir)
            pose_fit = estimate_camera_center_transform(
                mvs_centers, gt_centers, align_mode=align_mode
            )
            points = transform_points(points, pose_fit["transform"])
            align_stats.update(
                {
                    "sim3_scale": float(pose_fit["scale"]),
                    "pose_rmse_before": float(pose_fit["rmse_before"]),
                    "pose_rmse_after": float(pose_fit["rmse_after"]),
                    "pose_match_count": int(pose_fit["matches"]),
                    "rotation_det": float(np.linalg.det(pose_fit["rotation"])),
                }
            )
            print(
                f"[PPM] Pose alignment ({align_mode}) matched {pose_fit['matches']} cameras, "
                f"rmse {pose_fit['rmse_before']:.6f} -> {pose_fit['rmse_after']:.6f}, "
                f"scale={pose_fit['scale']:.6f}, det(R)={np.linalg.det(pose_fit['rotation']):.6f}"
            )
            if float(pose_fit["rmse_after"]) > 0.1:
                warnings.warn(
                    f"PPM pose alignment RMSE looks high: {pose_fit['rmse_after']:.6f}"
                )

    # Match the same final frame as Parser.points after pose replacement and normalization.
    points = transform_points(points, pose_alignment_transform)
    points = transform_points(points, normalize_transform)

    ppm_dir = os.path.join(result_dir, "ppm")
    if save_debug:
        os.makedirs(ppm_dir, exist_ok=True)
        save_points_ply(os.path.join(ppm_dir, "aligned_points.ply"), points, colors)

    voxel_points, voxel_colors = voxel_pool_points(points, colors, voxel_size)
    print(
        f"[PPM] Voxel pooling: {len(points)} -> {len(voxel_points)} "
        f"(voxel_size={voxel_size})"
    )
    pruned_points, pruned_colors, prune_debug = distance_adaptive_prune(
        voxel_points,
        voxel_colors,
        tau0=tau0,
        beta=beta,
        iters=iters,
        eps=eps,
        min_points_after_prune=min_points_after_prune,
        seed=seed,
    )
    print(f"[PPM] Final pruned points: {len(pruned_points)}")

    stats: Dict[str, object] = {
        "raw_points": int(len(points)),
        "voxel_points": int(len(voxel_points)),
        "final_points": int(len(pruned_points)),
        "tau0": float(tau0),
        "beta": float(beta),
        "iters": int(iters),
        "align_mode": align_stats["align_mode"],
        "sim3_scale": align_stats["sim3_scale"],
        "pose_rmse_before": align_stats["pose_rmse_before"],
        "pose_rmse_after": align_stats["pose_rmse_after"],
        "pose_match_count": align_stats["pose_match_count"],
        "dense_points_path": dense_points_path,
        "dense_points_rgb_path": dense_points_rgb_path,
        "gt_sparse_dir": gt_sparse_dir,
        "mvs_sparse_dir": mvs_sparse_dir,
        "prune_debug": prune_debug,
    }

    if save_debug:
        with open(os.path.join(ppm_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        if save_pruned_ply:
            save_points_ply(
                os.path.join(ppm_dir, "pruned_points.ply"), pruned_points, pruned_colors
            )

    return pruned_points.astype(np.float32), pruned_colors.astype(np.uint8), stats
