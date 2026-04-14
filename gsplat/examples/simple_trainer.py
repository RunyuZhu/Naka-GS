# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import math
import os
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent
for _path in (str(REPO_ROOT), str(EXAMPLES_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Older local CUDA toolkits may not ship libcudacxx's <cuda/std/optional>.
# This trainer does not need 3DGUT unless the user explicitly opts in, so
# default to disabling that part of the extension build for compatibility.
os.environ.setdefault("BUILD_3DGUT", "0")
# Keep torch extension builds in a writable location so headless/sandboxed runs
# do not fail on a read-only home cache.
os.environ.setdefault(
    "TORCH_EXTENSIONS_DIR",
    os.path.join(tempfile.gettempdir(), "torch_extensions"),
)

from gsplat.color_correct import color_correct_affine, color_correct_quadratic
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from preprocess.ppm_prune import run_ppm_preprocess
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    knn,
    rgb_to_sh,
    set_random_seed,
)

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization, RasterizeMode
from gsplat.cuda._wrapper import CameraModel
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap

# -----------------------------------------------------------------------------
# Local quickstart defaults:
# For quick scene switching, usually only change `--scene` in command line.
# -----------------------------------------------------------------------------
LOCAL_DATA_ROOT = "/home/zhu/3r_test/3D/recon/single-mul_noLOL_HF_0.05sigma_MSE_mask_test/"
LOCAL_RESULTS_ROOT = "/home/zhu/3r_test/gsplat_v2/results/test"
LOCAL_DEFAULT_SCENE = "Ujikintoki"
LOCAL_DEFAULT_COLMAP_PATH = "sparse"


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Path to checkpoint(s) to resume training from.
    resume_ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path: "interp", "ellipse", "spiral", or "testjson".
    render_traj_path: str = "testjson"

    # Scene name under LOCAL_DATA_ROOT, e.g. BlueHawaii/GearWorks/Cupcake.
    scene: str = LOCAL_DEFAULT_SCENE
    # Path to scene root. If empty, auto use f"{LOCAL_DATA_ROOT}/{scene}".
    data_dir: str = ""
    # Optional COLMAP path. Can be absolute path or relative to data_dir.
    # Example: "sparse", "sparse/0", "ma_sparse/sparse"
    colmap_path: Optional[str] = LOCAL_DEFAULT_COLMAP_PATH
    # Optional train/test transforms json paths for pose alignment and rendering.
    # Can be absolute path or relative to data_dir.
    transforms_train_path: Optional[str] = "transforms_train.json"
    transforms_test_path: Optional[str] = "transforms_test.json"
    # How to use transforms_train_path:
    # - "colmap": use COLMAP poses as-is
    # - "align": align COLMAP world to transforms world by similarity transform
    # - "replace": align first, then replace matched train poses by transforms poses
    pose_source: Literal["colmap", "align", "replace"] = "replace"
    # Coordinate convention in transforms json.
    # "opengl" matches common NeRF transforms files.
    transforms_coord: Literal["opengl", "opencv"] = "opengl"
    # Override COLMAP intrinsics by transforms_train intrinsics.
    use_transforms_intrinsics: bool = True
    # Downsample factor for the dataset
    data_factor: int = 1
    # Directory to save results. If empty, auto use:
    # f"{LOCAL_RESULTS_ROOT}/{scene}_{colmap_path with /->_}_{pose_source}".
    result_dir: str = ""
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: CameraModel = "pinhole"
    # Load EXIF exposure metadata from images (if available)
    load_exposure: bool = False

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 8_000
    # Evaluate and save rendered images every N steps. Set <=0 to disable interval-based eval.
    eval_every_steps: int = 2_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [8_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [8_000])
    # Save checkpoint every N steps. Set <=0 to disable interval-based saving.
    save_every_steps: int = 2_000
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [8_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = True

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Replace default sparse initialization with a dense VGGT/MVS point cloud.
    ppm_enable: bool = False
    # Dense point cloud path (.ply/.npy/.npz), absolute or relative to data_dir.
    ppm_dense_points_path: Optional[str] = None
    # Optional external RGB path for dense points when RGB is not embedded in the file.
    ppm_dense_points_rgb_path: Optional[str] = None
    # GT sparse directory used for optional camera-center alignment.
    ppm_gt_sparse_dir: Optional[str] = None
    # Sparse directory that matches the dense point cloud coordinates.
    ppm_mvs_sparse_dir: Optional[str] = None
    # Estimate a global transform from ppm_mvs_sparse_dir cameras to GT cameras.
    ppm_align_to_gt: bool = True
    # Global alignment mode for the dense point cloud.
    ppm_align_mode: Literal["sim3", "rigid", "none"] = "sim3"
    # Voxel size applied before stochastic pruning.
    ppm_voxel_size: float = 0.01
    # Initial threshold for distance-adaptive pruning.
    ppm_tau0: float = 0.005
    # Threshold update strength for distance-adaptive pruning.
    ppm_beta: float = 0.01
    # Number of pruning iterations.
    ppm_iters: int = 6
    # Small epsilon for pruning stability.
    ppm_eps: float = 1e-8
    # RNG seed for pruning.
    ppm_seed: int = 42
    # Roll back the current prune iteration if it would drop below this count.
    ppm_min_points_after_prune: int = 5000
    # Save debug stats and point clouds to result_dir/ppm.
    ppm_save_debug: bool = False
    # Save the final pruned point cloud as pruned_points.ply.
    ppm_save_pruned_ply: bool = False
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    # RGB loss to optimize.
    rgb_loss: Literal["l1_ssim", "l1", "mse"] = "l1_ssim"

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Post-processing method for appearance correction (experimental)
    post_processing: Optional[Literal["bilateral_grid", "ppisp"]] = None
    # Use fused implementation for bilateral grid (only applies when post_processing="bilateral_grid")
    bilateral_grid_fused: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    # Enable PPISP controller
    ppisp_use_controller: bool = True
    # Use controller distillation in PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_distillation: bool = True
    # Controller activation ratio for PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_activation_num_steps: int = 25_000
    # Color correction method for cc_* diagnostic metrics.
    color_correct_method: Literal["affine", "quadratic"] = "affine"
    # Compute color-corrected metrics (cc_psnr, cc_ssim, cc_lpips) during evaluation
    use_color_correction_metric: bool = False

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    # Which evaluation metrics to compute. "psnr" skips SSIM/LPIPS entirely.
    eval_metrics: Literal["all", "psnr"] = "all"
    # Quantize eval renders to 8-bit before computing PSNR so local scores match uploads.
    eval_quantize_metrics: bool = True
    # Retained for CLI compatibility; exported eval renders are always written as .JPG.
    eval_render_format: Literal["png", "jpg"] = "jpg"
    # JPEG quality for exported eval renders.
    eval_jpg_quality: int = 95
    # Save GT|prediction comparison canvases alongside prediction exports.
    eval_save_comparison: bool = False
    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    def resolve_paths(self):
        if self.data_dir:
            scene_name = os.path.basename(os.path.normpath(self.data_dir))
        else:
            scene_name = self.scene
            self.data_dir = os.path.join(LOCAL_DATA_ROOT, scene_name)

        if not self.result_dir:
            colmap_tag = (self.colmap_path or "colmap").replace("/", "_")
            if colmap_tag.startswith("ma_sparse"):
                colmap_tag = "ma_sparse"
            elif colmap_tag.startswith("sparse"):
                colmap_tag = "sparse"
            self.result_dir = os.path.join(
                LOCAL_RESULTS_ROOT, f"{scene_name}_{colmap_tag}_{self.pose_source}"
            )

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        if self.eval_every_steps > 0:
            self.eval_every_steps = max(1, int(self.eval_every_steps * factor))
        if self.save_every_steps > 0:
            self.save_every_steps = max(1, int(self.save_every_steps * factor))
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
            if strategy.noise_injection_stop_iter >= 0:
                strategy.noise_injection_stop_iter = int(
                    strategy.noise_injection_stop_iter * factor
                )
        else:
            assert_never(strategy)

def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True,
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            load_exposure=cfg.load_exposure,
            colmap_path=cfg.colmap_path,
            transforms_train_path=cfg.transforms_train_path,
            transforms_test_path=cfg.transforms_test_path,
            pose_source=cfg.pose_source,
            transforms_coord=cfg.transforms_coord,
            use_transforms_intrinsics=cfg.use_transforms_intrinsics,
        )
        self.ppm_stats: Optional[Dict[str, object]] = None
        if cfg.ppm_enable:
            if cfg.init_type != "sfm":
                raise ValueError("PPM currently requires init_type='sfm'.")
            if cfg.pose_opt:
                raise ValueError("PPM requires pose_opt=False.")
            if cfg.depth_loss:
                raise ValueError(
                    "PPM currently does not support depth_loss because "
                    "points3D.bin tracks no longer match the injected dense point cloud."
                )
            if cfg.resume_ckpt is not None:
                raise ValueError(
                    "PPM affects fresh initialization only; do not combine it with resume_ckpt."
                )
            self._apply_ppm_preprocess()
        # Optional GT images for transforms_test.json views.
        self.testjson_gt_paths: List[Optional[str]] = []
        self.testjson_has_gt = False
        if (
            self.parser.render_camtoworlds is not None
            and self.parser.render_frame_paths is not None
        ):
            for rel_path in self.parser.render_frame_paths:
                candidates = []
                if os.path.isabs(rel_path):
                    candidates.append(rel_path)
                else:
                    basename = os.path.basename(rel_path)
                    candidates.extend(
                        [
                            os.path.join(cfg.data_dir, rel_path),
                            os.path.join(cfg.data_dir, "test", basename),
                            os.path.join(cfg.data_dir, "images", basename),
                        ]
                    )
                gt_path = next((p for p in candidates if os.path.isfile(p)), None)
                self.testjson_gt_paths.append(gt_path)
            num_missing = sum(p is None for p in self.testjson_gt_paths)
            self.testjson_has_gt = (
                num_missing == 0
                and len(self.testjson_gt_paths) == len(self.parser.render_camtoworlds)
            )
            if self.testjson_has_gt:
                print(
                    f"[Eval] Found GT images for all {len(self.testjson_gt_paths)} "
                    "transforms_test views. Metrics will use test GT."
                )
            else:
                print(
                    f"[Eval] transforms_test GT images missing ({num_missing}/{len(self.testjson_gt_paths)}). "
                    "Metrics will be skipped; only rendered test-view images will be saved."
                )
        if world_rank == 0:
            pose_meta = {
                "pose_source": cfg.pose_source,
                "pose_match_count": self.parser.pose_match_count,
                "pose_alignment_transform": self.parser.pose_alignment_transform.tolist(),
                "normalize_transform": self.parser.transform.tolist(),
                "has_test_render_poses": self.parser.render_camtoworlds is not None,
                "testjson_has_gt": self.testjson_has_gt,
                "ppm_enabled": cfg.ppm_enable,
                "ppm_stats": self.ppm_stats,
            }
            with open(f"{cfg.result_dir}/pose_metadata.json", "w") as f:
                json.dump(pose_meta, f, indent=2)
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        if self.parser.num_cameras > 1 and cfg.batch_size != 1:
            raise ValueError(
                f"When using multiple cameras ({self.parser.num_cameras} found), batch_size must be 1, "
                f"but got batch_size={cfg.batch_size}."
            )
        if cfg.post_processing == "ppisp" and cfg.batch_size != 1:
            raise ValueError(
                f"PPISP post-processing requires batch_size=1, got batch_size={cfg.batch_size}"
            )
        if cfg.post_processing is not None and world_size > 1:
            raise ValueError(
                f"Post-processing ({cfg.post_processing}) requires single-GPU training, "
                f"but world_size={world_size}."
            )
        if cfg.post_processing == "ppisp" and isinstance(cfg.strategy, DefaultStrategy):
            raise ValueError(
                f"PPISP post-processing requires MCMCStrategy at the moment."
            )

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.post_processing_module = None
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_module = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
        elif cfg.post_processing == "ppisp":
            ppisp_config = PPISPConfig(
                use_controller=cfg.ppisp_use_controller,
                controller_distillation=cfg.ppisp_controller_distillation,
                controller_activation_ratio=cfg.ppisp_controller_activation_num_steps
                / cfg.max_steps,
            )
            self.post_processing_module = PPISP(
                num_cameras=self.parser.num_cameras,
                num_frames=len(self.trainset),
                config=ppisp_config,
            ).to(self.device)

        self.post_processing_optimizers = []
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_optimizers = [
                torch.optim.Adam(
                    self.post_processing_module.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]
        elif cfg.post_processing == "ppisp":
            self.post_processing_optimizers = (
                self.post_processing_module.create_optimizers()
            )

        # Losses & Metrics.
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = None
        self.lpips = None
        if cfg.eval_metrics == "all":
            self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            if cfg.lpips_net == "alex":
                self.lpips = LearnedPerceptualImagePatchSimilarity(
                    net_type="alex", normalize=True
                ).to(self.device)
            elif cfg.lpips_net == "vgg":
                # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
                self.lpips = LearnedPerceptualImagePatchSimilarity(
                    net_type="vgg", normalize=False
                ).to(self.device)
            else:
                raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

        # Track if Gaussians are frozen (for controller distillation)
        self._gaussians_frozen = False
        self.resume_step = 0

    def _apply_ppm_preprocess(self) -> None:
        cfg = self.cfg
        gt_sparse_dir = cfg.ppm_gt_sparse_dir or cfg.colmap_path or "sparse"
        points, colors, stats = run_ppm_preprocess(
            data_dir=cfg.data_dir,
            dense_points_path=cfg.ppm_dense_points_path,
            dense_points_rgb_path=cfg.ppm_dense_points_rgb_path,
            gt_sparse_dir=gt_sparse_dir,
            mvs_sparse_dir=cfg.ppm_mvs_sparse_dir,
            align_to_gt=cfg.ppm_align_to_gt,
            align_mode=cfg.ppm_align_mode,
            pose_alignment_transform=self.parser.pose_alignment_transform,
            normalize_transform=self.parser.transform,
            voxel_size=cfg.ppm_voxel_size,
            tau0=cfg.ppm_tau0,
            beta=cfg.ppm_beta,
            iters=cfg.ppm_iters,
            eps=cfg.ppm_eps,
            min_points_after_prune=cfg.ppm_min_points_after_prune,
            seed=cfg.ppm_seed,
            save_debug=cfg.ppm_save_debug and self.world_rank == 0,
            save_pruned_ply=cfg.ppm_save_pruned_ply and self.world_rank == 0,
            result_dir=cfg.result_dir,
        )
        self.parser.points = points
        self.parser.points_rgb = colors
        self.parser.points_err = np.full((len(points),), -1.0, dtype=np.float32)
        # Dense point injection breaks the original sparse track associations, so
        # we clear them instead of carrying stale indices into later code paths.
        self.parser.point_indices = {
            name: np.empty((0,), dtype=np.int32) for name in self.parser.image_names
        }
        self.ppm_stats = stats
        print(
            f"[PPM] Replaced parser sparse points with {len(points)} preprocessed dense points."
        )

    def freeze_gaussians(self):
        """Freeze all Gaussian parameters for controller distillation.

        This prevents Gaussians from being updated by any loss (including regularization)
        while the controller learns to predict per-frame corrections.
        """
        if self._gaussians_frozen:
            return

        for name, param in self.splats.items():
            param.requires_grad = False

        self._gaussians_frozen = True
        print("[Distillation] Gaussian parameters frozen")

    def _compute_rgb_loss(
        self, colors: Tensor, pixels: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        mse_loss = F.mse_loss(colors, pixels)
        stats: Dict[str, Tensor] = {"mse_loss": mse_loss}

        if self.cfg.rgb_loss == "mse":
            rgb_loss = mse_loss
        elif self.cfg.rgb_loss == "l1":
            l1loss = F.l1_loss(colors, pixels)
            stats["l1loss"] = l1loss
            rgb_loss = l1loss
        elif self.cfg.rgb_loss == "l1_ssim":
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2),
                pixels.permute(0, 3, 1, 2),
                padding="valid",
            )
            stats["l1loss"] = l1loss
            stats["ssimloss"] = ssimloss
            rgb_loss = torch.lerp(l1loss, ssimloss, self.cfg.ssim_lambda)
        else:
            assert_never(self.cfg.rgb_loss)

        stats["rgb_loss"] = rgb_loss
        stats["train_psnr"] = -10.0 * torch.log10(mse_loss.clamp_min(1e-10))
        return rgb_loss, stats

    def _format_metric_summary(self, stats: Dict[str, float], prefix: str = "") -> str:
        parts = [f"{prefix}PSNR: {stats['psnr']:.3f}"]
        if "ssim" in stats:
            parts.append(f"SSIM: {stats['ssim']:.4f}")
        if "lpips" in stats:
            parts.append(f"LPIPS: {stats['lpips']:.3f}")
        if "cc_psnr" in stats:
            parts.append(f"CC_PSNR: {stats['cc_psnr']:.3f}")
        if "cc_ssim" in stats:
            parts.append(f"CC_SSIM: {stats['cc_ssim']:.4f}")
        if "cc_lpips" in stats:
            parts.append(f"CC_LPIPS: {stats['cc_lpips']:.3f}")
        return ", ".join(parts)

    def _quantize_eval_image(self, image: Tensor) -> Tensor:
        image = torch.clamp(image, 0.0, 1.0)
        if not self.cfg.eval_quantize_metrics:
            return image
        return torch.round(image * 255.0) / 255.0

    def _eval_render_ext(self) -> str:
        return "JPG"

    def _write_eval_image(self, path_stem: str, image: np.ndarray) -> str:
        out_path = f"{path_stem}.{self._eval_render_ext()}"
        image_u8 = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        imageio.imwrite(out_path, image_u8, quality=getattr(self.cfg, "eval_jpg_quality", 95))
        return out_path

    def _save_eval_outputs(
        self,
        path_stem: str,
        pred: Tensor,
        gt: Optional[Tensor] = None,
    ) -> str:
        pred_np = pred.squeeze(0).cpu().numpy()
        pred_path = self._write_eval_image(path_stem, pred_np)
        if gt is not None and self.cfg.eval_save_comparison:
            compare_np = torch.cat([gt, pred], dim=2).squeeze(0).cpu().numpy()
            self._write_eval_image(f"{path_stem}_compare", compare_np)
        return pred_path

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[RasterizeMode] = None,
        camera_model: Optional[CameraModel] = None,
        frame_idcs: Optional[Tensor] = None,
        camera_idcs: Optional[Tensor] = None,
        exposure: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0

        needs_pixel_coords = self.cfg.post_processing is not None
        pixel_coords = None
        if needs_pixel_coords:
            # Create pixel coordinates [H, W, 2] with +0.5 center offset
            pixel_y, pixel_x = torch.meshgrid(
                torch.arange(height, device=self.device) + 0.5,
                torch.arange(width, device=self.device) + 0.5,
                indexing="ij",
            )
            pixel_coords = torch.stack([pixel_x, pixel_y], dim=-1)  # [H, W, 2]
        
        if self.cfg.post_processing is not None:
            # Split RGB from extra channels (e.g. depth) for post-processing
            rgb = render_colors[..., :3]
            extra = render_colors[..., 3:] if render_colors.shape[-1] > 3 else None

            if self.cfg.post_processing == "bilateral_grid":
                if frame_idcs is not None:
                    grid_xy = (
                        pixel_coords / torch.tensor([width, height], device=self.device)
                    ).unsqueeze(0)
                    rgb = slice(
                        self.post_processing_module,
                        grid_xy.expand(rgb.shape[0], -1, -1, -1),
                        rgb,
                        frame_idcs.unsqueeze(-1),
                    )["rgb"]
            elif self.cfg.post_processing == "ppisp":
                camera_idx = camera_idcs.item() if camera_idcs is not None else None
                frame_idx = frame_idcs.item() if frame_idcs is not None else None
                rgb = self.post_processing_module(
                    rgb=rgb,
                    pixel_coords=pixel_coords,
                    resolution=(width, height),
                    camera_idx=camera_idx,
                    frame_idx=frame_idx,
                    exposure_prior=exposure,
                )

            render_colors = (
                torch.cat([rgb, extra], dim=-1) if extra is not None else rgb
            )

        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = self.resume_step
        if init_step >= max_steps:
            raise ValueError(
                f"Resume step {init_step} must be smaller than max_steps {max_steps}."
            )

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        # Post-processing module has a learning rate schedule
        if cfg.post_processing == "bilateral_grid":
            # Linear warmup + exponential decay
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.post_processing_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.post_processing_optimizers[0],
                            gamma=0.01 ** (1.0 / max_steps),
                        ),
                    ]
                )
            )
        elif cfg.post_processing == "ppisp":
            ppisp_schedulers = self.post_processing_module.create_schedulers(
                self.post_processing_optimizers,
                max_optimization_iters=max_steps,
            )
            schedulers.extend(ppisp_schedulers)

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            # Freeze Gaussians when PPISP controller distillation starts
            if (
                cfg.post_processing == "ppisp"
                and cfg.ppisp_use_controller
                and cfg.ppisp_controller_distillation
                and step >= cfg.ppisp_controller_activation_num_steps
            ):
                self.freeze_gaussians()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            exposure = (
                data["exposure"].to(device) if "exposure" in data else None
            )  # [B,]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
                frame_idcs=image_ids,
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            loss, rgb_loss_stats = self._compute_rgb_loss(colors, pixels)

            depthloss = None
            if cfg.depth_loss:
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )
                grid = points.unsqueeze(2)
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )
                depths = depths.squeeze(3).squeeze(1)
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.post_processing == "bilateral_grid":
                post_processing_reg_loss = 10 * total_variation_loss(
                    self.post_processing_module.grids
                )
                loss += post_processing_reg_loss
            elif cfg.post_processing == "ppisp":
                post_processing_reg_loss = (
                    self.post_processing_module.get_regularization_loss()
                )
                loss += post_processing_reg_loss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if depthloss is not None:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar(
                    "train/rgb_loss", rgb_loss_stats["rgb_loss"].item(), step
                )
                self.writer.add_scalar(
                    "train/mse_loss", rgb_loss_stats["mse_loss"].item(), step
                )
                self.writer.add_scalar(
                    "train/psnr", rgb_loss_stats["train_psnr"].item(), step
                )
                if "l1loss" in rgb_loss_stats:
                    self.writer.add_scalar(
                        "train/l1loss", rgb_loss_stats["l1loss"].item(), step
                    )
                if "ssimloss" in rgb_loss_stats:
                    self.writer.add_scalar(
                        "train/ssimloss", rgb_loss_stats["ssimloss"].item(), step
                    )
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if depthloss is not None:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.post_processing is not None:
                    self.writer.add_scalar(
                        "train/post_processing_reg_loss",
                        post_processing_reg_loss.item(),
                        step,
                    )
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            should_save_by_interval = (
                cfg.save_every_steps > 0 and (step + 1) % cfg.save_every_steps == 0
            )
            should_save_by_list = (step + 1) in cfg.save_steps
            if should_save_by_interval or should_save_by_list or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                if self.post_processing_module is not None:
                    data["post_processing"] = self.post_processing_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if ((step + 1) in cfg.ply_steps or step == max_steps - 1) and cfg.save_ply:

                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]

                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.post_processing_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set / save test-view renders
            should_eval_by_interval = (
                cfg.eval_every_steps > 0 and (step + 1) % cfg.eval_every_steps == 0
            )
            should_eval_by_list = (step + 1) in cfg.eval_steps
            should_eval = (
                should_eval_by_interval
                or should_eval_by_list
                or step == max_steps - 1
            )
            if should_eval:
                self.eval(step)
                self.render_traj(step)

            # run compression
            if cfg.compression is not None and should_eval_by_list:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        step_id = step + 1
        if self.parser.render_camtoworlds is not None:
            self._eval_testjson(step=step, stage=stage)
            return

        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            # Exposure metadata is available for any image with EXIF data (train or val)
            exposure = data["exposure"].to(device) if "exposure" in data else None

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                frame_idcs=None,  # For novel views, pass None (no per-frame parameters available)
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = self._quantize_eval_image(colors)

            if world_rank == 0:
                self._save_eval_outputs(
                    f"{self.render_dir}/{stage}_step{step_id}_{i:04d}",
                    pred=colors,
                    gt=pixels,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                if self.ssim is not None:
                    metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                if self.lpips is not None:
                    metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                # Compute color-corrected metrics for fair comparison across methods
                if cfg.use_color_correction_metric:
                    if cfg.color_correct_method == "affine":
                        cc_colors = color_correct_affine(colors, pixels)
                    else:
                        cc_colors = color_correct_quadratic(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    if self.ssim is not None:
                        metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    if self.lpips is not None:
                        metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            metric_summary = self._format_metric_summary(stats)
            print(
                f"{metric_summary} Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step_id:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def _eval_testjson(self, step: int, stage: str = "val"):
        """Evaluate/render views from transforms_test.json."""
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        step_id = step + 1

        assert self.parser.render_camtoworlds is not None
        camtoworlds_all = (
            torch.from_numpy(self.parser.render_camtoworlds).float().to(device)
        )
        if self.parser.render_K is None:
            K_np = list(self.parser.Ks_dict.values())[0]
        else:
            K_np = self.parser.render_K
        if self.parser.render_imsize is None:
            width, height = list(self.parser.imsize_dict.values())[0]
        else:
            width, height = self.parser.render_imsize
        K = torch.from_numpy(K_np).float().to(device)[None]

        has_gt = self.testjson_has_gt and (
            len(self.testjson_gt_paths) == len(camtoworlds_all)
        )
        metrics = defaultdict(list)
        ellipse_time = 0.0

        for i in range(len(camtoworlds_all)):
            camtoworlds = camtoworlds_all[i : i + 1]
            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=K,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                frame_idcs=None,  # novel test views do not have train-time frame ids
                camera_idcs=None,
                exposure=None,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = self._quantize_eval_image(colors)

            if world_rank != 0:
                continue

            out_stem = f"{self.render_dir}/{stage}_testjson_step{step_id}_{i:04d}"
            if has_gt:
                gt_path = self.testjson_gt_paths[i]
                assert gt_path is not None
                pixels_np = imageio.imread(gt_path)
                if pixels_np.ndim == 2:
                    pixels_np = np.repeat(pixels_np[..., None], 3, axis=-1)
                pixels = (
                    torch.from_numpy(pixels_np[..., :3]).float().to(device)[None] / 255.0
                )  # [1, H, W, 3]
                if pixels.shape[1] != height or pixels.shape[2] != width:
                    pixels = F.interpolate(
                        pixels.permute(0, 3, 1, 2),
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)

                self._save_eval_outputs(out_stem, pred=colors, gt=pixels)

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                if self.ssim is not None:
                    metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                if self.lpips is not None:
                    metrics["lpips"].append(self.lpips(colors_p, pixels_p))

                if cfg.use_color_correction_metric:
                    if cfg.color_correct_method == "affine":
                        cc_colors = color_correct_affine(colors, pixels)
                    else:
                        cc_colors = color_correct_quadratic(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    if self.ssim is not None:
                        metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    if self.lpips is not None:
                        metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))
            else:
                self._save_eval_outputs(out_stem, pred=colors)

        if world_rank == 0:
            ellipse_time /= len(camtoworlds_all)
            stats = {
                "ellipse_time": ellipse_time,
                "num_GS": len(self.splats["means"]),
                "num_views": len(camtoworlds_all),
                "has_test_gt": int(has_gt),
            }
            if has_gt:
                stats.update({k: torch.stack(v).mean().item() for k, v in metrics.items()})
                metric_summary = self._format_metric_summary(stats, prefix="[testjson] ")
                print(
                    f"{metric_summary} Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"[testjson] No GT images found. Saved {len(camtoworlds_all)} rendered views to {self.render_dir}."
                )

            with open(f"{self.stats_dir}/{stage}_step{step_id:04d}.json", "w") as f:
                json.dump(stats, f)
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        if cfg.render_traj_path == "testjson":
            if self.parser.render_camtoworlds is None:
                raise ValueError(
                    "render_traj_path='testjson' requires --transforms_test_path."
                )
            camtoworlds_np = self.parser.render_camtoworlds
            if self.parser.render_K is None:
                K_np = list(self.parser.Ks_dict.values())[0]
            else:
                K_np = self.parser.render_K
            if self.parser.render_imsize is None:
                width, height = list(self.parser.imsize_dict.values())[0]
            else:
                width, height = self.parser.render_imsize
        else:
            camtoworlds_np = self.parser.camtoworlds[5:-5]
            if cfg.render_traj_path == "interp":
                camtoworlds_np = generate_interpolated_path(
                    camtoworlds_np, 1
                )  # [N, 3, 4]
            elif cfg.render_traj_path == "ellipse":
                height = camtoworlds_np[:, 2, 3].mean()
                camtoworlds_np = generate_ellipse_path_z(
                    camtoworlds_np, height=height
                )  # [N, 3, 4]
            elif cfg.render_traj_path == "spiral":
                camtoworlds_np = generate_spiral_path(
                    camtoworlds_np,
                    bounds=self.parser.bounds * self.scene_scale,
                    spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
                )
            else:
                raise ValueError(
                    f"Render trajectory type not supported: {cfg.render_traj_path}"
                )

            camtoworlds_np = np.concatenate(
                [
                    camtoworlds_np,
                    np.repeat(
                        np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_np), axis=0
                    ),
                ],
                axis=1,
            )  # [N, 4, 4]
            K_np = list(self.parser.Ks_dict.values())[0]
            width, height = list(self.parser.imsize_dict.values())[0]

        camtoworlds_all = torch.from_numpy(camtoworlds_np).float().to(device)
        K = torch.from_numpy(K_np).float().to(device)

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def export_ppisp_reports(self) -> None:
        """Export PPISP visualization reports (PDF) and parameter JSON."""
        if self.cfg.post_processing != "ppisp":
            return
        print("Exporting PPISP reports...")

        # Compute frames per camera from training dataset
        num_cameras = self.parser.num_cameras
        frames_per_camera = [0] * num_cameras
        for idx in self.trainset.indices:
            cam_idx = self.parser.camera_indices[idx]
            frames_per_camera[cam_idx] += 1

        # Generate camera names from COLMAP camera IDs
        # camera_id_to_idx maps COLMAP ID -> 0-based index
        idx_to_camera_id = {v: k for k, v in self.parser.camera_id_to_idx.items()}
        camera_names = [f"camera_{idx_to_camera_id[i]}" for i in range(num_cameras)]

        # Export reports
        output_dir = Path(self.cfg.result_dir) / "ppisp_reports"
        pdf_paths = export_ppisp_report(
            self.post_processing_module,
            frames_per_camera,
            output_dir,
            camera_names=camera_names,
        )
        print(f"PPISP reports saved to {output_dir}")
        for path in pdf_paths:
            print(f"  - {path.name}")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    cfg.resolve_paths()

    # Import post-processing modules based on configuration
    # These imports must be here (not in __main__) for distributed workers
    if cfg.post_processing == "bilateral_grid":
        global BilateralGrid, slice, total_variation_loss
        if cfg.bilateral_grid_fused:
            from fused_bilagrid import (
                BilateralGrid,
                slice,
                total_variation_loss,
            )
        else:
            from lib_bilagrid import (
                BilateralGrid,
                slice,
                total_variation_loss,
            )
    elif cfg.post_processing == "ppisp":
        global PPISP, PPISPConfig, export_ppisp_report
        from ppisp import PPISP, PPISPConfig
        from ppisp.report import export_ppisp_report

    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    def load_ckpts_into_runner(ckpt_files: List[str]) -> int:
        def infer_num_points(splat_state: Dict[str, Tensor]) -> int:
            for value in splat_state.values():
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
            raise ValueError("Checkpoint splat state is empty.")

        def get_compatible_splat_tensor(
            splat_state: Dict[str, Tensor], key: str
        ) -> Tensor:
            if key in splat_state:
                return splat_state[key]

            num_points = infer_num_points(splat_state)
            ref = runner.splats[key].detach()

            if key == "colors" and "sh0" in splat_state:
                c0 = 0.28209479177387814
                rgb = torch.clamp(splat_state["sh0"][:, 0, :] * c0 + 0.5, 1e-4, 1 - 1e-4)
                return torch.logit(rgb)
            if key == "features":
                return ref.new_zeros((num_points, *ref.shape[1:]))
            if key == "sh0" and "colors" in splat_state:
                rgb = torch.sigmoid(splat_state["colors"])
                return rgb_to_sh(rgb).unsqueeze(1)
            if key == "shN":
                return ref.new_zeros((num_points, *ref.shape[1:]))

            raise KeyError(key)

        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in ckpt_files
        ]
        for k in runner.splats.keys():
            try:
                runner.splats[k].data = torch.cat(
                    [get_compatible_splat_tensor(ckpt["splats"], k) for ckpt in ckpts]
                )
            except KeyError as exc:
                raise KeyError(
                    f"Checkpoint is missing splat parameter '{exc.args[0]}' and it "
                    "could not be derived automatically."
                ) from exc
        if cfg.pose_opt:
            pose_state = ckpts[0].get("pose_adjust")
            if pose_state is not None:
                pose_module = (
                    runner.pose_adjust.module
                    if isinstance(runner.pose_adjust, DDP)
                    else runner.pose_adjust
                )
                pose_module.load_state_dict(pose_state)
        if cfg.app_opt:
            app_state = ckpts[0].get("app_module")
            if app_state is not None:
                app_module = (
                    runner.app_module.module
                    if isinstance(runner.app_module, DDP)
                    else runner.app_module
                )
                app_module.load_state_dict(app_state)
            elif world_rank == 0:
                print("[Checkpoint] app_module state missing; using fresh app_opt initialization.")
        if runner.post_processing_module is not None:
            pp_state = ckpts[0].get("post_processing")
            if pp_state is not None:
                runner.post_processing_module.load_state_dict(pp_state)
            elif world_rank == 0:
                print(
                    "[Checkpoint] post_processing state missing; using fresh "
                    f"{cfg.post_processing} initialization."
                )
        return ckpts[0]["step"]

    if cfg.ckpt is not None:
        # run eval only
        if cfg.resume_ckpt is not None:
            raise ValueError("Use either --ckpt for eval or --resume_ckpt for training, not both.")
        step = load_ckpts_into_runner(cfg.ckpt)
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        if cfg.resume_ckpt is not None:
            step = load_ckpts_into_runner(cfg.resume_ckpt)
            runner.resume_step = step + 1
            if world_rank == 0:
                print(f"Resuming training from step {step}.")
        runner.train()
        runner.export_ppisp_reports()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)
