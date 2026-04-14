import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_NAKA_CKPT = (
    REPO_ROOT
    / "outputs"
    / "naka"
    / "checkpoints"
    / "latest.pth"
)
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value from: {value}")


def parse_int_list(value: str, fallback: List[int]) -> List[int]:
    if not value.strip():
        return list(fallback)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def list_image_files(folder: Path) -> List[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def ensure_import_paths() -> None:
    for path in (
        REPO_ROOT,
        REPO_ROOT / "vggt",
        REPO_ROOT / "gsplat",
        REPO_ROOT / "gsplat" / "examples",
    ):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def ensure_env_defaults() -> None:
    os.environ.setdefault("BUILD_3DGUT", "0")
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR",
        os.path.join(tempfile.gettempdir(), "torch_extensions"),
    )


def to_abs_path(raw: str, default: Path) -> Path:
    return Path(raw).expanduser().resolve() if raw else default.expanduser().resolve()


def to_abs_optional_path(raw: str) -> Optional[Path]:
    return Path(raw).expanduser().resolve() if raw else None


def resolve_scene_relative_path(raw: str, scene_dir: Path, default: Optional[Path] = None) -> Optional[Path]:
    if raw:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = scene_dir / candidate
        return candidate.resolve()
    return None if default is None else default.expanduser().resolve()


def maybe_str(path: Optional[Path]) -> Optional[str]:
    return None if path is None else str(path)


def load_latest_stats(stats_dir: Path) -> Dict[str, Any]:
    if not stats_dir.is_dir():
        return {"latest_stats_path": None, "latest_stats": None}
    stats_files = sorted(stats_dir.glob("*.json"))
    if not stats_files:
        return {"latest_stats_path": None, "latest_stats": None}
    latest_path = stats_files[-1]
    return {
        "latest_stats_path": str(latest_path),
        "latest_stats": json.loads(latest_path.read_text()),
    }


def save_summary(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def stage_banner(name: str) -> None:
    print(f"\n{'=' * 24} {name} {'=' * 24}")


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    lowlight_dir = to_abs_path(args.lowlight_dir, scene_dir / "train")
    enhanced_dir = to_abs_path(args.enhanced_dir, scene_dir / "images")
    sparse_dir = to_abs_path(args.sparse_dir, scene_dir / "sparse")
    result_dir = to_abs_path(args.result_dir, scene_dir / "gsplat_results")
    transforms_train_path = to_abs_path(
        args.transforms_train_path,
        scene_dir / "transforms_train.json",
    )
    transforms_test_path = to_abs_path(
        args.transforms_test_path,
        scene_dir / "transforms_test.json",
    )
    ppm_dense_points_path = resolve_scene_relative_path(
        args.gs_ppm_dense_points_path,
        scene_dir,
        sparse_dir / "points.ply",
    )
    summary_path = to_abs_path(
        args.summary_path,
        result_dir / "pipeline_summary.json",
    )
    return {
        "scene_dir": scene_dir,
        "lowlight_dir": lowlight_dir,
        "enhanced_dir": enhanced_dir,
        "sparse_dir": sparse_dir,
        "result_dir": result_dir,
        "transforms_train_path": transforms_train_path,
        "transforms_test_path": transforms_test_path,
        "ppm_dense_points_path": ppm_dense_points_path,
        "summary_path": summary_path,
    }


def run_naka_stage(args: argparse.Namespace, paths: Dict[str, Path]) -> Dict[str, Any]:
    stage_banner("Stage 1: Naka Enhancement")
    ensure_import_paths()
    import naka_color_correction as naka

    ckpt_path = to_abs_optional_path(args.naka_ckpt)
    if ckpt_path is None:
        ckpt_path = DEFAULT_NAKA_CKPT.resolve() if DEFAULT_NAKA_CKPT.exists() else None
    if ckpt_path is None or not ckpt_path.is_file():
        raise FileNotFoundError(
            "Naka checkpoint is required. Pass --naka_ckpt, or place the default checkpoint at "
            f"{DEFAULT_NAKA_CKPT}"
        )

    lowlight_dir = paths["lowlight_dir"]
    if not lowlight_dir.is_dir():
        raise FileNotFoundError(f"Low-light input directory does not exist: {lowlight_dir}")

    input_images = list_image_files(lowlight_dir)
    if not input_images:
        raise FileNotFoundError(f"No input images found in {lowlight_dir}")

    paths["enhanced_dir"].mkdir(parents=True, exist_ok=True)
    infer_args = argparse.Namespace(
        ckpt=str(ckpt_path),
        input_dir=str(lowlight_dir),
        output_dir=str(paths["enhanced_dir"]),
        base_ch=args.naka_base_ch,
        mul_range=args.naka_mul_range,
        add_range=args.naka_add_range,
        hf_kernel_size=args.naka_hf_kernel_size,
        hf_sigma=args.naka_hf_sigma,
        tile_size=args.naka_tile_size,
        tile_overlap=args.naka_tile_overlap,
    )

    start = time.time()
    naka.inference(infer_args)
    elapsed = time.time() - start
    output_images = list_image_files(paths["enhanced_dir"])

    print(
        f"[Naka] Enhanced {len(output_images)} images into {paths['enhanced_dir']} "
        f"in {elapsed:.2f}s"
    )
    return {
        "status": "completed",
        "checkpoint": str(ckpt_path),
        "input_dir": str(lowlight_dir),
        "output_dir": str(paths["enhanced_dir"]),
        "input_count": len(input_images),
        "output_count": len(output_images),
        "elapsed_sec": elapsed,
    }


def run_vggt_stage(args: argparse.Namespace, paths: Dict[str, Path]) -> Dict[str, Any]:
    stage_banner("Stage 2: VGGT Reconstruction")
    ensure_import_paths()
    import demo_colmap

    enhanced_images = list_image_files(paths["enhanced_dir"])
    if not enhanced_images:
        raise FileNotFoundError(
            f"No enhanced images found in {paths['enhanced_dir']}. "
            "Run Naka enhancement first or use --skip_naka with prepared outputs."
        )

    demo_args = argparse.Namespace(
        scene_dir=str(paths["scene_dir"]),
        seed=args.seed,
        sort_images=args.vggt_sort_images,
        use_ba=args.vggt_use_ba,
        max_reproj_error=args.vggt_max_reproj_error,
        shared_camera=args.vggt_shared_camera,
        camera_type=args.vggt_camera_type,
        vis_thresh=args.vggt_vis_thresh,
        query_frame_num=args.vggt_query_frame_num,
        max_query_pts=args.vggt_max_query_pts,
        fine_tracking=args.vggt_fine_tracking,
        conf_thres_value=args.vggt_conf_thres_value,
    )

    start = time.time()
    stats = demo_colmap.demo_fn(demo_args)
    elapsed = time.time() - start
    stats["elapsed_sec"] = elapsed

    print(
        f"[VGGT] Saved sparse reconstruction to {paths['sparse_dir']} "
        f"in {elapsed:.2f}s"
    )
    return {"status": "completed", **stats}


def build_gsplat_config(
    args: argparse.Namespace,
    paths: Dict[str, Path],
):
    ensure_import_paths()
    ensure_env_defaults()
    import simple_trainer as trainer

    if args.gs_mode == "default":
        cfg = trainer.Config(
            strategy=trainer.DefaultStrategy(verbose=True),
        )
    else:
        cfg = trainer.Config(
            init_opa=0.5,
            init_scale=0.1,
            opacity_reg=0.01,
            scale_reg=0.01,
            strategy=trainer.MCMCStrategy(verbose=True),
        )

    cfg.disable_viewer = args.gs_disable_viewer
    cfg.render_traj_path = args.gs_render_traj_path
    cfg.scene = paths["scene_dir"].name
    cfg.data_dir = str(paths["scene_dir"])
    cfg.colmap_path = str(paths["sparse_dir"])
    cfg.transforms_train_path = (
        str(paths["transforms_train_path"])
        if paths["transforms_train_path"].is_file()
        else None
    )
    cfg.transforms_test_path = (
        str(paths["transforms_test_path"])
        if paths["transforms_test_path"].is_file()
        else None
    )
    cfg.pose_source = args.gs_pose_source
    cfg.use_transforms_intrinsics = args.gs_use_transforms_intrinsics
    cfg.data_factor = args.gs_data_factor
    cfg.result_dir = str(paths["result_dir"])
    cfg.test_every = args.gs_test_every
    cfg.batch_size = args.gs_batch_size
    cfg.steps_scaler = args.gs_steps_scaler
    cfg.max_steps = args.gs_max_steps
    cfg.eval_every_steps = args.gs_eval_every_steps
    cfg.eval_steps = parse_int_list(args.gs_eval_steps, [args.gs_max_steps])
    cfg.save_steps = parse_int_list(args.gs_save_steps, [args.gs_max_steps])
    cfg.save_every_steps = args.gs_save_every_steps
    cfg.save_ply = args.gs_save_ply
    cfg.ply_steps = parse_int_list(args.gs_ply_steps, [args.gs_max_steps])
    cfg.disable_video = args.gs_disable_video
    cfg.sh_degree = args.gs_sh_degree
    cfg.ssim_lambda = args.gs_ssim_lambda
    cfg.rgb_loss = args.gs_rgb_loss
    cfg.eval_metrics = args.gs_eval_metrics
    cfg.eval_quantize_metrics = args.gs_eval_quantize_metrics
    cfg.eval_jpg_quality = args.gs_eval_jpg_quality
    cfg.eval_save_comparison = args.gs_eval_save_comparison
    cfg.use_color_correction_metric = args.gs_use_color_correction_metric
    cfg.color_correct_method = args.gs_color_correct_method
    cfg.ppm_enable = args.gs_ppm_enable
    cfg.ppm_dense_points_path = (
        str(paths["ppm_dense_points_path"]) if args.gs_ppm_enable else None
    )
    cfg.ppm_dense_points_rgb_path = maybe_str(
        resolve_scene_relative_path(args.gs_ppm_dense_points_rgb_path, paths["scene_dir"])
    )
    cfg.ppm_gt_sparse_dir = maybe_str(
        resolve_scene_relative_path(args.gs_ppm_gt_sparse_dir, paths["scene_dir"])
    )
    cfg.ppm_mvs_sparse_dir = maybe_str(
        resolve_scene_relative_path(args.gs_ppm_mvs_sparse_dir, paths["scene_dir"])
    )
    cfg.ppm_align_to_gt = args.gs_ppm_align_to_gt
    cfg.ppm_align_mode = args.gs_ppm_align_mode
    cfg.ppm_voxel_size = args.gs_ppm_voxel_size
    cfg.ppm_tau0 = args.gs_ppm_tau0
    cfg.ppm_beta = args.gs_ppm_beta
    cfg.ppm_iters = args.gs_ppm_iters
    cfg.ppm_eps = args.gs_ppm_eps
    cfg.ppm_min_points_after_prune = args.gs_ppm_min_points_after_prune
    cfg.ppm_seed = args.gs_ppm_seed
    cfg.ppm_save_debug = args.gs_ppm_save_debug
    cfg.ppm_save_pruned_ply = args.gs_ppm_save_pruned_ply

    if args.gs_ckpt:
        cfg.ckpt = [str(to_abs_path(args.gs_ckpt, Path(args.gs_ckpt)))]
    if args.gs_resume_ckpt:
        cfg.resume_ckpt = [str(to_abs_path(args.gs_resume_ckpt, Path(args.gs_resume_ckpt)))]

    cfg.adjust_steps(cfg.steps_scaler)
    return trainer, cfg


def run_gsplat_stage(args: argparse.Namespace, paths: Dict[str, Path]) -> Dict[str, Any]:
    stage_banner("Stage 3: Gaussian Splatting")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "The gsplat stage requires a visible CUDA device, but torch.cuda.is_available() is False."
        )

    trainer, cfg = build_gsplat_config(args, paths)

    if cfg.ppm_enable and not paths["ppm_dense_points_path"].is_file():
        raise FileNotFoundError(
            f"PPM dense point cloud is missing: {paths['ppm_dense_points_path']}"
        )

    if not paths["sparse_dir"].is_dir():
        raise FileNotFoundError(
            f"Sparse reconstruction directory does not exist: {paths['sparse_dir']}"
        )

    start = time.time()
    trainer.main(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
    elapsed = time.time() - start

    render_dir = paths["result_dir"] / "renders"
    stats_dir = paths["result_dir"] / "stats"
    latest_stats = load_latest_stats(stats_dir)
    render_files = sorted(render_dir.glob("*")) if render_dir.is_dir() else []

    print(
        f"[GSplat] Results written to {paths['result_dir']} "
        f"in {elapsed:.2f}s"
    )
    return {
        "status": "completed",
        "mode": args.gs_mode,
        "result_dir": str(paths["result_dir"]),
        "render_dir": str(render_dir),
        "stats_dir": str(stats_dir),
        "render_count": len(render_files),
        "elapsed_sec": elapsed,
        **latest_stats,
    }


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end low-light to novel-view rendering pipeline: "
            "Naka enhancement -> VGGT reconstruction -> PPM GSplat."
        )
    )
    parser.add_argument("--scene_dir", required=True, help="Scene root containing train/test/transforms files.")
    parser.add_argument("--lowlight_dir", default="", help="Low-light input directory. Default: <scene_dir>/train")
    parser.add_argument("--enhanced_dir", default="", help="Enhanced image output directory. Default: <scene_dir>/images")
    parser.add_argument("--sparse_dir", default="", help="VGGT sparse output directory. Default: <scene_dir>/sparse")
    parser.add_argument("--result_dir", default="", help="GSplat result directory. Default: <scene_dir>/gsplat_results")
    parser.add_argument("--transforms_train_path", default="", help="Train transforms JSON. Default: <scene_dir>/transforms_train.json")
    parser.add_argument("--transforms_test_path", default="", help="Test transforms JSON. Default: <scene_dir>/transforms_test.json")
    parser.add_argument("--summary_path", default="", help="Pipeline summary JSON path. Default: <result_dir>/pipeline_summary.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_naka", action="store_true")
    parser.add_argument("--skip_vggt", action="store_true")
    parser.add_argument("--skip_gsplat", action="store_true")

    parser.add_argument("--naka_ckpt", default="", help="Checkpoint for naka_color_correction.py inference")
    parser.add_argument("--naka_base_ch", type=int, default=32)
    parser.add_argument("--naka_mul_range", type=float, default=0.6)
    parser.add_argument("--naka_add_range", type=float, default=0.25)
    parser.add_argument("--naka_hf_kernel_size", type=int, default=5)
    parser.add_argument("--naka_hf_sigma", type=float, default=1.0)
    parser.add_argument("--naka_tile_size", type=int, default=0)
    parser.add_argument("--naka_tile_overlap", type=int, default=32)

    parser.add_argument("--vggt_use_ba", action="store_true")
    parser.add_argument(
        "--vggt_sort_images",
        type=str2bool,
        default=False,
        help="Sort enhanced image filenames before VGGT. Default: False, which preserves glob.glob() order.",
    )
    parser.add_argument("--vggt_max_reproj_error", type=float, default=8.0)
    parser.add_argument("--vggt_shared_camera", action="store_true")
    parser.add_argument("--vggt_camera_type", default="SIMPLE_PINHOLE")
    parser.add_argument("--vggt_vis_thresh", type=float, default=0.2)
    parser.add_argument("--vggt_query_frame_num", type=int, default=8)
    parser.add_argument("--vggt_max_query_pts", type=int, default=4096)
    parser.add_argument("--vggt_fine_tracking", type=str2bool, default=True)
    parser.add_argument("--vggt_conf_thres_value", type=float, default=2.0)

    parser.add_argument("--gs_mode", choices=["default", "mcmc"], default="default")
    parser.add_argument("--gs_ckpt", default="", help="Optional gsplat checkpoint for eval-only mode")
    parser.add_argument("--gs_resume_ckpt", default="", help="Optional gsplat checkpoint to resume training")
    parser.add_argument(
        "--gs_disable_viewer",
        "--gs-disable-viewer",
        "--disable-viewer",
        dest="gs_disable_viewer",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--gs_render_traj_path",
        "--gs-render-traj-path",
        "--render-traj-path",
        dest="gs_render_traj_path",
        default="testjson",
    )
    parser.add_argument(
        "--gs_pose_source",
        "--gs-pose-source",
        "--pose-source",
        dest="gs_pose_source",
        choices=["colmap", "align", "replace"],
        default="replace",
    )
    parser.add_argument("--gs_use_transforms_intrinsics", type=str2bool, default=True)
    parser.add_argument("--gs_data_factor", type=int, default=1)
    parser.add_argument("--gs_test_every", type=int, default=8)
    parser.add_argument("--gs_batch_size", type=int, default=1)
    parser.add_argument("--gs_steps_scaler", type=float, default=1.0)
    parser.add_argument("--gs_max_steps", type=int, default=8000)
    parser.add_argument("--gs_eval_every_steps", type=int, default=2000)
    parser.add_argument("--gs_eval_steps", default="", help="Comma-separated evaluation steps. Default: max_steps")
    parser.add_argument("--gs_save_steps", default="", help="Comma-separated checkpoint save steps. Default: max_steps")
    parser.add_argument("--gs_save_every_steps", type=int, default=2000)
    parser.add_argument("--gs_save_ply", type=str2bool, default=False)
    parser.add_argument("--gs_ply_steps", default="", help="Comma-separated ply export steps. Default: max_steps")
    parser.add_argument("--gs_disable_video", type=str2bool, default=True)
    parser.add_argument("--gs_sh_degree", type=int, default=3)
    parser.add_argument("--gs_ssim_lambda", type=float, default=0.2)
    parser.add_argument("--gs_rgb_loss", choices=["l1_ssim", "l1", "mse"], default="l1_ssim")
    parser.add_argument("--gs_eval_metrics", choices=["all", "psnr"], default="all")
    parser.add_argument("--gs_eval_quantize_metrics", type=str2bool, default=True)
    parser.add_argument("--gs_eval_jpg_quality", type=int, default=95)
    parser.add_argument("--gs_eval_save_comparison", type=str2bool, default=False)
    parser.add_argument("--gs_use_color_correction_metric", type=str2bool, default=False)
    parser.add_argument("--gs_color_correct_method", choices=["affine", "quadratic"], default="affine")

    parser.add_argument(
        "--gs_ppm_enable",
        "--gs-ppm-enable",
        "--ppm-enable",
        dest="gs_ppm_enable",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--gs_ppm_dense_points_path",
        "--gs-ppm-dense-points-path",
        "--ppm-dense-points-path",
        dest="gs_ppm_dense_points_path",
        default="",
        help="Dense point cloud path for PPM. Default: <sparse_dir>/points.ply",
    )
    parser.add_argument(
        "--gs_ppm_dense_points_rgb_path",
        "--gs-ppm-dense-points-rgb-path",
        "--ppm-dense-points-rgb-path",
        dest="gs_ppm_dense_points_rgb_path",
        default="",
    )
    parser.add_argument(
        "--gs_ppm_gt_sparse_dir",
        "--gs-ppm-gt-sparse-dir",
        "--ppm-gt-sparse-dir",
        dest="gs_ppm_gt_sparse_dir",
        default="",
    )
    parser.add_argument(
        "--gs_ppm_mvs_sparse_dir",
        "--gs-ppm-mvs-sparse-dir",
        "--ppm-mvs-sparse-dir",
        dest="gs_ppm_mvs_sparse_dir",
        default="",
    )
    parser.add_argument(
        "--gs_ppm_align_to_gt",
        "--gs-ppm-align-to-gt",
        "--ppm-align-to-gt",
        dest="gs_ppm_align_to_gt",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--gs_ppm_align_mode",
        "--gs-ppm-align-mode",
        "--ppm-align-mode",
        dest="gs_ppm_align_mode",
        choices=["sim3", "rigid", "none"],
        default="sim3",
    )
    parser.add_argument(
        "--gs_ppm_voxel_size",
        "--gs-ppm-voxel-size",
        "--ppm-voxel-size",
        dest="gs_ppm_voxel_size",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--gs_ppm_tau0",
        "--gs-ppm-tau0",
        "--ppm-tau0",
        dest="gs_ppm_tau0",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--gs_ppm_beta",
        "--gs-ppm-beta",
        "--ppm-beta",
        dest="gs_ppm_beta",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--gs_ppm_iters",
        "--gs-ppm-iters",
        "--ppm-iters",
        dest="gs_ppm_iters",
        type=int,
        default=6,
    )
    parser.add_argument("--gs_ppm_eps", type=float, default=1e-8)
    parser.add_argument("--gs_ppm_seed", type=int, default=42)
    parser.add_argument("--gs_ppm_min_points_after_prune", type=int, default=5000)
    parser.add_argument("--gs_ppm_save_debug", type=str2bool, default=False)
    parser.add_argument("--gs_ppm_save_pruned_ply", type=str2bool, default=False)

    return parser


def main() -> None:
    ensure_env_defaults()
    args = create_parser().parse_args()
    paths = resolve_paths(args)
    paths["result_dir"].mkdir(parents=True, exist_ok=True)

    test_gt_images = list_image_files(paths["scene_dir"] / "test")
    transforms_test_exists = paths["transforms_test_path"].is_file()

    summary: Dict[str, Any] = {
        "scene_dir": str(paths["scene_dir"]),
        "seed": args.seed,
        "paths": {key: str(value) for key, value in paths.items()},
        "has_test_gt_images": bool(test_gt_images),
        "test_gt_count": len(test_gt_images),
        "transforms_test_exists": transforms_test_exists,
        "expected_eval_behavior": (
            "compute_metrics_and_save_renders"
            if transforms_test_exists and test_gt_images
            else "save_renders_only"
        ),
        "stages": {},
    }
    save_summary(paths["summary_path"], summary)

    if args.skip_naka:
        summary["stages"]["naka"] = {"status": "skipped"}
    else:
        summary["stages"]["naka"] = run_naka_stage(args, paths)
    save_summary(paths["summary_path"], summary)

    if args.skip_vggt:
        summary["stages"]["vggt"] = {"status": "skipped"}
    else:
        summary["stages"]["vggt"] = run_vggt_stage(args, paths)
    save_summary(paths["summary_path"], summary)

    if args.skip_gsplat:
        summary["stages"]["gsplat"] = {"status": "skipped"}
    else:
        summary["stages"]["gsplat"] = run_gsplat_stage(args, paths)
    save_summary(paths["summary_path"], summary)

    print(f"\nPipeline summary saved to: {paths['summary_path']}")


if __name__ == "__main__":
    main()
