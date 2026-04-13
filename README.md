# NAKA-GS
This pipeline was bulided base on [VGGT](https://github.com/facebookresearch/vggt) and [gsplat](https://github.com/nerfstudio-project/gsplat), thanks for their excellent works.

NAKA-GS is an end-to-end pipeline for low-light 3D scene reconstruction and novel-view synthesis:

1. `Naka` enhances low-light training images.
2. `VGGT` reconstructs sparse cameras and geometry from the enhanced images.
3. `gsplat` performs Gaussian Splatting training, with optional `PPM` dense-point preprocessing.

The qualitative result (visual comparison on RealX3D) can be found at folder"asset"

## 1. What The Pipeline Expects

Each scene directory should look like this before the first run:

```text
data/
└── Scene1/
    ├── train/                  # low-light training images
    ├── transforms_train.json   # training camera poses
    ├── transforms_test.json    # render trajectory / test poses
    └── test/                   # optional GT test images for metrics
```

After the pipeline runs, it will automatically create:

```text
data/
└── Scene/
    ├── images/                 # Naka-enhanced images
    ├── sparse/                 # VGGT reconstruction outputs
    │   ├── cameras.bin
    │   ├── images.bin
    │   ├── points3D.bin
    │   └── points.ply
    └── gsplat_results/         # rendering results, stats, checkpoints
```

Notes:

- `images/`, `sparse/`, and `gsplat_results/` do not need to exist before the first run.
- `sparse/points.ply` is produced by the VGGT stage and then reused by the PPM stage.
- If a scene does not contain ground-truth test images, the pipeline still renders novel views but skips reference-image metrics.

## 2. System Requirements

- Linux
- NVIDIA GPU
- CUDA-compatible PyTorch environment
- A working CUDA toolkit / `nvcc` visible to the environment for `gsplat` extension compilation

All experiments and internal validation for this repository were tested on an NVIDIA RTX A6000 GPU.

## 3. Install The Environment

We recommend Conda for reproducibility.

If the unified environment in this README does not solve cleanly on your machine, use the original environment setup procedures from the two upstream components instead:

- `vggt/README.md`
- `gsplat/README.md`

In that fallback workflow, configure the `VGGT` and `gsplat` environments separately first, then return to this repository and run the unified pipeline script.

### Option A: Conda

From the repository root:

```bash
conda env create -f environment.yaml
conda activate naka-gs
pip install git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5
pip install git+https://github.com/harry7557558/fused-bilagrid@90f9788e57d3545e3a033c1038bb9986549632fe
pip install git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787
pip install ppisp @ git+https://github.com/nv-tlabs/ppisp@v1.0.0
```

If your Conda solver is slow, you can use:

```bash
conda env create -f environment.yaml --solver=libmamba
```

### Option B: Pip

If you already have a matching CUDA PyTorch installation:

```bash
pip install -r requirements.txt
```

## 4. Download The VGGT Checkpoint

The repository does not include the `VGGT` model weight. Download the official checkpoint and place it at:

```text
vggt/checkpoint/model.pt
```

Official model page:

- https://huggingface.co/facebook/VGGT-1B

Direct checkpoint URL:

- https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt

Example:

```bash
mkdir -p vggt/checkpoint
wget -O vggt/checkpoint/model.pt \
  https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt
```

## 5. Naka Checkpoint

By default, the pipeline looks for the Naka checkpoint at:

```text
outputs/naka/checkpoints/latest.pth
```

## 6. Prepare The Scene

Put your scene under `data/` or any other location you prefer. The important part is that `--scene_dir` points to the scene root.

Example:

```text
/path/to/naka-gs/data/Scene/
├── train/
├── transforms_train.json
├── transforms_test.json
└── test/        # optional
```

`train/` is required.  
`transforms_train.json` is required when using `--pose-source replace`.  
`transforms_test.json` is required when using `--render-traj-path testjson`.

## 7. Reproduce The Unified Pipeline Command

From the repository root, run:

```bash
python run_lowlight_reconstruction.py \
  --scene_dir /path/to/naka-gs/data/Your_Scene \
  --pose-source replace \
  --render-traj-path testjson \
  --disable-viewer \
  --ppm-enable \
  --ppm-dense-points-path sparse/points.ply \
  --ppm-align-mode none \
  --ppm-voxel-size 0.01 \
  --ppm-tau0 0.005 \
  --ppm-beta 0.01 \
  --ppm-iters 6
```

This command runs the full pipeline:

1. Low-light `train/` images are enhanced into `images/`.
2. `VGGT` reconstructs the scene and writes `sparse/` plus `sparse/points.ply`.
3. `gsplat` uses `PPM` to preprocess `sparse/points.ply`, then trains and renders the target trajectory from `transforms_test.json`.

## 8. Example With A Local Conda Python Path

If you want to use a specific Python interpreter inside a Conda environment, the command is equivalent to:

```bash
/path/to/conda/env/bin/python /path/to/naka-gs/run_lowlight_reconstruction.py \
  --scene_dir /path/to/naka-gs/data/Your_Scene \
  --pose-source replace \
  --render-traj-path testjson \
  --disable-viewer \
  --ppm-enable \
  --ppm-dense-points-path sparse/points.ply \
  --ppm-align-mode none \
  --ppm-voxel-size 0.01 \
  --ppm-tau0 0.005 \
  --ppm-beta 0.01 \
  --ppm-iters 6
```

## 9. Main Outputs

After a successful run, check:

- `data/Laboratory/images/` for enhanced images
- `data/Laboratory/sparse/` for the VGGT sparse reconstruction
- `data/Laboratory/gsplat_results/` for rendered views, metrics, checkpoints, and logs
- `data/Laboratory/gsplat_results/pipeline_summary.json` for a stage-by-stage summary

## 10. Useful Variants

### Reuse Existing Enhanced Images

```bash
python run_lowlight_reconstruction.py \
  --scene_dir /path/to/scene \
  --skip_naka
```

### Reuse Existing Sparse Reconstruction

```bash
python run_lowlight_reconstruction.py \
  --scene_dir /path/to/scene \
  --skip_naka \
  --skip_vggt
```

### Disable PPM

```bash
python run_lowlight_reconstruction.py \
  --scene_dir /path/to/scene \
  --ppm-enable false
```

## 11. Common Issues

### `FileNotFoundError: Naka checkpoint is required`

Provide `--naka_ckpt /path/to/latest.pth`, or place the checkpoint at the default path shown above.

### `No enhanced images found`

Make sure `train/` contains valid image files and the Naka stage finished successfully.

### `PPM dense point cloud is missing: .../sparse/points.ply`

This usually means the VGGT stage did not finish successfully, so `sparse/points.ply` was not generated.

### `torch.cuda.is_available() is False`

The `gsplat` stage requires a visible CUDA GPU.

### `gsplat` spends a long time on the first run

This is expected when the CUDA extension is compiled for the first time.

## 12. Minimal Checklist Before Running

- Environment created successfully
- `vggt/checkpoint/model.pt` downloaded
- Naka checkpoint available, either at the default path or via `--naka_ckpt`
- Scene directory contains `train/`
- `transforms_train.json` exists for `--pose-source replace`
- `transforms_test.json` exists for `--render-traj-path testjson`
