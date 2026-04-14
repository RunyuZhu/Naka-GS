import sys
import os
import math
import glob
import random
import argparse
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision.models import vgg19, VGG19_Weights
except Exception:
    vgg19 = None
    VGG19_Weights = None

# -----------------------------------------------------------------------------
# Try importing the provided Naka function.
# Supports either:
#   1) retina.phototransduction.Phototransduction
#   2) phototransduction.Phototransduction
# -----------------------------------------------------------------------------
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from retina.phototransduction import Phototransduction
except Exception:
    from phototransduction import Phototransduction


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def save_rgb_tensor(tensor: torch.Tensor, path: str) -> None:
    arr = tensor.detach().cpu().clamp(0, 1)
    if arr.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape: {tuple(arr.shape)}")

    if arr.shape[0] == 1:
        arr = (arr.squeeze(0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(arr, mode="L").save(path)
    elif arr.shape[0] == 3:
        arr = (arr.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(arr).save(path)
    else:
        raise ValueError(f"save_rgb_tensor only supports 1 or 3 channels, got: {arr.shape[0]}")


def load_torch_checkpoint(path: str, map_location) -> Dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    if img.dtype != np.float32:
        img = img.astype(np.float32) / 255.0
    if img.max() > 1.0:
        img = img / 255.0
    img = np.ascontiguousarray(img)
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().float()


def list_image_files(folder: str) -> List[str]:
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.PNG", "*.JPG", "*.JPEG"]
    files: List[str] = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def paired_paths(low_dir: str, normal_dir: str) -> List[Tuple[str, str]]:
    low_files = list_image_files(low_dir)
    normal_files = list_image_files(normal_dir)
    normal_map = {os.path.basename(p): p for p in normal_files}
    pairs = []
    for low_path in low_files:
        name = os.path.basename(low_path)
        if name in normal_map:
            pairs.append((low_path, normal_map[name]))
    if not pairs:
        raise RuntimeError(f"No paired files found between {low_dir} and {normal_dir}")
    return pairs


def ensure_min_size_pair(a: np.ndarray, b: np.ndarray, min_size: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = a.shape[:2]
    if h >= min_size and w >= min_size:
        return a, b
    scale = max(min_size / max(h, 1), min_size / max(w, 1))
    nh, nw = int(math.ceil(h * scale)), int(math.ceil(w * scale))
    a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return a, b


def random_rescale_pair(
    a: np.ndarray,
    b: np.ndarray,
    min_scale: float = 0.7,
    max_scale: float = 1.4,
    min_after_scale: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    scale = random.uniform(min_scale, max_scale)
    h, w = a.shape[:2]
    nh = max(min_after_scale, int(round(h * scale)))
    nw = max(min_after_scale, int(round(w * scale)))
    a = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return a, b


def random_crop_pair(a: np.ndarray, b: np.ndarray, crop_size: int) -> Tuple[np.ndarray, np.ndarray]:
    a, b = ensure_min_size_pair(a, b, crop_size)
    h, w = a.shape[:2]
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    a = a[top:top + crop_size, left:left + crop_size]
    b = b[top:top + crop_size, left:left + crop_size]
    return a, b


def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def edge_map(x: torch.Tensor) -> torch.Tensor:
    c = x.shape[1]
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    sobel_x = sobel_x.repeat(c, 1, 1, 1)
    sobel_y = sobel_y.repeat(c, 1, 1, 1)
    gx = F.conv2d(x, sobel_x, padding=1, groups=c)
    gy = F.conv2d(x, sobel_y, padding=1, groups=c)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def gaussian_window(
    window_size: int = 11,
    sigma: float = 1.5,
    channels: int = 3,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = torch.outer(g, g)
    w = w.view(1, 1, window_size, window_size)
    return w.repeat(channels, 1, 1, 1)



def gaussian_blur_tensor(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")
    c = x.shape[1]
    window = gaussian_window(kernel_size, sigma, c, x.device, x.dtype)
    return F.conv2d(x, window, padding=kernel_size // 2, groups=c)


def ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    c = x.shape[1]
    window = gaussian_window(window_size, 1.5, c, x.device, x.dtype)
    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=c)
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=c)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=c) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=c) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=c) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_n = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    ssim_d = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = ssim_n / (ssim_d + 1e-8)
    return 1.0 - ssim_map.mean()


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class NakaPairDataset(Dataset):
    """
    Directory layout:
      root/
        train/
          low/
          normal/
        val/
          low/
          normal/

    Files are paired by the same filename.
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        crop_size: int = 256,
        is_train: bool = True,
        cache_naka: bool = False,
        min_scale: float = 0.7,
        max_scale: float = 1.4,
    ) -> None:
        super().__init__()
        low_dir = os.path.join(root, split, "low")
        normal_dir = os.path.join(root, split, "normal")
        self.pairs = paired_paths(low_dir, normal_dir)
        self.crop_size = crop_size
        self.is_train = is_train
        self.cache_naka = cache_naka and (not is_train)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.naka_cache: Dict[str, np.ndarray] = {}

        self.naka_processor = Phototransduction(
            mode="naka",
            per_channel=True,
            naka_sigma=0.05,
            clip_percentile=99.9,
            out_mode="0_1",
            out_method="linear",
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def _apply_naka(self, low_rgb: np.ndarray, key: str) -> np.ndarray:
        if self.cache_naka and key in self.naka_cache:
            return self.naka_cache[key]

        low_bgr = cv2.cvtColor(low_rgb, cv2.COLOR_RGB2BGR)
        naka_bgr = self.naka_processor(low_bgr)
        naka_rgb = cv2.cvtColor(naka_bgr.astype(np.float32), cv2.COLOR_BGR2RGB)
        naka_rgb = np.clip(naka_rgb, 0.0, 1.0).astype(np.float32)

        if self.cache_naka:
            self.naka_cache[key] = naka_rgb
        return naka_rgb

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        low_path, gt_path = self.pairs[idx]
        low = load_rgb(low_path)
        gt = load_rgb(gt_path)

        if self.is_train:
            low, gt = random_rescale_pair(low, gt, self.min_scale, self.max_scale, min_after_scale=self.crop_size)
            low, gt = random_crop_pair(low, gt, self.crop_size)
            if random.random() < 0.5:
                low = np.ascontiguousarray(np.fliplr(low))
                gt = np.ascontiguousarray(np.fliplr(gt))
            if random.random() < 0.5:
                low = np.ascontiguousarray(np.flipud(low))
                gt = np.ascontiguousarray(np.flipud(gt))
            if random.random() < 0.5:
                low = np.ascontiguousarray(np.rot90(low))
                gt = np.ascontiguousarray(np.rot90(gt))
            cache_key = f"{low_path}_train_no_cache"
        else:
            cache_key = low_path

        naka = self._apply_naka(low, cache_key)
        low_t = to_tensor(low)
        gt_t = to_tensor(gt)
        naka_t = to_tensor(naka)

        return {
            "low": low_t,
            "naka": naka_t,
            "gt": gt_t,
            "name": os.path.basename(low_path),
            "hw": torch.tensor([low_t.shape[1], low_t.shape[2]], dtype=torch.int32),
        }


# -----------------------------------------------------------------------------
# Model blocks
# -----------------------------------------------------------------------------
class InputStandardizer(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(self.eps)
        return (x - mean) / std


class ConvAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        act: bool = True,
        use_norm: bool = True,
    ):
        super().__init__()
        if p is None:
            p = k // 2

        layers = [nn.Conv2d(in_ch, out_ch, k, s, p, bias=not use_norm)]

        if use_norm:
            groups = min(8, out_ch)
            while groups > 1 and out_ch % groups != 0:
                groups -= 1
            layers.append(nn.GroupNorm(groups, out_ch))

        if act:
            layers.append(nn.GELU())

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = ConvAct(ch, ch, 3, 1)
        self.conv2 = ConvAct(ch, ch, 3, 1, act=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return self.act(out + x)


class SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        mid = max(8, ch // r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, mid, 1),
            nn.GELU(),
            nn.Conv2d(mid, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_res: int = 2):
        super().__init__()
        blocks = [ConvAct(in_ch, out_ch, 3, 1)]
        for _ in range(num_res):
            blocks.append(ResidualBlock(out_ch))
        blocks.append(SEBlock(out_ch))
        self.block = nn.Sequential(*blocks)
        self.down = ConvAct(out_ch, out_ch, 3, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.block(x)
        down = self.down(feat)
        return feat, down


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, num_res: int = 2):
        super().__init__()
        self.reduce = ConvAct(in_ch + skip_ch, out_ch, 3, 1)
        blocks = []
        for _ in range(num_res):
            blocks.append(ResidualBlock(out_ch))
        blocks.append(SEBlock(out_ch))
        self.block = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.reduce(x)
        return self.block(x)


class ChromaGuidedUNet(nn.Module):
    """
    Input features:
      raw branch:  [low(3), naka(3), delta(3)]
      norm branch: [low_norm(3), naka_norm(3), delta_norm(3)]
      total input channels = 18

    Output:
      mul_map: [B,1,H,W], single-channel multiplicative correction
      add_map: [B,3,H,W], additive correction
      naka is decomposed into low/high frequency parts:
        naka = naka_lf + naka_hf
      correction is applied only on low-frequency content:
        base = naka_lf * mul_map + add_map
        enhanced = clamp(base + naka_hf, 0, 1)
    """
    def __init__(
        self,
        base_ch: int = 32,
        mul_range: float = 0.6,
        add_range: float = 0.25,
        hf_kernel_size: int = 5,
        hf_sigma: float = 1.0,
    ):
        super().__init__()
        self.mul_range = mul_range
        self.add_range = add_range
        self.hf_kernel_size = hf_kernel_size
        self.hf_sigma = hf_sigma
        self.input_std = InputStandardizer()

        self.stem = ConvAct(18, base_ch, 3, 1)
        self.down1 = DownBlock(base_ch, base_ch, num_res=2)
        self.down2 = DownBlock(base_ch, base_ch * 2, num_res=2)
        self.down3 = DownBlock(base_ch * 2, base_ch * 4, num_res=3)

        self.bottleneck = nn.Sequential(
            ConvAct(base_ch * 4, base_ch * 8, 3, 1),
            ResidualBlock(base_ch * 8),
            ResidualBlock(base_ch * 8),
            SEBlock(base_ch * 8),
        )

        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4, num_res=2)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2, num_res=2)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch, num_res=2)

        self.fuse = nn.Sequential(
            ConvAct(base_ch, base_ch, 3, 1),
            ResidualBlock(base_ch),
        )

        self.mul_head = nn.Conv2d(base_ch, 1, 3, 1, 1)
        self.add_head = nn.Conv2d(base_ch, 3, 3, 1, 1)

    def forward(self, low: torch.Tensor, naka: torch.Tensor) -> Dict[str, torch.Tensor]:
        delta = naka - low

        low_n = self.input_std(low)
        naka_n = self.input_std(naka)
        delta_n = self.input_std(delta)
        x = torch.cat([low, naka, delta, low_n, naka_n, delta_n], dim=1)

        x0 = self.stem(x)
        s1, d1 = self.down1(x0)
        s2, d2 = self.down2(d1)
        s3, d3 = self.down3(d2)

        b = self.bottleneck(d3)
        u3 = self.up3(b, s3)
        u2 = self.up2(u3, s2)
        u1 = self.up1(u2, s1)
        feat = self.fuse(u1)

        mul_res = torch.tanh(self.mul_head(feat)) * self.mul_range
        add_map = torch.tanh(self.add_head(feat)) * self.add_range
        mul_map = 1.0 + mul_res

        naka_lf = gaussian_blur_tensor(naka, kernel_size=self.hf_kernel_size, sigma=self.hf_sigma)
        naka_hf = naka - naka_lf

        base = naka_lf * mul_map + add_map
        enhanced = torch.clamp(base + naka_hf, 0.0, 1.0)
        return {
            "enhanced": enhanced,
            "mul_map": mul_map,
            "add_map": add_map,
        }


def adapt_mul_head_to_single_channel(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Backward compatibility:
    convert an old 3-channel mul_head checkpoint to the new single-channel mul_head
    by averaging the three output filters/biases.
    """
    adapted = dict(state_dict)

    if "mul_head.weight" in adapted and adapted["mul_head.weight"].ndim == 4 and adapted["mul_head.weight"].shape[0] == 3:
        adapted["mul_head.weight"] = adapted["mul_head.weight"].mean(dim=0, keepdim=True)

    if "mul_head.bias" in adapted and adapted["mul_head.bias"].ndim == 1 and adapted["mul_head.bias"].shape[0] == 3:
        adapted["mul_head.bias"] = adapted["mul_head.bias"].mean(dim=0, keepdim=True)

    return adapted


def load_model_state_flexible(model: nn.Module, ckpt_obj: Dict[str, torch.Tensor]) -> None:
    state_dict = ckpt_obj["model"] if "model" in ckpt_obj else ckpt_obj
    state_dict = adapt_mul_head_to_single_channel(state_dict)
    model.load_state_dict(state_dict, strict=True)


# -----------------------------------------------------------------------------
# Perceptual feature extractor
# -----------------------------------------------------------------------------
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_ids: Tuple[int, ...] = (3, 8, 17, 26)):
        super().__init__()
        self.enabled = vgg19 is not None
        self.layer_ids = layer_ids
        if not self.enabled:
            self.features = None
            self.mean = None
            self.std = None
            return

        try:
            model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        except Exception:
            model = vgg19(weights=None)
        self.features = model.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.features is None:
            return []
        x = (x - self.mean) / self.std
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_ids:
                feats.append(x)
        return feats


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------
class NakaCorrectionLoss(nn.Module):
    def __init__(
        self,
        lambda_rgb: float = 1.0,
        lambda_chroma: float = 0.5,
        lambda_ssim: float = 0.3,
        lambda_edge: float = 0.2,
        lambda_feat: float = 0.15,
        lambda_reg: float = 0.02,
        lambda_mse: float = 0.0,
        mse_on: str = "rgb",
    ):
        super().__init__()
        self.lambda_rgb = lambda_rgb
        self.lambda_chroma = lambda_chroma
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge
        self.lambda_feat = lambda_feat
        self.lambda_reg = lambda_reg
        self.lambda_mse = lambda_mse
        self.mse_on = mse_on
        self.vgg = VGGFeatureExtractor()

    def forward(self, pred_dict: Dict[str, torch.Tensor], gt: torch.Tensor, naka: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred = pred_dict["enhanced"]
        mul_map = pred_dict["mul_map"]
        add_map = pred_dict["add_map"]

        loss_rgb = charbonnier_loss(pred, gt) + 0.5 * F.l1_loss(pred, gt)

        pred_ycc = rgb_to_ycbcr(pred)
        gt_ycc = rgb_to_ycbcr(gt)
        loss_chroma = F.l1_loss(pred_ycc[:, 1:], gt_ycc[:, 1:]) + 0.2 * F.l1_loss(pred_ycc[:, :1], gt_ycc[:, :1])

        loss_ssim = ssim_loss(pred, gt)
        loss_edge = F.l1_loss(edge_map(pred), edge_map(gt))

        loss_feat = pred.new_tensor(0.0)
        pred_feats = self.vgg(pred)
        gt_feats = self.vgg(gt)
        if len(pred_feats) == len(gt_feats) and len(pred_feats) > 0:
            for pf, gf in zip(pred_feats, gt_feats):
                loss_feat = loss_feat + F.l1_loss(pf, gf)
            loss_feat = loss_feat / len(pred_feats)

        id_mul = F.l1_loss(mul_map, torch.ones_like(mul_map))
        id_add = F.l1_loss(add_map, torch.zeros_like(add_map))
        smooth_mul = F.l1_loss(mul_map[:, :, :, 1:], mul_map[:, :, :, :-1]) + F.l1_loss(mul_map[:, :, 1:, :], mul_map[:, :, :-1, :])
        smooth_add = F.l1_loss(add_map[:, :, :, 1:], add_map[:, :, :, :-1]) + F.l1_loss(add_map[:, :, 1:, :], add_map[:, :, :-1, :])
        improve_consistency = 0.1 * torch.relu(F.l1_loss(pred, gt) - F.l1_loss(naka, gt))
        loss_reg = id_mul + id_add + 0.5 * (smooth_mul + smooth_add) + improve_consistency

        if self.mse_on == "rgb":
            loss_mse = F.mse_loss(pred, gt)
        elif self.mse_on == "chroma":
            loss_mse = F.mse_loss(pred_ycc[:, 1:], gt_ycc[:, 1:])
        elif self.mse_on == "y":
            loss_mse = F.mse_loss(pred_ycc[:, :1], gt_ycc[:, :1])
        else:
            raise ValueError(f"Unsupported mse_on: {self.mse_on}")

        total = (
            self.lambda_rgb * loss_rgb
            + self.lambda_chroma * loss_chroma
            + self.lambda_ssim * loss_ssim
            + self.lambda_edge * loss_edge
            + self.lambda_feat * loss_feat
            + self.lambda_reg * loss_reg
            + self.lambda_mse * loss_mse
        )

        metrics = {
            "loss": float(total.detach().item()),
            "rgb": float(loss_rgb.detach().item()),
            "chroma": float(loss_chroma.detach().item()),
            "ssim": float(loss_ssim.detach().item()),
            "edge": float(loss_edge.detach().item()),
            "feat": float(loss_feat.detach().item()),
            "reg": float(loss_reg.detach().item()),
            "mse": float(loss_mse.detach().item()),
        }
        return total, metrics


# -----------------------------------------------------------------------------
# Validation / inference helpers
# -----------------------------------------------------------------------------
def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    return 10.0 * torch.log10(1.0 / (mse + 1e-8))


def make_naka_processor() -> Phototransduction:
    return Phototransduction(
        mode="naka",
        per_channel=True,
        naka_sigma=0.05,
        clip_percentile=99.9,
        out_mode="0_1",
        out_method="linear",
    )


@torch.no_grad()
def forward_full_or_tiled(
    model: nn.Module,
    low: torch.Tensor,
    naka: torch.Tensor,
    tile_size: int = 0,
    tile_overlap: int = 32,
) -> Dict[str, torch.Tensor]:
    _, _, h, w = low.shape
    if tile_size <= 0 or (h <= tile_size and w <= tile_size):
        return model(low, naka)

    step = max(tile_size - tile_overlap, 1)
    enhanced_acc = torch.zeros_like(naka)
    add_acc = torch.zeros_like(naka)
    weight_acc = torch.zeros_like(naka)

    b = low.shape[0]
    mul_acc = low.new_zeros((b, 1, h, w))
    mul_weight_acc = low.new_zeros((b, 1, h, w))

    for top in range(0, h, step):
        for left in range(0, w, step):
            bottom = min(top + tile_size, h)
            right = min(left + tile_size, w)
            top = max(0, bottom - tile_size)
            left = max(0, right - tile_size)

            low_tile = low[:, :, top:bottom, left:right]
            naka_tile = naka[:, :, top:bottom, left:right]
            pred = model(low_tile, naka_tile)

            weight = torch.ones_like(pred["enhanced"])
            mul_weight = torch.ones_like(pred["mul_map"])

            enhanced_acc[:, :, top:bottom, left:right] += pred["enhanced"] * weight
            mul_acc[:, :, top:bottom, left:right] += pred["mul_map"] * mul_weight
            add_acc[:, :, top:bottom, left:right] += pred["add_map"] * weight
            weight_acc[:, :, top:bottom, left:right] += weight
            mul_weight_acc[:, :, top:bottom, left:right] += mul_weight

    enhanced = enhanced_acc / weight_acc.clamp_min(1e-6)
    mul_map = mul_acc / mul_weight_acc.clamp_min(1e-6)
    add_map = add_acc / weight_acc.clamp_min(1e-6)
    enhanced = torch.clamp(enhanced, 0.0, 1.0)
    return {"enhanced": enhanced, "mul_map": mul_map, "add_map": add_map}


@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: NakaCorrectionLoss,
    loader: DataLoader,
    device: torch.device,
    save_dir: Optional[str] = None,
    max_save: int = 8,
    tile_size: int = 0,
    tile_overlap: int = 32,
) -> Dict[str, float]:
    model.eval()
    loss_sum = 0.0
    psnr_sum = 0.0
    count = 0
    saved = 0

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for batch in loader:
        low = batch["low"].to(device, non_blocking=True)
        naka = batch["naka"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)
        names = batch["name"]

        pred_dict = forward_full_or_tiled(model, low, naka, tile_size=tile_size, tile_overlap=tile_overlap)
        loss, _ = criterion(pred_dict, gt, naka)

        bs = low.size(0)
        loss_sum += float(loss.item()) * bs
        psnr_sum += float(psnr(pred_dict["enhanced"], gt).item()) * bs
        count += bs

        if save_dir is not None and saved < max_save:
            for i in range(bs):
                if saved >= max_save:
                    break
                stem, _ = os.path.splitext(names[i])
                sample_dir = os.path.join(save_dir, stem)
                os.makedirs(sample_dir, exist_ok=True)
                save_rgb_tensor(low[i], os.path.join(sample_dir, f"{stem}_low.JPG"))
                save_rgb_tensor(naka[i], os.path.join(sample_dir, f"{stem}_naka.JPG"))
                save_rgb_tensor(pred_dict["enhanced"][i], os.path.join(sample_dir, f"{stem}_enhanced.JPG"))
                save_rgb_tensor(gt[i], os.path.join(sample_dir, f"{stem}_gt.JPG"))
                save_rgb_tensor(pred_dict["mul_map"][i].clamp(0, 2) / 2.0, os.path.join(sample_dir, f"{stem}_mul_map_vis.JPG"))
                save_rgb_tensor((pred_dict["add_map"][i] + 0.25) / 0.5, os.path.join(sample_dir, f"{stem}_add_map_vis.JPG"))
                saved += 1

    return {
        "val_loss": loss_sum / max(count, 1),
        "val_psnr": psnr_sum / max(count, 1),
    }
class NakaCorrectionLossWithMasks(nn.Module):
    def __init__(self, base_loss: nn.Module, lambda_gray_edge: float = 0.5, lambda_bright: float = 0.8):
        """
        base_loss: 原始 NakaCorrectionLoss
        lambda_gray_edge: 灰度边缘 mask 权重
        lambda_bright: 亮区 mask 权重
        """
        super().__init__()
        self.base_loss = base_loss
        self.lambda_gray_edge = lambda_gray_edge
        self.lambda_bright = lambda_bright

    @staticmethod
    def compute_gray_laplacian_mask(img: torch.Tensor) -> torch.Tensor:
        """B x C x H x W -> gray edge mask B x 1 x H x W"""
        img_np = img.permute(0, 2, 3, 1).cpu().numpy()
        lap_masks = []
        for i in range(img.shape[0]):
            gray = 0.299*img_np[i,:,:,0] + 0.587*img_np[i,:,:,1] + 0.114*img_np[i,:,:,2]
            lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            lap = np.abs(lap)
            lap /= (lap.max() + 1e-8)
            lap = np.sqrt(lap)  # 压缩极端值
            lap_masks.append(lap)
        lap_masks = np.stack(lap_masks, axis=0)
        lap_masks = torch.from_numpy(lap_masks).float().unsqueeze(1).to(img.device)
        return lap_masks

    @staticmethod
    def compute_bright_mask(img: torch.Tensor, percentile: float = 0.85) -> torch.Tensor:
        """B x C x H x W -> bright mask B x 1 x H x W"""
        img_gray = 0.299*img[:,0:1] + 0.587*img[:,1:2] + 0.114*img[:,2:3]
        threshold = torch.quantile(img_gray.view(img.shape[0], -1), percentile, dim=1).view(-1,1,1,1)
        mask = (img_gray >= threshold).float()
        return mask

    def forward(self, pred_dict: Dict[str, torch.Tensor], gt: torch.Tensor, naka: torch.Tensor):
        # 原始 base loss
        total_loss, metrics = self.base_loss(pred_dict, gt, naka)

        pred = pred_dict["enhanced"]

        # 灰度边缘 mask
        gray_mask = self.compute_gray_laplacian_mask(gt)
        loss_gray = (gray_mask * torch.abs(pred - gt)).mean()

        # 亮区 mask
        bright_mask = self.compute_bright_mask(pred)
        loss_bright = (bright_mask * torch.abs(pred - gt)).mean()

        # 总 loss
        total_loss = total_loss + self.lambda_gray_edge * loss_gray + self.lambda_bright * loss_bright

        # 更新 metrics
        metrics["gray_edge"] = float(loss_gray.detach().item())
        metrics["bright_mask"] = float(loss_bright.detach().item())
        metrics["loss"] = float(total_loss.detach().item())

        return total_loss, metrics

# -----------------------------------------------------------------------------
# Training / inference
# -----------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    vis_dir = os.path.join(args.output_dir, "val_vis")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    train_set = NakaPairDataset(
        root=args.data_root,
        split="train",
        crop_size=args.crop_size,
        is_train=True,
        cache_naka=False,
        min_scale=args.train_min_scale,
        max_scale=args.train_max_scale,
    )
    val_set = NakaPairDataset(
        root=args.data_root,
        split="val",
        crop_size=args.crop_size,
        is_train=False,
        cache_naka=args.cache_naka,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    # Validation uses full-resolution images, so batch_size must stay at 1.
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = ChromaGuidedUNet(base_ch=args.base_ch, mul_range=args.mul_range, add_range=args.add_range, hf_kernel_size=args.hf_kernel_size, hf_sigma=args.hf_sigma).to(device)
    base_loss = NakaCorrectionLoss(
        lambda_rgb=args.lambda_rgb,
        lambda_chroma=args.lambda_chroma,
        lambda_ssim=args.lambda_ssim,
        lambda_edge=args.lambda_edge,
        lambda_feat=args.lambda_feat,
        lambda_reg=args.lambda_reg,
        lambda_mse=args.lambda_mse,
        mse_on=args.mse_on,
    ).to(device)

    criterion = NakaCorrectionLossWithMasks(
        base_loss=base_loss,
        lambda_gray_edge=1,  # 可调
        lambda_bright=0.8  # 可调
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    start_epoch = 1
    best_psnr = -1e9

    if args.resume_ckpt:
        ckpt = load_torch_checkpoint(args.resume_ckpt, map_location=device)
        load_model_state_flexible(model, ckpt)
        if not args.reset_optimizer:
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler"])
                except Exception as e:
                    print(f"[Warning] Failed to load scheduler state: {e}. Scheduler will be reinitialized.")
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_psnr = float(ckpt.get("best_psnr", -1e9))
        print(f"Loaded resume checkpoint: {args.resume_ckpt}")
    elif args.init_ckpt:
        ckpt = load_torch_checkpoint(args.init_ckpt, map_location=device)
        load_model_state_flexible(model, ckpt)
        print(f"Loaded init checkpoint: {args.init_ckpt}")

    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        running_loss = 0.0
        running_psnr = 0.0
        count = 0

        for batch in train_loader:
            low = batch["low"].to(device, non_blocking=True)
            naka = batch["naka"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                pred_dict = model(low, naka)
                loss, _ = criterion(pred_dict, gt, naka)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_psnr = psnr(pred_dict["enhanced"], gt)
            running_loss += float(loss.item()) * low.size(0)
            running_psnr += float(batch_psnr.item()) * low.size(0)
            count += low.size(0)

        scheduler.step()

        train_log = {
            "train_loss": running_loss / max(count, 1),
            "train_psnr": running_psnr / max(count, 1),
        }
        val_log = validate(
            model,
            criterion,
            val_loader,
            device,
            save_dir=os.path.join(vis_dir, f"epoch_{epoch:03d}"),
            max_save=4,
            tile_size=args.val_tile_size,
            tile_overlap=args.tile_overlap,
        )

        print(
            f"Epoch [{epoch:03d}/{end_epoch:03d}] "
            f"train_loss={train_log['train_loss']:.4f} "
            f"train_psnr={train_log['train_psnr']:.2f} "
            f"val_loss={val_log['val_loss']:.4f} "
            f"val_psnr={val_log['val_psnr']:.2f}"
        )

        latest_path = os.path.join(ckpt_dir, "latest.pth")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
                "best_psnr": best_psnr,
            },
            latest_path,
        )

        if val_log["val_psnr"] > best_psnr:
            best_psnr = val_log["val_psnr"]
            best_path = os.path.join(ckpt_dir, "best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "args": vars(args),
                    "best_psnr": best_psnr,
                },
                best_path,
            )
            print(f"Saved best checkpoint to: {best_path}")


@torch.no_grad()
def inference(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model = ChromaGuidedUNet(base_ch=args.base_ch, mul_range=args.mul_range, add_range=args.add_range, hf_kernel_size=args.hf_kernel_size, hf_sigma=args.hf_sigma).to(device)
    ckpt = load_torch_checkpoint(args.ckpt, map_location=device)
    load_model_state_flexible(model, ckpt)
    model.eval()

    naka_processor = make_naka_processor()
    paths = list_image_files(args.input_dir)

    for path in paths:
        low_rgb = load_rgb(path)
        low_float = low_rgb.astype(np.float32) / 255.0
        low_bgr = cv2.cvtColor(low_rgb, cv2.COLOR_RGB2BGR)
        naka_bgr = naka_processor(low_bgr)
        naka_rgb = cv2.cvtColor(naka_bgr.astype(np.float32), cv2.COLOR_BGR2RGB)
        naka_rgb = np.clip(naka_rgb, 0.0, 1.0).astype(np.float32)

        low_t = torch.from_numpy(np.ascontiguousarray(low_float)).permute(2, 0, 1).unsqueeze(0).float().to(device)
        naka_t = torch.from_numpy(np.ascontiguousarray(naka_rgb)).permute(2, 0, 1).unsqueeze(0).float().to(device)
        pred_dict = forward_full_or_tiled(
            model,
            low_t,
            naka_t,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
        )

        name = os.path.splitext(os.path.basename(path))[0]
        save_rgb_tensor(pred_dict["enhanced"][0], os.path.join(args.output_dir, f"{name}_enhanced.JPG"))
        #save_rgb_tensor(pred_dict["mul_map"][0].clamp(0, 2) / 2.0, os.path.join(args.output_dir, f"{name}_mul_vis.JPG"))
        #save_rgb_tensor((pred_dict["add_map"][0] + 0.25) / 0.5, os.path.join(args.output_dir, f"{name}_add_vis.JPG"))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Naka-guided color-correction network (multi-scale + adaptive input standardization)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "infer"])
    parser.add_argument("--data_root", type=str, default="./datasets/LOLv1")
    parser.add_argument("--input_dir", type=str, default="./test_images")
    parser.add_argument("--output_dir", type=str, default="./outputs/naka_color_correction_v2")
    parser.add_argument("--ckpt", type=str, default="./outputs/naka_color_correction_v2/checkpoints/best.pth")
    parser.add_argument("--resume_ckpt", type=str, default="", help="Resume training from a saved checkpoint and continue epoch count.")
    parser.add_argument("--init_ckpt", type=str, default="", help="Initialize model weights from a checkpoint and start a fresh optimization run.")
    parser.add_argument("--reset_optimizer", action="store_true", help="When used with --resume_ckpt, only load model weights and reset optimizer/scheduler.")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_naka", action="store_true")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--mul_range", type=float, default=0.6)
    parser.add_argument("--add_range", type=float, default=0.25)
    parser.add_argument("--hf_kernel_size", type=int, default=5, help="Odd Gaussian kernel size for low/high-frequency decomposition.")
    parser.add_argument("--hf_sigma", type=float, default=1.0, help="Gaussian sigma for low/high-frequency decomposition.")
    parser.add_argument("--train_min_scale", type=float, default=0.7)
    parser.add_argument("--train_max_scale", type=float, default=1.4)

    parser.add_argument("--val_tile_size", type=int, default=0, help="0 means full-resolution validation without tiling")
    parser.add_argument("--tile_size", type=int, default=0, help="0 means full-resolution inference without tiling")
    parser.add_argument("--tile_overlap", type=int, default=32)

    parser.add_argument("--lambda_rgb", type=float, default=1.0)
    parser.add_argument("--lambda_chroma", type=float, default=0.5)
    parser.add_argument("--lambda_ssim", type=float, default=0.3)
    parser.add_argument("--lambda_edge", type=float, default=0.2)
    parser.add_argument("--lambda_feat", type=float, default=0.15)
    parser.add_argument("--lambda_reg", type=float, default=0.02)
    parser.add_argument("--lambda_mse", type=float, default=0.0, help="Weight for extra MSE loss term. Keep small to avoid oversmoothing.")
    parser.add_argument("--mse_on", type=str, default="rgb", choices=["rgb", "chroma", "y"], help="Where to apply the extra MSE term.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.mode == "train":
        train(args)
    else:
        inference(args)
