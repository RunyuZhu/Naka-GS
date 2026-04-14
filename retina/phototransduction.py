import numpy as np
from typing import Literal, Optional
import cv2


Array = np.ndarray
_Mode = Literal["log", "naka"]
_Out  = Literal["zero_mean", "0_1"]


class Phototransduction:
    def __init__(
        self,
        log_sigma: Optional[float] = None,
        mode: _Mode = "log",
        naka_n: float = 1.0, 
        naka_sigma: Optional[float] = None,
        clip_percentile: Optional[float] = 99.9,
        local_radius: int = 5,
        per_channel: bool = True, 
        eps: float = 1e-3,
        out_mode: _Out = "zero_mean",
        sym_clip_tau: float = 3,
        out_method: str = "symmetric", 
        out_dtype = np.float32,
    ):
        self.log_sigma = log_sigma
        self.mode = mode
        self.naka_n = float(naka_n)
        self.naka_sigma = naka_sigma 
        self.clip_percentile = clip_percentile
        self.local_radius = int(local_radius)
        self.per_channel = bool(per_channel)
        self.eps = float(eps)
        self.out_mode = out_mode
        self.sym_clip_tau = float(sym_clip_tau)
        self.out_method = out_method
        self.out_dtype = out_dtype

    # ---------- public API ----------
    def __call__(self, I: Array) -> Array:
        x = self._to_float01(I)

        if self.mode == "log":
            effective_log_sigma = self._auto_log_sigma(x) if self.log_sigma is None else self.log_sigma
            x = self._log_compress(x, effective_log_sigma)
        elif self.mode == "naka":
            effective_naka_sigma = self._auto_naka_sigma(x) if self.naka_sigma is None else self.naka_sigma
            x = self._naka_rushton(x, n=self.naka_n, sigma=effective_naka_sigma)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.out_mode == "0_1":
            if self.out_method == "symmetric":
                x = self._to_01_symmetric(x, tau=self.sym_clip_tau)
            elif self.out_method == "percentile":
                x = self._to_01_percentile(x, lower_pct=2.5, upper_pct=97.5)
            elif self.out_method == "linear":
                x = self._to_01_linear(x)
            elif self.out_method == "histogram":
                x = self._to_01_histogram(x)
            else:
                raise ValueError(f"Unknown out_method: {self.out_method}")
        elif self.out_mode == "zero_mean":
            pass
        else:
            raise ValueError(f"Unknown out_mode: {self.out_mode}")

        return x.astype(self.out_dtype, copy=False)
    
    @staticmethod
    def _to_01_symmetric(x: Array, tau: float = 3.0) -> Array:
        x_clip = np.clip(x, -tau, tau)
        return (x_clip + tau) / (2.0 * tau)
    
    @staticmethod
    def _to_01_percentile(x: Array, lower_pct: float = 1.0, upper_pct: float = 99.0) -> Array:
        lower = np.percentile(x, lower_pct)
        upper = np.percentile(x, upper_pct)
        
        x_clip = np.clip(x, lower, upper)
        return (x_clip - lower) / (upper - lower + 1e-12)
    
    @staticmethod
    def _to_01_linear(x: Array) -> Array:
        x_min = x.min()
        x_max = x.max()
        
        if x_max - x_min < 1e-6:
            return np.zeros_like(x)
        
        return (x - x_min) / (x_max - x_min)
    
    @staticmethod
    def _to_01_histogram(x: Array) -> Array:
        x_min = x.min()
        x_max = x.max()
        
        if x_max - x_min < 1e-6:
            return np.zeros_like(x)
        
        x_norm = (x - x_min) / (x_max - x_min)
        x_uint8 = (x_norm * 255).astype(np.uint8)
        
        if len(x.shape) == 3:  
            x_yuv = cv2.cvtColor(x_uint8, cv2.COLOR_RGB2YUV)
            x_yuv[:,:,0] = cv2.equalizeHist(x_yuv[:,:,0])
            x_eq = cv2.cvtColor(x_yuv, cv2.COLOR_YUV2RGB)
        else: 
            x_eq = cv2.equalizeHist(x_uint8)
            
        return x_eq.astype(np.float32) / 255.0
    
    def _to_float01(self, I: Array) -> Array:
        if np.issubdtype(I.dtype, np.integer):
            maxv = np.iinfo(I.dtype).max
            x = I.astype(np.float32) / float(maxv)
            return np.clip(x, 0.0, 1.0)
        x = I.astype(np.float32, copy=False)
        if self.clip_percentile is None:
            maxv = float(np.max(x)) if x.size else 1.0
            if maxv <= 1.0 + 1e-6:
                return np.clip(x, 0.0, 1.0)
            return np.clip(x / (maxv + 1e-12), 0.0, 1.0)

        hi = np.percentile(x, self.clip_percentile)
        if hi <= 1e-12:
            return np.zeros_like(x, dtype=np.float32)
        return np.clip(x / hi, 0.0, 1.0)

    def _auto_log_sigma(self, x: Array) -> float:
        if x.ndim == 3:
            brightness = np.mean(x, axis=2)
        else:
            brightness = x
        
        median_brightness = np.median(brightness)
        
        median_brightness = np.clip(median_brightness, 0.05, 0.95)
        
        auto_sigma = median_brightness * 0.4
        
        auto_sigma = np.clip(auto_sigma, 0.02, 0.5)
        
        return float(auto_sigma)

    def _auto_naka_sigma(self, x: Array) -> float:
        if x.ndim == 3:
            brightness = np.mean(x, axis=2)
        else:
            brightness = x
        
        median_brightness = np.median(brightness)
        
        auto_sigma = median_brightness * 0.25
        
        auto_sigma = np.clip(auto_sigma, 0.01, 0.8)
        
        if median_brightness < 0.05:
            auto_sigma = max(auto_sigma, 0.05)
        
        return float(auto_sigma)

    @staticmethod
    def _log_compress(x: Array, sigma: float) -> Array:
        denom = np.log1p(1.0 / (sigma + 1e-12))
        return np.log1p(x / (sigma + 1e-12)) / (denom + 1e-12)

    @staticmethod
    def _naka_rushton(x: Array, n: float, sigma: float) -> Array:
        xn = np.power(np.clip(x, 0.0, None), n)
        sig = np.power(max(sigma, 1e-8), n)
        return xn / (xn + sig)

    @staticmethod
    def _zero_center(x: Array) -> Array:
        if x.ndim == 3:
            mu = np.mean(x, axis=(0, 1), keepdims=True)
        else:
            mu = np.mean(x, keepdims=True)
        return x - mu

    @staticmethod
    def _to_01_from_zero_mean(x: Array, tau: float = 3.0) -> Array:
        x_clip = np.clip(x, -tau, tau)
        return (x_clip + tau) / (2.0 * tau)

    def _gaussian_blur(self, x: Array, radius: int, per_channel: bool) -> Array:
        if radius <= 0:
            return x.copy()

        if x.ndim == 2:
            xx = x[..., None]
        else:
            xx = x

        if not per_channel and xx.shape[2] > 1:
            mean_ch = np.mean(xx, axis=2, keepdims=True)
            sm = self._gauss_sep(mean_ch, radius)
            sm = np.repeat(sm, xx.shape[2], axis=2)
            return sm.squeeze() if x.ndim == 2 else sm

        sm = self._gauss_sep(xx, radius)
        return sm.squeeze() if x.ndim == 2 else sm

    @staticmethod
    def _gauss_kernel1d(radius: int) -> Array:
        sigma = max(radius / 3.0, 1e-6)
        ax = np.arange(-radius, radius + 1, dtype=np.float32)
        k = np.exp(-0.5 * (ax / sigma) ** 2)
        k /= np.sum(k)
        return k.astype(np.float32)

    def _gauss_sep(self, x: Array, radius: int) -> Array:
        k = self._gauss_kernel1d(radius)
        y = self._conv1d_h(x, k)
        y = self._conv1d_v(y, k)
        return y

    @staticmethod
    def _pad_reflect(x: Array, pad: int, axis: int) -> Array:
        pad_width = [(0, 0)] * x.ndim
        pad_width[axis] = (pad, pad)
        return np.pad(x, pad_width, mode="reflect")

    def _conv1d_h(self, x: Array, k: Array) -> Array:
        pad = k.size // 2
        xp = self._pad_reflect(x, pad, axis=1)
        out = np.empty_like(xp[:, pad:-pad, :])
        for c in range(x.shape[2]):
            out[..., c] = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 1, xp[..., c])
        return out

    def _conv1d_v(self, x: Array, k: Array) -> Array:
        pad = k.size // 2
        xp = self._pad_reflect(x, pad, axis=0)
        out = np.empty_like(xp[pad:-pad, :, :])
        for c in range(x.shape[2]):
            out[..., c] = np.apply_along_axis(lambda r: np.convolve(r, k, mode="valid"), 0, xp[..., c])
        return out
