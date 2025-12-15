"""GUI-friendly Dr.Bokeh rendering engine (preprocess once, render many).

This module wraps the existing forward-rendering pipeline used by
`app/bokeh_rendering/Inference.py` into a reusable API suitable for a GUI:

- One-time preprocessing per image (RGB -> disp/alpha -> layered RGBAD)
- Fast re-rendering for varying focal plane and blur intensity (K)
- Disk caching to speed up repeated runs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class PreprocessResult:
    """Holds preprocessed representations for a single RGB image."""

    rgb: np.ndarray  # H x W x 3 float32 in [0, 1]
    disp: np.ndarray  # H x W x 1 float32 in [0, 1]
    alpha: np.ndarray  # H x W x 1 float32 in [0, 1]
    fg_rgbad: np.ndarray  # H x W x 5 float32
    bg_rgbad: np.ndarray  # H x W x 5 float32


class BokehEngine:
    """A small wrapper around DPT/LDF/LaMa + DScatter for GUI use."""

    def set_lens(self, lens: int) -> None:
        """Update the lens kernel size used by the scatter renderer.

        The scatter renderer is constructed lazily and cached. Changing `lens`
        requires rebuilding it, so this method resets the internal renderer
        instance.

        Args:
            lens: Largest lens kernel size (must be odd).

        Raises:
            ValueError: If `lens` is even.
        """
        if lens % 2 == 0:
            raise ValueError(f"`lens` must be odd, got {lens}.")
        if int(lens) == int(self.lens):
            return
        self.lens = int(lens)
        # Force rebuild on next render.
        self._renderer = None

    def __init__(
        self,
        lens: int = 71,
        cache_dir: Optional[Path] = None,
        device: Optional[torch.device] = None,
        *,
        use_cuda_scatter: Optional[bool] = None,
        gpu_occlusion: bool = True,
    ) -> None:
        """Initialize the engine.

        Args:
            lens: Largest lens kernel size (must be odd).
            cache_dir: Directory to store preprocessing cache.
            device: Torch device. Defaults to CUDA if available.
            use_cuda_scatter: Whether to use the CUDA scatter backend.
                - None: auto (use CUDA if available and `scatter_cuda` imports)
                - True: force CUDA (raise if unavailable)
                - False: force CPU (slow but works without `scatter_cuda`)
            gpu_occlusion: Whether to use the occlusion-aware variant (recommended).
        """
        if lens % 2 == 0:
            raise ValueError(f"`lens` must be odd, got {lens}.")

        self.device: torch.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.lens: int = lens
        self.use_cuda_scatter: Optional[bool] = use_cuda_scatter
        self.gpu_occlusion: bool = bool(gpu_occlusion)
        # Construct the scatter renderer lazily so `Preprocess` can run even if
        # `scatter_cuda` isn't built yet (matches upstream behavior).
        self._renderer = None
        # Device to run the scatter renderer on. If the CUDA extension is not available,
        # we force CPU to avoid running Python-loop fallback kernels on the GPU (which is
        # typically even slower due to kernel launch overhead).
        self._scatter_device: torch.device = self.device

        # Standard cache location: bokeh_rendering_and_focus_stacking_suite/outputs/bokeh_rendering/cache/
        base_cache = cache_dir or Path(__file__).resolve().parents[2] / "outputs" / "bokeh_rendering" / "cache"
        self.cache_dir: Path = base_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_renderer(self) -> Any:
        """Create (or return) the DScatter renderer."""
        if self._renderer is not None:
            return self._renderer

        # Lazy import: avoids importing `scatter_cuda` on preprocess.
        from app.bokeh_rendering.DScatter.scatter import Multi_Layer_Renderer  # type: ignore

        renderer = Multi_Layer_Renderer(
            self.lens,
            use_cuda=self.use_cuda_scatter,
            gpu_occlusion=self.gpu_occlusion,
        ).to(self.device)

        # If the renderer fell back to CPU scatter, force the module + subsequent inference
        # inputs onto CPU for sanity/perf.
        try:
            use_cuda_selected = bool(renderer.renderer.use_cuda)
        except Exception:
            use_cuda_selected = False

        if not use_cuda_selected:
            self._scatter_device = torch.device("cpu")
            renderer = renderer.to(self._scatter_device)
        else:
            self._scatter_device = self.device

        self._renderer = renderer
        return renderer

    @staticmethod
    def _read_rgb(rgb_path: Path) -> np.ndarray:
        """Read RGB as float32 in [0, 1]."""
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {rgb_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb

    @staticmethod
    def _read_alpha(alpha_path: Path) -> np.ndarray:
        """Read an RGBA image and return alpha as float32 HxWx1 in [0, 1]."""
        rgba = cv2.imread(str(alpha_path), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise FileNotFoundError(f"Failed to read alpha: {alpha_path}")
        if rgba.ndim != 3 or rgba.shape[2] < 4:
            raise ValueError("Alpha input must be an RGBA image (alpha in the last channel).")
        alpha = rgba[..., 3].astype(np.float32) / 255.0
        return alpha[..., None]

    @staticmethod
    def _read_disp_npz(disp_path: Path) -> np.ndarray:
        """Read disparity from an .npz containing key `data` and return HxWx1 float32."""
        data = np.load(str(disp_path))
        if "data" not in data:
            raise KeyError('Disp npz must contain key "data".')
        disp = data["data"]
        if disp.ndim != 2:
            raise ValueError(f"Disp must be HxW, got shape {disp.shape}.")
        disp = disp.astype(np.float32)
        return disp[..., None]

    @staticmethod
    def _normalize_01(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        mn = float(np.min(x))
        mx = float(np.max(x))
        if mx - mn < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return (x - mn) / (mx - mn)

    @staticmethod
    def _resize_hw1(
        x: np.ndarray,
        *,
        h: int,
        w: int,
        interpolation: int,
    ) -> np.ndarray:
        """Resize an HxW or HxWx1 array to (h, w) and return HxWx1.

        Note:
            For best visual quality, use:
            - `cv2.INTER_CUBIC` for disparity/depth-like continuous maps
            - `cv2.INTER_NEAREST` for masks/alpha to avoid edge bleeding (halos)
        """
        if x.ndim == 3:
            if x.shape[2] != 1:
                raise ValueError(f"Expected HxWx1, got shape {x.shape}.")
            x2 = x[..., 0]
        elif x.ndim == 2:
            x2 = x
        else:
            raise ValueError(f"Expected 2D or 3D (HxW or HxWx1), got shape {x.shape}.")
        resized = cv2.resize(x2, (w, h), interpolation=interpolation).astype(np.float32)
        return resized[..., None]

    def _cache_key(
        self,
        rgb_path: Path,
        *,
        threshold: float,
        fg_erode: int,
        fg_iters: int,
        inpaint_kernel: int,
        mask_filter: bool,
    ) -> str:
        """Build a cache key for preprocessing outputs.

        Important:
            Preprocessing results depend not only on the input image, but also on
            user-controlled preprocessing parameters (e.g., mask filtering). Those
            parameters must be part of the key; otherwise toggling a setting would
            incorrectly re-use stale cached results.
        """
        st = rgb_path.stat()
        thr = float(threshold)
        mf = 1 if bool(mask_filter) else 0
        return (
            f"{rgb_path.stem}"
            f"__{int(st.st_mtime)}__{st.st_size}"
            f"__thr{thr:.4f}__er{int(fg_erode)}__it{int(fg_iters)}__ik{int(inpaint_kernel)}__mf{mf}"
        )

    def preprocess(
        self,
        rgb_path: Path,
        disp_path: Optional[Path] = None,
        alpha_path: Optional[Path] = None,
        *,
        force: bool = False,
        threshold: float = 0.1,
        fg_erode: int = 5,
        fg_iters: int = 2,
        inpaint_kernel: int = 7,
        mask_filter: bool = False,
    ) -> PreprocessResult:
        """Preprocess an image into layered RGBAD (with caching).

        Args:
            rgb_path: Path to RGB image.
            disp_path: Optional disparity `.npz` with key `data`.
            alpha_path: Optional RGBA image path to use its alpha channel.
            force: If True, ignore cache and recompute.
            threshold: Foreground mask threshold.
            fg_erode: Foreground dilation/erosion setting used in preprocessing.
            fg_iters: Iterations for morphological operations.
            inpaint_kernel: Kernel size for depth inpainting.

        Returns:
            PreprocessResult with rgb/disp/alpha and fg/bg RGBAD layers.
        """
        rgb_path = rgb_path.resolve()
        cache_key = self._cache_key(
            rgb_path,
            threshold=threshold,
            fg_erode=fg_erode,
            fg_iters=fg_iters,
            inpaint_kernel=inpaint_kernel,
            mask_filter=mask_filter,
        )
        cache_file = self.cache_dir / f"{cache_key}.npz"

        if cache_file.exists() and not force and disp_path is None and alpha_path is None:
            cached = np.load(str(cache_file))
            return PreprocessResult(
                rgb=cached["rgb"].astype(np.float32),
                disp=cached["disp"].astype(np.float32),
                alpha=cached["alpha"].astype(np.float32),
                fg_rgbad=cached["fg_rgbad"].astype(np.float32),
                bg_rgbad=cached["bg_rgbad"].astype(np.float32),
            )

        rgb = self._read_rgb(rgb_path)
        h, w = rgb.shape[:2]

        # Import lazily: `image_preprocessing.py` constructs model inferencers at import time.
        from app.bokeh_rendering.image_preprocessing import (  # type: ignore
            RGBAD2layers,
            depth_predict,
            salient_segmentation,
        )

        if disp_path is not None:
            disp = self._read_disp_npz(disp_path)
        else:
            disp = depth_predict(rgb)
            if disp.ndim == 2:
                disp = disp[..., None]
        disp = self._resize_hw1(disp, h=h, w=w, interpolation=cv2.INTER_CUBIC)
        disp = self._normalize_01(disp).astype(np.float32)

        if alpha_path is not None:
            alpha = self._read_alpha(alpha_path)
        else:
            alpha = salient_segmentation(rgb)
            if alpha.ndim == 2:
                alpha = alpha[..., None]
        # Masks should be resized with nearest-neighbor to preserve crisp edges.
        alpha = self._resize_hw1(alpha, h=h, w=w, interpolation=cv2.INTER_NEAREST)
        alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)

        params: dict[str, Any] = {
            "threshold": float(threshold),
            "fg_erode": int(fg_erode),
            "fg_iters": int(fg_iters),
            "inpaint_kernel": int(inpaint_kernel),
            "mask_filter": bool(mask_filter),
        }
        layers = RGBAD2layers(rgb, alpha, disp, params)
        fg_rgbad = layers["fg_rgbad"].astype(np.float32)
        bg_rgbad = layers["bg_rgbad"].astype(np.float32)

        # Only cache when inputs are fully automatic (to avoid mixing user-provided disp/alpha).
        if disp_path is None and alpha_path is None:
            np.savez_compressed(
                str(cache_file),
                rgb=rgb.astype(np.float32),
                disp=disp.astype(np.float32),
                alpha=alpha.astype(np.float32),
                fg_rgbad=fg_rgbad.astype(np.float32),
                bg_rgbad=bg_rgbad.astype(np.float32),
            )

        return PreprocessResult(rgb=rgb, disp=disp, alpha=alpha, fg_rgbad=fg_rgbad, bg_rgbad=bg_rgbad)

    def render(
        self,
        pre: PreprocessResult,
        *,
        focal: float,
        k_blur: float,
        gamma: float = 2.2,
        offset: float = 0.0,
        highlight: bool = False,
        highlight_threshold: float = 200.0 / 255.0,
        highlight_enhance_ratio: float = 0.2,
    ) -> np.ndarray:
        """Render bokeh for a preprocessed image.

        Args:
            pre: Preprocess result.
            focal: Focal plane in normalized disparity space [0, 1].
            k_blur: Blur strength (K) scaling factor.
            gamma: Gamma correction exponent.
            offset: Offset used in gamma correction.

        Returns:
            HxWx3 float32 image in [0, 1] (clipped).
        """
        rgb = pre.rgb
        disp = pre.disp
        fg_rgbad = pre.fg_rgbad.copy()
        bg_rgbad = pre.bg_rgbad.copy()

        # Optional highlight enhancement (matches `Inference.py` behavior)
        if highlight:
            disp = pre.disp
            rgb = pre.rgb
            focal_f = float(focal)

            # out-of-focus areas
            mask1 = np.clip(np.tanh(200.0 * (np.abs(disp - focal_f) ** 2 - 0.01)), 0.0, 1.0)
            # highlight areas
            mask2 = np.clip(np.tanh(10.0 * (rgb - float(highlight_threshold))), 0.0, 1.0)
            mask = mask1 * mask2

            fg_rgbad[..., :3] = fg_rgbad[..., :3] * (1.0 + mask * float(highlight_enhance_ratio))
            bg_rgbad[..., :3] = bg_rgbad[..., :3] * (1.0 + mask * float(highlight_enhance_ratio))

        # Gamma correction before rendering (matches `Inference.py` behavior)
        fg_rgbad[..., :3] = (fg_rgbad[..., :3] + float(offset)) ** float(gamma)
        bg_rgbad[..., :3] = (bg_rgbad[..., :3] + float(offset)) ** float(gamma)

        with torch.no_grad():
            renderer = self._get_renderer()
            bokeh = renderer.inference(
                [fg_rgbad, bg_rgbad],
                float(k_blur),
                float(focal),
                device=self._scatter_device,
            )

        # Invert gamma correction
        bokeh = (bokeh ** (1.0 / float(gamma))) - float(offset)
        return np.clip(bokeh.astype(np.float32), 0.0, 1.0)

    def coc_map(self, pre: PreprocessResult, *, focal: float, k_blur: float) -> np.ndarray:
        """Compute a Circle-of-Confusion (CoC) magnitude map for visualization.

        This mirrors the renderer's internal definition used by the scatter model:
        CoC is proportional to the magnitude of *relative disparity* w.r.t. the
        chosen focal plane, scaled by the blur strength K.

        Concretely:
            coc = abs(disp - focal) * k_blur

        Args:
            pre: Preprocess result containing normalized disparity in [0, 1].
            focal: Focal plane in normalized disparity space [0, 1].
            k_blur: Blur strength (K) scaling factor.

        Returns:
            HxW float32 CoC magnitude map (arbitrary units consistent with the renderer).
        """
        disp = pre.disp
        if disp.ndim != 3 or disp.shape[2] != 1:
            raise ValueError(f"Expected disp as HxWx1, got shape {disp.shape}.")
        coc = np.abs(disp[..., 0] - float(focal)) * float(k_blur)
        return coc.astype(np.float32)

    def focus_mask(
        self,
        pre: PreprocessResult,
        *,
        focal: float,
        k_blur: float,
        coc_threshold: float = 1.0,
    ) -> np.ndarray:
        """Compute an in-focus boolean mask from CoC.

        Args:
            pre: Preprocess result.
            focal: Focal plane in normalized disparity space [0, 1].
            k_blur: Blur strength (K) scaling factor.
            coc_threshold: Pixels with CoC <= threshold are treated as in-focus.

        Returns:
            HxW bool mask where True indicates in-focus region.
        """
        if coc_threshold <= 0:
            raise ValueError(f"`coc_threshold` must be > 0, got {coc_threshold}.")
        coc = self.coc_map(pre, focal=float(focal), k_blur=float(k_blur))
        return (coc <= float(coc_threshold))


