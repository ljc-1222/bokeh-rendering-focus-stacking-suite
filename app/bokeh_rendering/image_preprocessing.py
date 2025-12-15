"""Preprocessing utilities for the forward bokeh rendering pipeline.

This module builds a two-layer representation (foreground/background) from an RGB image:

- Predict (or load) disparity via DPT
- Predict (or load) alpha matte via LDF
- Inpaint disocclusions in the background via LaMa

Important:
    This file initializes model inferencers at import time (see the globals near the bottom).
    That behavior is relied upon by the original codebase to amortize model load costs.
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)

from salient import Salient_Inference
from rgb_inpainting import RGB_Inpainting_Inference
from depth_predict import Depth_Inference
import bilateral_median_filter

import cv2


def RGBAD2layers(rgb: np.ndarray, mask: np.ndarray, disp: np.ndarray, params: dict) -> dict[str, np.ndarray]:
    """Convert RGB + alpha mask + disparity into foreground/background RGBAD layers.

    Args:
        rgb: RGB image (H x W x 3).
        mask: Alpha/mask (H x W x 1).
        disp: Disparity (H x W x 1).
        params: Dict of preprocessing parameters.

    Returns:
        A dict containing `fg_rgbad` and `bg_rgbad` (each H x W x 5).
    """
    assert rgb.shape[-1] == 3, f"RGB should have 3 channels ({rgb.shape[-1]})"
    assert mask.shape[-1] == 1, f"mask should have 1 channel ({mask.shape[-1]})"
    assert disp.shape[-1] == 1, f"disp should have 1 channel ({disp.shape[-1]})"

    h, w = rgb.shape[:2]

    threshold = params["threshold"]
    fg_erode = params["fg_erode"]
    fg_iters = params["fg_iters"]
    inpaint_kernel = params["inpaint_kernel"]

    # Optional (slow) bilateral-median mask denoising before thresholding.
    # Off by default to keep the preprocessing fast.
    mask_filter = bool(params.get("mask_filter", False))

    fg_rgbad = process_fg(
        rgb,
        mask,
        disp,
        threshold=threshold,
        fg_erode=fg_erode,
        iterations=fg_iters,
        filter_input=mask_filter,
        inpaint_kernel=inpaint_kernel,
    )
    bg_rgbad = process_bg(rgb, mask, disp, threshold=threshold, filter_input=mask_filter)

    return {
        "fg_rgbad": fg_rgbad,
        "bg_rgbad": bg_rgbad,
    }


def RGBD2layers(rgb: np.ndarray, disp: np.ndarray, params: dict) -> dict[str, np.ndarray]:
    """Build layered representation from RGB + disparity (alpha obtained via saliency).

    Args:
        rgb: RGB image (H x W x 3).
        disp: Disparity map (H x W x 1).
        params: Dict containing 'Salient' key.

    Returns:
        A dict containing `fg_rgbad` and `bg_rgbad` (each H x W x 5).
    """
    assert rgb.shape[-1] == 3, f"Input rgb should have 3 channels ({rgb.shape[-1]})"
    assert disp.shape[-1] == 1, f"Input disp should have 1 channel ({disp.shape[-1]})"
    assert "Salient" in params, "Which Salient algorithm?"

    h, w = rgb.shape[:2]
    rgbd = np.concatenate([rgb, disp], axis=2)

    salient_mask = salient_segmentation(rgbd, params)
    if len(salient_mask.shape) == 2:
        salient_mask = salient_mask[..., None]

    return RGBAD2layers(rgb, salient_mask, disp, params)


def RGB2layers(rgb: np.ndarray, params: dict) -> dict[str, np.ndarray]:
    """Build layered representation from RGB (disparity is predicted internally).

    Args:
        rgb: RGB image (H x W x 3).
        params: Dict containing 'Salient' key.

    Returns:
        A dict containing `fg_rgbad` and `bg_rgbad` (each H x W x 5).
    """
    assert rgb.shape[-1] == 3, f"Input rgb should have 3 channels ({rgb.shape[-1]})"
    assert "Salient" in params, "Which Salient algorithm?"

    disp = depth_predict(rgb)

    if len(disp.shape) == 2:
        disp = disp[..., None]

    disp = (disp - disp.min()) / (disp.max() - disp.min())

    return RGBD2layers(rgb, disp, params)


def preprocess_mask(
    mask: np.ndarray,
    threshold: float = 0.1,
    filter_input: bool = False,
    kernel_size: int = 11,
    sigma_s: float = 4.0,
    sigma_r: float = 0.9,
) -> np.ndarray:
    """Threshold (and optionally denoise) the alpha mask.

    Ref: https://cs.brown.edu/courses/csci1290/labs/lab_bilateral/index.html
         https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=b086cfe529d132feb7accf10f4c35555bf9a96bb

    Args:
        mask: Mask as H x W x 1 numpy array.
        threshold: Binarization threshold.
        filter_input: If True, run bilateral-median filtering before thresholding.
        kernel_size: Filter kernel size (must be odd).
        sigma_s: Spatial sigma for the filter.
        sigma_r: Range sigma for the filter.

    Returns:
        Thresholded (and optionally filtered) mask as H x W x 1.
    """
    assert kernel_size % 2 == 1, "Kernel should be odd number"

    if filter_input:
        # NOTE: `bilateral_median_filter` is imported as a module; the filtering function
        # lives at `bilateral_median_filter.filter(...)`. The current call path is kept
        # unchanged to preserve runtime behavior.
        ret = bilateral_median_filter.filter(mask, kernel_size=kernel_size, sigma_s=sigma_s, sigma_r=sigma_r)
    else:
        ret = mask.copy()

    ret[ret < threshold] = 0.0
    ret[ret > threshold] = 1.0

    if len(ret.shape) == 2:
        ret = ret[..., None]

    return ret


def process_fg(
    rgb: np.ndarray,
    mask: np.ndarray,
    disp: np.ndarray,
    threshold: float = 0.1,
    fg_erode: int = 5,
    iterations: int = 3,
    filter_input: bool = False,
    inpaint_kernel: int = 11,
) -> np.ndarray:
    """Compute the foreground RGBAD layer (mask + inpainted disparity near boundaries).

    Args:
        rgb: RGB image (H x W x 3).
        mask: Alpha mask (H x W x 1).
        disp: Disparity map (H x W x 1).
        threshold: Mask binarization threshold.
        fg_erode: Foreground erosion size.
        iterations: Number of morphological iterations.
        filter_input: Whether to apply bilateral-median filtering.
        inpaint_kernel: Inpainting kernel size.

    Returns:
        Foreground RGBAD layer (H x W x 5).
    """
    h, w = rgb.shape[:2]

    hard_mask = preprocess_mask(mask, threshold, filter_input)

    # inpaint_mask = erode(hard_mask, size=fg_erode, iterations=iterations)
    inpaint_mask = dilate(1.0 - hard_mask, size=fg_erode, iterations=iterations)

    if inpaint_mask.shape != disp.shape:
        inpaint_mask = cv2.resize(inpaint_mask, (w, h))

    if len(inpaint_mask.shape) == 2:
        inpaint_mask = inpaint_mask[..., None]

    # inpaint_disp = disp * inpaint_mask
    # inpaint_mask = ((1.0-inpaint_mask) * 255.0).astype(np.uint8)
    inpaint_mask = (inpaint_mask * 255.0).astype(np.uint8)
    inpaint_disp = cv2.inpaint(
        (disp * 65535).astype(np.uint16),
        inpaint_mask,
        inpaint_kernel,
        cv2.INPAINT_TELEA,
    ) / 65535.0
    inpaint_disp = inpaint_disp[..., None]

    # inpaint_disp = depth_inpainting(inpaint_disp, 1.0-inpaint_mask)

    fg_rgbad = np.concatenate([rgb, mask, inpaint_disp], axis=2)
    # fg_rgbad = np.concatenate([rgb, hard_mask, inpaint_disp], axis=2)

    return fg_rgbad


def process_bg(
    rgb: np.ndarray,
    mask: np.ndarray,
    disp: np.ndarray,
    threshold: float,
    filter_input: bool = False,
) -> np.ndarray:
    """Compute the background RGBAD layer (RGB + disparity are inpainted behind foreground).

    Args:
        rgb: RGB image (H x W x 3).
        mask: Alpha mask (H x W x 1).
        disp: Disparity map (H x W x 1).
        threshold: Mask binarization threshold.
        filter_input: Whether to apply bilateral-median filtering.

    Returns:
        Background RGBAD layer (H x W x 5).
    """
    h, w = rgb.shape[:2]

    hard_mask = preprocess_mask(mask, threshold, filter_input)
    binary_salient_mask = hard_mask.copy()
    # binary_salient_mask[salient_mask>0.001] = 1.0
    # binary_salient_mask[salient_mask<0.001] = 0.0
    enlarge_mask = dilate(binary_salient_mask, size=5)
    bg_mask = 1.0 - enlarge_mask
    bg_rgb = rgb * bg_mask
    bg_depth = disp * bg_mask

    bg_rgb = RGB_inpainting(bg_rgb, enlarge_mask)
    bg_depth = depth_inpainting(bg_depth, enlarge_mask)

    # bg_depth = cv2.GaussianBlur(bg_depth, (5, 5),0)
    # if len(bg_depth.shape) == 2:
    #     bg_depth = bg_depth[..., None]

    bg_rgbad = np.concatenate([bg_rgb, np.ones((h, w, 1)), bg_depth], axis=2)
    return bg_rgbad


salient_inferencer = Salient_Inference()


def salient_segmentation(rgb: np.ndarray) -> np.ndarray:
    """Run the LDF saliency model and return an alpha-like saliency map.

    Args:
        rgb: RGB image (H x W x 3).

    Returns:
        Saliency map (H x W) or (H x W x 1).
    """
    return salient_inferencer.inference(rgb)


rgb_inpainting_inferencer = RGB_Inpainting_Inference()


def RGB_inpainting(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint masked RGB regions using LaMa.

    Args:
        rgb: RGB image (H x W x 3).
        mask: Inpainting mask (H x W x 1).

    Returns:
        Inpainted RGB image (H x W x 3).
    """
    inpainted = rgb_inpainting_inferencer.inference(rgb, mask)
    return inpainted


depth_inferencer = Depth_Inference()


def depth_predict(rgb: np.ndarray) -> np.ndarray:
    """Predict disparity using the depth model.

    Args:
        rgb: RGB image (H x W x 3).

    Returns:
        Disparity map (H x W).
    """
    return depth_inferencer.inference(rgb)


def depth_inpainting(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Inpaint depth/disparity using the RGB inpainting backend (LaMa).

    Args:
        depth: Depth map (H x W x 1).
        mask: Inpainting mask (H x W x 1).

    Returns:
        Inpainted depth map (H x W x 1).
    """
    assert depth.shape[-1] == 1 and mask.shape[-1] == 1, (
        f"depth and mask channels should be 1 ({depth.shape[-1]}) and 1 ({mask.shape[-1]})"
    )

    tmp_depth = np.repeat(depth, 3, axis=2)
    inpainted = rgb_inpainting_inferencer.inference(tmp_depth, mask)[..., :1]
    return inpainted


def dilate(img: np.ndarray, size: int = 5, iterations: int = 2) -> np.ndarray:
    """Dilate a single-channel image.

    Args:
        img: Input image (H x W x 1).
        size: Kernel size.
        iterations: Number of iterations.

    Returns:
        Dilated image (H x W x 1).
    """
    assert img.shape[-1] == 1, f"Dilation assumes img has 1 channel ({img.shape[-1]})"

    kernel = np.ones((size, size), np.float64)
    img_dilation = cv2.dilate(img, kernel, iterations=iterations)
    return img_dilation[..., None]


def erode(img: np.ndarray, size: int = 5, iterations: int = 2) -> np.ndarray:
    """Erode a single-channel image.

    Args:
        img: Input image (H x W x 1).
        size: Kernel size.
        iterations: Number of iterations.

    Returns:
        Eroded image (H x W x 1).
    """
    assert img.shape[-1] == 1, f"Erosion assumes img has 1 channel ({img.shape[-1]})"

    kernel = np.ones((size, size), np.float64)
    img_erosion = cv2.erode(img, kernel, iterations=iterations)
    return img_erosion[..., None]




if __name__ == "__main__":
    # Intentionally left empty.
    # This project is trimmed to forward rendering; paper/figure debug entrypoints were removed.
    pass
