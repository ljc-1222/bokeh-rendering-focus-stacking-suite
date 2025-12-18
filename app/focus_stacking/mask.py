"""Decision mask utilities for focus stacking.

The mask selects which source image contributes to each pyramid level/pixel.
We start with a hard argmax mask and optionally apply Gaussian smoothing and
normalization (soft masks) to reduce seams/halos.
"""

from __future__ import annotations

import cv2
import numpy as np


def build_raw_masks(sharpness_maps: list[list[np.ndarray]], mode: str = "max") -> list[list[np.ndarray]]:
    """Build hard (one-hot) decision masks from sharpness maps.
    
    Args:
        sharpness_maps: List of sharpness maps.
        mode: "max" for sharpest pixels (default), "min" for blurriest pixels.
    """
    num_images = len(sharpness_maps)
    if num_images == 0:
        return []
    
    num_levels = len(sharpness_maps[0])

    # Initialize raw_masks: a list for each image, initially containing None for each level
    raw_masks = [[None] * num_levels for _ in range(num_images)]

    # Process level by level
    for k in range(num_levels):
        # Stack all images' sharpness maps at level k: (N, H_k, W_k)
        Ek_stack = np.stack(
            [sharpness_maps[i][k] for i in range(num_images)],
            axis=0
        )

        # Find the index of the image with the maximum (or minimum) sharpness for each pixel
        if mode == "min":
            idx_selected = np.argmin(Ek_stack, axis=0)
        else:
            idx_selected = np.argmax(Ek_stack, axis=0)

        # Create a one-hot mask for each image
        for i in range(num_images):
            mask = (idx_selected == i).astype(np.float32)  # (H_k, W_k), 0/1
            raw_masks[i][k] = mask

    return raw_masks


def smooth_and_normalize_masks(
    raw_masks: list[list[np.ndarray]],
    sigma: float = 1.0,
    ksize: int = 5,
) -> list[list[np.ndarray]]:
    """Smooth raw masks and normalize so masks sum to ~1 per pixel/level."""
    num_images = len(raw_masks)
    if num_images == 0:
        return []

    num_levels = len(raw_masks[0])

    smoothed_masks = []
    for i in range(num_images):
        smoothed_masks.append([None] * num_levels)

    for k in range(num_levels):
        # First, apply Gaussian blur to each image's mask at this level
        blurred_list = []
        for i in range(num_images):
            m = raw_masks[i][k]
            # Ensure the mask is not empty and convert type to float32
            m = m.astype(np.float32)
            # Gaussian blur to avoid hard edges causing artifacts like jaggedness or halos during reconstruction
            mb = cv2.GaussianBlur(m, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
            blurred_list.append(mb)

        # Stack into (N, H, W)
        stack = np.stack(blurred_list, axis=0)

        # Normalize along the 0th dimension (image index)
        denom = np.sum(stack, axis=0, keepdims=True) + 1e-8  # Avoid division by zero
        norm_stack = stack / denom

        # Unpack back to list[list[np.ndarray]]
        for i in range(num_images):
            smoothed_masks[i][k] = norm_stack[i]

    return smoothed_masks

def build_masks(sharpness_maps: list[list[np.ndarray]], sigma: float = 1.0, ksize: int = 5, mode: str = "max") -> list[list[np.ndarray]]:
    """Convenience wrapper: sharpness maps -> soft/normalized masks."""
    raw_masks = build_raw_masks(sharpness_maps, mode=mode)
    return smooth_and_normalize_masks(raw_masks, sigma=sigma, ksize=ksize)