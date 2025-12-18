"""Sharpness map computation for focus stacking.

We use a simple, effective baseline: local energy of the Laplacian levels,
computed as a small Gaussian blur of the squared Laplacian responses.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def compute_sharpness_map(
    laplacian_pyramids: list[list[np.ndarray]],
    output_dir: str | None = None,
    definition: str = "GaussianBlur(L^2)",
) -> list[list[np.ndarray]]:
    """Compute per-level sharpness maps from Laplacian pyramids.

    Args:
        laplacian_pyramids: Laplacian pyramids for all images.
        output_dir: Optional directory to save debug visualizations.
        definition: Sharpness definition ("L", "GaussianBlur(L)", "GaussianBlur(L^2)",
                    "Tenengrad+Blur", "Variance(L)", "SML+Blur").

    Returns:
        Sharpness maps with the same nested structure as `laplacian_pyramids`.
    """

    num_images = len(laplacian_pyramids)
    if num_images == 0:
        return []

    num_levels = len(laplacian_pyramids[0])

    sharpness_maps = []

    for i in range(num_images):
        lap_pyr = laplacian_pyramids[i]
        level_sharpness = []

        for k in range(num_levels):
            Lk = lap_pyr[k]

            # Compute sharpness metric based on definition
            if definition == "L":
                # Just magnitude of Laplacian
                Ek = np.abs(Lk)
            elif definition == "GaussianBlur(L)":
                # Smoothed magnitude
                Ek = cv2.GaussianBlur(np.abs(Lk), (3, 3), 0)
            elif definition == "Tenengrad+Blur":
                # Tenengrad (Sobel energy) + Blur
                gx = cv2.Sobel(Lk, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(Lk, cv2.CV_32F, 0, 1, ksize=3)
                E = gx**2 + gy**2
                Ek = cv2.GaussianBlur(E, (3, 3), 0)
            elif definition == "Variance(L)":
                # Local variance: Blur(L^2) - (Blur(L))^2
                mean_L = cv2.GaussianBlur(Lk, (3, 3), 0)
                mean_L2 = cv2.GaussianBlur(Lk**2, (3, 3), 0)
                Ek = mean_L2 - mean_L**2
                Ek = np.maximum(Ek, 0)
            elif definition == "SML+Blur":
                # Sum Modified Laplacian + Blur
                # ML = |d2L/dx2| + |d2L/dy2|
                k_x = np.array([[-1, 2, -1]], dtype=np.float32)
                k_y = np.array([[-1], [2], [-1]], dtype=np.float32)
                Lx = cv2.filter2D(Lk, cv2.CV_32F, k_x)
                Ly = cv2.filter2D(Lk, cv2.CV_32F, k_y)
                ML = np.abs(Lx) + np.abs(Ly)
                Ek = cv2.GaussianBlur(ML, (3, 3), 0)
            else: # "GaussianBlur(L^2)" or default
                # Smoothed squared Laplacian (local energy)
                Ek = cv2.GaussianBlur(Lk * Lk, (3, 3), 0)
            
            level_sharpness.append(Ek)

        sharpness_maps.append(level_sharpness)

    # print("Sharpness maps shape:", [[sharpness_maps[i][k].shape for k in range(num_levels)] for i in range(num_images)])

    # Save sharpness maps for debugging, if requested.
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        for i in range(num_images):
            for k in range(num_levels):
                sharp_map = sharpness_maps[i][k]
                # Normalize for visualization
                sharp_map_norm = cv2.normalize(sharp_map, None, 0, 255, cv2.NORM_MINMAX)
                sharp_map_uint8 = sharp_map_norm.astype(np.uint8)
                output_path = os.path.join(output_dir, f"image_{i}_level_{k}_sharpness.png")
                cv2.imwrite(output_path, sharp_map_uint8)

    return sharpness_maps

