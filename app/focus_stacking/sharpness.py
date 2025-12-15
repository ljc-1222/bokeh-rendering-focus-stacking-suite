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
) -> list[list[np.ndarray]]:
    """Compute per-level sharpness maps from Laplacian pyramids.

    Args:
        laplacian_pyramids: Laplacian pyramids for all images.
        output_dir: Optional directory to save debug visualizations.

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

            # Compute sharpness metric
            # Using Gaussian smoothed squared Laplacian (local energy)
            # For color images, this produces a per-channel sharpness map
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

