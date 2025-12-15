"""Focus stacking CLI entry point.

Note:
    The GUI (`gui/gui.py`) is the recommended way to run the demos. This module is
    kept as a lightweight CLI for the focus stacking pipeline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.focus_stacking.fusion import fuse_pyramids_and_reconstruct
from app.focus_stacking.mask import build_masks
from app.focus_stacking.preprocess import preprocess_image_stack
from app.focus_stacking.pyramids import build_pyramids_stack
from app.focus_stacking.sharpness import compute_sharpness_map


def run_focus_stacking(*, dataset_dir: Path, output_dir: Path, levels: int = 4) -> Path:
    """Run the focus stacking pipeline on a dataset directory.

    Args:
        dataset_dir: Directory containing an image stack (e.g., multiple focused shots).
        output_dir: Output directory where the fused image will be written.
        levels: Number of pyramid levels.

    Returns:
        Path to the written fused image.
    """
    base_name = dataset_dir.name

    print("Preprocessing image stack...")
    images = preprocess_image_stack(str(dataset_dir))

    print("Building pyramids...")
    gaussian_dir = output_dir / "gaussian_pyramids" / base_name
    laplacian_dir = output_dir / "laplacian_pyramids" / base_name
    _, laplacian_pyrs, top_gaussians = build_pyramids_stack(
        images,
        levels,
        gaussian_pyramid_dir=str(gaussian_dir),
        laplacian_pyramid_dir=str(laplacian_dir),
    )

    print("Computing sharpness maps...")
    sharp_dir = output_dir / "sharpness_maps" / base_name
    sharpness_maps = compute_sharpness_map(laplacian_pyrs, output_dir=str(sharp_dir))

    print("Building decision masks...")
    smoothed_masks = build_masks(sharpness_maps, sigma=1.2, ksize=7)

    print("Fusing pyramids and reconstructing fused image...")
    fused_dir = output_dir / "fused_pyramids" / base_name
    fused_image = fuse_pyramids_and_reconstruct(
        laplacian_pyrs,
        top_gaussians,
        smoothed_masks,
        top_fusion_method="max",
        output_dir=str(fused_dir),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{base_name}_fused.png"
    cv2.imwrite(str(output_path), fused_image.astype(np.uint8))
    print(f"Saved fused image to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Example (run from repo root):
    # python -m app.focus_stacking.cli_main
    project_root = Path(__file__).resolve().parents[2]
    dataset_dir = project_root / "Imgs" / "focus_stacking"
    output_dir = project_root / "outputs" / "focus_stacking"

    name = input(f"Enter dataset folder name under '{dataset_dir}': ").strip()
    run_focus_stacking(dataset_dir=dataset_dir / name, output_dir=output_dir)