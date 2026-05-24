"""Canonical focus stacking runner."""

from __future__ import annotations

from pathlib import Path


class DependencyError(RuntimeError):
    """Raised when required runtime dependencies are not installed."""


def _import_focus_modules():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise DependencyError("missing Python module `cv2`") from exc

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise DependencyError("missing Python module `numpy`") from exc

    from app.focus_stacking.fusion import fuse_pyramids_and_reconstruct
    from app.focus_stacking.mask import build_masks, build_raw_masks
    from app.focus_stacking.preprocess import preprocess_image_stack
    from app.focus_stacking.pyramids import build_pyramids_stack
    from app.focus_stacking.sharpness import compute_sharpness_map

    return {
        "cv2": cv2,
        "np": np,
        "fuse_pyramids_and_reconstruct": fuse_pyramids_and_reconstruct,
        "build_masks": build_masks,
        "build_raw_masks": build_raw_masks,
        "preprocess_image_stack": preprocess_image_stack,
        "build_pyramids_stack": build_pyramids_stack,
        "compute_sharpness_map": compute_sharpness_map,
    }


def run_focus_stacking(
    *,
    dataset_dir: Path,
    output_path: Path,
    levels: int = 3,
    mask: str = "soft",
    top: str = "mean",
    sharpness: str = "Tenengrad+Blur",
) -> Path:
    """Run the focus stacking pipeline and write one fused image."""
    mods = _import_focus_modules()
    cv2 = mods["cv2"]
    np = mods["np"]

    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)

    if levels < 1:
        raise ValueError(f"`levels` must be >= 1, got {levels}.")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    print("Preprocessing image stack...")
    images = mods["preprocess_image_stack"](str(dataset_dir))

    print("Building pyramids...")
    _gaussian_pyrs, laplacian_pyrs, top_gaussians = mods["build_pyramids_stack"](images, levels)

    print("Computing sharpness maps...")
    sharpness_maps = mods["compute_sharpness_map"](laplacian_pyrs, definition=sharpness)

    print(f"Building {mask} masks...")
    if mask == "soft":
        masks = mods["build_masks"](sharpness_maps, sigma=1.2, ksize=7)
    elif mask == "hard-min":
        masks = mods["build_raw_masks"](sharpness_maps, mode="min")
    elif mask == "hard":
        masks = mods["build_raw_masks"](sharpness_maps, mode="max")
    else:
        raise ValueError(f"Unsupported mask mode: {mask}")

    print(f"Fusing pyramids (top={top})...")
    fused_image = mods["fuse_pyramids_and_reconstruct"](
        laplacian_pyrs,
        top_gaussians,
        masks,
        top_fusion_method=top,
        output_dir=None,
    )
    if fused_image is None:
        raise RuntimeError("Focus stacking returned no fused image.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.clip(fused_image, 0.0, 255.0).astype(np.uint8))
    print(f"Saved fused image to: {output_path}")
    return output_path
