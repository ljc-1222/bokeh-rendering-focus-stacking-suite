"""Focus stacking preprocessing utilities.

This module loads an image stack from disk, ensures a consistent spatial size,
optionally aligns the images, and caches the preprocessed stack to speed up
repeated runs.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import cv2
import numpy as np


def load_image_stack(folder_path: str, file_extension: str = "png") -> np.ndarray:
    """Load a stack of images from a folder.

    Args:
        folder_path: Path to the folder containing images.
        file_extension: Image file extension to load (without the dot).

    Returns:
        A float32 array with shape (N, H, W, 3) in BGR order, or an empty array if
        no images are found.
    """
    image_files = sorted(glob.glob(os.path.join(folder_path, f"*.{file_extension}")))
    image_stack: list[np.ndarray] = []
    for image_file in image_files:
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        if image is None:
            continue
        image_stack.append(image.astype(np.float32))

    if not image_stack:
        return np.array([])

    # Shape: (N, H, W, C=3)
    return np.stack(image_stack, axis=0)
    
def ensure_same_size(image_stack: np.ndarray) -> np.ndarray:
    """Resize all images in a stack to match the first image's size."""
    # print("Image stack size:", image_stack.size)

    if image_stack.size == 0:
        return image_stack
    
    target_shape = image_stack[0].shape
    resized_stack = []
    for image in image_stack:
        if image.shape != target_shape:
            resized_image = cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
            resized_stack.append(resized_image)
        else:
            resized_stack.append(image)

    return np.stack(resized_stack, axis=0)

def align_images(image_stack: np.ndarray) -> np.ndarray:
    """Align images in the stack using ECC maximization (affine warp)."""
    if image_stack.size == 0:
        return image_stack

    aligned_stack = []

    # Use the first image as the reference
    reference_image = image_stack[0]
    aligned_stack.append(reference_image)

    # Convert reference to grayscale for ECC
    if reference_image.ndim == 3 and reference_image.shape[2] > 1:
        ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = reference_image

    H, W = ref_gray.shape[:2]

    # Define the motion model
    # MOTION_AFFINE handles translation, rotation, scale, and shear
    warp_mode = cv2.MOTION_AFFINE

    # Set termination criteria
    number_of_iterations = 500
    termination_eps = 1e-5
    # Criteria: either 500 iterations or epsilon of 1e-5
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    for i in range(1, len(image_stack)):
        image = image_stack[i]
        
        # Convert to grayscale for ECC
        if image.ndim == 3 and image.shape[2] > 1:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image

        # Initialize warp matrix
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            # Run the ECC algorithm. The results are stored in warp_matrix.
            # findTransformECC finds the transform that maps the input image (img_gray) to the template (ref_gray)
            (_, warp_matrix) = cv2.findTransformECC(ref_gray, img_gray, warp_matrix, warp_mode, criteria)
            
            # Use warpAffine with the calculated matrix.
            aligned_image = cv2.warpAffine(
                image, 
                warp_matrix, 
                (W, H), 
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
        except cv2.error as e:
            print(f"Alignment failed for image {i}, keeping original. Error: {e}")
            aligned_image = image

        aligned_stack.append(aligned_image)

    return np.stack(aligned_stack, axis=0)


def preprocess_image_stack(folder_path: str, file_extension: str = "png", use_cache: bool = True) -> np.ndarray:
    """Load, resize, and align images from a folder (with optional caching).

    Args:
        folder_path: Folder containing an image stack.
        file_extension: Image file extension to load (without the dot).
        use_cache: If True, read/write an aligned stack cache under `outputs/`.

    Returns:
        A float32 array with shape (N, H, W, 3) in BGR order.
    """
    # Determine cache path (store under bokeh_rendering_and_focus_stacking_suite/outputs/focus_stacking/cache)
    base_name = os.path.basename(os.path.normpath(folder_path))
    project_root = Path(__file__).resolve().parents[2]  # bokeh_rendering_and_focus_stacking_suite/
    cache_dir = project_root / "outputs" / "focus_stacking" / "cache"
    cache_file = cache_dir / f"{base_name}_aligned.npy"

    if use_cache and cache_file.exists():
        try:
            return np.load(str(cache_file))
        except Exception as e:
            print(f"Failed to load cache: {e}. Reprocessing...")

    image_stack = load_image_stack(folder_path, file_extension)

    if image_stack.size == 0:
        raise ValueError(f"No images found in {folder_path} with extension .{file_extension}")

    image_stack = ensure_same_size(image_stack)
    image_stack = align_images(image_stack)

    if use_cache:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_file), image_stack)
        except Exception as e:
            print(f"Failed to save cache: {e}")

    return image_stack
 