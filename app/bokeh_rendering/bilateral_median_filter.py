"""Bilateral median filtering utilities.

This module provides a slow but simple reference implementation used during
mask preprocessing. It is not performance-critical in the default pipeline
configuration (filtering is typically disabled).
"""

from __future__ import annotations

from functools import reduce

import numpy as np

def filter(
    img: np.ndarray,
    kernel_size: int = 11,
    sigma_s: float = 4.0,
    sigma_r: float = 0.5,
) -> np.ndarray:
    """Apply a bilateral median-like filter to a single-channel image.

    Note:
        This implementation is intentionally slow and is mainly intended as a
        reference / optional denoiser.

    Args:
        img: Input image (HxW or HxWx1).
        kernel_size: Odd kernel size.
        sigma_s: Spatial sigma.
        sigma_r: Range sigma.

    Returns:
        Filtered image, shape HxW.
    """
    img = np.squeeze(img)

    h, w = img.shape[:2]
    pad_size = kernel_size // 2
    padded_mask = np.pad(img, pad_size, "edge")
    midpt = pad_size

    ax = np.arange(-midpt, midpt + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    spatial_term = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_s**2))

    padded_maskh_patches = rolling_window(padded_mask, [kernel_size, kernel_size], [1, 1])
    pH, pW = padded_maskh_patches.shape[:2]

    output = img.copy()
    for pi in range(pH):
        for pj in range(pW):
            patch = padded_maskh_patches[pi, pj]
            depth_order = patch.ravel().argsort()
            patch_midpt = patch[kernel_size // 2, kernel_size // 2]
            range_term = np.exp(-((patch - patch_midpt) ** 2) / (2.0 * sigma_r**2))

            coef = spatial_term * range_term

            if coef.sum() == 0:
                output[pi, pj] = patch_midpt
            else:
                coef = coef / (coef.sum())
                coef_order = coef.ravel()[depth_order]
                cum_coef = np.cumsum(coef_order)
                ind = np.digitize(0.5, cum_coef)
                output[pi, pj] = patch.ravel()[depth_order][ind]

    return output


def rolling_window(a: np.ndarray, window: list[int], strides: list[int]) -> np.ndarray:
    """Create a rolling window view of an array using stride tricks.

    Args:
        a: Input array.
        window: Window size for each dimension.
        strides: Stride size for each dimension.

    Returns:
        Rolling window view of the array.
    """
    assert len(a.shape) == len(window) == len(strides), "'a', 'window', 'strides' dimension mismatch"

    def shape_fn(i: int, w: int, s: int) -> int:
        return (a.shape[i] - w) // s + 1

    shape = [shape_fn(i, w, s) for i, (w, s) in enumerate(zip(window, strides))] + list(window)

    def acc_shape(i: int) -> int:
        if i + 1 >= len(a.shape):
            return 1
        else:
            return reduce(lambda x, y: x * y, a.shape[i + 1 :])

    _strides = [acc_shape(i) * s * a.itemsize for i, s in enumerate(strides)] + list(a.strides)

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=_strides)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test = plt.imread("img/test.png")[..., 0]

    kernel_size = 15
    sigma_r_ = [0.1, 0.5, 0.9, 1.0]
    sigma_s_ = [4, 8, 16, 32, 64]

    for sigma_r in sigma_r_:
        for sigma_s in sigma_s_:
            filted = filter(
                test,
                kernel_size=kernel_size,
                sigma_r=sigma_r,
                sigma_s=sigma_s,
            )
            diff = np.abs(test - filted)
            print(diff.sum(), diff.min(), diff.max())

            plt.imsave(f"img/{kernel_size:02d}_{sigma_r:.3f}_{sigma_s:.3f}_filted.png", filted, cmap="gray")
            plt.imsave(f"img/{kernel_size:02d}_{sigma_r:.3f}_{sigma_s:.3f}_diff.png", diff)
