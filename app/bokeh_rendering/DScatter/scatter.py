"""Occlusion-aware bokeh renderer (DScatter).

This module exposes the `Multi_Layer_Renderer` used by both the GUI engine and
the CLI pipeline.

Important:
    The scattering kernel can be implemented either:

    - in CUDA via the optional `scatter_cuda` extension (fast), or
    - in Python (slow reference) when CUDA extension is unavailable.

    The original upstream DrBokeh code supports both GPU and CPU execution. In
    this repo, we preserve that behavior by *not* importing the CUDA extension
    at module import-time. Instead, we attempt to enable it lazily.
"""

from __future__ import annotations

import time
import logging
import matplotlib.pyplot as plt
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils, transforms
import numpy as np

import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)

import CPU_scatter
from CPU_scatter import distance_kernel, lens_shape_mask


def _try_import_gpu_scatter() -> tuple[bool, object | None, str | None]:
    """Best-effort import of the CUDA scatter wrapper.

    Returns:
        (ok, module, error_message)
    """
    try:
        # Import locally to avoid raising `ModuleNotFoundError: scatter_cuda` at import-time.
        import GPU_scatter  # type: ignore

        return True, GPU_scatter, None
    except Exception as exc:  # noqa: BLE001 - we want to surface any import/build failures
        return False, None, str(exc)


class Multi_Layer_Renderer(nn.Module):
    def __init__(self, lens_mask: int, use_cuda: bool | None = None, gpu_occlusion: bool = True) -> None:
        super().__init__()

        self.gpu_occlusion = gpu_occlusion
        self.renderer = Scatter_Rendering(lens_mask, use_cuda=use_cuda, gpu_occlusion=gpu_occlusion)


    def forward(self, rgbad_layers, lens_effect, focal):
        """Render lens blur from a multi-layer RGBAD representation.

        Note:
            The disparity channel is expected to be *relative* to the focal plane.
        """
        n_layer = rgbad_layers.shape[1] // 5

        assert rgbad_layers.shape[1] % 5 == 0, f"Layer number needs to be multiple of 5 ({rgbad_layers.shape[1] % 5})"
        assert n_layer >= 0, "Layer number needs to be greater than 0"
        assert focal.shape == rgbad_layers.shape, "rgbad_layers should have the same shape with focal"
        assert (rgbad_layers.dtype == torch.float) and (lens_effect.dtype == torch.float), (
            "Blur rendering assumes input tensors to be float"
        )

        eps = 1e-6
        rgbad_layers = rgbad_layers - focal

        return self.defocus_render(rgbad_layers, lens_effect)


    def defocus_render(self, rgbad_layers, lens_effect):
        """Render lens blur with occlusion reasoning (multi-layer)."""
        if not self.gpu_occlusion:
            return self.no_occlusion_defocus_render(rgbad_layers, lens_effect)

        n_layer = rgbad_layers.shape[1] // 5

        assert rgbad_layers.shape[1] % 5 == 0, f"Layer number needs to be multiple of 5 ({rgbad_layers.shape[1] % 5})"
        assert n_layer >= 0, "Layer number needs to be greater than 0"
        assert (rgbad_layers.dtype == torch.float) and (lens_effect.dtype == torch.float), (
            "Blur rendering assumes input tensors to be float"
        )

        eps = 1e-8
        blur_list = [
            self.renderer.forward(rgbad_layers[:, 5 * i : 5 * (i + 1)], lens_effect)
            for i in range(n_layer)
        ]

        # we need to render the blur from the first layer to the last layer
        blur_rgb = blur_list[0][:, :-2]
        blur_w = blur_list[0][:, -2:-1]
        blur_occu = blur_list[0][:, -1:]

        blur_rgb = blur_rgb / (blur_w + eps) * blur_occu
        blur_occu = 1.0 - blur_occu

        for li in range(1, n_layer):
            layer_rgb = blur_list[li][:, :-2]
            layer_w = blur_list[li][:, -2:-1]
            layer_occu = blur_occu * blur_list[li][:, -1:]

            layer_blur = layer_rgb / (layer_w + eps) * layer_occu
            blur_occu = blur_occu * (1.0 - blur_list[li][:, -1:])

            blur_rgb = blur_rgb + layer_blur

        return blur_rgb


    def inference(
        self,
        rgbad_list: list[np.ndarray],
        lens_effect: float,
        focal: float,
        *,
        device: torch.device | None = None,
    ) -> np.ndarray:
        """Run inference from numpy inputs and return the rendered RGB image.

        Args:
            rgbad_list: List of HxWx5 float numpy arrays.
            lens_effect: Blur strength scaling factor.
            focal: Focal plane in normalized disparity space [0, 1].
            device: Optional torch device override. Defaults to the module device.
        """
        assert len(rgbad_list) > 0, "The RGBAD list input is empty"
        for i in range(len(rgbad_list)):
            assert rgbad_list[i].shape[2] == 5, "The RGBAD list elements should have 5 channel"

        inferred_device = device
        if inferred_device is None:
            # fall back to any parameter/buffer device; default to CPU if the module is empty
            try:
                inferred_device = next(self.parameters()).device
            except StopIteration:
                inferred_device = torch.device("cpu")

        rgbad_layer = torch.cat(
            [torch.tensor(rgbad.transpose(2, 0, 1))[None, ...] for rgbad in rgbad_list],
            dim=1,
        ).to(inferred_device).float()
        lens_effect_tensor = (torch.ones(1, 1, device=inferred_device) * lens_effect).float()

        # Build a per-layer focal tensor: only the disparity channel (last of each RGBAD block)
        # is shifted by the focal plane.
        focal_tensor = torch.zeros_like(rgbad_layer).to(rgbad_layer)
        _, c, h, w = rgbad_layer.shape
        n_layer = c // 5
        for i in range(n_layer):
            disp_ch = i * 5 + 4
            focal_tensor[:, disp_ch, :, :] = focal


        blur = self.forward(rgbad_layer, lens_effect_tensor, focal_tensor)
        blur = blur[0].detach().cpu().numpy().transpose((1, 2, 0))

        return blur


    def no_occlusion_defocus_render(self, rgbad_layers, lens_effect):
        """

        Render the lens blur given the multi-layer representation.
        Note, the disparity layer is the relative disparity!

        @param rgbad_layers:  [B, 5 * n, H, W] float tensor (n: layer #)
        @param lens_effects: [B, 1] float tensor
        @param focal: [B, 5 * n, H, W] float tensor

        @return: blur: [B, 3, H, W]
        """
        n_layer = rgbad_layers.shape[1] // 5

        assert rgbad_layers.shape[1] == 5, "Layer number needs to be 5({})".format(rgbad_layers.shape[1])
        assert n_layer >= 0, "Layer number needs to be greater than 0"
        assert (rgbad_layers.dtype == torch.float) and (lens_effect.dtype == torch.float), \
                "Blur rendering assumes input tensors to be float"

        rgbd_layer = torch.cat([rgbad_layers[:,:3], 
                                rgbad_layers[:,-1:]], dim=1) 
        eps = 1e-8
        bokeh = self.renderer.forward(rgbd_layer, lens_effect)
        return bokeh


class Scatter_Rendering(nn.Module):
    """Scatter rendering layer (depth-aware scattering blur)."""

    def __init__(self, lens_mask: int, use_cuda: bool | None = None, gpu_occlusion: bool = True) -> None:
        if lens_mask % 2 == 0:
            raise ValueError("Lens mask {} is even".format(lens_mask))
        super(Scatter_Rendering, self).__init__()

        self.lens, self.padding = lens_mask, torch.nn.ReplicationPad2d(lens_mask // 2)
        self.diskernel = nn.Parameter(distance_kernel(self.lens).float(), requires_grad=False)
        self.lens_mask = nn.Parameter(lens_shape_mask(self.lens).float(), requires_grad=False)

        # Decide backend.
        # - use_cuda=None means "auto": use CUDA only if (a) CUDA is available and (b) the extension imports.
        # - use_cuda=True means "force": raise if not available.
        # - use_cuda=False means "force CPU".
        if use_cuda is None:
            want_cuda = torch.cuda.is_available()
        else:
            want_cuda = bool(use_cuda)

        ok, gpu_mod, err = (False, None, None)
        if want_cuda:
            ok, gpu_mod, err = _try_import_gpu_scatter()

        if want_cuda and not ok:
            if use_cuda is True:
                raise RuntimeError(
                    "Requested CUDA scattering, but GPU backend failed to import. "
                    "This usually means `scatter_cuda` is not built/installed or CUDA/toolchain mismatches.\n"
                    f"Import error: {err}"
                )
            # auto-mode fallback
            want_cuda = False

        self.use_cuda = want_cuda

        if self.use_cuda:
            assert gpu_mod is not None
            if gpu_occlusion:
                self.scatter = gpu_mod.Scatter.apply  # type: ignore[attr-defined]
            else:
                self.scatter = gpu_mod.Scatter_no_occlusion.apply  # type: ignore[attr-defined]
        else:
            self.scatter = CPU_scatter.Scatter(self.lens)


    def forward(self, x, lens_effects):
        """Forward pass for scattering blur.

        Args:
            x: Tensor shaped `B x 4/5 x H x W` (RGBD or RGBAD) with *relative* disparity.
        """
        b, c, h, w = x.shape
        assert c == 4 or c == 5, "Scattering Input is wrong. {}".format(c)

        if not self.use_cuda and c == 4:
            # CPU implementation expects RGBAD; synthesize alpha=1 for RGBD input.
            alpha = torch.ones((b, 1, h, w), dtype=x.dtype, device=x.device)
            x = torch.cat([x[:, :3], alpha, x[:, 3:4]], dim=1)

        if self.use_cuda:
            ret = self.scatter(x, lens_effects, self.diskernel, self.lens_mask)
        else:
            ret = self.scatter(x, lens_effects)

        # ret = ret/(ret[:, -1:] + 1e-8)
        return ret


    def inference(self, rgbd: np.array, lens_effect: float, focal: float):
        """
        Given the numpy array of RGBD, float lens_effect, render the results
        @param rgbd: RGBD numpy array. NOTE, the D is disparity
        @param lens_effect: lens strength
        @param focal: camera's focal length
        @return: blur image
        """
        h, w, c = rgbd.shape
        assert c == 4, 'Input channel should be 4({})'.format(c)

        rgb  = rgbd[..., :3]
        disp = rgbd[..., -1:] - focal
        rgbd = np.concatenate([rgb, disp], axis=2)

        rgbd_layer = torch.tensor(rgbd.transpose(2,0,1))[None, ...].cuda().float()
        lens_effect_tensor = (torch.ones(1, 1) * lens_effect).cuda().float()

        blur = self.forward(rgbd_layer, lens_effect_tensor)
        blur = blur[0].detach().cpu().numpy().transpose((1, 2, 0))

        return blur


if __name__ == '__main__':
    renderer = Multi_Layer_Renderer(21)
    h, w = 256, 256

    rgbad_layers = torch.randn(5, 5, 5, h, w).cuda()
    lens_effect  = torch.ones(5, 1).cuda()

    renderer = renderer.cuda()
    blur = renderer.forward(rgbad_layers, lens_effect, 0.3)
