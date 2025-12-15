"""Depth/disparity prediction wrapper around the vendored DPT implementation.

This module is a thin adapter used by the rendering pipeline. It modifies `sys.path`
to match the expected upstream DPT import layout.

Third-party notice:
    This project vendors code from "DPT" (`isl-org/DPT`) which is licensed under MIT.
    See:
    - `bokeh_rendering_and_focus_stacking_suite/app/bokeh_rendering/Depth/DPT/LICENSE`
    - `bokeh_rendering_and_focus_stacking_suite/THIRD_PARTY_NOTICES.md`
"""

import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
sys.path.insert(0, 'app/bokeh_rendering/Depth/DPT')

import glob

import torch
from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet


class Depth_Inference:
    def __init__(self):
        model_path = 'app/bokeh_rendering/Depth/DPT/weights/dpt_large-midas-2f21e586.pt'
        model_type = 'dpt_large'

        # set torch options
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        net_w, net_h = 384, 384
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )

        self.transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        # Upstream code was CUDA-only; we allow CPU fallback for environments without CUDA.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(self.device)


    def inference(self, rgb: np.array):
        """ Given the RGB np.array image, compute the salient map and return

        :param rgb: H x W x 3, np.array
        :returns: H x W x 1, np.array

        """
        assert rgb.shape[2] == 3, 'RGB input should have 3 channels({})'.format(rgb.shape[2])

        device = self.device
        model  = self.model

        img_input = self.transform({"image": rgb})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size= rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            return prediction

