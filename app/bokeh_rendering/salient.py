"""Saliency/alpha matte inference wrapper around the vendored LDF model.

The rendering pipeline uses this module to obtain an alpha-like mask for
foreground/background separation. The underlying network code lives under
`app/bokeh_rendering/Salient/LDF/` and is treated as vendored.
"""

import numpy as np
import torch

import cv2

from app.bokeh_rendering.Salient.LDF.train_fine.net import LDF


class Salient_Inference:
    def __init__(self):
        model_snapshot = 'app/bokeh_rendering/Salient/LDF/train_fine/out/model-40'
        resnet_path    = 'app/bokeh_rendering/Salient/LDF/res/resnet50-19c8e357.pth'

        # Minimal runtime config: training dataset + dataloaders are not needed for inference.
        self.cfg = _LDFConfig(snapshot=model_snapshot)
        self.net = LDF(self.cfg, resnet_path)

        self.net.train(False)
        # Upstream code was CUDA-only; we allow CPU fallback for environments without CUDA.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.normalize = Normalize(mean=self.cfg.mean, std=self.cfg.std)
        self.resize    = Resize(352, 352)
        self.totensor  = ToTensor()



    def inference(self, rgb: np.array):
        """Compute a saliency/alpha map from an RGB image."""
        assert rgb.shape[2] == 3, 'Salient inference input should have 3 channels({})'.format(rgb.shape[2])

        with torch.no_grad():
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255.0).astype(np.uint8)
            # image = torch.tensor(rgb.transpose((2,0,1)))[None, ...].cuda().float()

            H, W = rgb.shape[:2]
            image = self.transform(rgb).to(self.device).float()
            image, shape = image, (H, W)
            outb1, outd1, out1, outb2, outd2, out2 = self.net(image, shape)
            out  = out2
            pred = torch.sigmoid(out[0,0]).cpu().numpy()

            return pred


    def transform(self, img: np.array):
        """Transform an RGB array into the model's normalized tensor input."""
        img = self.normalize(img)
        img = self.resize(img)
        img = self.totensor(img)[None, ...]
        return img



"""
---------------------------------------------------
"""
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, image, mask=None, body=None, detail=None):
        image = (image - self.mean)/self.std
        if mask is None:
            return image
        return image, mask/255, body/255, detail/255


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None, body=None, detail=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        body  = cv2.resize( body, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        detail= cv2.resize( detail, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, body, detail


class ToTensor(object):
    def __call__(self, image, mask=None, body=None, detail=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        mask  = torch.from_numpy(mask)
        body  = torch.from_numpy(body)
        detail= torch.from_numpy(detail)
        return image, mask, body, detail


class _LDFConfig(object):
    """Minimal config shim for LDF inference.

    The upstream code used `train_fine/dataset.py::Config`, but that file also
    contains training datasets/augmentations which are not needed in the
    rendering pipeline. LDF only relies on:

    - `snapshot`: path to the trained model weights (torch state_dict)
    - `mean` / `std`: RGB normalization constants
    """

    def __init__(self, *, snapshot: str):
        self.snapshot = snapshot
        # Upstream normalization constants (BGR->RGB already handled by caller).
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])

