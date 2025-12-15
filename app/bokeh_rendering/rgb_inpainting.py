"""RGB inpainting wrapper around the vendored LaMa model.

This module loads the "big-lama" generator checkpoint and exposes a small
`RGB_Inpainting_Inference` class used by the rendering pipeline.

Note:
    The implementation intentionally follows the upstream LaMa project layout
    by modifying `sys.path` to make `saicinpainting` importable.

Third-party notice:
    This project vendors code from "LaMa" (`advimman/lama`) which is licensed under
    Apache License 2.0.
    See:
    - `bokeh_rendering_and_focus_stacking_suite/app/bokeh_rendering/Inpainting/lama/LICENSE`
    - `bokeh_rendering_and_focus_stacking_suite/THIRD_PARTY_NOTICES.md`
"""

import os
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf

# Make `saicinpainting` importable as a top-level package (matches upstream LaMa layout).
sys.path.insert(0, ".")
sys.path.insert(0, "./app/bokeh_rendering/Inpainting/lama")

from saicinpainting.training.modules import make_generator  # type: ignore

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


class RGB_Inpainting_Inference:
    def __init__(self):
        """Initialize a minimal LaMa generator-only inference stack.

        This intentionally avoids importing LaMa training/evaluation code. We only:
        - build the generator from `big-lama/config.yaml`
        - load `big-lama/models/best.ckpt` (Lightning checkpoint) and extract `generator.*` weights
        """
        predict_cfg_path = "app/bokeh_rendering/Inpainting/lama/configs/prediction/default.yaml"
        with open(predict_cfg_path, "r") as f:
            predict_cfg = yaml.safe_load(f)

        model_dir = "app/bokeh_rendering/Inpainting/lama/big-lama"
        train_cfg_path = os.path.join(model_dir, "config.yaml")
        # The training config uses OmegaConf interpolations (e.g. `${generator.*}` and `${env:VAR}`).
        # Some OmegaConf versions ship a built-in `env` resolver that *errors* when a variable is
        # missing (e.g. TORCH_HOME). For inference, we prefer a permissive resolver that returns
        # the provided default (or None) instead of crashing.
        OmegaConf.register_new_resolver(
            "env",
            lambda k, default=None: os.environ.get(k, default),
            replace=True,
        )
        train_cfg = OmegaConf.to_container(OmegaConf.load(train_cfg_path), resolve=True)

        # Minimal fields needed for inference
        self.out_key: str = str(predict_cfg.get("out_key", "inpainted"))
        self.pad_out_to_modulo: int = int(predict_cfg.get("dataset", {}).get("pad_out_to_modulo", 8))
        self.concat_mask: bool = bool(train_cfg.get("training_model", {}).get("concat_mask", True))

        gen_cfg = dict(train_cfg["generator"])
        gen_kind = gen_cfg.pop("kind")
        self.generator = make_generator(None, kind=gen_kind, **gen_cfg)

        # Lightning checkpoint; in newer PyTorch versions `weights_only=True` is default and will fail.
        ckpt_path = os.path.join(model_dir, "models", str(predict_cfg.get("model", {}).get("checkpoint", "best.ckpt")))
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            # Older PyTorch versions do not support the `weights_only` argument.
            ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        gen_state = {k[len("generator.") :]: v for k, v in state.items() if k.startswith("generator.")}
        self.generator.load_state_dict(gen_state, strict=True)

        # Upstream code was CUDA-only; we allow CPU fallback for environments without CUDA.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator.eval()
        self.generator.to(self.device)



    def inference(self, rgb: np.array, mask: np.array):
        """Inpaint masked regions of an RGB image."""
        assert rgb.shape[-1] == 3 and mask.shape[-1] == 1, \
            'RGB and mask channels should be 3({}) and 1({}).'.format(rgb.shape[-1], mask.shape[-1])

        device = self.device

        image = torch.tensor(rgb.transpose((2, 0, 1)), dtype=torch.float32)[None, ...]
        m = torch.tensor(mask.transpose((2, 0, 1)), dtype=torch.float32)[None, ...]
        m = (m > 0).float()

        ori_h, ori_w = rgb.shape[:2]
        modulo = int(self.pad_out_to_modulo)
        if ori_h % modulo != 0 or ori_w % modulo != 0:
            pad_h = (modulo - ori_h % modulo) % modulo
            pad_w = (modulo - ori_w % modulo) % modulo
            image = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
            m = F.pad(m, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            image = image.to(device)
            m = m.to(device)

            masked = image * (1.0 - m)
            x = torch.cat([masked, m], dim=1) if self.concat_mask else masked
            predicted = self.generator(x)
            inpainted = m * predicted + (1.0 - m) * image

            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = cur_res[:ori_h, :ori_w]

        return cur_res

