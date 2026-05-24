"""Centralized repository paths for BRnFS.

Entry points should use this module for canonical data/model/output locations.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

EXAMPLES_DIR = REPO_ROOT / "examples"
EXAMPLE_BOKEH_DIR = EXAMPLES_DIR / "bokeh"
EXAMPLE_FOCUS_DIR = EXAMPLES_DIR / "focus"

OUTPUTS_DIR = REPO_ROOT / "outputs"
BOKEH_OUTPUT_DIR = OUTPUTS_DIR / "bokeh"
FOCUS_OUTPUT_DIR = OUTPUTS_DIR / "focus"
CACHE_DIR = OUTPUTS_DIR / "cache"
BOKEH_CACHE_DIR = CACHE_DIR / "bokeh"
FOCUS_CACHE_DIR = CACHE_DIR / "focus"

MODELS_DIR = REPO_ROOT / "models"
DPT_MODELS_DIR = MODELS_DIR / "dpt"
LAMA_MODELS_DIR = MODELS_DIR / "lama"
LDF_MODELS_DIR = MODELS_DIR / "ldf"

VENDOR_DIR = REPO_ROOT / "vendor"

BOKEH_APP_DIR = REPO_ROOT / "app" / "bokeh_rendering"
DPT_VENDOR_DIR = BOKEH_APP_DIR / "Depth" / "DPT"
LAMA_VENDOR_DIR = BOKEH_APP_DIR / "Inpainting" / "lama"
LDF_VENDOR_DIR = BOKEH_APP_DIR / "Salient" / "LDF"

CUDA_SRC_DIR = REPO_ROOT / "brnfs" / "cuda_src"


def bokeh_input_dir() -> Path:
    """Return the canonical bokeh example directory."""
    return EXAMPLE_BOKEH_DIR


def focus_input_dir() -> Path:
    """Return the canonical focus example root."""
    return EXAMPLE_FOCUS_DIR


def ensure_runtime_dirs() -> None:
    """Create runtime output/cache directories."""
    for path in (BOKEH_OUTPUT_DIR, FOCUS_OUTPUT_DIR, BOKEH_CACHE_DIR, FOCUS_CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def dpt_weight_path() -> Path:
    return DPT_MODELS_DIR / "dpt_large-midas-2f21e586.pt"


def ldf_resnet_path() -> Path:
    return LDF_MODELS_DIR / "resnet50-19c8e357.pth"


def ldf_snapshot_path() -> Path:
    return LDF_MODELS_DIR / "model-40"


def lama_checkpoint_path() -> Path:
    return LAMA_MODELS_DIR / "big-lama" / "models" / "best.ckpt"


def lama_train_config_path() -> Path:
    return LAMA_VENDOR_DIR / "big-lama" / "config.yaml"


def lama_predict_config_path() -> Path:
    return LAMA_VENDOR_DIR / "configs" / "prediction" / "default.yaml"
