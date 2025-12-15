#!/usr/bin/env bash
# Deterministic environment setup script for the `bokeh_rendering_and_focus_stacking_suite/` project (WSL/Linux).
#
# This version uses a **single Python venv** and:
# - installs the pinned legacy scientific stack (numpy==1.19.5, scikit-image==0.17.2, ...)
# - installs PyTorch CUDA wheels (cu117)
# - builds and installs the custom CUDA extension (`scatter_cuda`)
#
# If you are currently in a conda session (e.g. `(base)`), the script can optionally
# bootstrap a Python 3.9 conda env automatically and re-run itself via `conda run`.
#
# Usage:
#   bash setup.sh              # creates/updates .venv and installs deps
#
# After it finishes:
#   source .venv/bin/activate
#   python -m gui.gui
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

# You may override the interpreter explicitly:
#   PYTHON_BIN=python3.9 bash setup.sh
#
# This project pins a legacy scientific stack (numpy==1.19.5, scikit-image==0.17.2, ...),
# so it must run on Python 3.9 (and, in practice, cannot run on Python 3.12+).
if [ -n "${PYTHON_BIN:-}" ]; then
  : # user-specified
elif command -v python3.9 >/dev/null 2>&1; then
  PYTHON_BIN="python3.9"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: python not found on PATH." >&2
  exit 1
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Using python: ${PYTHON_BIN}"

PY_MINOR="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [ "${PY_MINOR}" != "3.9" ]; then
  # If we're already in a conda shell, we can bootstrap Python 3.9 automatically.
  # We intentionally use `conda run` (instead of `conda activate`) so this script
  # remains non-interactive and doesn't rely on shell state.
  if [ "${BRNFS_CONDA_BOOTSTRAP:-1}" = "1" ] \
    && [ -n "${CONDA_PREFIX:-}" ] \
    && command -v conda >/dev/null 2>&1 \
    && [ -z "${BRNFS_RERUN_IN_CONDA:-}" ]; then
    CONDA_ENV_NAME="${BRNFS_CONDA_ENV_NAME:-BRnFS}"

    echo "Detected conda shell but Python is ${PY_MINOR}; bootstrapping conda env '${CONDA_ENV_NAME}' with Python 3.9..." >&2

    if ! conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
      conda create -n "${CONDA_ENV_NAME}" python=3.9 -y
    fi

    echo "Re-running setup inside conda env '${CONDA_ENV_NAME}' (via conda run)..." >&2
    export BRNFS_RERUN_IN_CONDA=1
    export BRNFS_CONDA_ENV_NAME="${CONDA_ENV_NAME}"
    exec conda run -n "${CONDA_ENV_NAME}" bash "$0"
  fi

  echo "ERROR: Unsupported Python ${PY_MINOR} (from: ${PYTHON_BIN})." >&2
  echo "This project requires Python 3.9 due to pinned legacy deps (e.g. numpy==1.19.5)." >&2
  echo "" >&2
  echo "Fix options:" >&2
  echo "  - If you're using conda:" >&2
  echo "      conda create -n BRnFS python=3.9 -y && conda activate BRnFS" >&2
  echo "      bash setup.sh" >&2
  echo "  - Or install a system python3.9 and run:" >&2
  echo "      PYTHON_BIN=python3.9 bash setup.sh" >&2
  exit 1
fi

###############################################################################
# Environment strategy
#
# Default: always use a project-local venv at `${PROJECT_ROOT}/.venv` and
# activate it for the remainder of the script.
#
# If you really want to install into an active conda env instead, you may set:
#   BRNFS_USE_VENV=0 bash setup.sh
###############################################################################

BRNFS_USE_VENV="${BRNFS_USE_VENV:-1}"

if [ "${BRNFS_USE_VENV}" = "1" ]; then
  # If an old/broken venv exists (wrong python, or moved directory), recreate it.
  if [ -d "${VENV_DIR}" ]; then
    if [ ! -x "${VENV_DIR}/bin/python" ] || [ ! -f "${VENV_DIR}/bin/activate" ]; then
      echo "Removing broken venv: ${VENV_DIR}"
      rm -rf "${VENV_DIR}"
    else
      VENV_PY_MINOR="$("${VENV_DIR}/bin/python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
      if [ "${VENV_PY_MINOR}" != "3.9" ]; then
        echo "Removing venv with wrong Python (${VENV_PY_MINOR}): ${VENV_DIR}"
        rm -rf "${VENV_DIR}"
      elif ! grep -Fq "export VIRTUAL_ENV=${VENV_DIR}" "${VENV_DIR}/bin/activate"; then
        echo "Removing moved/corrupted venv (activate path mismatch): ${VENV_DIR}"
        rm -rf "${VENV_DIR}"
      fi
    fi
  fi

  if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating venv: ${VENV_DIR}"
    "${PYTHON_BIN}" -m venv --prompt "BRnFS" "${VENV_DIR}"
  fi

  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
else
  echo "Using active environment (no project venv)."
fi

echo "Upgrading pip tooling..."
# Pin pip to a conservative range for best compatibility with legacy wheels.
# (Newer pip versions occasionally drop support for older manylinux wheel tags.)
python -m pip install --upgrade "pip<24" setuptools wheel

echo "Installing pinned numpy first (build prerequisite for legacy packages)..."
# Force wheels-only: source builds are fragile and (on newer Pythons) will fail.
python -m pip install --no-cache-dir --only-binary=:all: "numpy==1.19.5"

echo "Installing PyTorch CUDA 11.7 wheels..."
python -m pip install --upgrade \
  "torch==2.0.0+cu117" \
  "torchvision==0.15.0+cu117" \
  "torchaudio==2.0.0+cu117" \
  --extra-index-url https://download.pytorch.org/whl/cu117

echo "Installing remaining Python dependencies from requirements.txt..."
# Note: keeping this as a single command preserves the pinned legacy stack in requirements.txt.
python -m pip install --no-cache-dir -r "${PROJECT_ROOT}/requirements.txt"

echo "Installing OpenCV (pinned) WITHOUT upgrading numpy..."
python -m pip install --no-deps --no-cache-dir "opencv-python-headless==4.5.5.64"

echo "Installing minimal PyTorch Lightning bits (for LaMa checkpoint loading) WITHOUT upgrading numpy..."
python -m pip install --no-deps --no-cache-dir \
  "pytorch-lightning==1.9.5" \
  "torchmetrics==0.11.4" \
  "lightning-utilities==0.15.2"
python -m pip install --no-deps --no-cache-dir "lightning-fabric==1.9.5"

echo "Uninstalling TensorFlow if it was pulled in previously (optional dependency)..."
python -m pip uninstall -y tensorflow tensorflow-io-gcs-filesystem >/dev/null 2>&1 || true

###############################################################################
# Model weights (LDF + LaMa + MiDaS/DPT)
#
# Upstream DrBokeh expects users to download weights manually. In this merged
# project, we make `setup.sh` ensure the files exist.
#
# To skip (e.g., offline install), run:
#   BRNFS_SKIP_WEIGHTS=1 bash setup.sh
###############################################################################

download_file() {
  local url="$1"
  local dest="$2"

  mkdir -p "$(dirname "${dest}")"

  if command -v wget >/dev/null 2>&1; then
    # -c: resume if partially downloaded
    wget -c -O "${dest}" --tries=3 --timeout=30 "${url}"
    return 0
  fi

  if command -v curl >/dev/null 2>&1; then
    # -C - : resume if partially downloaded
    curl -L --fail --retry 3 --retry-delay 2 -C - -o "${dest}" "${url}"
    return 0
  fi

  echo "ERROR: neither wget nor curl is available to download model weights." >&2
  return 1
}

ensure_weight() {
  local name="$1"
  local url="$2"
  local dest="$3"
  local min_bytes="$4"

  if [ -f "${dest}" ]; then
    local size
    size="$(stat -c%s "${dest}" 2>/dev/null || echo 0)"
    if [ "${size}" -ge "${min_bytes}" ]; then
      echo "✓ ${name} weight present: ${dest} (${size} bytes)"
      return 0
    fi
    echo "WARNING: ${name} weight exists but looks too small (${size} bytes). Re-downloading..."
    rm -f "${dest}"
  fi

  echo "Downloading ${name} weight..."
  echo "  -> ${dest}"
  download_file "${url}" "${dest}"

  local final_size
  final_size="$(stat -c%s "${dest}" 2>/dev/null || echo 0)"
  if [ "${final_size}" -lt "${min_bytes}" ]; then
    echo "ERROR: downloaded ${name} weight looks incomplete (${final_size} bytes): ${dest}" >&2
    echo "ERROR: check network access and the URL: ${url}" >&2
    return 1
  fi
  echo "✓ Downloaded ${name} weight (${final_size} bytes)"
}

if [ "${BRNFS_SKIP_WEIGHTS:-0}" != "1" ]; then
  echo "Ensuring model weights are available (set BRNFS_SKIP_WEIGHTS=1 to skip)..."

  ensure_weight \
    "LDF (salient detection backbone)" \
    "https://huggingface.co/ysheng/DrBokeh/resolve/main/resnet50-19c8e357.pth?download=true" \
    "${PROJECT_ROOT}/app/bokeh_rendering/Salient/LDF/res/resnet50-19c8e357.pth" \
    50000000

  ensure_weight \
    "LaMa (RGB inpainting)" \
    "https://huggingface.co/ysheng/DrBokeh/resolve/main/best.ckpt?download=true" \
    "${PROJECT_ROOT}/app/bokeh_rendering/Inpainting/lama/big-lama/models/best.ckpt" \
    150000000

  ensure_weight \
    "MiDaS/DPT (monocular depth)" \
    "https://huggingface.co/ysheng/DrBokeh/resolve/main/dpt_large-midas-2f21e586.pt?download=true" \
    "${PROJECT_ROOT}/app/bokeh_rendering/Depth/DPT/weights/dpt_large-midas-2f21e586.pt" \
    300000000
else
  echo "Skipping model weight downloads (BRNFS_SKIP_WEIGHTS=1)."
fi

echo "Configuring runtime library paths for CUDA extensions..."
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.7}"
if [ ! -d "${CUDA_HOME}" ] && [ -d "/usr/local/cuda" ]; then
  CUDA_HOME="/usr/local/cuda"
fi
export CUDA_HOME="${CUDA_HOME}"

TORCH_LIB_DIR="$(python - <<'PY'
import pathlib
import torch
print(pathlib.Path(torch.__file__).resolve().parent / "lib")
PY
)"
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

if [ "${BRNFS_SKIP_SCATTER_CUDA:-0}" = "1" ]; then
  echo "Skipping scatter_cuda build (BRNFS_SKIP_SCATTER_CUDA=1)."
  echo "NOTE: bokeh rendering requires scatter_cuda; focus stacking does not."
else
  echo "Configuring host compiler for CUDA 11.7 (needs GCC/G++ < 12)..."

  if command -v g++ >/dev/null 2>&1; then
    GXX_MAJOR="$(g++ -dumpversion | cut -d. -f1 || echo 0)"
  else
    GXX_MAJOR="0"
  fi

  if [ "${GXX_MAJOR}" -ge 12 ]; then
    # Prefer g++-11, otherwise try 10/9.
    for ver in 11 10 9; do
      if command -v "g++-${ver}" >/dev/null 2>&1 && command -v "gcc-${ver}" >/dev/null 2>&1; then
        export CC="gcc-${ver}"
        export CXX="g++-${ver}"
        export CUDAHOSTCXX="${CXX}"
        echo "Using ${CXX} (via CC=${CC}) for CUDA builds."
        break
      fi
    done

    if [ -z "${CXX:-}" ]; then
      # Prefer installing a compatible toolchain via conda (works on WSL without sudo).
      if [ -n "${CONDA_PREFIX:-}" ] && command -v conda >/dev/null 2>&1; then
        echo "Attempting to install a GCC 11 toolchain into the active conda env (no sudo)..."
        conda install -y "gcc_linux-64=11" "gxx_linux-64=11" >/dev/null 2>&1 \
          || conda install -y -c conda-forge "gcc_linux-64=11" "gxx_linux-64=11"

        if command -v x86_64-conda-linux-gnu-gcc >/dev/null 2>&1 && command -v x86_64-conda-linux-gnu-g++ >/dev/null 2>&1; then
          export CC="x86_64-conda-linux-gnu-gcc"
          export CXX="x86_64-conda-linux-gnu-g++"
          export CUDAHOSTCXX="${CXX}"
          echo "Using ${CXX} (conda toolchain) for CUDA builds."
        fi
      fi

      if [ -z "${CXX:-}" ] && command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
        echo "Installing gcc-11/g++-11 via apt (passwordless sudo detected)..."
        sudo -n apt-get update
        sudo -n apt-get install -y gcc-11 g++-11
        export CC="gcc-11"
        export CXX="g++-11"
        export CUDAHOSTCXX="${CXX}"
        echo "Using ${CXX} (via CC=${CC}) for CUDA builds."
      fi

      if [ -z "${CXX:-}" ]; then
        echo "ERROR: CUDA 11.7 + PyTorch requires GCC/G++ < 12, but your default g++ is ${GXX_MAJOR}." >&2
        echo "Install a compatible compiler and re-run, e.g. on Ubuntu/WSL:" >&2
        echo "  sudo apt-get update && sudo apt-get install -y gcc-11 g++-11" >&2
        echo "" >&2
        echo "Or (to proceed without CUDA scattering, focus stacking only):" >&2
        echo "  BRNFS_SKIP_SCATTER_CUDA=1 bash setup.sh" >&2
        exit 1
      fi
    fi
  fi

  echo "Building the CUDA extension (scatter_cuda) against the CURRENT PyTorch..."
  pushd "${PROJECT_ROOT}/app/cuda-src" >/dev/null
  rm -rf build/ dist/ *.egg-info/ __pycache__/ || true
  find . -name "*.so" -delete || true
  python -m pip install --no-build-isolation --force-reinstall --no-cache-dir .
  popd >/dev/null
fi

echo "Quick sanity checks..."
python - << 'PY'
import numpy as np
import torch
import cv2
import pandas as pd
import matplotlib
import scatter_cuda

print("✓ numpy:", np.__version__)
print("✓ torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("✓ cv2:", cv2.__version__)
print("✓ pandas:", pd.__version__)
print("✓ matplotlib:", matplotlib.__version__)
print("✓ scatter_cuda import OK")
PY

echo "DONE."
echo "Next:"
if [ "${BRNFS_USE_VENV}" = "1" ]; then
  echo "  source .venv/bin/activate"
else
  echo "  conda activate ${BRNFS_CONDA_ENV_NAME:-BRnFS}"
fi
echo "  python -m gui.gui"
