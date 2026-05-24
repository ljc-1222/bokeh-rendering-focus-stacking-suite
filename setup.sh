#!/usr/bin/env bash
# Deterministic environment setup script for BRnFS (Linux / WSL2).
#
# This script creates/updates `.venv`, installs the pinned Python 3.9 runtime,
# ensures model weights, and builds the optional fast `scatter_cuda` renderer.
#
# Usage:
#   bash setup.sh
#
# Useful overrides:
#   PYTHON_BIN=python3.9 bash setup.sh
#   BRNFS_SKIP_WEIGHTS=1 bash setup.sh
#   BRNFS_SKIP_SCATTER_CUDA=1 bash setup.sh
#   CUDA_HOME=/usr/local/cuda-11.7 bash setup.sh
#
# After it finishes:
#   source .venv/bin/activate
#   python -m brnfs gui
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

# You may override the interpreter used to create `.venv` explicitly:
#   PYTHON_BIN=python3.9 bash setup.sh
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
echo "Python for new venv: ${PYTHON_BIN}"

PY_MINOR="$("${PYTHON_BIN}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [ "${PY_MINOR}" != "3.9" ]; then
  echo "ERROR: BRnFS requires Python 3.9, but ${PYTHON_BIN} is Python ${PY_MINOR}." >&2
  echo "Use PYTHON_BIN=python3.9 bash setup.sh, or create a Python 3.9 environment first." >&2
  exit 1
fi

###############################################################################
# Environment strategy
#
# Always use a project-local venv at `${PROJECT_ROOT}/.venv` and activate it
# for the remainder of the script before installing packages.
###############################################################################

# If an old/broken/wrong-Python venv exists, recreate it.
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
      echo "Removing venv with unsupported Python ${VENV_PY_MINOR}: ${VENV_DIR}"
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
echo "Using venv python: $(python -c 'import sys; print(sys.executable)')"

echo "Upgrading pip tooling..."
# Pin pip to a conservative range for best compatibility with older wheels.
# Newer pip versions occasionally drop support for older manylinux wheel tags.
python -m pip install --upgrade "pip<24" "setuptools<81" wheel

echo "Installing pinned numpy first (build prerequisite for older packages)..."
# Force wheels-only: source builds are fragile and (on newer Pythons) will fail.
python -m pip install --no-cache-dir --only-binary=:all: "numpy==1.19.5"

echo "Installing PyTorch CUDA 11.7 wheels..."
python -m pip install --upgrade \
  "torch==2.0.0+cu117" \
  "torchvision==0.15.0+cu117" \
  "torchaudio==2.0.0+cu117" \
  --extra-index-url https://download.pytorch.org/whl/cu117

echo "Installing remaining Python dependencies from requirements.txt..."
# requirements.txt intentionally excludes torch/opencv/lightning packages that
# are installed explicitly above/below to avoid accidental numpy upgrades.
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

echo "Installing this project in editable mode (without dependency resolution)..."
python -m pip install -e "${PROJECT_ROOT}" --no-deps

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

ensure_existing_weight() {
  local name="$1"
  local dest="$2"
  local min_bytes="$3"

  local size
  size="$(stat -c%s "${dest}" 2>/dev/null || echo 0)"
  if [ "${size}" -lt "${min_bytes}" ]; then
    echo "ERROR: ${name} is missing or incomplete (${size} bytes): ${dest}" >&2
    echo "ERROR: this file should be present in the repository under models/." >&2
    return 1
  fi
  echo "✓ ${name} weight present: ${dest} (${size} bytes)"
}

if [ "${BRNFS_SKIP_WEIGHTS:-0}" != "1" ]; then
  echo "Ensuring model weights are available (set BRNFS_SKIP_WEIGHTS=1 to skip)..."

  ensure_existing_weight \
    "LDF snapshot" \
    "${PROJECT_ROOT}/models/ldf/model-40" \
    50000000

  ensure_weight \
    "LDF (salient detection backbone)" \
    "https://huggingface.co/ysheng/DrBokeh/resolve/main/resnet50-19c8e357.pth?download=true" \
    "${PROJECT_ROOT}/models/ldf/resnet50-19c8e357.pth" \
    50000000

  ensure_weight \
    "LaMa (RGB inpainting)" \
    "https://huggingface.co/ysheng/DrBokeh/resolve/main/best.ckpt?download=true" \
    "${PROJECT_ROOT}/models/lama/big-lama/models/best.ckpt" \
    150000000

  ensure_weight \
    "MiDaS/DPT (monocular depth)" \
    "https://huggingface.co/ysheng/DrBokeh/resolve/main/dpt_large-midas-2f21e586.pt?download=true" \
    "${PROJECT_ROOT}/models/dpt/dpt_large-midas-2f21e586.pt" \
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
  pushd "${PROJECT_ROOT}/brnfs/cuda_src" >/dev/null
  rm -rf build/ dist/ *.egg-info/ __pycache__/ || true
  find . -name "*.so" -delete || true
  python -m pip install --no-build-isolation --force-reinstall --no-cache-dir .
  popd >/dev/null
fi

echo "Quick sanity checks..."
if [ "${BRNFS_SKIP_SCATTER_CUDA:-0}" = "1" ]; then
python - << 'PY'
import numpy as np
import torch
import cv2
import pandas as pd
import matplotlib

print("✓ numpy:", np.__version__)
print("✓ torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("✓ cv2:", cv2.__version__)
print("✓ pandas:", pd.__version__)
print("✓ matplotlib:", matplotlib.__version__)
print("! scatter_cuda not built (BRNFS_SKIP_SCATTER_CUDA=1)")
PY
else
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
fi

echo "DONE."
echo "Next:"
echo "  source .venv/bin/activate"
echo "  python -m brnfs gui"
