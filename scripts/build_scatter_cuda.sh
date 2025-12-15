#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Build and install the CUDA/C++ extension that provides the `scatter_cuda` Python module.
# This reuses the currently-active environment (e.g., `conda activate drbokeh`).

pushd app/cuda-src >/dev/null
python - <<'PY'
import re
import subprocess
import sys

def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()

try:
    import torch
except Exception as exc:  # noqa: BLE001
    print("ERROR: PyTorch must be installed in the active environment before building scatter_cuda.")
    print(f"Import error: {exc}")
    sys.exit(1)

torch_cuda = getattr(torch.version, "cuda", None)
if not torch_cuda:
    print("ERROR: Your installed PyTorch does not have CUDA support (torch.version.cuda is empty).")
    print("Install a CUDA-enabled PyTorch build, or use CPU fallback (the app will run but rendering is very slow).")
    sys.exit(1)

nvcc_ver = None
try:
    out = run(["nvcc", "--version"])
    m = re.search(r"release\s+(\d+\.\d+)", out)
    if m:
        nvcc_ver = m.group(1)
except Exception:
    nvcc_ver = None

if not nvcc_ver:
    print("ERROR: `nvcc` not found (CUDA toolkit is not installed or not on PATH).")
    print(f"Your PyTorch expects CUDA {torch_cuda}. Install the matching CUDA toolkit and ensure `nvcc` is available.")
    sys.exit(1)

if nvcc_ver != torch_cuda:
    print("ERROR: CUDA version mismatch between nvcc toolkit and PyTorch.")
    print(f"  - nvcc detected CUDA: {nvcc_ver}")
    print(f"  - torch was built with CUDA: {torch_cuda}")
    print("")
    print("Fix options:")
    print("  - This project targets CUDA 11.7. Recommended: install a cu117 PyTorch build (see `setup.sh`), or")
    print("  - Install a CUDA toolkit matching your installed PyTorch build.")
    sys.exit(1)

print(f"OK: CUDA toolchain matches torch (CUDA {torch_cuda}). Proceeding with build...")
PY

# CUDA 11.7 requires g++ < 12. If the system default is too new, try to use g++-11.
if [ -x "$(command -v g++)" ]; then
  GXX_MAJOR="$(g++ -dumpversion | cut -d. -f1 || echo "")"
else
  GXX_MAJOR=""
fi

if [ "${GXX_MAJOR:-}" != "" ] && [ "${GXX_MAJOR}" -ge 12 ]; then
  if command -v g++-11 >/dev/null 2>&1 && command -v gcc-11 >/dev/null 2>&1; then
    echo "Detected g++ ${GXX_MAJOR} (too new for CUDA 11.7). Using gcc-11/g++-11 for this build."
    export CC="$(command -v gcc-11)"
    export CXX="$(command -v g++-11)"
  else
    echo "ERROR: Detected g++ ${GXX_MAJOR}, but CUDA 11.7 requires g++ < 12." >&2
    echo "Install a compatible compiler and retry:" >&2
    echo "  sudo apt-get update && sudo apt-get install -y gcc-11 g++-11" >&2
    exit 1
  fi
fi

# Prefer pip build/install over deprecated `setup.py install`.
# NOTE: `--no-build-isolation` ensures we build against the active env's torch.
python -m pip install -v --no-build-isolation .
popd >/dev/null

echo "Done. You should now be able to import scatter_cuda (used by app/bokeh_rendering/DScatter/GPU_scatter.py)."


