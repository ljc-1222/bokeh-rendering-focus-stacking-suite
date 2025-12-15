#!/bin/bash
# Set library paths for PyTorch and CUDA extensions (avoid hardcoding Python version).
TORCH_LIB="$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")"
export LD_LIBRARY_PATH="${TORCH_LIB}:${CUDA_HOME:-/usr/local/cuda}/lib64:${CUDA_HOME:-/usr/local/cuda}/lib:${LD_LIBRARY_PATH}"

set -euo pipefail

# NOTE:
# - This script is kept for parity with an earlier demo layout that expected `Imgs/00007.png`, etc.
# - The current `bokeh_rendering_and_focus_stacking_suite/Imgs/` directory primarily contains datasets under subfolders such as
#   `Imgs/bokeh_rendering/` and `Imgs/focus_stacking/`.

# Demo 1: full automatic pipeline (depth via DPT, salient via LDF, inpainting via LaMa)
id=00007
file="Imgs/${id}.png"

# basename w. extension and w.o. extension
fbasename=$(basename "$file")
fbasename_wo_ext="${fbasename%.*}"

focal=0.1
python app/bokeh_rendering/Inference.py --rgb "$file" -K 30.0 --focal "$focal" --ofile "outputs/bokeh_rendering/${fbasename_wo_ext}-focal-${focal}.png" --verbose --lens 71 --gamma 2.2

focal=0.8
python app/bokeh_rendering/Inference.py --rgb "$file" -K 30.0 --focal "$focal" --ofile "outputs/bokeh_rendering/${fbasename_wo_ext}-focal-${focal}.png" --verbose --lens 71 --gamma 2.2

# Demo 2: provide an alpha matte (optional)
id=00044
file="Imgs/${id}.png"
alpha="Imgs/${id}-alpha2.png"
fbasename=$(basename "$file")
fbasename_wo_ext="${fbasename%.*}"
focal=0.2
python app/bokeh_rendering/Inference.py --rgb "$file" --alpha "$alpha" -K 30.0 --focal "$focal" --ofile "outputs/bokeh_rendering/${fbasename_wo_ext}-alpha-focal-${focal}.png" --verbose --lens 71 --gamma 2.2