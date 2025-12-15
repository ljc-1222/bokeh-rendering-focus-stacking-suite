# Bokeh Rendering + Focus Stacking Suite — Image Processing Suite

## Project description
This project is a **DSP lab image-processing suite** that merges two pipelines into a single runnable demo:
- **Bokeh rendering**: depth estimation + layered defocus + inpainting + CUDA-accelerated scattering.
- **Focus stacking**: Laplacian-pyramid-based multi-focus fusion to produce an all-in-focus image.

It includes a **unified GUI** (two tabs), a consistent dataset layout under `Imgs/`, and output/cache folders under `outputs/` for repeatable runs.

## Requirements
- **OS**: Linux (WSL2 works) or Windows 10/11
- **GPU**: NVIDIA GPU with CUDA support (**required for bokeh rendering**)
- **Tooling**: a working CUDA toolkit install + a C++ toolchain for building the CUDA extension
- **Python**: **3.9** (this project pins a legacy NumPy/Skimage stack)

## Repository layout

```
bokeh_rendering_and_focus_stacking_suite/
  setup.sh                           # venv-based environment provisioning script
  setup.ps1                          # Windows PowerShell environment provisioning script
  requirements.txt                   # python deps (legacy pins; torch installed by env scripts)
  gui/
    gui.py                           # unified GUI entry point (tabs)
    gui_bokeh_rendering.py           # bokeh rendering tab
    gui_focus_stacking.py            # focus stacking tab (manual save)
  app/
    bokeh_rendering/                 # bokeh pipeline (DPT + LDF + LaMa + DScatter)
    focus_stacking/                  # focus stacking pipeline (vendored + adapted)
      preprocess.py                  # load/resize/align (+ cache) the input stack
      pyramids.py                    # build Gaussian/Laplacian pyramids
      sharpness.py                   # compute sharpness maps from Laplacians
      mask.py                        # build/smooth decision masks
      fusion.py                      # fuse pyramids and reconstruct final image
      cli_main.py                    # optional focus stacking CLI entry point
    cuda-src/                        # CUDA/C++ extension sources for `scatter_cuda`
  Imgs/
    bokeh_rendering/                 # sample images for bokeh tab
    focus_stacking/                  # datasets for focus stacking tab (each subfolder is a dataset)
  outputs/
    bokeh_rendering/cache/           # bokeh preprocessing cache (.npz)
    bokeh_rendering/                 # bokeh outputs (manual save)
    focus_stacking/cache/            # focus stacking preprocessing cache (.npz)
    focus_stacking/                  # focus stacking outputs (manual save)
  scripts/
    build_scatter_cuda.sh
    build_scatter_cuda.ps1
```

## Installation

### Linux / WSL2

From `bokeh_rendering_and_focus_stacking_suite/`:

```bash
cd "bokeh_rendering_and_focus_stacking_suite"
bash setup.sh
source .venv/bin/activate
```

If you are currently in conda `(base)` (or otherwise don't have `python3.9` on PATH),
`setup.sh` will bootstrap a Python 3.9 conda env automatically. The default env name is
`BRnFS` (override with `BRNFS_CONDA_ENV_NAME=...`).

### Windows (PowerShell)

Prereqs (bokeh rendering):
- NVIDIA driver + **CUDA Toolkit 11.7** installed (so `CUDA_PATH` is set)
- **Visual Studio Build Tools** (C++ build tools) to compile the CUDA extension
- **Python 3.9**

From `bokeh_rendering_and_focus_stacking_suite\` in **PowerShell**:

```powershell
cd ".\bokeh_rendering_and_focus_stacking_suite"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```

If you prefer `cmd.exe` activation instead of PowerShell:

```bat
.\.venv\Scripts\activate.bat
```

## Building the CUDA Extension

The bokeh rendering pipeline uses a CUDA-accelerated scattering extension (`scatter_cuda`) for fast rendering. The setup scripts (`setup.sh` and `setup.ps1`) attempt to build this extension automatically, but if the build fails or you need to rebuild it manually, you can use the dedicated build scripts.

### Linux / WSL2

From `bokeh_rendering_and_focus_stacking_suite/` with the virtual environment activated:

```bash
cd "bokeh_rendering_and_focus_stacking_suite"
source .venv/bin/activate
bash scripts/build_scatter_cuda.sh
```

**Prerequisites:**
- CUDA toolkit installed and `nvcc` available on PATH
- CUDA version must match your PyTorch build (this project targets CUDA 11.7)
- C++ compiler (g++ < 12 for CUDA 11.7; the script will attempt to use g++-11 if available)

The script will:
- Verify PyTorch is installed and has CUDA support
- Check that `nvcc` version matches PyTorch's CUDA version
- Handle compiler version compatibility (CUDA 11.7 requires g++ < 12)
- Build and install the `scatter_cuda` extension

### Windows (PowerShell)

From `bokeh_rendering_and_focus_stacking_suite\` with the virtual environment activated:

```powershell
cd ".\bokeh_rendering_and_focus_stacking_suite"
.\.venv\Scripts\Activate.ps1
.\scripts\build_scatter_cuda.ps1
```

**Prerequisites:**
- CUDA Toolkit 11.7 installed (with `CUDA_PATH` environment variable set)
- Visual Studio Build Tools (C++ build tools) installed
- PyTorch with CUDA support installed in the active environment

**Note:** If the CUDA extension is not built, the bokeh rendering pipeline will automatically fall back to a slow CPU-based rendering implementation. The GUI will display a warning when rendering without the CUDA extension.

## Quickstart: Unified GUI (recommended)

```bash
cd "bokeh_rendering_and_focus_stacking_suite"
source .venv/bin/activate
python gui/gui.py
```

Windows (PowerShell):

```powershell
cd ".\bokeh_rendering_and_focus_stacking_suite"
.\.venv\Scripts\Activate.ps1
python .\gui\gui.py
```

### GUI workflow

#### Bokeh Rendering tab
- **Select Image**: loaded from `Imgs/bokeh_rendering/`
- **Preprocess**: runs DPT + LDF + LaMa and builds foreground/background layers  
  - results cached under `outputs/bokeh_rendering/cache/`
- **Render**: renders the current preview (no automatic file write)
- **Save rendered image**: manual save dialog (default: `outputs/bokeh_rendering/`)

#### Focus Stacking tab
- **Select Image Set**: choose a dataset folder under `Imgs/focus_stacking/`
- **Generate Fused Image**: computes fused result and updates preview (**no auto save**)
- **Save fused image**: manual save dialog (default: `outputs/focus_stacking/`)

## Image Organization and Pipeline Overview

This section explains how to organize your images and how each pipeline processes them.

### Bokeh Rendering: Single Image Processing

#### Image Placement

Place your input images directly in the `Imgs/bokeh_rendering/` folder:

```
Imgs/
  bokeh_rendering/
    IMG_1275.png          # Your input image
    photo1.jpg            # Another image
    photo2.jpeg           # Supported formats: .png, .jpg, .jpeg
```

**Requirements:**
- Supported formats: `.png`, `.jpg`, `.jpeg`
- Each image is processed independently
- No specific naming convention required (avoid filenames containing "alpha" as they are filtered out)
- Images can be of any resolution (will be processed at their original size)

#### Pipeline Stages

When you select an image and click **Preprocess**, the following steps occur:

1. **Depth Estimation (DPT)**
   - Uses a Dense Prediction Transformer (DPT) model to estimate depth/disparity from the RGB image
   - Produces a normalized disparity map (H×W×1) where values represent relative depth
   - This map determines which parts of the image will be blurred based on their distance from the focal plane

2. **Salient Segmentation (LDF)**
   - Uses a Layered Defocus (LDF) model to predict an alpha matte
   - Identifies the foreground subject from the background
   - Produces a mask (H×W×1) used to separate foreground and background layers

3. **Inpainting (LaMa)**
   - Uses Large Mask Inpainting (LaMa) to fill disocclusions in the background
   - When the foreground is separated, gaps appear in the background; LaMa fills these regions
   - Ensures the background layer is complete for proper bokeh rendering

4. **Layer Construction**
   - Combines RGB + alpha + disparity into two RGBAD layers (5 channels: R, G, B, A, D)
   - **Foreground layer (`fg_rgbad`)**: The subject with its alpha and depth
   - **Background layer (`bg_rgbad`)**: The background with inpainted regions and depth
   - These layers are cached in `outputs/bokeh_rendering/cache/` for fast re-rendering

5. **Rendering (DScatter)**
   - When you click **Render**, the system uses the preprocessed layers
   - Applies depth-of-field blur based on:
     - **Focal plane** (0.0-1.0): Which depth plane stays sharp
     - **Intensity K** (0.0-60.0): How strong the blur effect is
     - **Lens kernel size** (7-151, odd): The size of the blur kernel (larger = more bokeh)
   - Uses CUDA-accelerated scattering for fast rendering (falls back to slow CPU if CUDA extension is missing)
   - The rendered result is displayed in the preview panel

**Caching:**
- Preprocessing results are cached in `outputs/bokeh_rendering/cache/` as `.npz` files
- Cache keys include image filename, modification time, size, and preprocessing parameters
- Changing preprocessing settings (e.g., mask filter) invalidates the cache and triggers recomputation

### Focus Stacking: Multi-Image Fusion

#### Image Placement

Organize your focus stack images into **dataset folders** under `Imgs/focus_stacking/`:

```
Imgs/
  focus_stacking/
    dataset1/              # First dataset (e.g., "macro_flower")
      IMG_001.png          # Image focused on foreground
      IMG_002.png          # Image focused on mid-ground
      IMG_003.png          # Image focused on background
      IMG_004.png          # More images at different focus distances
    dataset2/              # Second dataset (e.g., "landscape")
      photo1.png
      photo2.png
      photo3.png
```

**Requirements:**
- Each subfolder under `Imgs/focus_stacking/` is treated as a separate dataset
- All images in a dataset folder are loaded together as a stack
- Supported formats: `.png` (default), `.jpg`, `.jpeg` (configurable)
- Images should be captured at different focus distances of the same scene
- Images can have different resolutions (will be resized to match the first image)
- No specific naming convention required, but images are processed in alphabetical order

#### Pipeline Stages

When you select a dataset and click **Generate Fused Image**, the following steps occur:

1. **Image Loading and Preprocessing**
   - Loads all images from the selected dataset folder
   - Resizes all images to match the first image's dimensions (ensures consistent stack size)
   - Aligns images using ECC (Enhanced Correlation Coefficient) maximization with affine warping
     - Compensates for slight camera movement between shots
     - Uses the first image as the reference frame
   - Preprocessed stack is cached in `outputs/focus_stacking/cache/` as `.npy` files for faster subsequent runs

2. **Pyramid Construction**
   - Builds Gaussian and Laplacian pyramids for each image in the stack
   - **Gaussian pyramid**: Multi-scale representation (blurred versions at different resolutions)
   - **Laplacian pyramid**: Difference between consecutive Gaussian levels (captures detail at each scale)
   - Number of pyramid levels is user-configurable (default: 5, range: 2-20)
   - More levels capture finer detail but increase computation time

3. **Sharpness Map Computation**
   - Computes sharpness maps from the Laplacian pyramids
   - Measures local variance/energy in the Laplacian coefficients
   - Higher values indicate sharper regions in each image
   - Produces one sharpness map per image in the stack

4. **Mask Generation**
   - Builds decision masks based on sharpness maps
   - **Soft masks** (default): Normalized masks with Gaussian smoothing for smooth transitions
   - **Hard masks**: Binary masks (sharpest region wins) for more aggressive fusion
   - Each mask indicates which image contributes most to each pixel at each pyramid level

5. **Pyramid Fusion**
   - Fuses Laplacian pyramids using the generated masks
   - At each pyramid level, combines coefficients from different images based on their sharpness
   - Top-level (coarsest) Gaussian can be fused using:
     - **Max**: Takes the maximum value (preserves highlights)
     - **Mean**: Takes the average (smoother transitions)

6. **Image Reconstruction**
   - Reconstructs the final all-in-focus image from the fused pyramid
   - Combines all pyramid levels back into a single high-resolution image
   - The result contains the sharpest regions from each input image

**Caching:**
- Aligned image stacks are cached in `outputs/focus_stacking/cache/` as `{dataset_name}_aligned.npy`
- Cache is based on dataset folder name
- Adding/removing images or changing image files invalidates the cache (detected by file modification times)

### Example Workflows

#### Example 1: Bokeh Rendering Workflow

1. **Prepare your image:**
   ```bash
   # Copy your photo to the bokeh rendering folder
   cp ~/Pictures/portrait.jpg Imgs/bokeh_rendering/portrait.jpg
   ```

2. **Run the GUI:**
   ```bash
   python gui/gui.py
   ```

3. **In the Bokeh Rendering tab:**
   - Select `portrait.jpg` from the dropdown
   - Click **Preprocess** (this may take 30-60 seconds for the first run)
     - Depth map is estimated
     - Foreground/background are separated
     - Background gaps are inpainted
     - Results are cached for future renders
   - Adjust **Focal plane** slider (0.0 = near, 1.0 = far) to choose what stays sharp
   - Adjust **Intensity (K)** slider to control blur strength
   - Adjust **Lens (kernel)** slider to control bokeh size
   - Click **Render** to see the result (fast if CUDA is available, slow on CPU)
   - Click **Save rendered image** to export the final result

#### Example 2: Focus Stacking Workflow

1. **Prepare your focus stack:**
   ```bash
   # Create a dataset folder
   mkdir -p Imgs/focus_stacking/macro_flower
   
   # Copy your focus stack images (captured at different focus distances)
   cp ~/Photos/focus_stack/*.png Imgs/focus_stacking/macro_flower/
   ```

2. **Run the GUI:**
   ```bash
   python gui/gui.py
   ```

3. **In the Focus Stacking tab:**
   - Select `macro_flower` from the dataset dropdown
   - The left panel shows an animated preview of all input images
   - Adjust **Pyramid Levels** (default: 5) if needed
   - Choose **Mask Type**: "Normalized Soft" (smooth) or "Hard" (sharp transitions)
   - Choose **Top Layer Fusion**: "Max" (preserves highlights) or "Mean" (smoother)
   - Click **Generate Fused Image** (this may take 1-5 minutes depending on image count and size)
     - Images are aligned
     - Pyramids are built
     - Sharpness maps are computed
     - Masks are generated
     - Images are fused
   - The right panel shows the all-in-focus result
   - Click **Save fused image** to export the final result

## CLI usage (RGB → bokeh)

The CLI is implemented in `app/bokeh_rendering/Inference.py`.

```bash
cd "bokeh_rendering_and_focus_stacking_suite"
source .venv/bin/activate
python app/bokeh_rendering/Inference.py \
  --rgb Imgs/bokeh_rendering/IMG_1275.png \
  -K 30.0 \
  --focal 0.10 \
  --lens 71 \
  --gamma 2.2 \
  --ofile outputs/IMG_1275-focal-0.10.png \
  --verbose
```

## Troubleshooting
- **`ImportError: scatter_cuda ...` / `No module named 'scatter_cuda'`**
  - The GUI now **falls back to CPU scatter automatically** if `scatter_cuda` is missing.
    - Preprocess will work.
    - Rendering will be **very slow** without the CUDA extension.
  - To enable fast GPU rendering, build/install `scatter_cuda`:
    - Linux/WSL2: `bash scripts/build_scatter_cuda.sh`
    - Windows: `.\scripts\build_scatter_cuda.ps1` (after activating the venv)
  - If the build script reports a CUDA version mismatch, install a CUDA toolkit that matches `torch.version.cuda`
    (or install a PyTorch build that matches your installed CUDA toolkit).
- **CUDA not available**
  - Bokeh rendering can run on CPU, but it will be extremely slow; focus stacking does not require CUDA.

## Third-party code and licenses
This repo vendors parts of upstream projects for inference:
- DPT: `app/bokeh_rendering/Depth/DPT/` (see `app/bokeh_rendering/Depth/DPT/LICENSE`)
- LaMa: `app/bokeh_rendering/Inpainting/lama/` (see `app/bokeh_rendering/Inpainting/lama/LICENSE`)

For redistribution, see `THIRD_PARTY_NOTICES.md` and the copied license texts under `LICENSES/`.
