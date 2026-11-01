# Setup and command-line reference

Run the commands below from the repository root. Use the CPU environment for
the focus-stacking CLI, or the full environment for bokeh rendering and the
two-tab GUI. Both paths target Python 3.9, as required by
[`pyproject.toml`](../pyproject.toml).

## Focus stacking on CPU

The focus CLI uses NumPy and OpenCV; it does not require PyTorch, model
weights, a CUDA toolkit, or a GPU. The minimal dependency set follows the
[focus runner](../brnfs/focus/runner.py) and its imports, with versions matching
the full setup.

Use a separate `.venv-focus` environment for this workflow.

**Linux / WSL2**

```bash
python3.9 -m venv .venv-focus
source .venv-focus/bin/activate
python -m pip install --upgrade "pip<24" "setuptools<81" wheel
python -m pip install --only-binary=:all: "numpy==1.19.5"
python -m pip install --no-deps "opencv-python-headless==4.5.5.64"
python -m pip install --no-deps -e .
python -m brnfs doctor --focus
```

**Windows PowerShell**

```powershell
py -3.9 -m venv .venv-focus
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv-focus\Scripts\Activate.ps1
python -m pip install --upgrade "pip<24" "setuptools<81" wheel
python -m pip install --only-binary=:all: "numpy==1.19.5"
python -m pip install --no-deps "opencv-python-headless==4.5.5.64"
python -m pip install --no-deps -e .
python -m brnfs doctor --focus
```

Then run the same command on either platform:

```bash
python -m brnfs focus --dataset examples/focus/IMG_2649 --output outputs/focus/IMG_2649_fused.png --levels 3 --mask soft --top mean --sharpness "Tenengrad+Blur"
```

The command writes `outputs/focus/IMG_2649_fused.png`. The first run also creates
an aligned-stack cache under `outputs/cache/focus/`.

This environment covers the focus CLI, not the GUI. The GUI requires additional
dependencies, including Pillow and Tkinter; use the full setup below for that
workflow. `doctor --focus` does not check GUI requirements.

## Full setup: bokeh rendering and GUI

The full setup scripts create or update `.venv`, install the dependency stack,
check or download bokeh weights, install BRnFS in editable mode, and build the
`scatter_cuda` extension unless explicitly skipped. They require an existing
Python 3.9 interpreter; they do not install Python itself. An existing `.venv`
that is broken or uses a different Python version is recreated by the scripts.

For CUDA rendering, provide an NVIDIA driver, CUDA Toolkit 11.7 including
`nvcc`, and a compatible C++ compiler. The Linux scripts look for a compiler
compatible with CUDA 11.7, preferring GCC/G++ 11 when the default is too new.
On Windows, install Visual Studio Build Tools with C++ support and run from a
shell where the compiler is available. The GUI additionally needs Python's
Tcl/Tk bindings and a working graphical display, including on WSL2.

### Linux / WSL2

```bash
PYTHON_BIN=python3.9 bash setup.sh
source .venv/bin/activate
python -m brnfs doctor
python -m brnfs gui
```

If your CUDA toolkit is installed elsewhere, set `CUDA_HOME` when running
setup:

```bash
CUDA_HOME=/usr/local/cuda-11.7 PYTHON_BIN=python3.9 bash setup.sh
```

### Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
.\.venv\Scripts\Activate.ps1
python -m brnfs doctor
python -m brnfs gui
```

The Windows script prefers `PYTHON_BIN`, then `py -3.9`, then a `python` command
that reports Python 3.9. For an explicit interpreter or toolkit location:

```powershell
$env:PYTHON_BIN = "C:\Path\To\Python39\python.exe"
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7"
.\setup.ps1
```

### Dependency versions and setup options

The table records the versions selected by
[`setup.sh`](../setup.sh), [`setup.ps1`](../setup.ps1), and
[`requirements.txt`](../requirements.txt). Some remaining dependencies use
ranges or no version pin; the scripts do not provide a complete lockfile.

| Component | Version selected by the full setup |
| --- | --- |
| Python | 3.9 |
| pip / setuptools | `pip<24`, `setuptools<81` |
| NumPy | `1.19.5` |
| PyTorch | `2.0.0+cu117` |
| torchvision / torchaudio | `0.15.0+cu117` / `2.0.0+cu117` |
| OpenCV | `opencv-python-headless==4.5.5.64` |
| PyTorch Lightning / torchmetrics | `1.9.5` / `0.11.4` |
| lightning-utilities / lightning-fabric | `0.15.2` / `1.9.5` |
| scikit-image / scikit-learn | `0.18.3` / `0.24.2` |
| kornia / timm | `0.5.0` / `1.0.22` |
| hydra-core / easydict | `1.1.0` / `1.9.0` |
| pandas / matplotlib | `<1.4` / `<3.6` |

The scripts install NumPy first, obtain the CUDA 11.7 PyTorch wheels from the
PyTorch wheel index, then install the remaining requirements. OpenCV and the
Lightning packages are installed with `--no-deps` to preserve this older
dependency stack. Installing `requirements.txt` alone does not reproduce the
full setup. An editable BRnFS install alone also does not install runtime
dependencies, because the package metadata declares an empty dependency list.

| Environment variable | Effect |
| --- | --- |
| `PYTHON_BIN` | Select the Python 3.9 interpreter used to create `.venv`. |
| `BRNFS_SKIP_WEIGHTS=1` | Skip bokeh weight checks and downloads. |
| `BRNFS_SKIP_SCATTER_CUDA=1` | Skip compiling the CUDA scatter extension. |
| `CUDA_HOME` | Select the installed CUDA toolkit location. |

To omit both weight preparation and the extension build:

```bash
BRNFS_SKIP_WEIGHTS=1 BRNFS_SKIP_SCATTER_CUDA=1 bash setup.sh
```

```powershell
$env:BRNFS_SKIP_WEIGHTS = "1"
$env:BRNFS_SKIP_SCATTER_CUDA = "1"
.\setup.ps1
```

Both skip flags still install the full dependency stack, including CUDA-enabled
PyTorch wheels. They do not select the minimal CPU installation above. In
PowerShell, remove the variables before a later full setup if you want those
steps enabled again:

```powershell
Remove-Item Env:BRNFS_SKIP_WEIGHTS -ErrorAction SilentlyContinue
Remove-Item Env:BRNFS_SKIP_SCATTER_CUDA -ErrorAction SilentlyContinue
```

## Bokeh weights and CUDA extension

The automatic bokeh pipeline expects these files at the paths registered in
[`brnfs/paths.py`](../brnfs/paths.py):

| Model | Required path relative to the repository root | Setup behavior |
| --- | --- | --- |
| LDF snapshot | `models/ldf/model-40` | Included in the repository; setup checks the existing file. |
| LDF backbone | `models/ldf/resnet50-19c8e357.pth` | Downloads if missing or below the script's minimum size. |
| LaMa | `models/lama/big-lama/models/best.ckpt` | Downloads if missing or below the script's minimum size. |
| DPT | `models/dpt/dpt_large-midas-2f21e586.pt` | Downloads if missing or below the script's minimum size. |

The three downloads use the URLs in the setup scripts under the upstream
[`ysheng/DrBokeh` model repository](https://huggingface.co/ysheng/DrBokeh/tree/main).
LaMa's configuration files already live in the vendored source tree. Setup's
file-size checks can detect obviously incomplete downloads; they are not
checksum verification. If `model-40` is absent, restore it from the repository
rather than expecting setup to download it.

To rebuild the extension against the PyTorch in an activated full environment:

```bash
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-11.7
export PATH="$CUDA_HOME/bin:$PATH"
bash scripts/build_scatter_cuda.sh
```

```powershell
.\.venv\Scripts\Activate.ps1
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7"
$env:Path = "$env:CUDA_HOME\bin;$env:Path"
.\scripts\build_scatter_cuda.ps1
```

The [Linux rebuild script](../scripts/build_scatter_cuda.sh) checks that the
`nvcc` version matches `torch.version.cuda` before compiling. The
[Windows rebuild script](../scripts/build_scatter_cuda.ps1) directly invokes the
build; check the toolkit and compiler setup first. Verify the active environment
with:

```bash
nvcc --version
python -c "import torch; print(torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); import scatter_cuda; print('scatter_cuda import OK')"
```

An importable CUDA extension and a usable GPU are needed for the intended fast
bokeh-rendering path. The renderer includes a CPU scatter fallback, but it is
not the recommended path for full-resolution bokeh examples.

## Command-line reference

Use `python -m brnfs` or the `brnfs` console script installed by the editable
installation. Available commands and options can also be inspected with:

```bash
python -m brnfs --help
python -m brnfs focus --help
python -m brnfs bokeh --help
python -m brnfs --version
```

### Focus stacking

```bash
python -m brnfs focus --dataset examples/focus/IMG_2649 --output outputs/focus/IMG_2649_fused.png --levels 3 --mask soft --top mean --sharpness "Tenengrad+Blur"
```

| Argument | Meaning and allowed values |
| --- | --- |
| `--dataset` | Required directory containing the input focus stack. |
| `--output` | Required output image filename; parent directories are created. |
| `--levels` | Pyramid levels, at least `1`; default `3`. |
| `--mask` | `soft` (default), `hard`, or `hard-min`. Soft masks smooth the maximum-sharpness selection; `hard-min` selects minimum sharpness for comparison. |
| `--top` | Fuse the top Gaussian level using `mean` (default) or `max`. |
| `--sharpness` | `L`, `GaussianBlur(L)`, `GaussianBlur(L^2)`, `Tenengrad+Blur` (default), `Variance(L)`, or `SML+Blur`. Quote values containing parentheses. |

The current CLI loads lowercase `*.png` files in filename order and aligns
them to the first image using affine ECC. Supply same-sized images of one
static scene with different focus settings. An ECC failure is reported and
that frame is kept unaligned; inspect the result if this occurs.

### Bokeh rendering

```bash
python -m brnfs bokeh --rgb examples/bokeh/IMG_2649.png --output outputs/bokeh/IMG_2649_bokeh.png --k-blur 30.0 --focal 0.10 --lens 71 --gamma 2.2
```

| Argument | Meaning and allowed values |
| --- | --- |
| `--rgb` | Required input image path. |
| `--output` | Required output image filename; parent directories are created. |
| `--k-blur` | Required blur-strength parameter `K`. |
| `--focal` | Required focal plane in normalized disparity coordinates, conventionally `[0, 1]`; not a distance in meters. |
| `--lens` | Required odd kernel size; the example uses `71`. |
| `--gamma` | Gamma exponent; default `2.2`. |
| `--disp` | Optional `.npz` disparity file containing an `H x W` array under key `data`. The pipeline resizes and normalizes it. |
| `--alpha` | Optional RGBA image; its fourth channel supplies the alpha mask. |
| `--verbose` | Also write disparity, alpha, and foreground/background layer debug images beside the output. |

Providing `--disp` or `--alpha` changes the preprocessing inputs. It does not
provide a supported way to omit all model dependencies: the current
preprocessing module initializes model inferencers when imported.

### GUI and demo assets

Launch the two-tab application from the full environment:

```bash
python -m brnfs gui
```

The asset command writes demo images and a `demo_status.md` file into the
selected directory. To keep a local regeneration separate from the committed
showcase assets, use an output directory:

```bash
python -m brnfs demo-assets --output outputs/demo-assets
```

Add `--focus-only` in the minimal CPU environment, or `--bokeh-only` to regenerate
just the bokeh example. Omitting `--output` writes to `docs/assets/readme`.
Each run replaces the status file in its output directory. The bokeh asset generator requires an importable
`scatter_cuda` module in its prerequisite check. Read `demo_status.md` after
generation: individual steps can be skipped or fail even when the command
returns successfully. Input filenames for the committed README images are
listed in [the asset notes](assets/readme/demo_status.md).

## Diagnostics, outputs, and troubleshooting

```bash
python -m brnfs doctor --focus
python -m brnfs doctor --bokeh
python -m brnfs doctor
```

`doctor --focus` checks whether NumPy and OpenCV can be located and whether the
example focus root exists. The bokeh check similarly looks for selected
modules, weight paths, and the example root. It labels a missing `scatter_cuda`
as `OPTIONAL`; that alone does not fail `doctor --bokeh`. These checks do not
execute inference, validate weight contents, or test `torch.cuda.is_available()`.
Use the CUDA verification command above before attempting bokeh rendering.

Outputs go to the filename passed to `--output`. Preprocessing caches use
`outputs/cache/focus/` and `outputs/cache/bokeh/`. Focus cache filenames depend
only on the dataset directory's basename, such as `IMG_2649_aligned.npy`.
After changing a stack, or when two different input directories share a name,
remove the corresponding generated cache file before rerunning. Bokeh caches
include input-file metadata and preprocessing settings, but not model weight
versions; clear the relevant generated cache when changing weights. The CLI
does not expose a cache-bypass flag.

| Symptom | What to check |
| --- | --- |
| Python version rejected or a pinned wheel cannot be installed | Confirm `python --version` is 3.9 and that the intended environment is active. The full scripts accept `PYTHON_BIN`; the focus path uses a separate `.venv-focus`. |
| `python` or `brnfs` cannot be found | Activate the environment. On Linux/WSL2, its interpreter can also be called directly, for example `.venv-focus/bin/python -m brnfs doctor --focus`. |
| Missing `cv2` | In the active environment, install `opencv-python-headless==4.5.5.64` with `--no-deps`; check that NumPy is still `1.19.5`. |
| Missing `pkg_resources` during bokeh imports | Install `setuptools<81` in the full environment, as the setup scripts do. |
| Missing weights or checkpoint-loading failure | Check the exact paths above. Rerun full setup without the weight-skip flag; restore the tracked LDF snapshot if absent. File existence alone is not proof of a valid checkpoint. |
| Missing `nvcc`, CUDA-version mismatch, or extension-build failure | Check `nvcc --version`, `torch.version.cuda`, `CUDA_HOME`, and the C++ compiler. Use the CUDA 11.7 toolchain expected by the current setup, then rebuild in the same environment. |
| Bokeh rendering is unexpectedly slow | Check GPU availability and actually import `scatter_cuda`; `doctor --bokeh` alone can pass with the extension missing. The initial uncached run also includes model preprocessing. |
| Missing `tkinter` or no display for the GUI | Install Tcl/Tk bindings matching the Python 3.9 interpreter and use a graphical session. The focus CLI can run without either. |
| No focus images found, shape errors, or alignment warnings | Use readable lowercase `.png` files with equal dimensions; review any frames whose ECC alignment failed. |
| Focus output ignores changed inputs | Remove that dataset's generated aligned-stack cache and rerun. Use distinct directory basenames for distinct stacks. |

For the pipeline stages behind these commands, see
[`architecture.md`](architecture.md).

