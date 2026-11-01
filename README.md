# BRnFS — Bokeh Rendering & Focus Stacking

**Two ways to change depth of field:** synthesize shallow-focus bokeh from a single image, or combine a focus bracket into an all-in-focus image. BRnFS brings both workflows into one Tkinter GUI and a common command-line interface.

[**Watch the original demo**](https://youtu.be/h5yaBS06S4U) · [Results & evaluation](docs/results.md) · [Setup & CLI](docs/setup.md) · [Architecture](docs/architecture.md)

## Results at a glance

### 1. Bokeh rendering — reduce depth of field

A single RGB image is processed into depth, foreground, and inpainted background layers before an occlusion-aware scatter renderer produces the defocused result.

<table>
<tr><th>Input image</th><th>Synthetic bokeh</th></tr>
<tr>
<td><img src="docs/assets/readme/bokeh_input.png" alt="Original all-sharp photo of flowers" width="440"></td>
<td><img src="docs/assets/readme/bokeh_rendered.png" alt="Synthetic shallow-depth-of-field result on the same scene" width="440"></td>
</tr>
</table>

Shown setting: `K=30`, `focal=0.10`, `lens=71`, `gamma=2.2`. The focal parameter is relative to normalized disparity, not a distance in meters. The bokeh renderer builds on [Dr.Bokeh](https://github.com/ShengCN/DrBokeh-Src); the integration and interface are described [below](#implementation-and-provenance).

### 2. Focus stacking — extend depth of field

The input stack is registered by ECC, decomposed into Laplacian pyramids, fused using sharpness-driven masks, and reconstructed into a single image.

<table>
<tr><th colspan="3">Three representative input frames</th><th>Final fusion of all 15 frames</th></tr>
<tr>
<td><img src="docs/assets/readme/focus_input_1.png" alt="Near-focused input frame 1 of 15" width="165"></td>
<td><img src="docs/assets/readme/focus_input_2.png" alt="Middle-focused input frame 8 of 15" width="165"></td>
<td><img src="docs/assets/readme/focus_input_3.png" alt="Far-focused input frame 15 of 15" width="165"></td>
<td><img src="docs/assets/readme/focus_fused.png" alt="All-in-focus output composed from 15 frames" width="390"></td>
</tr>
</table>

Shown setting: **Tenengrad+Blur**, **soft mask**, **mean** coarsest-level fusion, and **3 pyramid levels**. The three source images above are previews only; the actual output uses **all 15** images in `examples/focus/IMG_2649/`.

## Recorded GUI walkthrough

The original [8-minute demonstration](https://youtu.be/h5yaBS06S4U) includes both working GUIs. Jump directly to the relevant operations:

| Workflow | Watch | What the recording demonstrates |
| --- | --- | --- |
| Focus stacking | [06:07 — selecting frames and fusion](https://youtu.be/h5yaBS06S4U?t=367) | Source preview, fusion controls, Generate, and a fused output |
| Negative control | [06:28 — Hard (Argmin)](https://youtu.be/h5yaBS06S4U?t=388) | Deliberately selecting less-sharp content produces a blurred output; **this is not** the normal hard-argmax baseline |
| Bokeh rendering | [07:03 — preprocessing and focus controls](https://youtu.be/h5yaBS06S4U?t=423) | Reusing scene preprocessing, adjusting focus/blur settings, and viewing the red focus overlay |
| Final render | [07:39 — Render](https://youtu.be/h5yaBS06S4U?t=459) | Computing the bokeh image separately from the overlay preview |

The video is an **archived project demonstration**; the GUI and repository have since been reorganized. The red outline is a **focus preview**, not a live bokeh render. Rendering is triggered explicitly with **Render**. See [Results & evaluation](docs/results.md#recorded-demonstration) for what can and cannot be concluded from this recording.

## What we implemented

- **One interactive suite:** a two-tab GUI, preview and save actions, a unified `python -m brnfs` CLI, and shared path/asset handling.
- **Reusable bokeh preprocessing:** cache depth, saliency, inpainting, and foreground/background layers so successive parameter renders need not recompute the automatic model stages.
- **Configurable multi-scale focus fusion:** ECC alignment, Gaussian/Laplacian pyramids, six sharpness definitions, hard/soft selection masks, and mean/max residual fusion.
- **Experimental analysis:** an ablation of fusion design choices across a recorded **791-stack, 24-configuration** evaluation.

### Focus-stacking experimental highlights

| Recorded outcome | Result |
| --- | ---: |
| Highest mean edge-preservation score (Tenengrad+Blur / soft / mean) | `Q_AB/F = 0.8854 ± 0.0057` (95% CI) |
| Soft masks vs. normal hard masks, paired difference | `+0.0106 ± 0.0005` |
| Stacks improved by soft masks under the recorded protocol | **98.0%** |

These are results of the **original project evaluation**, not a fresh benchmark run on the current checkout. The extended multi-source `Q_AB/F` metric selects references using sharpness maps; comparisons across different sharpness definitions therefore have an evaluation-dependence caveat. See the [full tables and interpretation](docs/results.md#focus-stacking-experiments).

## Quick start — focus stacking on CPU

The focus CLI does not require PyTorch, bokeh weights, or CUDA. It uses **Python 3.9** and pinned NumPy/OpenCV versions. From Linux or WSL2:

```bash
git clone https://github.com/ljc-1222/bokeh-rendering-focus-stacking-suite.git
cd bokeh-rendering-focus-stacking-suite
python3.9 -m venv .venv-focus
source .venv-focus/bin/activate
python -m pip install --upgrade "pip<24" "setuptools<81" wheel
python -m pip install --only-binary=:all: numpy==1.19.5
python -m pip install --no-deps opencv-python-headless==4.5.5.64
python -m pip install --no-deps -e .
python -m brnfs focus --dataset examples/focus/IMG_2649 \
  --output outputs/focus/IMG_2649_fused.png \
  --levels 3 --mask soft --top mean --sharpness Tenengrad+Blur
```

Output: `outputs/focus/IMG_2649_fused.png`. The aligned input stack is cached for subsequent runs. [Windows instructions and troubleshooting](docs/setup.md#focus-stacking-on-cpu) are in the setup guide.

## Full GUI and bokeh rendering

Bokeh uses pretrained DPT/LDF/LaMa models and a CUDA 11.7-compatible PyTorch/scatter extension for practical performance. On Linux or WSL2 with Python 3.9 and the required NVIDIA CUDA toolchain:

```bash
PYTHON_BIN=python3.9 bash setup.sh
source .venv/bin/activate
python -m brnfs doctor
python -m brnfs gui
```

For the full requirements, PowerShell instructions, model downloads, CLI rendering commands, and GPU troubleshooting, use [Setup & CLI](docs/setup.md). A check reported by `doctor` does not constitute a successful GPU inference test.

## Implementation and provenance

- [Architecture](docs/architecture.md) — processing stages, masks, parameter meanings, and module map.
- [Results & evaluation](docs/results.md) — detailed ablations, scoring caveats, and recorded GUI examples.
- [Third-party notices](THIRD_PARTY_NOTICES.md) — upstream sources, model provenance, and license status.

The bokeh scattering renderer and its CUDA kernels come from **[Dr.Bokeh](https://github.com/ShengCN/DrBokeh-Src)**, whose existing pipeline also integrates DPT, LDF, and LaMa. This project contributes the shared presentation/interaction layer, workflow organization, and its focus-stacking implementation and evaluation; it does **not** claim the upstream renderer or pretrained models as newly developed here.

**Limitations:** bokeh may produce artifacts near thin foreground structures or inaccurate depth/alpha boundaries; focus fusion can be affected by imperfect registration, scene motion, and mask selection. The recorded 791-stack evaluation's raw input manifest and per-stack scores are not included in this repository. No root license has yet been assigned to the combined codebase; consult [Third-party notices](THIRD_PARTY_NOTICES.md) before redistribution.

**Authors:** Jin-Cheng Lin and Hsin-Chang Tsai · National Tsing Hua University, Electrical Engineering · EE3662 Digital Signal Processing Laboratory final project.
