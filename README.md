# BRnFS — Bokeh Rendering & Focus Stacking

An interactive computational photography suite for exploring depth of field in
two directions: **render a shallow-focus image from one all-sharp input**, or
**combine a focus stack into an all-in-focus image**. Both workflows are available
through a shared Tkinter interface and a command-line interface.

Final project for **EE3662 Digital Signal Processing Laboratory**, National Tsing
Hua University. By **Jin-Cheng Lin** and **Hsin-Chang Tsai**.

[Results and experiments](docs/results.md) · [Setup and usage](docs/setup.md) ·
[How it works](docs/architecture.md)

## See the results

### Bokeh rendering

The renderer uses estimated disparity, foreground separation, and an inpainted
background to synthesize defocus. Adjust the focal plane, blur strength, and lens
kernel size to control where the scene stays sharp and how strongly it blurs.

<table>
  <tr>
    <th>All-sharp input</th>
    <th>Rendered bokeh</th>
  </tr>
  <tr>
    <td><img src="docs/assets/readme/bokeh_input.png" alt="All-sharp flower scene used as the bokeh input" width="440"></td>
    <td><img src="docs/assets/readme/bokeh_rendered.png" alt="Synthetic bokeh output from the same flower scene" width="440"></td>
  </tr>
</table>

Included example: `IMG_2649`, rendered with `K=30`, `focal=0.10`, `lens=71`, and
`gamma=2.2`. Focal values refer to normalized disparity; the lens setting is a
kernel size. The GUI provides a focus overlay for previewing parameter changes;
press **Render** to compute the final image.

### Focus stacking

The fusion pipeline aligns the inputs and selects sharp detail at multiple
spatial scales. This example uses **all 15 images** in the included stack; three
representative input frames are shown below.

<table>
  <tr>
    <th>Input 1 of 15</th>
    <th>Input 8 of 15</th>
    <th>Input 15 of 15</th>
    <th>Fused output</th>
  </tr>
  <tr>
    <td><img src="docs/assets/readme/focus_input_1.png" alt="First input frame, with the nearer flowers in focus" width="220"></td>
    <td><img src="docs/assets/readme/focus_input_2.png" alt="Eighth input frame, with the middle flowers in focus" width="220"></td>
    <td><img src="docs/assets/readme/focus_input_3.png" alt="Fifteenth input frame, with the background in focus" width="220"></td>
    <td><img src="docs/assets/readme/focus_fused.png" alt="All-in-focus result fused from all 15 input frames" width="220"></td>
  </tr>
</table>

Settings: **Tenengrad+Blur**, **soft masks**, **mean** top-layer fusion, and
**3 pyramid levels**. Image filenames and regeneration commands are documented in
[Results and experiments](docs/results.md).

The original project evaluation compared **24 configurations on 791 image
stacks**. Tenengrad+Blur with soft masks and mean top-layer fusion achieved the
highest average score, **Q_AB/F = 0.8854 ± 0.0057** (95% confidence interval),
under the project's multi-source evaluation rule. The
[experiment summary](docs/results.md#focus-stacking-experiments) explains the
ablations, the metric's source-selection dependence, and which results can be
regenerated from this checkout.

## What this project brings together

- **Interactive depth-of-field editing:** two independent workflows in one GUI,
  with input previews, explicit rendering/fusion, and manual image saving.
- **Reusable bokeh preprocessing:** cache estimated scene layers, preview the
  focus region, and render additional parameter settings without repeating the
  automatic depth, saliency, and inpainting stages.
- **Configurable focus fusion:** ECC alignment and Laplacian-pyramid fusion with
  six sharpness definitions, hard or soft masks, and mean or max top-layer fusion.
- **Design analysis:** compare fusion choices and pyramid depth, and expose
  intermediate bokeh images through the CLI for inspecting artifacts.

The bokeh pipeline adapts [Dr.Bokeh](https://github.com/ShengCN/DrBokeh-Src),
including its CUDA renderer and pretrained DPT, LDF, and LaMa components. The
suite's interface, workflow integration, and focus-stacking experiments build on
these existing methods. See [How it works](docs/architecture.md) for module
boundaries and [Third-party notices](THIRD_PARTY_NOTICES.md) for source attribution
and license status.

## Try focus stacking on CPU

The focus CLI needs Python 3.9, NumPy, and OpenCV. It does not load the bokeh models
or require a CUDA build. From Linux or WSL2:

```bash
git clone https://github.com/ljc-1222/bokeh-rendering-focus-stacking-suite.git
cd bokeh-rendering-focus-stacking-suite
python3.9 -m venv .venv-focus
source .venv-focus/bin/activate
python -m pip install --upgrade "pip<24" "setuptools<81" wheel
python -m pip install --only-binary=:all: numpy==1.19.5
python -m pip install --no-deps opencv-python-headless==4.5.5.64
python -m pip install --no-deps -e .
python -m brnfs doctor --focus
python -m brnfs focus \
  --dataset examples/focus/IMG_2649 \
  --output outputs/focus/IMG_2649_fused.png \
  --levels 3 --mask soft --top mean --sharpness Tenengrad+Blur
```

Open `outputs/focus/IMG_2649_fused.png` to inspect the result. The first run also
aligns and caches the input stack. For Windows commands, dependency details, and
cache behavior, see [Setup and usage](docs/setup.md).

## Run the GUI and bokeh workflow

The full suite uses the pinned Python 3.9 environment and pretrained model
weights. Practical bokeh rendering requires an NVIDIA GPU and the `scatter_cuda`
extension; the setup scripts target PyTorch with CUDA 11.7.

From the repository root on Linux or WSL2:

```bash
bash setup.sh
source .venv/bin/activate
python -m brnfs doctor
python -m brnfs gui
```

The **Bokeh Rendering** tab separates preprocessing from rendering and provides
a focus overlay. The **Focus Stacking** tab previews the source sequence and lets
you choose the fusion settings before generating and saving the result.

For CLI rendering, Windows setup, model locations, intermediate outputs, and
CUDA troubleshooting, follow [Setup and usage](docs/setup.md). The minimal CPU
environment above is for the focus CLI; use the full environment for the GUI.

## Scope and limitations

Bokeh quality depends on the estimated disparity, foreground mask, and inpainted
background, especially around thin structures and occlusion boundaries. The
current renderer uses a circular aperture model. Focus stacking depends on image
alignment and the sharpness cues used to select detail; scene motion and blending
can introduce artifacts. These dependencies are described in
[How it works](docs/architecture.md).

This repository includes representative inputs and saved demo outputs. The
historical 791-stack evaluation is summarized with its limitations in
[Results and experiments](docs/results.md). Upstream license texts are retained
under `LICENSES/`; the combined project has no root license. Consult
[Third-party notices](THIRD_PARTY_NOTICES.md) for the status of each component.
