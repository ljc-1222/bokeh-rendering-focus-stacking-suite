# BRnFS Architecture

BRnFS combines two independent image-processing pipelines behind one CLI and GUI. The `brnfs` package owns the public entry points and path registry; algorithm code that originated from the source projects remains under `app/`.

## Entry Points

```mermaid
flowchart LR
  CLI["python -m brnfs"] --> GUI["brnfs gui"]
  CLI --> BOKEH["brnfs bokeh"]
  CLI --> FOCUS["brnfs focus"]
  CLI --> DOCTOR["brnfs doctor"]
  GUI --> BokehTab["Bokeh Rendering tab"]
  GUI --> FocusTab["Focus Stacking tab"]
```

The public command surface is `python -m brnfs` or the installed `brnfs` console script.

## Bokeh Rendering Pipeline

```mermaid
flowchart TD
  RGB["RGB image"] --> DPT["DPT disparity"]
  RGB --> LDF["LDF saliency / alpha"]
  RGB --> LAMA["LaMa background inpainting"]
  DPT --> Layers["Foreground/background RGBAD layers"]
  LDF --> Layers
  LAMA --> Layers
  Layers --> Scatter["DScatter renderer"]
  Scatter --> Output["Rendered bokeh image"]
```

The bokeh path uses:

- `models/dpt/` for DPT depth weights.
- `models/ldf/` for LDF/ResNet weights.
- `models/lama/` for LaMa checkpoints.
- `outputs/cache/bokeh/` for preprocessed RGBAD layers.

`scatter_cuda` is the fast renderer backend. If it is unavailable, the code can fall back to a Python/CPU reference implementation, but this is too slow for normal demo use.

## Focus Stacking Pipeline

```mermaid
flowchart TD
  Stack["Focus stack dataset"] --> Preprocess["Load, resize, ECC align"]
  Preprocess --> Pyramids["Gaussian + Laplacian pyramids"]
  Pyramids --> Sharpness["Sharpness maps"]
  Sharpness --> Masks["Hard or soft decision masks"]
  Masks --> Fusion["Pyramid fusion"]
  Fusion --> Reconstruct["Reconstruct all-in-focus image"]
```

Focus stacking reads datasets from `examples/focus/<dataset>/`, writes final images under `outputs/focus/`, and caches aligned stacks under `outputs/cache/focus/`.

## Path Policy

All new code should import paths from `brnfs.paths`; root-relative data/model/output paths should not be hardcoded elsewhere.

Canonical data locations:

```text
examples/       sample inputs
models/         downloaded weights
outputs/        generated outputs and caches
docs/assets/    documentation assets
```
