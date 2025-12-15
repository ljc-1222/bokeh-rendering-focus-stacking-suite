# Third-Party Notices

This project vendors (copies) portions of upstream open-source projects for inference/runtime use.
The upstream license texts are included in this repository and must be preserved when redistributing.

## DPT (Depth Prediction Transformer)

- **Upstream project**: `isl-org/DPT` (a.k.a. "DPT")
- **Upstream URL**: `https://github.com/isl-org/DPT`
- **License**: MIT
- **Vendored code location**: `bokeh_rendering_and_focus_stacking_suite/app/bokeh_rendering/Depth/DPT/`
- **Upstream license in-tree**: `bokeh_rendering_and_focus_stacking_suite/app/bokeh_rendering/Depth/DPT/LICENSE`
- **Copied license text**: `bokeh_rendering_and_focus_stacking_suite/LICENSES/DPT_MIT.txt`

Notes:
- Some integration glue (paths / wrappers) may differ from upstream to fit this demo pipeline.

## LaMa (Large Mask Inpainting)

- **Upstream project**: `advimman/lama` (LaMa)
- **Upstream URL**: `https://github.com/advimman/lama`
- **License**: Apache License 2.0
- **Vendored code location**: `bokeh_rendering_and_focus_stacking_suite/app/bokeh_rendering/Inpainting/lama/`
- **Upstream license in-tree**: `bokeh_rendering_and_focus_stacking_suite/app/bokeh_rendering/Inpainting/lama/LICENSE`
- **Copied license text**: `bokeh_rendering_and_focus_stacking_suite/LICENSES/LAMA_APACHE-2.0.txt`

Notes:
- This repo uses the vendored LaMa generator for inference; training utilities remain vendored but are not required for the demo.


