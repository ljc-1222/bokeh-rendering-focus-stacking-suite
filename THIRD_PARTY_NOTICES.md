# Third-party notices

BRnFS includes source code adapted or copied from the projects below. Retain the
included upstream license texts and notices when redistributing those components.
Code, model weights, and example images have separate provenance.

The combined project currently has no standalone root license. These entries
record the license texts available in this checkout and the remaining gaps.

## Dr.Bokeh

- **Upstream:** [ShengCN/DrBokeh-Src](https://github.com/ShengCN/DrBokeh-Src).
- **Role:** layered, occlusion-aware bokeh rendering and its associated
  depth/saliency/inpainting integration.
- **Local code:** [`app/bokeh_rendering/DScatter/`](app/bokeh_rendering/DScatter/),
  [`brnfs/cuda_src/`](brnfs/cuda_src/), and adapted preprocessing/inference code
  under [`app/bokeh_rendering/`](app/bokeh_rendering/).
- **CUDA provenance:** `scatter_cuda.cpp` and `scatter_cuda_kernel.cu` are
  unchanged from the corresponding upstream files compared at
  [`24d96e4`](https://github.com/ShengCN/DrBokeh-Src/tree/24d96e4fb76a698494e86835c84a7c45ae2ebc10/app/cuda-src).
- **Local integration:** shared entry points, GUI controls, preprocessing caches,
  and centralized paths around the renderer.
- **License status:** no standalone Dr.Bokeh license text is included in this
  checkout. Establish the applicable upstream terms before redistributing these
  components under a project-wide license.

Dr.Bokeh's upstream pipeline already integrates LDF, LaMa, and MiDaS/DPT. Those
components retain their own attribution below.

## DPT — Dense Prediction Transformer

- **Upstream:** [isl-org/DPT](https://github.com/isl-org/DPT).
- **Role:** monocular disparity prediction using the pretrained DPT-Large model.
- **Local code:** [`app/bokeh_rendering/Depth/DPT/`](app/bokeh_rendering/Depth/DPT/).
- **License:** MIT, retained in the
  [vendored license](app/bokeh_rendering/Depth/DPT/LICENSE) and
  [`LICENSES/DPT_MIT.txt`](LICENSES/DPT_MIT.txt).

Path handling and inference wrappers have been adapted for this suite.

## LaMa — Large Mask Inpainting

- **Upstream:** [advimman/lama](https://github.com/advimman/lama).
- **Role:** fill the background RGB and disparity behind the salient foreground.
- **Local code:** [`app/bokeh_rendering/Inpainting/lama/`](app/bokeh_rendering/Inpainting/lama/).
- **License:** Apache License 2.0, retained in the
  [vendored license](app/bokeh_rendering/Inpainting/lama/LICENSE) and
  [`LICENSES/LAMA_APACHE-2.0.txt`](LICENSES/LAMA_APACHE-2.0.txt).

The suite uses the pretrained generator for inference.

## LDF — Label Decoupling Framework

- **Upstream:** [weijun-arc/LDF](https://github.com/weijun-arc/LDF), also integrated
  into Dr.Bokeh's rendering pipeline.
- **Role:** salient-object inference, used as an alpha-like foreground map.
- **Local code:** [`app/bokeh_rendering/Salient/LDF/`](app/bokeh_rendering/Salient/LDF/).
- **License status:** no standalone LDF license text is included in this
  checkout. Check the applicable upstream terms before redistribution.

## Model weights and example images

[`setup.sh`](setup.sh) and [`setup.ps1`](setup.ps1) download the DPT, LaMa, and
ResNet backbone weights from the upstream
[ysheng/DrBokeh model repository](https://huggingface.co/ysheng/DrBokeh).
The LDF `model-40` snapshot is tracked under `models/ldf/`. Model paths and
retrieval behavior are listed in [Setup and usage](docs/setup.md).

Inputs under `examples/` and outputs under `docs/assets/readme/` are included for
demonstrating the workflows. This checkout does not include a separate
image-source manifest or image license; their terms are not established by the
DPT or LaMa source-code licenses.
