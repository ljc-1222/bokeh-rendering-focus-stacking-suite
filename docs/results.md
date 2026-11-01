# Results and Evaluation

BRnFS demonstrates two complementary depth-of-field operations: rendering
synthetic bokeh from one image and combining a focus stack into an all-in-focus
image. The numerical results below are recorded results from the **original
project evaluation**. The included examples provide a smaller, runnable view of
the two workflows.

## Focus Stacking Experiments

The original evaluation used **791 image stacks** and **24 configurations**:
six sharpness definitions, two mask types (`soft` and `hard`), and two rules for
the coarsest Gaussian level (`mean` and `max`). Quality was measured using the
multi-source `Q_AB/F` implementation described below; higher scores indicate
better preservation of the selected source edges.

The highest recorded mean was **0.8854 ± 0.0057** for
**Tenengrad+Blur / soft / mean**. All `±` values on this page are the recorded
95% confidence-interval half-widths.

The following table isolates the main effect of sharpness by averaging over
both mask types and both top-level rules. These averages describe groups of
configurations, so they should be compared within this table.

| Sharpness definition | Mean Q_AB/F ± 95% CI |
| --- | ---: |
| `L` | 0.8017 ± 0.0049 |
| `GaussianBlur(L)` | 0.8567 ± 0.0056 |
| `GaussianBlur(L^2)` | 0.8644 ± 0.0057 |
| `Tenengrad+Blur` | 0.8823 ± 0.0057 |
| `Variance(L)` | 0.8707 ± 0.0060 |
| `SML+Blur` | 0.8715 ± 0.0064 |

Paired comparisons quantify changes on the same stacks. For the mask and
top-level comparisons, results were averaged over the other configuration
choices.

| Comparison | Paired improvement in Q_AB/F ± 95% CI | Stacks improved |
| --- | ---: | ---: |
| `Tenengrad+Blur` vs. `L`, with `soft / mean` fixed | +0.0725 ± 0.0015 | 98.5% |
| `soft` vs. `hard` masks | +0.0106 ± 0.0005 | 98.0% |
| `mean` vs. `max` top-level fusion | +0.0007 ± 0.0002 | 65.1% |

Under the original protocol, sharpness choice produced the largest differences,
soft masks usually improved the result, and the top-level rule had a smaller
effect. The `0.8017` main-effect average for `L` is not the score of the specific
`L / soft / mean` baseline used in the first paired comparison.

A separate pyramid-depth study showed a strong increase in mean quality from
one to three levels, followed by a plateau. This observation motivated the
current three-level default. The available results do not include measured
runtime curves, so the depth choice is supported here by quality saturation.

## How to Interpret Q_AB/F

The [evaluation implementation](../app/focus_stacking/evaluation.py) measures
edge-strength and edge-orientation preservation. For more than two source
images, it extends the two-source metric by choosing the two highest-scoring
sources **at each pixel**, using supplied sharpness maps, then aggregating their
edge-preservation scores with edge-strength weights.

The [focus-stacking GUI](../brnfs/ui/focus_stacking.py) supplies the finest-level
sharpness maps produced by the selected fusion configuration. Consequently,
changing the sharpness definition changes both the fused image and the sources
used to evaluate it. The cross-sharpness ranking above is therefore specific to
this evaluation protocol. With the input stack, preprocessing, and sharpness
definition held fixed, mask and top-level comparisons retain the same selected
source references.

A comparison intended to isolate sharpness choice should select evaluation
sources with one fixed rule shared by all candidates. That would constitute a
separate evaluation; the historical numbers above belong to the original rule.
`Q_AB/F` measures edge preservation and does not by itself establish perceptual
quality across all scenes.

## Recorded Demonstration

The [original project video](https://youtu.be/h5yaBS06S4U) provides visual evidence for the application workflow; the recording is **not** a new run of this repository revision.

| Recording time | Observed operation | Interpretation |
| --- | --- | --- |
| [06:07–06:27](https://youtu.be/h5yaBS06S4U?t=367) | Focus stacking: select the flower stack, use the fusion controls, and generate a result with an on-screen score. | The program produces an all-in-focus result. The recording does not establish the dataset-wide scores tabulated above. |
| [06:28–06:42](https://youtu.be/h5yaBS06S4U?t=388) | Change the mask mode to **Hard (Argmin)** and regenerate; the flowers become noticeably blurred. | This is an intentionally incorrect **minimum-sharpness** negative control, not a comparison of normal hard-argmax and soft masks. |
| [06:45–06:52](https://youtu.be/h5yaBS06S4U?t=405) | Fuse a second scene featuring a “VICINITY MAP” sign. | Demonstrates another recorded scene; its full source stack is **not** included with the published example inputs. |
| [07:03–07:38](https://youtu.be/h5yaBS06S4U?t=423) | Select an image, preprocess the scene, enable the red **focus overlay**, and adjust bokeh controls. | The overlay shows a predicted focus region and updates as controls change; it is **not** the final bokeh image. |
| [07:39–07:53](https://youtu.be/h5yaBS06S4U?t=459) | Press **Render** and display the bokeh result. | Actual bokeh rendering is a separate operation; this single screen recording is not a timing benchmark. |

The recorded interface predates later GUI/CLI and path changes. For reproducible **current-version examples**, use the commands in [Setup & CLI](setup.md). The clean, high-resolution images embedded in the [README](../README.md) are preferable to extracting low-resolution results from a screen recording.

## Visual Examples

The existing [README showcase](../README.md#results-at-a-glance) displays the original flower scene and the saved bokeh/focus outputs. Its focus result is generated from the **complete 15-image stack**, even though only three representative inputs are displayed. The current demo asset status and input filenames are documented in [the generated asset notes](assets/readme/demo_status.md).

For bokeh, the qualitative observation is control over **how much** defocus is applied and **where** the focal region lies. The recorded focus overlay is a preview signal, not an additional rendered output. Actual bokeh quality depends on relative disparity, foreground separation, inpainting, and the occlusion-aware renderer.

For focus stacking, the recorded Hard (Argmin) failure illustrates the role of selecting strong local detail. It should not be used as evidence that ordinary hard selection is intrinsically worse than soft blending; the 791-stack **soft vs. hard-argmax** ablation above is a separate comparison.

## Reproduction Scope

The repository currently contains one example focus stack and its associated
bokeh input. The complete 791-stack manifest, dataset provenance, per-stack
results, batch evaluation runner, and confidence-interval calculation procedure
are not included. Reconstructing the original tables requires those materials,
including the exact preprocessing settings and rules for failures and ties.

After [installation](setup.md), regenerate both demo workflows from the
repository root in the full environment:

```bash
python -m brnfs demo-assets --output outputs/demo-assets
```

Add `--focus-only` when using the minimal CPU environment. Practical bokeh
rendering requires the model weights and CUDA renderer. Each run writes its
outcome to `outputs/demo-assets/demo_status.md`; inspect it for skipped or failed
steps. The [committed asset notes](assets/readme/demo_status.md) identify the
source frames used for the showcase. These commands regenerate example images.
The focus GUI also displays a `Q_AB/F` score for the selected example;
the [CLI focus runner](../brnfs/focus/runner.py) writes the fused image without
calculating a score.
