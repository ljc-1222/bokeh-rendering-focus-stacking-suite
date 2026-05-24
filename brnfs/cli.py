"""Unified command line interface for BRnFS."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

from brnfs import __version__
from brnfs import paths


SHARPNESS_CHOICES = [
    "L",
    "GaussianBlur(L)",
    "GaussianBlur(L^2)",
    "Tenengrad+Blur",
    "Variance(L)",
    "SML+Blur",
]


def _add_common_output_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", required=True, type=Path, help="Output file or directory path.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brnfs",
        description="Bokeh rendering and focus stacking suite.",
    )
    parser.add_argument("--version", action="version", version=f"brnfs {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    gui = subparsers.add_parser("gui", help="Launch the unified Tkinter GUI.")
    gui.set_defaults(func=run_gui)

    bokeh = subparsers.add_parser("bokeh", help="Render synthetic bokeh from one RGB image.")
    bokeh.add_argument("--rgb", required=True, type=Path, help="Input RGB image.")
    _add_common_output_arg(bokeh)
    bokeh.add_argument("--k-blur", required=True, type=float, help="Blur strength K.")
    bokeh.add_argument("--focal", required=True, type=float, help="Focal plane in normalized disparity space.")
    bokeh.add_argument("--lens", required=True, type=int, help="Odd lens kernel size.")
    bokeh.add_argument("--disp", type=Path, default=None, help="Optional disparity .npz path.")
    bokeh.add_argument("--alpha", type=Path, default=None, help="Optional RGBA alpha path.")
    bokeh.add_argument("--gamma", type=float, default=2.2, help="Gamma correction exponent.")
    bokeh.add_argument("--verbose", action="store_true", help="Write intermediate debug images.")
    bokeh.set_defaults(func=run_bokeh)

    focus = subparsers.add_parser("focus", help="Fuse a multi-focus image stack.")
    focus.add_argument("--dataset", required=True, type=Path, help="Directory containing focus stack images.")
    _add_common_output_arg(focus)
    focus.add_argument("--levels", type=int, default=3, help="Number of pyramid levels.")
    focus.add_argument("--mask", choices=["soft", "hard", "hard-min"], default="soft", help="Decision mask mode.")
    focus.add_argument("--top", choices=["mean", "max"], default="mean", help="Top Gaussian fusion method.")
    focus.add_argument(
        "--sharpness",
        choices=SHARPNESS_CHOICES,
        default="Tenengrad+Blur",
        help="Sharpness metric used for mask selection.",
    )
    focus.set_defaults(func=run_focus)

    doctor = subparsers.add_parser("doctor", help="Inspect local runtime prerequisites.")
    doctor.add_argument("--bokeh", action="store_true", help="Only check bokeh prerequisites.")
    doctor.add_argument("--focus", action="store_true", help="Only check focus stacking prerequisites.")
    doctor.set_defaults(func=run_doctor)

    assets = subparsers.add_parser("demo-assets", help="Generate README demo assets from examples.")
    assets.add_argument("--output", type=Path, default=paths.REPO_ROOT / "docs" / "assets" / "readme")
    group = assets.add_mutually_exclusive_group()
    group.add_argument("--focus-only", action="store_true", help="Only generate focus stacking assets.")
    group.add_argument("--bokeh-only", action="store_true", help="Only generate bokeh assets.")
    assets.set_defaults(func=run_demo_assets)

    return parser


def run_gui(_args: argparse.Namespace) -> int:
    from brnfs.ui.gui import main as gui_main

    gui_main()
    return 0


def run_bokeh(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np

    from app.bokeh_rendering.gui_engine import BokehEngine

    engine = BokehEngine(lens=args.lens)
    pre = engine.preprocess(
        rgb_path=args.rgb,
        disp_path=args.disp,
        alpha_path=args.alpha,
    )
    bokeh = engine.render(
        pre,
        focal=args.focal,
        k_blur=args.k_blur,
        gamma=args.gamma,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bokeh_u8 = np.clip(bokeh * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(args.output), cv2.cvtColor(bokeh_u8, cv2.COLOR_RGB2BGR))
    print(f"Saved bokeh image to: {args.output}")

    if args.verbose:
        prefix = args.output.with_suffix("")
        cv2.imwrite(str(prefix) + "_disp.png", np.clip(pre.disp[..., 0] * 255.0, 0, 255).astype(np.uint8))
        cv2.imwrite(str(prefix) + "_alpha.png", np.clip(pre.alpha[..., 0] * 255.0, 0, 255).astype(np.uint8))
        for name, layer in (("fg", pre.fg_rgbad), ("bg", pre.bg_rgbad)):
            rgb_u8 = np.clip(layer[..., :3] * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(str(prefix) + f"_{name}_rgb.png", cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(prefix) + f"_{name}_alpha.png", np.clip(layer[..., 3] * 255.0, 0, 255).astype(np.uint8))
            cv2.imwrite(str(prefix) + f"_{name}_disp.png", np.clip(layer[..., 4] * 255.0, 0, 255).astype(np.uint8))
    return 0


def run_focus(args: argparse.Namespace) -> int:
    from brnfs.focus.runner import run_focus_stacking

    run_focus_stacking(
        dataset_dir=args.dataset,
        output_path=args.output,
        levels=args.levels,
        mask=args.mask,
        top=args.top,
        sharpness=args.sharpness,
    )
    return 0


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _check_path(label: str, path: Path, *, required: bool = True) -> bool:
    exists = path.exists()
    prefix = "OK" if exists else ("MISSING" if required else "OPTIONAL")
    print(f"[{prefix}] {label}: {path}")
    return exists or not required


def check_focus() -> bool:
    ok = True
    print("Focus stacking")
    for mod in ("numpy", "cv2"):
        present = _has_module(mod)
        print(f"[{'OK' if present else 'MISSING'}] Python module: {mod}")
        ok = ok and present
    ok = _check_path("example focus root", paths.focus_input_dir()) and ok
    return ok


def check_bokeh(*, require_scatter: bool = False) -> bool:
    ok = True
    print("Bokeh rendering")
    for mod in ("numpy", "torch", "cv2", "pkg_resources"):
        present = _has_module(mod)
        print(f"[{'OK' if present else 'MISSING'}] Python module: {mod}")
        ok = ok and present
    scatter_present = _has_module("scatter_cuda")
    scatter_state = "OK" if scatter_present else ("MISSING" if require_scatter else "OPTIONAL")
    print(f"[{scatter_state}] Python module: scatter_cuda")
    if require_scatter:
        ok = ok and scatter_present
    ok = _check_path("DPT weight", paths.dpt_weight_path()) and ok
    ok = _check_path("LDF snapshot", paths.ldf_snapshot_path()) and ok
    ok = _check_path("LDF ResNet weight", paths.ldf_resnet_path()) and ok
    ok = _check_path("LaMa checkpoint", paths.lama_checkpoint_path()) and ok
    ok = _check_path("example bokeh image root", paths.bokeh_input_dir()) and ok
    return ok


def run_doctor(args: argparse.Namespace) -> int:
    check_all = not args.bokeh and not args.focus
    ok = True
    if check_all or args.focus:
        ok = check_focus() and ok
    if check_all or args.bokeh:
        if check_all:
            print()
        ok = check_bokeh() and ok
    return 0 if ok else 1


def _write_status(output_dir: Path, lines: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "demo_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(paths.REPO_ROOT))
    except ValueError:
        return str(path)


def run_demo_assets(args: argparse.Namespace) -> int:
    from brnfs.focus.runner import DependencyError, run_focus_stacking

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# README demo asset status", ""]

    want_focus = not args.bokeh_only
    want_bokeh = not args.focus_only

    if want_focus:
        dataset = paths.EXAMPLE_FOCUS_DIR / "IMG_2649"
        source_images = sorted(dataset.glob("*.png"))
        if source_images:
            sample_indices = sorted({0, len(source_images) // 2, len(source_images) - 1})
            copied_inputs = []
            for output_index, source_index in enumerate(sample_indices, start=1):
                source_image = source_images[source_index]
                output_name = f"focus_input_{output_index}.png"
                shutil.copyfile(source_image, output_dir / output_name)
                copied_inputs.append(f"`{output_name}` from `{_display_path(source_image)}`")
            lines.append("- focus inputs: copied " + ", ".join(copied_inputs) + ".")
        else:
            lines.append("- focus input: skipped; no sample image found.")

        try:
            run_focus_stacking(
                dataset_dir=dataset,
                output_path=output_dir / "focus_fused.png",
                levels=3,
                mask="soft",
                top="mean",
                sharpness="Tenengrad+Blur",
            )
            lines.append("- focus fused result: generated `focus_fused.png`.")
        except DependencyError as exc:
            lines.append(f"- focus fused result: skipped; {exc}.")
        except Exception as exc:  # noqa: BLE001 - asset generation should report and continue
            lines.append(f"- focus fused result: failed; {type(exc).__name__}: {exc}.")

    if want_bokeh:
        bokeh_input = paths.EXAMPLE_BOKEH_DIR / "IMG_2649.png"
        if bokeh_input.exists():
            shutil.copyfile(bokeh_input, output_dir / "bokeh_input.png")
            lines.append(f"- bokeh input: copied `{_display_path(bokeh_input)}`.")
        else:
            lines.append("- bokeh input: skipped; no sample image found.")

        bokeh_ok = check_bokeh(require_scatter=True)
        if bokeh_ok:
            bokeh_output = output_dir / "bokeh_rendered.png"
            bokeh_args = argparse.Namespace(
                rgb=bokeh_input,
                output=bokeh_output,
                k_blur=30.0,
                focal=0.10,
                lens=71,
                disp=None,
                alpha=None,
                gamma=2.2,
                verbose=False,
            )
            try:
                run_bokeh(bokeh_args)
                lines.append("- bokeh rendered result: generated `bokeh_rendered.png`.")
            except Exception as exc:  # noqa: BLE001
                lines.append(f"- bokeh rendered result: failed; {type(exc).__name__}: {exc}.")
        else:
            lines.append("- bokeh rendered result: skipped; bokeh demo prerequisites did not pass.")

    _write_status(output_dir, lines)
    print(output_dir / "demo_status.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    paths.ensure_runtime_dirs()
    return int(args.func(args))
