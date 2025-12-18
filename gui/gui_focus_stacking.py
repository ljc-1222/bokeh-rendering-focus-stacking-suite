"""Tkinter GUI for Laplacian-Pyramid-Based Focus Stacking (vendored into `bokeh_rendering_and_focus_stacking_suite/`).

Key behavior changes vs the standalone focus-stacking project:
- Datasets are loaded from `bokeh_rendering_and_focus_stacking_suite/Imgs/focus_stacking/<set_name>/`
- **No auto-saving**: generating a fused image only updates the preview; saving is manual
  (default directory: `bokeh_rendering_and_focus_stacking_suite/outputs/focus_stacking/`).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

from app.focus_stacking.preprocess import preprocess_image_stack
from app.focus_stacking.pyramids import build_pyramids_stack
from app.focus_stacking.sharpness import compute_sharpness_map
from app.focus_stacking.mask import build_masks, build_raw_masks
from app.focus_stacking.fusion import fuse_pyramids_and_reconstruct
from app.focus_stacking.evaluation import compute_q_abf


class FocusStackingGUI:
    """Focus stacking GUI that can be embedded into another Tkinter container."""

    def __init__(self, root: tk.Tk, *, parent: Optional[tk.Misc] = None) -> None:
        self.root = root
        self.parent: tk.Misc = parent or root

        project_root = Path(__file__).resolve().parents[1]  # bokeh_rendering_and_focus_stacking_suite/
        self.data_dir = project_root / "Imgs" / "focus_stacking"
        self.output_dir = project_root / "outputs" / "focus_stacking"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Animation state
        self.anim_frames: list[ImageTk.PhotoImage] = []
        self.anim_id: Optional[str] = None
        self.anim_idx: int = 0
        self.is_playing: bool = True

        # Current results (for manual saving)
        self.current_dataset: Optional[str] = None
        self.fused_u8_bgr: Optional[np.ndarray] = None  # HxWx3 uint8 BGR

        # Preview cache / async guards
        self._preview_dataset: Optional[str] = None
        self._preview_stack: Optional[np.ndarray] = None
        self._preview_req_id: int = 0
        self._fusion_req_id: int = 0

        # Debounced resize redraw for preview panels.
        self._resize_job_id: Optional[str] = None
        self._last_preview_size: Optional[tuple[int, int]] = None  # (w, h) used to build Tk images

        self._build_widgets()

    # -------------------------
    # UI construction
    # -------------------------

    def _build_widgets(self) -> None:
        # 1. Image Selection
        frame_select = ttk.LabelFrame(self.parent, text="1. Select Image Set")
        frame_select.pack(fill="x", padx=10, pady=5)

        self.folder_var = tk.StringVar()
        self.folder_combo = ttk.Combobox(frame_select, textvariable=self.folder_var, state="readonly")
        self.folder_combo.pack(fill="x", padx=10, pady=10)
        self._refresh_folders()
        # Switch preview immediately when the user selects another dataset.
        self.folder_combo.bind("<<ComboboxSelected>>", self._on_folder_selected)

        # 2. Fusion Settings
        frame_settings = ttk.LabelFrame(self.parent, text="2. Fusion Settings")
        frame_settings.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_settings, text="Mask Type:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.mask_var = tk.StringVar(value="Soft")
        frame_mask_opts = ttk.Frame(frame_settings)
        frame_mask_opts.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(frame_mask_opts, text="Normalized Soft", variable=self.mask_var, value="Soft").pack(
            side="left", padx=5
        )
        ttk.Radiobutton(frame_mask_opts, text="Hard", variable=self.mask_var, value="Hard").pack(side="left", padx=5)
        ttk.Radiobutton(frame_mask_opts, text="Hard (Argmin)", variable=self.mask_var, value="HardMin").pack(side="left", padx=5)

        ttk.Label(frame_settings, text="Top Layer Fusion:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.top_fusion_var = tk.StringVar(value="mean")
        frame_top_opts = ttk.Frame(frame_settings)
        frame_top_opts.grid(row=1, column=1, sticky="w")
        ttk.Radiobutton(frame_top_opts, text="Max", variable=self.top_fusion_var, value="max").pack(
            side="left", padx=5
        )
        ttk.Radiobutton(frame_top_opts, text="Mean", variable=self.top_fusion_var, value="mean").pack(
            side="left", padx=5
        )

        ttk.Label(frame_settings, text="Sharpness Def:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.sharpness_var = tk.StringVar(value="Tenengrad+Blur")
        self.sharpness_combo = ttk.Combobox(frame_settings, textvariable=self.sharpness_var, state="readonly", width=20)
        self.sharpness_combo["values"] = [
            "L", 
            "GaussianBlur(L)", 
            "GaussianBlur(L^2)", 
            "Tenengrad+Blur", 
            "Variance(L)", 
            "SML+Blur"
        ]
        self.sharpness_combo.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        # 3. Pyramid Levels
        frame_levels = ttk.LabelFrame(self.parent, text="3. Pyramid Levels")
        frame_levels.pack(fill="x", padx=10, pady=5)

        self.level_var = tk.IntVar(value=3)
        self.level_label = ttk.Label(frame_levels, text="Levels: 3")
        self.level_label.pack(pady=5)

        self.level_scale = ttk.Scale(
            frame_levels, from_=1, to=20, variable=self.level_var, orient="horizontal", command=self._update_level_label
        )
        self.level_scale.pack(fill="x", padx=10, pady=10)

        # 4. Actions & Progress
        frame_action = ttk.Frame(self.parent)
        frame_action.pack(fill="x", padx=10, pady=10)

        self.btn_generate = ttk.Button(frame_action, text="Generate Fused Image", command=self._start_generation)
        self.btn_generate.pack(fill="x", pady=5)

        self.btn_save = ttk.Button(frame_action, text="Save fused image", command=self._save_fused_image)
        self.btn_save.pack(fill="x", pady=5)
        self.btn_save.config(state="disabled")

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(frame_action, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=5)

        self.status_label = ttk.Label(frame_action, text="Ready")
        self.status_label.pack()

        # 5. Image Display Area
        self.display_frame = ttk.Frame(self.parent)
        self.display_frame.pack(expand=True, fill="both", padx=10, pady=10)

        self.display_frame.columnconfigure(0, weight=1)
        self.display_frame.columnconfigure(1, weight=1)
        self.display_frame.rowconfigure(1, weight=1)

        self.lbl_source_title = ttk.Label(self.display_frame, text="Source Images", font=("Arial", 12))
        self.lbl_source_title.grid(row=0, column=0, pady=5)

        self.lbl_result_title = ttk.Label(self.display_frame, text="Fused Result", font=("Arial", 12))
        self.lbl_result_title.grid(row=0, column=1, pady=5)

        # Panels for previews: we use `place()` to guarantee that content stays centered
        # (ttk.Label's `anchor` can be inconsistent across themes for image-only labels).
        self.source_panel = ttk.Frame(self.display_frame)
        self.source_panel.grid(row=1, column=0, sticky="nsew", padx=5)
        self.result_panel = ttk.Frame(self.display_frame)
        self.result_panel.grid(row=1, column=1, sticky="nsew", padx=5)

        self.anim_label = ttk.Label(self.source_panel, text="", anchor="center", justify="center")
        self.anim_label.place(relx=0.5, rely=0.5, anchor="center")

        self.result_label = ttk.Label(self.result_panel, text="", anchor="center", justify="center")
        self.result_label.place(relx=0.5, rely=0.5, anchor="center")

        # Keep placeholder texts readable as the window resizes.
        self.source_panel.bind("<Configure>", self._on_source_panel_resize)
        self.result_panel.bind("<Configure>", self._on_result_panel_resize)

        self.controls_frame = ttk.Frame(self.display_frame)
        # IMPORTANT: span both columns, otherwise this control row biases the grid
        # and makes the two preview panels not get equal width (causing "off-center"
        # previews and size changes after Generate).
        self.controls_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5, padx=5)

        self.btn_play = ttk.Button(self.controls_frame, text="Pause", command=self._toggle_play)
        self.btn_play.pack(side="left", padx=5)

        self.anim_slider_var = tk.IntVar()
        self.anim_slider = ttk.Scale(
            self.controls_frame, from_=0, to=0, variable=self.anim_slider_var, orient="horizontal", command=self._on_slider_change
        )
        self.anim_slider.pack(side="left", fill="x", expand=True, padx=5)

        # Show a preview for the default selection (if any).
        self.root.after(0, self._request_preview_load)

    # -------------------------
    # UI callbacks
    # -------------------------

    def _refresh_folders(self) -> None:
        if not self.data_dir.exists():
            self.folder_combo["values"] = []
            return

        folders = sorted([p.name for p in self.data_dir.iterdir() if p.is_dir()])
        self.folder_combo["values"] = folders
        if folders:
            self.folder_combo.current(0)

    def _update_level_label(self, value: str) -> None:
        self.level_label.config(text=f"Levels: {int(float(value))}")

    def _toggle_play(self) -> None:
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.config(text="Pause")
            if self.anim_id:
                self.root.after_cancel(self.anim_id)
                self.anim_id = None
            self._animate_loop()
        else:
            self.btn_play.config(text="Play")
            if self.anim_id:
                self.root.after_cancel(self.anim_id)
                self.anim_id = None

    def _on_slider_change(self, value: str) -> None:
        if not self.anim_frames:
            return
        idx = int(float(value))
        self.anim_idx = idx
        img = self.anim_frames[self.anim_idx]
        self.anim_label.config(image=img, text="")

    def _on_source_panel_resize(self, event: tk.Event) -> None:
        # `wraplength` wants pixels; keep some padding.
        width = max(1, int(getattr(event, "width", 1)) - 20)
        self.anim_label.config(wraplength=width)
        self._schedule_preview_redraw()

    def _on_result_panel_resize(self, event: tk.Event) -> None:
        width = max(1, int(getattr(event, "width", 1)) - 20)
        self.result_label.config(wraplength=width)
        self._schedule_preview_redraw()

    def _schedule_preview_redraw(self) -> None:
        """Debounce preview redraw during window resizing."""
        if self._resize_job_id is not None:
            try:
                self.root.after_cancel(self._resize_job_id)
            except Exception:
                pass
        self._resize_job_id = self.root.after(80, self._redraw_previews)

    def _redraw_previews(self) -> None:
        """Rebuild preview images to fit the current panel sizes."""
        self._resize_job_id = None
        source_images = self._preview_stack
        if source_images is None or source_images.size == 0:
            return

        # Determine the available space inside panels (minus padding).
        default_w, default_h = 425, 450

        src_pw = int(self.source_panel.winfo_width() or 0)
        src_ph = int(self.source_panel.winfo_height() or 0)
        src_w = max(1, (src_pw if src_pw > 1 else default_w) - 20)
        src_h = max(1, (src_ph if src_ph > 1 else default_h) - 20)

        fused_u8_bgr = self.fused_u8_bgr
        if fused_u8_bgr is not None:
            res_pw = int(self.result_panel.winfo_width() or 0)
            res_ph = int(self.result_panel.winfo_height() or 0)
            res_w = max(1, (res_pw if res_pw > 1 else default_w) - 20)
            res_h = max(1, (res_ph if res_ph > 1 else default_h) - 20)
            max_w = min(src_w, res_w)
            max_h = min(src_h, res_h)
        else:
            max_w, max_h = src_w, src_h

        # Fit by aspect ratio based on the first source frame.
        frame0 = source_images[0].astype(np.uint8)
        h, w = frame0.shape[:2]
        scale = min(float(max_h) / float(max(h, 1)), float(max_w) / float(max(w, 1)))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # If size didn't change, no need to rebuild the (potentially large) PhotoImage lists.
        if self._last_preview_size != (new_w, new_h):
            self._last_preview_size = (new_w, new_h)

            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

            # Preserve current selection where possible.
            keep_idx = int(self.anim_slider_var.get() if hasattr(self, "anim_slider_var") else self.anim_idx)

            self.anim_frames = []
            for i in range(source_images.shape[0]):
                frame_bgr = source_images[i].astype(np.uint8)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=interp)
                self.anim_frames.append(ImageTk.PhotoImage(Image.fromarray(frame_resized)))

            self.anim_slider.config(to=max(0, len(self.anim_frames) - 1))
            if self.anim_frames:
                idx = max(0, min(keep_idx, len(self.anim_frames) - 1))
                self.anim_idx = idx
                self.anim_slider_var.set(idx)
                img = self.anim_frames[idx]
                self.anim_label.config(image=img, text="")

        # Update the fused result image to match the same preview size.
        if fused_u8_bgr is not None:
            rgb = cv2.cvtColor(fused_u8_bgr, cv2.COLOR_BGR2RGB)
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            rgb_resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)
            img_tk = ImageTk.PhotoImage(Image.fromarray(rgb_resized))
            self.result_label.config(image=img_tk, text="")
            self.result_label.image = img_tk  # type: ignore[attr-defined]

    def _on_folder_selected(self, _event: object) -> None:
        # Dataset changed: update left preview immediately and clear right result.
        self._request_preview_load()

    def _request_preview_load(self) -> None:
        folder_name = self.folder_var.get()
        if not folder_name:
            return

        # Invalidate any in-flight fusion/preview updates from older selections.
        self._preview_req_id += 1
        self._fusion_req_id += 1
        preview_req_id = self._preview_req_id

        self._stop_animation()
        self._clear_result_preview()

        self.fused_u8_bgr = None
        self.current_dataset = folder_name
        self.btn_save.config(state="disabled")
        self.btn_generate.config(state="normal")

        self.progress_var.set(0)
        self.status_label.config(text="Loading preview...")

        thread = threading.Thread(target=self._load_preview_stack, args=(folder_name, preview_req_id), daemon=True)
        thread.start()

    def _load_preview_stack(self, folder_name: str, preview_req_id: int) -> None:
        try:
            data_path = self.data_dir / folder_name

            # Reuse last preview if it matches the dataset (fast path).
            if self._preview_dataset == folder_name and self._preview_stack is not None:
                images = self._preview_stack
            else:
                images = preprocess_image_stack(str(data_path))

            # If the user switched datasets while we were loading, drop the update.
            if preview_req_id != self._preview_req_id:
                return

            self._preview_dataset = folder_name
            self._preview_stack = images

            self.root.after(0, lambda: self._show_source_preview(source_images=images))
            self.root.after(0, lambda: self.status_label.config(text="Preview ready (click Generate to fuse)"))
        except Exception as exc:
            if preview_req_id != self._preview_req_id:
                return
            self.root.after(0, lambda: messagebox.showerror("Preview load failed", str(exc)))
            self.root.after(0, lambda: self.status_label.config(text="Preview load failed"))

    def _start_generation(self) -> None:
        folder_name = self.folder_var.get()
        if not folder_name:
            messagebox.showerror("Error", "Please select an image set.")
            return

        # Only pause animation (don't clear frames) if we're generating the same dataset
        # This preserves the source images when regenerating
        if self.current_dataset == folder_name and self._preview_stack is not None:
            self._pause_animation()
        else:
            self._stop_animation()
        self.btn_generate.config(state="disabled")
        self.btn_save.config(state="disabled")
        self.fused_u8_bgr = None
        self.current_dataset = folder_name
        self._fusion_req_id += 1
        fusion_req_id = self._fusion_req_id

        self.progress_var.set(0)
        self.status_label.config(text="Starting...")

        thread = threading.Thread(target=self._run_fusion_pipeline, args=(folder_name, fusion_req_id), daemon=True)
        thread.start()

    # -------------------------
    # Pipeline
    # -------------------------

    def _run_fusion_pipeline(self, folder_name: str, fusion_req_id: int) -> None:
        try:
            levels = int(self.level_var.get())
            mask_type = self.mask_var.get()
            top_method = self.top_fusion_var.get()
            sharpness_def = self.sharpness_var.get()

            data_path = self.data_dir / folder_name

            self._update_status_if_current(fusion_req_id, "Preprocessing images...", 10)
            if self._preview_dataset == folder_name and self._preview_stack is not None:
                images = self._preview_stack
            else:
                images = preprocess_image_stack(str(data_path))

            self._update_status_if_current(fusion_req_id, "Building pyramids...", 30)
            _gaussian_pyrs, laplacian_pyrs, top_gaussians = build_pyramids_stack(images, levels)

            self._update_status_if_current(fusion_req_id, "Computing sharpness maps...", 50)
            sharpness_maps = compute_sharpness_map(laplacian_pyrs, definition=sharpness_def)

            self._update_status_if_current(fusion_req_id, f"Building {mask_type} masks...", 70)
            if mask_type == "Soft":
                masks = build_masks(sharpness_maps, sigma=1.2, ksize=7)
            elif mask_type == "HardMin":
                masks = build_raw_masks(sharpness_maps, mode="min")
            else:
                masks = build_raw_masks(sharpness_maps, mode="max")

            self._update_status_if_current(fusion_req_id, f"Fusing images (Top: {top_method})...", 90)
            fused_image = fuse_pyramids_and_reconstruct(
                laplacian_pyrs,
                top_gaussians,
                masks,
                top_fusion_method=top_method,
                output_dir=None,  # IMPORTANT: no auto-saving of debug images
            )

            fused_u8 = np.clip(fused_image, 0.0, 255.0).astype(np.uint8)  # BGR
            # If the user switched datasets while we were fusing, drop the update.
            if fusion_req_id != self._fusion_req_id or self.folder_var.get() != folder_name:
                return

            self.fused_u8_bgr = fused_u8

            # Compute Q_AB/F score
            self._update_status_if_current(fusion_req_id, "Computing Q_AB/F score...", 95)
            # Use finest level sharpness maps (index 0)
            finest_sharpness = [sm[0] for sm in sharpness_maps]
            # Convert images to list for the metric function
            source_images_list = [images[i] for i in range(images.shape[0])]
            score = compute_q_abf(fused_u8, source_images_list, finest_sharpness)

            # Avoid fancy punctuation that can look odd depending on font/theme.
            self._update_status_if_current(fusion_req_id, f"Done! Q_AB/F: {score:.4f} (Preview updated)", 100)
            self.root.after(0, lambda: self._show_result(fused_u8_bgr=fused_u8, source_images=images))
        except Exception as exc:
            if fusion_req_id != self._fusion_req_id:
                return
            self.root.after(0, lambda: messagebox.showerror("Error", str(exc)))
            self._update_status_if_current(fusion_req_id, "Error occurred", 0)
        finally:
            self.root.after(0, lambda: self.btn_generate.config(state="normal"))
            if fusion_req_id == self._fusion_req_id and self.fused_u8_bgr is not None:
                self.root.after(0, lambda: self.btn_save.config(state="normal"))
            else:
                self.root.after(0, lambda: self.btn_save.config(state="disabled"))

    # -------------------------
    # Rendering / saving
    # -------------------------

    def _update_status(self, text: str, progress: float) -> None:
        self.root.after(0, lambda: self.status_label.config(text=text))
        self.root.after(0, lambda: self.progress_var.set(progress))

    def _update_status_if_current(self, fusion_req_id: int, text: str, progress: float) -> None:
        if fusion_req_id != self._fusion_req_id:
            return
        self._update_status(text, progress)

    def _clear_result_preview(self) -> None:
        # Clear right-side result preview (and drop old Tk image references).
        self.result_label.config(image="", text="(Result will appear after you click Generate)", justify="center")
        self.result_label.image = None  # type: ignore[attr-defined]

    def _show_source_preview(self, *, source_images: np.ndarray) -> None:
        """Render just the left-side source preview, keeping the right side blank."""
        if source_images.size == 0:
            self.anim_label.config(image="", text="No images found")
            return

        # Persist raw preview stack so we can rebuild frames on window resize.
        self._preview_stack = source_images
        self._last_preview_size = None  # force rebuild at current panel size
        # Defer until Tk geometry is updated; otherwise winfo_width/height may be 1.
        # IMPORTANT: also start animation *after* frames are built, otherwise the loop
        # returns early and autoplay never starts.
        def redraw_then_play() -> None:
            self._redraw_previews()
            if self.anim_id is None and self.anim_frames:
                self.is_playing = True
                self.btn_play.config(text="Pause")
                self._start_animation()

        self.root.after(0, redraw_then_play)

    def _show_result(self, *, fused_u8_bgr: np.ndarray, source_images: np.ndarray) -> None:
        # Persist raw arrays so we can rebuild previews on resize.
        self.fused_u8_bgr = fused_u8_bgr
        self._preview_stack = source_images
        # Keep the same preview size as the current source preview so clicking Generate
        # doesn't cause the image size to jump.
        # (We still allow resizing via window resize events.)
        def redraw_then_play() -> None:
            self._redraw_previews()
            if self.anim_id is None and self.anim_frames:
                self.is_playing = True
                self.btn_play.config(text="Pause")
                # Resume from current position instead of resetting to beginning
                self._resume_animation()

        self.root.after(0, redraw_then_play)

    def _save_fused_image(self) -> None:
        if self.fused_u8_bgr is None or self.current_dataset is None:
            messagebox.showerror("Error", "Nothing to save yet. Generate a fused image first.")
            return

        levels = int(self.level_var.get())
        mask_type = self.mask_var.get()
        top_method = self.top_fusion_var.get()

        default_name = f"{self.current_dataset}_{mask_type}_{top_method}_L{levels}_fused.png"
        out_path = filedialog.asksaveasfilename(
            initialdir=str(self.output_dir),
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            title="Save fused image",
        )
        if not out_path:
            return

        try:
            cv2.imwrite(out_path, self.fused_u8_bgr)
            self.status_label.config(text=f"Saved: {Path(out_path).name}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    # -------------------------
    # Animation helpers
    # -------------------------

    def _start_animation(self) -> None:
        """Start animation from the beginning (frame 0)."""
        self.anim_idx = 0
        self._animate_loop()

    def _resume_animation(self) -> None:
        """Resume animation from the current frame position."""
        if not self.anim_frames:
            return
        # Ensure anim_idx is within valid range
        self.anim_idx = max(0, min(self.anim_idx, len(self.anim_frames) - 1))
        self._animate_loop()

    def _animate_loop(self) -> None:
        if not self.anim_frames:
            return
        if self.is_playing:
            img = self.anim_frames[self.anim_idx]
            self.anim_label.config(image=img, text="")
            self.anim_slider_var.set(self.anim_idx)
            self.anim_idx = (self.anim_idx + 1) % len(self.anim_frames)
            self.anim_id = self.root.after(500, self._animate_loop)

    def _pause_animation(self) -> None:
        """Pause the animation loop without clearing frames or labels."""
        if self.anim_id:
            self.root.after_cancel(self.anim_id)
            self.anim_id = None

    def _stop_animation(self) -> None:
        """Stop animation and clear all frames and labels."""
        if self.anim_id:
            self.root.after_cancel(self.anim_id)
            self.anim_id = None
        self.anim_frames = []
        self.anim_label.config(image="", text="")
        self.result_label.config(image="", text="(Result will appear after you click Generate)", justify="center")
        self.anim_label.image = None  # type: ignore[attr-defined]
        self.result_label.image = None  # type: ignore[attr-defined]


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Focus Stacking GUI")
    root.geometry("900x1000")
    FocusStackingGUI(root)
    root.mainloop()


