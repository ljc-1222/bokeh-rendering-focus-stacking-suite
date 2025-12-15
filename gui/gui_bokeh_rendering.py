"""Tkinter GUI for Dr.Bokeh-style single-image bokeh rendering.

This UI is a thin wrapper around `app.bokeh_rendering.gui_engine.BokehEngine`.
It intentionally:

- **Loads models lazily** (on first preprocess/render) to keep startup responsive.
- **Caches preprocessing** (depth/alpha/layers) to speed up repeated renders.
- **Avoids automatic disk writes**: rendering updates the preview; saving is explicit.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk


@dataclass
class _UIState:
    pre: Optional[Any] = None
    rgb_path: Optional[Path] = None
    last_render_key: Optional[str] = None
    busy: bool = False
    last_render_u8: Optional[np.ndarray] = None  # HxWx3 uint8 RGB
    # `source_base_u8` is the unmodified source image.
    # `last_source_u8` is whatever is currently displayed (may include overlays).
    source_base_u8: Optional[np.ndarray] = None  # HxWx3 uint8 RGB
    last_source_u8: Optional[np.ndarray] = None  # HxWx3 uint8 RGB
    resize_job_id: Optional[str] = None
    focus_overlay_job_id: Optional[str] = None
    focus_overlay_token: int = 0
    focus_overlay_enabled: bool = False
    # Downscaled preview buffers for fast, real-time overlay.
    overlay_src_u8: Optional[np.ndarray] = None  # h x w x 3 uint8 RGB
    overlay_disp_f32: Optional[np.ndarray] = None  # h x w float32 in [0, 1]


class DrBokehGUI:
    def __init__(self, root: tk.Tk, *, parent: Optional[tk.Misc] = None) -> None:
        """Create the bokeh rendering UI.

        Args:
            root: The Tk root used for `.after()` scheduling and dialogs.
            parent: Optional container widget. If omitted, the UI is attached to `root`.
        """
        self.root = root
        self.parent: tk.Misc = parent or root

        # Only configure the top-level window when we're attached directly to it.
        if parent is None:
            self.root.title("Bokeh Rendering GUI")
            # Base/default window size for scaling calculations
            self.base_width = 820
            self.base_height = 760
            self.root.geometry(f"{self.base_width}x{self.base_height}")
        else:
            # Fallback sizing constants (used for preview scaling computations).
            self.base_width = 820
            self.base_height = 760

        project_root = Path(__file__).resolve().parents[1]  # bokeh_rendering_and_focus_stacking_suite/
        self.img_dir = project_root / "Imgs" / "bokeh_rendering"
        self.out_dir = project_root / "outputs" / "bokeh_rendering"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.state = _UIState()

        # Lens kernel size control (odd int). Use DoubleVar because ttk.Scale produces floats.
        self.lens_var = tk.DoubleVar(value=71.0)

        # Optional preprocessing controls.
        # Bilateral-median mask filtering is intentionally OFF by default because it's slow.
        self.mask_filter_var = tk.BooleanVar(value=False)

        # Engine is initialized lazily when the user clicks Preprocess/Render.
        self.engine: Optional[Any] = None

        # Base preview panel size (each side gets half).
        self.base_preview_h = 460
        self.base_preview_w_total = 780
        self.base_preview_w_half = self.base_preview_w_total // 2

        self._build_widgets()
        # Delay filesystem scanning until after the window is displayed.
        self.root.after(0, self._refresh_images)

        # A single persistent worker thread for focus overlay updates. This avoids
        # spawning a new thread on every slider motion and ensures we always render
        # only the *latest* slider values.
        self._overlay_event = threading.Event()
        self._overlay_lock = threading.Lock()
        self._overlay_request: dict[str, float | int] = {"token": 0, "focal": 0.0, "k_blur": 0.0}
        self._overlay_thread: Optional[threading.Thread] = None

    def _ensure_engine(self) -> bool:
        """Lazy-init the rendering engine only when needed."""
        if self.engine is not None:
            return True
        try:
            # Import lazily: importing the engine pulls in heavy ML models.
            from app.bokeh_rendering.gui_engine import BokehEngine  # type: ignore

            # default matches README demo
            self.engine = BokehEngine(lens=self._current_lens())
            return True
        except Exception as exc:
            self.engine = None
            messagebox.showerror("Engine init failed", str(exc))
            return False

    @staticmethod
    def _normalize_lens(value: float) -> int:
        """Normalize lens value from UI to a valid odd kernel size."""
        raw = int(round(float(value)))
        lens = raw if (raw % 2 == 1) else raw + 1
        # Keep a safe/reasonable range for the demo UI.
        return max(7, min(151, lens))

    def _current_lens(self) -> int:
        """Current normalized lens kernel size."""
        return self._normalize_lens(float(self.lens_var.get()))


    def _build_widgets(self) -> None:
        # Configure ttk style for larger default fonts
        style = ttk.Style()
        style.configure("TLabel", font=("TkDefaultFont", 14))
        style.configure("TButton", font=("TkDefaultFont", 14))
        style.configure("TCheckbutton", font=("TkDefaultFont", 14))

        frame_select = ttk.LabelFrame(self.parent, text="1. Select Image")
        frame_select.pack(fill="x", padx=10, pady=8)

        self.image_var = tk.StringVar()
        self.image_combo = ttk.Combobox(frame_select, textvariable=self.image_var, state="readonly")
        self.image_combo.pack(fill="x", padx=10, pady=8)
        self.image_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_image_selected())

        frame_settings = ttk.LabelFrame(self.parent, text="2. Controls (Refocus + Intensity)")
        frame_settings.pack(fill="x", padx=10, pady=8)

        self.k_var = tk.DoubleVar(value=30.0)
        ttk.Label(frame_settings, text="Intensity (K):").grid(row=0, column=0, sticky="w", padx=10, pady=6)
        self.k_scale = ttk.Scale(frame_settings, from_=0.0, to=60.0, orient="horizontal", variable=self.k_var)
        self.k_scale.grid(row=0, column=1, sticky="ew", padx=10, pady=6)
        self.k_value_label = ttk.Label(frame_settings, text="30.0")
        self.k_value_label.grid(row=0, column=2, sticky="w", padx=10, pady=6)

        self.focal_var = tk.DoubleVar(value=0.10)
        ttk.Label(frame_settings, text="Focal plane:").grid(row=1, column=0, sticky="w", padx=10, pady=6)
        self.focal_scale = ttk.Scale(
            frame_settings, from_=0.0, to=1.0, orient="horizontal", variable=self.focal_var
        )
        self.focal_scale.grid(row=1, column=1, sticky="ew", padx=10, pady=6)
        self.focal_value_label = ttk.Label(frame_settings, text="0.10")
        self.focal_value_label.grid(row=1, column=2, sticky="w", padx=10, pady=6)

        ttk.Label(frame_settings, text="Lens (kernel):").grid(row=2, column=0, sticky="w", padx=10, pady=6)
        self.lens_scale = ttk.Scale(frame_settings, from_=7, to=151, orient="horizontal", variable=self.lens_var)
        self.lens_scale.grid(row=2, column=1, sticky="ew", padx=10, pady=6)
        self.lens_value_label = ttk.Label(frame_settings, text=str(self._current_lens()))
        self.lens_value_label.grid(row=2, column=2, sticky="w", padx=10, pady=6)

        # Preprocessing option: mask denoising (slow).
        self.mask_filter_check = ttk.Checkbutton(
            frame_settings,
            text="Enable mask bilateral-median filter (slow)",
            variable=self.mask_filter_var,
            command=self._on_mask_filter_toggled,
        )
        self.mask_filter_check.grid(row=3, column=0, columnspan=3, sticky="w", padx=10, pady=6)

        frame_settings.columnconfigure(1, weight=1)

        # Slider updates only update labels; rendering is triggered explicitly by the button.
        self.k_scale.bind("<B1-Motion>", lambda _e: self._on_slider_change())
        self.k_scale.bind("<ButtonRelease-1>", lambda _e: self._on_slider_change())
        self.focal_scale.bind("<B1-Motion>", lambda _e: self._on_slider_change())
        self.focal_scale.bind("<ButtonRelease-1>", lambda _e: self._on_slider_change())
        self.lens_scale.bind("<B1-Motion>", lambda _e: self._on_lens_change())
        self.lens_scale.bind("<ButtonRelease-1>", lambda _e: self._on_lens_change())

        frame_action = ttk.Frame(self.parent)
        frame_action.pack(fill="x", padx=10, pady=8)

        self.btn_preprocess = ttk.Button(frame_action, text="Preprocess (Depth + Alpha + Layers)", command=self._start_preprocess)
        self.btn_preprocess.pack(fill="x", pady=4)

        # Focus overlay toggle (preview-only): OFF by default.
        self.btn_toggle_overlay = ttk.Button(
            frame_action,
            text="Enable focus overlay (preview)",
            command=self._toggle_focus_overlay,
        )
        self.btn_toggle_overlay.pack(fill="x", pady=4)

        self.btn_render = ttk.Button(frame_action, text="Render (Current Params)", command=self._start_render)
        self.btn_render.pack(fill="x", pady=4)

        self.btn_save = ttk.Button(frame_action, text="Save rendered image", command=self._save_rendered_image)
        self.btn_save.pack(fill="x", pady=4)

        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(frame_action, variable=self.progress_var, maximum=100)
        self.progress.pack(fill="x", pady=6)

        self.status_label = ttk.Label(frame_action, text="Ready")
        self.status_label.pack()

        # Preview panel - mirror focus stacking layout so tabs align visually:
        # title row + two equal-width panels (grid) with consistent padding.
        self.display = ttk.Frame(self.parent)
        self.display.pack(expand=True, fill="both", padx=10, pady=10)

        self.display.columnconfigure(0, weight=1)
        self.display.columnconfigure(1, weight=1)
        self.display.rowconfigure(1, weight=1)
        # Reserve a bottom row like focus stacking's play/slider controls so the
        # preview panels have the same effective height across tabs.
        self.display.rowconfigure(2, minsize=44)

        lbl_source_title = ttk.Label(self.display, text="Source Image", font=("Arial", 12))
        lbl_source_title.grid(row=0, column=0, pady=5)
        lbl_result_title = ttk.Label(self.display, text="Rendered Result", font=("Arial", 12))
        lbl_result_title.grid(row=0, column=1, pady=5)

        self._src_panel = ttk.Frame(self.display)
        self._src_panel.grid(row=1, column=0, sticky="nsew", padx=5)
        self._res_panel = ttk.Frame(self.display)
        self._res_panel.grid(row=1, column=1, sticky="nsew", padx=5)

        self.src_canvas = tk.Canvas(self._src_panel, highlightthickness=1)
        self.src_canvas.pack(fill="both", expand=True)
        self.res_canvas = tk.Canvas(self._res_panel, highlightthickness=1)
        self.res_canvas.pack(fill="both", expand=True)

        # Spacer row to match focus stacking's bottom controls area (visual alignment).
        self._controls_spacer = ttk.Frame(self.display, height=44)
        self._controls_spacer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5, padx=5)
        self._controls_spacer.grid_propagate(False)

        # Keep strong refs to PhotoImage to avoid garbage collection.
        self._src_imgtk: Optional[ImageTk.PhotoImage] = None
        self._res_imgtk: Optional[ImageTk.PhotoImage] = None

        self._clear_source_preview()
        self._clear_result_preview()
        # Redraw (center) images when the user resizes the window/panels.
        self.src_canvas.bind("<Configure>", lambda _e: self._schedule_redraw())
        self.res_canvas.bind("<Configure>", lambda _e: self._schedule_redraw())

    def _refresh_images(self) -> None:
        if not self.img_dir.exists():
            self.image_combo["values"] = []
            return
        images = sorted([p.name for p in self.img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"} and "alpha" not in p.name])
        self.image_combo["values"] = images
        if images:
            # Do not auto-load or preprocess anything before the user interacts.
            self.image_combo.current(0)
            # Show a lightweight source preview (no model/preprocess).
            self._on_image_selected()
        else:
            self._clear_source_preview()
            self._clear_result_preview()
            self._set_status("No images found in Imgs/.", 0)

    def _on_image_selected(self) -> None:
        if self.state.busy:
            return
        name = self.image_var.get()
        if not name:
            return
        self.state.pre = None
        self.state.rgb_path = (self.img_dir / name).resolve()
        self.state.last_render_key = None
        self.state.last_render_u8 = None
        self.state.source_base_u8 = None
        self.state.last_source_u8 = None
        self.state.overlay_src_u8 = None
        self.state.overlay_disp_f32 = None
        self._cancel_focus_overlay()
        # Keep overlay disabled unless the user explicitly enables it again.
        self._set_focus_overlay_enabled(False)
        self._show_source(self.state.rgb_path)
        self._clear_result_preview()
        self._set_status("Selected image. Click Preprocess.", 0)

    def _show_source(self, rgb_path: Path) -> None:
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            self._clear_source_preview()
            return
        rgb_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.state.source_base_u8 = rgb_u8
        self.state.last_source_u8 = rgb_u8
        self._set_image_on_canvas(self.src_canvas, rgb_u8, which="src")


    def _clear_source_preview(self) -> None:
        self.src_canvas.delete("all")
        # Leave the canvas empty. The section title above the panel already labels it,
        # and drawing placeholder text here can get clipped during initial layout.
        self._src_imgtk = None
        self.state.source_base_u8 = None
        self.state.last_source_u8 = None
        self.state.overlay_src_u8 = None
        self.state.overlay_disp_f32 = None

    def _clear_result_preview(self) -> None:
        self.res_canvas.delete("all")
        # Leave the canvas empty (see `_clear_source_preview`).
        self._res_imgtk = None
        self.state.last_render_u8 = None
        self.progress_var.set(0)

    def _set_image_on_canvas(self, canvas: tk.Canvas, rgb: np.ndarray, *, which: str) -> None:
        # Use *actual* size after layout; fall back to configured size if not ready yet.
        target_w = int(canvas.winfo_width() or canvas.cget("width"))
        target_h = int(canvas.winfo_height() or canvas.cget("height"))
        h, w = rgb.shape[:2]
        scale = min(target_h / max(h, 1), target_w / max(w, 1))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_tk = ImageTk.PhotoImage(Image.fromarray(resized))

        canvas.delete("all")
        x = (target_w - new_w) // 2
        y = (target_h - new_h) // 2
        canvas.create_image(x, y, image=img_tk, anchor="nw")

        if which == "src":
            self._src_imgtk = img_tk
        else:
            self._res_imgtk = img_tk

    def _schedule_redraw(self) -> None:
        """Debounce redraw requests during window resizing."""
        if self.state.resize_job_id is not None:
            try:
                self.root.after_cancel(self.state.resize_job_id)
            except Exception:
                pass
        self.state.resize_job_id = self.root.after(80, self._redraw_previews)

    def _redraw_previews(self) -> None:
        """Re-center the currently displayed source/result images."""
        if self.state.last_source_u8 is not None:
            self._set_image_on_canvas(self.src_canvas, self.state.last_source_u8, which="src")
        if self.state.last_render_u8 is not None:
            self._set_image_on_canvas(self.res_canvas, self.state.last_render_u8, which="res")

    def _set_status(self, text: str, progress: float) -> None:
        self.root.after(0, lambda: self.status_label.config(text=text))
        self.root.after(0, lambda: self.progress_var.set(progress))

    def _set_busy(self, busy: bool) -> None:
        self.state.busy = busy
        state = "disabled" if busy else "normal"
        readonly_or_disabled = "disabled" if busy else "readonly"

        # Disable everything that could cause conflicting operations.
        self.image_combo.config(state=readonly_or_disabled)
        self.k_scale.config(state=state)
        self.focal_scale.config(state=state)
        self.lens_scale.config(state=state)
        self.btn_preprocess.config(state=state)
        self.btn_toggle_overlay.config(state=state)
        self.btn_render.config(state=state)
        self.btn_save.config(state=state)

    def _on_slider_change(self) -> None:
        self.k_value_label.config(text=f"{self.k_var.get():.1f}")
        self.focal_value_label.config(text=f"{self.focal_var.get():.2f}")
        # Real-time focus visualization (no full render): when enabled and after preprocess,
        # update the source preview with focus highlighting.
        self._schedule_focus_overlay()

    def _on_lens_change(self) -> None:
        """Update lens kernel size (odd integer) and propagate to engine if present."""
        lens = self._current_lens()
        self.lens_var.set(float(lens))
        self.lens_value_label.config(text=str(lens))

        # Lens affects rendering backend; force key change so re-renders happen.
        self.state.last_render_key = None

        if self.engine is not None:
            try:
                self.engine.set_lens(int(lens))  # type: ignore[attr-defined]
            except Exception as exc:
                messagebox.showerror("Invalid lens", str(exc))

    def _set_focus_overlay_enabled(self, enabled: bool) -> None:
        """Update overlay state and UI label."""
        self.state.focus_overlay_enabled = enabled
        self.btn_toggle_overlay.config(
            text="Disable focus overlay (preview)" if enabled else "Enable focus overlay (preview)"
        )

    def _toggle_focus_overlay(self) -> None:
        """Toggle real-time focus overlay on the source preview."""
        if self.state.busy:
            return
        enabled = not self.state.focus_overlay_enabled
        self._set_focus_overlay_enabled(enabled)

        if not enabled:
            # Cancel pending work and restore plain source.
            self._cancel_focus_overlay()
            self.state.focus_overlay_token += 1  # invalidate in-flight worker results
            if self.state.source_base_u8 is not None:
                self.state.last_source_u8 = self.state.source_base_u8
                self._set_image_on_canvas(self.src_canvas, self.state.source_base_u8, which="src")
            return

        # Enabled: compute overlay immediately if we can.
        self._schedule_focus_overlay()

    def _cancel_focus_overlay(self) -> None:
        """Cancel any pending focus-overlay job."""
        if self.state.focus_overlay_job_id is not None:
            try:
                self.root.after_cancel(self.state.focus_overlay_job_id)
            except Exception:
                pass
            self.state.focus_overlay_job_id = None

    def _schedule_focus_overlay(self) -> None:
        """Debounce focus-overlay recomputation during slider motion."""
        if self.state.busy:
            return
        if not self.state.focus_overlay_enabled:
            return
        if self.state.pre is None or self.state.source_base_u8 is None:
            return
        if self.state.overlay_src_u8 is None or self.state.overlay_disp_f32 is None:
            # We can still compute on the full source, but it will be slower. Prefer
            # generating the preview buffers right after preprocess.
            return
        self._cancel_focus_overlay()
        # Keep it responsive without recomputing for every mousemove.
        self.state.focus_overlay_job_id = self.root.after(10, self._request_focus_overlay)

    def _ensure_overlay_worker(self) -> None:
        """Start the persistent overlay worker thread if it isn't running yet."""
        if self._overlay_thread is not None and self._overlay_thread.is_alive():
            return

        def loop() -> None:
            while True:
                self._overlay_event.wait()
                self._overlay_event.clear()

                try:
                    with self._overlay_lock:
                        token = int(self._overlay_request["token"])
                        focal = float(self._overlay_request["focal"])
                        k_blur = float(self._overlay_request["k_blur"])

                    # Grab downscaled buffers for speed.
                    src_small = self.state.overlay_src_u8
                    disp_small = self.state.overlay_disp_f32
                    if src_small is None or disp_small is None:
                        continue

                    # If overlay toggled off or token advanced, skip work.
                    if not self.state.focus_overlay_enabled:
                        continue
                    if token != self.state.focus_overlay_token:
                        continue

                    # Compute in-focus mask from CoC on the *small* preview.
                    if k_blur <= 1e-6:
                        out_u8 = src_small
                    else:
                        coc = np.abs(disp_small - focal) * k_blur
                        focus_mask = (coc <= 1.0)
                        mask_u8 = (focus_mask.astype(np.uint8) * 255)

                        # Light denoise (cheap at preview size).
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
                        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

                        # Soft focus mask for dimming.
                        soft_mask = cv2.GaussianBlur(mask_u8, (0, 0), sigmaX=1.0, sigmaY=1.0)
                        soft = (soft_mask.astype(np.float32) / 255.0)[..., None]  # h x w x 1

                        # Darken out-of-focus region to enhance focus.
                        dim_factor = 0.35  # 0=black OOF, 1=no dimming
                        factor = dim_factor + (1.0 - dim_factor) * soft
                        out_u8 = np.clip(src_small.astype(np.float32) * factor, 0.0, 255.0).astype(np.uint8)

                        # Fast contour/edge overlay: edges of the mask.
                        edges = cv2.Canny(mask_u8, 50, 150)
                        edges = cv2.dilate(edges, kernel, iterations=1)
                        out_u8[edges > 0] = (255, 0, 0)  # red in RGB

                    def apply() -> None:
                        if token != self.state.focus_overlay_token:
                            return
                        if not self.state.focus_overlay_enabled:
                            return
                        self.state.last_source_u8 = out_u8
                        self._set_image_on_canvas(self.src_canvas, out_u8, which="src")

                    self.root.after(0, apply)
                except Exception:
                    # Best-effort; never crash GUI.
                    continue

        self._overlay_thread = threading.Thread(target=loop, daemon=True)
        self._overlay_thread.start()

    def _request_focus_overlay(self) -> None:
        """Request a focus-overlay update (processed by the persistent worker)."""
        if self.state.busy:
            return
        if not self.state.focus_overlay_enabled:
            return
        if self.state.pre is None:
            return
        if self.state.overlay_src_u8 is None or self.state.overlay_disp_f32 is None:
            return

        self._ensure_overlay_worker()

        # Token allows us to drop stale overlay results if the user keeps moving sliders.
        self.state.focus_overlay_token += 1
        token = int(self.state.focus_overlay_token)
        with self._overlay_lock:
            self._overlay_request["token"] = token
            self._overlay_request["focal"] = float(self.focal_var.get())
            self._overlay_request["k_blur"] = float(self.k_var.get())
        self._overlay_event.set()

    def _on_mask_filter_toggled(self) -> None:
        """Invalidate preprocessing when the mask-filter toggle changes.

        This does NOT auto-run preprocessing or rendering; the user must click the
        corresponding buttons. We only ensure we never render with stale layers.
        """
        if self.state.busy:
            return

        # Invalidate cached preprocessing + render memoization.
        self.state.pre = None
        self.state.last_render_key = None
        self.state.last_render_u8 = None
        self._set_status("Mask filter changed. Please click 'Preprocess' again.", 0)

    def _start_preprocess(self) -> None:
        if not self._ensure_engine():
            return
        if self.state.rgb_path is None:
            messagebox.showerror("Error", "Please select an image.")
            return
        if self.state.busy:
            return
        self._set_busy(True)
        self._set_status("Preprocessing... (DPT + LDF + LaMa)", 10)

        def worker() -> None:
            try:
                t0 = time.time()
                pre = self.engine.preprocess(
                    self.state.rgb_path,
                    mask_filter=bool(self.mask_filter_var.get()),
                )
                dt = time.time() - t0
                self.state.pre = pre
                self._set_status(f"Preprocess done in {dt:.1f}s. Ready to render.", 100)
                # Build downscaled preview buffers for fast, real-time overlay.
                def build_preview_buffers() -> None:
                    if self.state.source_base_u8 is None:
                        return
                    src = self.state.source_base_u8
                    disp = pre.disp[..., 0] if (pre.disp.ndim == 3 and pre.disp.shape[2] == 1) else None
                    if disp is None:
                        return

                    h, w = src.shape[:2]
                    max_dim = 720  # tune: larger = sharper overlay, smaller = faster
                    scale = min(1.0, float(max_dim) / float(max(h, w)))
                    new_w = max(1, int(round(w * scale)))
                    new_h = max(1, int(round(h * scale)))
                    if new_w == w and new_h == h:
                        self.state.overlay_src_u8 = src
                        self.state.overlay_disp_f32 = disp.astype(np.float32)
                    else:
                        self.state.overlay_src_u8 = cv2.resize(src, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        self.state.overlay_disp_f32 = cv2.resize(disp.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)

                    # If overlay is enabled, update it now that preprocessing is ready.
                    self._schedule_focus_overlay()

                self.root.after(0, build_preview_buffers)
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Preprocess failed", str(exc)))
                self._set_status("Preprocess failed.", 0)
            finally:
                self.root.after(0, lambda: self._set_busy(False))

        threading.Thread(target=worker, daemon=True).start()

    def _render_key(self) -> str:
        assert self.state.rgb_path is not None
        return f"{self.state.rgb_path.stem}__L{self._current_lens()}__K{self.k_var.get():.2f}__f{self.focal_var.get():.3f}"

    def _start_render(self) -> None:
        if not self._ensure_engine():
            return
        if self.state.pre is None:
            self._set_status("Please preprocess first.", 0)
            return
        if self.state.busy:
            return

        # If `scatter_cuda` is missing, rendering will fall back to the Python-loop
        # implementation and can take *minutes to hours* for normal image sizes/lens.
        # Make this explicit so it doesn't look like the GUI is "stuck".
        try:
            import scatter_cuda  # type: ignore

            has_scatter_cuda = True
        except Exception:
            has_scatter_cuda = False

        if not has_scatter_cuda:
            ok = messagebox.askyesno(
                "Slow rendering (CUDA extension missing)",
                "The optional CUDA extension `scatter_cuda` is not installed, so rendering will use a slow fallback.\n\n"
                "To enable fast GPU rendering (CUDA 11.7), run:\n"
                "  - `bash setup.sh` (recommended)\n"
                "  - or `bash scripts/build_scatter_cuda.sh` from the correct environment\n\n"
                "Continue with slow rendering anyway?",
            )
            if not ok:
                self._set_status("Render cancelled (build scatter_cuda for speed).", 0)
                return

        key = self._render_key()
        if self.state.last_render_key == key:
            return
        self.state.last_render_key = key

        self._set_busy(True)
        self._set_status("Rendering...", 50)

        def worker() -> None:
            try:
                t0 = time.time()
                out = self.engine.render(
                    self.state.pre,
                    focal=float(self.focal_var.get()),
                    k_blur=float(self.k_var.get()),
                )
                dt = time.time() - t0

                out_u8 = (out * 255.0).astype(np.uint8)
                self.state.last_render_u8 = out_u8

                self.root.after(0, lambda: self._set_image_on_canvas(self.res_canvas, out_u8, which="res"))
                self._set_status(f"Rendered in {dt:.2f}s. Click 'Save rendered image' to export.", 100)
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Render failed", str(exc)))
                self._set_status("Render failed.", 0)
            finally:
                self.root.after(0, lambda: self._set_busy(False))

        threading.Thread(target=worker, daemon=True).start()

    def _save_rendered_image(self) -> None:
        if self.state.busy:
            return
        if self.state.rgb_path is None:
            messagebox.showerror("Error", "Please select an image first.")
            return
        if self.state.last_render_u8 is None:
            messagebox.showerror("Error", "Nothing to save yet. Render an image first.")
            return

        default_name = f"{self._render_key()}.png"
        out_path = filedialog.asksaveasfilename(
            initialdir=str(self.out_dir),
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            title="Save rendered image",
        )
        if not out_path:
            return

        try:
            # cv2 expects BGR
            cv2.imwrite(out_path, cv2.cvtColor(self.state.last_render_u8, cv2.COLOR_RGB2BGR))
            self._set_status(f"Saved: {Path(out_path).name}", 0)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))


if __name__ == "__main__":
    root = tk.Tk()
    app = DrBokehGUI(root)
    root.mainloop()


