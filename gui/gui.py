"""Unified demo GUI for `bokeh_rendering_and_focus_stacking_suite/`.

This is the single entry point that merges:
- Dr.Bokeh-style **single-image bokeh rendering**
- Laplacian-pyramid-based **focus stacking**

Both demos use datasets under `bokeh_rendering_and_focus_stacking_suite/Imgs/`:
- `Imgs/bokeh_rendering/`
- `Imgs/focus_stacking/`
"""

from __future__ import annotations

import warnings
import tkinter as tk
from tkinter import ttk

from gui.gui_focus_stacking import FocusStackingGUI
from gui.gui_bokeh_rendering import DrBokehGUI


def main() -> None:
    # Silence noisy third-party deprecation warnings that appear on startup in
    # this legacy stack (e.g. `lightning_fabric` importing `pkg_resources`).
    warnings.filterwarnings(
        "ignore",
        message=r".*pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )

    root = tk.Tk()
    root.title("DSP Lab - Image Processing Suite")
    root.geometry("980x1040")

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)

    bokeh_frame = ttk.Frame(nb)
    focus_frame = ttk.Frame(nb)
    nb.add(bokeh_frame, text="Bokeh Rendering")
    nb.add(focus_frame, text="Focus Stacking")

    # Embed both UIs into tabs (they still use `root` for scheduling + dialogs).
    DrBokehGUI(root, parent=bokeh_frame)
    FocusStackingGUI(root, parent=focus_frame)

    root.mainloop()


if __name__ == "__main__":
    main()


