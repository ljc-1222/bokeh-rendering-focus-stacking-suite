"""Unified GUI entry point."""

from __future__ import annotations

import tkinter as tk
import warnings
from tkinter import ttk

from brnfs.ui.bokeh_rendering import DrBokehGUI
from brnfs.ui.focus_stacking import FocusStackingGUI


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )

    root = tk.Tk()
    root.title("BRnFS - Image Processing Suite")
    root.geometry("980x1040")

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True)

    bokeh_frame = ttk.Frame(nb)
    focus_frame = ttk.Frame(nb)
    nb.add(bokeh_frame, text="Bokeh Rendering")
    nb.add(focus_frame, text="Focus Stacking")

    DrBokehGUI(root, parent=bokeh_frame)
    FocusStackingGUI(root, parent=focus_frame)

    root.mainloop()
