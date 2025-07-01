from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ----------------------------------------------------------------------
#  import algorithms from descriptors.py
# ----------------------------------------------------------------------

from descriptors import (
    hsv_hist, colour_moments,
    chi_square, euclidean,
)

DESCRIPTORS: dict[str, tuple[str,
Callable[[np.ndarray], np.ndarray],
Callable[[np.ndarray, np.ndarray], float]]] = {
    "hsv_hist": ("HSV histogram", hsv_hist, chi_square),
    "moments": ("Colour moments", colour_moments, euclidean),
}


# ----------------------------------------------------------------------
#  Visualisation helpers  (UI only)
# ----------------------------------------------------------------------


def _split_hsv(hist: np.ndarray, bins: int = 16):
    h = hist.reshape(bins, bins, bins).sum(axis=(1, 2))
    s = hist.reshape(bins, bins, bins).sum(axis=(0, 2))
    v = hist.reshape(bins, bins, bins).sum(axis=(0, 1))
    return h, s, v


# HSV histogram plot ----------------------------------------------------

def draw_hsv(fig, img1, img2, d1, d2):
    fig.clear()
    gs = fig.add_gridspec(2, 3)
    ax1 = fig.add_subplot(gs[0, 0]);
    ax2 = fig.add_subplot(gs[0, 1])
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB));
    ax1.axis("off");
    ax1.set_title("Image 1")
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB));
    ax2.axis("off");
    ax2.set_title("Image 2")

    colours = ["red", "green", "blue"]
    names = ["Hue", "Saturation", "Value"]
    for i, (h1, h2) in enumerate(zip(_split_hsv(d1), _split_hsv(d2))):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(h1, color=colours[i], label="Img1")
        ax.plot(h2, color=colours[i], linestyle="--", label="Img2")
        ax.set_title(f"{names[i]} histogram");
        ax.set_xlabel("Bin");
        ax.set_ylabel("Norm count");
        ax.legend()


# Colour‑moments bar plot ----------------------------------------------

def draw_moments(fig, img1, img2, d1, d2):
    fig.clear()
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2])
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB));
    ax1.axis("off");
    ax1.set_title("Image 1")
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB));
    ax2.axis("off");
    ax2.set_title("Image 2")

    ax_bar = fig.add_subplot(gs[1, :])
    labels = ["μ_H", "σ_H", "sk_H", "μ_S", "σ_S", "sk_S", "μ_V", "σ_V", "sk_V"]
    x = np.arange(len(labels));
    w = 0.35
    ax_bar.bar(x - w / 2, d1, w, label="Img1");
    ax_bar.bar(x + w / 2, d2, w, label="Img2")
    ax_bar.set_xticks(x);
    ax_bar.set_xticklabels(labels, rotation=45, ha="right")
    ax_bar.set_ylabel("Value");
    ax_bar.set_title("Colour moments");
    ax_bar.legend()


DRAWERS = {
    "hsv_hist": draw_hsv,
    "moments": draw_moments,
}


# ----------------------------------------------------------------------
#  Tkinter application – UI only
# ----------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Colour Descriptor Comparator")
        self.geometry("1000x650")
        self.img_paths = {"img1": None, "img2": None}
        self.selected_mode = tk.StringVar(value="hsv_hist")
        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self);
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        ttk.Button(top, text="Load Image 1", command=lambda: self._load("img1")).pack(side=tk.LEFT)
        ttk.Button(top, text="Load Image 2", command=lambda: self._load("img2")).pack(side=tk.LEFT, padx=5)
        for k, (label, _, _) in DESCRIPTORS.items():
            ttk.Radiobutton(top, text=label, variable=self.selected_mode, value=k).pack(side=tk.LEFT, padx=5)
        ttk.Button(top, text="Compare", command=self._compare).pack(side=tk.LEFT, padx=10)

        self.fig = plt.Figure(figsize=(9, 5.5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self);
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.status = ttk.Label(self, text="Load two images to begin.");
        self.status.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    def _load(self, slot):
        fname = filedialog.askopenfilename(title="Select image",
                                           filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if fname:
            self.img_paths[slot] = Path(fname);
            self.status.config(text=f"Loaded {slot}: {fname}")

    def _compare(self):
        if None in self.img_paths.values():
            messagebox.showwarning("Missing images", "Load both images first.");
            return
        key = self.selected_mode.get();
        label, extractor, metric = DESCRIPTORS[key]
        img1, img2 = cv2.imread(str(self.img_paths["img1"])), cv2.imread(str(self.img_paths["img2"]))
        if img1 is None or img2 is None:
            messagebox.showerror("Error", "Could not open one of the images.");
            return
        d1, d2 = extractor(img1), extractor(img2)
        dist = metric(d1, d2)
        sim = 100 * (1 - dist / (dist + 1))
        DRAWERS[key](self.fig, img1, img2, d1, d2)
        self.fig.suptitle(f"{label} | distance={dist:.3f} | similarity≈{sim:.1f}%")
        self.canvas.draw()
        self.status.config(text=f"{label}: distance={dist:.3f}, similarity≈{sim:.1f}%")


if __name__ == "__main__":
    App().mainloop()
