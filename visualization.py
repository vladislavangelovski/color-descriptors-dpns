import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from descriptors import hsv_hist, chi_square  # reuse your functions


def split_hsv_hist(hist: np.ndarray, bins: int = 16):
    h_size = s_size = v_size = bins
    total = bins ** 3
    h_hist = hist.reshape(bins, bins, bins).sum(axis=(1, 2))
    s_hist = hist.reshape(bins, bins, bins).sum(axis=(0, 2))
    v_hist = hist.reshape(bins, bins, bins).sum(axis=(0, 1))
    return h_hist, s_hist, v_hist


def show(img1_path: Path, img2_path: Path, bins: int = 16):
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        sys.exit("Could not open one of the images.")

    d1, d2 = hsv_hist(img1, bins), hsv_hist(img2, bins)
    dist = chi_square(d1, d2)
    sim = 100 * (1 - dist / (dist + 1))

    print(f"Chi-square distance: {dist:.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    fig.suptitle(
        f"HSV-Histogram Comparison   |   χ² = {dist:.3f}   |   Similarity ≈ {sim:.1f}%",
        fontsize=14,
    )

    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Image 1")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Image 2")
    axes[0, 1].axis("off")

    axes[0, 2].axis("off")

    channel_names = ["Hue", "Saturation", "Value"]
    colours = ["red", "green", "blue"]
    for idx, (h1, h2) in enumerate(zip(split_hsv_hist(d1, bins),
                                       split_hsv_hist(d2, bins))):
        ax = axes[1, idx]
        ax.plot(h1, label="Img 1", color=colours[idx], alpha=0.7)
        ax.plot(h2, label="Img 2", color=colours[idx], alpha=0.4, linestyle="--")
        ax.set_title(f"{channel_names[idx]} histogram")
        ax.set_xlabel("Bin")
        ax.set_ylabel("Normalised count")
        ax.legend(frameon=False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python viz.py <image1> <image2>")
        sys.exit(1)
    show(Path(sys.argv[1]), Path(sys.argv[2]))
