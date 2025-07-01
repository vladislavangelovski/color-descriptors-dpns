from pathlib import Path
import cv2
import numpy as np


def hsv_hist(img: np.ndarray, bins: int = 16) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins] * 3, [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()


def chi_square(d1: np.ndarray, d2: np.ndarray) -> float:
    return cv2.compareHist(d1.astype("float32"), d2.astype("float32"), cv2.HISTCMP_CHISQR)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print("Usage: python descriptors.py <image1> <image2>")
        sys.exit(1)

    img1 = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2])
    if img1 is None or img2 is None:
        sys.exit("Could not open one of the images")

    d1, d2 = hsv_hist(img1), hsv_hist(img2)
    dist = chi_square(d1, d2)
    sim = 100 * (1 - dist / (dist + 1))
    print(f"Chi-square distance: {dist: .3f}")
    print(f"Similarity (0%-100%): {sim: .1f}")
