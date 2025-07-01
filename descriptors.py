from pathlib import Path
import cv2
import numpy as np


def hsv_hist(img: np.ndarray, bins: int = 16) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins] * 3, [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()


def chi_square(d1: np.ndarray, d2: np.ndarray) -> float:
    return cv2.compareHist(d1.astype("float32"), d2.astype("float32"), cv2.HISTCMP_CHISQR)


# ------------------------------------------------------------------
# Descriptor B : Colour moments (mean, σ, skewness)  ---  9 numbers
# ------------------------------------------------------------------

def colour_moments(img: np.ndarray) -> np.ndarray:
    """
    Return a 9-D vector:
        [μ_H, σ_H, skew_H,  μ_S, σ_S, skew_S,  μ_V, σ_V, skew_V]
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    feats = []
    for ch in cv2.split(hsv):  # H, S, V
        mu = ch.mean()
        sd = ch.std()
        skew = np.mean(((ch - mu) / (sd + 1e-6)) ** 3)
        feats.extend([mu, sd, skew])
    return np.array(feats, dtype=np.float32)


def euclidean(d1: np.ndarray, d2: np.ndarray) -> float:
    """Plain L2 distance between two 9-D vectors."""
    return float(np.linalg.norm(d1 - d2))


if __name__ == "__main__":
    import sys

    USAGE = ("Usage:\n"
             "  python descriptors.py <img1> <img2> hist    # HSV histogram\n"
             "  python descriptors.py <img1> <img2> moments # colour moments")

    if len(sys.argv) != 4 or sys.argv[3] not in {"hist", "moments"}:
        print(USAGE)
        sys.exit(1)

    img1 = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2])
    if img1 is None or img2 is None:
        sys.exit("⚠️  Could not open one of the images.")

    mode = sys.argv[3]
    if mode == "hist":
        d1, d2 = hsv_hist(img1), hsv_hist(img2)
        dist = chi_square(d1, d2)
    else:  # moments
        d1, d2 = colour_moments(img1), colour_moments(img2)
        dist = euclidean(d1, d2)

    sim = 100 * (1 - dist / (dist + 1))  # simple 0-100 mapping
    print(f"Mode: {mode}")
    print(f"Distance    : {dist:.3f}")
    print(f"Similarity %: {sim:.1f}")
