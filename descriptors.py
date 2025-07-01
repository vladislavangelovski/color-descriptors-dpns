from pathlib import Path
import cv2, numpy as np

# ------------------------------------------------------------------
# Descriptor A : HSV histogram
# ------------------------------------------------------------------

def hsv_hist(img: np.ndarray, bins: int = 16) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins] * 3, [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()


def chi_square(d1: np.ndarray, d2: np.ndarray) -> float:
    return cv2.compareHist(d1.astype("float32"), d2.astype("float32"), cv2.HISTCMP_CHISQR)


# ------------------------------------------------------------------
# Descriptor B : Colour moments (mean, σ, skew)
# ------------------------------------------------------------------

def colour_moments(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    feats = []
    for ch in cv2.split(hsv):
        mu, sd = ch.mean(), ch.std()
        skew = np.mean(((ch - mu) / (sd + 1e-6)) ** 3)
        feats.extend([mu, sd, skew])
    return np.array(feats, dtype=np.float32)


def euclidean(d1: np.ndarray, d2: np.ndarray) -> float:
    return float(np.linalg.norm(d1 - d2))


# ------------------------------------------------------------------
# Descriptor C : RGB histogram + intersection
# ------------------------------------------------------------------

def rgb_hist(img: np.ndarray, bins: int = 32) -> np.ndarray:
    hist = cv2.calcHist([img], [0, 1, 2], None, [bins] * 3, [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()


def hist_intersection(d1: np.ndarray, d2: np.ndarray) -> float:
    return 1.0 - float(np.minimum(d1, d2).sum())


# ------------------------------------------------------------------
# CLI test harness (supports 3 modes)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    USAGE = ("Usage:\n"
             "  python descriptors.py <img1> <img2> hsv       # HSV histogram\n"
             "  python descriptors.py <img1> <img2> moments   # Colour moments\n"
             "  python descriptors.py <img1> <img2> rgb       # RGB histogram intersection")

    if len(sys.argv) != 4 or sys.argv[3] not in {"hsv", "moments", "rgb"}:
        print(USAGE)
        sys.exit(1)

    img1, img2 = cv2.imread(sys.argv[1]), cv2.imread(sys.argv[2])
    if img1 is None or img2 is None:
        sys.exit("Could not open one of the images.")

    mode = sys.argv[3]
    if mode == "hsv":
        d1, d2 = hsv_hist(img1), hsv_hist(img2)
        dist = chi_square(d1, d2)
    elif mode == "moments":
        d1, d2 = colour_moments(img1), colour_moments(img2)
        dist = euclidean(d1, d2)
    else:  # rgb
        d1, d2 = rgb_hist(img1), rgb_hist(img2)
        dist = hist_intersection(d1, d2)

    sim = 100 * (1 - dist / (dist + 1))
    print(f"Mode: {mode}\nDistance: {dist:.3f}\nSimilarity: {sim:.1f} %")
