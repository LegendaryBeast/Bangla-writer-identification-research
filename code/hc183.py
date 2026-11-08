#!/usr/bin/env python3
"""HC-183 gradient descriptor of the ink mask (copied unchanged from the authors' handedness study).

image_to_ink  binary ink of a grey photograph at 768 px width (local-contrast threshold, 2x2 opening)
ink_features  183 values: 4x4 grid of 9-bin unsigned gradient-orientation histograms and ink density (160),
              row and column ink-projection statistics (18) and connected-component area quantiles (5)
"""
import numpy as np
import scipy.ndimage as ndi
from PIL import Image


def image_to_ink(im: Image.Image, width: int = 768) -> np.ndarray:
    # Aspect-preserving resize: stroke orientation angles must not be distorted
    w, h = im.size
    im = im.resize((width, max(32, round(width * h / w))), Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.float32)
    shade = ndi.gaussian_filter(a, sigma=17)
    ink = (shade - a > 17) & (shade > 120) & (a < 205)
    ink[:4, :] = False; ink[-4:, :] = False; ink[:, :4] = False; ink[:, -4:] = False
    ink = ndi.binary_opening(ink, structure=np.ones((2, 2)))
    return ink.astype(np.float32)

def ink_features(ink: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(ndi.gaussian_filter(ink, .7))
    mag = np.hypot(gx, gy)
    angle = (np.arctan2(gy, gx) + np.pi) % np.pi
    features = []
    # 4x4 spatial orientation histogram
    for row in np.array_split(np.arange(ink.shape[0]), 4):
        for col in np.array_split(np.arange(ink.shape[1]), 4):
            a = angle[np.ix_(row, col)].ravel()
            m = mag[np.ix_(row, col)].ravel()
            hist = np.histogram(a, bins=9, range=(0, np.pi), weights=m)[0]
            features.extend((hist / (hist.sum() + 1e-8)).tolist())
            features.append(float(ink[np.ix_(row, col)].mean()))
    # Horizontal and vertical ink projections
    for axis in (0, 1):
        proj = ink.mean(axis=axis)
        features.extend(np.quantile(proj, [0, .1, .25, .5, .75, .9, 1]).tolist())
        features.extend([float(proj.mean()), float(proj.std())])
    # Connected component size quantiles
    labels, n = ndi.label(ink)
    sizes = np.bincount(labels.ravel().astype(int))[1:]
    sizes = sizes[sizes >= 3]
    features.extend(np.quantile(sizes, [.1, .25, .5, .75, .9]).tolist() if sizes.size else [0.]*5)
    return np.asarray(features, dtype=np.float32)
