#!/usr/bin/env python3
"""Features of whole paragraphs and of horizontal strips of paragraphs (writer identification study).

A unit is a whole paragraph (n = 1) or one of n horizontal strips of it (n = 2, 4, 8; paragraphs have about 7 lines,
so a strip holds about 3.5, 1.75 or 0.9 lines). The text block of the
paragraph (rows of the 1,536-px STROKE ink mask, as in stroke_features.ink_mask) is cut into n bands of equal height,
each cut moved to the emptiest row of the smoothed ink profile within half a line pitch, so that cuts fall between
text lines where possible.
Strips keep the full width of the photograph, so every representation sees them at the scale of the whole paragraph.

Representations are those of the PRL study, computed on the unit:
  hc183            ink_features of the 768-px ink mask (run_crop_evaluation.image_to_ink) restricted to the strip rows
  stroke           208-d STROKE features of the 1,536-px ink restricted to the strip rows, cropped to its text block
  stroke_cnn       ResNet-18 on black-on-white, stroke-width-normalised tiles of that ink
  resnet18, efficientnet_b0, vit_b_16   frozen ImageNet embeddings of 224-px grey tiles at 1,568 px width
  global_intensity 7 grey-level statistics of the 768-px grey strip (acquisition control)
  ink_density      fraction of ink pixels of the 768-px mask in the strip (covariate of the adjusted model)
For n = 1 every representation equals the PRL extraction (extract_final_features.py, extract_stroke_features.py).
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
import torchvision
import torchvision.models as tvm
from PIL import Image
from skimage.morphology import remove_small_objects

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    sys.path.insert(0, str(p))
import stroke_features as sf  # noqa: E402
from run_crop_evaluation import image_to_ink, ink_features  # noqa: E402

FRACTIONS = (1, 2, 4, 8)
LINE_PITCH = 96          # median text-line pitch of the dataset at 1,536 px width (row-profile autocorrelation)
TILE, TILE_W = 224, 1568
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DEEP = ["resnet18", "efficientnet_b0", "vit_b_16"]


def intensity_features(values):
    """final_study_pipeline._intensity_features."""
    values = np.asarray(values, dtype=float)
    return np.array([values.mean(), values.std(), *np.quantile(values, [.05, .25, .5, .75, .95])])


def load_gray(path):
    with Image.open(path) as im:
        return im.convert("L").copy()


def stroke_ink_uncropped(gray):
    """stroke_features.ink_mask before the crop to the text block."""
    w, h = gray.size
    a = np.asarray(gray.resize((sf.WIDTH, round(sf.WIDTH * h / w)), Image.Resampling.LANCZOS), np.float32)
    shade = ndi.gaussian_filter(a, 17 * sf.WIDTH / 768)
    ink = (shade - a > 17) & (shade > 120) & (a < 205)
    b = 8
    ink[:b] = ink[-b:] = False; ink[:, :b] = ink[:, -b:] = False
    ink = ndi.binary_opening(ink, np.ones((2, 2)))
    return remove_small_objects(ink, 12)


def text_rows(ink):
    rows = np.flatnonzero(ink.mean(1) > 0.01)
    r0, r1 = np.percentile(rows, [0.5, 99.5]).astype(int)
    return r0, r1


def crop_text_block(ink):
    """The crop of stroke_features.ink_mask."""
    rows = np.flatnonzero(ink.mean(1) > 0.01); cols = np.flatnonzero(ink.mean(0) > 0.005)
    if rows.size == 0 or cols.size == 0:
        return None
    r0, r1 = np.percentile(rows, [0.5, 99.5]).astype(int); c0, c1 = np.percentile(cols, [0.5, 99.5]).astype(int)
    pad = 10
    return ink[max(0, r0 - pad):r1 + pad, max(0, c0 - pad):c1 + pad]


def strip_bounds(ink, n):
    """Row fractions [(f0, f1), ...] of the n strips of the text block (whole image for n = 1)."""
    H = ink.shape[0]
    if n == 1:
        return [(0.0, 1.0)]
    r0, r1 = text_rows(ink)
    block = r1 - r0
    prof = ndi.gaussian_filter1d(ink.mean(1).astype(float), LINE_PITCH / 5)
    win = min(LINE_PITCH // 2, max(3, int(block / (3 * n))))
    cuts = [r0]
    for j in range(1, n):
        nom = int(round(r0 + block * j / n))
        lo, hi = max(cuts[-1] + 1, nom - win), min(r1 - 1, nom + win)
        cuts.append(lo + int(np.argmin(prof[lo:hi + 1])) if hi >= lo else nom)
    cuts.append(r1 + 1)
    return [(cuts[j] / H, cuts[j + 1] / H) for j in range(n)]


def rows_of(f, height):
    return slice(int(round(f[0] * height)), max(int(round(f[0] * height)) + 1, int(round(f[1] * height))))


def stroke_unit(ink_strip):
    """stroke_features.stroke_features on an ink strip; None if the strip has too little writing."""
    ink = crop_text_block(ink_strip)
    if ink is None or ink.sum() < 200:
        return None
    try:
        sw, sk = sf.stroke_width(ink)
        L = max(4, int(round(2 * sw)))
        d1, hg, tn = sf.contour_features(ink, L)
        e, o, sl, dens = sf.skeleton_features(ink, sk, L)
    except (ValueError, ZeroDivisionError):
        return None
    v = np.concatenate([d1, hg, tn, e, o, sl, dens]).astype(np.float32)
    if not np.isfinite(v).all():
        return None
    return v, ink, sw


def grey_tiles(a, ink):
    """extract_final_features.tiles on a (strip of the) 1,568-px grey image and its ink mask."""
    rows = int(np.ceil(a.shape[0] / TILE))
    paper = float(np.median(a))
    pad = rows * TILE - a.shape[0]
    a = np.pad(a, ((0, pad), (0, 0)), constant_values=paper)
    ink = np.pad(ink, ((0, pad), (0, 0)), constant_values=False)
    out, frac = [], []
    for r in range(rows):
        for c in range(TILE_W // TILE):
            sl = (slice(r * TILE, (r + 1) * TILE), slice(c * TILE, (c + 1) * TILE))
            out.append(a[sl]); frac.append(ink[sl].mean())
    keep = np.flatnonzero(np.array(frac) >= 0.01)
    if keep.size == 0:
        keep = np.arange(len(out))
    x = np.stack([out[i] for i in keep]) / 255.0
    x = np.repeat(x[:, None], 3, axis=1)
    x = (x - MEAN[None, :, None, None]) / STD[None, :, None, None]
    return torch.from_numpy(x.astype(np.float32))


def binary_tile_batch(ink, sw):
    t = torch.from_numpy(sf.binary_tiles(ink, sw)).float()[:, None].repeat(1, 3, 1, 1)
    return (t - torch.from_numpy(MEAN).view(1, 3, 1, 1)) / torch.from_numpy(STD).view(1, 3, 1, 1)


def backbones(device):
    r18 = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1); r18.fc = torch.nn.Identity()
    eff = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1); eff.classifier = torch.nn.Identity()
    vit = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1); vit.heads = torch.nn.Identity()
    return {k: m.eval().to(device) for k, m in [("resnet18", r18), ("efficientnet_b0", eff), ("vit_b_16", vit)]}


def embed(net, x, device):
    with torch.no_grad(), torch.autocast(device.type, enabled=False):
        return torch.cat([net(x[j:j + 64].to(device)).float().cpu() for j in range(0, len(x), 64)]).mean(0).numpy()


def image_units(gray, nets, device, fractions=FRACTIONS):
    w, h = gray.size
    ink768 = image_to_ink(gray)
    g768 = np.asarray(gray.resize((768, max(32, round(768 * h / w))), Image.Resampling.LANCZOS), float)
    a1568 = np.asarray(gray.resize((TILE_W, round(TILE_W * h / w)), Image.Resampling.LANCZOS), np.float32)
    ink1568 = np.asarray(Image.fromarray(np.uint8(ink768 * 255)).resize((a1568.shape[1], a1568.shape[0]),
                                                                        Image.Resampling.NEAREST)) > 0
    ink1536 = stroke_ink_uncropped(gray)
    out = []
    for n in fractions:
        for j, f in enumerate(strip_bounds(ink1536, n)):
            st = stroke_unit(ink1536[rows_of(f, ink1536.shape[0])])
            ink_s = ink768[rows_of(f, ink768.shape[0])]
            if st is None or ink_s.sum() < 50:
                out.append(dict(n=n, strip=j, f0=f[0], f1=f[1], valid=False)); continue
            v, ink_c, sw = st
            r = dict(n=n, strip=j, f0=f[0], f1=f[1], valid=True, stroke_width=sw,
                     hc183=ink_features(ink_s), stroke=v, ink_density=float(ink_s.mean()),
                     global_intensity=intensity_features(g768[rows_of(f, g768.shape[0])]))
            x = grey_tiles(a1568[rows_of(f, a1568.shape[0])], ink1568[rows_of(f, a1568.shape[0])])
            r["n_tiles"] = len(x)
            for k in DEEP:
                r[k] = embed(nets[k], x, device)
            r["stroke_cnn"] = embed(nets["resnet18"], binary_tile_batch(ink_c, sw), device)
            out.append(r)
    return out


REPS = ["hc183", "stroke", "stroke_cnn"] + DEEP + ["global_intensity"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="folder with metadata/images.csv and the class folders")
    ap.add_argument("--output", required=True)
    ap.add_argument("--writers", type=int, default=0, help="first k writers of each class only (testing)")
    a = ap.parse_args()
    D, out = Path(a.dataset), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(D / "metadata/images.csv")
    if a.writers:
        keep = []
        for h in ("L", "R"):
            ws = sorted({w for w in meta.writer_id if w.startswith(h)}, key=lambda s: int(s.split("_p")[1]))[:a.writers]
            keep += ws
        meta = meta[meta.writer_id.isin(keep)].reset_index(drop=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    nets = backbones(device)
    rows, X = [], {k: [] for k in REPS}
    t0 = time.time()
    for i, m in enumerate(meta.itertuples()):
        for u in image_units(load_gray(D / m.relative_path), nets, device):
            rows.append(dict(unit_id=f"{m.image_id}_n{u['n']}_s{u['strip']}", image_id=m.image_id, writer_id=m.writer_id,
                             label=int(m.label), para=int(m.paragraph_number), n=u["n"], strip=u["strip"],
                             f0=u["f0"], f1=u["f1"], valid=u["valid"], stroke_width=u.get("stroke_width", np.nan),
                             ink_density=u.get("ink_density", np.nan), n_tiles=u.get("n_tiles", 0)))
            for k in REPS:
                X[k].append(u[k] if u["valid"] else None)
        if i % 10 == 0:
            print(f"{i + 1}/{len(meta)} {m.image_id} {time.time() - t0:.0f}s", flush=True)
    units = pd.DataFrame(rows)
    arrays = {}
    for k, v in X.items():
        dim = next(len(x) for x in v if x is not None)
        arrays[k] = np.asarray([x if x is not None else np.full(dim, np.nan) for x in v], np.float32)
    np.savez_compressed(out / "unit_features.npz", unit_id=units.unit_id.to_numpy(), **arrays)
    units.to_csv(out / "units.csv", index=False)
    (out / "unit_feature_provenance.json").write_text(json.dumps(dict(
        fractions=FRACTIONS, dims={k: int(v.shape[1]) for k, v in arrays.items()}, device=str(device),
        python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
        n_units=len(units), n_invalid=int((~units.valid).sum()), seconds=round(time.time() - t0)), indent=2))
    print("saved", len(units), "units,", int((~units.valid).sum()), "invalid")


if __name__ == "__main__":
    main()
