#!/usr/bin/env python3
"""CSAFE whole-page features using the unchanged original extractors."""
import argparse
import hashlib
import json
import platform
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.models as tvm
from PIL import Image

import wid_features as vf
from hc183 import ink_features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/csafe_replication_2026-11-08"
DATA = ROOT / "dataset/csafe"
REPS = ("hc183", "stroke", "resnet18", "vit_b_16", "global_intensity")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--wait-for-download", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.hub.set_dir(str(ROOT / "dataset/model_cache"))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoints = {}
    for weight in (tvm.ResNet18_Weights.IMAGENET1K_V1, tvm.ViT_B_16_Weights.IMAGENET1K_V1):
        path = ROOT / "dataset/model_cache/checkpoints" / weight.url.rsplit("/", 1)[-1]
        if not path.exists():
            raise FileNotFoundError(f"Acquire exact checkpoint first: {path}")
        checkpoints[str(weight)] = dict(file=path.name, sha256=digest(path), url=weight.url)
    r18 = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    r18.fc = torch.nn.Identity()
    vit = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
    vit.heads = torch.nn.Identity()
    nets = {"resnet18": r18.eval().to(device), "vit_b_16": vit.eval().to(device)}
    source_files = [Path(__file__), Path(vf.__file__), ROOT / "code/hc183.py", ROOT / "code/stroke_features.py"]
    signature = dict(code={p.name: digest(p) for p in source_files}, checkpoints=checkpoints,
        manifest_sha256=digest(OUT / "images.csv"), neural_device=str(device))
    cache = DATA / "feature_cache"
    cache.mkdir(exist_ok=True)
    images = pd.read_csv(OUT / "images.csv")
    acquisition_path = OUT / "acquisition.json"
    acquired = json.loads(acquisition_path.read_text())["image_sha256"] if acquisition_path.exists() else {}
    if not acquired and not args.wait_for_download:
        raise FileNotFoundError("Run acquisition first, or pass --wait-for-download")
    expected = {}
    for session in (1,2,3):
        for entry in json.loads((DATA / f"Session {session}_inventory.json").read_text()):
            expected[str(Path('selected') / entry['name'])] = entry
    rows, features = [], {rep: [] for rep in REPS}
    start = time.time()
    for i, row in enumerate(images.itertuples(), 1):
        path = DATA / row.relative_path
        deadline = time.time() + 1800
        while True:
            if path.exists():
                content = path.read_bytes()
                entry = expected[row.relative_path]
                if len(content) == entry['size'] and zlib.crc32(content) == entry['crc']:
                    break
            if not args.wait_for_download or time.time() > deadline:
                raise FileNotFoundError(f"Verified download not ready: {path}")
            time.sleep(1)
        image_hash = hashlib.sha256(content).hexdigest()
        if row.relative_path in acquired:
            assert image_hash == acquired[row.relative_path]
        sig = json.dumps(dict(**signature, image_sha256=image_hash), sort_keys=True)
        key = hashlib.sha256(sig.encode()).hexdigest()
        file = cache / f"{row.image_id}_{key}.npz"
        if file.exists():
            with np.load(file) as saved:
                valid = bool(saved["valid"])
                values = {rep: saved[rep] for rep in REPS if rep in saved} if valid else {}
                density = float(saved["ink_density"])
                tiles = int(saved["n_tiles"])
        else:
            gray = vf.load_gray(path)
            w, h = gray.size
            ink768 = vf.image_to_ink(gray)
            g768 = np.asarray(gray.resize((768, max(32, round(768*h/w))), Image.Resampling.LANCZOS), float)
            ink1536 = vf.stroke_ink_uncropped(gray)
            stroke = vf.stroke_unit(ink1536)
            valid = stroke is not None and ink768.sum() >= 50
            values, density, tiles = {}, float(ink768.mean()), 0
            if valid:
                if row.deep_required:
                    gray1568 = np.asarray(gray.resize((vf.TILE_W, round(vf.TILE_W*h/w)), Image.Resampling.LANCZOS), np.float32)
                    ink1568 = np.asarray(Image.fromarray(np.uint8(ink768*255)).resize(
                        (gray1568.shape[1], gray1568.shape[0]), Image.Resampling.NEAREST)) > 0
                    x = vf.grey_tiles(gray1568, ink1568)
                    tiles = len(x)
                values = dict(hc183=ink_features(ink768), stroke=stroke[0], global_intensity=vf.intensity_features(g768))
                if row.deep_required:
                    for rep, net in nets.items():
                        values[rep] = vf.embed(net, x, device)
                valid = all(np.isfinite(v).all() for v in values.values())
            np.savez_compressed(file, valid=valid, ink_density=density, n_tiles=tiles, **values)
        rows.append(dict(image_id=row.image_id, writer_id=row.writer_id, label=row.label,
            session=row.session, prompt=row.prompt, repetition=row.repetition,
            attribute_train=row.attribute_train, gallery=row.gallery, query=row.query, condition=row.condition,
            deep_required=row.deep_required, parent_form_id=row.parent_form_id, valid=valid,
            ink_density=density, n_tiles=tiles, image_sha256=image_hash))
        for rep in REPS:
            features[rep].append(values.get(rep))
        if i % 10 == 0 or i == len(images):
            print(f"Features {i}/{len(images)}; {time.time()-start:.0f}s; {device}", flush=True)
    units = pd.DataFrame(rows)
    arrays = {}
    for rep, values in features.items():
        dim = next(len(v) for v in values if v is not None)
        arrays[rep] = np.asarray([v if v is not None else np.full(dim, np.nan) for v in values], np.float32)
    units.to_csv(OUT / "units.csv", index=False)
    np.savez_compressed(OUT / "features.npz", image_id=units.image_id.to_numpy().astype(str), **arrays)
    report = dict(**signature, python=platform.python_version(), torch=torch.__version__,
        torchvision=torchvision.__version__, device=str(device), threads=args.threads,
        n_images=len(units), invalid_images=units.loc[~units.valid, "image_id"].tolist(),
        dimensions={k: list(v.shape) for k,v in arrays.items()}, seconds=time.time()-start,
        crop_policy="Original provider-cropped handwriting pages; no demographic survey, filenames or printed headers supplied to descriptors.",
        preprocessing="Original n=1 extraction applied to whole handwriting pages: HC-183 at 768px, STROKE at 1536px, ImageNet tiles at 1568px. Neural embeddings only for gallery/query pages; all six session-1 pages supply handcrafted auxiliary features.")
    (OUT / "feature_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Feature extraction complete", report["dimensions"], flush=True)


if __name__ == "__main__":
    main()
