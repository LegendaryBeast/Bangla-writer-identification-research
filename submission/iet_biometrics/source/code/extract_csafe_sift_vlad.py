#!/usr/bin/env python3
"""SIFT-VLAD writer-identification features for the CSAFE cohort (protocol in
results/p2_extensions_2026-11-09/protocol.json).

RootSIFT descriptors -> PCA(64) -> 100-cluster VLAD with intra-normalisation,
signed square root and L2 -> PCA whitening to 128 dimensions -> L2. The codebook,
PCA and whitening use only session-1 LND/WOZ repetition-2 pages (never gallery or
query pages) and no labels. Output: features_sift_vlad.npz aligned to units.csv.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
CS = ROOT / 'results/csafe_replication_2026-11-08'
OUT = ROOT / 'results/p2_extensions_2026-11-09'
DATA = ROOT / 'dataset/csafe'
SEED = 20261109
WIDTH, NFEAT, PCA_DIM, K, WHITE = 1600, 4000, 64, 100, 128


def rootsift(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert img is not None, path
    h, w = img.shape
    img = cv2.resize(img, (WIDTH, int(round(h * WIDTH / w))), interpolation=cv2.INTER_AREA)
    kp, d = cv2.SIFT_create(nfeatures=NFEAT).detectAndCompute(img, None)
    assert d is not None and len(d) > 100, (path, 0 if d is None else len(d))
    d = d.astype(np.float64)
    d /= np.maximum(d.sum(1, keepdims=True), 1e-12)
    return np.sqrt(d).astype(np.float32)


def vlad(desc, pca, km):
    x = pca.transform(desc)
    a = km.predict(x)
    c = km.cluster_centers_
    v = np.zeros_like(c)
    for k in range(len(c)):
        m = a == k
        if m.any():
            v[k] = (x[m] - c[k]).sum(0)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    v = v.ravel()
    v = np.sign(v) * np.sqrt(np.abs(v))
    return v / max(np.linalg.norm(v), 1e-12)


def main():
    start = time.time()
    U = pd.read_csv(CS / 'units.csv'); A = pd.read_csv(CS / 'images.csv')
    assert np.array_equal(U.image_id, A.image_id)
    paths = [DATA / p for p in A.relative_path]
    codebook = (U.session == 1) & (U.repetition == 2)
    needed = codebook | U.gallery | U['query']
    idx = np.flatnonzero(needed)
    for i in idx:
        assert paths[i].exists(), paths[i]
        assert hashlib.sha256(paths[i].read_bytes()).hexdigest() == U.image_sha256[i]
    desc = Parallel(n_jobs=6)(delayed(rootsift)(paths[i]) for i in idx)
    D = dict(zip(idx, desc))
    print(f'RootSIFT on {len(idx)} pages; median {int(np.median([len(d) for d in desc]))} descriptors; {time.time()-start:.0f}s', flush=True)
    rng = np.random.default_rng(SEED)
    cb = np.flatnonzero(codebook)
    sample = np.concatenate([D[i][rng.choice(len(D[i]), min(300, len(D[i])), replace=False)] for i in cb])
    pca = PCA(PCA_DIM, random_state=SEED).fit(sample)
    km = MiniBatchKMeans(K, random_state=SEED, n_init=3, batch_size=4096).fit(pca.transform(sample))
    V = {i: vlad(D[i], pca, km) for i in idx}
    white = PCA(WHITE, whiten=True, random_state=SEED).fit(np.stack([V[i] for i in cb]))
    F = np.full((len(U), WHITE), np.nan)
    for i in idx:
        z = white.transform(V[i][None])[0]
        F[i] = z / max(np.linalg.norm(z), 1e-12)
    np.savez_compressed(OUT / 'features_sift_vlad.npz', image_id=U.image_id.to_numpy(), sift_vlad=F.astype(np.float32))
    prov = dict(script=str(Path(__file__).relative_to(ROOT)), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                opencv=cv2.__version__, pages_with_features=int(len(idx)), codebook_pages=int(len(cb)),
                descriptors_per_page_median=int(np.median([len(d) for d in desc])), pca_dim=PCA_DIM, clusters=K,
                whitened_dim=WHITE, codebook_sample=int(len(sample)), seconds=round(time.time() - start, 1),
                note='Codebook, PCA and whitening fitted on session-1 repetition-2 pages only; no labels used.')
    (OUT / 'features_sift_vlad_provenance.json').write_text(json.dumps(prov, indent=2) + '\n')
    print(json.dumps(prov, indent=2))


if __name__ == '__main__':
    sys.exit(main())
