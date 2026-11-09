#!/usr/bin/env python3
"""Warm auxiliary-only caches with the identical extraction operations.

Optional speed optimization. The main extractor can reproduce every value
without this helper. No prediction or model selection is performed here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from PIL import Image
import torchvision.models as tvm

import wid_features as vf
from hc183 import ink_features

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/csafe_replication_2026-11-08'
DATA=ROOT/'dataset/csafe'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def one(row,signature,expected):
    assert not row.deep_required
    path=DATA/row.relative_path;image_hash=sha(path)
    assert image_hash==expected[row.relative_path]
    key=hashlib.sha256(json.dumps(dict(**signature,image_sha256=image_hash),sort_keys=True).encode()).hexdigest()
    file=DATA/'feature_cache'/f'{row.image_id}_{key}.npz'
    if file.exists():return row.image_id,'existing'
    gray=vf.load_gray(path);w,h=gray.size
    ink768=vf.image_to_ink(gray)
    g768=np.asarray(gray.resize((768,max(32,round(768*h/w))),Image.Resampling.LANCZOS),float)
    stroke=vf.stroke_unit(vf.stroke_ink_uncropped(gray))
    valid=stroke is not None and ink768.sum()>=50
    values={}
    if valid:
        values=dict(hc183=ink_features(ink768),stroke=stroke[0],global_intensity=vf.intensity_features(g768))
        valid=all(np.isfinite(v).all() for v in values.values())
    temporary=file.with_name(file.stem+f'.{os.getpid()}.partial.npz')
    np.savez_compressed(temporary,valid=valid,ink_density=float(ink768.mean()),n_tiles=0,**values)
    temporary.replace(file)
    return row.image_id,'computed'


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--start-index',type=int,default=250)
    ap.add_argument('--jobs',type=int,default=3);args=ap.parse_args()
    codes=[ROOT/'code/extract_csafe_features.py',Path(vf.__file__),ROOT/'code/hc183.py',ROOT/'code/stroke_features.py']
    checkpoints={}
    for weight in (tvm.ResNet18_Weights.IMAGENET1K_V1,tvm.ViT_B_16_Weights.IMAGENET1K_V1):
        path=ROOT/'dataset/model_cache/checkpoints'/weight.url.rsplit('/',1)[-1]
        checkpoints[str(weight)]=dict(file=path.name,sha256=sha(path),url=weight.url)
    signature=dict(code={p.name:sha(p) for p in codes},checkpoints=checkpoints,
        manifest_sha256=sha(OUT/'images.csv'),neural_device='mps')
    images=pd.read_csv(OUT/'images.csv').iloc[args.start_index:]
    images=images[~images.deep_required]
    expected=json.loads((OUT/'acquisition.json').read_text())['image_sha256']
    with parallel_config(backend='loky',n_jobs=args.jobs,inner_max_num_threads=1):
        results=Parallel()(delayed(one)(row,signature,expected) for row in images.itertuples(index=False))
    report=dict(optional=True,code_sha256=sha(__file__),images=len(results),
        main_extractor_signature=signature,results=results,
        equivalence='Identical vf.image_to_ink, stroke_ink_uncropped, stroke_unit, ink_features and intensity_features operations, no neural features for auxiliary-only pages.')
    (OUT/'handcrafted_cache_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Auxiliary-only cache warming complete:',len(results),'pages',flush=True)


if __name__=='__main__':main()
