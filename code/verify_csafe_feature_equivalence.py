#!/usr/bin/env python3
"""Compare the CSAFE adapter with original n=1 extraction on fixed samples."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.models as tvm

import wid_features as vf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/csafe_replication_2026-11-08'


def main():
    U=pd.read_csv(OUT/'units.csv');A=pd.read_csv(OUT/'images.csv')
    F=np.load(OUT/'features.npz');torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT/'dataset/model_cache'))
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    r=tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1);r.fc=torch.nn.Identity()
    v=tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1);v.heads=torch.nn.Identity()
    nets={'resnet18':r.eval().to(device),'vit_b_16':v.eval().to(device),
        'efficientnet_b0':torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten()).eval().to(device)}
    indices=[int(U[(U.label==h)&U.gallery].index[0]) for h in (0,1)]
    warm=json.loads((OUT/'handcrafted_cache_provenance.json').read_text())
    warmed=next(name for name,status in warm['results'] if status=='computed')
    indices.append(int(U[U.image_id==warmed].index[0]))
    result=[]
    for i in indices:
        original=vf.image_units(vf.load_gray(ROOT/'dataset/csafe'/A.iloc[i].relative_path),nets,device,fractions=(1,))[0]
        reps=('hc183','stroke','global_intensity')+ (('resnet18','vit_b_16') if U.iloc[i].deep_required else ())
        errors={rep:float(np.max(np.abs(np.asarray(original[rep],np.float32)-F[rep][i]))) for rep in reps}
        for rep in reps:
            assert np.allclose(np.asarray(original[rep],np.float32),F[rep][i],rtol=1e-5,atol=1e-6),(i,rep,errors)
        result.append(dict(image_id=U.iloc[i].image_id,neural_comparison=bool(U.iloc[i].deep_required),max_absolute_difference=errors))
        print(result[-1],flush=True)
    (OUT/'feature_equivalence.json').write_text(json.dumps(dict(passed=True,device=str(device),
        samples=result,tolerance=dict(rtol=1e-5,atol=1e-6),
        scope='Original image_units n=1 followed by original float32 storage; one gallery sample per hand plus one prefetched auxiliary page. Unused EfficientNet stand-in excluded from comparison.'),indent=2)+'\n')


if __name__=='__main__':main()
