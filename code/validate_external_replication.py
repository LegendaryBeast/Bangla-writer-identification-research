#!/usr/bin/env python3
"""Audit external fitting exclusions, input hashes and reported outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/external_replication_2026-11-08"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-only",action="store_true",help="Audit saved numerical outputs without downloading raw images/checkpoints")
    args=parser.parse_args()
    run = json.loads((OUT/"run_provenance.json").read_text())
    assert run["completed"] and run["galleries"]==21 and run["settings"]==168
    for name,digest in run["input_code_sha256"].items():
        assert sha(ROOT/name)==digest, ("Input changed",name)
    features=json.loads((OUT/"feature_provenance.json").read_text())
    assert not features["invalid_images"]
    for name,digest in features["code"].items():
        assert sha(ROOT/"code"/name)==digest
    if not args.saved_only:
        for info in features["checkpoints"].values():
            assert sha(ROOT/"dataset/model_cache/checkpoints"/info["file"])==info["sha256"]
    U=pd.read_csv(OUT/"units.csv")
    A=pd.read_csv(OUT/"images.csv")
    assert np.array_equal(U.image_id,A.image_id) and U.image_sha256.is_unique
    if not args.saved_only:
        for row in A.itertuples():
            assert sha(ROOT/"dataset/mphd"/row.relative_path)==U.loc[U.image_id==row.image_id,"image_sha256"].iloc[0]
    manifest=json.loads((OUT/"fitting_manifest.json").read_text())
    n_outer,n_models,n_calibration=0,0,0
    for cell in manifest:
        writers=set(cell["writers"])
        refs=U[U.image_id.isin(cell["reference_image_ids"])]
        queries=U[U.image_id.isin(cell["query_image_ids"])]
        assert set(refs.writer_id)==writers==set(queries.writer_id)
        assert not set(refs.image_id)&set(queries.image_id)
        assert set(refs.parent_form_id)==set(queries.parent_form_id)
        nested=json.loads((OUT/"attribute_cache"/(cell["cache_key"]+".json")).read_text())
        assert {r["outer_writer"] for r in nested}==writers
        for row in nested:
            n_outer+=1
            outer=row["outer_writer"]
            train=set(row["final_attribute_train_writers"])
            assert train==writers-{outer}
            stages=[(train,row["final_calibration_folds"],set())]
            all_held=[]
            for dev in row["development"]:
                dt,dh=set(dev["train_writers"]),set(dev["held_writers"])
                assert not dt&dh and dt|dh==writers-{outer}
                all_held.extend(dh)
                stages.append((dt,dev["calibration_folds"],dh))
            assert len(all_held)==len(writers)-1 and set(all_held)==writers-{outer}
            for expected, folds, forbidden in stages:
                n_models+=1
                held=[]
                for fold in folds:
                    n_calibration+=1
                    a,b=set(fold["train_writers"]),set(fold["calibration_writers"])
                    assert not a&b and a|b==expected
                    assert not (a|b)&(forbidden|{outer})
                    held.extend(b)
                assert len(held)==len(expected) and set(held)==expected
    S=pd.read_csv(OUT/"id_summary.csv")
    W=pd.read_csv(OUT/"writer_outcomes.csv.gz")
    D=pd.read_csv(OUT/"diagnostics.csv")
    T=pd.read_csv(OUT/"paired_tests.csv")
    keys=["cohort","direction","rep","method"]
    by=W.groupby(keys).agg(top1=("correct","mean"),n_writers=("writer_id","nunique"))
    paired=S.set_index(keys).join(by,rsuffix="_recomputed")
    assert len(S)==840 and len(D)==168
    assert np.allclose(paired.top1,paired.top1_recomputed,atol=1e-14)
    assert np.array_equal(paired.n_writers,paired.n_writers_recomputed)
    assert np.allclose(D.filtering_exact_joint,D.filtering_observed,atol=1e-14)
    assert (D.random_balanced_partition_expected_top1>=S[S.method=="identity_only"].top1.to_numpy()-1e-14).all()
    base=W[W.method=="identity_only"].set_index(["cohort","direction","rep","writer_id"]).correct
    for method in W.method.unique():
        rows=W[W.method==method].set_index(["cohort","direction","rep","writer_id"])
        assert np.array_equal(rows.delta_vs_identity,rows.correct-base.reindex(rows.index))
    primary=T[T.primary].sort_values("p_signflip")
    assert len(primary)==4 and (primary.cohort=="full_balanced").all() and (primary.direction=="T1_to_T2").all()
    adjusted=np.maximum.accumulate([min(1.,(4-i)*p) for i,p in enumerate(primary.p_signflip)])
    assert np.allclose(primary.p_holm_primary,adjusted)
    assert ((T.ci_lo<=T.delta_top1+1e-14)&(T.ci_hi>=T.delta_top1-1e-14)).all()
    result=dict(passed=True,outer_writer_fits_audited=n_outer,attribute_models_audited=n_models,
        inner_calibration_folds_audited=n_calibration,settings=168,summary_rows=len(S),
        saved_writer_outcomes=len(W),primary_tests=4,input_code_hashes_verified=True,
        raw_image_checkpoint_hashes_verified=not args.saved_only,
        zero_outer_writer_in_attribute_fitting=True,zero_development_writer_in_its_attribute_fitting=True,
        exact_filter_identity_verified=True,margin_reconstruction_verified_during_run=True,
        same_source_form_limitation_preserved=True,full_archive_md5_independently_verified=False)
    (OUT/("validation_saved_only.json" if args.saved_only else "validation.json")).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
