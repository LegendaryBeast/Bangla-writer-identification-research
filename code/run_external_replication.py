#!/usr/bin/env python3
"""Frozen external MPHD replication with writer-nested attribute fusion.

Each text pair belongs to one scan. This is independent-dataset evaluation,
not separate-page/session validation or cross-dataset model transfer.
"""
import argparse
import hashlib
import json
import math
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed, parallel_config
from scipy.special import expit, logsumexp
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import revised_protocol as rp
import run_wid_handedness as wh
from run_nested_fusion import choose_lambda

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/external_replication_2026-11-08"
SEED = 20261108
REPS = ("resnet18", "vit_b_16", "hc183", "stroke")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_manifest(train, groups, labels):
    tr = np.flatnonzero(train)
    rows = []
    for a,b in StratifiedGroupKFold(5, shuffle=True, random_state=2031).split(tr, labels[tr], groups[tr]):
        left, right = sorted(set(groups[tr[a]])), sorted(set(groups[tr[b]]))
        assert not set(left) & set(right)
        rows.append(dict(train_writers=left, calibration_writers=right))
    return rows


def outer_attribute(outer, writers, X, y, groups, text, reference, query):
    ref = text == reference
    cov = np.zeros((len(y), 1))  # unused: primary HC-183 is unadjusted
    train = ref & np.isin(groups, writers) & (groups != outer)
    test = np.flatnonzero((text == query) & (groups == outer))
    assert len(test) == 1 and outer not in set(groups[train])
    _, raw, prediction = wh.lowo_writer(X, y, cov, groups, train, test, False, 2031)
    dev = np.array([w for w in writers if w != outer])
    labels = np.array([y[np.flatnonzero(groups == w)[0]] for w in dev])
    logits = np.full(len(writers), np.nan)
    index = {w:i for i,w in enumerate(writers)}
    manifest = dict(outer_writer=outer, final_attribute_train_writers=sorted(set(groups[train])),
        final_calibration_folds=calibration_manifest(train, groups, y), development=[])
    for fold, (a,b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2031).split(dev, labels)):
        trainw, heldw = dev[a], dev[b]
        assert outer not in set(trainw) | set(heldw) and not set(trainw) & set(heldw)
        mask = ref & np.isin(groups, trainw)
        idx = np.flatnonzero((text == query) & np.isin(groups, heldw))
        _, _, ell = wh.lowo_writer(X, y, cov, groups, mask, idx, False, 2031)
        for writer, value in zip(groups[idx], ell):
            logits[index[writer]] = value
        inner = calibration_manifest(mask, groups, y)
        for row in inner:
            assert not (set(row["train_writers"]) | set(row["calibration_writers"])) & (set(heldw) | {outer})
        manifest["development"].append(dict(fold=fold, train_writers=list(trainw),
            held_writers=list(heldw), calibration_folds=inner))
    assert np.isfinite(logits[np.array(writers) != outer]).all()
    assert np.isnan(logits[index[outer]])
    return float(raw[0]), float(prediction[0]), logits, manifest


def attribute_predictions(U, F, writers, reference, query, cache, jobs, signature):
    key = hashlib.sha256(json.dumps(dict(writers=writers, reference=reference,
        query=query, signature=signature), sort_keys=True).encode()).hexdigest()
    arrays, meta = cache / (key + ".npz"), cache / (key + ".json")
    if arrays.exists() and meta.exists():
        with np.load(arrays) as z:
            raw, outer, development = z["raw"], z["outer"], z["development"]
        manifest = json.loads(meta.read_text())
    else:
        with parallel_config(backend="loky", n_jobs=jobs, inner_max_num_threads=1):
            fitted = Parallel()(delayed(outer_attribute)(w, writers, F["hc183"],
                U.label.to_numpy(), U.writer_id.to_numpy(), U.text.to_numpy(), reference, query) for w in writers)
        raw = np.array([r[0] for r in fitted])
        outer = np.array([r[1] for r in fitted])
        development = np.stack([r[2] for r in fitted])
        manifest = [r[3] for r in fitted]
        np.savez_compressed(arrays, raw=raw, outer=outer, development=development)
        meta.write_text(json.dumps(manifest, indent=2) + "\n")
    assert np.isnan(np.diag(development)).all()
    assert np.isfinite(development[~np.eye(len(writers), dtype=bool)]).all()
    return raw, outer, development, key, manifest


def conditional_inference(delta, labels, rng):
    delta = np.asarray(delta, float)
    draws = np.zeros(10000)
    for hand in (0,1):
        values = delta[labels == hand]
        draws += .5 * values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    lo, hi = np.quantile(draws, [.025, .975])
    flips = rng.choice([-1,1], size=(9999, len(delta)))
    observed = delta.mean()
    p = (1 + np.sum(np.abs(flips @ delta / len(delta)) >= abs(observed)-1e-12))/10000
    return dict(delta_top1=observed, ci_lo=lo, ci_hi=hi, p_signflip=p)


def gallery_definitions(U):
    writers = sorted(U.writer_id.unique())
    labels = U.groupby("writer_id").label.first()
    cohorts = [("full_balanced", writers)]
    rng = np.random.default_rng(SEED)
    subsets = []
    for i in range(20):
        subset = sorted(w for h in (0,1) for w in rng.choice(labels.index[labels==h], 21, replace=False))
        subsets.append(dict(name=f"balanced42_{i:02d}", writers=subset))
    cohorts.extend((r["name"], r["writers"]) for r in subsets)
    (OUT/"gallery_manifest.json").write_text(json.dumps(dict(seed=SEED, full=writers,
        matched_primary_descriptive_gallery=subsets[0]["name"], sensitivity=subsets), indent=2)+"\n")
    return cohorts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--sensitivity-galleries", type=int, default=20, choices=range(21))
    args = parser.parse_args()
    start = time.time()
    U = pd.read_csv(OUT / "units.csv")
    if not U.valid.all():
        raise ValueError("Feature-invalid image: record and resolve eligibility before any evaluation")
    assert not U.duplicated(["writer_id", "text"]).any()
    assert (U.groupby("writer_id").size() == 2).all()
    assert U.image_sha256.is_unique
    with np.load(OUT / "features.npz") as z:
        assert np.array_equal(z["image_id"], U.image_id)
        F = {rep:z[rep].astype(float) for rep in REPS}
    sources = [Path(__file__), Path(wh.__file__), Path(rp.__file__), ROOT/"code/run_nested_fusion.py",
        OUT/"protocol_v1.json", OUT/"units.csv", OUT/"features.npz"]
    signature = {str(p.relative_to(ROOT)):sha(p) for p in sources}
    cache = OUT/"attribute_cache";cache.mkdir(exist_ok=True)
    cohorts = gallery_definitions(U)
    cohorts = [cohorts[0]] + cohorts[1:1+args.sensitivity_galleries]
    summaries, outcomes, parameters, tests, diagnostics, fitting = [], [], [], [], [], []
    loss_profiles = []
    index = {(r.writer_id, r.text):i for i,r in enumerate(U.itertuples())}
    rng = np.random.default_rng(SEED+1)
    for ci,(cohort,writers) in enumerate(cohorts):
        labels = U.groupby("writer_id").label.first().reindex(writers).to_numpy()
        assert np.sum(labels==0)==np.sum(labels==1)
        size = len(writers)
        for reference,query in [(1,2),(2,1)]:
            direction = f"T{reference}_to_T{query}"
            ri = np.array([index[(w,reference)] for w in writers])
            qi = np.array([index[(w,query)] for w in writers])
            assert not set(ri) & set(qi)
            assert np.array_equal(U.parent_form_id.to_numpy()[ri], U.parent_form_id.to_numpy()[qi])
            raw, ell, development, cache_key, manifest = attribute_predictions(
                U,F,writers,reference,query,cache,args.jobs,signature)
            fitting.append(dict(cohort=cohort,direction=direction,cache_key=cache_key,
                writers=writers,reference_image_ids=U.image_id.to_numpy()[ri].tolist(),
                query_image_ids=U.image_id.to_numpy()[qi].tolist()))
            M = rp.match_logprob(ell,labels)
            truth = np.arange(size)
            weights = np.full(size,1/size)
            qw = np.array(writers)
            probability = expit(ell)
            hand_acc = float(np.mean((ell>0)==labels))
            hand_brier = float(np.mean((probability-labels)**2))
            hand_nll = float(np.mean(np.logaddexp(0,ell)-labels*ell))
            for rep in REPS:
                X = F[rep]
                scaler = StandardScaler().fit(X[ri])
                zr,zq = scaler.transform(X[ri]),scaler.transform(X[qi])
                zr /= np.maximum(np.linalg.norm(zr,axis=1,keepdims=True),1e-12)
                zq /= np.maximum(np.linalg.norm(zq,axis=1,keepdims=True),1e-12)
                S = zq @ zr.T
                T = rp.fit_temperatures(S,truth,qw,weights)
                lambdas = np.zeros(size)
                for j,w in enumerate(writers):
                    lambdas[j], losses = choose_lambda(S,truth,qw,weights,development[j],w,T[j],labels)
                    parameters.append(dict(cohort=cohort,direction=direction,rep=rep,writer_id=w,
                        temperature=T[j],fusion_weight=lambdas[j],raw_hand_score=raw[j],hand_logit=ell[j]))
                    if cohort in ("full_balanced","balanced42_00"):
                        for lam,loss in zip(wh.LAM_GRID,losses):
                            loss_profiles.append(dict(cohort=cohort,direction=direction,rep=rep,writer_id=w,
                                fusion_weight=lam,development_nll=loss))
                scores = dict(identity_only=S,
                    oracle_known_hand=S+10*(labels[:,None]==labels[None,:]),
                    predicted_hard_filter=S+10*((ell>0)[:,None]==labels[None,:]),
                    fixed_fusion_lambda1=S/T[:,None]+M,
                    nested_fusion=S/T[:,None]+lambdas[:,None]*M)
                ranks = {method:rp.ranks(score,truth) for method,score in scores.items()}
                base = ranks["identity_only"]==1
                oracle = ranks["oracle_known_hand"]==1
                hand_ok = (ell>0)==labels
                filtered = ranks["predicted_hard_filter"]==1
                assert np.array_equal(filtered, hand_ok & oracle)
                # True writer must win its own hand group; fusion cannot reorder that group.
                same = labels[:,None]==labels[None,:]
                own = np.argmax(np.where(same,S,-np.inf),axis=1)
                opposite = np.argmax(np.where(~same,S,-np.inf),axis=1)
                margin = (S[np.arange(size),truth]-S[np.arange(size),opposite])/T
                hand_logodds = (2*labels-1)*ell
                for method,lam in [("fixed_fusion_lambda1",np.ones(size)),("nested_fusion",lambdas)]:
                    boundary = margin + lam*hand_logodds
                    reconstructed = (own==truth) & ((boundary>0) | ((boundary==0)&(truth<opposite)))
                    assert np.array_equal(reconstructed,ranks[method]==1)
                for method,rank in ranks.items():
                    correct = rank==1
                    row = dict(cohort=cohort,direction=direction,rep=rep,method=method,
                        n_writers=size,top1=float(correct.mean()),top5=float((rank<=5).mean()),
                        mrr=float((1/rank).mean()),left_top1=float(correct[labels==1].mean()),
                        right_top1=float(correct[labels==0].mean()),hand_balanced_accuracy=hand_acc,
                        hand_nll=hand_nll,hand_brier=hand_brier,
                        rescued=float((correct&~base).mean()),damaged=float((~correct&base).mean()))
                    summaries.append(row)
                    for j,w in enumerate(writers):
                        outcomes.append(dict(cohort=cohort,direction=direction,rep=rep,method=method,
                            writer_id=w,label=int(labels[j]),rank=int(rank[j]),correct=int(correct[j]),
                            delta_vs_identity=int(correct[j])-int(base[j]),hand_correct=int(hand_ok[j]),
                            identity_margin=margin[j],true_hand_logodds=hand_logodds[j]))
                    if method!="identity_only" and cohort in ("full_balanced","balanced42_00"):
                        primary = (cohort=="full_balanced" and direction=="T1_to_T2"
                            and rep in ("resnet18","vit_b_16") and method in ("fixed_fusion_lambda1","nested_fusion"))
                        tests.append(dict(cohort=cohort,direction=direction,rep=rep,method=method,
                            primary=primary,**conditional_inference(correct.astype(float)-base,labels,rng)))
                baseline_ranks=ranks["identity_only"]
                partition=np.array([math.comb(size-size//2,int(r)-1)/math.comb(size-1,int(r)-1)
                    if int(r)-1 <= size-size//2 else 0. for r in baseline_ranks])
                diagnostics.append(dict(cohort=cohort,direction=direction,rep=rep,n_writers=size,
                    random_balanced_partition_expected_top1=float(partition.mean()),
                    random_partition_expected_gain=float(partition.mean()-base.mean()),
                    oracle_gain=float(oracle.mean()-base.mean()),
                    filtering_exact_joint=float((hand_ok&oracle).mean()),filtering_observed=float(filtered.mean()),
                    filtering_independence_approximation=float(hand_ok.mean()*oracle.mean()),
                    filtering_independence_residual=float(filtered.mean()-hand_ok.mean()*oracle.mean()),
                    top_score_ties=int(np.sum((S==S.max(1,keepdims=True)).sum(1)>1)),
                    fixed_boundary_ties=int(np.sum(margin+hand_logodds==0)),
                    nested_boundary_ties=int(np.sum(margin+lambdas*hand_logodds==0))))
            print(f"{cohort} {direction}: {size} writers; {time.time()-start:.1f}s",flush=True)
            # Incremental outputs are explicitly not a completed scientific run.
            pd.DataFrame(summaries).to_csv(OUT/"id_summary.partial.csv",index=False)
    test = pd.DataFrame(tests)
    test["p_holm_primary"] = np.nan
    selected = test[test.primary].sort_values("p_signflip")
    assert len(selected)==4
    test.loc[selected.index,"p_holm_primary"] = np.maximum.accumulate([
        min(1.,(len(selected)-i)*p) for i,p in enumerate(selected.p_signflip)])
    pd.DataFrame(summaries).to_csv(OUT/"id_summary.csv",index=False)
    pd.DataFrame(outcomes).to_csv(OUT/"writer_outcomes.csv.gz",index=False)
    pd.DataFrame(parameters).to_csv(OUT/"selected_parameters.csv",index=False)
    pd.DataFrame(loss_profiles).to_csv(OUT/"lambda_loss_profiles.csv.gz",index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/"diagnostics.csv",index=False)
    test.to_csv(OUT/"paired_tests.csv",index=False)
    (OUT/"fitting_manifest.json").write_text(json.dumps(fitting,indent=2)+"\n")
    report = dict(completed=True,protocol_sha256=sha(OUT/"protocol_v1.json"),input_code_sha256=signature,
        python=platform.python_version(),sklearn=sklearn.__version__,seconds=time.time()-start,
        galleries=len(cohorts),sensitivity_galleries=args.sensitivity_galleries,
        representations=list(REPS),settings=len(cohorts)*2*len(REPS),primary_tests=4,
        query_ref_image_overlap=False,query_ref_parent_form_overlap=True,
        interpretation="Independent-dataset replication with within-dataset refitting. Same-form paragraph pairs; no new-session/separate-page validation. CIs conditional on fitted predictions and enrolled gallery; overlapping galleries are dependent.",
        inferential_scope="Four prespecified comparisons at full balanced gallery, T1->T2, ResNet18/ViT; all other comparisons descriptive.")
    (OUT/"run_provenance.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
