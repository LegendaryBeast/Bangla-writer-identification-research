#!/usr/bin/env python3
"""Additional, explicitly post-review checks of the BanglaHand ID experiment.

The original 2026-10-29 results are read-only inputs. Identification scores use
revised_protocol.iter_score_cells: its scaler is fitted separately to the selected
gallery references. Every setting gives equal weight to each writer, and reported
overall effects give equal weight to all 120 settings. Bootstrap intervals resample
writers (stratified by hand), conditional on the fixed gallery and fitted evidence;
they do not claim uncertainty for a wholly new gallery or refitted predictors.

Run from the project root:
  python code/run_revision_checks.py --both-controls --jobs 8

Outputs are named checks_* and written only to the new revision result directory.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import expit, log_softmax, logsumexp
from scipy.stats import hypergeom

import run_wid_handedness as wh
import run_both_control as bc
import revised_protocol as rp


SEED = 20261108
KEY = ["rep", "query", "n", "k"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def writer_average(x, wi, nw):
    x = np.asarray(x, dtype=float)
    return np.bincount(wi, weights=x, minlength=nw) / np.bincount(wi, minlength=nw)


def safe_ratio(a, b):
    return np.divide(a, b, out=np.full_like(np.asarray(a, float), np.nan), where=np.asarray(b) > 0)


def inference(d, boots, flips):
    d = np.asarray(d, float)
    obs = float(d.mean())
    lo, hi = np.quantile(d[boots].mean(1), [.025, .975])
    p = (1 + np.sum(np.abs(flips @ d / len(d)) >= abs(obs) - 1e-12)) / (len(flips) + 1)
    return dict(estimate=obs, ci_lo=float(lo), ci_hi=float(hi), p_signflip=float(p))


def macro_temperature(S, t, wi, nw):
    """ID-only temperature; loss gives equal weight to each training writer."""
    losses = np.stack([-log_softmax(S / T, axis=1)[np.arange(len(t)), t] for T in wh.T_GRID], 1)
    wm = np.stack([losses[wi == w].mean(0) for w in range(nw)])
    best = np.argmin(wm.sum(0)[None, :] - wm, axis=1)
    return wh.T_GRID[best[wi]], wh.T_GRID[best]


def known_match(ell, ghand):
    lp = np.stack([-np.logaddexp(0, ell), -np.logaddexp(0, -ell)], 1)
    return lp[:, ghand]


def both_match(ell, gl):
    lq = np.stack([-np.logaddexp(0, ell), -np.logaddexp(0, -ell)], 1)
    lg = np.stack([-np.logaddexp(0, gl), -np.logaddexp(0, -gl)], 2)
    return logsumexp(lq[:, None, :] + lg, axis=2)


def gallery_evidence(U, e, writers, configs, refsets, shape):
    gl = np.empty(shape, float)
    for c, refs in enumerate(refsets):
        vals = [np.mean([e.loc[f"{w}_para{p:02d}_n1_s0"] for p in refs]) for w in writers]
        gl[configs == c] = vals
    return gl


def unpack_cell(raw):
    """Support the documented dict form and older (metadata, arrays) form."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[1], dict):
        return {**raw[0], **raw[1]}
    raise TypeError("Unexpected revised_protocol cell format")


def partition_controls(S, t, wi, labels, nw):
    """Oracle random partitions with tied top rank handled exactly as wh.ranks.

    Each draw uniformly assigns 21 of 42 writers to group 1. True query membership
    is supplied. A query succeeds iff none of its strictly higher-scoring writers
    has the same assigned membership. uint64 masks avoid a Q x draws x W tensor.
    """
    v = S[np.arange(len(t)), t][:, None]
    higher = (S > v) | ((S == v) & (np.arange(nw)[None, :] < t[:, None]))
    bits = np.left_shift(np.uint64(1), np.arange(nw, dtype=np.uint64))
    masks = (higher.astype(np.uint64) * bits[None, :]).sum(1, dtype=np.uint64)
    draw_masks = (labels.astype(np.uint64) * bits[None, :]).sum(1, dtype=np.uint64)
    all_bits = np.uint64((1 << nw) - 1)
    weight = 1 / (nw * np.bincount(wi, minlength=nw)[wi])
    draws = np.zeros(len(labels))
    for start in range(0, len(t), 1024):
        stop = min(start + 1024, len(t))
        same_group = np.where(labels[:, t[start:stop]], draw_masks[:, None],
                              (all_bits ^ draw_masks)[:, None])
        correct = (same_group & masks[None, start:stop]) == 0
        draws += correct @ weight[start:stop]
    # Conditional on true-writer membership, all outrankers must be among the
    # 21 opposite-group candidates out of 41 other writers.
    nr = higher.sum(1)
    exact = hypergeom.pmf(nr, nw - 1, nw // 2, nr)
    return draws, writer_average(exact, wi, nw)


def calibration(U, H, writers, boots, out):
    rows, bins, writer_rows = [], [], []
    nw = len(writers)
    widx = {w: i for i, w in enumerate(writers)}
    for hm, E in H.groupby("hand_model", sort=True):
        e = E.set_index("unit_id").logit
        for n in wh.NS:
            for part in ("shared", "own"):
                m = (U.n == n) & U.valid & ((U.para <= 6) if part == "shared" else (U.para >= 7))
                D = U[m]
                ell = e.reindex(D.unit_id).to_numpy()
                y = D.label.to_numpy()
                p = expit(ell)
                wi = D.writer_id.map(widx).to_numpy()
                values = dict(accuracy=((ell > 0) == y), brier=(p-y)**2,
                              log_loss=np.logaddexp(0, ell) - y * ell)
                row = dict(hand_model=hm, n=n, part=part, n_units=len(D), n_writers=nw)
                for name, v in values.items():
                    wm = writer_average(v, wi, nw)
                    lo, hi = np.quantile(wm[boots].mean(1), [.025, .975])
                    row.update({name: float(wm.mean()), name+"_ci_lo": float(lo), name+"_ci_hi": float(hi)})
                    for w, val in zip(writers, wm):
                        writer_rows.append(dict(hand_model=hm, n=n, part=part, writer_id=w, metric=name, value=val))
                row.update(brier_uninformative=.25, log_loss_uninformative=float(np.log(2)))
                rows.append(row)
                edges = np.linspace(0, 1, 6)
                bi = np.clip(np.digitize(p, edges, right=False)-1, 0, 4)
                weights = 1 / (nw * np.bincount(wi, minlength=nw)[wi])
                for b in range(5):
                    hit = bi == b
                    den_w = writer_average(hit, wi, nw)
                    num_w = writer_average(hit*y, wi, nw)
                    freq_boot = safe_ratio(num_w[boots].mean(1), den_w[boots].mean(1))
                    finite = freq_boot[np.isfinite(freq_boot)]
                    lo, hi = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
                    den = weights[hit].sum()
                    bins.append(dict(hand_model=hm, n=n, part=part, bin=b, lower=edges[b], upper=edges[b+1],
                                     n_units=int(hit.sum()), n_writers=int(len(np.unique(wi[hit]))),
                                     weight=float(den), mean_probability=float(np.sum(weights[hit]*p[hit])/den) if den else np.nan,
                                     observed_fraction=float(np.sum(weights[hit]*y[hit])/den) if den else np.nan,
                                     fraction_ci_lo=lo, fraction_ci_hi=hi))
    pd.DataFrame(rows).to_csv(out/"checks_calibration.csv", index=False)
    pd.DataFrame(bins).to_csv(out/"checks_reliability_bins.csv", index=False)
    pd.DataFrame(writer_rows).to_csv(out/"checks_calibration_writer.csv", index=False)
    return pd.DataFrame(rows)


def control_evidence(U, F, real, out, jobs, n_pseudo, n_random):
    """Cache all refitted pseudo scores. No files in the original run are changed."""
    cache = out/"checks_both_evidence.npz"
    meta = out/"checks_both_evidence_provenance.json"
    desired = dict(seed=SEED+7, n_pseudo=n_pseudo, n_random=n_random,
                   unit_order_sha256=hashlib.sha256("\n".join(U.unit_id).encode()).hexdigest(),
                   real_scores_sha256=hashlib.sha256(real.to_numpy().tobytes()).hexdigest())
    if cache.exists() and meta.exists():
        saved = json.loads(meta.read_text())
        if all(saved.get(k) == v for k, v in desired.items()):
            z = np.load(cache)
            return {(k.split("__")[0], int(k.split("__")[1])): pd.Series(z[k], index=U.unit_id) for k in z.files}
    rng = np.random.default_rng(SEED+7)
    writers = sorted(U.writer_id.unique())
    L, R = [w for w in writers if w.startswith("L")], [w for w in writers if w.startswith("R")]
    evid, groups = {}, []
    for b in range(n_pseudo):
        nl = len(L)//2 + b % 2 * (len(L)%2)
        g1 = set(rng.choice(L, nl, replace=False)) | set(rng.choice(R, len(writers)//2-nl, replace=False))
        labels = {w: int(w in g1) for w in writers}
        groups.append(sorted(g1))
        e = bc.pseudo_logits(U, F, labels, jobs)
        evid[("pseudo", b)] = e
        # Same descriptive spread matching as the original controls; this uses
        # the full cohort score spread and is not a deployable calibration fit.
        evid[("pseudo_scaled", b)] = bc.match_spread(U, e.to_numpy(), real)
        print(f"both control pseudo {b+1}/{n_pseudo} fitted", flush=True)
    for b in range(n_random):
        evid[("random", b)] = bc.random_logits(U, F, real, rng)
    np.savez_compressed(cache, **{f"{kind}__{b}": e.to_numpy() for (kind,b),e in evid.items()})
    meta.write_text(json.dumps({**desired, "pseudo_groups":groups,
                               "description":"Writer-cross-fitted descriptors; other gallery folds can contain queried writer training data. Scaled/random controls match full-cohort score spread."},indent=2)+"\n")
    return evid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="results/wid_handedness_2026-10-29")
    ap.add_argument("--out", default="results/wid_revision_2026-11-08")
    ap.add_argument("--handedness", help="Defaults to original writer-independent predictions")
    ap.add_argument("--partitions",type=int,default=2000)
    ap.add_argument("--bootstrap",type=int,default=10000)
    ap.add_argument("--permutations",type=int,default=9999)
    ap.add_argument("--both-controls",action="store_true")
    ap.add_argument("--jobs",type=int,default=8)
    ap.add_argument("--pseudo",type=int,default=20)
    ap.add_argument("--random",type=int,default=100)
    a = ap.parse_args()
    t0 = time.time()
    run,out=Path(a.run),Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    upath=run/"features"/"units.csv"; fpath=run/"features"/"unit_features.npz"
    hpath=Path(a.handedness) if a.handedness else run/"handedness_lowo.csv.gz"
    U=pd.read_csv(upath)
    z=np.load(fpath,allow_pickle=True)
    assert np.array_equal(z["unit_id"].astype(str),U.unit_id.to_numpy())
    F={k:z[k] for k in z.files if k!="unit_id"}
    H=pd.read_csv(hpath)
    writers=sorted(U.writer_id.unique()); nw=len(writers)
    assert nw==42
    widx={w:i for i,w in enumerate(writers)}
    gh=np.array([int(w.startswith("L")) for w in writers])
    rng=np.random.default_rng(SEED)
    boots=np.concatenate([rng.choice(np.flatnonzero(gh==h),(a.bootstrap,int((gh==h).sum()))) for h in (0,1)],1)
    flips=rng.choice([-1,1],(a.permutations,nw))
    partition_labels=np.zeros((a.partitions,nw),bool)
    for row in partition_labels:
        row[rng.choice(nw,nw//2,replace=False)]=True
    np.savez_compressed(out/"checks_random_partitions.npz", labels=partition_labels,writer_id=np.array(writers))
    A=calibration(U,H,writers,boots,out)
    hp={hm:D.set_index("unit_id").logit for hm,D in H.groupby("hand_model")}
    ce=control_evidence(U,F,hp["hc183"],out,a.jobs,a.pseudo,a.random) if a.both_controls else {}
    identity,iwriter,mainwriter,oracle,oracle_draws,control_writer,temperature,cross=[],[],[],[],[],[],[],[]
    for raw in rp.iter_score_cells(U,F):
        c=unpack_cell(raw)
        key={k:c[k] for k in KEY}
        q,S,config,t=np.asarray(c["q"]),np.asarray(c["S"]),np.asarray(c["config"]),np.asarray(c["t"])
        qw=U.writer_id.to_numpy()[q]; wi=np.array([widx[w] for w in qw])
        qhand=gh[t]
        assert list(c["writers"])==writers
        counts=np.bincount(wi,minlength=nw)
        assert np.all(counts>0)
        base=rp.ranks(S,t)==1
        known=rp.ranks(S+10*(gh[None,:]==qhand[:,None]),t)==1
        Tq=rp.fit_temperatures(S,t,qw,c["weights"])
        Tw=np.array([Tq[wi==j][0] for j in range(nw)])
        base_w=writer_average(base,wi,nw); known_w=writer_average(known,wi,nw)
        for w,T in zip(writers,Tw):
            temperature.append({**key,"writer_id":w,"temperature":T})
        rd,expected_w=partition_controls(S,t,wi,partition_labels,nw)
        oracle_draws.append(rd)
        diff=known_w-expected_w
        inf=inference(diff,boots,flips)
        oracle.append({**key,"base":base_w.mean(),"known_hand":known_w.mean(),
                       "random_partition_exact_mean":expected_w.mean(),"random_partition_mc_mean":rd.mean(),
                       "random_partition_q025":np.quantile(rd,.025),"random_partition_q975":np.quantile(rd,.975),
                       "hand_minus_random":inf["estimate"],"difference_ci_lo":inf["ci_lo"],"difference_ci_hi":inf["ci_hi"],
                       "p_control_ge_hand":(1+np.sum(rd>=known_w.mean()))/(len(rd)+1)})
        pred=S.argmax(1); wrong=~base
        for i,w in enumerate(writers):
            m=wi==i
            cross.append({**key,"writer_id":w,"errors":int(wrong[m].sum()),
                          "cross_hand_errors":int((wrong[m] & (gh[pred[m]]!=qhand[m])).sum())})
        evid={hm:e.reindex(U.unit_id.iloc[q]).to_numpy() for hm,e in hp.items()}
        if c["query"]=="own":
            own=U[(U.n==c["n"])&U.valid&(U.para>=7)]
            means=pd.Series(hp["hc183"].reindex(own.unit_id).to_numpy(),index=own.writer_id).groupby(level=0).mean()
            evid["hc183_writerlevel"]=means.reindex(qw).to_numpy()
        results={"none":base_w,"known":known_w,"known_random_expected":expected_w}
        for hm,ell in evid.items():
            hand=(ell>0)==qhand
            filt=rp.ranks(S+10*(gh[None,:]==(ell>0)[:,None]),t)==1
            assert np.array_equal(filt,hand & known),"Exact filter event identity failed"
            ah=writer_average(hand,wi,nw); kk=known_w; ff=writer_average(filt,wi,nw)
            am,km,fm=ah.mean(),kk.mean(),ff.mean()
            cov=fm-am*km
            bootcov=ff[boots].mean(1)-ah[boots].mean(1)*kk[boots].mean(1)
            lo,hi=np.quantile(bootcov,[.025,.975])
            identity.append({**key,"hand_model":hm,"n_query_configs":len(q),"attribute_accuracy":am,
                             "known_top1":km,"base_top1":base_w.mean(),"filter_top1":fm,
                             "product_approximation":am*km,"covariance":cov,"covariance_ci_lo":lo,"covariance_ci_hi":hi,
                             "known_success_given_hand_correct":fm/am if am else np.nan,
                             "known_success_given_hand_wrong":(km-fm)/(1-am) if am<1 else np.nan,
                             "frechet_lower":max(0.,am+km-1),"frechet_upper":min(am,km),
                             "independence_break_even":base_w.mean()/km if km else np.nan,
                             "exact_identity_error":float(fm-(am*km+cov))})
            for i,w in enumerate(writers):
                iwriter.append({**key,"hand_model":hm,"writer_id":w,"attribute_accuracy":ah[i],
                                "known_top1":kk[i],"filter_top1":ff[i],"base_top1":base_w[i]})
            if hm=="hc183":
                querybayes=rp.ranks(S/Tq[:,None]+known_match(ell,gh),t)==1
                gl=gallery_evidence(U,hp["hc183"],writers,config,c["refs"],S.shape)
                both=rp.ranks(S/Tq[:,None]+both_match(ell,gl),t)==1
                results.update(query_bayes=writer_average(querybayes,wi,nw),query_filter=ff,
                               both_bayes=writer_average(both,wi,nw))
                if ce:
                    for (kind,b),e in ce.items():
                        ec=e.reindex(U.unit_id.iloc[q]).to_numpy()
                        gc=gallery_evidence(U,e,writers,config,c["refs"],S.shape)
                        corr=rp.ranks(S/Tq[:,None]+both_match(ec,gc),t)==1
                        cw=writer_average(corr,wi,nw)
                        for i,w in enumerate(writers):
                            control_writer.append({**key,"control":kind,"instance":b,"writer_id":w,
                                                   "top1":cw[i],"base_top1":base_w[i],"hand_top1":results["both_bayes"][i]})
        for meth,vals in results.items():
            for w,v in zip(writers,vals):
                mainwriter.append({**key,"method":meth,"writer_id":w,"top1":v})
        print(f"checks {key} done ({time.time()-t0:.0f}s)",flush=True)
    I=pd.DataFrame(identity); W=pd.DataFrame(mainwriter)
    I.to_csv(out/"checks_filter_identity.csv",index=False)
    pd.DataFrame(iwriter).to_csv(out/"checks_filter_identity_writer.csv",index=False)
    pd.DataFrame(temperature).to_csv(out/"checks_temperature_writer.csv",index=False)
    pd.DataFrame(cross).to_csv(out/"checks_cross_hand_writer.csv",index=False)
    pd.DataFrame(oracle).to_csv(out/"checks_oracle_partitions.csv",index=False)
    od=np.stack(oracle_draws)
    np.savez_compressed(out/"checks_oracle_partition_draws.npz",top1=od,setting=np.array([str({k:r[k] for k in KEY}) for r in oracle]))
    W.to_csv(out/"checks_identification_writer.csv",index=False)
    pivot=W.pivot(index=KEY+["writer_id"],columns="method",values="top1").reset_index()
    summaries=[]
    for method in ["known","query_bayes","query_filter","both_bayes","known_random_expected"]:
        d=(pivot[method]-pivot["none"]).groupby(pivot.writer_id).mean().reindex(writers).to_numpy()
        summaries.append(dict(scope="all_120_settings",contrast=method+"_minus_none",**inference(d,boots,flips)))
    d=(pivot.known-pivot.known_random_expected).groupby(pivot.writer_id).mean().reindex(writers).to_numpy()
    summaries.append(dict(scope="all_120_settings",contrast="known_hand_minus_random_partition_expectation",**inference(d,boots,flips)))
    for qy,n,k,rep in wh.PRIMARY:
        C=pivot[(pivot["query"]==qy)&(pivot.n==n)&(pivot.k==k)&(pivot.rep==rep)].set_index("writer_id").reindex(writers)
        summaries.append(dict(scope=f"primary:{rep}:{qy}:n{n}:k{k}",contrast="query_bayes_minus_none",
                              **inference((C.query_bayes-C["none"]).to_numpy(),boots,flips)))
    S=pd.DataFrame(summaries)
    pm=S.scope.str.startswith("primary:")
    P=S[pm].sort_values("p_signflip")
    S["p_holm"]=np.nan
    S.loc[P.index,"p_holm"]=np.maximum.accumulate([min(1,(len(P)-j)*p) for j,p in enumerate(P.p_signflip)])
    S.to_csv(out/"checks_effect_intervals.csv",index=False)
    if control_writer:
        C=pd.DataFrame(control_writer)
        C.to_csv(out/"checks_both_control_writer.csv.gz",index=False)
        paired=[]
        for kind,D in C.groupby("control"):
            # Equal settings; each instance gets equal weight. Direct paired
            # writer difference uses the mean control, not tests vs zero.
            bywriter=(D.hand_top1-D.top1).groupby(D.writer_id).mean().reindex(writers).to_numpy()
            gains=(D.top1-D.base_top1).groupby(D.instance).mean()
            handgain=(D.hand_top1-D.base_top1).mean()
            paired.append(dict(control=kind,instances=len(gains),hand_mean_gain=handgain,
                               control_mean_gain=gains.mean(),control_q025=gains.quantile(.025),control_q975=gains.quantile(.975),
                               p_control_ge_hand=(1+(gains>=handgain).sum())/(len(gains)+1),
                               **inference(bywriter,boots,flips)))
        pd.DataFrame(paired).to_csv(out/"checks_both_direct_paired.csv",index=False)
    mean_rd=od.mean(0)
    summary=dict(n_settings=len(oracle),n_filter_combinations=len(I),n_writers=nw,
                 filter_identity_max_error=float(I.exact_identity_error.abs().max()),
                 product_mae=float(I.covariance.abs().mean()),product_bias=float(I.covariance.mean()),
                 product_r=float(I[["product_approximation","filter_top1"]].corr().iloc[0,1]),
                 oracle_hand_mean=float(pivot.known.mean()),random_partition_exact_mean=float(pivot.known_random_expected.mean()),
                 random_partition_draw_mean=float(mean_rd.mean()),random_partition_draw_q025=float(np.quantile(mean_rd,.025)),
                 random_partition_draw_q975=float(np.quantile(mean_rd,.975)),
                 p_random_partition_ge_hand=float((1+np.sum(mean_rd>=pivot.known.mean()))/(len(mean_rd)+1)),
                 calibration_hc183_own_paragraph=A[(A.hand_model=="hc183")&(A.n==1)&(A.part=="own")].to_dict("records"),
                 effects=S.astype(object).where(pd.notna(S), None).to_dict("records"))
    (out/"checks_findings.json").write_text(json.dumps(summary,indent=2)+"\n")
    provenance=dict(seed=SEED,parameters=vars(a),seconds=round(time.time()-t0,2),
                    input_sha256={str(p):sha256(p) for p in [upath,fpath,hpath,Path(__file__),Path(rp.__file__),Path(wh.__file__),Path(bc.__file__)]},
                    python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
                    weighting="Equal writer weight within setting, equal setting weight across the 120 settings.",
                    inference="Post-review exploratory analyses. Stratified writer bootstrap/sign flips conditional on the enrolled gallery, fitted evidence and selected-reference score protocol. Gallery composition and fitting are not resampled.",
                    random_partition="Uniform 21/21 writer partitions; query membership and gallery memberships supplied as oracle labels. Draw quantiles measure partition variability, not sampling confidence.",
                    both_scenario="Exploratory descriptor agreement. Gallery writer-cross-fitted predictors can include the querywriter in their training cohort; no attribution to handedness specifically is established.")
    (out/"checks_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    lines=["# Post-review checks (2026-11-08)","",provenance["weighting"],"",provenance["inference"],"",provenance["random_partition"],"",
           f"Exact filter identity error: {summary['filter_identity_max_error']:.3g}.",
           f"Independence approximation: MAE {100*summary['product_mae']:.2f} points; bias {100*summary['product_bias']:.2f} points; pooled r {summary['product_r']:.4f}.",
           f"Oracle known hand mean {100*summary['oracle_hand_mean']:.2f}%; expected random balanced partition {100*summary['random_partition_exact_mean']:.2f}%.","",
           S.to_string(index=False),"",provenance["both_scenario"],""]
    (out/"checks_findings.md").write_text("\n".join(lines))
    print(json.dumps(summary,indent=2),flush=True)
    print(f"Finished in {(time.time()-t0)/60:.1f} min",flush=True)


if __name__=="__main__":
    main()
