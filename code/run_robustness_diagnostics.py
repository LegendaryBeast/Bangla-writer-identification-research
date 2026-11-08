#!/usr/bin/env python3
"""Post-review calibration-partition, temperature and evaluation-writer diagnostics.

This characterizes sensitivity of fixed fitted scores on the observed gallery.
Additional seeds change grouped Platt calibration folds only. They are not new
datasets, independent writer samples, neural training replicates, or external
validation. Evaluation-writer omission leaves the gallery and all models fixed.
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
from joblib import Parallel, delayed
from scipy.special import expit, logsumexp
from sklearn.model_selection import StratifiedGroupKFold

import revised_protocol as rp
import run_wid_handedness as wh

KEY = ["rep", "query", "n", "k"]
SEEDS = (2031, 2032, 2033, 2034)
MULTIPLIERS = (.5, 1., 2.)
NEAR_OPTIMAL_NATS = .01
BOOTSTRAP_SEED = 20261109


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    def clean(v):
        if isinstance(v, dict):
            return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v, (list,tuple)):
            return [clean(x) for x in v]
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (float,np.floating)):
            return float(v) if np.isfinite(v) else None
        if isinstance(v, np.bool_):
            return bool(v)
        return v
    Path(path).write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+"\n")


def writer_mean(values, qw, uid, writers):
    return rp.mean_by_writer(values,qw,uid).reindex(writers).to_numpy()


def boot_indices(writers, n_boot):
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    hand=np.array([int(w.startswith("L")) for w in writers])
    return np.concatenate([rng.choice(np.flatnonzero(hand==h),(n_boot,int((hand==h).sum()))) for h in (0,1)],1)


def interval(values, boot):
    v=np.asarray(values,float)
    lo,hi=np.quantile(v[boot].mean(1),[.025,.975])
    return dict(estimate=float(v.mean()),ci_lo=float(lo),ci_hi=float(hi))


def calibrator_variants(U,F,H,out,jobs,source):
    """Exactly original outer held-writer-out fits, changing inner grouping seed."""
    cache=out/"calibration_seed_scores.npz"
    meta=out/"calibration_seed_provenance.json"
    required={str(p):digest(p) for p in [source/"features/units.csv",source/"features/unit_features.npz",
                                        source/"handedness_lowo.csv.gz",Path(wh.__file__)]}
    if cache.exists() and meta.exists():
        M=json.loads(meta.read_text())
        if M.get("input_sha256")==required and M.get("seeds")==list(SEEDS):
            with np.load(cache,allow_pickle=True) as Z:
                assert np.array_equal(Z["unit_id"],U.unit_id.to_numpy())
                scores={s:Z[f"logit_{s}"].copy() for s in SEEDS}
            return scores,M
    base=H[H.hand_model=="hc183"].set_index("unit_id")
    ell={SEEDS[0]:base.logit.reindex(U.unit_id).to_numpy()}
    raw0=base.raw.reindex(U.unit_id).to_numpy()
    X=F["hc183"]
    cov=np.column_stack([U.ink_density.to_numpy(),F["global_intensity"]])
    y=U.label.to_numpy(); g=U.writer_id.to_numpy()
    writers=sorted(U.writer_id.unique())
    manifest=[]; deviations=[]
    for seed in SEEDS[1:]:
        logit=np.full(len(U),np.nan)
        raw=np.full(len(U),np.nan)
        for n in wh.NS:
            valid=((U.n==n)&U.valid).to_numpy()
            tasks=[]
            for w in writers:
                train=valid&(U.para.to_numpy()<=6)&(g!=w)
                test=np.flatnonzero(valid&(g==w))
                tr=np.flatnonzero(train)
                assert w not in set(g[tr])
                folds=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
                for fold,(a,b) in enumerate(folds.split(tr,y[tr],g[tr])):
                    train_w=sorted(set(g[tr[a]])); val_w=sorted(set(g[tr[b]]))
                    assert not (set(train_w)&set(val_w)) and w not in set(train_w+val_w)
                    manifest.append(dict(seed=seed,n=n,outer_writer=w,fold=fold,
                                         train_writers=train_w,validation_writers=val_w))
                tasks.append(delayed(wh.lowo_writer)(X,y,cov,g,train,test,False,seed))
            for idx,r,l in Parallel(n_jobs=jobs)(tasks):
                raw[idx]=r; logit[idx]=l
            print(f"calibration seed={seed} n={n} complete",flush=True)
        m=U.valid.to_numpy()
        deviation=float(np.max(np.abs(raw[m]-raw0[m])))
        assert deviation<1e-8,(seed,deviation)
        deviations.append(dict(seed=seed,maximum_raw_score_change=deviation))
        ell[seed]=logit
    np.savez_compressed(cache,unit_id=U.unit_id.to_numpy(dtype=str),**{f"logit_{s}":e for s,e in ell.items()})
    M=dict(seeds=list(SEEDS),input_sha256=required,raw_score_checks=deviations,
           interpretation="Only inner grouped calibration partitions vary; full outer SVM decision scores are invariant. Seed replicates are not independent writers or independent validations.",
           full_outer_model="Linear HC183 SVM C=.1, training shared units of other41writers; inner5writer-group folds for Platt scaling as in original analysis.")
    write_json(meta,M);write_json(out/"calibration_seed_fold_manifest.json",manifest)
    return ell,M


def calibration_summaries(U,ell,writers,boot,out):
    unit_rows=[];wm_rows=[];summ=[]
    V=U[U.valid].copy();idx=V.index.to_numpy()
    probabilities=np.stack([expit(ell[s][idx]) for s in SEEDS])
    decisions=np.stack([ell[s][idx]>0 for s in SEEDS])
    for j,r in enumerate(V.itertuples()):
        unit_rows.append(dict(unit_id=r.unit_id,writer_id=r.writer_id,n=r.n,
                              part="shared" if r.para<=6 else "own",label=r.label,
                              probability_mean=probabilities[:,j].mean(),probability_sd=probabilities[:,j].std(ddof=1),
                              probability_min=probabilities[:,j].min(),probability_max=probabilities[:,j].max(),
                              n_left_decisions=int(decisions[:,j].sum()),any_disagreement=bool(decisions[:,j].any()!=decisions[:,j].all()),
                              fraction_new_seed_disagreeing_with_original=float((decisions[1:,j]!=decisions[0,j]).mean())))
    for seed,e in ell.items():
        for n in wh.NS:
            for part in ("shared","own"):
                mask=((U.n==n)&U.valid&((U.para<=6) if part=="shared" else (U.para>=7))).to_numpy()
                D=U[mask];ee=e[mask];p=expit(ee);y=D.label.to_numpy()
                vals={"accuracy":(ee>0)==y,"brier":(p-y)**2,"nll":np.logaddexp(0,ee)-y*ee}
                row=dict(seed=seed,n=n,part=part)
                for name,v in vals.items():
                    wm=writer_mean(v,D.writer_id.to_numpy(),D.unit_id.to_numpy(),writers)
                    inf=interval(wm,boot)
                    row.update({name:inf["estimate"],name+"_ci_lo":inf["ci_lo"],name+"_ci_hi":inf["ci_hi"]})
                    for w,value in zip(writers,wm):
                        wm_rows.append(dict(seed=seed,n=n,part=part,writer_id=w,metric=name,value=value))
                summ.append(row)
    D=pd.DataFrame(unit_rows)
    D.to_csv(out/"calibration_prediction_variability_unit.csv",index=False)
    pd.DataFrame(wm_rows).to_csv(out/"calibration_seed_writer_metrics.csv",index=False)
    pd.DataFrame(summ).to_csv(out/"calibration_seed_metrics.csv",index=False)
    S=[]
    for (n,part),G in D.groupby(["n","part"]):
        for metric in ("any_disagreement","probability_sd","fraction_new_seed_disagreeing_with_original"):
            wm=G.groupby("writer_id")[metric].mean().reindex(writers).to_numpy()
            S.append(dict(n=n,part=part,metric=metric,**interval(wm,boot)))
    pd.DataFrame(S).to_csv(out/"calibration_prediction_variability_summary.csv",index=False)
    return pd.DataFrame(summ)


def scopes(D):
    yield "all_120_settings",D
    for query,n,k,rep in wh.PRIMARY:
        yield f"designated:{rep}:{query}:n{n}:k{k}",D[(D["query"]==query)&(D.n==n)&(D.k==k)&(D.rep==rep)]


def effect_summaries(W,writers,boot,out):
    rows=[]
    for seed,D in W.groupby("seed"):
        P=D.pivot(index=KEY+["writer_id"],columns="method",values="top1").reset_index()
        for scope,E in scopes(P):
            for meth in ("known","fixed","filter"):
                delta=(E[meth]-E["none"]).groupby(E.writer_id).mean().reindex(writers).to_numpy()
                rows.append(dict(seed=seed,scope=scope,method=meth,**interval(delta,boot)))
    E=pd.DataFrame(rows);E.to_csv(out/"calibration_seed_identification_effects.csv",index=False)
    ranges=E.groupby(["scope","method"]).agg(seed_min=("estimate","min"),seed_max=("estimate","max"),
                   seed_mean=("estimate","mean"),n_seeds=("seed","size")).reset_index()
    ranges["seed_range"]=ranges.seed_max-ranges.seed_min
    ranges.to_csv(out/"calibration_seed_identification_effect_ranges.csv",index=False)


def influence_summaries(W,writers,boot,out,nested_path=None):
    D=W[W.seed==SEEDS[0]][KEY+["writer_id","method","top1"]].copy()
    nested_included=False
    if nested_path is not None and Path(nested_path).exists():
        N=pd.read_csv(nested_path)
        N=N[N.method=="nested"][KEY+["writer_id","method","top1"]]
        assert len(N)==120*42
        D=pd.concat([D,N],ignore_index=True);nested_included=True
    P=D.pivot(index=KEY+["writer_id"],columns="method",values="top1").reset_index()
    hand=np.array([int(w.startswith("L")) for w in writers])
    influence=[];summary=[];classes=[]
    for scope,E in scopes(P):
        means=E.groupby("writer_id")[[c for c in ("none","known","fixed","filter","nested") if c in E]].mean().reindex(writers)
        for method in means:
            v=means[method].to_numpy();b=means["none"].to_numpy()
            for metric,values in [("top1",v),("gain_vs_none",v-b)]:
                full=.5*values[hand==0].mean()+.5*values[hand==1].mean()
                omissions=[]
                for j,w in enumerate(writers):
                    keep=np.arange(len(writers))!=j
                    omitted=.5*values[(hand==0)&keep].mean()+.5*values[(hand==1)&keep].mean()
                    omissions.append(omitted)
                    influence.append(dict(scope=scope,method=method,metric=metric,omitted_writer=w,
                                          omitted_hand="left" if hand[j] else "right",full_estimate=full,
                                          omitted_estimate=omitted,change_from_full=omitted-full))
                arr=np.array(omissions);imax=int(np.argmax(np.abs(arr-full)))
                summary.append(dict(scope=scope,method=method,metric=metric,full_estimate=full,
                                    omission_min=arr.min(),omission_max=arr.max(),omission_q025=np.quantile(arr,.025),
                                    omission_q975=np.quantile(arr,.975),max_absolute_change=np.abs(arr-full).max(),
                                    most_influential_writer=writers[imax],most_influential_change=arr[imax]-full))
                for label,h in (("all",None),("right",0),("left",1)):
                    if h is None:
                        inf=interval(values,boot)
                    else:
                        selected=values[hand==h]
                        # The stratified bootstrap has right samples first, left
                        # samples second; map selected global indices directly.
                        cols=slice(0,21) if h==0 else slice(21,42)
                        draws=values[boot[:,cols]].mean(1)
                        lo,hi=np.quantile(draws,[.025,.975])
                        inf=dict(estimate=selected.mean(),ci_lo=lo,ci_hi=hi)
                    classes.append(dict(scope=scope,method=method,metric=metric,hand=label,
                                        n_writers=42 if h is None else 21,**inf))
    pd.DataFrame(influence).to_csv(out/"evaluation_writer_omission.csv",index=False)
    pd.DataFrame(summary).to_csv(out/"evaluation_writer_omission_summary.csv",index=False)
    pd.DataFrame(classes).to_csv(out/"class_specific_identification.csv",index=False)
    write_json(out/"evaluation_writer_omission_provenance.json",dict(
        bootstrap_seed=BOOTSTRAP_SEED,n_bootstrap=len(boot),nested_included=nested_included,
        omission="Omit a writer only from evaluation averaging, keeping gallery, fitted models and predictions fixed; give each hand .5 weight among remaining writers.",
        inference="Exploratory influence/sensitivity diagnostic, not leave-one-writer-out model refitting or independent validation; no writer is excluded from reported main results.",
        source_sha256={str(out/"fixed_seed_writer_outcomes.csv"):digest(out/"fixed_seed_writer_outcomes.csv"),
                       **({str(nested_path):digest(nested_path)} if nested_included else {})}))
    return pd.DataFrame(summary)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",default="results/wid_handedness_2026-10-29")
    ap.add_argument("--revision",default="results/wid_revision_2026-11-08")
    ap.add_argument("--out",default="results/wid_fusion_robustness_2026-11-08")
    ap.add_argument("--jobs",type=int,default=6)
    ap.add_argument("--bootstrap",type=int,default=10000)
    ap.add_argument("--nested",help="Optional fully nested writer outcomes, added only to influence/class summaries")
    ap.add_argument("--influence-only",action="store_true")
    args=ap.parse_args();start=time.time()
    src,out=Path(args.source),Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if src.resolve()==out.resolve():raise ValueError("Do not overwrite original results")
    U,F,H=rp.load_inputs(src);writers=sorted(U.writer_id.unique());boot=boot_indices(writers,args.bootstrap)
    if args.influence_only:
        W=pd.read_csv(out/"fixed_seed_writer_outcomes.csv")
        S=influence_summaries(W,writers,boot,out,args.nested)
        print(S[(S.scope=="all_120_settings")&(S.metric=="gain_vs_none")].to_string(index=False));return
    ell,calmeta=calibrator_variants(U,F,H,out,args.jobs,src)
    handmetrics=calibration_summaries(U,ell,writers,boot,out)
    profile=[];selected=[];temperature=[];rows=[];multrows=[];max_T_error=0.
    for i,c in enumerate(rp.iter_score_cells(U,F)):
        key={k:c[k] for k in KEY};S,t,qw,uid=c["S"],c["t"],c["qw"],c["unit_id"]
        losses=np.stack([logsumexp(S/T,axis=1)-S[np.arange(len(t)),t]/T for T in wh.T_GRID],1)
        weighted=losses*c["weights"][:,None];total=weighted.sum(0)
        T=rp.fit_temperatures(S,t,qw,c["weights"])
        bounds=[]
        for w in writers:
            m=qw==w;devcost=total-weighted[m].sum(0);den=1-c["weights"][m].sum()
            dev=devcost/den;best=int(np.argmin(devcost));bestT=wh.T_GRID[best]
            max_T_error=max(max_T_error,float(np.max(np.abs(T[m]-bestT))))
            near=dev<=dev[best]+NEAR_OPTIMAL_NATS
            second=np.sort(dev)[1]-dev[best]
            selected.append(dict(**key,writer_id=w,T=bestT,lower_boundary=best==0,
                                 upper_boundary=best==len(wh.T_GRID)-1,dev_nll=dev[best],second_best_gap=second,
                                 near_optimal_nats=NEAR_OPTIMAL_NATS,near_optimal_count=int(near.sum()),
                                 near_optimal_T_min=wh.T_GRID[near].min(),near_optimal_T_max=wh.T_GRID[near].max()))
            for j,gT in enumerate(wh.T_GRID):
                profile.append(dict(**key,outer_writer=w,T=gT,dev_nll=dev[j],gap_from_selected=dev[j]-dev[best],
                                    selected=j==best,near_optimal=bool(near[j])))
            bounds.append((best==0,best==len(wh.T_GRID)-1,bestT,second,near.sum()))
        B=np.array(bounds)
        temperature.append(dict(**key,n_writers=len(writers),lower_boundary_fraction=B[:,0].mean(),
                                 upper_boundary_fraction=B[:,1].mean(),T_min=B[:,2].min(),T_median=np.median(B[:,2]),
                                 T_max=B[:,2].max(),median_second_best_gap=np.median(B[:,3]),
                                 median_near_optimal_candidates=np.median(B[:,4])))
        base=rp.ranks(S,t);known=rp.ranks(S+10*(c["ghand"][None,:]==c["qhand"][:,None]),t)
        basew=writer_mean(base==1,qw,uid,writers)
        for seed,e in ell.items():
            qell=e[c["q"]]
            M=rp.match_logprob(qell,c["ghand"])
            fused_score=S/T[:,None]+M
            ranks={"none":base,"known":known,"fixed":rp.ranks(fused_score,t),
                   "filter":rp.ranks(S+10*(c["ghand"][None,:]==(qell>0)[:,None]),t)}
            for method,r in ranks.items():
                acc=writer_mean(r==1,qw,uid,writers)
                for w,a in zip(writers,acc):rows.append(dict(**key,seed=seed,method=method,writer_id=w,top1=a))
            if seed==SEEDS[0]:
                for multiplier in MULTIPLIERS:
                    A=S/(T[:,None]*multiplier)+M;r=rp.ranks(A,t)
                    acc=writer_mean(r==1,qw,uid,writers)
                    nll=writer_mean(logsumexp(A,axis=1)-A[np.arange(len(t)),t],qw,uid,writers)
                    for j,w in enumerate(writers):
                        multrows.append(dict(**key,multiplier=multiplier,writer_id=w,top1=acc[j],
                                             base_top1=basew[j],gain=acc[j]-basew[j],fused_nll=nll[j]))
        if (i+1)%20==0:print(f"diagnostics {i+1}/120 ({time.time()-start:.0f}s)",flush=True)
    assert max_T_error==0.,max_T_error
    P=pd.DataFrame(selected);W=pd.DataFrame(rows);MW=pd.DataFrame(multrows)
    pd.DataFrame(profile).to_csv(out/"temperature_validation_grid.csv.gz",index=False)
    P.to_csv(out/"temperature_selected_writer.csv",index=False)
    TS=pd.DataFrame(temperature);TS.to_csv(out/"temperature_setting_summary.csv",index=False)
    W.to_csv(out/"fixed_seed_writer_outcomes.csv",index=False)
    MW.to_csv(out/"temperature_multiplier_writer_outcomes.csv",index=False)
    effect_summaries(W,writers,boot,out)
    effects=[]
    for multiplier,D in MW.groupby("multiplier"):
        for scope,E in scopes(D):
            for metric in ("gain","fused_nll"):
                values=E.groupby("writer_id")[metric].mean().reindex(writers).to_numpy()
                effects.append(dict(multiplier=multiplier,scope=scope,metric=metric,**interval(values,boot)))
    pd.DataFrame(effects).to_csv(out/"temperature_multiplier_effects.csv",index=False)
    influence=influence_summaries(W,writers,boot,out,args.nested)
    original=pd.read_csv(Path(args.revision)/"writer_outcomes.csv")
    compare=[]
    for method,scenario,m in [("none","none","none"),("known","oracle","filter"),("fixed","query","bayes"),("filter","query","filter")]:
        old=original[(original.scenario==scenario)&(original.method==m)&(original.hand_model==("hc183" if scenario=="query" else "-"))]
        new=W[(W.seed==SEEDS[0])&(W.method==method)]
        J=new.merge(old,on=KEY+["writer_id"],suffixes=("_new","_old"),validate="one_to_one")
        assert len(J)==120*42
        error=float(np.max(np.abs(J.top1_new-J.top1_old)));assert error<1e-12
        compare.append(dict(method=method,rows=len(J),maximum_error=error))
    one=MW[MW.multiplier==1].merge(W[(W.seed==SEEDS[0])&(W.method=="fixed")],on=KEY+["writer_id"],suffixes=("_mult","_seed"))
    assert len(one)==5040 and np.allclose(one.top1_mult,one.top1_seed,rtol=0,atol=1e-14)
    provenance=dict(parameters=vars(args),seconds=time.time()-start,seeds=list(SEEDS),multipliers=list(MULTIPLIERS),
                    near_optimal_threshold_nats=NEAR_OPTIMAL_NATS,bootstrap_seed=BOOTSTRAP_SEED,
                    python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
                    interpretation="Post-review sensitivity diagnostics on the same writer cohort/gallery. Calibrator seeds are correlated partition replicates. Writer bootstrap conditions on gallery and fitted models. Omission changes evaluation averaging only; no exclusions or full model refits.",
                    temperature="Original 31point grid, ID-only writer-balanced development NLL excluding scored writer. Fixed multipliers .5,1,2 applied without outcome-driven selection or clipping.",
                    input_sha256={str(p):digest(p) for p in [src/"features/units.csv",src/"features/unit_features.npz",src/"handedness_lowo.csv.gz",Path(__file__),Path(rp.__file__),Path(wh.__file__)]})
    write_json(out/"diagnostics_provenance.json",provenance)
    write_json(out/"diagnostics_validation.json",dict(status="pass",temperature_matches_helper_exactly=max_T_error==0,
                 baseline_seed_matches_main=compare,multiplier_one_matches_fixed=True,calibration_raw_score_checks=calmeta["raw_score_checks"],
                 settings=len(TS),writer_temperatures=len(P),grid_rows=len(profile),seed_writer_rows=len(W),multiplier_writer_rows=len(MW)))
    findings=dict(temperature_boundary_fractions=dict(lower=float(P.lower_boundary.mean()),upper=float(P.upper_boundary.mean())),
                  temperature_range=[float(P["T"].min()),float(P["T"].max())],
                  fixed_fusion_seed_effects=pd.read_csv(out/"calibration_seed_identification_effects.csv").query("scope == 'all_120_settings' and method == 'fixed'").to_dict("records"),
                  temperature_multiplier_effects=pd.read_csv(out/"temperature_multiplier_effects.csv").query("scope == 'all_120_settings'").to_dict("records"),
                  writer_omission=influence[(influence.scope=="all_120_settings")&(influence.metric=="gain_vs_none")].to_dict("records"),
                  own_paragraph_calibration=handmetrics[(handmetrics.n==1)&(handmetrics.part=="own")].to_dict("records"))
    write_json(out/"diagnostics_findings.json",findings)
    lines=["# Robustness diagnostics", "",provenance["interpretation"],"",json.dumps(findings,indent=2),""]
    (out/"diagnostics_findings.md").write_text("\n".join(lines))
    print(json.dumps(findings,indent=2),flush=True)
    print(f"Done in {time.time()-start:.1f}s",flush=True)


if __name__=="__main__":main()
