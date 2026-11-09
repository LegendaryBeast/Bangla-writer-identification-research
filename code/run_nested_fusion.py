#!/usr/bin/env python3
"""Strictly nested, writer-held-out selection of a nonnegative fusion weight.

The original query-writer-held-out attribute predictions supply OUTER TEST
scores only. Development probabilities are regenerated inside each outer writer
exclusion, using five writer folds and another five writer folds for calibration.
The full known-hand gallery is available metadata in this closed-set task.
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
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

import revised_protocol as rp
import run_wid_handedness as wh

KEY=['rep','query','n','k']
LAM_GRID=np.array([0,.1,.2,.3,.5,.7,1,1.5,2,3,5],float)
FOLD_SEED=2031
INFERENCE_SEED=20261110


def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def json_write(p,obj):
    def clean(x):
        if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v) for v in x]
        if isinstance(x,(np.bool_,)):return bool(x)
        if isinstance(x,(np.integer,)):return int(x)
        if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
        return x
    Path(p).write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n')


def fit_development(X,y,cov,groups,train,test,seed):
    _,raw,logit=wh.lowo_writer(X,y,cov,groups,train,test,False,seed)
    return test,logit


def development_scores(U,F,source,out,jobs):
    cache=out/'nested_development_logits.npz';meta=out/'nested_development_provenance.json'
    manifest_path=out/'nested_fold_manifest.json'
    inputs=[source/'features/units.csv',source/'features/unit_features.npz',Path(__file__),Path(wh.__file__)]
    signature={str(p):digest(p) for p in inputs}
    writers=np.array(sorted(U.writer_id.unique()))
    g=U.writer_id.to_numpy();y=U.label.to_numpy();valid=U.valid.to_numpy();ns=U.n.to_numpy()
    para=U.para.to_numpy();cov=np.column_stack([U.ink_density.to_numpy(),F['global_intensity']])
    ell=np.full((len(writers),len(U)),np.nan);completed=[];manifest=[]
    if cache.exists() and meta.exists() and manifest_path.exists():
        m=json.loads(meta.read_text())
        if m.get('input_sha256')==signature and m.get('fold_seed')==FOLD_SEED:
            with np.load(cache,allow_pickle=True) as z:
                assert np.array_equal(z['unit_id'],U.unit_id.to_numpy())
                assert np.array_equal(z['writers'],writers)
                ell=z['logits'].copy()
            completed=m['completed_writers'];manifest=json.loads(manifest_path.read_text())
    for oi,outer in enumerate(writers):
        if outer in completed:continue
        dev=writers[writers!=outer];labels=np.array([int(w.startswith('L')) for w in dev])
        devfolds=list(StratifiedKFold(5,shuffle=True,random_state=FOLD_SEED).split(dev,labels))
        tasks=[]
        for n in wh.NS:
            for fold,(it,iv) in enumerate(devfolds):
                trainw=dev[it];heldw=dev[iv]
                assert outer not in set(trainw)|set(heldw)
                assert not set(trainw)&set(heldw)
                train=valid&(ns==n)&(para<=6)&np.isin(g,trainw)
                test=np.flatnonzero(valid&(ns==n)&np.isin(g,heldw))
                tr=np.flatnonzero(train)
                inner=[]
                for j,(a,b) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=FOLD_SEED).split(tr,y[tr],g[tr])):
                    tw=sorted(set(g[tr[a]]));vw=sorted(set(g[tr[b]]))
                    assert outer not in tw+vw and not (set(tw)|set(vw))&set(heldw)
                    assert not set(tw)&set(vw)
                    inner.append(dict(fold=j,train_writers=tw,validation_writers=vw))
                manifest.append(dict(outer_writer=outer,n=n,development_fold=fold,
                                     model_train_writers=list(trainw),development_validation_writers=list(heldw),
                                     inner_calibration_folds=inner))
                tasks.append(delayed(fit_development)(F['hc183'],y,cov,g,train,test,FOLD_SEED))
        for idx,e in Parallel(n_jobs=jobs)(tasks):ell[oi,idx]=e
        assert np.isfinite(ell[oi,valid&(g!=outer)]).all()
        assert np.isnan(ell[oi,g==outer]).all()
        completed.append(str(outer))
        np.savez_compressed(cache,unit_id=U.unit_id.to_numpy(),writers=writers,logits=ell)
        json_write(manifest_path,manifest)
        json_write(meta,dict(input_sha256=signature,fold_seed=FOLD_SEED,completed_writers=completed,
                            outer_folds=len(writers),development_folds=5,calibration_folds=5,
                            training='Each development SVM/preprocessor/calibrator excludes outer writer and entire development validation fold; shared units only for fitting.',
                            development_train_writers='32 or 33; final outer test attribute model has 41 training writers.',
                            gallery_metadata='All 42 known gallery hands/references remain permitted candidate metadata; outer writer query outcomes and samples never enter development attribute fits or lambda targets.'))
        print(f'nested development {len(completed)}/42 outer writers complete',flush=True)
    return writers,ell,manifest


def choose_lambda(S,t,qw,weights,e,outer,T,ghand):
    """Development NLL only; outer query rows are discarded before computation."""
    keep=qw!=outer
    X=S[keep]/T;tt=t[keep];ee=e[keep];ww=weights[keep];ww=ww/ww.sum()
    logs=np.stack([-np.logaddexp(0.,ee),-np.logaddexp(0.,-ee)],1)
    z=np.stack([logsumexp(X[:,ghand==h],axis=1) for h in (0,1)],1)
    true=logs[np.arange(len(tt)),ghand[tt]]
    loss=np.stack([logsumexp(z+lam*logs,axis=1)-X[np.arange(len(tt)),tt]-lam*true for lam in LAM_GRID],1)
    means=ww@loss
    best=int(np.argmin(means))
    return LAM_GRID[best],means


def inference(d,boot,flips):
    d=np.asarray(d,float);estimate=d.mean();draws=d[boot].mean(1)
    lo,hi=np.quantile(draws,[.025,.975])
    p=(1+np.sum(np.abs(flips@d/len(d))>=abs(estimate)-1e-12))/(len(flips)+1)
    return dict(estimate=estimate,ci_lo=lo,ci_hi=hi,p_signflip=p)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',default='results/wid_handedness_2026-10-29')
    ap.add_argument('--revision',default='results/wid_revision_2026-11-08')
    ap.add_argument('--out',default='results/wid_fusion_robustness_2026-11-08')
    ap.add_argument('--jobs',type=int,default=3)
    a=ap.parse_args();start=time.time();source=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    U,F,H=rp.load_inputs(source)
    writers,dev,manifest=development_scores(U,F,source,out,a.jobs)
    hp=H[H.hand_model=='hc183'].set_index('unit_id').logit
    params=[];grid=[];rows=[];summary=[];max_formula_error=0.;invariance=True
    for ci,c in enumerate(rp.iter_score_cells(U,F)):
        key={k:c[k] for k in KEY};S,t,qw,uid=c['S'],c['t'],c['qw'],c['unit_id']
        T=rp.fit_temperatures(S,t,qw,c['weights']);eq=hp.reindex(uid).to_numpy();M=rp.match_logprob(eq,c['ghand'])
        base_score=S/T[:,None];nested_score=np.empty_like(S)
        for oi,outer in enumerate(writers):
            hit=qw==outer;temp=T[hit][0]
            lam,loss=choose_lambda(S,t,qw,c['weights'],dev[oi,c['q']],outer,temp,c['ghand'])
            assert loss.min()<=loss[0]+1e-12 and loss.min()<=loss[np.flatnonzero(LAM_GRID==1)[0]]+1e-12
            nested_score[hit]=base_score[hit]+lam*M[hit]
            params.append(dict(**key,writer_id=outer,T=temp,lam=lam,dev_nll=loss.min(),dev_nll_lambda0=loss[0],
                               dev_nll_lambda1=loss[np.flatnonzero(LAM_GRID==1)[0]],n_development_writers=41))
            for l,v in zip(LAM_GRID,loss):grid.append(dict(**key,outer_writer=outer,T=temp,lam=l,dev_nll=v,selected=bool(l==lam)))
            # Direct 42-candidate formula and exclusion checks on one outer fold/cell.
            if oi==0:
                keep=qw!=outer;ix=np.flatnonzero(keep)[:7];ee=dev[oi,c['q']][ix]
                logp=rp.match_logprob(ee,c['ghand']);xx=S[ix]/temp
                for l in LAM_GRID:
                    full=logsumexp(xx+l*logp,axis=1)-xx[np.arange(len(ix)),t[ix]]-l*logp[np.arange(len(ix)),t[ix]]
                    logs=np.stack([-np.logaddexp(0.,ee),-np.logaddexp(0.,-ee)],1)
                    zz=np.stack([logsumexp(xx[:,c['ghand']==h],axis=1) for h in (0,1)],1)
                    reduced=logsumexp(zz+l*logs,axis=1)-xx[np.arange(len(ix)),t[ix]]-l*logs[np.arange(len(ix)),c['ghand'][t[ix]]]
                    max_formula_error=max(max_formula_error,float(np.max(np.abs(full-reduced))))
                altered=dev[oi,c['q']].copy();altered[hit]=1e6
                tt=t.copy();tt[hit]=(tt[hit]+1)%len(writers)
                lam2,loss2=choose_lambda(S,tt,qw,c['weights'],altered,outer,temp,c['ghand'])
                invariance=invariance and lam==lam2 and np.array_equal(loss,loss2)
        for method,score in [('none',base_score),('fixed',base_score+M),('nested',nested_score)]:
            r=rp.ranks(score,t);acc=rp.mean_by_writer(r==1,qw,uid).reindex(writers)
            nll=rp.mean_by_writer(logsumexp(score,axis=1)-score[np.arange(len(t)),t],qw,uid).reindex(writers)
            mrr=rp.mean_by_writer(1/r,qw,uid).reindex(writers)
            summary.append(dict(**key,method=method,top1=acc.mean(),nll=nll.mean(),mrr=mrr.mean(),n_writers=42))
            for w in writers:rows.append(dict(**key,method=method,writer_id=w,top1=acc[w],nll=nll[w],mrr=mrr[w]))
        assert np.array_equal(rp.ranks(base_score,t),rp.ranks(S,t))
        if (ci+1)%20==0:print(f'nested selection {ci+1}/120 settings complete ({time.time()-start:.0f}s)',flush=True)
    W=pd.DataFrame(rows);P=pd.DataFrame(params)
    W.to_csv(out/'nested_writer_outcomes.csv',index=False);P.to_csv(out/'nested_selected_parameters.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'nested_summary.csv',index=False)
    pd.DataFrame(grid).to_csv(out/'nested_weight_validation_grid.csv.gz',index=False)
    pivot=W.pivot(index=KEY+['writer_id'],columns='method',values=['top1','nll'])
    rng=np.random.default_rng(INFERENCE_SEED);hand=np.array([w.startswith('L') for w in writers])
    boot=np.concatenate([rng.choice(np.flatnonzero(hand==h),(10000,int((hand==h).sum()))) for h in (False,True)],1)
    flips=rng.choice([-1,1],(9999,len(writers)));tests=[]
    for setting,D in W.groupby(KEY,sort=False):
        key=dict(zip(KEY,setting));pp=D.pivot(index='writer_id',columns='method',values='top1').reindex(writers)
        designated=(key['query'],key['n'],key['k'],key['rep']) in wh.PRIMARY
        for comparator in ['none','fixed']:
            tests.append(dict(**key,contrast='nested_minus_'+comparator,primary=designated,
                              **inference(pp.nested-pp[comparator],boot,flips)))
    tests=pd.DataFrame(tests);tests['p_holm']=np.nan
    Q=tests[tests.primary].sort_values('p_signflip')
    tests.loc[Q.index,'p_holm']=np.maximum.accumulate([min(1.,(len(Q)-j)*v) for j,v in enumerate(Q.p_signflip)])
    tests.to_csv(out/'nested_paired_tests.csv',index=False)
    aggregate=[]
    for scope,sub in [('all_120_settings',W),('own_text_96_settings',W[W['query']=='own'])]:
        for metric in ['top1','nll']:
            pp=sub.pivot(index=KEY+['writer_id'],columns='method',values=metric).reset_index()
            for comparator in ['none','fixed']:
                d=(pp.nested-pp[comparator]).groupby(pp.writer_id).mean().reindex(writers)
                aggregate.append(dict(scope=scope,metric=metric,contrast='nested_minus_'+comparator,**inference(d,boot,flips)))
    A=pd.DataFrame(aggregate);A.to_csv(out/'nested_effect_intervals.csv',index=False)
    original=pd.read_csv(Path(a.revision)/'writer_outcomes.csv');comparisons=[]
    for method,scenario,m,hm in [('none','none','none','-'),('fixed','query','bayes','hc183')]:
        old=original[(original.scenario==scenario)&(original.method==m)&(original.hand_model==hm)]
        J=W[W.method==method].merge(old,on=KEY+['writer_id'],validate='one_to_one',suffixes=('_new','_old'))
        err=float(np.max(np.abs(J.top1_new-J.top1_old)))
        assert len(J)==5040 and err<1e-12
        comparisons.append(dict(method=method,rows=len(J),max_top1_difference=err))
    assert invariance and max_formula_error<1e-10
    json_write(out/'nested_validation.json',dict(status='pass',comparators_match_main=comparisons,
        excluded_outer_query_targets_invariance=invariance,two_hand_nll_max_error=max_formula_error,
        outer_attribute_exclusion_manifests=len(manifest),settings=120,writer_outcomes=len(W),parameters=len(P),
        selected_dev_nll_no_worse_than_lambda0_or1=True))
    inputs=[source/'features/units.csv',source/'features/unit_features.npz',source/'handedness_lowo.csv.gz',Path(__file__),Path(rp.__file__),Path(wh.__file__)]
    json_write(out/'nested_provenance.json',dict(parameters=vars(a),seconds=time.time()-start,
        lambda_grid=list(LAM_GRID),tie_rule='lowest lambda among exact development NLL ties',
        temperature='Fixed at the same ID-only temperature used by fixed fusion; outer query writer excluded from temperature loss.',
        nested_training='Outer query writer excluded from all development attribute fits/calibrators; five development writer folds, with additional five-fold writer-group calibration inside each model training set.',
        gallery='42 known gallery hands and reference templates, including the outer writer, remain available candidate metadata throughout.',
        primary_family='Holm across eight exploratory comparisons: four designated settings, nested vs none and nested vs fixed.',
        uncertainty='10000 hand-stratified writer bootstrap and 9999 sign flips; fixed gallery and fitted nested predictions; no new-dataset or full-refitting bootstrap uncertainty.',
        inference_seed=INFERENCE_SEED,fold_seed=FOLD_SEED,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
        input_sha256={str(p):digest(p) for p in inputs}))
    findings=dict(overall=A.to_dict('records'),designated=tests[tests.primary].to_dict('records'),
                  selected_weight_counts={str(k):int(v) for k,v in P.lam.value_counts().sort_index().items()},
                  zero_weight_fraction=float((P.lam==0).mean()),median_weight=float(P.lam.median()))
    json_write(out/'nested_findings.json',findings)
    (out/'nested_findings.md').write_text('# Nested fusion-weight comparison\n\n'+A.to_string(index=False)+'\n\n'+tests[tests.primary].to_string(index=False)+'\n')
    print(A.to_string(index=False),flush=True);print(tests[tests.primary].to_string(index=False),flush=True)
    print(f'Finished in {time.time()-start:.1f}s',flush=True)


if __name__=='__main__':main()
