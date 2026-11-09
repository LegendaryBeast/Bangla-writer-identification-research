#!/usr/bin/env python3
"""Independent invariants for nested-weight and within-cohort sensitivity outputs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import log_softmax

import revised_protocol as rp
from run_nested_fusion import choose_lambda,LAM_GRID

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results/wid_fusion_robustness_2026-11-08'
B=ROOT/'results/wid_revision_2026-11-08'
OLD=ROOT/'results/wid_handedness_2026-10-29'
KEY=['rep','query','n','k']


def main():
    passed=[]
    def check(x,label):
        if not bool(x):raise AssertionError(label)
        passed.append(label)
    W=pd.read_csv(R/'nested_writer_outcomes.csv');S=pd.read_csv(R/'nested_summary.csv')
    P=pd.read_csv(R/'nested_selected_parameters.csv');G=pd.read_csv(R/'nested_weight_validation_grid.csv.gz')
    E=pd.read_csv(R/'nested_effect_intervals.csv');T=pd.read_csv(R/'nested_paired_tests.csv')
    ow=pd.read_csv(B/'writer_outcomes.csv')
    check(len(W)==15120 and W.groupby(KEY+['method']).writer_id.nunique().eq(42).all(),'All 120 settings retain 42 writers under all three rules')
    check(W.top1.between(0,1).all() and np.isfinite(W.nll).all(),'Finite, bounded held-out metrics')
    mean=W.groupby(KEY+['method'])[['top1','nll','mrr']].mean().sort_index()
    saved=S.set_index(KEY+['method'])[['top1','nll','mrr']].sort_index()
    check(np.allclose(mean,saved,rtol=0,atol=2e-14),'Writer outcomes reconstruct every nested summary')
    for method,scen,meth,hm in [('none','none','none','-'),('fixed','query','bayes','hc183')]:
        A=W[W.method==method].set_index(KEY+['writer_id']).top1.sort_index()
        C=ow[(ow.scenario==scen)&(ow.method==meth)&(ow.hand_model==hm)].set_index(KEY+['writer_id']).top1.sort_index()
        check(np.allclose(A,C,rtol=0,atol=2e-14),f'Exact same-cohort comparator: {method}')
    check(len(P)==5040 and P.lam.isin(LAM_GRID).all(),'All 5040 selections belong to the preset grid')
    ix=KEY+['outer_writer'];mins=G.groupby(ix).dev_nll.min()
    pp=P.rename(columns={'writer_id':'outer_writer'}).set_index(ix).sort_index()
    check(np.allclose(pp.dev_nll,mins.sort_index(),rtol=0,atol=1e-12),'Each selected weight minimizes writer-held-out development NLL')
    check((pp.dev_nll<=pp.dev_nll_lambda0+1e-12).all() and (pp.dev_nll<=pp.dev_nll_lambda1+1e-12).all(),'No worse than grid endpoints in development, without asserting test improvement')
    check(G.groupby(ix).size().eq(11).all(),'Complete eleven-weight validation grid')
    manifest=json.loads((R/'nested_fold_manifest.json').read_text())
    check(len(manifest)==840,'All outer/unit-size/development-fold manifests present')
    u=pd.read_csv(OLD/'features/units.csv');writers=set(u.writer_id)
    seen={}
    for m in manifest:
        outer=m['outer_writer'];train=set(m['model_train_writers']);val=set(m['development_validation_writers'])
        check(not train&val and outer not in train|val and train|val==writers-{outer},'Outer and development writer exclusions hold')
        seen.setdefault((outer,m['n']),[]).extend(val)
        for inner in m['inner_calibration_folds']:
            a=set(inner['train_writers']);b=set(inner['validation_writers'])
            check(not a&b and a|b==train,'Inner calibration entirely within development training writers')
    check(all(len(v)==41 and len(set(v))==41 for v in seen.values()),'Each development writer scored out of fold once per outer exclusion/size')
    with np.load(R/'nested_development_logits.npz',allow_pickle=True) as z:
        check(np.array_equal(z['unit_id'],u.unit_id.to_numpy()),'Nested cached unit IDs align')
        for j,writer in enumerate(z['writers']):
            selfmask=u.writer_id.to_numpy()==writer
            check(np.isnan(z['logits'][j,selfmask]).all(),'Outer-writer development predictions absent')
            check(np.isfinite(z['logits'][j,u.valid.to_numpy()&~selfmask]).all(),'All development valid units have finite predictions')
    # Independent full-candidate softmax verifies the reduced two-hand objective.
    rng=np.random.default_rng(77);score=rng.normal(size=(18,42));target=rng.integers(0,42,18)
    group=np.repeat(['outer','dev1','dev2'],6);weights=np.full(18,1/18);ell=rng.normal(size=18)
    gh=np.array([0]*21+[1]*21);temp=.37
    lam,loss=choose_lambda(score,target,group,weights,ell,'outer',temp,gh)
    keep=group!='outer';M=rp.match_logprob(ell[keep],gh)
    independent=[]
    for v in LAM_GRID:
        logp=log_softmax(score[keep]/temp+v*M,axis=1)
        independent.append(-np.mean(logp[np.arange(keep.sum()),target[keep]]))
    check(np.allclose(loss,independent,rtol=0,atol=1e-12),'Full 42-candidate softmax independently reconstructs lambda loss')
    altered=ell.copy();altered[~keep]=1000;targets=target.copy();targets[~keep]=(target[~keep]+3)%42
    lam2,loss2=choose_lambda(score,targets,group,weights,altered,'outer',temp,gh)
    check(lam==lam2 and np.array_equal(loss,loss2),'Weight selection invariant to outer-query evidence/targets')
    for scale in [.5,1,2]:
        check(np.array_equal(rp.ranks(score/(scale*temp)+rp.match_logprob(ell,gh),target),rp.ranks(score/temp+scale*rp.match_logprob(ell,gh),target)),'Temperature perturbation equivalent to relative attribute weighting for ranking')
    Q=T[T.primary].sort_values('p_signflip')
    expected=np.maximum.accumulate([min(1.,(len(Q)-j)*p) for j,p in enumerate(Q.p_signflip)])
    check(len(Q)==8 and np.allclose(Q.p_holm,expected),'Holm correction covers eight new designated comparisons')
    for scope in ['all_120_settings','own_text_96_settings']:
        D=W if scope.startswith('all') else W[W['query']=='own']
        for metric in ['top1','nll']:
            ppiv=D.pivot(index=KEY+['writer_id'],columns='method',values=metric)
            for comp in ['none','fixed']:
                val=(ppiv.nested-ppiv[comp]).mean()
                savedval=E[(E.scope==scope)&(E.metric==metric)&(E.contrast=='nested_minus_'+comp)].estimate.iloc[0]
                check(abs(val-savedval)<2e-14,'Held-out aggregate contrasts independently reconstructed')
    F=pd.read_csv(R/'fixed_seed_writer_outcomes.csv');cal=pd.read_csv(R/'calibration_seed_metrics.csv')
    check(set(F.seed)=={2031,2032,2033,2034},'All four correlated calibration partitions retained')
    for seed in [2031,2032,2033,2034]:
        d=F[F.seed==seed].pivot(index=KEY+['writer_id'],columns='method',values='top1')
        value=(d.fixed-d['none']).mean()
        SE=pd.read_csv(R/'calibration_seed_identification_effects.csv')
        recorded=SE[(SE.seed==seed)&(SE.scope=='all_120_settings')&(SE.method=='fixed')].estimate.iloc[0]
        check(abs(value-recorded)<2e-14,'Every seed-specific fusion effect reconstructed')
    meta=json.loads((R/'calibration_seed_provenance.json').read_text())
    check(all(d['maximum_raw_score_change']<1e-8 for d in meta['raw_score_checks']),'Full outer SVM scores invariant to calibration partitions')
    O=pd.read_csv(R/'evaluation_writer_omission.csv');OS=pd.read_csv(R/'evaluation_writer_omission_summary.csv')
    check(set(O.method)=={'none','known','fixed','nested','filter'},'Influence summaries include learned weighting')
    check(O.groupby(['scope','method','metric']).omitted_writer.nunique().eq(42).all(),'Every writer represented in omission diagnostics')
    for r in OS.itertuples():
        d=O[(O.scope==r.scope)&(O.method==r.method)&(O.metric==r.metric)]
        check(abs(d.omitted_estimate.min()-r.omission_min)<1e-14 and abs(d.omitted_estimate.max()-r.omission_max)<1e-14,'Omission ranges reconstruct from full writer list')
    classes=pd.read_csv(R/'class_specific_identification.csv')
    for key,d in classes.groupby(['scope','method','metric']):
        a=d.set_index('hand').estimate
        check(abs(.5*(a.left+a.right)-a['all'])<2e-14,'Class-specific means reconstruct balanced aggregate')
    tempstats=pd.read_csv(R/'temperature_selected_writer.csv')
    check(len(tempstats)==5040 and (tempstats.near_optimal_count>=1).all(),'Complete selected-temperature diagnostics')
    for p in R.glob('*.json'):
        json.loads(p.read_text(),parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Invalid JSON constant: '+v)))
    result=dict(passed=True,numerical_and_scope_checks=len(passed),unique_checks=list(dict.fromkeys(passed)),
                scope='Nested writer-exclusion, score/metric equivalence, conditional sensitivity and output integrity; not independent-dataset validation.')
    (R/'additional_validation_report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='unique_checks'},indent=2))


if __name__=='__main__':main()
