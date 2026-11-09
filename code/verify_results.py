#!/usr/bin/env python3
"""Validate the released numerical results and scientific invariants.

This is a reproducibility audit, not a certification of external generalization.
No dataset photographs or manuscript source are required for verification.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/wid_revision_2026-11-08'
OLD=ROOT/'results/wid_handedness_2026-10-29'
KEY=['rep','query','n','k']


def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    args=ap.parse_args()
    checks=[]
    def require(condition,label):
        if not bool(condition):raise AssertionError(label)
        checks.append(label)
    S=pd.read_csv(RUN/'id_summary.csv')
    W=pd.read_csv(RUN/'writer_outcomes.csv')
    T=pd.read_csv(RUN/'id_paired_tests.csv')
    I=pd.read_csv(RUN/'checks_filter_identity.csv')
    O=pd.read_csv(RUN/'checks_oracle_partitions.csv')
    C=pd.read_csv(RUN/'checks_identification_writer.csv')
    E=pd.read_csv(RUN/'checks_effect_intervals.csv')
    U=pd.read_csv(OLD/'features/units.csv')
    with np.load(OLD/'features/unit_features.npz',allow_pickle=True) as z:
        require(np.array_equal(z['unit_id'],U.unit_id.to_numpy()),'Feature/unit IDs align exactly')
        require(all(np.isfinite(z[k][U.valid]).all() for k in z.files if k!='unit_id'),'Every retained feature is finite')
    require(len(U)==5040 and U.writer_id.nunique()==42,'Expected unit and writer counts')
    require((~U.valid).sum()==39 and U[~U.valid].n.min()>1,'Exactly 39 excluded strips and no excluded paragraphs')
    require(S[S.scenario=='none'][KEY].drop_duplicates().shape[0]==120,'Exactly 120 unique baseline settings')
    require((S.n_writers==42).all(),'Every reported setting retains all 42 writers')
    require(S.top1.between(0,1).all(),'Top-1 accuracies are valid probabilities')
    GROUP=KEY+['scenario','hand_model','method']
    wm=W.groupby(GROUP).top1.mean().sort_index()
    sm=S.set_index(GROUP).top1.sort_index()
    require(np.allclose(wm,sm,atol=2e-14,rtol=0),'Writer outcomes reconstruct every summary')
    base=S[S.scenario=='none'].set_index(KEY).top1
    oracle=S[S.scenario=='oracle'].set_index(KEY).top1
    require((oracle-base>=-1e-14).all(),'Correct oracle filtering never reduces top-1')
    require(len(I)==456,'Exactly 456 filtering combinations with declared model coverage')
    require(np.max(np.abs(I.filter_top1-I.attribute_accuracy*I.known_top1-I.covariance))<2e-14,'Exact covariance filtering identity')
    require(((I.filter_top1>=I.frechet_lower-1e-14)&(I.filter_top1<=I.frechet_upper+1e-14)).all(),'Every filter outcome obeys Frechet bounds')
    require(np.max(np.abs(I.filter_top1-I.attribute_accuracy*I.known_success_given_hand_correct))<2e-14,'Exact conditional filtering identity')
    require((O.random_partition_exact_mean>=O.base-1e-14).all(),'Expected known random partition cannot harm top-1')
    require(np.max(np.abs(O.random_partition_mc_mean-O.random_partition_exact_mean))<.02,'Monte Carlo partition means agree with exact expectation within two points')
    require(len(T[T.primary])==4 and T[T.primary].p_holm.notna().all(),'Four designated comparisons have Holm adjustments')
    require((T.ci_lo<=T.diff_top1+1e-14).all() and (T.ci_hi>=T.diff_top1-1e-14).all(),'Point estimates lie within reported paired intervals')
    P=C.pivot(index=KEY+['writer_id'],columns='method',values='top1')
    for scenario,method in [('none','none'),('oracle','known')]:
        d=W[W.scenario==scenario].set_index(KEY+['writer_id']).top1.sort_index()
        require(np.allclose(d,P[method].sort_index(),atol=2e-14,rtol=0),f'Independent recomputation agrees: {method}')
    for method in ['query_bayes','query_filter']:
        meth='bayes' if method=='query_bayes' else 'filter'
        d=W[(W.scenario=='query')&(W.hand_model=='hc183')&(W.method==meth)].set_index(KEY+['writer_id']).top1.sort_index()
        require(np.allclose(d,P[method].sort_index(),atol=2e-14,rtol=0),f'Independent recomputation agrees: {method}')
    for method in ['known','query_bayes','query_filter','both_bayes','known_random_expected']:
        expected=(P[method]-P['none']).mean()
        value=E[(E.scope=='all_120_settings')&(E.contrast==method+'_minus_none')].estimate.iloc[0]
        require(abs(expected-value)<2e-14,f'Across-setting mean reconstructs: {method}')
    sim=pd.read_csv(RUN/'breakeven_simulation.csv.gz')
    require(len(sim)==52800,'Complete synthetic simulation grid')
    require((sim.groupby(KEY+['acc']).size()==200).all(),'Exactly 200 simulation draws per target and setting')
    require(np.allclose(sim[np.isclose(sim.acc,.5)].bayes,sim[np.isclose(sim.acc,.5)].base,atol=1e-14),'Zero-information fusion equals baseline')
    at1=sim[np.isclose(sim.acc,1.)]
    require(np.allclose(at1.bayes,at1.oracle,atol=1e-14) and np.allclose(at1['filter'],at1.oracle,atol=1e-14),'Perfect synthetic evidence equals exact oracle')
    prev=pd.read_csv(RUN/'prevalence.csv')
    require(len(prev)==288 and (prev.draws==1000).all(),'Complete prevalence sensitivity grid with 1000 draws')
    require(prev.top1.between(0,1).all(),'Prevalence accuracies are valid probabilities')
    require((pd.read_csv(RUN/'ranking_ties.csv').tied_top==0).all(),'No exact top-score ties in observed ID settings')
    extra=ROOT/'results/wid_fusion_robustness_2026-11-08'
    representation=pd.read_csv(extra/'representation_utility_intervals.csv')
    provenance=json.loads((extra/'representation_utility_provenance.json').read_text())
    require(provenance['passed'],'Representation utility generator validation passed')
    for name,expected in provenance['inputs'].items():
        require(sha256(ROOT/name)==expected,'Representation utility input hash: '+name)
    require(len(representation)==30 and (representation.n_writers==42).all()
            and (representation.settings_per_writer==20).all(),'Complete all-writer representation contrasts')
    NP=pd.read_csv(extra/'nested_writer_outcomes.csv').pivot(index=KEY+['writer_id'],columns='method',values='top1')
    for r in representation.itertuples():
        method=r.contrast.removesuffix('_minus_none')
        if r.contrast=='nested_minus_fixed':values=NP.loc[r.rep]['nested']-NP.loc[r.rep]['fixed']
        elif method=='nested':values=NP.loc[r.rep]['nested']-NP.loc[r.rep]['none']
        else:
            col={'known':'known','fixed':'query_bayes','filter':'query_filter'}[method]
            values=P.loc[r.rep][col]-P.loc[r.rep]['none']
        require(abs(values.mean()-r.estimate)<2e-14,'Representation contrast reconstructs: '+r.rep+' '+r.contrast)
        require(r.ci_lo<=r.estimate<=r.ci_hi,'Representation interval contains estimate: '+r.rep+' '+r.contrast)
    margin_report=json.loads((extra/'margin_validation.json').read_text())
    require(margin_report['passed'] and margin_report['settings']==120 and margin_report['writers']==42,
            'Margin reconstruction covers all settings and writers')
    for name,expected in margin_report['input_sha256'].items():
        require(sha256(ROOT/name)==expected,'Margin diagnostic input hash: '+name)
    margins=pd.read_csv(extra/'margin_writer_outcomes.csv')
    require(len(margins)==10080,'Complete fixed/nested margin writer rows')
    require(np.allclose(margins.gain,margins.rescued-margins.damaged,atol=2e-14,rtol=0),
            'Every writer/setting gain equals rescued minus damaged')
    require(np.allclose(margins.gain,margins.top1-margins.base,atol=2e-14,rtol=0),
            'Every writer/setting gain agrees with direct top-1 change')
    MP=margins.pivot(index=KEY+['writer_id'],columns='method',values='top1')
    require(np.allclose(MP[['fixed','nested']],NP[['fixed','nested']],atol=2e-14,rtol=0),
            'Margin reconstruction matches independently computed fixed/nested writer outcomes')
    for r in pd.read_csv(extra/'margin_effect_intervals.csv').itertuples():
        expected=margins[margins.method==r.method][r.metric].mean()
        require(abs(expected-r.estimate)<2e-14,'Margin diagnostic mean reconstructs: '+r.method+' '+r.metric)
        require(r.ci_lo<=r.estimate<=r.ci_hi,'Margin interval contains estimate: '+r.method+' '+r.metric)
    out={'passed':True,'checks':len(checks),'checks_detail':list(dict.fromkeys(checks)),
         'scope':'Numerical invariants, writer-level reconstruction, scientific input integrity; conditional on the observed cohort.'}
    (RUN/'analysis_validation_report.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='checks_detail'},indent=2))


if __name__=='__main__':main()
