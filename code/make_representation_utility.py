#!/usr/bin/env python3
"""Exploratory representation-level utility, conditional on the fitted gallery.

All 20 settings per representation and all 42 writers are retained. Bootstrap
indices are shared across every plotted contrast, preserving paired dependence.
Intervals are marginal, descriptive intervals, not simultaneous coverage claims.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from make_revision_paper import ROOT, OUT, REPS, NAMES, save, panel
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RUN = ROOT / 'results/wid_fusion_robustness_2026-11-08'
BASE = ROOT / 'results/wid_revision_2026-11-08'
KEY = ['rep', 'query', 'n', 'k', 'writer_id']


def main():
    paths = [BASE/'checks_identification_writer.csv', RUN/'nested_writer_outcomes.csv']
    fixed = pd.read_csv(paths[0]).pivot(index=KEY, columns='method', values='top1')
    nested = pd.read_csv(paths[1]).pivot(index=KEY, columns='method', values='top1')
    assert len(fixed) == 5040 and len(nested) == 5040
    assert fixed.index.equals(nested.index)
    assert np.allclose(fixed['none'], nested['none'], atol=2e-14, rtol=0)
    assert np.allclose(fixed['query_bayes'], nested['fixed'], atol=2e-14, rtol=0)
    X = fixed[['none', 'known', 'query_bayes', 'query_filter']].rename(
        columns={'query_bayes':'fixed', 'query_filter':'filter'})
    X['nested'] = nested['nested']
    assert (X.groupby(level=['rep','writer_id']).size() == 20).all()
    X = X.groupby(level=['rep', 'writer_id']).mean()
    writers = sorted(X.index.get_level_values('writer_id').unique())
    assert len(writers) == 42
    left = np.array([i for i,w in enumerate(writers) if w.startswith('L_')])
    right = np.array([i for i,w in enumerate(writers) if w.startswith('R_')])
    assert len(left) == len(right) == 21
    rng = np.random.default_rng(20361108)
    samples = np.concatenate([rng.choice(left,(10000,21)), rng.choice(right,(10000,21))],axis=1)
    methods = ['known', 'fixed', 'nested', 'filter']
    rows = []
    for rep in REPS:
        x = X.loc[rep].reindex(writers)
        for method in methods:
            values = (x[method]-x['none']).to_numpy()
            boot = values[samples].mean(axis=1)
            lo,hi = np.quantile(boot,[.025,.975])
            rows.append(dict(rep=rep, contrast=method+'_minus_none', estimate=values.mean(),
                             ci_lo=lo, ci_hi=hi, n_writers=42, settings_per_writer=20))
        values = (x['nested']-x['fixed']).to_numpy()
        lo,hi = np.quantile(values[samples].mean(axis=1),[.025,.975])
        rows.append(dict(rep=rep, contrast='nested_minus_fixed', estimate=values.mean(),
                         ci_lo=lo, ci_hi=hi, n_writers=42, settings_per_writer=20))
    E = pd.DataFrame(rows)
    E.to_csv(RUN/'representation_utility_intervals.csv',index=False)
    # Independent reconstruction from setting-level published summaries.
    S = pd.read_csv(BASE/'id_summary.csv')
    NS = pd.read_csv(RUN/'nested_summary.csv')
    for rep in REPS:
        base = S[(S.rep==rep)&(S.scenario=='none')].top1.mean()
        learned = NS[(NS.rep==rep)&(NS.method=='nested')].top1.mean()
        observed = E[(E.rep==rep)&(E.contrast=='nested_minus_none')].estimate.iloc[0]
        assert abs(observed-(learned-base)) < 2e-14
    fig,(a,b) = plt.subplots(1,2,figsize=(6.9,2.85),sharey=True,gridspec_kw={'width_ratios':[1.25,1]})
    colors = ['#009E73','#0072B2','#CC79A7','#D55E00']
    markers = ['s','o','D','^']
    offsets = [-.24,-.08,.08,.24]
    for j,rep in enumerate(REPS):
        for method,color,marker,offset in zip(methods,colors,markers,offsets):
            r = E[(E.rep==rep)&(E.contrast==method+'_minus_none')].iloc[0]
            a.errorbar(100*r.estimate,j+offset,
                       xerr=[[100*(r.estimate-r.ci_lo)],[100*(r.ci_hi-r.estimate)]],
                       fmt=marker,color=color,markersize=3.8,capsize=1.8,lw=.9)
        r = E[(E.rep==rep)&(E.contrast=='nested_minus_fixed')].iloc[0]
        b.errorbar(100*r.estimate,j,xerr=[[100*(r.estimate-r.ci_lo)],[100*(r.ci_hi-r.estimate)]],
                   fmt='D',color='#CC79A7',markersize=3.8,capsize=2,lw=1)
    for ax,letter in [(a,'a'),(b,'b')]:
        ax.axvline(0,color='gray',ls=':',lw=.8)
        ax.set_yticks(range(6),[NAMES[r] for r in REPS])
        panel(ax,letter)
    a.invert_yaxis()
    b.tick_params(axis='y',labelleft=False)
    a.set_xlabel('Change from no fusion (points)')
    b.set_xlabel('Nested minus fixed fusion (points)')
    handles = [Line2D([],[],color=c,marker=m,ls='',markersize=4) for c,m in zip(colors,markers)]
    fig.legend(handles,['Known','Fixed fusion','Nested fusion','Hard filter'],loc='upper center',
               ncol=4,frameon=False,bbox_to_anchor=(.52,1.01),handletextpad=.3,columnspacing=1.1)
    fig.tight_layout(rect=(0,0,1,.89),pad=.5,w_pad=2)
    save(fig,'representation_utility')
    M={}
    for contrast,label in [('nested_minus_none','RepNested'),('nested_minus_fixed','RepCorrection')]:
        d=E[E.contrast==contrast]
        M[label+'Min']=f'{100*d.estimate.min():+.2f}'
        M[label+'Max']=f'{100*d.estimate.max():+.2f}'
    (OUT/'representation_utility_macros.tex').write_text(
        '% Generated by code/make_representation_utility.py\n'+
        ''.join(f'\\newcommand{{\\r{k}}}{{{v}}}\n' for k,v in M.items()))
    audit={'passed':True,'seed':20361108,'draws':10000,'writers':42,'settings_per_representation':20,
           'scope':'Conditional paired writer bootstrap; fixed gallery and predictions; marginal exploratory intervals.',
           'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
           'checks':['Identical 5040 writer/setting keys','Baseline and fixed comparators match',
                     'All 42 writers and all 20 settings retained per representation',
                     'Nested gains reconstruct independently from setting summaries']}
    (RUN/'representation_utility_provenance.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(E.to_string(index=False))


if __name__=='__main__':
    main()
