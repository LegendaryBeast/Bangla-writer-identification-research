#!/usr/bin/env python3
"""Reconstruct fusion decisions and separate rescued from damaged matches.

Uses only existing features, calibrated scores and selected parameters. No
training, selection, writer exclusion or new dataset is introduced.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from make_revision_paper import ROOT, OUT, save, panel
import matplotlib.pyplot as plt
import revised_protocol as rp

RUN=ROOT/'results/wid_fusion_robustness_2026-11-08'
BASE=ROOT/'results/wid_revision_2026-11-08'
SOURCE=ROOT/'results/wid_handedness_2026-10-29'
KEY=['rep','query','n','k','writer_id']


def main():
    U,F,H=rp.load_inputs(SOURCE)
    ell=H[H.hand_model=='hc183'].set_index('unit_id').logit.reindex(U.unit_id).to_numpy()
    parameters=pd.read_csv(RUN/'nested_selected_parameters.csv').set_index(KEY)
    rows=[];example=None;decisions=0;ties=0
    for cell in rp.iter_score_cells(U,F):
        rep,query,n,k=[cell[x] for x in ['rep','query','n','k']]
        key=(rep,query,n,k)
        S=cell['S'];t=cell['t'];qw=cell['qw'];uid=cell['unit_id'];qh=cell['qhand']
        par=parameters.loc[key].reindex(qw)
        T=par['T'].to_numpy();same=qh[:,None]==cell['ghand'][None,:]
        known=np.argmax(np.where(same,S,-3.),axis=1)==t
        opposite=np.argmax(np.where(~same,S,-3.),axis=1)
        margin=(S[np.arange(len(S)),t]-S[np.arange(len(S)),opposite])/T
        odds=(2*qh-1)*ell[cell['q']]
        base=np.argmax(S,axis=1)==t
        evidence=rp.match_logprob(ell[cell['q']],cell['ghand'])
        for method,lam in [('fixed',np.ones(len(S))),('nested',par.lam.to_numpy())]:
            fused=np.argmax(S/T[:,None]+lam[:,None]*evidence,axis=1)==t
            boundary=margin+lam*odds
            reconstructed=known&((boundary>0)|((boundary==0)&(t<opposite)))
            assert np.array_equal(fused,reconstructed),f'Margin identity failed: {key}, {method}'
            rescued=(~base)&fused;damaged=base&(~fused)
            assert np.array_equal(fused.astype(int)-base.astype(int),rescued.astype(int)-damaged.astype(int))
            decisions+=len(S);ties+=int((boundary==0).sum())
            values={'base':base,'top1':fused,'gain':fused.astype(float)-base,
                    'rescued':rescued,'damaged':damaged,'known':known}
            W={name:rp.mean_by_writer(v,qw,uid) for name,v in values.items()}
            for w in cell['writers']:
                rows.append(dict(zip(KEY,key+(w,)),method=method,**{name:float(v.loc[w]) for name,v in W.items()}))
            if key==('hc183','own',1,1) and method=='fixed':
                status=np.select([rescued,damaged,base&fused],['rescued','damaged','correct_unchanged'],default='wrong_unchanged')
                example=pd.DataFrame(dict(unit_id=uid,writer_id=qw,config=cell['config'],margin=margin,
                                          true_hand_logodds=odds,known=known,base=base,fused=fused,status=status))
        print(f'{rep} {query} n={n} k={k}',flush=True)
    W=pd.DataFrame(rows)
    W.to_csv(RUN/'margin_writer_outcomes.csv',index=False)
    expected=pd.read_csv(RUN/'nested_writer_outcomes.csv').set_index(KEY+['method']).top1
    actual=W.set_index(KEY+['method']).top1
    assert np.allclose(expected.reindex(actual.index),actual,atol=2e-14,rtol=0)
    assert len(W)==10080 and W.writer_id.nunique()==42
    writers=sorted(W.writer_id.unique());rng=np.random.default_rng(20361109)
    L=np.array([i for i,w in enumerate(writers) if w.startswith('L')]);R=np.array([i for i,w in enumerate(writers) if w.startswith('R')])
    boot=np.concatenate([rng.choice(L,(10000,21)),rng.choice(R,(10000,21))],axis=1)
    estimates=[]
    for method in ['fixed','nested']:
        x=W[W.method==method].groupby('writer_id').mean(numeric_only=True).reindex(writers)
        for metric in ['rescued','damaged','gain']:
            v=x[metric].to_numpy();lo,hi=np.quantile(v[boot].mean(axis=1),[.025,.975])
            estimates.append(dict(method=method,metric=metric,estimate=v.mean(),ci_lo=lo,ci_hi=hi))
    E=pd.DataFrame(estimates);E.to_csv(RUN/'margin_effect_intervals.csv',index=False)
    example.to_csv(RUN/'margin_example.csv',index=False)
    fig,(a,b)=plt.subplots(1,2,figsize=(6.9,3.15),gridspec_kw={'width_ratios':[.85,1.15]})
    for i,metric,color in [(0,'rescued','#0072B2'),(1,'damaged','#D55E00')]:
        d=E[E.metric==metric].set_index('method').reindex(['fixed','nested'])
        a.bar(np.arange(2)+(i-.5)*.29,100*d.estimate,width=.27,color=color,label=metric.capitalize())
        a.errorbar(np.arange(2)+(i-.5)*.29,100*d.estimate,
                   yerr=100*np.array([d.estimate-d.ci_lo,d.ci_hi-d.estimate]),fmt='none',ecolor='black',capsize=2,lw=.8)
    a.set_xticks([0,1],['Fixed','Nested']);a.set_ylabel('Share of decisions (%)');a.set_xlabel('Fusion rule')
    a.legend(frameon=False,loc='upper right',fontsize=8)
    known=example[example.known]
    styles=[('wrong_unchanged','Wrong, unchanged','#bbb','.'),('correct_unchanged','Correct, unchanged','#009E73','.'),
            ('rescued','Rescued','#0072B2','o'),('damaged','Damaged','#D55E00','x')]
    for state,label,color,marker in styles:
        d=known[known.status==state]
        b.scatter(d.margin,d.true_hand_logodds,color=color,marker=marker,s=11,alpha=.6,label=label,linewidths=.7)
    b.axvline(0,color='gray',ls=':',lw=.8)
    xlim=b.get_xlim();ylim=b.get_ylim();xx=np.array(xlim)
    b.plot(xx,-xx,color='black',ls='--',lw=.9)
    b.set_xlim(xlim);b.set_ylim(ylim)
    b.set_xlabel('Scaled identity margin, $m$');b.set_ylabel('True-hand log-odds, $r$')
    b.legend(frameon=False,fontsize=7.5,loc='upper right',handletextpad=.2)
    for ax,letter in [(a,'a'),(b,'b')]:panel(ax,letter)
    fig.tight_layout(pad=1,w_pad=2);save(fig,'margin_diagnostic')
    macros={}
    for method,prefix in [('fixed','Fixed'),('nested','Nested')]:
        for metric,suffix in [('rescued','Rescued'),('damaged','Damaged')]:
            e=E[(E.method==method)&(E.metric==metric)].iloc[0]
            macros[prefix+suffix]=f'{100*e.estimate:.2f}'
    (OUT/'margin_macros.tex').write_text('% Generated by code/run_margin_diagnostic.py\n'+
        ''.join(f'\\newcommand{{\\r{k}}}{{{v}}}\n' for k,v in macros.items()))
    inputs=[SOURCE/'features/units.csv',SOURCE/'features/unit_features.npz',SOURCE/'handedness_lowo.csv.gz',
            RUN/'nested_selected_parameters.csv',RUN/'nested_writer_outcomes.csv',
            ROOT/'code/revised_protocol.py',Path(__file__)]
    report=dict(passed=True,decisions_reconstructed=decisions,exact_boundary_ties=ties,
                writer_rows=len(W),writers=42,settings=120,bootstrap_draws=10000,bootstrap_seed=20361109,
                input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                checks=['All direct fusion decisions match margin identity including tie precedence',
                        'All writer-level top-1 values match independently generated nested/fixed results',
                        'Gain equals rescued minus damaged for every individual decision'],
                scope='Descriptive, conditional on the same gallery, predictions and parameters; no new datasets or models.')
    (RUN/'margin_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(E.to_string(index=False))


if __name__=='__main__':
    main()
