#!/usr/bin/env python3
"""Generate CSAFE report, tables and figures exclusively from audited outputs."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/csafe_replication_2026-11-08'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'dataset/plot_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPS={'resnet18':'ResNet-18','vit_b_16':'ViT-B/16','hc183':'HC-183','stroke':'STROKE'}
CONDITIONS={'S2_LND_r01':'Session 2, same prompt','S2_WOZ_r01':'Session 2, different prompt','S3_WOZ_r01':'Session 3, different prompt'}
HANDS={'hc183':'HC-183','stroke':'STROKE','hc183_adj':'HC-183 adjusted'}
METHODS={'identity_only':'Identity','oracle_known_hand':'Oracle','predicted_hard_filter':'Hard filter',
 'fixed_fusion_lambda1':'Fixed','nested_fusion':'Nested'}
COLORS={'identity_only':'#555555','oracle_known_hand':'#009E73','predicted_hard_filter':'#D55E00',
 'fixed_fusion_lambda1':'#0072B2','nested_fusion':'#CC79A7'}


def accuracy_intervals(W,H):
    rng=np.random.default_rng(20261120);rows=[]
    for row in H[H.cohort=='full_balanced'].itertuples():
        w=W[(W.cohort==row.cohort)&(W.condition==row.condition)&(W.hand_model==row.hand_model)
            &(W.rep=='resnet18')&(W.method=='identity_only')].sort_values('writer_id')
        draws=np.zeros(10000)
        for hand in (0,1):
            values=w.loc[w.label==hand,'hand_correct'].to_numpy()
            draws+=.5*values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
        lo,hi=np.quantile(draws,[.025,.975])
        rows.append(dict(condition=row.condition,hand_model=row.hand_model,accuracy=row.accuracy,
            ci_lo=lo,ci_hi=hi,left_recall=row.left_recall,right_recall=row.right_recall,
            raw_auc=row.raw_auc,calibrated_auc=row.calibrated_auc,nll=row.nll,brier=row.brier))
    result=pd.DataFrame(rows);result.to_csv(OUT/'handedness_conditional_intervals.csv',index=False)
    return result


def main():
    validation=json.loads((OUT/'validation.json').read_text());assert validation['passed']
    S=pd.read_csv(OUT/'id_summary.csv');T=pd.read_csv(OUT/'paired_tests.csv')
    H=pd.read_csv(OUT/'handedness_summary.csv');D=pd.read_csv(OUT/'diagnostics.csv')
    W=pd.read_csv(OUT/'writer_outcomes.csv.gz');P=pd.read_csv(OUT/'selected_parameters.csv')
    full=S[(S.cohort=='full_balanced')&(S.hand_model=='hc183')]
    hci=accuracy_intervals(W,H)
    selected=T[T.primary]
    table=full.pivot(index=['condition','rep'],columns='method',values='top1')
    table.mul(100).to_csv(OUT/'full_primary_model_accuracy_percent.csv',float_format='%.6f')
    stats=P[P.cohort=='full_balanced'].groupby(['condition','rep','hand_model']).agg(
        zero_weight_fraction=('fusion_weight',lambda v:(v==0).mean()),
        median_weight=('fusion_weight','median'),median_temperature=('temperature','median'),
        temperature_lower_endpoint_fraction=('temperature',lambda v:np.isclose(v,.001).mean()),
        temperature_upper_endpoint_fraction=('temperature',lambda v:np.isclose(v,1).mean()))
    stats.to_csv(OUT/'parameter_diagnostics.csv')
    fit_manifest=json.loads((OUT/'fitting_manifest.json').read_text())
    cal_rows=[]
    for cell in fit_manifest:
        if cell['cohort']!='full_balanced' or cell['condition']!='S2_LND_r01':continue
        records=json.loads((OUT/'attribute_cache'/(cell['cache_key']+'.json')).read_text())
        for record in records:
            cal_rows.append(dict(hand_model=cell['hand_model'],writer_id=record['outer_writer'],**record['final_platt_fit']))
    pd.DataFrame(cal_rows).to_csv(OUT/'final_calibrator_diagnostics.csv',index=False)
    negative_slopes=pd.DataFrame(cal_rows).groupby('hand_model').raw_slope.apply(lambda v:int((v<0).sum())).to_dict()
    significant=selected[selected.p_holm_primary<.05]
    positive=significant[significant.delta_top1>0]
    negative=significant[significant.delta_top1<0]
    lines=['# CSAFE cross-session external replication','',
        'Completed from the original frozen design in protocol_v1.json and the numerical implementation amendment in protocol_v1_1.json. Dataset: [CSAFE Handwriting Database v2](https://doi.org/10.25380/iastate.10062203.v2), CC BY 4.0. The [original Data in Brief article](https://doi.org/10.1016/j.dib.2019.105059) describes an earlier 90-writer release; the version-2 survey contains 475 writers. Counts below come from the actual version-2 data, not extrapolation from that article.','',
        '## Design and scope','',
        'The cohort consists of all 48 left-handed participants and 48 seeded right-handed participants. Three ambidextrous participants and one missing-hand participant were excluded before sampling. All 864 selected pages passed the frozen feature-validity rules. Each participant supplies six long-text session-1 pages for auxiliary-model training, one of which (LND repetition 1) is the identity gallery reference. The other pages of a queried writer never enter its handedness fitting or calibration. Queries use separate physical pages from session 2 (same/different prompt) and session 3 (different prompt).','',
        'Reported timing is positive for all available query-to-gallery day differences: session 2 minus session 1 has 94 nonmissing values, median 33 days and range 18–252; session 3 minus session 1 has 95 nonmissing values, median 68 days and range 42–272. Missing timing entries are retained under their original session labels. The data do not support claiming that every observed gap is at least 21 days.','',
        'The same original descriptors, pretrained neural checkpoints, SVM C=0.1, logistic-calibration objective, temperature grid and fusion-weight grid were retained. Numerical convergence warnings in an initial sensitivity-model pass were resolved by centering/scaling training OOF SVM scores and transforming the L2 penalty algebraically to preserve the original C=10000 objective; every fit was rerun and checked by its analytic gradient. This amendment preceded inspection of CSAFE performance tables. There is no network fine-tuning or classifier selection based on achieved CSAFE accuracy. The primary handedness model is HC-183; STROKE and intensity-adjusted HC-183 are predefined sensitivity models. The protocol was specified before CSAFE outcomes, as an extension of the existing study; it is not a publicly registered preregistration.','',
        'This evaluates a new dataset, writer population, physical pages and acquisition sessions with within-dataset model refitting. It is not transfer of a frozen Bangla-trained predictor. The 20 balanced 42-writer galleries are overlapping subsamples from this same selected 96-writer pool, not independent dataset replications.','',
        '## Handedness prediction','',
        '| Query condition | Attribute model | Balanced accuracy (%) | Conditional 95% CI (%) | Raw / calibrated AUC | NLL / Brier |',
        '|---|---|---:|---:|---:|---:|']
    for r in hci.itertuples():
        lines.append(f'| {CONDITIONS[r.condition]} | {HANDS[r.hand_model]} | {100*r.accuracy:.2f} | [{100*r.ci_lo:.2f}, {100*r.ci_hi:.2f}] | {r.raw_auc:.3f} / {r.calibrated_auc:.3f} | {r.nll:.3f} / {r.brier:.3f} |')
    lines+=['','The cohort is balanced, so ordinary and balanced accuracy coincide. Intervals resample writers within handedness and condition on the fitted predictors. AUC is descriptive discrimination across writer-held-out scores. The reported rates characterize these fitted classifiers and depend on prompt, session, representation and calibration.','',
        f"The adjusted sensitivity model has negative fitted Platt slopes in {negative_slopes['hc183_adj']} of 96 outer fits, compared with {negative_slopes['hc183']} for unadjusted HC-183 and {negative_slopes['stroke']} for STROKE. Its weaker calibrated accuracy is retained. Raw and calibrated AUCs pool different writer-held-out score sets, rather than a single common monotone calibration map. These are descriptive calibration diagnostics, not evidence of a particular causal acquisition mechanism.",'',
        '## Identification accuracy','',
        'The following table uses the primary HC-183 handedness predictor. ResNet-18 and ViT-B/16 in the two session-2 conditions define the primary inferential comparisons; other rows are secondary. All values are top-1 identification percentages.','',
        '| Query condition | Identity representation | Identity only | Oracle hand | Hard filter | Fixed fusion | Nested fusion |',
        '|---|---|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        for rep in REPS:
            row=table.loc[(condition,rep)]
            lines.append('| '+CONDITIONS[condition]+' | '+REPS[rep]+' | '+' | '.join(f'{100*row[m]:.2f}' for m in METHODS)+' |')
    lines+=['','## Primary paired contrasts','',
        '| Query condition | Representation | Fusion | Change (points) | Conditional 95% CI | Holm-adjusted p |',
        '|---|---|---|---:|---:|---:|']
    for r in selected.itertuples():
        lines.append(f'| {CONDITIONS[r.condition]} | {REPS[r.rep]} | {METHODS[r.method]} | {100*r.delta_top1:+.2f} | [{100*r.ci_lo:.2f}, {100*r.ci_hi:.2f}] | {r.p_holm_primary:.4f} |')
    lines+=['',f'Of the eight predefined contrasts, {len(positive)} show a positive difference and {len(negative)} a negative difference with Holm-adjusted p < 0.05. All eight comparisons are retained. Intervals use 10,000 hand-stratified writer-bootstrap draws, and paired sign-flip tests use 9,999 draws. These condition on fitted models, selected pages and the enrolled gallery; they do not propagate full fitting or population-sampling uncertainty. Individual intervals are not simultaneous or multiplicity-adjusted, and percentile-bootstrap intervals are not inversions of the discrete sign-flip tests; an interval can therefore exclude zero while a Holm-adjusted test does not reject. An observed [0,0] interval indicates identical correctness outcomes for the fitted rules, not population equivalence.','',
        '## Gallery sensitivity','',
        'Each of 20 seeded balanced 21/21 galleries refits the primary HC-183 attribute model and its nested development models. The following ranges concern fixed/nested changes relative to identity only in those galleries; they are descriptive ranges, not confidence intervals.','',
        '| Condition | Representation | Fixed change range (points) | Nested change range (points) |',
        '|---|---|---:|---:|']
    small=S[(S.cohort!='full_balanced')&(S.hand_model=='hc183')].pivot(
        index=['cohort','condition','rep'],columns='method',values='top1')
    for condition in list(CONDITIONS)[:2]:
        for rep in ('resnet18','vit_b_16'):
            group=small.xs((condition,rep),level=('condition','rep'));ranges=[]
            for method in ('fixed_fusion_lambda1','nested_fusion'):
                delta=100*(group[method]-group.identity_only);ranges.append(f'[{delta.min():+.2f}, {delta.max():+.2f}]')
            lines.append('| '+CONDITIONS[condition]+' | '+REPS[rep]+' | '+' | '.join(ranges)+' |')
    lines+=['','## Interpretation and checks','',
        'Oracle and random-partition comparisons, rescue/damage counts, class-specific identification, all secondary attribute-model results, calibration diagnostics, chosen temperatures/weights and every writer outcome are retained in the CSV outputs. Evidence of utility must be evaluated jointly with predictor quality, identity evidence and decision margins; attribute-classification accuracy alone is not the endpoint. A favorable or unfavorable external result should not be treated as a universal claim about handedness or forensic attribution.','',
        f'The full audit checked {validation["outer_writer_fits_audited"]:,} outer writer exclusions, {validation["attribute_models_audited"]:,} final/development attribute models and {validation["calibration_folds_audited"]:,} calibration folds. It verifies that auxiliary training uses session 1 only, query pages/sessions are distinct, scored writers and development validation writers are absent from their attribute fits, all input/checkpoint hashes match, reported accuracies equal saved writer outcomes, and the exact filtering identity and fusion-margin reconstruction hold.','',
        'The raw dataset is downloaded separately under its source terms. No dataset images or neural checkpoint binaries are included in the reproducibility archive. The existing manuscript and submission package are unchanged by this experiment.','',
        '![Primary identification utility](csafe_identification_utility.png)','',
        'Figure 1. Full 96-writer results using the primary HC-183 attribute model. Left: identity, oracle, filtering and fusion accuracy. Right: fixed/nested changes with conditional writer-bootstrap 95% intervals. Both session-2 query conditions are shown.','',
        '![Attribute and gallery robustness](csafe_attribute_robustness.png)','',
        'Figure 2. Left: writer-held-out handedness accuracy across the three predefined models and query conditions, with conditional 95% intervals. Right: fixed/nested identification changes across 20 dependent balanced 42-writer galleries.']
    if (OUT/'independent_recompute_validation.json').exists():
        assert json.loads((OUT/'independent_recompute_validation.json').read_text())['passed']
        lines += ['', 'A fresh run from the extracted package, with supplied attribute caches removed, reproduced all nine core numerical/manifest output files exactly. This verifies computational reproducibility using identical frozen features; it is not another empirical replication.']
    lines += ['', 'The predefined primary family refers to the CSAFE contrasts. Dataset selection was not part of a globally preregistered dataset-sampling program; the CSAFE dataset and its protocol were fixed before inspecting its performance outcomes.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.bbox':'tight'})
    cases=[(c,r) for c in list(CONDITIONS)[:2] for r in ('resnet18','vit_b_16')]
    labels=[('Same prompt' if c=='S2_LND_r01' else 'Different prompt')+' · '+REPS[r] for c,r in cases]
    y=np.arange(4)
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.5),gridspec_kw={'width_ratios':[1.1,1]})
    for j,method in enumerate(METHODS):
        vals=[100*table.loc[(c,r),method] for c,r in cases]
        axes[0].plot(vals,y+(j-2)*.12,'o',ms=4,color=COLORS[method],label=METHODS[method])
    axes[0].set(yticks=y,yticklabels=labels,xlabel='Top-1 identification (%)',xlim=(0,104))
    for method,offset in [('fixed_fusion_lambda1',-.1),('nested_fusion',.1)]:
        subset=selected[selected.method==method].set_index(['condition','rep']).reindex(cases)
        vals=100*subset.delta_top1.to_numpy()
        errors=np.stack([vals-100*subset.ci_lo.to_numpy(),100*subset.ci_hi.to_numpy()-vals])
        axes[1].errorbar(vals,y+offset,xerr=np.maximum(errors,0),fmt='o',ms=4,capsize=2,color=COLORS[method])
    axes[1].set(yticks=y,yticklabels=[],xlabel='Change from identity (percentage points)')
    axes[1].axvline(0,color='#888',ls='--',lw=.8)
    for ax,title in zip(axes,['(a) Cross-session identification','(b) Conditional uncertainty']):
        ax.set_title(title,loc='left',fontsize=10);ax.set_ylim(3.45,-.45);ax.grid(axis='x',alpha=.15)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,ncol=5,loc='upper center',bbox_to_anchor=(.57,1.03),frameon=False)
    fig.subplots_adjust(left=.28,right=.99,top=.78,bottom=.18,wspace=.14)
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'csafe_identification_utility.{ext}',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.6),gridspec_kw={'width_ratios':[1,1.15]})
    handcolors={'hc183':'#0072B2','stroke':'#D55E00','hc183_adj':'#009E73'}
    for j,model in enumerate(HANDS):
        subset=hci[hci.hand_model==model].set_index('condition').reindex(CONDITIONS)
        vals=100*subset.accuracy.to_numpy();error=np.stack([vals-100*subset.ci_lo.to_numpy(),100*subset.ci_hi.to_numpy()-vals])
        axes[0].errorbar(vals,np.arange(3)+(j-1)*.15,xerr=np.maximum(error,0),fmt='o',ms=4,capsize=2,
            color=handcolors[model],label=HANDS[model])
    axes[0].set(yticks=np.arange(3),yticklabels=['S2, same prompt','S2, different prompt','S3, different prompt'],
        xlabel='Handedness balanced accuracy (%)',ylim=(2.45,-.45),xlim=(0,100))
    axes[0].axvline(50,color='#888',ls='--',lw=.8);axes[0].legend(loc='lower left',bbox_to_anchor=(-.12,1.16),frameon=False,ncol=3,fontsize=8)
    for method,offset in [('fixed_fusion_lambda1',-.11),('nested_fusion',.11)]:
        for i,(condition,rep) in enumerate(cases):
            group=small.xs((condition,rep),level=('condition','rep'))
            delta=100*(group[method]-group.identity_only)
            axes[1].scatter(delta,np.full(len(delta),i+offset)+np.linspace(-.05,.05,len(delta)),s=11,
                color=COLORS[method],alpha=.65,edgecolors='none',label=METHODS[method] if i==0 else None)
    axes[1].set(yticks=np.arange(4),yticklabels=labels,xlabel='Identification change (points)',ylim=(3.45,-.45))
    axes[1].axvline(0,color='#888',ls='--',lw=.8);axes[1].legend(loc='lower left',bbox_to_anchor=(.05,1.16),frameon=False,ncol=2,fontsize=8)
    for ax,title in zip(axes,['(a) Auxiliary predictor checks','(b) Overlapping-gallery sensitivity']):
        ax.set_title(title,loc='left',fontsize=10,pad=7);ax.grid(axis='x',alpha=.15)
    fig.subplots_adjust(left=.16,right=.98,top=.77,bottom=.18,wspace=.62)
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'csafe_attribute_robustness.{ext}',dpi=220)
    plt.close(fig)
    print(OUT/'RESULTS.md')


if __name__=='__main__':main()
