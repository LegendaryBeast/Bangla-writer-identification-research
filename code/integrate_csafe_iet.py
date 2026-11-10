#!/usr/bin/env python3
"""Integrate audited CSAFE validation while preserving the approved IET framing."""
import json
import re
import shutil
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper/iet_biometrics'
RESULTS=ROOT/'results/csafe_replication_2026-11-08'
CONDITIONS=('S2_LND_r01','S2_WOZ_r01','S3_WOZ_r01')
REPS=('resnet18','vit_b_16','hc183','stroke')
NAMES={'resnet18':'ResNet-18','vit_b_16':'ViT-B/16','hc183':'HC-183','stroke':'STROKE','hc183_adj':'HC-183 adj.'}


def pct(x):return f'{100*x:.2f}'


def write_tables():
    S=pd.read_csv(RESULTS/'id_summary.csv');H=pd.read_csv(RESULTS/'handedness_conditional_intervals.csv')
    T=pd.read_csv(RESULTS/'paired_tests.csv')
    full=S[S.cohort=='full_balanced'].pivot(index=['condition','hand_model','rep'],columns='method',values='top1')
    main=[]
    for condition,word in zip(CONDITIONS[:2],('Same','Different')):
        for rep in REPS[:2]:
            r=full.loc[(condition,'hc183',rep)]
            main.append(' & '.join([word,NAMES[rep]]+[pct(r[m]) for m in ('identity_only','oracle_known_hand','predicted_hard_filter','fixed_fusion_lambda1','nested_fusion')])+r' \\')
    (PAPER/'tables/csafe_main_rows.tex').write_text('\n'.join(main)+'\n')
    hrows=[]
    for condition in CONDITIONS:
        for model in ('hc183','stroke','hc183_adj'):
            r=H[(H.condition==condition)&(H.hand_model==model)].iloc[0]
            hrows.append(' & '.join([condition[:2]+'/'+('LND' if 'LND' in condition else 'WOZ'),NAMES[model],pct(r.accuracy),
                f'[{pct(r.ci_lo)}, {pct(r.ci_hi)}]',f'{r.nll:.3f}',f'{r.brier:.3f}',f'{r.raw_auc:.3f}',f'{r.calibrated_auc:.3f}'])+r' \\')
    (PAPER/'tables/csafe_hand_rows.tex').write_text('\n'.join(hrows)+'\n')
    tests=[]
    for r in T[T.primary].itertuples():
        tests.append(' & '.join(['Same' if 'LND' in r.condition else 'Different',NAMES[r.rep],
            'Fixed' if r.method=='fixed_fusion_lambda1' else 'Nested',f'{100*r.delta_top1:+.2f}',
            f'[{100*r.ci_lo:.2f}, {100*r.ci_hi:.2f}]',f'{r.p_holm_primary:.4f}'])+r' \\')
    (PAPER/'tables/csafe_test_rows.tex').write_text('\n'.join(tests)+'\n')
    rows=[]
    for condition in CONDITIONS:
        for model in ('hc183','stroke','hc183_adj'):
            for rep in REPS:
                r=full.loc[(condition,model,rep)]
                rows.append(' & '.join([condition[:2]+'/'+('LND' if 'LND' in condition else 'WOZ'),NAMES[model],NAMES[rep]]+
                    [pct(r[m]) for m in ('identity_only','oracle_known_hand','predicted_hard_filter','fixed_fusion_lambda1','nested_fusion')])+r' \\')
    (PAPER/'tables/csafe_full_rows.tex').write_text('\n'.join(rows)+'\n')
    # Editable complete grid accompanies the submission separately.
    S.to_csv(PAPER/'tables/CSAFE_complete_196_settings.csv',index=False)


def external_methods():
    return r'''
\subsection{External validation across capture sessions}
\label{sec:csafemethods}
The CSAFE Handwriting Database provides an independent English-language evaluation \cite{crawford2020database,crawford2026csafe}. Its version-2 release contains 475 writers; this differs from the 90-writer release described in the original article. The released survey identifies 48 left-handed, 423 right-handed, three ambidextrous and one unreported-hand writer. We include all 48 left-handed writers and 48 right-handed writers sampled with seed 20261108. Each selected writer contributes six session-1 pages (London Letter, LND, and Wizard of Oz, WOZ, each repeated three times) for auxiliary-model training. One LND page is the sole identity-gallery reference. Queries are separate session-2 pages using LND or WOZ, plus a session-3 WOZ page for secondary evaluation. Thus the primary conditions distinguish matching prompts from cross-prompt identification across sessions.

The original whole-unit extraction rules, SVM $C=0.1$, checkpoints and fusion grids are retained. We evaluate ResNet-18, ViT-B/16, HC-183 and STROKE identity features, with HC-183 as the primary handedness predictor and the two original sensitivity predictors. All 864 selected pages pass the validity rules. Each outer query writer's six training pages are excluded from its attribute model and calibrator. Nested development probabilities additionally exclude their validation writers; temperature and weight targets exclude the outer query writer. Known gallery references and hands remain permitted metadata. Numerical score scaling preserves the original logistic-calibration objective; Supporting Section S13 documents the convergence check and its implementation amendment.

Eight primary contrasts compare fixed and nested fusion with identity only for ResNet-18 and ViT-B/16 in both session-2 conditions, with Holm correction across all eight. Session 3, alternate attribute models and 20 refitted balanced 21/21 galleries are secondary checks. Confidence intervals condition on the fitted models and enrolled gallery. The CSAFE design was specified before inspecting its performance outcomes; it is an extension of the existing study, not a globally preregistered dataset-sampling plan. This is within-dataset refitting on an external cohort, not transfer of a frozen Bangla-trained predictor.
'''


def external_results():
    rows=(PAPER/'tables/csafe_main_rows.tex').read_text()
    return r'''
\subsection{External validation on CSAFE}
\label{sec:csafeexternal}
The primary handedness predictor reaches 67.71\% accuracy on session-2 LND queries (conditional 95\% interval 58.33--77.08\%), 58.33\% on session-2 WOZ (48.96--67.71\%), and 65.62\% on session-3 WOZ (56.25--75.00\%). Balanced and ordinary accuracy coincide in this 48/48 cohort. These values separate attribute predictability from the identity-ranking changes in Table~\ref{tab:csafemain} and Fig.~\ref{fig:csafeutility}.

With matching prompts across sessions, known handedness increases ResNet-18 identification from 61.46\% to 68.75\%, whereas fixed and nested fusion both achieve 62.50\%. ViT-B/16 changes from 64.58\% to 62.50\% under either fusion rule. With different prompts, ResNet-18 changes from 54.17\% to 48.96\% under fixed fusion and remains at 54.17\% under nested fusion; ViT-B/16 changes from 55.21\% to 54.17\%. None of the eight primary contrasts is significant after Holm correction (all adjusted $p\geq0.4888$). This does not establish equivalence. Individual percentile-bootstrap intervals are conditional, are not simultaneous, and are not inversions of the discrete sign-flip tests.

Hard filtering reduces primary neural identification by 17.71--25.00 percentage points across the two session-2 conditions. The STROKE handedness sensitivity model attains 67.71\% and 60.42\%, while adjusted HC-183 attains 34.38\% in both; these differences are retained in the supplement. Negative fitted Platt slopes occur in 24 of 96 adjusted-model outer fits, so its pooled raw and calibrated score sets differ substantially. This sensitivity failure does not identify a causal acquisition mechanism. Session-3 comparisons and the 20 overlapping gallery subsamples further characterize variability without adding independent dataset replications.

\begin{table*}[!t]
\caption{CSAFE session-2 top-1 identification accuracy (\%) with the primary HC-183 handedness model and one session-1 reference per writer. The gallery contains 48 writers of each hand. ``Same'' and ``Different'' refer to the query prompt relative to the LND reference. All eight fixed/nested primary contrasts have Holm-adjusted $p\geq0.4888$.}\label{tab:csafemain}
\centering\small
\begin{tabular}{llrrrrr}\toprule
Prompt & Identity features & Identity only & Known hand & Hard filter & Fixed & Nested\\\midrule
''' + rows + r'''\bottomrule
\end{tabular}
\end{table*}

\begin{figure*}[!t]
\centering
\includegraphics[width=\textwidth]{assets/csafe_identification_utility.pdf}
\caption{External CSAFE validation using separate gallery/query sessions and the primary HC-183 handedness model. (a) Top-1 identification in the 96-writer gallery. (b) Fixed and nested changes from identity only with conditional 95\% writer-bootstrap intervals. Both session-2 conditions belong to the eight-comparison primary family. Intervals retain fitted models and the gallery; they are not multiplicity-adjusted.}\label{fig:csafeutility}
\end{figure*}
'''


def external_supplement():
    hand=(PAPER/'tables/csafe_hand_rows.tex').read_text()
    tests=(PAPER/'tables/csafe_test_rows.tex').read_text()
    full=(PAPER/'tables/csafe_full_rows.tex').read_text()
    return r'''
\section{External CSAFE validation}
\label{sec:csafesupplement}
\subsection{Source, cohort and capture conditions}
The secondary data are the \href{https://doi.org/10.25380/iastate.10062203.v2}{CSAFE Handwriting Database, version 2}, credited to Amy Crawford, Anyesha Ray, Alicia Carriquiry, James Kruse and Marc Peterson, under CC BY 4.0. Crawford, Ray and Carriquiry describe the original 90-writer release in \emph{Data in Brief} 28 (2020), 105059, doi:10.1016/j.dib.2019.105059. The version-2 survey instead has 475 unique writer IDs: 48 left-handed, 423 right-handed, three ambidextrous and one missing hand. Every survey ID matches nine original image filenames in each of three session archives (12,825 images in total).

The analysis includes all 48 left-handed writers and 48 right-handed writers sampled without replacement from sorted IDs using NumPy \texttt{default\_rng(20261108)}. The four ambiguous/missing-hand records are excluded before sampling. No writer is removed according to classification accuracy. The selected IDs and source-image hashes are provided in the archive. Each writer supplies the nine pages in Table~\ref{tab:csafepages}. Printed headers have been removed by the provider. Distinct sessions and source pages are preserved; no crop of a gallery page is used as a query.

\begin{table}[ht]
\caption{Source pages selected for each of the 96 CSAFE writers. LND is the London Letter; WOZ is the Wizard of Oz passage.}\label{tab:csafepages}
\centering\small
\begin{tabular}{llrl}\toprule
Session & Prompt/repetition & Pages & Role\\\midrule
1 & LND, repetition 1 & 96 & Gallery and auxiliary training\\
1 & LND, repetitions 2--3 & 192 & Auxiliary training only\\
1 & WOZ, repetitions 1--3 & 288 & Auxiliary training only\\
2 & LND, repetition 1 & 96 & Primary, same prompt\\
2 & WOZ, repetition 1 & 96 & Primary, different prompt\\
3 & WOZ, repetition 1 & 96 & Secondary, different prompt\\\bottomrule
\end{tabular}
\end{table}

Where enrollment-relative timing is available, session 2 follows session 1 for 94/96 writers (median 33 days; range 18--252), and session 3 follows session 1 for 95/96 (median 68; range 42--272). Missing timing values are retained under their original session labels. These records do not support claiming that every observed interval is at least 21 days.

\subsection{Features and nested fitting}
The original $n=1$ extraction is applied to whole handwriting pages, with $k=1$ gallery reference. HC-183, STROKE and the eight nuisance covariates are extracted on all 864 pages. Frozen ResNet-18 and ViT-B/16 embeddings are needed for the 96 gallery and 288 query pages. Auxiliary-only pages have no neural embedding; their intentional missing entries never enter identity scoring. All pages pass the common validity rules. Exact checkpoint names, URLs, hashes, source hashes and accelerator information are recorded. A fixed-sample comparison against the original extraction function, including a prefetched auxiliary page, gives zero difference after the original float32 storage in all compared features.

For the full gallery, an outer attribute fit uses six session-1 pages from each of the other 95 writers. Its calibrator uses their five-fold writer-group out-of-fold SVM scores. Five development folds inside this outer exclusion regenerate attribute evidence, with another five writer-group folds for each development calibrator. Thus both the outer writer and each development validation group are absent from the corresponding attribute fit. All 96 known gallery hands and selected references remain available as candidate metadata. Temperature and fusion weight are selected separately for each query condition using other query writers. The grids and the $C=0.1$ linear SVM are unchanged. The same fitting rules are refitted within each balanced 42-writer gallery.

The primary HC-183 model and the original STROKE and adjusted HC-183 sensitivity models are fixed before CSAFE outcome inspection. Eight primary comparisons concern fixed/nested versus identity-only ranking for ResNet-18 and ViT-B/16 in both session-2 conditions. Holm adjustment covers all eight. Session 3, alternate predictors, handcrafted identity representations and overlapping galleries are secondary descriptive analyses. Local specification of these CSAFE contrasts is not a globally preregistered external-dataset sampling plan.

\subsection{Numerically equivalent logistic calibration}
Initial fitting produced L-BFGS convergence warnings for the adjusted sensitivity model. Before inspecting CSAFE performance tables, an implementation amendment centered and scaled the training out-of-fold SVM scores while retaining the original logistic objective. Let $z_i=(s_i-\bar s)/\sigma_s$ and let $C_{\rm cal}=10^4$. The fitted objective is
\begin{equation}
L(a,b)=\frac1N\sum_i\bigl[\log(1+e^{a z_i+b})-y_i(a z_i+b)\bigr]+\frac{a^2}{2C_{\rm cal}N\sigma_s^2}.
\label{eq:csafecalibration}
\end{equation}
The raw-score slope is $a/\sigma_s$; the intercept is unpenalized. The transformed penalty therefore equals the original raw-slope L2 penalty. A zero score standard deviation uses scale one. L-BFGS-B uses analytic gradients, a maximum of 2,000 iterations and a gradient certificate of $10^{-7}$. All primary and sensitivity fits are regenerated with this implementation; no feature, writer, SVM penalty, endpoint or fusion grid is selected from outcomes. The largest observed gradient infinity norm is $9.51\times10^{-9}$, with every optimizer reporting success. An independent numerical check matches the original logistic estimator's probabilities within $2.27\times10^{-11}$ on a well-conditioned reference case and verifies the raw/transformed objective equality. Numerical optimizer convergence is distinct from predictive calibration quality.

\subsection{Attribute and identification results}
Table~\ref{tab:csafehand} reports all nine full-gallery handedness conditions. The balanced cohort makes ordinary accuracy equal to balanced accuracy. Its intervals resample writers within hand and retain the fitted model. Raw and calibrated AUCs pool different writer-held-out score sets. Adjusted HC-183 has negative Platt slopes in 24/96 outer fits, compared with none for the unadjusted models. This diagnostic does not prove a specific acquisition or leakage mechanism.

\begin{table}[ht]
\caption{CSAFE handedness prediction on the full 48/48 cohort. Intervals are conditional 95\% writer-bootstrap intervals; NLL and Brier scores are losses. AUC columns concern raw and calibrated scores.}\label{tab:csafehand}
\centering\footnotesize
\begin{tabular}{llrlrrrr}\toprule
Query & Model & Acc. (\%) & 95\% interval & NLL & Brier & Raw AUC & Cal. AUC\\\midrule
''' + hand + r'''\bottomrule
\end{tabular}
\end{table}

Table~\ref{tab:csafetests} gives the eight primary paired contrasts. The percentile-bootstrap intervals are conditional, not simultaneous and not inversions of the discrete paired sign-flip tests. An interval can exclude zero while the Holm-adjusted test does not reject; a $[0,0]$ interval only describes identical correctness outcomes of the fitted rules. No primary adjusted test rejects at 0.05. Table~\ref{tab:csafefull} retains every full-gallery comparator, including the secondary session-3 and alternative-predictor results.

\begin{table}[ht]
\caption{Primary CSAFE fixed/nested changes from identity only, in percentage points. Prompt relation is to the session-1 LND gallery; all queries are session 2.}\label{tab:csafetests}
\centering\small
\begin{tabular}{lllrlr}\toprule
Prompt & ID features & Fusion & Change & 95\% interval & Holm $p$\\\midrule
''' + tests + r'''\bottomrule
\end{tabular}
\end{table}

\begin{longtable}{lllrrrrr}
\caption{Complete 96-writer CSAFE top-1 comparisons (\%). ID is identity only; Known supplies the true query hand; Filter uses the predicted hand. Alternate attribute models and session 3 are secondary.}\label{tab:csafefull}\\
\toprule Query & Hand model & ID features & ID & Known & Filter & Fixed & Nested\\\midrule
\endfirsthead
\toprule Query & Hand model & ID features & ID & Known & Filter & Fixed & Nested\\\midrule
\endhead
''' + full + r'''\bottomrule
\end{longtable}

\subsection{Gallery sensitivity and reproducibility}
Twenty seeded balanced 21/21 galleries are drawn from the selected 96-writer pool and refit independently for the primary HC-183 predictor. Their overlapping memberships make them dependent sensitivity samples. Figure~\ref{fig:csafesensitivity} shows their changes alongside the attribute checks. They characterize gallery composition within this pool, not a population-level distribution or 20 independent dataset replications. The complete 196-setting summary, all 50,880 writer-method outcomes and selected parameters accompany the external reproducibility archive.

\begin{figure}[ht]
\centering
\includegraphics[width=\linewidth]{assets/csafe_attribute_robustness.pdf}
\caption{CSAFE secondary checks. (a) All predefined attribute models and query conditions, with conditional writer-bootstrap 95\% intervals. (b) Fixed/nested identification changes across 20 overlapping balanced 42-writer galleries. Points are dependent descriptive subsamples.}\label{fig:csafesensitivity}
\end{figure}

The archive \texttt{csafe-replication-2026-11-08.zip} supplies the frozen features, original and amended protocol specifications, source/member hashes, checkpoint references, fitting manifests, numerical outputs and figure/table scripts. Raw CSAFE images and neural checkpoint binaries are obtained separately. The downloaded survey MD5 and every selected image's source ZIP size/CRC and SHA-256 are checked; the complete multi-gigabyte source ZIP MD5s are not claimed to have been verified after partial retrieval. Validation audits 1,128 outer exclusions, 6,768 final/development attribute models and 33,840 calibration folds, verifies all image roles and reported accuracies, and checks the filtering identity and exact fusion-margin reconstruction. A fresh run from the extracted package, with supplied fitting caches removed, reproduces all nine core numerical and manifest outputs exactly. This is computational reproducibility, not an additional empirical replication.

From the archive root, saved-feature reproduction uses:
\begin{verbatim}
python code/test_csafe_calibration.py
python code/run_csafe_replication.py --jobs 4
python code/validate_csafe_replication.py --saved-only
python code/summarize_csafe_replication.py
\end{verbatim}
Move the supplied \texttt{attribute\_cache/} aside to force fresh fitting. For source-image reproduction, run \texttt{acquire\_csafe\_replication.py}, then \texttt{extract\_csafe\_features.py}; full validation omits \texttt{--saved-only}. Saved-only validation covers numerical outputs; the original full audit additionally verifies images and checkpoints.
'''


def main():
    assert json.loads((RESULTS/'validation.json').read_text())['passed']
    assert json.loads((RESULTS/'independent_recompute_validation.json').read_text())['passed']
    write_tables()
    for stem in ('csafe_identification_utility','csafe_attribute_robustness'):
        for ext in ('pdf','png'):shutil.copy2(RESULTS/f'{stem}.{ext}',PAPER/'assets'/f'{stem}.{ext}')
    body=(PAPER/'manuscript_body.tex').read_text()
    if r'\label{sec:csafeexternal}' not in body:
        anchor=r'\section{Related work}'
        body=body.replace(anchor,'The BanglaHand analysis is complemented by an external CSAFE evaluation of the same utility question on independent writers, separate source pages and capture sessions \\cite{crawford2020database,crawford2026csafe}.\n\n'+anchor,1)
        body=body.replace(r'\section{Results}',external_methods()+'\n'+r'\section{Results}',1)
        body=body.replace(r'\section{Discussion}',external_results()+'\n'+r'\section{Discussion}',1)
        body=body.replace('In this cohort, neither calibrated fixed fusion nor properly nested learned weighting establishes a gain in the designated comparisons.',
            'In BanglaHand, neither calibrated fixed fusion nor properly nested learned weighting establishes a gain in the designated comparisons. The CSAFE validation extends this distinction to a separate population, English writing and distinct source pages and sessions: even the 67.71\\% same-prompt handedness predictor yields no reliably demonstrated primary fusion gain. The cross-prompt and session-3 differences show why this result must remain tied to its declared query conditions and fitted rules.')
        body=body.replace('Broader transfer requires other writers and acquisition conditions; the companion study shares this cohort and is not independent validation.',
            'The companion study shares this cohort and is not independent validation. CSAFE supplies an independent population and separate-session scans, extending the evaluation beyond the original photographic acquisition; it does not establish transfer to all scripts, devices or deployment conditions.')
        body=body.replace('The uncertainty analyses characterize internal stability conditional on the enrolled gallery and fitted predictions. They do not propagate new-gallery, new-session or full-pipeline-refitting variability. Repeated calibration partitions vary one fitting stage; writer omission changes evaluation averaging while retaining the gallery. These distinctions specify which sources of variation future replication should assess.',
            'The uncertainty intervals condition on the enrolled gallery and fitted predictions; they do not propagate full model-refitting or population-sampling variability. BanglaHand calibration partitions vary one fitting stage and writer omission changes evaluation averaging while retaining its gallery. CSAFE tests new sessions and refits models in overlapping gallery subsamples, but those dependent subsamples do not provide independent-cohort confidence intervals. Its whole-page, one-reference conditions do not reproduce every BanglaHand text-size/reference setting, and same-prompt matching must be distinguished from cross-prompt identification.')
        body=body.replace('On this Bangla collection, correct query and gallery handedness improves writer ranking, whereas fixed and properly nested learned-weight fusion of the tested query predictor produce no reliably demonstrated gain over identification alone, and hard filtering causes substantial losses.',
            'On BanglaHand, correct query and gallery handedness improves writer ranking, whereas fixed and properly nested learned-weight fusion of the tested query predictor produce no reliably demonstrated gain over identification alone, and hard filtering causes substantial losses. Independent CSAFE writers and capture sessions reproduce the distinction between oracle opportunity and achieved predicted-attribute utility in the tested comparisons.')
        body=body.replace('Independent and cross-session studies can assess transfer to other populations and acquisition conditions.',
            'Further datasets, fitting objectives and deployment conditions can assess the remaining limits of transfer; the present results do not establish a universal absence of handedness utility.')
        body=body.replace('Code, frozen features, setting manifests, numerical results and self-contained supplementary methods are archived as',
            'The BanglaHand code, frozen features, setting manifests, numerical results and self-contained supplementary methods are archived as')
        availability=(' The separate CSAFE extension, \\texttt{csafe-replication-2026-11-08.zip}, is supplied with the submission and contains its complete code, frozen features, protocols, fitting manifests and reproduction instructions. It is distinct from public release v1.6.0. CSAFE version 2 is obtained separately at doi:10.25380/iastate.10062203.v2 under CC BY 4.0 \\cite{crawford2026csafe}; original images and checkpoint binaries are excluded from the analysis archive.')
        body=body.replace(r'\section*{Author contributions}',availability+'\n\n'+r'\section*{Author contributions}',1)
        body=body.replace('Informed consent was obtained voluntarily from all participants before data collection.',
            'For BanglaHand, informed consent was obtained voluntarily from all participants before data collection.')
        body=body.replace(r'\section*{Funding}',
            'The external evaluation reuses publicly released, pseudonymous CSAFE images and handedness labels under CC BY 4.0 \\cite{crawford2020database,crawford2026csafe}. No additional participants were recruited or contacted for that secondary analysis.\n\n'+r'\section*{Funding}',1)
        body=body.replace('the complete 120-setting grid, and reproduction instructions.',
            'the complete 120-setting BanglaHand grid, the external CSAFE protocol and full results, and reproduction instructions.')
    (PAPER/'manuscript_body.tex').write_text(body)
    meta=json.loads((PAPER/'metadata.json').read_text())
    abstract=meta['abstract']
    if 'CSAFE' not in abstract:
        abstract=abstract.replace('Robustness analyses characterize sensitivity within this cohort.',
            'An external CSAFE evaluation uses 96 writers and separate gallery/query capture sessions. Handedness accuracy is 67.7\\% for same-prompt and 58.3\\% for cross-prompt session-2 queries; none of eight primary fusion comparisons establishes a Holm-adjusted gain. Session-3 and gallery-subsample analyses provide secondary checks.')
    meta.update(abstract=abstract,new_experiments=True,new_datasets=True,external_dataset='CSAFE Handwriting Database v2',
        framing='Original controlled framework and title retained; CSAFE is external validation, not a paper-wide empirical-study reframing.')
    (PAPER/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');(PAPER/'abstract.tex').write_text(abstract+'\n')
    maintex=(PAPER/'manuscript.tex').read_text();header=maintex.split(r'\section{Introduction}',1)[0]
    header=re.sub(r'\\abstract\[ABSTRACT\]\{.*?\}\n\\maketitle',lambda m:r'\abstract[ABSTRACT]{'+abstract+'}\n'+r'\maketitle',header,flags=re.S)
    (PAPER/'manuscript.tex').write_text(header+body+'\n\\bibliography{refs}\n\\end{document}\n')
    sup=(PAPER/'supporting_information.tex').read_text()
    if r'\label{sec:csafesupplement}' not in sup:
        sup=sup.replace('Analyses are retrospective and were not preregistered.',
            'The primary BanglaHand analyses are retrospective and were not preregistered. Section S13 documents the separately specified CSAFE extension and its numerical implementation amendment.')
        sup=sup.replace('Independent-cohort and cross-session replication can assess transfer beyond these acquisition conditions.',
            'The external CSAFE evaluation in Section S13 assesses a separate cohort and capture sessions; its conditional intervals still do not propagate full fitting uncertainty.')
        sup=sup.replace(r'\end{document}',external_supplement()+'\n'+r'\end{document}')
    (PAPER/'supporting_information.tex').write_text(sup)
    refs=(PAPER/'refs.bib').read_text()
    if 'crawford2020database' not in refs:
        refs+=r'''
@article{crawford2020database,
 author={Amy Crawford and Anyesha Ray and Alicia Carriquiry},
 title={A database of handwriting samples for applications in forensic statistics},
 journal={Data in Brief}, volume={28}, pages={105059}, year={2020},
 doi={10.1016/j.dib.2019.105059}
}
@misc{crawford2026csafe,
 author={Amy Crawford and Anyesha Ray and Alicia Carriquiry and James Kruse and Marc Peterson},
 title={{CSAFE} Handwriting Database}, year={2026},
 howpublished={Iowa State University DataShare, version 2},
 doi={10.25380/iastate.10062203.v2},
 url={https://doi.org/10.25380/iastate.10062203.v2},
 note={Released 23 June 2026}
}
'''
    (PAPER/'refs.bib').write_text(refs)
    cover=(PAPER/'cover_letter.md').read_text()
    if '96 CSAFE' not in cover:
        cover=cover.replace('The distinct companion preprint,',
            'An external validation uses 96 CSAFE writers, balanced by handedness, with gallery and query pages from different capture sessions. The primary predictor achieves 67.71% same-prompt and 58.33% cross-prompt session-2 handedness accuracy, yet none of eight primary fixed/nested identification comparisons establishes a Holm-adjusted gain. This tests the central methodological distinction on an independent population and acquisition regime; it does not imply a universal null effect.\n\nThe distinct companion preprint,')
        cover=cover.replace('Supporting information provides the complete 120-setting grid and reproduction details.',
            'Supporting information provides the complete BanglaHand grid, external protocol, all CSAFE results and reproduction details. A separate versioned CSAFE code/features/results archive accompanies the submission; CSAFE source data are available at doi:10.25380/iastate.10062203.v2.')
    (PAPER/'cover_letter.md').write_text(cover)
    print('Integrated CSAFE while preserving the current title, framing, original figures and PRL files.')


if __name__=='__main__':main()
