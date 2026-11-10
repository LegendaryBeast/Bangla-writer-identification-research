#!/usr/bin/env python3
"""LaTeX blocks for the revision sections of the supporting information.

Writes paper/iet_biometrics/tables/p1_*.tex from saved outputs only. The blocks are
inserted verbatim into supporting_information.tex so that the Markdown/Word export
(which does not resolve \\input) receives the same content.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / 'results/p1_revision_2026-11-09'
TAB = ROOT / 'paper/iet_biometrics/tables'
NAMES = {'stroke': 'STROKE', 'hc183': 'HC-183', 'resnet18': 'ResNet-18', 'efficientnet_b0': 'EffNet-B0',
         'vit_b_16': 'ViT-B/16', 'stroke_cnn': 'Stroke-CNN'}
COND = {'S2_LND_r01': 'Same', 'S2_WOZ_r01': 'Different', 'S3_WOZ_r01': 'S3 different'}


def f(x, d=2, sign=True):
    return f'{x:+.{d}f}' if sign else f'{x:.{d}f}'


def ci(lo, hi, d=2):
    return f'[{lo:+.{d}f}, {hi:+.{d}f}]'


def designated():
    fx = pd.read_csv(ROOT / 'results/wid_revision_2026-11-08/id_paired_tests.csv')
    fx = fx[fx.primary == True]  # noqa: E712
    ns = pd.read_csv(ROOT / 'results/wid_fusion_robustness_2026-11-08/nested_paired_tests.csv')
    ns = ns[ns.primary]
    rows = []
    for (n, k, lab) in ((1, 1, 'Paragraph, $k=1$'), (4, 6, 'Quarter, $k=6$')):
        for rep in ('resnet18', 'vit_b_16'):
            a = fx[(fx.rep == rep) & (fx.n == n) & (fx.k == k)].iloc[0]
            b = ns[(ns.rep == rep) & (ns.n == n) & (ns.k == k) & (ns.contrast == 'nested_minus_none')].iloc[0]
            c = ns[(ns.rep == rep) & (ns.n == n) & (ns.k == k) & (ns.contrast == 'nested_minus_fixed')].iloc[0]
            rows.append(' & '.join([lab, NAMES[rep], f(100 * a.diff_top1) + ' ' + ci(100 * a.ci_lo, 100 * a.ci_hi),
                                    f'{a.p_holm:.3f}', f(100 * b.estimate) + ' ' + ci(100 * b.ci_lo, 100 * b.ci_hi),
                                    f(100 * c.estimate) + ' ' + ci(100 * c.ci_lo, 100 * c.ci_hi),
                                    f'{b.p_holm:.3f}', f'{c.p_holm:.3f}']) + r' \\')
    return '\n'.join([r'\begin{table}[ht]', r'\caption{Four designated BanglaHand settings (self-chosen queries). Entries are paired top-1 differences in percentage points with conditional 95\% writer-bootstrap intervals. Fixed and Nested are differences from no fusion. Holm adjustment covers the four fixed-fusion comparisons and, separately, the eight learned-weight comparisons (four settings, two comparators).}\label{tab:designated}',
                      r'\centering\footnotesize\setlength{\tabcolsep}{2.2pt}', r'\begin{tabular}{llrrrrrr}\toprule',
                      r' & & & Holm $p$ & & & \multicolumn{2}{c}{Holm $p$, nested vs}\\\cmidrule(lr){7-8}',
                      r'Query & Features & Fixed [95\% CI] & fixed & Nested [95\% CI] & Nested$-$fixed [95\% CI] & None & Fixed\\\midrule']
                     + rows + [r'\bottomrule', r'\end{tabular}', r'\end{table}'])


def precision():
    P = pd.read_csv(P1 / 'precision_equivalence.csv')
    label = {'all 120 settings': 'All 120 settings',
             'resnet18 own n=1 k=1': 'ResNet-18, paragraph, $k=1$', 'vit_b_16 own n=1 k=1': 'ViT-B/16, paragraph, $k=1$',
             'resnet18 own n=4 k=6': 'ResNet-18, quarter, $k=6$', 'vit_b_16 own n=4 k=6': 'ViT-B/16, quarter, $k=6$',
             'resnet18 S2_LND_r01': 'ResNet-18, same prompt', 'vit_b_16 S2_LND_r01': 'ViT-B/16, same prompt',
             'resnet18 S2_WOZ_r01': 'ResNet-18, different prompt', 'vit_b_16 S2_WOZ_r01': 'ViT-B/16, different prompt'}
    rows, last = [], None
    for r in P.itertuples():
        if r.dataset != last:
            if last is not None:
                rows.append(r'\midrule')
            rows.append(r'\multicolumn{8}{l}{\emph{' + ('BanglaHand, 42 writers' if r.dataset == 'BanglaHand' else 'CSAFE, 96 writers, session 2') + r'}}\\')
            last = r.dataset
        kind = 'Fixed' if r.contrast.startswith('fixed') else 'Nested'
        eq = 'none changed' if r.degenerate else f'{r.smallest_equivalence_margin_pts:.2f}'
        mde = '--' if r.degenerate else f'{r.mde80_pts:.2f}'
        share = '--' if r.degenerate or r.oracle_gain_pts <= 0 else f'{100 * max(r.excludable_gain_share_of_oracle, 0):.0f}\\%'
        rows.append(' & '.join([label[r.setting], kind, f(r.estimate_pts), ci(r.ci90_lo, r.ci90_hi), eq, mde,
                                f(r.oracle_gain_pts), share]) + r' \\')
    return '\n'.join([r'\begin{table}[ht]', r'\caption{Post hoc precision of the primary fusion contrasts. $\Delta^*$ is the smallest symmetric margin for which the 90\% conditional writer-bootstrap interval lies inside $(-\Delta^*,\Delta^*)$, i.e.\ two one-sided tests at $\alpha=.05$. MDE$_{80}$ is the minimum detectable effect at 80\% power, $2.80\times$ the bootstrap standard error. The last column is the 90\% upper bound as a share of the known-hand gain in the same setting. All values are percentage points except the share. ``None changed'' marks a contrast in which no writer decision changed, so the interval is degenerate.}\label{tab:precision}',
                      r'\centering\footnotesize\setlength{\tabcolsep}{3.5pt}', r'\begin{tabular}{llrrrrrr}\toprule',
                      r'Setting & Fusion & Change & 90\% CI & $\Delta^*$ & MDE$_{80}$ & Known gain & Upper/known\\\midrule']
                     + rows + [r'\bottomrule', r'\end{tabular}', r'\end{table}'])


def decomposition():
    S = pd.read_csv(P1 / 'mechanism_bangla_settings.csv')
    J = pd.read_csv(P1 / 'mechanism_bangla_j1_by_rep.csv').set_index('rep')
    g = S.groupby('rep')[['gap_total_pts', 'j=1_gap_pts', 'j=2_gap_pts', 'j=3_gap_pts', 'j>=4_gap_pts']].mean()
    rows = []
    for rep in NAMES:
        r, j = g.loc[rep], J.loc[rep]
        rows.append(' & '.join([NAMES[rep], f(r.gap_total_pts), f(r['j=1_gap_pts']), f(r['j=2_gap_pts']), f(r['j=3_gap_pts']),
                                f(r['j>=4_gap_pts']), f'{100 * j.j1_queries_weight_share:.1f}',
                                f'{100 * j.j1_opposite_hand_share:.1f} [{100 * j.ci95_lo:.1f}, {100 * j.ci95_hi:.1f}]']) + r' \\')
    m = S[['gap_total_pts', 'j=1_gap_pts', 'j=2_gap_pts', 'j=3_gap_pts', 'j>=4_gap_pts']].mean()
    rows.append(r'\midrule')
    rows.append(' & '.join(['All settings', f(m.gap_total_pts), f(m['j=1_gap_pts']), f(m['j=2_gap_pts']), f(m['j=3_gap_pts']),
                            f(m['j>=4_gap_pts']), '', '']) + r' \\')
    C = pd.read_csv(P1 / 'mechanism_csafe_settings.csv')
    C = C[C.condition != 'S3_WOZ_r01']
    cp = pd.read_csv(P1 / 'mechanism_csafe_j1_pooled.csv')
    neural = cp[cp.reps.str.startswith('neural')].iloc[0]
    allr = cp[cp.reps == 'all four'].iloc[0]
    cm = C[['gap_total_pts', 'j=1_gap_pts', 'j=2_gap_pts', 'j=3_gap_pts', 'j>=4_gap_pts']].mean()
    cn = C[C.rep.isin(['resnet18', 'vit_b_16'])][['gap_total_pts', 'j=1_gap_pts', 'j=2_gap_pts', 'j=3_gap_pts', 'j>=4_gap_pts']].mean()
    rows.append(r'\midrule')
    rows.append(' & '.join(['CSAFE, ResNet-18 and ViT-B/16', f(cn.gap_total_pts), f(cn['j=1_gap_pts']), f(cn['j=2_gap_pts']),
                            f(cn['j=3_gap_pts']), f(cn['j>=4_gap_pts']), '',
                            f'{100 * neural.j1_opposite_hand_share:.1f} [{100 * neural.ci95_lo:.1f}, {100 * neural.ci95_hi:.1f}]']) + r' \\')
    rows.append(' & '.join(['CSAFE, all four', f(cm.gap_total_pts), f(cm['j=1_gap_pts']), f(cm['j=2_gap_pts']),
                            f(cm['j=3_gap_pts']), f(cm['j>=4_gap_pts']), '',
                            f'{100 * allr.j1_opposite_hand_share:.1f} [{100 * allr.ci95_lo:.1f}, {100 * allr.ci95_hi:.1f}]']) + r' \\')
    return '\n'.join([r'\begin{table}[ht]', r'\caption{Decomposition of the known-hand minus random-partition gap by the number $j$ of impostors outranking the true writer (Eq.~\ref{eq:supdecomp}), in percentage points averaged over settings. The $j=1$ share is the writer-weighted fraction of queries with exactly one outranking impostor; the last column is the fraction of those impostors with the opposite hand, with conditional 95\% writer-bootstrap intervals. Random references: 51.2\% (BanglaHand) and 50.5\% (CSAFE). CSAFE rows average the two session-2 conditions; their $j=1$ shares pool decisions across conditions and representations.}\label{tab:decomposition}',
                      r'\centering\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{lrrrrrrr}\toprule',
                      r'Representation & Gap & $j=1$ & $j=2$ & $j=3$ & $j\geq4$ & $j=1$ share, \% & Opposite hand at $j=1$, \%\\\midrule']
                     + rows + [r'\bottomrule', r'\end{tabular}', r'\end{table}'])


def correction():
    old = pd.read_csv(ROOT / 'results/csafe_replication_2026-11-08/handedness_conditional_intervals.csv')
    new = pd.read_csv(P1 / 'csafe_corrected/handedness_conditional_intervals.csv')
    so = pd.read_csv(ROOT / 'results/csafe_replication_2026-11-08/id_summary.csv')
    sn = pd.read_csv(P1 / 'csafe_corrected/id_summary.csv')
    rows = []
    for cond in COND:
        o = old[(old.condition == cond) & (old.hand_model == 'hc183_adj')].iloc[0]
        n = new[(new.condition == cond) & (new.hand_model == 'hc183_adj')].iloc[0]
        def acc(x):
            return f'{100 * x.accuracy:.2f} [{100 * x.ci_lo:.2f}, {100 * x.ci_hi:.2f}]'
        rows.append(' & '.join([COND[cond].replace('Same', 'S2 same').replace('Different', 'S2 different'), acc(o), f'{o.calibrated_auc:.3f}',
                                acc(n), f'{n.calibrated_auc:.3f}', f'{n.raw_auc:.3f}']) + r' \\')
    return '\n'.join([r'\begin{table}[ht]', r'\caption{CSAFE adjusted HC-183 handedness prediction before and after the numerical correction. Accuracy in \% with conditional 95\% writer-bootstrap intervals. The raw-score AUC is unchanged in substance; only the calibration collapse is removed. All other CSAFE models and every primary result are unaffected.}\label{tab:correction}',
                      r'\centering\footnotesize', r'\begin{tabular}{lrrrrr}\toprule',
                      r' & \multicolumn{2}{c}{Original} & \multicolumn{3}{c}{Corrected}\\\cmidrule(lr){2-3}\cmidrule(lr){4-6}',
                      r'Query & Accuracy & Cal. AUC & Accuracy & Cal. AUC & Raw AUC\\\midrule']
                     + rows + [r'\bottomrule', r'\end{tabular}', r'\end{table}'])


def by_hand():
    D = pd.read_csv(P1 / 'csafe_by_query_hand.csv')
    rows = []
    for r in D.itertuples():
        rows.append(' & '.join([COND[r.condition], NAMES[r.rep], r.query_hand, f'{r.identity_top1:.2f}', f(r.known_gain),
                                f(r.fixed_change), f(r.nested_change), f(r.filter_change), f'{r.hand_recall:.1f}']) + r' \\')
    return '\n'.join([r'\begin{table}[ht]', r'\caption{CSAFE session-2 results split by query hand in the balanced 96-writer gallery (48 writers per hand; one writer is 2.08 points). Changes are percentage points from identity-only ranking with the primary HC-183 predictor; recall is the handedness accuracy for that hand. Descriptive only.}\label{tab:csafebyhand}',
                      r'\centering\footnotesize\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{lllrrrrrr}\toprule',
                      r'Prompt & ID features & Query hand & Identity, \% & Known & Fixed & Nested & Filter & Recall, \%\\\midrule']
                     + rows + [r'\bottomrule', r'\end{tabular}', r'\end{table}'])


def csafe_rows():
    """Corrected rows for Tables S17 (handedness) and S19 (complete identification)."""
    H = pd.read_csv(P1 / 'csafe_corrected/handedness_conditional_intervals.csv')
    S = pd.read_csv(P1 / 'csafe_corrected/id_summary.csv')
    S = S[(S.cohort == 'full_balanced')]
    q = {'S2_LND_r01': 'S2/LND', 'S2_WOZ_r01': 'S2/WOZ', 'S3_WOZ_r01': 'S3/WOZ'}
    hand = {}
    for r in H[H.hand_model == 'hc183_adj'].itertuples():
        hand[q[r.condition]] = (f'{q[r.condition]} & HC-183 adj. & {100 * r.accuracy:.2f} & [{100 * r.ci_lo:.2f}, {100 * r.ci_hi:.2f}] & '
                                f'{r.nll:.3f} & {r.brier:.3f} & {r.raw_auc:.3f} & {r.calibrated_auc:.3f} \\\\')
    full = {}
    reps = {'resnet18': 'ResNet-18', 'vit_b_16': 'ViT-B/16', 'hc183': 'HC-183', 'stroke': 'STROKE'}
    for cond in q:
        for rep in reps:
            s = S[(S.condition == cond) & (S.rep == rep) & (S.hand_model == 'hc183_adj')].set_index('method').top1 * 100
            full[(q[cond], reps[rep])] = (f'{q[cond]} & HC-183 adj. & {reps[rep]} & {s.identity_only:.2f} & {s.oracle_known_hand:.2f} & '
                                          f'{s.predicted_hard_filter:.2f} & {s.fixed_fusion_lambda1:.2f} & {s.nested_fusion:.2f} \\\\')
    return hand, full


def p2_tables():
    P2 = ROOT / 'results/p2_extensions_2026-11-09'
    A = pd.read_csv(P2 / 'attribute_summary.csv')
    I = pd.read_csv(P2 / 'identification_summary.csv')
    M = pd.read_csv(P2 / 'mechanism.csv')
    T = pd.read_csv(P2 / 'paired_tests.csv')
    RN = dict(NAMES, sift_vlad='SIFT-VLAD')
    CN = {'S2_LND_r01': 'S2 same', 'S2_WOZ_r01': 'S2 different', 'S3_WOZ_r01': 'S3 different'}
    rows = [' & '.join([r.attribute.capitalize(), CN[r.condition], f'{100*r.accuracy:.1f}', f'{100*r.balanced_accuracy:.1f}',
                        f'{100*r.recall_1:.1f} / {100*r.recall_0:.1f}', f'{r.auc:.3f}', f'{r.nll:.3f}', f'{r.brier:.3f}']) + r' \\' for r in A.itertuples()]
    attr = '\n'.join([r'\begin{table}[ht]', r'\caption{Writer-held-out attribute prediction on CSAFE with the primary HC-183 model. Handedness: 48 left (label 1) and 48 right. Gender: 63 female (label 1) and 33 male. Recall is given for label 1 / label 0; AUC uses calibrated scores; NLL and Brier are losses.}\label{tab:p2attr}',
                      r'\centering\footnotesize', r'\begin{tabular}{llrrrrrr}\toprule',
                      r'Attribute & Query & Accuracy, \% & Balanced, \% & Recall, \% & AUC & NLL & Brier\\\midrule'] + rows + [r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    piv = I.pivot_table(index=['attribute', 'condition', 'rep'], columns='method', values='top1')
    mm = M.set_index(['attribute', 'condition', 'rep'])
    rows, last = [], None
    for att in ('handedness', 'gender'):
        for cond in CN:
            for rep in ('resnet18', 'vit_b_16', 'sift_vlad', 'hc183', 'stroke'):
                r = piv.loc[(att, cond, rep)]
                rand = r.identity_only + mm.loc[(att, cond, rep), 'random_gain']
                if last is not None and att != last:
                    rows.append(r'\midrule')
                last = att
                rows.append(' & '.join([att.capitalize(), CN[cond], RN[rep], f'{r.identity_only:.2f}', f'{r.known:.2f}', f'{rand:.2f}',
                                        f'{r.hard_filter:.2f}', f'{r.fixed:.2f}', f'{r.nested:.2f}']) + r' \\')
    ident = '\n'.join([r'\begin{longtable}{lllrrrrrr}', r'\caption{CSAFE top-1 identification (\%) with SIFT-VLAD added as a handwriting-specific matcher and gender as a second attribute. Random: exact expectation of a known random partition with the same group sizes as the attribute. Filter, Fixed and Nested use the predicted attribute.}\label{tab:p2ident}\\',
                       r'\toprule Attribute & Query & ID features & ID & Known & Random & Filter & Fixed & Nested\\\midrule', r'\endfirsthead',
                       r'\toprule Attribute & Query & ID features & ID & Known & Random & Filter & Fixed & Nested\\\midrule', r'\endhead'] + rows + [r'\bottomrule', r'\end{longtable}'])
    P = T[T.primary].copy()
    order = {('handedness', 'sift_vlad'): 0, ('gender', 'resnet18'): 1, ('gender', 'vit_b_16'): 2, ('gender', 'sift_vlad'): 3}
    P['o'] = [order[(a, r)] for a, r in zip(P.attribute, P.rep)]
    P = P.sort_values(['o', 'condition', 'method'])
    rows = []
    for r in P.itertuples():
        eq = 'none changed' if r.n_changed == 0 else f'{r.smallest_equivalence_margin:.2f}'
        rows.append(' & '.join([r.attribute.capitalize(), RN[r.rep], CN[r.condition], r.method.capitalize(), f(r.delta_pts),
                                ci(r.ci95_lo, r.ci95_hi), eq, str(r.n_changed), f'{r.p_holm:.3f}']) + r' \\')
    tests = '\n'.join([r'\begin{table}[ht]', r'\caption{The 16 pre-specified primary extension contrasts (fixed or nested fusion minus identity-only ranking, percentage points), with conditional 95\% writer-bootstrap intervals, the smallest equivalence margin supported by the 90\% interval, the number of writer decisions changed, and Holm-adjusted sign-flip $p$-values across all 16.}\label{tab:p2tests}',
                       r'\centering\footnotesize\setlength{\tabcolsep}{3.5pt}', r'\begin{tabular}{lllrrrrrr}\toprule',
                       r'Attribute & ID features & Query & Fusion & Change & 95\% CI & $\Delta^*$ & Changed & Holm $p$\\\midrule'] + rows + [r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    return attr, ident, tests


def p3_tables():
    P3 = ROOT / 'results/p3_sweep_2026-11-09'
    RN = dict(NAMES, sift_vlad='SIFT-VLAD')
    S = pd.read_csv(P3 / 'sweep_summary.csv')
    SET = {'own n=1 k=1': 'own, paragraph, $k=1$', 'own n=1 k=6': 'own, paragraph, $k=6$', 'own n=4 k=6': 'own, quarter, $k=6$',
           'shared n=4 k=5': 'shared, quarter, $k=5$', 'S2_LND_r01': 'S2 same', 'S2_WOZ_r01': 'S2 different'}
    def pc(x):
        return '--' if pd.isna(x) else f'{100*x:.0f}'
    rows, last = [], None
    for r in S.itertuples():
        if last is not None and r.dataset != last:
            rows.append(r'\midrule')
        last = r.dataset
        rows.append(' & '.join([r.dataset, RN[r.rep], SET[r.setting], f'{r.oracle_gain:.1f}', f'{100*r.real_acc:.1f}', f(r.sim_gain_at_real),
                                f(r.observed_fixed_gain), pc(r.acc_for_1pt), pc(r.acc_for_half_oracle)]) + r' \\')
    sweep = '\n'.join([r'{\footnotesize\setlength{\tabcolsep}{2.6pt}', r'\begin{longtable}{lllrrrrrr}', r'\caption{Accuracy sweep with synthetic calibrated attribute evidence that is independent of the identity scores given the attribute (200 draws per accuracy, fixed fusion). Simulated: gain at the achieved accuracy of the real HC-183 predictor; Observed: fixed-fusion gain of the real predictor in the same setting. The last two columns give the simulated accuracy (\%) at which the mean gain first reaches $+1$ point and half the oracle gain; -- means never. Gains are percentage points.}\label{tab:sweep}\\',
                       r'\toprule Data & ID features & Setting & Oracle & Achieved, \% & Simulated & Observed & $+1$ pt & Half\\\midrule', r'\endfirsthead',
                       r'\toprule Data & ID features & Setting & Oracle & Achieved, \% & Simulated & Observed & $+1$ pt & Half\\\midrule', r'\endhead']
                      + rows + [r'\bottomrule', r'\end{longtable}', '}'])
    R = pd.read_csv(P3 / 'rank_endpoints.csv')
    rows = []
    B = R[R.dataset == 'BanglaHand']
    for r in B.itertuples():
        val = f(r.delta, 2) + ' ' + ci(r.ci_lo, r.ci_hi, 2) if r.metric == 'top5' else f(r.delta, 3) + ' ' + ci(r.ci_lo, r.ci_hi, 3)
        setting = 'all 120 settings' if r.setting.startswith('all') else RN[r.rep] + ', ' + ('paragraph, $k=1$' if 'n=1' in r.setting else 'quarter, $k=6$')
        rows.append(' & '.join(['BanglaHand', 'hand', setting, r.method.replace('_', ' '), 'top-5' if r.metric == 'top5' else 'MRR', val]) + r' \\')
    rows.append(r'\midrule')
    C = R[(R.dataset == 'CSAFE') & R.rep.isin(['resnet18', 'vit_b_16', 'sift_vlad'])]
    for (att, rep, st, m), g in C.groupby(['attribute', 'rep', 'setting', 'method'], sort=True):
        g = g.set_index('metric')
        t5, mr = g.loc['top5'], g.loc['mrr']
        rows.append(' & '.join(['CSAFE', att[:6] if att == 'gender' else 'hand', RN[rep] + ', ' + SET[st], m.replace('_', ' '),
                                'top-5 / MRR', f(t5.delta, 2) + ' ' + ci(t5.ci_lo, t5.ci_hi, 1) + ' / ' + f(mr.delta, 3)]) + r' \\')
    rank = '\n'.join([r'{\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{longtable}{llllll}', r'\caption{Rank-based endpoints: changes in top-5 accuracy (percentage points) and mean reciprocal rank (MRR) from identity-only ranking, with conditional 95\% hand-stratified writer-bootstrap intervals (CSAFE intervals shown for top-5). Unadjusted; descriptive.}\label{tab:rank}\\',
                      r'\toprule Data & Attribute & Setting & Rule & Metric & Change [95\% CI]\\\midrule', r'\endfirsthead',
                      r'\toprule Data & Attribute & Setting & Rule & Metric & Change [95\% CI]\\\midrule', r'\endhead']
                     + rows + [r'\bottomrule', r'\end{longtable}', '}'])
    K = pd.read_csv(P3 / 'cluster_counts.csv')
    rows, last = [], None
    for r in K.itertuples():
        key = (r.dataset, r.attribute)
        if last is not None and key != last:
            rows.append(r'\midrule')
        last = key
        cival = '--' if pd.isna(r.ci_lo) else f'[{100*r.ci_lo:.1f}, {100*r.ci_hi:.1f}]'
        rows.append(' & '.join([r.dataset, r.attribute, RN[r.rep], f'{int(r.decisions):,}', f'{int(r.units):,}', str(int(r.writers)),
                                f'{100*r.share:.1f}', cival, f'{100*r.reference:.1f}']) + r' \\')
    counts = '\n'.join([r'\begin{table}[ht]', r'\caption{Sample sizes behind the within-group clustering result: queries whose true writer is outranked by exactly one impostor. BanglaHand decisions are query-unit/reference-configuration pairs pooled over the 20 settings per representation (units: distinct query units); CSAFE decisions pool both session-2 conditions. Share: fraction of sole outranking impostors in the opposite group, with 95\% intervals (writer bootstrap for BanglaHand, decision bootstrap stratified by hand for CSAFE); Random: the expected share. CSAFE rows rest on few decisions and are descriptive.}\label{tab:counts}',
                        r'\centering\footnotesize\setlength{\tabcolsep}{3.5pt}', r'\begin{tabular}{lllrrrrrr}\toprule',
                        r'Data & Attribute & ID features & Decisions & Units & Writers & Share, \% & 95\% CI & Random, \%\\\midrule']
                       + rows + [r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    return sweep, rank, counts


def main():
    TAB.mkdir(exist_ok=True)
    for name, fn in (('designated', designated), ('precision', precision), ('decomposition', decomposition),
                     ('correction', correction), ('csafe_by_hand', by_hand)):
        (TAB / f'p1_{name}.tex').write_text(fn() + '\n')
    for name, block in zip(('p2_attr', 'p2_ident', 'p2_tests'), p2_tables()):
        (TAB / f'{name}.tex').write_text(block + '\n')
    for name, block in zip(('p3_sweep', 'p3_rank', 'p3_counts'), p3_tables()):
        (TAB / f'{name}.tex').write_text(block + '\n')
    hand, full = csafe_rows()
    (TAB / 'p1_csafe_hand_rows.tex').write_text('\n'.join(hand.values()) + '\n')
    (TAB / 'p1_csafe_full_rows.tex').write_text('\n'.join(full.values()) + '\n')
    print('written', sorted(p.name for p in TAB.glob('p1_*')))


if __name__ == '__main__':
    main()
