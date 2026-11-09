#!/usr/bin/env python3
"""P1 revision analyses from saved outputs (no refitting).

1. Precision of the primary fusion contrasts: hand-stratified writer-bootstrap
   SE, 90% intervals (TOST at alpha=.05), the smallest equivalence margin the
   data support, the minimum detectable effect at 80% power, and the share of
   the oracle (known-hand) gain that the data can exclude.
2. Same-hand confusability: the known-hand minus random-partition oracle gap
   decomposed by the number j of impostors outranking the true writer. For
   j impostors a known hand partition rescues the query only if all j are of
   the opposite hand; a uniform balanced partition does so with probability
   C(H,j)/C(N-1,j), where H is the opposite-hand gallery size.

Inputs are the saved BanglaHand and CSAFE writer/rank outputs. Margins are
reported as a continuum because no margin was declared before the data were
seen; any margin chosen for the manuscript must be disclosed as post hoc.
"""
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/p1_revision_2026-11-09'
B = 10000
SEED = 20261109
Z80 = 1.959964 + 0.841621  # two-sided alpha=.05, power .80
MARGINS = (1.0, 2.0, 3.0)


def stratified_boot(delta, labels, rng, n=B):
    """Equal-hand-weight writer bootstrap of the mean paired difference."""
    delta = np.asarray(delta, float)
    draws = np.zeros(n)
    for hand in (0, 1):
        v = delta[labels == hand]
        draws += .5 * v[rng.integers(len(v), size=(n, len(v)))].mean(1)
    return draws


def precision_row(tag, delta, labels, oracle_gain, rng):
    delta = 100 * np.asarray(delta, float)
    est = .5 * (delta[labels == 0].mean() + delta[labels == 1].mean())
    draws = 100 * stratified_boot(delta / 100, labels, rng)
    se = float(draws.std(ddof=1))
    lo90, hi90 = np.quantile(draws, [.05, .95])
    lo95, hi95 = np.quantile(draws, [.025, .975])
    eq_margin = float(max(abs(lo90), abs(hi90)))
    row = dict(**tag, n_writers=len(delta), n_writers_changed=int(np.sum(delta != 0)), estimate_pts=est, boot_se_pts=se,
               degenerate=bool(np.all(delta == 0)),
               ci90_lo=lo90, ci90_hi=hi90, ci95_lo=lo95, ci95_hi=hi95,
               smallest_equivalence_margin_pts=eq_margin,
               mde80_pts=Z80 * se, upper_bound_gain_pts=float(hi90),
               oracle_gain_pts=oracle_gain,
               excludable_gain_share_of_oracle=(float(hi90) / oracle_gain if oracle_gain > 0 else np.nan))
    for m in MARGINS:
        row[f'equivalent_at_{m:g}pt'] = bool(-m < lo90 and hi90 < m)
    return row


def bangla_precision(rng):
    d = pd.read_csv(ROOT / 'results/wid_fusion_robustness_2026-11-08/nested_writer_outcomes.csv')
    k = pd.read_csv(ROOT / 'results/wid_revision_2026-11-08/checks_identification_writer.csv')
    k = k[k.method == 'known'].rename(columns={'top1': 'known'})
    p = d.pivot_table(index=['rep', 'query', 'n', 'k', 'writer_id'], columns='method', values='top1').reset_index()
    p = p.merge(k[['rep', 'query', 'n', 'k', 'writer_id', 'known']], on=['rep', 'query', 'n', 'k', 'writer_id'])
    assert len(p) == 120 * 42 and p[['none', 'fixed', 'nested', 'known']].notna().all().all()
    p['label'] = p.writer_id.str.startswith('L').astype(int)
    rows = []
    designated = [(r, n, kk) for r in ('resnet18', 'vit_b_16') for n, kk in ((1, 1), (4, 6))]
    for rep, n, kk in designated:
        s = p[(p.rep == rep) & (p['query'] == 'own') & (p.n == n) & (p.k == kk)].sort_values('writer_id')
        lab = s.label.to_numpy()
        og = 100 * float((s.known - s.none).mean())
        for m in ('fixed', 'nested'):
            tag = dict(dataset='BanglaHand', family='designated', setting=f'{rep} own n={n} k={kk}', contrast=f'{m}_minus_none')
            rows.append(precision_row(tag, (s[m] - s.none).to_numpy(), lab, og, rng))
    g = p.groupby('writer_id')[['none', 'fixed', 'nested', 'known', 'label']].mean()
    lab = g.label.to_numpy().astype(int)
    og = 100 * float((g.known - g.none).mean())
    for m in ('fixed', 'nested'):
        tag = dict(dataset='BanglaHand', family='aggregate', setting='all 120 settings', contrast=f'{m}_minus_none')
        rows.append(precision_row(tag, (g[m] - g.none).to_numpy(), lab, og, rng))
    return rows


def csafe_precision(rng):
    w = pd.read_csv(ROOT / 'results/csafe_replication_2026-11-08/writer_outcomes.csv.gz')
    w = w[(w.cohort == 'full_balanced') & (w.hand_model == 'hc183')]
    rows = []
    for cond in ('S2_LND_r01', 'S2_WOZ_r01'):
        for rep in ('resnet18', 'vit_b_16'):
            s = w[(w.condition == cond) & (w.rep == rep)]
            piv = s.pivot_table(index=['writer_id', 'label'], columns='method', values='correct').reset_index()
            lab = piv.label.to_numpy()
            og = 100 * float((piv.oracle_known_hand - piv.identity_only).mean())
            for m, name in (('fixed_fusion_lambda1', 'fixed'), ('nested_fusion', 'nested')):
                tag = dict(dataset='CSAFE', family='primary', setting=f'{rep} {cond}', contrast=f'{name}_minus_none')
                rows.append(precision_row(tag, (piv[m] - piv.identity_only).to_numpy(), lab, og, rng))
    return rows


def decompose(ranks_none, known_success, weights, opposite, others):
    """Gap known - random partition, decomposed by j = baseline rank - 1."""
    j = np.asarray(ranks_none) - 1
    rnd = np.array([comb(opposite, int(x)) / comb(others, int(x)) if x <= opposite else 0. for x in j])
    w = np.asarray(weights, float) / np.sum(weights)
    out = {}
    for lab, mask in (('j=1', j == 1), ('j=2', j == 2), ('j=3', j == 3), ('j>=4', j >= 4)):
        out[lab] = dict(weight=float(w[mask].sum()),
                        known_rescue=float(np.sum(w[mask] * known_success[mask]) / max(w[mask].sum(), 1e-15)),
                        random_rescue=float(np.sum(w[mask] * rnd[mask]) / max(w[mask].sum(), 1e-15)),
                        gap_contribution=float(np.sum(w[mask] * (known_success[mask] - rnd[mask]))))
    out['gap_total'] = float(np.sum(w * (known_success - rnd)))
    return out


def bangla_mechanism(rng):
    d = pd.read_csv(ROOT / 'results/wid_revision_2026-11-08/id_ranks.csv.gz',
                    usecols=['rep', 'query', 'n', 'k', 'scenario', 'method', 'unit_id', 'writer_id', 'config', 'rank', 'weight'])
    base = d[d.scenario == 'none'].drop(columns=['scenario', 'method'])
    known = d[d.scenario == 'oracle'][['rep', 'query', 'n', 'k', 'unit_id', 'config', 'rank']].rename(columns={'rank': 'rank_known'})
    m = base.merge(known, on=['rep', 'query', 'n', 'k', 'unit_id', 'config'], validate='one_to_one')
    assert len(m) == len(base)
    m['j'] = m['rank'] - 1
    m['known_success'] = (m.rank_known == 1).astype(float)
    m['label'] = m.writer_id.str.startswith('L').astype(int)
    # Sanity: a known partition can never rescue more than all-opposite outrankers allow
    assert (m.loc[m.j == 0, 'known_success'] == 1).all()
    rows = []
    for (rep, q, n, k), s in m.groupby(['rep', 'query', 'n', 'k']):
        dec = decompose(s['rank'].to_numpy(), s.known_success.to_numpy(), s.weight.to_numpy(), 21, 41)
        r = dict(dataset='BanglaHand', rep=rep, query=q, n=n, k=k, gap_total_pts=100 * dec['gap_total'])
        for lab in ('j=1', 'j=2', 'j=3', 'j>=4'):
            r[f'{lab}_weight'] = dec[lab]['weight']
            r[f'{lab}_known_rescue'] = dec[lab]['known_rescue']
            r[f'{lab}_random_rescue'] = dec[lab]['random_rescue']
            r[f'{lab}_gap_pts'] = 100 * dec[lab]['gap_contribution']
        rows.append(r)
    settings = pd.DataFrame(rows)
    # j=1 opposite-hand share per representation with writer bootstrap
    one = m[m.j == 1].copy()
    one['ww'] = one.weight
    rep_rows = []
    for rep, s in one.groupby('rep'):
        wsum = s.groupby('writer_id').apply(lambda x: pd.Series(dict(num=(x.ww * x.known_success).sum(), den=x.ww.sum(),
                                                                       label=x.label.iloc[0])), include_groups=False)
        lab = wsum.label.to_numpy().astype(int)
        num, den = wsum.num.to_numpy(), wsum.den.to_numpy()
        est = num.sum() / den.sum()
        draws = np.empty(B)
        idx = {h: np.flatnonzero(lab == h) for h in (0, 1)}
        for b in range(B):
            pick = np.concatenate([rng.choice(idx[h], len(idx[h])) for h in (0, 1)])
            draws[b] = num[pick].sum() / den[pick].sum()
        rep_total = m.loc[m.rep == rep, 'weight'].sum()
        rep_rows.append(dict(dataset='BanglaHand', rep=rep, j1_queries_weight_share=float(den.sum() / rep_total),
                             j1_opposite_hand_share=est, ci95_lo=np.quantile(draws, .025), ci95_hi=np.quantile(draws, .975),
                             uniform_reference=21 / 41))
    return settings, pd.DataFrame(rep_rows)


def csafe_mechanism(rng):
    w = pd.read_csv(ROOT / 'results/csafe_replication_2026-11-08/writer_outcomes.csv.gz')
    w = w[(w.cohort == 'full_balanced') & (w.hand_model == 'hc183')]
    rows, reps = [], []
    for (cond, rep), s in w.groupby(['condition', 'rep']):
        piv = s.pivot_table(index=['writer_id', 'label'], columns='method', values=['rank', 'correct']).reset_index()
        rk = piv[('rank', 'identity_only')].to_numpy().astype(int)
        ks = piv[('correct', 'oracle_known_hand')].to_numpy().astype(float)
        lab = piv['label'].to_numpy().astype(int)
        dec = decompose(rk, ks, np.ones(len(rk)), 48, 95)
        r = dict(dataset='CSAFE', condition=cond, rep=rep, gap_total_pts=100 * dec['gap_total'])
        for key in ('j=1', 'j=2', 'j=3', 'j>=4'):
            r[f'{key}_weight'] = dec[key]['weight']
            r[f'{key}_known_rescue'] = dec[key]['known_rescue']
            r[f'{key}_gap_pts'] = 100 * dec[key]['gap_contribution']
        rows.append(r)
        mask = rk == 2
        reps.append(dict(dataset='CSAFE', condition=cond, rep=rep, n_j1_writers=int(mask.sum()),
                         j1_opposite_hand_share=float(ks[mask].mean()) if mask.any() else np.nan,
                         uniform_reference=48 / 95))
    return pd.DataFrame(rows), pd.DataFrame(reps)


def csafe_opposite_share_pooled(rng):
    """Pooled CSAFE j=1 opposite-hand share across conditions/reps, writer bootstrap."""
    w = pd.read_csv(ROOT / 'results/csafe_replication_2026-11-08/writer_outcomes.csv.gz')
    w = w[(w.cohort == 'full_balanced') & (w.hand_model == 'hc183') & (w.condition != 'S3_WOZ_r01')]
    base = w[w.method == 'identity_only'][['condition', 'rep', 'writer_id', 'label', 'rank']]
    known = w[w.method == 'oracle_known_hand'][['condition', 'rep', 'writer_id', 'correct']]
    m = base.merge(known, on=['condition', 'rep', 'writer_id'])
    out = []
    for name, sel in (('neural (ResNet-18, ViT-B/16)', m.rep.isin(['resnet18', 'vit_b_16'])), ('all four', m.rep.notna())):
        s = m[sel & (m['rank'] == 2)]
        g = s.groupby('writer_id').agg(num=('correct', 'sum'), den=('correct', 'size'), label=('label', 'first'))
        lab = g.label.to_numpy()
        num, den = g.num.to_numpy(float), g.den.to_numpy(float)
        draws = np.empty(B)
        idx = {h: np.flatnonzero(lab == h) for h in (0, 1)}
        for b in range(B):
            pick = np.concatenate([rng.choice(idx[h], len(idx[h])) for h in (0, 1)])
            draws[b] = num[pick].sum() / den[pick].sum()
        out.append(dict(dataset='CSAFE', reps=name, n_decisions=int(den.sum()), n_writers=len(g),
                        j1_opposite_hand_share=num.sum() / den.sum(),
                        ci95_lo=np.quantile(draws, .025), ci95_hi=np.quantile(draws, .975), uniform_reference=48 / 95))
    return pd.DataFrame(out)


def csafe_by_query_hand():
    """Balanced-gallery CSAFE effects split by query hand (no refitting); 1 writer = 2.08 pts."""
    w = pd.read_csv(ROOT / 'results/csafe_replication_2026-11-08/writer_outcomes.csv.gz')
    w = w[(w.cohort == 'full_balanced') & (w.hand_model == 'hc183') & w.rep.isin(['resnet18', 'vit_b_16'])
          & (w.condition != 'S3_WOZ_r01')]
    p = w.pivot_table(index=['condition', 'rep', 'writer_id', 'label'], columns='method', values='correct').reset_index()
    rows = []
    for (c, r, lab), s in p.groupby(['condition', 'rep', 'label']):
        rows.append(dict(condition=c, rep=r, query_hand='left' if lab else 'right', n_writers=len(s),
                         identity_top1=100 * s.identity_only.mean(),
                         known_gain=100 * (s.oracle_known_hand - s.identity_only).mean(),
                         fixed_change=100 * (s.fixed_fusion_lambda1 - s.identity_only).mean(),
                         nested_change=100 * (s.nested_fusion - s.identity_only).mean(),
                         filter_change=100 * (s.predicted_hard_filter - s.identity_only).mean(),
                         hand_recall=100 * s.merge(w[w.method == 'identity_only'][['condition', 'rep', 'writer_id', 'hand_correct']],
                                                   on=['condition', 'rep', 'writer_id']).hand_correct.mean()))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    csafe_by_query_hand().to_csv(OUT / 'csafe_by_query_hand.csv', index=False)
    rng = np.random.default_rng(SEED)
    prec = pd.DataFrame(bangla_precision(rng) + csafe_precision(rng))
    prec.to_csv(OUT / 'precision_equivalence.csv', index=False)
    bs, br = bangla_mechanism(rng)
    cs, cr = csafe_mechanism(rng)
    cp = csafe_opposite_share_pooled(rng)
    bs.to_csv(OUT / 'mechanism_bangla_settings.csv', index=False)
    br.to_csv(OUT / 'mechanism_bangla_j1_by_rep.csv', index=False)
    cs.to_csv(OUT / 'mechanism_csafe_settings.csv', index=False)
    cr.to_csv(OUT / 'mechanism_csafe_j1_by_rep.csv', index=False)
    cp.to_csv(OUT / 'mechanism_csafe_j1_pooled.csv', index=False)
    # Cross-checks: decomposition totals must equal the saved oracle-minus-random differences
    ref = pd.read_csv(ROOT / 'results/wid_revision_2026-11-08/checks_oracle_partitions.csv')
    chk = bs.merge(ref, on=['rep', 'query', 'n', 'k'])
    assert len(chk) == 120
    bangla_err = float(np.max(np.abs(chk.gap_total_pts - 100 * (chk.known_hand - chk.random_partition_exact_mean))))
    cref = pd.read_csv(ROOT / 'results/csafe_replication_2026-11-08/diagnostics.csv')
    cref = cref[(cref.cohort == 'full_balanced') & (cref.hand_model == 'hc183')]
    cchk = cs.merge(cref, on=['condition', 'rep'])
    assert len(cchk) == 12
    csafe_err = float(np.max(np.abs(cchk.gap_total_pts - 100 * (cchk.oracle_gain - cchk.random_partition_expected_gain))))
    assert bangla_err < 1e-6 and csafe_err < 1e-6, (bangla_err, csafe_err)
    gap = bs.gap_total_pts.mean()
    summary = dict(
        bootstrap=dict(B=B, seed=SEED, scheme='equal-hand-weight writer bootstrap; conditional on fitted models and gallery'),
        bangla_mean_gap_known_minus_random_pts=gap,
        bangla_mean_gap_by_j_pts={k: float(bs[f'{k}_gap_pts'].mean()) for k in ('j=1', 'j=2', 'j=3', 'j>=4')},
        bangla_j1_known_rescue_mean=float(np.average(bs['j=1_known_rescue'], weights=bs['j=1_weight'])),
        csafe_mean_gap_known_minus_random_pts=float(cs[cs.condition != 'S3_WOZ_r01'].gap_total_pts.mean()),
        validation=dict(bangla_max_abs_error_vs_saved_partition_pts=bangla_err,
                        csafe_max_abs_error_vs_saved_diagnostics_pts=csafe_err),
        note='Margins are reported as a continuum; no equivalence margin was specified before seeing the data.')
    (OUT / 'p1_precision_mechanism_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    pd.set_option('display.width', 220)
    print(prec[['dataset', 'setting', 'contrast', 'estimate_pts', 'ci90_lo', 'ci90_hi', 'smallest_equivalence_margin_pts',
                'mde80_pts', 'oracle_gain_pts', 'excludable_gain_share_of_oracle']].round(2).to_string())
    print(br.round(3).to_string())
    print(cp.round(3).to_string())
    print(cr.round(3).to_string())
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
