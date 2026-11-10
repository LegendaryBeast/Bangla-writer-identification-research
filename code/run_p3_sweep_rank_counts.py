#!/usr/bin/env python3
"""Accuracy sweep, rank-based endpoints and clustering sample sizes.

1. Accuracy sweep. Synthetic calibrated attribute evidence, independent of the
   identity scores given the attribute, is fused with the fixed rule for every CSAFE
   matcher (session 2, handedness): z | h ~ N((2h-1)d/2, 1), d = 2 Phi^{-1}(a),
   log-odds = d z. 200 draws per accuracy. The BanglaHand sweep (four settings, six
   matchers) is the archived breakeven_simulation. Because synthetic evidence carries
   no redundancy with the matcher, the curve is an optimistic reference; the gap to
   the real predictor at its achieved accuracy measures redundancy plus calibration.
2. Rank endpoints: top-5 and mean reciprocal rank (MRR) changes for the primary
   contrasts, with hand-stratified writer-bootstrap intervals.
3. Sample sizes behind the within-group clustering result, per matcher.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revised_protocol as rp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CS = ROOT / 'results/csafe_replication_2026-11-08'
P2 = ROOT / 'results/p2_extensions_2026-11-09'
OUT = ROOT / 'results/p3_sweep_2026-11-09'
REV = ROOT / 'results/wid_revision_2026-11-08'
ROB = ROOT / 'results/wid_fusion_robustness_2026-11-08'
ACC = np.round(np.arange(.5, 1.0001, .05), 2)
DRAWS = 200
SEED = 20261110


def boot(delta, hand, rng, n=10000):
    draws = np.zeros(n)
    for h in (0, 1):
        v = delta[hand == h]
        draws += .5 * v[rng.integers(len(v), size=(n, len(v)))].mean(1)
    return np.quantile(draws, [.025, .975])


def csafe_sweep(rng):
    U = pd.read_csv(CS / 'units.csv')
    with np.load(CS / 'features.npz') as z:
        F = {k: z[k].astype(float) for k in ('resnet18', 'vit_b_16', 'hc183', 'stroke')}
    with np.load(P2 / 'features_sift_vlad.npz', allow_pickle=True) as z:
        F['sift_vlad'] = z['sift_vlad'].astype(float)
    writers = sorted(U.writer_id.unique()); n = len(writers)
    hand = U.groupby('writer_id').label.first().reindex(writers).to_numpy()
    look = {(r.writer_id, r.condition): i for i, r in enumerate(U.itertuples())}
    ri = np.array([look[(w, 'S1_LND_r01')] for w in writers])
    rows = []
    for rep in F:
        for cond in ('S2_LND_r01', 'S2_WOZ_r01'):
            qi = np.array([look[(w, cond)] for w in writers])
            sc = StandardScaler().fit(F[rep][ri])
            zr, zq = sc.transform(F[rep][ri]), sc.transform(F[rep][qi])
            zr /= np.linalg.norm(zr, axis=1, keepdims=True); zq /= np.linalg.norm(zq, axis=1, keepdims=True)
            S = zq @ zr.T; t = np.arange(n); qw = np.array(writers)
            T = rp.fit_temperatures(S, t, qw, np.full(n, 1 / n))
            base = (rp.ranks(S, t) == 1).mean()
            oracle = (rp.ranks(S + 10 * (hand[:, None] == hand[None, :]), t) == 1).mean()
            for a in ACC:
                if a >= 1:
                    rows.append(dict(rep=rep, condition=cond, acc=a, base=base, oracle=oracle, fused=oracle, fused_lo=oracle,
                                     fused_hi=oracle, filt=oracle)); continue
                d = 2 * norm.ppf(a); fused, filt = [], []
                for _ in range(DRAWS):
                    zz = rng.normal((2 * hand - 1) * d / 2, 1.0); ell = d * zz
                    if a == .5:
                        ell = np.zeros(n); pred = rng.integers(0, 2, n)
                    else:
                        pred = (ell > 0).astype(int)
                    M = rp.match_logprob(ell, hand)
                    fused.append((rp.ranks(S / T[:, None] + M, t) == 1).mean())
                    filt.append((rp.ranks(S + 10 * (pred[:, None] == hand[None, :]), t) == 1).mean())
                fused = np.array(fused)
                rows.append(dict(rep=rep, condition=cond, acc=a, base=base, oracle=oracle, fused=fused.mean(),
                                 fused_lo=np.quantile(fused, .05), fused_hi=np.quantile(fused, .95), filt=np.mean(filt)))
    return pd.DataFrame(rows)


def bangla_sweep():
    s = pd.read_csv(REV / 'breakeven_simulation.csv.gz')
    g = s.groupby(['rep', 'query', 'n', 'k', 'acc'])[['base', 'oracle', 'bayes', 'filter']].mean().reset_index()
    q = s.groupby(['rep', 'query', 'n', 'k', 'acc']).bayes.quantile([.05, .95]).unstack().reset_index()
    g = g.merge(q.rename(columns={.05: 'fused_lo', .95: 'fused_hi'}), on=['rep', 'query', 'n', 'k', 'acc'])
    return g.rename(columns={'bayes': 'fused', 'filter': 'filt'})


def crossing(acc, gain, level):
    """Smallest accuracy at which the mean gain reaches level (linear interpolation)."""
    acc, gain = np.asarray(acc), np.asarray(gain)
    above = np.flatnonzero(gain >= level)
    if not len(above):
        return np.nan
    i = above[0]
    if i == 0:
        return acc[0]
    return acc[i - 1] + (level - gain[i - 1]) * (acc[i] - acc[i - 1]) / (gain[i] - gain[i - 1])


def sweep_summary(B, C):
    # Real predictor accuracy and observed fixed-fusion gain in the same setting.
    real = []
    HA = pd.read_csv(REV / 'handedness_accuracy.csv')
    nested = pd.read_csv(ROB / 'nested_writer_outcomes.csv')
    for (rep, q, n, k), g in B.groupby(['rep', 'query', 'n', 'k']):
        g = g.sort_values('acc'); gain = 100 * (g.fused - g.base); og = 100 * (g.oracle.iloc[0] - g.base.iloc[0])
        a_real = float(HA[(HA.hand_model == 'hc183') & (HA['n'] == n)][q].iloc[0])
        nw = nested[(nested.rep == rep) & (nested['query'] == q) & (nested.n == n) & (nested.k == k)]
        piv = nw.pivot_table(index='writer_id', columns='method', values='top1')
        obs = 100 * float((piv.fixed - piv['none']).mean())
        real.append(dict(dataset='BanglaHand', rep=rep, setting=f'{q} n={n} k={k}', oracle_gain=og, real_acc=a_real,
                         sim_gain_at_real=float(np.interp(a_real, g.acc, gain)), observed_fixed_gain=obs,
                         acc_for_1pt=crossing(g.acc, gain, 1.0), acc_for_half_oracle=crossing(g.acc, gain, og / 2)))
    H = pd.read_csv(P2 / 'attribute_summary.csv'); H = H[H.attribute == 'handedness'].set_index('condition')
    I = pd.read_csv(P2 / 'identification_summary.csv'); I = I[(I.attribute == 'handedness')]
    for (rep, cond), g in C.groupby(['rep', 'condition']):
        g = g.sort_values('acc'); gain = 100 * (g.fused - g.base); og = 100 * (g.oracle.iloc[0] - g.base.iloc[0])
        a_real = float(H.loc[cond, 'accuracy'])
        x = I[(I.rep == rep) & (I.condition == cond)].set_index('method').top1
        real.append(dict(dataset='CSAFE', rep=rep, setting=cond, oracle_gain=og, real_acc=a_real,
                         sim_gain_at_real=float(np.interp(a_real, g.acc, gain)), observed_fixed_gain=float(x.fixed - x.identity_only),
                         acc_for_1pt=crossing(g.acc, gain, 1.0), acc_for_half_oracle=crossing(g.acc, gain, og / 2) if og > 0 else np.nan))
    return pd.DataFrame(real)


def rank_endpoints(rng):
    rows = []
    # CSAFE, all matchers and both attributes (session 2), from the extension outcomes.
    W = pd.read_csv(P2 / 'writer_outcomes.csv.gz')
    W = W[W.condition != 'S3_WOZ_r01']
    W['top5'] = (W['rank'] <= 5).astype(float); W['rr'] = 1 / W['rank']
    for (att, rep, cond), g in W.groupby(['attribute', 'rep', 'condition']):
        p5 = g.pivot_table(index=['writer_id', 'hand'], columns='method', values='top5').reset_index()
        pr = g.pivot_table(index=['writer_id', 'hand'], columns='method', values='rr').reset_index()
        hand = p5.hand.to_numpy()
        for m in ('known', 'hard_filter', 'fixed', 'nested'):
            for metric, P in (('top5', p5), ('mrr', pr)):
                d = (P[m] - P.identity_only).to_numpy()
                lo, hi = boot(d, hand, rng)
                rows.append(dict(dataset='CSAFE', attribute=att, rep=rep, setting=cond, method=m, metric=metric,
                                 base=100 * P.identity_only.mean() if metric == 'top5' else P.identity_only.mean(),
                                 delta=100 * d.mean() if metric == 'top5' else d.mean(),
                                 ci_lo=100 * lo if metric == 'top5' else lo, ci_hi=100 * hi if metric == 'top5' else hi))
    # BanglaHand MRR (none, fixed, nested) over all 120 settings and the four designated settings.
    N = pd.read_csv(ROB / 'nested_writer_outcomes.csv')
    N['hand'] = N.writer_id.str.startswith('L').astype(int)
    scopes = [('all 120 settings', N)] + [(f'{r} own n={n} k={k}', N[(N.rep == r) & (N['query'] == 'own') & (N.n == n) & (N.k == k)])
                                          for r in ('resnet18', 'vit_b_16') for n, k in ((1, 1), (4, 6))]
    for label, g in scopes:
        P = g.groupby(['writer_id', 'method']).mrr.mean().unstack()
        hand = np.asarray(P.index.str.startswith('L'), dtype=int)
        for m in ('fixed', 'nested'):
            d = (P[m] - P['none']).to_numpy(); lo, hi = boot(d, hand, rng)
            rows.append(dict(dataset='BanglaHand', attribute='handedness', rep='all' if label.startswith('all') else label.split()[0],
                             setting=label, method=m, metric='mrr', base=P['none'].mean(), delta=d.mean(), ci_lo=lo, ci_hi=hi))
    # BanglaHand top-5 from the archived ranks (none, known, fixed = query bayes, filter), all 120 settings.
    R = pd.read_csv(REV / 'id_ranks.csv.gz', usecols=['rep', 'query', 'n', 'k', 'scenario', 'hand_model', 'method', 'writer_id', 'rank', 'weight'])
    R = R[(R.scenario == 'none') | (R.scenario == 'oracle') | ((R.scenario == 'query') & (R.hand_model == 'hc183'))]
    R['key'] = R.scenario + '_' + R.method
    R['top5'] = (R['rank'] <= 5) * R.weight
    per = R.groupby(['rep', 'query', 'n', 'k', 'key', 'writer_id']).agg(t5=('top5', 'sum'), w=('weight', 'sum')).reset_index()
    per['t5'] = per.t5 / per.w
    P = per.groupby(['writer_id', 'key']).t5.mean().unstack()
    hand = np.asarray(P.index.str.startswith('L'), dtype=int)
    for key, m in (('oracle_filter', 'known'), ('query_bayes', 'fixed'), ('query_filter', 'hard_filter')):
        d = (P[key] - P['none_none']).to_numpy(); lo, hi = boot(d, hand, rng)
        rows.append(dict(dataset='BanglaHand', attribute='handedness', rep='all', setting='all 120 settings', method=m, metric='top5',
                         base=100 * P['none_none'].mean(), delta=100 * d.mean(), ci_lo=100 * lo, ci_hi=100 * hi))
    return pd.DataFrame(rows)


def cluster_counts(rng):
    rows = []
    R = pd.read_csv(REV / 'id_ranks.csv.gz', usecols=['rep', 'query', 'n', 'k', 'scenario', 'method', 'unit_id', 'writer_id', 'config', 'rank', 'weight'])
    base = R[R.scenario == 'none']
    known = R[R.scenario == 'oracle'][['rep', 'query', 'n', 'k', 'unit_id', 'config', 'rank']].rename(columns={'rank': 'rk'})
    m = base.merge(known, on=['rep', 'query', 'n', 'k', 'unit_id', 'config'])
    m = m[m['rank'] == 2]
    J = pd.read_csv(ROOT / 'results/p1_revision_2026-11-09/mechanism_bangla_j1_by_rep.csv').set_index('rep')
    for rep, g in m.groupby('rep'):
        rows.append(dict(dataset='BanglaHand', attribute='handedness', rep=rep, decisions=len(g), units=g.unit_id.nunique(),
                         writers=g.writer_id.nunique(), settings=g.groupby(['query', 'n', 'k']).ngroups,
                         share=J.loc[rep, 'j1_opposite_hand_share'], ci_lo=J.loc[rep, 'ci95_lo'], ci_hi=J.loc[rep, 'ci95_hi'], reference=21 / 41))
    W = pd.read_csv(P2 / 'writer_outcomes.csv.gz'); W = W[W.condition != 'S3_WOZ_r01']
    b = W[W.method == 'identity_only']; k = W[W.method == 'known']
    x = b.merge(k[['attribute', 'rep', 'condition', 'writer_id', 'correct']], on=['attribute', 'rep', 'condition', 'writer_id'], suffixes=('', '_k'))
    x = x[x['rank'] == 2]
    sizes = {'handedness': (48, 48), 'gender': (33, 63)}
    for (att, rep), g in x.groupby(['attribute', 'rep']):
        n0, n1 = sizes[att]
        ref = np.where(g.attribute_label == 1, n0, n1) / 95
        hand = g.hand.to_numpy(); v = g.correct_k.to_numpy().astype(float)
        if len(np.unique(hand)) == 2:
            lo, hi = boot(v, hand, rng)
        else:
            lo, hi = np.nan, np.nan
        rows.append(dict(dataset='CSAFE', attribute=att, rep=rep, decisions=len(g), units=len(g), writers=g.writer_id.nunique(),
                         settings=2, share=v.mean(), ci_lo=lo, ci_hi=hi, reference=ref.mean()))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    C = csafe_sweep(rng); C.to_csv(OUT / 'csafe_accuracy_sweep.csv', index=False)
    B = bangla_sweep(); B.to_csv(OUT / 'banglahand_accuracy_sweep.csv', index=False)
    S = sweep_summary(B, C); S.to_csv(OUT / 'sweep_summary.csv', index=False)
    R = rank_endpoints(rng); R.to_csv(OUT / 'rank_endpoints.csv', index=False)
    K = cluster_counts(rng); K.to_csv(OUT / 'cluster_counts.csv', index=False)
    pd.set_option('display.width', 220)
    print(S.round(3).to_string())
    print(R[(R.method.isin(['fixed', 'nested']))].round(3).to_string())
    print(R[(R.method.isin(['known', 'hard_filter']))].round(3).to_string())
    print(K.round(3).to_string())
    (OUT / 'provenance.json').write_text(json.dumps(dict(seed=SEED, draws=DRAWS, accuracies=ACC.tolist(),
        note='Synthetic evidence is independent of identity scores given the attribute; an optimistic reference. Post hoc.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
