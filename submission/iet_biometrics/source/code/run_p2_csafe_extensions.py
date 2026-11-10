#!/usr/bin/env python3
"""CSAFE extensions: SIFT-VLAD identity matcher and gender as a second attribute.

Protocol: results/p2_extensions_2026-11-09/protocol.json (written before outcomes).
Reuses the CSAFE outer-writer / development / calibration hierarchy exactly
(run_csafe_replication.outer_attribute) and the same ranking rules, temperature
and nested-weight selection. Handedness predictions are refitted and must
reproduce the archived logits.
"""
import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revised_protocol as rp  # noqa: E402
import run_csafe_replication as rcs  # noqa: E402
from run_nested_fusion import choose_lambda  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CS = ROOT / 'results/csafe_replication_2026-11-08'
OUT = ROOT / 'results/p2_extensions_2026-11-09'
CONDITIONS = ('S2_LND_r01', 'S2_WOZ_r01', 'S3_WOZ_r01')
REPS = ('resnet18', 'vit_b_16', 'hc183', 'stroke', 'sift_vlad')
SEED = 20261109
Z80 = 1.959964 + 0.841621
PRIMARY = {('handedness', 'sift_vlad'), ('gender', 'resnet18'), ('gender', 'vit_b_16'), ('gender', 'sift_vlad')}


def load():
    U = pd.read_csv(CS / 'units.csv')
    with np.load(CS / 'features.npz') as z:
        assert np.array_equal(z['image_id'], U.image_id)
        F = {k: z[k].astype(float) for k in ('resnet18', 'vit_b_16', 'hc183', 'stroke', 'global_intensity')}
    with np.load(OUT / 'features_sift_vlad.npz', allow_pickle=True) as z:
        assert np.array_equal(z['image_id'].astype(str), U.image_id.to_numpy().astype(str))
        F['sift_vlad'] = z['sift_vlad'].astype(float)
    survey = pd.read_csv(ROOT / 'dataset/csafe/surveydata_v2.csv').set_index('wid')
    wid = U.writer_id.str.replace('CSAFE_', '').astype(int)
    U['gender'] = (survey.loc[wid, 'gender'].to_numpy() == 'female').astype(int)
    assert set(survey.loc[wid.unique(), 'gender']) == {'female', 'male'}
    return U, F


def fit_attribute(U, F, y, name, jobs=6):
    path = OUT / f'attribute_{name}.npz'
    if path.exists():
        with np.load(path) as z:
            return z['raw'], z['outer'], z['development']
    writers = sorted(U.writer_id.unique())
    cov = np.column_stack([U.ink_density.to_numpy(), F['global_intensity']])
    with parallel_config(backend='loky', n_jobs=jobs, inner_max_num_threads=1):
        fitted = Parallel()(delayed(rcs.outer_attribute)(w, writers, F['hc183'], y, cov, U.writer_id.to_numpy(),
                                                         U.attribute_train.to_numpy(), U['query'].to_numpy(),
                                                         U.condition.to_numpy(), False) for w in writers)
    raw = np.stack([r[0] for r in fitted], 1)
    outer = np.stack([r[1] for r in fitted], 1)
    dev = np.stack([r[2] for r in fitted])
    np.savez_compressed(path, raw=raw, outer=outer, development=dev)
    (OUT / f'attribute_{name}_manifest.json').write_text(json.dumps([r[3] for r in fitted], indent=1) + '\n')
    return raw, outer, dev


def boot90(delta, hand, rng, n=10000):
    draws = np.zeros(n)
    for h in (0, 1):
        v = delta[hand == h]
        draws += .5 * v[rng.integers(len(v), size=(n, len(v)))].mean(1)
    lo, hi = np.quantile(draws, [.05, .95])
    return lo, hi, draws.std(ddof=1)


def main():
    start = time.time()
    U, F = load()
    writers = sorted(U.writer_id.unique())
    n = len(writers)
    hand = U.groupby('writer_id').label.first().reindex(writers).to_numpy()
    gender = U.groupby('writer_id').gender.first().reindex(writers).to_numpy()
    attrs = {'handedness': (U.label.to_numpy(), hand), 'gender': (U.gender.to_numpy(), gender)}
    lookup = {(r.writer_id, r.condition): i for i, r in enumerate(U.itertuples())}
    ri = np.array([lookup[(w, 'S1_LND_r01')] for w in writers])
    preds = {a: fit_attribute(U, F, y, a) for a, (y, _) in attrs.items()}
    # Regression check: refitted handedness logits equal the archived primary model.
    W0 = pd.read_csv(CS / 'writer_outcomes.csv.gz')
    W0 = W0[(W0.cohort == 'full_balanced') & (W0.hand_model == 'hc183') & (W0.rep == 'resnet18') & (W0.method == 'identity_only')]
    for c, cond in enumerate(CONDITIONS):
        arch = W0[W0.condition == cond].set_index('writer_id').reindex(writers).true_hand_logodds.to_numpy()
        mine = (2 * hand - 1) * preds['handedness'][1][c]
        assert np.max(np.abs(arch - mine)) < 1e-9, cond
    rng = np.random.default_rng(SEED)
    attr_rows, summ, tests, wrows, mech = [], [], [], [], []
    for a, (_, lab) in attrs.items():
        raw, outer, dev = preds[a]
        for c, cond in enumerate(CONDITIONS):
            ell = outer[c]; p = expit(ell); ok = (ell > 0) == lab
            attr_rows.append(dict(attribute=a, condition=cond, accuracy=ok.mean(),
                                  balanced_accuracy=.5 * (ok[lab == 0].mean() + ok[lab == 1].mean()),
                                  recall_1=ok[lab == 1].mean(), recall_0=ok[lab == 0].mean(),
                                  auc=roc_auc_score(lab, ell), raw_auc=roc_auc_score(lab, raw[c]),
                                  nll=np.mean(np.logaddexp(0, ell) - lab * ell), brier=np.mean((p - lab) ** 2),
                                  n_label1=int(lab.sum()), n_label0=int((1 - lab).sum())))
    for rep in REPS:
        X = F[rep]
        for cond in CONDITIONS:
            c = CONDITIONS.index(cond)
            qi = np.array([lookup[(w, cond)] for w in writers])
            sc = StandardScaler().fit(X[ri])
            zr, zq = sc.transform(X[ri]), sc.transform(X[qi])
            zr /= np.maximum(np.linalg.norm(zr, axis=1, keepdims=True), 1e-12)
            zq /= np.maximum(np.linalg.norm(zq, axis=1, keepdims=True), 1e-12)
            S = zq @ zr.T; truth = np.arange(n); qw = np.array(writers); wts = np.full(n, 1 / n)
            T = rp.fit_temperatures(S, truth, qw, wts)
            base_rank = rp.ranks(S, truth); base = base_rank == 1
            for a, (_, lab) in attrs.items():
                raw, outer, dev = preds[a]
                ell = outer[c]; M = rp.match_logprob(ell, lab); ok = (ell > 0) == lab
                lam = np.array([choose_lambda(S, truth, qw, wts, dev[j, c], w, T[j], lab)[0] for j, w in enumerate(writers)])
                scores = dict(identity_only=S, known=S + 10 * (lab[:, None] == lab[None, :]),
                              hard_filter=S + 10 * ((ell > 0)[:, None] == lab[None, :]),
                              fixed=S / T[:, None] + M, nested=S / T[:, None] + lam[:, None] * M)
                rk = {m: rp.ranks(s, truth) for m, s in scores.items()}
                assert np.array_equal(rk['hard_filter'] == 1, ok & (rk['known'] == 1))
                opp = np.array([np.sum(lab != lab[i]) for i in range(n)])
                jj = base_rank - 1
                rand = np.array([comb(int(opp[i]), int(jj[i])) / comb(n - 1, int(jj[i])) if jj[i] <= opp[i] else 0. for i in range(n)])
                known = (rk['known'] == 1).astype(float)
                one = jj == 1
                mech.append(dict(attribute=a, rep=rep, condition=cond, known_gain=100 * (known.mean() - base.mean()),
                                 random_gain=100 * (rand.mean() - base.mean()), gap=100 * (known.mean() - rand.mean()),
                                 j1_n=int(one.sum()), j1_opposite_share=known[one].mean() if one.any() else np.nan,
                                 j1_reference=(opp[one] / (n - 1)).mean() if one.any() else np.nan))
                for m, r in rk.items():
                    corr = r == 1
                    summ.append(dict(attribute=a, rep=rep, condition=cond, method=m, top1=100 * corr.mean(),
                                     top5=100 * (r <= 5).mean(), mrr=float((1 / r).mean()),
                                     rescued=int((corr & ~base).sum()), damaged=int((~corr & base).sum()),
                                     median_lambda=float(np.median(lam)) if m == 'nested' else np.nan))
                    for j, w in enumerate(writers):
                        wrows.append(dict(attribute=a, rep=rep, condition=cond, method=m, writer_id=w, hand=int(hand[j]),
                                          attribute_label=int(lab[j]), rank=int(r[j]), correct=int(corr[j])))
                    if m in ('fixed', 'nested', 'known', 'hard_filter'):
                        d = corr.astype(float) - base
                        inf = rcs.inference(d, hand, rng)
                        lo90, hi90, se = boot90(d, hand, rng)
                        primary = (a, rep) in PRIMARY and cond != 'S3_WOZ_r01' and m in ('fixed', 'nested')
                        tests.append(dict(attribute=a, rep=rep, condition=cond, method=m, primary=primary,
                                          delta_pts=100 * inf['delta_top1'], ci95_lo=100 * inf['ci_lo'], ci95_hi=100 * inf['ci_hi'],
                                          ci90_lo=100 * lo90, ci90_hi=100 * hi90, smallest_equivalence_margin=100 * max(abs(lo90), abs(hi90)),
                                          mde80=100 * Z80 * se, p_signflip=inf['p_signflip'],
                                          n_changed=int(np.sum(d != 0))))
        print(f'{rep}: done {time.time()-start:.0f}s', flush=True)
    T_ = pd.DataFrame(tests); T_['p_holm'] = np.nan
    sel = T_[T_.primary].sort_values('p_signflip')
    assert len(sel) == 16
    T_.loc[sel.index, 'p_holm'] = np.maximum.accumulate([min(1., (len(sel) - i) * p) for i, p in enumerate(sel.p_signflip)])
    pd.DataFrame(attr_rows).to_csv(OUT / 'attribute_summary.csv', index=False)
    pd.DataFrame(summ).to_csv(OUT / 'identification_summary.csv', index=False)
    T_.to_csv(OUT / 'paired_tests.csv', index=False)
    pd.DataFrame(wrows).to_csv(OUT / 'writer_outcomes.csv.gz', index=False)
    pd.DataFrame(mech).to_csv(OUT / 'mechanism.csv', index=False)
    # Pooled j=1 opposite-group share for SIFT-VLAD handedness and for gender (neural + SIFT-VLAD), session 2.
    W = pd.DataFrame(wrows)
    pooled = []
    for a, reps in (('handedness', ['sift_vlad']), ('gender', ['resnet18', 'vit_b_16', 'sift_vlad'])):
        b = W[(W.attribute == a) & W.rep.isin(reps) & (W.condition != 'S3_WOZ_r01') & (W.method == 'identity_only')]
        k = W[(W.attribute == a) & W.rep.isin(reps) & (W.condition != 'S3_WOZ_r01') & (W.method == 'known')]
        m = b.merge(k[['rep', 'condition', 'writer_id', 'correct']], on=['rep', 'condition', 'writer_id'], suffixes=('', '_known'))
        m = m[m['rank'] == 2]
        lab = m.attribute_label.to_numpy()
        opp = np.where(lab == 1, (attrs[a][1] == 0).sum(), (attrs[a][1] == 1).sum())
        g = m.groupby('writer_id').agg(num=('correct_known', 'sum'), den=('correct_known', 'size'), hand=('hand', 'first'))
        draws = np.empty(10000)
        idx = {h: np.flatnonzero(g.hand.to_numpy() == h) for h in (0, 1)}
        for i in range(10000):
            pick = np.concatenate([rng.choice(idx[h], len(idx[h])) for h in (0, 1) if len(idx[h])])
            draws[i] = g.num.to_numpy()[pick].sum() / g.den.to_numpy()[pick].sum()
        pooled.append(dict(attribute=a, reps='+'.join(reps), n_decisions=int(len(m)), share=m.correct_known.mean(),
                           ci95_lo=np.quantile(draws, .025), ci95_hi=np.quantile(draws, .975), reference=float((opp / (n - 1)).mean())))
    pd.DataFrame(pooled).to_csv(OUT / 'mechanism_pooled_j1.csv', index=False)
    print(pd.DataFrame(attr_rows).round(3).to_string())
    I = pd.DataFrame(summ).pivot_table(index=['attribute', 'condition', 'rep'], columns='method', values='top1').round(2)
    print(I.to_string())
    print(T_[T_.primary].round(3).to_string())
    print(pd.DataFrame(mech).round(2).to_string())
    print(pd.DataFrame(pooled).round(3).to_string())
    print(f'total {time.time()-start:.0f}s')


if __name__ == '__main__':
    main()
