#!/usr/bin/env python3
"""Rerun the CSAFE full-gallery analysis with a numerically stable nuisance regression.

Diagnosis (see results/p1_revision_2026-11-09/README.md): on CSAFE scans the
grayscale q50/q75/q95 covariates are exactly 255 on every page and q25 is 255 on
98% of pages. The design matrix [1, covariates] is therefore rank deficient, and
np.linalg.lstsq with the default rcond keeps rounding-level singular values. In
some training splits this produced residualised features, and hence SVM
out-of-fold scores, of order 1e15. Two such pages with opposite labels flatten
the Platt slope to ~0 (|slope| < 1e-2 in scaled units, reported as "negative"),
so the calibrated logit becomes the training prior. With leave-one-writer-out
training that prior is opposite to the held-out writer's own hand, giving
below-chance accuracy (34%) despite raw AUC 0.73.

The stable version standardises the training covariates, drops columns with
zero training variance, and keeps exactly constant feature columns at exact zero
training residual (a second rounding path: HC-183 column 178 is 4 on all but two
pages, whose residual std 4.8e-16 was then divided out by StandardScaler). In
exact arithmetic the fitted
projection is the same as the original estimator; only rounding behaviour
changes. BanglaHand covariates are non-degenerate, so BanglaHand is unaffected.

This change was made after the CSAFE performance tables had been seen and must
be reported as a post hoc implementation correction to a secondary model.
Unadjusted HC-183 results are recomputed in the same run as a regression check
and must reproduce the archived outputs exactly.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_csafe_replication as rcs  # noqa: E402
import run_wid_handedness as wh  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'results/csafe_replication_2026-11-08'
OUT = ROOT / 'results/p1_revision_2026-11-09/csafe_adjusted_fix'


def residualise_stable(Xtr, Ctr, X, C):
    """Same projection as run_wid_handedness.residualise, computed stably.

    (1) Covariates are standardised on the training split and zero-variance
        covariates (collinear with the intercept) are dropped.
    (2) Feature columns that are exactly constant on the training split have
        exact training residual 0; they are centred directly instead of being
        regressed, so StandardScaler sees zero variance (scale 1) rather than
        dividing by rounding noise of order 1e-16.
    """
    mu, sd = Ctr.mean(0), Ctr.std(0)
    keep = sd > 1e-9 * (1 + np.abs(mu))
    Ztr = (Ctr[:, keep] - mu[keep]) / sd[keep]
    Z = (C[:, keep] - mu[keep]) / sd[keep]
    A = np.column_stack([np.ones(len(Ztr)), Ztr])
    B = np.linalg.lstsq(A, Xtr, rcond=None)[0]
    out = X - np.column_stack([np.ones(len(Z)), Z]) @ B
    const = np.all(Xtr == Xtr[:1], axis=0)
    out[:, const] = X[:, const] - Xtr[0, const]
    return out


_ORIGINAL_OUTER = rcs.outer_attribute


def outer_attribute_stable(*args, **kwargs):
    """Worker-side wrapper: loky workers re-import modules, so patch there."""
    import run_wid_handedness as worker_wh
    worker_wh.residualise = residualise_stable
    return _ORIGINAL_OUTER(*args, **kwargs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(OUT / 'attribute_cache', ignore_errors=True)  # never reuse fits from another implementation
    for name in ('units.csv', 'images.csv', 'features.npz', 'protocol_v1_1.json'):
        shutil.copy2(SRC / name, OUT / name)
    wh.residualise = residualise_stable
    rcs.outer_attribute = outer_attribute_stable
    rcs.OUT = OUT
    rcs.HAND_MODELS = ('hc183', 'hc183_adj')
    original_cohorts = rcs.cohorts
    rcs.cohorts = lambda U: original_cohorts(U)[:1]  # full 96-writer gallery only
    sys.argv = [sys.argv[0], '--jobs', '6']
    rcs.main()

    # Regression check: unadjusted HC-183 must match the archived run exactly.
    old = pd.read_csv(SRC / 'writer_outcomes.csv.gz')
    new = pd.read_csv(OUT / 'writer_outcomes.csv.gz')
    key = ['cohort', 'condition', 'rep', 'hand_model', 'method', 'writer_id']
    o = old[(old.cohort == 'full_balanced') & (old.hand_model == 'hc183')].sort_values(key).reset_index(drop=True)
    n = new[new.hand_model == 'hc183'].sort_values(key).reset_index(drop=True)
    assert len(o) == len(n)
    exact = bool((o[key + ['rank', 'correct', 'hand_correct']] == n[key + ['rank', 'correct', 'hand_correct']]).all().all())
    max_margin = float(np.max(np.abs(o.identity_margin - n.identity_margin)))
    max_logodds = float(np.max(np.abs(o.true_hand_logodds - n.true_hand_logodds)))
    print(f'hc183 regression check: exact ranks/decisions={exact}; max |margin diff|={max_margin:.2e}; '
          f'max |log-odds diff|={max_logodds:.2e}')
    assert exact and max_margin < 1e-9 and max_logodds < 1e-9

    hs = pd.read_csv(OUT / 'handedness_summary.csv')
    hs_old = pd.read_csv(SRC / 'handedness_summary.csv')
    hs_old = hs_old[(hs_old.cohort == 'full_balanced') & (hs_old.hand_model == 'hc183_adj')]
    cols = ['condition', 'hand_model', 'accuracy', 'nll', 'brier', 'raw_auc', 'calibrated_auc']
    print('original adjusted model:\n', hs_old[cols].round(4).to_string(index=False))
    print('stable adjusted model:\n', hs[hs.hand_model == 'hc183_adj'][cols].round(4).to_string(index=False))
    import json
    meta = sorted((OUT / 'attribute_cache').glob('*.json'))
    slopes = {}
    for m in meta:
        rec = json.loads(m.read_text())
        slopes[m.stem] = [r['final_platt_fit']['raw_slope'] for r in rec]
    for k, v in slopes.items():
        v = np.array(v)
        print(f'cache {k[:10]}: outer Platt raw slopes min={v.min():.4f} n<=1e-6: {(v <= 1e-6).sum()}')
    raw = {}
    for f in sorted((OUT / 'attribute_cache').glob('*.npz')):
        with np.load(f) as z:
            raw[f.stem] = float(np.nanmax(np.abs(z['raw'])))
    print('max |raw outer score| per cache:', {k[:10]: round(v, 3) for k, v in raw.items()})


if __name__ == '__main__':
    main()
