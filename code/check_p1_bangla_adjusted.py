#!/usr/bin/env python3
"""Check whether the CSAFE residualisation rounding issue affects BanglaHand.

Recomputes the BanglaHand adjusted-HC-183 writer-held-out predictions for full
paragraphs (n=1) with (a) the original residualise and (b) the stable version
from run_p1_csafe_adjusted_fix.py, verifies (a) against the archived
handedness_lowo.csv.gz, and compares self-chosen paragraph accuracy.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_wid_handedness as wh  # noqa: E402
from run_p1_csafe_adjusted_fix import residualise_stable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / 'results/wid_handedness_2026-10-29/features'
OUT = ROOT / 'results/p1_revision_2026-11-09'


def fit(w, stable, X, y, C, g, train_base, units_w):
    if stable:
        wh.residualise = residualise_stable
    return wh.lowo_writer(X, y, C, g, train_base & (g != w), units_w, True, wh.SEED)


def main():
    U = pd.read_csv(FEAT / 'units.csv')
    with np.load(FEAT / 'unit_features.npz', allow_pickle=True) as z:  # project's own archived file
        assert np.array_equal(z['unit_id'], U.unit_id)
        X = z['hc183'].astype(float)
        G = z['global_intensity'].astype(float)
    C = np.column_stack([U.ink_density.to_numpy(), G])
    y, g = U.label.to_numpy(), U.writer_id.to_numpy()
    un = ((U.n == 1) & U.valid).to_numpy()
    base = un & (U.para <= 6).to_numpy()
    writers = sorted(U.writer_id.unique())
    rows = {}
    for stable in (False, True):
        res = Parallel(n_jobs=6)(delayed(fit)(w, stable, X, y, C, g, base, np.flatnonzero(un & (g == w))) for w in writers)
        idx = np.concatenate([r[0] for r in res])
        rows[stable] = pd.DataFrame(dict(unit_id=U.unit_id.to_numpy()[idx], raw=np.concatenate([r[1] for r in res]),
                                         logit=np.concatenate([r[2] for r in res])))
    arch = pd.read_csv(ROOT / 'results/wid_handedness_2026-10-29/handedness_lowo.csv.gz')
    arch = arch[arch.hand_model == 'hc183_adj'].set_index('unit_id')
    orig = rows[False].set_index('unit_id')
    reproduce = float(np.max(np.abs(orig.logit - arch.loc[orig.index, 'logit'])))
    meta = U.set_index('unit_id')
    report = dict(max_abs_logit_diff_original_vs_archive=reproduce)
    for stable, d in rows.items():
        d = d.set_index('unit_id')
        own = meta.loc[d.index, 'para'] >= 7
        lab = meta.loc[d.index, 'label']
        acc = float(((d.logit > 0) == (lab == 1))[own].mean())
        report[('stable' if stable else 'original') + '_own_paragraph_accuracy'] = acc
        report[('stable' if stable else 'original') + '_max_abs_raw'] = float(d.raw.abs().max())
    s = rows[True].set_index('unit_id')
    report['max_abs_logit_diff_stable_vs_original'] = float(np.max(np.abs(s.logit - orig.loc[s.index, 'logit'])))
    report['units_changing_predicted_hand'] = int(np.sum((s.logit > 0) != (orig.loc[s.index, 'logit'] > 0)))
    report['note'] = 'n=1 units only; identification-level adjusted-model results were not recomputed.'
    (OUT / 'bangla_adjusted_numerical_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
