#!/usr/bin/env python3
"""Assemble corrected CSAFE outputs: archived rows plus the stable adjusted-HC-183 refit.

Only rows with cohort == 'full_balanced' and hand_model == 'hc183_adj' are replaced,
using results/p1_revision_2026-11-09/csafe_adjusted_fix (see run_p1_csafe_adjusted_fix.py).
All primary (unadjusted HC-183) rows, sub-gallery rows and p-values are kept exactly as
archived. The summary script is then rerun on the corrected directory to regenerate the
RESULTS.md report, calibration diagnostics, interval tables and the two CSAFE figures.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'results/csafe_replication_2026-11-08'
FIX = ROOT / 'results/p1_revision_2026-11-09/csafe_adjusted_fix'
OUT = ROOT / 'results/p1_revision_2026-11-09/csafe_corrected'
TABLES = ['id_summary.csv', 'paired_tests.csv', 'handedness_summary.csv', 'diagnostics.csv',
          'writer_outcomes.csv.gz', 'selected_parameters.csv', 'lambda_loss_profiles.csv.gz']


def replace(name):
    old = pd.read_csv(SRC / name)
    new = pd.read_csv(FIX / name)
    mask_old = (old.cohort == 'full_balanced') & (old.hand_model == 'hc183_adj')
    mask_new = (new.cohort == 'full_balanced') & (new.hand_model == 'hc183_adj')
    assert mask_old.sum() == mask_new.sum() > 0, name
    if 'primary' in new:
        assert not new.loc[mask_new, 'primary'].any()
    # Replace in place, aligned by identifying keys, so row order (and therefore any
    # downstream seeded bootstrap sequence) is preserved for every untouched row.
    key = [c for c in ('cohort', 'condition', 'rep', 'hand_model', 'method', 'writer_id', 'fusion_weight') if c in old]
    sub = new[mask_new].reindex(columns=old.columns)
    sub = old.loc[mask_old, key].merge(sub, on=key, how='left', validate='one_to_one')
    assert len(sub) == mask_old.sum()
    merged = old.copy()
    merged.loc[mask_old, :] = sub[old.columns].to_numpy()
    merged = merged.astype(old.dtypes.to_dict())
    merged.to_csv(OUT / name, index=False)
    return int(mask_old.sum())


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    for p in SRC.iterdir():
        if p.is_file() and p.name not in TABLES:
            shutil.copy2(p, OUT / p.name)
    shutil.copytree(SRC / 'attribute_cache', OUT / 'attribute_cache')
    replaced = {name: replace(name) for name in TABLES}
    old_manifest = json.loads((SRC / 'fitting_manifest.json').read_text())
    new_manifest = json.loads((FIX / 'fitting_manifest.json').read_text())
    new_adj = [m for m in new_manifest if m['hand_model'] == 'hc183_adj']
    merged = [m for m in old_manifest if not (m['cohort'] == 'full_balanced' and m['hand_model'] == 'hc183_adj')] + new_adj
    assert len(merged) == len(old_manifest)
    (OUT / 'fitting_manifest.json').write_text(json.dumps(merged, indent=2) + '\n')
    for m in new_adj:
        for ext in ('.json', '.npz'):
            shutil.copy2(FIX / 'attribute_cache' / (m['cache_key'] + ext), OUT / 'attribute_cache' / (m['cache_key'] + ext))
    # Primary rows must be untouched.
    for name in ('id_summary.csv', 'paired_tests.csv'):
        a, b = pd.read_csv(SRC / name), pd.read_csv(OUT / name)
        ka = a[a.hand_model == 'hc183'].sort_values(list(a.columns[:5])).reset_index(drop=True)
        kb = b[b.hand_model == 'hc183'].sort_values(list(b.columns[:5])).reset_index(drop=True)
        pd.testing.assert_frame_equal(ka, kb)
    note = dict(corrected_rows=replaced, source=str(SRC.relative_to(ROOT)), correction=str(FIX.relative_to(ROOT)),
                scope='full_balanced cohort, hc183_adj sensitivity model only; primary and sub-gallery rows unchanged',
                timing='post hoc implementation correction after CSAFE performance tables were inspected',
                adjusted_paired_test_p_values='from the corrected rerun (descriptive, outside the primary family)')
    (OUT / 'CORRECTION.json').write_text(json.dumps(note, indent=2) + '\n')
    import summarize_csafe_replication as scr
    scr.OUT = OUT
    scr.main()
    cal = pd.read_csv(OUT / 'final_calibrator_diagnostics.csv')
    print('nonpositive raw slopes by model:', cal.groupby('hand_model').raw_slope.apply(lambda v: int((v <= 0).sum())).to_dict())
    print(json.dumps(note, indent=2))


if __name__ == '__main__':
    main()
