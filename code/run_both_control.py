#!/usr/bin/env python3
"""Control for the gain of scenario "both" (query and gallery hands predicted) in run_wid_handedness.py.

When both hands are predicted, the fusion term rewards gallery writers whose predicted log-odds agree with the query's.
A writer's units get similar scores from any fixed projection of their features, so the agreement can carry writer
identity without carrying handedness. The same fusion is repeated with handedness evidence replaced by
  pseudo     the same leave-one-writer-out HC-183 SVM and Platt calibration, trained on a random 21/21 split of the
             writers balanced for hand (10 or 11 left-handed writers per group, so the split is unrelated to hand);
  pseudo_scaled  the same pseudo-label scores, scaled to the spread of the real log-odds (calibration shrinks
             uninformative scores towards 0, which alone would remove their effect);
  random     a random direction of the standardised HC-183 features, scaled to the spread of the real log-odds.
If the handedness gain lies within the distribution of these controls, it reflects identity, not handedness.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import log_softmax

import run_wid_handedness as wh

N_PSEUDO, N_RANDOM, SEED = 20, 100, 2041


def pseudo_logits(U, F, labels, jobs):
    """Leave-one-writer-out calibrated log-odds of a writer-level pseudo label, for every unit and n."""
    y = U.writer_id.map(labels).to_numpy()
    C = np.column_stack([U.ink_density.to_numpy(), F["global_intensity"]])
    out = pd.Series(np.nan, index=U.unit_id.to_numpy())
    for n in wh.NS:
        un = (U.n == n) & U.valid
        res = Parallel(n_jobs=jobs)(
            delayed(wh.lowo_writer)(F["hc183"], y, C, U.writer_id.to_numpy(),
                                    (un & (U.para <= 6) & (U.writer_id != w)).to_numpy(),
                                    np.flatnonzero((un & (U.writer_id == w)).to_numpy()), False, wh.SEED)
            for w in sorted(U.writer_id.unique()))
        for idx, _, logit in res:
            out.iloc[idx] = logit
    return out


def match_spread(U, p, real):
    """Centre p and scale it per n to the standard deviation of the real log-odds."""
    out = pd.Series(np.nan, index=U.unit_id.to_numpy())
    for n in wh.NS:
        m = ((U.n == n) & U.valid).to_numpy()
        out[m] = (p[m] - np.nanmean(p[m])) / np.nanstd(p[m]) * real.reindex(U.unit_id[m]).std()
    return out


def random_logits(U, F, real, rng):
    """A random direction of standardised HC-183, scaled per n to the spread of the real log-odds."""
    Z = wh.normalised(F, U, "hc183")
    return match_spread(U, Z @ rng.standard_normal(Z.shape[1]), real)


def both_bayes_top1(S, t, qw, ell, gl, ghand):
    M = wh.match_logprob(ell, gl, ghand)
    ll = np.stack([log_softmax(S / T, 1)[np.arange(len(t)), t] for T in wh.T_GRID], 1)
    Tq = wh.T_GRID[wh.fit_by_writer(-ll, qw)][:, None]
    return float(np.mean(wh.ranks(log_softmax(S / Tq + M, 1), t) == 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--run", required=True, help="output folder of run_wid_handedness.py (handedness_lowo.csv.gz)")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    fin, run = Path(a.features), Path(a.run)
    t0 = time.time()
    U = pd.read_csv(fin / "units.csv")
    z = np.load(fin / "unit_features.npz", allow_pickle=True)
    F = {k: z[k] for k in z.files if k != "unit_id"}
    writers = sorted(U.writer_id.unique())
    ghand = np.array([1 if w.startswith("L") else 0 for w in writers])
    n_pseudo, n_random = (3, 5) if a.quick else (N_PSEUDO, N_RANDOM)
    rng = np.random.default_rng(SEED)

    H = pd.read_csv(run / "handedness_lowo.csv.gz")
    real = H[H.hand_model == "hc183"].set_index("unit_id").logit
    evid = {("hand", 0): real}
    L, R = [w for w in writers if w.startswith("L")], [w for w in writers if w.startswith("R")]
    splits = []
    for b in range(n_pseudo):
        nl = len(L) // 2 + b % 2 * (len(L) % 2)                    # 10 or 11 of the 21 left-handed writers in group 1
        g1 = set(rng.choice(L, nl, replace=False)) | set(rng.choice(R, len(writers) // 2 - nl, replace=False))
        labels = {w: int(w in g1) for w in writers}
        splits.append(sorted(g1))
        evid[("pseudo", b)] = pseudo_logits(U, F, labels, a.jobs)
        evid[("pseudo_scaled", b)] = match_spread(U, evid[("pseudo", b)].to_numpy(), real)
        print(f"  pseudo split {b} trained ({time.time() - t0:.0f}s)", flush=True)
    for b in range(n_random):
        evid[("random", b)] = random_logits(U, F, real, rng)

    widx = {w: i for i, w in enumerate(writers)}
    full_idx = {(w, p): i for i, (w, p, n) in enumerate(zip(U.writer_id, U.para, U.n)) if n == 1}
    rows = []
    for rep in wh.ID_REPS:
        Z = wh.normalised(F, U, rep)
        for cell in wh.cells():
            parts = list(wh.cell_queries(U, writers, cell, Z, full_idx))
            q = np.concatenate([p[2] for p in parts]); S = np.concatenate([p[3] for p in parts])
            conf = np.concatenate([np.full(len(p[2]), p[0]) for p in parts])
            t = np.array([widx[w] for w in U.writer_id.to_numpy()[q]]); qw = U.writer_id.to_numpy()[q]
            base = float(np.mean(wh.ranks(S, t) == 1))
            for (kind, b), e in evid.items():
                ell = e.reindex(U.unit_id.to_numpy()[q]).to_numpy()
                gl = np.empty_like(S)
                for c, (_, refs, _, _) in enumerate(parts):
                    gl[conf == c] = [np.mean([e.get(f"{w}_para{r:02d}_n1_s0", np.nan) for r in refs]) for w in writers]
                rows.append(dict(rep=rep, **cell, control=kind, instance=b, base=base,
                                 both_bayes=both_bayes_top1(S, t, qw, ell, gl, ghand),
                                 query_acc_own_label=np.nan if kind != "hand" else float(np.mean((ell > 0) == ghand[t]))))
        print(f"  {rep} done ({time.time() - t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D["gain"] = D.both_bayes - D.base
    D.to_csv(run / "both_control.csv", index=False)

    # empirical p: share of control instances whose mean gain over all 120 cells is >= that of handedness
    agg = D.groupby(["control", "instance"]).gain.mean().reset_index()
    hand = agg[agg.control == "hand"].gain.iloc[0]
    summ = []
    for kind in ["pseudo", "pseudo_scaled", "random"]:
        g = agg[agg.control == kind].gain.to_numpy()
        summ.append(dict(control=kind, instances=len(g), hand_mean_gain=hand, control_mean_gain=g.mean(),
                         control_sd=g.std(ddof=1), control_q05=np.quantile(g, .05), control_q95=np.quantile(g, .95),
                         p_ge_hand=(1 + np.sum(g >= hand)) / (len(g) + 1)))
    pd.DataFrame(summ).to_csv(run / "both_control_summary.csv", index=False)
    (run / "both_control_provenance.json").write_text(json.dumps(dict(
        n_pseudo=n_pseudo, n_random=n_random, seed=SEED, pseudo_groups=splits, seconds=round(time.time() - t0)), indent=2))
    print(pd.DataFrame(summ).round(4).to_string(index=False))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
