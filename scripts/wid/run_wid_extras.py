#!/usr/bin/env python3
"""Leakage control, confidence gating and the filter law for the handedness / writer identification study.

Run after run_wid_handedness.py (uses its handedness_lowo.csv.gz, id_summary.csv and breakeven_simulation.csv.gz).

1. Leakage control. HC-183 SVM handedness evidence from two training regimes, each with the real labels and with
   N_PSEUDO random 21/21 writer splits balanced for hand (10 or 11 left-handed writers per group):
     writer-independent  leave one writer out (as in run_wid_handedness.py);
     enrolment-trained   trained on the shared units of all enrolled writers, the queried writer included: shared
                         units scored by 5-fold cross-validation over paragraphs (strips of one paragraph kept
                         together), own-text units by the model fitted on all shared units; Platt calibration on the
                         cross-validated scores.
   Reported: unit accuracy of the label, and the change of top-1 when the evidence is fused with identification.
2. Confidence gating. The writer-independent evidence is used (filter) only for the most confident share of queries.
3. Filter law. Top-1 with the hand filter against handedness accuracy x top-1 with known hands, for every cell.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import log_softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

import run_wid_handedness as wh

N_PSEUDO, SEED = 20, 2051
COVERAGES = (1.0, 0.75, 0.5, 0.3, 0.2, 0.1)


def splits(writers, n, rng):
    L, R = [w for w in writers if w[0] == "L"], [w for w in writers if w[0] == "R"]
    out = []
    for b in range(n):
        nl = len(L) // 2 + b % 2 * (len(L) % 2)
        g1 = set(rng.choice(L, nl, replace=False)) | set(rng.choice(R, len(writers) // 2 - nl, replace=False))
        out.append({w: int(w in g1) for w in writers})
    return out


def lowo(U, F, C, y, jobs):
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


def enrolled(U, F, C, y):
    out = pd.Series(np.nan, index=U.unit_id.to_numpy())
    X = F["hc183"]
    for n in wh.NS:
        tr = np.flatnonzero(((U.n == n) & U.valid & (U.para <= 6)).to_numpy())
        own = np.flatnonzero(((U.n == n) & U.valid & (U.para >= 7)).to_numpy())
        oof = np.empty(len(tr))
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=wh.SEED).split(tr, y[tr], U.image_id.to_numpy()[tr]):
            oof[b] = wh.fit_score(X[tr[a]], y[tr[a]], C[tr[a]], X[tr[b]], C[tr[b]], False)
        raw_own = wh.fit_score(X[tr], y[tr], C[tr], X[own], C[own], False)
        p = LogisticRegression(C=1e4).fit(oof[:, None], y[tr])
        out.iloc[tr] = p.coef_[0, 0] * oof + p.intercept_[0]
        out.iloc[own] = p.coef_[0, 0] * raw_own + p.intercept_[0]
    return out


def fused_top1(S, t, qw, ell, glab):
    """Bayes fusion (gallery labels known) and hand filter; T fitted leaving the query writer out."""
    M = wh.match_logprob(ell, None, glab)
    ll = np.stack([log_softmax(S / T, 1)[np.arange(len(t)), t] for T in wh.T_GRID], 1)
    Tq = wh.T_GRID[wh.fit_by_writer(-ll, qw)][:, None]
    bayes = np.mean(wh.ranks(log_softmax(S / Tq + M, 1), t) == 1)
    filt = np.mean(wh.ranks(S + 10.0 * (glab[None] == (ell > 0)[:, None]), t) == 1)
    return bayes, filt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    fin, run = Path(a.features), Path(a.run)
    t0 = time.time()
    U = pd.read_csv(fin / "units.csv")
    z = np.load(fin / "unit_features.npz", allow_pickle=True)
    F = {k: z[k] for k in z.files if k != "unit_id"}
    writers = sorted(U.writer_id.unique()); widx = {w: i for i, w in enumerate(writers)}
    ghand = np.array([1 if w.startswith("L") else 0 for w in writers])
    C = np.column_stack([U.ink_density.to_numpy(), F["global_intensity"]])
    rng = np.random.default_rng(SEED)
    n_pseudo = 3 if a.quick else N_PSEUDO
    H = pd.read_csv(run / "handedness_lowo.csv.gz")

    # ---- 1. leakage control: evidence for every label set and regime
    labsets = [("hand", 0, {w: int(w[0] == "L") for w in writers})] + \
              [("pseudo", b, lab) for b, lab in enumerate(splits(writers, n_pseudo, rng))]
    evid = {}
    for kind, b, lab in labsets:
        y = U.writer_id.map(lab).to_numpy()
        evid[(kind, b, "writer-independent")] = (H[H.hand_model == "hc183"].set_index("unit_id").logit
                                                 if kind == "hand" else lowo(U, F, C, y, a.jobs))
        evid[(kind, b, "enrolment-trained")] = enrolled(U, F, C, y)
        print(f"  evidence {kind} {b} ({time.time() - t0:.0f}s)", flush=True)
    acc_rows = []
    for (kind, b, regime), e in evid.items():
        lab = dict(labsets[0 if kind == "hand" else b + 1][2])
        ok = U.valid.to_numpy()
        y = U.writer_id.map(lab).to_numpy()
        corr = (e.reindex(U.unit_id).to_numpy() > 0) == (y == 1)
        for n in wh.NS:
            for part, pm in [("shared", U.para <= 6), ("own", U.para >= 7)]:
                m = ok & (U.n == n).to_numpy() & pm.to_numpy()
                acc_rows.append(dict(label=kind, split=b, regime=regime, n=n, part=part, acc=corr[m].mean()))
    A = pd.DataFrame(acc_rows)
    A.to_csv(run / "leakage_accuracy.csv", index=False)

    full_idx = {(w, p): i for i, (w, p, n) in enumerate(zip(U.writer_id, U.para, U.n)) if n == 1}
    id_rows, gate_rows = [], []
    Hp = evid[("hand", 0, "writer-independent")]
    for rep in wh.ID_REPS:
        Z = wh.normalised(F, U, rep)
        for cell in wh.cells():
            parts = list(wh.cell_queries(U, writers, cell, Z, full_idx))
            q = np.concatenate([p[2] for p in parts]); S = np.concatenate([p[3] for p in parts])
            t = np.array([widx[w] for w in U.writer_id.to_numpy()[q]]); qw = U.writer_id.to_numpy()[q]
            base_r = wh.ranks(S, t) == 1
            for (kind, b, regime), e in evid.items():
                lab = labsets[0 if kind == "hand" else b + 1][2]
                glab = np.array([lab[w] for w in writers])
                ell = e.reindex(U.unit_id.to_numpy()[q]).to_numpy()
                bayes, filt = fused_top1(S, t, qw, ell, glab)
                id_rows.append(dict(rep=rep, **cell, label=kind, split=b, regime=regime, base=base_r.mean(),
                                    acc=np.mean((ell > 0) == glab[t]), bayes=bayes, filter=filt))
            # ---- 2. confidence gating (writer-independent handedness)
            ell = Hp.reindex(U.unit_id.to_numpy()[q]).to_numpy()
            for cov in COVERAGES:
                g = np.abs(ell) >= np.quantile(np.abs(ell), 1 - cov)
                filt = wh.ranks(S + 10.0 * (g[:, None] & (ghand[None] == (ell > 0)[:, None])), t) == 1
                gate_rows.append(dict(rep=rep, **cell, coverage=cov, acc_gated=np.mean((ell[g] > 0) == ghand[t][g]),
                                      base=base_r.mean(), filter_gated=filt.mean(),
                                      oracle_gated=np.mean(np.where(g, wh.ranks(S + 10.0 * (ghand[None] == ghand[t][:, None]), t) == 1, base_r))))
        print(f"  {rep} done ({time.time() - t0:.0f}s)", flush=True)
    I = pd.DataFrame(id_rows)
    I["gain_bayes"] = I.bayes - I.base; I["gain_filter"] = I["filter"] - I.base
    I.to_csv(run / "leakage_identification.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(run / "confidence_gating.csv", index=False)

    # ---- 3. filter law and Bayes gap to simulation
    S_ = pd.read_csv(run / "id_summary.csv"); key = ["rep", "query", "n", "k"]
    orc = S_[S_.scenario == "oracle"].set_index(key).top1
    law = []
    for hm in ["hc183", "hc183_adj", "stroke", "hc183_writerlevel"]:
        f = S_[(S_.scenario == "query") & (S_.hand_model == hm) & (S_.method == "filter")].set_index(key)
        b = S_[(S_.scenario == "query") & (S_.hand_model == hm) & (S_.method == "bayes")].set_index(key).top1
        law.append(pd.DataFrame(dict(hand_model=hm, hand_acc=f.hand_acc, oracle=orc.reindex(f.index), filter=f.top1,
                                     predicted=f.hand_acc * orc.reindex(f.index), bayes=b.reindex(f.index))).reset_index())
    Lw = pd.concat(law, ignore_index=True)
    sim = pd.read_csv(run / "breakeven_simulation.csv.gz").groupby(key + ["acc"]).bayes.mean()
    Lw["bayes_sim_same_acc"] = [np.interp(r.hand_acc, sim.loc[k].index, sim.loc[k].values) if k in sim.index.droplevel("acc") else np.nan
                                for k, r in zip(Lw[key].itertuples(index=False, name=None), Lw.itertuples())]
    Lw.to_csv(run / "filter_law.csv", index=False)
    fit = dict(r=float(np.corrcoef(Lw.predicted, Lw["filter"])[0, 1]),
               mae=float(np.mean(np.abs(Lw.predicted - Lw["filter"]))), bias=float(np.mean(Lw["filter"] - Lw.predicted)),
               n_points=int(len(Lw)))
    (run / "extras_provenance.json").write_text(json.dumps(dict(
        n_pseudo=n_pseudo, seed=SEED, coverages=COVERAGES, filter_law=fit, seconds=round(time.time() - t0)), indent=2))
    print(A.groupby(["label", "regime", "part"]).acc.mean().unstack().round(3))
    print("filter law", fit)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
