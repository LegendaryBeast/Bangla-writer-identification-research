#!/usr/bin/env python3
"""Reference-only scoring and writer-balanced estimands for the revision.

No questioned image enters the identification scaler.  The scaler is refitted
for each reference configuration using only its selected whole paragraphs.
The handedness evidence remains the independently calibrated writer-held-out
evidence of the original run.  Temperature uses identity scores only, with the
scored writer excluded from its fitting loss.
"""
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.preprocessing import StandardScaler

import run_wid_handedness as wh

ID_REPS = wh.ID_REPS
T_GRID = wh.T_GRID
NS = wh.NS
OWN_K = wh.OWN_K


def load_inputs(run):
    run = Path(run)
    U = pd.read_csv(run / "features/units.csv")
    with np.load(run / "features/unit_features.npz", allow_pickle=True) as z:
        if not np.array_equal(z["unit_id"], U.unit_id.to_numpy()):
            raise ValueError("Feature rows do not match units.csv")
        F = {k: z[k] for k in z.files if k != "unit_id"}
    H = pd.read_csv(run / "handedness_lowo.csv.gz")
    return U, F, H


def ranks(score, t):
    """One-based rank, resolving exact ties by fixed gallery column order."""
    score = np.asarray(score)
    if not np.isfinite(score).all():
        raise ValueError("Nonfinite ranking score")
    v = score[np.arange(len(t)), t][:, None]
    earlier = np.arange(score.shape[1])[None, :] < np.asarray(t)[:, None]
    return 1 + ((score > v) | ((score == v) & earlier)).sum(axis=1)


def mean_by_writer(values, qw, unit_id=None):
    """Average configurations per query unit, then units per writer."""
    D = pd.DataFrame({"writer_id": qw, "value": values})
    if unit_id is not None:
        D["unit_id"] = unit_id
        D = D.groupby(["writer_id", "unit_id"], sort=True).value.mean().reset_index()
    return D.groupby("writer_id", sort=True).value.mean()


def row_weights(qw, unit_id):
    D = pd.DataFrame({"writer_id": qw, "unit_id": unit_id})
    configs = D.groupby(["writer_id", "unit_id"]).unit_id.transform("size").to_numpy()
    units = D.groupby("writer_id").unit_id.transform("nunique").to_numpy()
    return 1.0 / (D.writer_id.nunique() * units * configs)


def fit_temperatures(S, t, qw, weights=None):
    """Temperature per query; fit ID-only NLL on the other query writers."""
    if weights is None:
        _, inv, counts = np.unique(qw, return_inverse=True, return_counts=True)
        weights = 1.0 / counts[inv]
    weights = np.asarray(weights)
    loss = np.stack([logsumexp(S / T, axis=1) - S[np.arange(len(t)), t] / T
                     for T in T_GRID], axis=1)
    weighted = loss * weights[:, None]
    total = weighted.sum(axis=0)
    out = np.empty(len(t))
    for writer in np.unique(qw):
        m = qw == writer
        out[m] = T_GRID[np.argmin(total - weighted[m].sum(axis=0))]
    return out


def match_logprob(ell, ghand):
    """Stable log evidence for each known gallery hand (balanced source prior)."""
    ell = np.asarray(ell)
    logs = np.stack([-np.logaddexp(0.0, ell), -np.logaddexp(0.0, -ell)], axis=1)
    return logs[:, ghand]


def iter_score_cells(U, F, reps=None):
    """Yield one dictionary per setting, including strict reference-only scores.

    Keys: rep/query/n/k/q/S/config/refs/writers/t/qw/weights/unit_id/qhand/ghand.
    refs[c] gives the paragraph numbers used in configuration c. All four query
    sizes share each scaler fit; their question images never enter that fit.
    """
    writers = tuple(sorted(U.writer_id.unique()))
    widx = {w: j for j, w in enumerate(writers)}
    full = {(w, p): i for i, (w, p, n) in enumerate(zip(U.writer_id, U.para, U.n)) if n == 1}
    ghand = np.array([int(w.startswith("L")) for w in writers])
    un = U.n.to_numpy()
    for rep in (ID_REPS if reps is None else reps):
        X = F[rep].astype(float)
        for query, k in [("shared", 5)] + [("own", k) for k in OWN_K]:
            configs = ([(p, tuple(r for r in range(1, 7) if r != p)) for p in range(1, 7)]
                       if query == "shared" else
                       [(None, s) for s in itertools.combinations(range(1, 7), k)])
            pieces = []
            for c, (p, refs) in enumerate(configs):
                ri = np.array([full[(w, r)] for w in writers for r in refs])
                q = U.index[U.valid & ((U.para == p) if query == "shared" else (U.para >= 7))].to_numpy()
                if np.intersect1d(q, ri).size:
                    raise AssertionError("Question image included in reference scaler")
                scaler = StandardScaler().fit(X[ri])
                zr = scaler.transform(X[ri]); zq = scaler.transform(X[q])
                zr /= np.maximum(np.linalg.norm(zr, axis=1, keepdims=True), 1e-12)
                zq /= np.maximum(np.linalg.norm(zq, axis=1, keepdims=True), 1e-12)
                templates = zr.reshape(len(writers), len(refs), -1).mean(axis=1)
                templates /= np.maximum(np.linalg.norm(templates, axis=1, keepdims=True), 1e-12)
                pieces.append((c, q, zq @ templates.T))
            for n in NS:
                q = np.concatenate([qi[un[qi] == n] for _, qi, _ in pieces])
                S = np.concatenate([s[un[qi] == n] for _, qi, s in pieces])
                config = np.concatenate([np.full(np.sum(un[qi] == n), c) for c, qi, _ in pieces])
                qw = U.writer_id.to_numpy()[q]
                uid = U.unit_id.to_numpy()[q]
                yield dict(rep=rep, query=query, n=n, k=k, q=q, S=S, config=config,
                           refs=[r for _, r in configs], writers=writers,
                           t=np.array([widx[w] for w in qw]), qw=qw,
                           weights=row_weights(qw, uid), unit_id=uid,
                           qhand=U.label.to_numpy()[q], ghand=ghand)
