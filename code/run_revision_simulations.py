#!/usr/bin/env python3
"""Conditional Gaussian-evidence and gallery-prevalence sensitivity analyses.

All feature scaling uses selected reference paragraphs only. Simulated evidence
is generated once per distinct question unit and reused across its reference
configurations. The endpoints a=.5 and a=1 are explicit baseline/oracle limits.
Prevalence uses fixed full-cohort ID-only temperature calibration, always
excluding the scored query writer, while refitting the scaler in each reduced
gallery. It is a conditional gallery sensitivity analysis, not new populations.
"""
import argparse
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

import revised_protocol as rp

SEED = 20261109
SIM_CELLS = [("own", 1, 1), ("own", 4, 6), ("own", 1, 6), ("shared", 4, 5)]
PREV_CELLS = [("own", 1, 1), ("own", 4, 6)]


def simulate(c, n_sim, noise_cache):
    S, t, weights = c["S"], c["t"], c["weights"]
    T = rp.fit_temperatures(S, t, c["qw"], weights)
    scaled = S / T[:, None]
    base = float(np.dot(weights, S.argmax(axis=1) == t))
    oracle_scores = S + 10.0 * (c["ghand"][None, :] == c["qhand"][:, None])
    oracle = float(np.dot(weights, oracle_scores.argmax(axis=1) == t))
    uid, inv = np.unique(c["unit_id"], return_inverse=True)
    cache_key = (c["query"], c["n"])
    if cache_key not in noise_cache:
        # Same distinct units occur across k and representations for a text type
        # and size, so common noise permits paired simulation comparisons.
        seed = SEED + 100 * c["n"] + (0 if c["query"] == "own" else 1)
        noise_cache[cache_key] = (uid, np.random.default_rng(seed).standard_normal((n_sim, len(uid))))
    cached_uid, noise = noise_cache[cache_key]
    if not np.array_equal(uid, cached_uid):
        raise AssertionError("Distinct query units differ across simulation cells")
    rows = []
    for a in np.round(np.arange(.5, 1.00001, .05), 2):
        for draw in range(n_sim):
            if a == 1.0:
                realised, bayes, filt = 1.0, oracle, oracle
            else:
                if a == .5:
                    ell = np.zeros(len(t))
                    # A fair random hand decision when posterior evidence ties.
                    predicted = noise[draw, inv] > 0
                else:
                    d = 2 * norm.ppf(a)
                    ell = d * ((2 * c["qhand"] - 1) * d / 2 + noise[draw, inv])
                    predicted = ell > 0
                realised = float(np.dot(weights, predicted == c["qhand"]))
                bayes = float(np.dot(weights, (scaled + rp.match_logprob(ell, c["ghand"])).argmax(axis=1) == t))
                filt = float(np.dot(weights, (S + 10 * (c["ghand"][None, :] == predicted[:, None])).argmax(axis=1) == t))
                if a == .5 and abs(bayes - base) > 1e-12:
                    raise AssertionError("Uninformative probabilistic fusion changed the baseline")
            rows.append(dict(rep=c["rep"], query=c["query"], n=c["n"], k=c["k"], acc=a, sim=draw,
                             realised_acc=realised, base=base, oracle=oracle, bayes=bayes, filter=filt))
    return rows


def gallery_result(U, X, c, left_subset, ell_by_id, full_temperature):
    """A reduced gallery with its own selected-reference-only scaler."""
    writers = tuple(w for w in c["writers"] if w.startswith("R") or w in left_subset)
    writer_set = set(writers)
    widx = {w: j for j, w in enumerate(writers)}
    ghand = np.array([w.startswith("L") for w in writers])
    full = {(w, p): i for i, (w, p, n) in enumerate(zip(U.writer_id, U.para, U.n)) if n == 1}
    q = U.index[U.valid & (U.n == c["n"]) & (U.para >= 7) & U.writer_id.isin(writer_set)].to_numpy()
    qw = U.writer_id.to_numpy()[q]
    t = np.array([widx[w] for w in qw])
    qhand = U.label.to_numpy()[q]
    ell = ell_by_id.reindex(U.unit_id.to_numpy()[q]).to_numpy()
    T = np.array([full_temperature[w] for w in qw])
    refs = list(itertools.combinations(range(1, 7), c["k"]))
    accum = {method: [] for method in ("none", "bayes", "filter", "filter_prior")}
    for reference in refs:
        ri = np.array([full[(w, r)] for w in writers for r in reference])
        if np.intersect1d(q, ri).size:
            raise AssertionError("Question in reduced-gallery scaler")
        scaler = StandardScaler().fit(X[ri])
        zr, zq = scaler.transform(X[ri]), scaler.transform(X[q])
        zr /= np.maximum(np.linalg.norm(zr, axis=1, keepdims=True), 1e-12)
        zq /= np.maximum(np.linalg.norm(zq, axis=1, keepdims=True), 1e-12)
        templates = zr.reshape(len(writers), len(reference), -1).mean(axis=1)
        templates /= np.maximum(np.linalg.norm(templates, axis=1, keepdims=True), 1e-12)
        S = zq @ templates.T
        prior = np.log(len(left_subset) / np.sum(~ghand))
        scores = dict(none=S, bayes=S / T[:, None] + rp.match_logprob(ell, ghand.astype(int)),
                      filter=S + 10 * (ghand[None] == (ell > 0)[:, None]),
                      filter_prior=S + 10 * (ghand[None] == (ell + prior > 0)[:, None]))
        for method, score in scores.items():
            accum[method].append(rp.ranks(score, t))
    rows = []
    for method, rank_list in accum.items():
        ranks = np.stack(rank_list)
        # Configurations are averaged per unit before equal writer averaging.
        top = rp.mean_by_writer((ranks == 1).mean(axis=0), qw)
        pen = rp.mean_by_writer(ranks.mean(axis=0) / len(writers), qw)
        for hand in (0, 1):
            group = [w for w in top.index if w.startswith("L") == bool(hand)]
            rows.append(dict(method=method, query_hand="left" if hand else "right",
                             top1=float(top.loc[group].mean()), mean_pen=float(pen.loc[group].mean()),
                             gallery=len(writers), n_queries=int(np.sum(qhand == hand))))
    return rows


def prevalence(U, F, c, ell_by_id, n_draws):
    X = F[c["rep"]].astype(float)
    T = rp.fit_temperatures(c["S"], c["t"], c["qw"], c["weights"])
    temperatures = pd.Series(T, index=c["qw"]).groupby(level=0).first().to_dict()
    left = [w for w in c["writers"] if w.startswith("L")]
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for m in (1, 2, 3):
        subsets = Counter(tuple(sorted(rng.choice(left, m, replace=False))) for _ in range(n_draws))
        sums = {}
        for subset, count in subsets.items():
            for row in gallery_result(U, X, c, subset, ell_by_id, temperatures):
                key = (row["method"], row["query_hand"])
                if key not in sums:
                    sums[key] = dict(top1=0., mean_pen=0., n_queries=0.)
                for metric in sums[key]:
                    sums[key][metric] += count * row[metric] / n_draws
        for (method, hand), values in sums.items():
            rows.append(dict(rep=c["rep"], query=c["query"], n=c["n"], k=c["k"], n_left=m,
                             method=method, query_hand=hand, **values, gallery=21+m,
                             draws=n_draws, unique_galleries=len(subsets)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/wid_handedness_2026-10-29")
    ap.add_argument("--output", default="results/wid_revision_2026-11-08")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-prevalence", action="store_true")
    ap.add_argument("--prevalence-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    if Path(a.source).resolve() == out.resolve():
        raise ValueError("Do not overwrite historical results")
    U, F, H = rp.load_inputs(a.source)
    ell = H[H.hand_model == "hc183"].set_index("unit_id").logit
    n_sim, n_draws = (5, 20) if a.quick else (200, 1000)
    start = time.time()
    sim_rows, prev_rows, noise_cache = [], [], {}
    for c in rp.iter_score_cells(U, F):
        key = (c["query"], c["n"], c["k"])
        if key in SIM_CELLS and not a.prevalence_only:
            sim_rows.extend(simulate(c, n_sim, noise_cache))
            print(f"Simulation {c['rep']} {key} complete ({time.time()-start:.1f}s)", flush=True)
        if not a.skip_prevalence and key in PREV_CELLS:
            prev_rows.extend(prevalence(U, F, c, ell, n_draws))
            print(f"Prevalence {c['rep']} {key} complete ({time.time()-start:.1f}s)", flush=True)
    if sim_rows:
        pd.DataFrame(sim_rows).to_csv(out / "breakeven_simulation.csv.gz", index=False)
    if prev_rows:
        pd.DataFrame(prev_rows).to_csv(out / "prevalence.csv", index=False)
    provenance = dict(seed=SEED, n_sim=n_sim, n_draws=n_draws, quick=a.quick,
                      gaussian="z|h ~ Normal((2h-1)d/2,1); ell=d*z; d=2*Phi^-1(a)",
                      query_noise="one draw per unique query unit, reused across reference configurations, k, and representations",
                      endpoint_half="ell=0; score fusion exactly baseline; fair random hand for filtering",
                      endpoint_one="both fusion and filtering exactly oracle, explicit limiting value",
                      weights="configuration mean per unit, unit mean per writer, equal writer mean",
                      prevalence_scaler="selected reference paragraphs of selected reduced-gallery writers only",
                      prevalence_temperature="frozen full-42-cohort ID-only calibration for the cell, scored query writer excluded",
                      prevalence_attribute="original writer-held-out HC183 logodds; approximate balanced source-prior evidence retained unchanged",
                      prevalence_prior_filter="add log(m/21) to evidence only for filter_prior; score fusion unchanged",
                      inference="Monte Carlo variability conditional on observed cohort and fixed fitted models; no new-population validation",
                      seconds=round(time.time()-start,2))
    provenance_name = "prevalence_provenance.json" if a.prevalence_only else "simulation_provenance.json"
    (out / provenance_name).write_text(json.dumps(provenance, indent=2)+"\n")
    print(f"Saved {len(sim_rows)} simulation rows and {len(prev_rows)} prevalence rows in {time.time()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
