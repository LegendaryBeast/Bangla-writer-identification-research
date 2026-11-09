#!/usr/bin/env python3
"""Writer-cluster uncertainty for the saved post-review check outcomes.

Run after run_revision_checks.py. Intervals condition on the fitted scores and
enrolled gallery, with equal writer and setting weights. No models are refitted.
"""
import argparse
import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261108
KEY = ["rep", "query", "n", "k", "hand_model"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def infer(values, boot, flips):
    values = np.asarray(values, float)
    estimate = values.mean()
    lo, hi = np.quantile(values[boot].mean(1), [.025, .975])
    p = (1+np.sum(np.abs(flips @ values / len(values)) >= abs(estimate)-1e-12))/(len(flips)+1)
    return dict(estimate=float(estimate), ci_lo=float(lo), ci_hi=float(hi), p_signflip=float(p))


def own_text_summaries(out, writers, boot, flips, args):
    """Use saved outcomes only; shared questioned pages are excluded."""
    mw_path = out / "checks_identification_writer.csv"
    MW = pd.read_csv(mw_path)
    M = MW[MW["query"] == "own"].pivot(index=["rep", "query", "n", "k", "writer_id"],
                                       columns="method", values="top1").reset_index()
    assert len(M[["rep", "query", "n", "k"]].drop_duplicates()) == 96
    effect = (M.both_bayes-M["none"]).groupby(M.writer_id).mean().reindex(writers).to_numpy()
    effect_row = dict(scope="own_text_96_settings", contrast="both_bayes_minus_none", **infer(effect,boot,flips))
    pd.DataFrame([effect_row]).to_csv(out / "checks_both_own_effect.csv", index=False)
    both_path = out / "checks_both_control_writer.csv.gz"
    pairs = []
    if both_path.exists():
        D = pd.read_csv(both_path)
        D = D[D["query"] == "own"]
        for kind, E in D.groupby("control"):
            delta = (E.hand_top1-E.top1).groupby(E.writer_id).mean().reindex(writers).to_numpy()
            gains = (E.top1-E.base_top1).groupby(E.instance).mean()
            hg = (E.hand_top1-E.base_top1).mean()
            pairs.append(dict(control=kind, instances=len(gains), hand_mean_gain=hg,
                              control_mean_gain=gains.mean(), control_q025=gains.quantile(.025),
                              control_q975=gains.quantile(.975),
                              p_control_ge_hand=(1+(gains >= hg).sum())/(len(gains)+1),
                              **infer(delta,boot,flips)))
        pd.DataFrame(pairs).to_csv(out / "checks_both_direct_paired_own.csv", index=False)
    (out / "checks_both_own_provenance.json").write_text(json.dumps(dict(
        seed=SEED, bootstrap=args.bootstrap, permutations=args.permutations,
        scope="Own-text only: 96 settings; paragraphs 7--8 never enter shared-paragraph auxiliary model training.",
        inference="Post-review exploratory paired writer bootstrap/sign flips, stratified by true hand and conditional on fitted evidence, enrolled gallery and realized control families.",
        remaining_dependency="Gallery writer-cross-fitted descriptor models can train on the queried writer's shared enrollment pages; this is enrolment-conditioned descriptor agreement, not globally unseen-query-writer training.",
        control_fraction="p_control_ge_hand is a finite-draw empirical exceedance fraction against designed control families, not an exchangeable permutation-test p-value for handedness itself.",
        input_sha256={str(p):digest(p) for p in [Path(__file__),mw_path]+([both_path] if both_path.exists() else [])}), indent=2)+"\n")
    print(pd.DataFrame([effect_row]).to_string(index=False))
    print(pd.DataFrame(pairs).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/wid_revision_2026-11-08")
    ap.add_argument("--units", default="results/wid_handedness_2026-10-29/features/units.csv")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--permutations", type=int, default=9999)
    ap.add_argument("--own-only", action="store_true", help="Write only own-text descriptor summaries from saved outcomes")
    args = ap.parse_args()
    out = Path(args.out)
    W = pd.read_csv(out / "checks_filter_identity_writer.csv")
    U = pd.read_csv(args.units)
    writers = sorted(W.writer_id.unique())
    h = np.array([w.startswith("L") for w in writers])
    rng = np.random.default_rng(SEED)
    boot = np.concatenate([rng.choice(np.flatnonzero(h == x), (args.bootstrap, int((h == x).sum())))
                           for x in (False, True)], 1)
    flips = rng.choice([-1,1], (args.permutations,len(writers)))
    if args.own_only:
        own_text_summaries(out,writers,boot,flips,args)
        return
    bw = np.stack([np.bincount(b, minlength=len(writers)) for b in boot]) / len(writers)
    rows = []
    groups = [("all_456", W)] + [(hm, D) for hm, D in W.groupby("hand_model")]
    for group, D in groups:
        matrices = [D.pivot(index=KEY, columns="writer_id", values=col).reindex(columns=writers).to_numpy()
                    for col in ("attribute_accuracy", "known_top1", "filter_top1")]
        am, km, fm = [X.mean(1) for X in matrices]
        ab, kb, fb = [bw @ X.T for X in matrices]
        pb, pm = ab * kb, am * km
        cb, cm = fb - pb, fm - pm
        # Correlations are descriptive across correlated setting/model points;
        # the bootstrap resamples writers, never those points independently.
        xp = pb - pb.mean(1, keepdims=True)
        yp = fb - fb.mean(1, keepdims=True)
        rb = (xp * yp).sum(1) / np.sqrt((xp*xp).sum(1) * (yp*yp).sum(1))
        row = dict(group=group, combinations=len(am))
        for name, observed, draws in [("bias", cm.mean(), cb.mean(1)),
                                      ("mae", np.abs(cm).mean(), np.abs(cb).mean(1)),
                                      ("pooled_r", np.corrcoef(pm, fm)[0, 1], rb)]:
            lo, hi = np.quantile(draws, [.025, .975])
            row.update({name: float(observed), name+"_ci_lo": float(lo), name+"_ci_hi": float(hi)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(out / "checks_dependence_uncertainty.csv", index=False)

    C = pd.read_csv(out / "checks_cross_hand_writer.csv")
    counts = {}
    for (writer, n), D in U[U.valid].groupby(["writer_id", "n"]):
        counts[(writer, n, "shared")] = int((D.para <= 6).sum())
        counts[(writer, n, "own")] = int((D.para >= 7).sum())
    C["n_query_configs"] = [counts[(r.writer_id, r.n, r.query)] * (comb(6, r.k) if r.query == "own" else 1)
                             for r in C.itertuples(index=False)]
    C["error_rate"] = C.errors / C.n_query_configs
    C["cross_error_rate"] = C.cross_hand_errors / C.n_query_configs
    result = []
    for rep, D in C.groupby("rep"):
        wm = D.groupby("writer_id")[["error_rate", "cross_error_rate"]].mean().reindex(writers)
        err, cross = wm.error_rate.to_numpy(), wm.cross_error_rate.to_numpy()
        draws = cross[boot].mean(1) / err[boot].mean(1)
        lo, hi = np.quantile(draws, [.025, .975])
        result.append(dict(rep=rep, cross_hand_share=float(cross.mean()/err.mean()),
                           ci_lo=float(lo), ci_hi=float(hi), uniform_wrong_writer_reference=21/41,
                           interpretation="Proximity to this reference does not establish independence."))
    C.to_csv(out / "checks_cross_hand_writer_weighted.csv", index=False)
    pd.DataFrame(result).to_csv(out / "checks_cross_hand_summary.csv", index=False)

    own_text_summaries(out,writers,boot,flips,args)
    both_path = out / "checks_both_control_writer.csv.gz"
    (out / "checks_followup_provenance.json").write_text(json.dumps(dict(
        seed=SEED, bootstrap=args.bootstrap,
        inference="Post-review exploratory fixed-gallery writer-stratified bootstrap; models and gallery not refitted/resampled.",
        weighting="Equal writers per setting; equal settings within representation/model summaries.",
        input_sha256={str(p):digest(p) for p in [Path(__file__), Path(args.units),
                          out/"checks_filter_identity_writer.csv", out/"checks_cross_hand_writer.csv",
                          out/"checks_identification_writer.csv"] + ([both_path] if both_path.exists() else [])}), indent=2)+"\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(result).to_string(index=False))


if __name__ == "__main__":
    main()
