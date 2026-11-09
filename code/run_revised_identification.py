#!/usr/bin/env python3
"""Reanalyse identification with reference-only scaling and equal writer weights.

Preserves every historical result.  Learned log-linear weights and both-predicted
gallery analyses are deliberately outside this main reanalysis.  The four named
comparisons are retrospective designated comparisons, not preregistered tests.
"""
import argparse
import gzip
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

import revised_protocol as rp
import run_wid_handedness as wh

SEED = 20261108
KEY = ["rep", "query", "n", "k"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def paired_test(d, rng, n_perm, n_boot):
    values = d.to_numpy()
    obs = values.mean()
    flips = rng.choice([-1, 1], size=(n_perm, len(values)))
    p = (1 + np.sum(np.abs(flips @ values / len(values)) >= abs(obs) - 1e-12)) / (n_perm + 1)
    hand = np.array([w.startswith("L") for w in d.index])
    draws = [values[hand == h][rng.integers(0, np.sum(hand == h), size=(n_boot, np.sum(hand == h)))]
             for h in (False, True)]
    boot = np.concatenate(draws, axis=1).mean(axis=1)
    lo, hi = np.quantile(boot, [.025, .975])
    return dict(diff_top1=obs, ci_lo=lo, ci_hi=hi, p_signflip=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="results/wid_handedness_2026-10-29")
    ap.add_argument("--output", default="results/wid_revision_2026-11-08")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    src, out = Path(a.source), Path(a.output)
    if src.resolve() == out.resolve():
        raise ValueError("Revision outputs must not overwrite historical results")
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    U, F, H = rp.load_inputs(src)
    hp = {hm: H[H.hand_model == hm].set_index("unit_id").logit
          for hm in ("hc183", "hc183_adj", "stroke")}
    rng = np.random.default_rng(SEED)
    n_perm, n_boot = (999, 500) if a.quick else (9999, 10000)
    summaries, writers_out, tests, errors, ties = [], [], [], [], []
    first = True
    with gzip.open(out / "id_ranks.csv.gz", "wt") as rank_file:
        for j, c in enumerate(rp.iter_score_cells(U, F)):
            key = {k: c[k] for k in KEY}
            S, t, qw = c["S"], c["t"], c["qw"]
            T = rp.fit_temperatures(S, t, qw, c["weights"])
            base = rp.ranks(S, t)
            oracle = rp.ranks(S + 10.0 * (c["ghand"][None, :] == c["qhand"][:, None]), t)
            pred = S.argmax(axis=1)
            wrong = base != 1
            cross = c["ghand"][pred] != c["qhand"]
            ew = float(np.dot(c["weights"], wrong))
            ec = float(np.dot(c["weights"], wrong & cross))
            errors.append(dict(**key, n_queries=len(t), errors=int(wrong.sum()),
                               cross_hand_errors=int((wrong & cross).sum()),
                               weighted_error=ew, weighted_cross_hand_error=ec,
                               cross_hand_share=ec / ew if ew else np.nan))
            ties.append(dict(**key, tied_top=int(((S == S.max(axis=1, keepdims=True)).sum(axis=1) > 1).sum())))
            outcomes = [("none", "-", "none", base, None),
                        ("oracle", "-", "filter", oracle, None)]
            evidence = [(hm, e.reindex(c["unit_id"]).to_numpy()) for hm, e in hp.items()]
            if c["query"] == "own":
                own = U[(U.n == c["n"]) & U.valid & (U.para >= 7)]
                avg = pd.Series(hp["hc183"].reindex(own.unit_id).to_numpy(), index=own.writer_id).groupby(level=0).mean()
                evidence.append(("hc183_writerlevel", avg.reindex(qw).to_numpy()))
            for hm, ell in evidence:
                if not np.isfinite(ell).all():
                    raise ValueError("Missing attribute evidence")
                correct = ((ell > 0) == c["qhand"]).astype(float)
                fusion = rp.ranks(S / T[:, None] + rp.match_logprob(ell, c["ghand"]), t)
                filt = rp.ranks(S + 10.0 * (c["ghand"][None, :] == (ell > 0)[:, None]), t)
                outcomes.extend([("query", hm, "bayes", fusion, correct),
                                 ("query", hm, "filter", filt, correct)])
            b_writer = rp.mean_by_writer(base == 1, qw, c["unit_id"])
            for scenario, hm, method, rank, correct in outcomes:
                tag = dict(**key, scenario=scenario, hand_model=hm, method=method)
                w1 = rp.mean_by_writer(rank == 1, qw, c["unit_id"])
                w5 = rp.mean_by_writer(rank <= 5, qw, c["unit_id"])
                wm = rp.mean_by_writer(1 / rank, qw, c["unit_id"])
                ha = rp.mean_by_writer(correct, qw, c["unit_id"]) if correct is not None else None
                summaries.append(dict(**tag, top1=w1.mean(), top5=w5.mean(), mrr=wm.mean(),
                                      hand_acc=ha.mean() if ha is not None else np.nan,
                                      n_rows=len(rank), n_writers=len(w1)))
                for writer in w1.index:
                    writers_out.append(dict(**tag, writer_id=writer, top1=w1[writer], top5=w5[writer], mrr=wm[writer],
                                            hand_acc=ha[writer] if ha is not None else np.nan))
                if scenario != "none":
                    d = w1 - b_writer
                    designated = (c["query"], c["n"], c["k"], c["rep"]) in wh.PRIMARY and hm == "hc183" and method == "bayes"
                    tests.append(dict(**tag, **paired_test(d, rng, n_perm, n_boot), primary=designated))
                rr = pd.DataFrame(dict(**tag, unit_id=c["unit_id"], writer_id=qw, config=c["config"], rank=rank,
                                       weight=c["weights"], hand_correct=correct if correct is not None else np.nan,
                                       T=T if method == "bayes" else np.nan))
                rr.to_csv(rank_file, index=False, header=first)
                first = False
            if (j + 1) % 20 == 0:
                print(f"{j+1}/120 settings complete ({time.time()-start:.1f}s)", flush=True)
    summary, test = pd.DataFrame(summaries), pd.DataFrame(tests)
    P = test[test.primary].sort_values("p_signflip")
    test["p_holm"] = np.nan
    test.loc[P.index, "p_holm"] = np.maximum.accumulate([min(1.0, (len(P) - i) * p) for i, p in enumerate(P.p_signflip)])
    summary.to_csv(out / "id_summary.csv", index=False)
    test.to_csv(out / "id_paired_tests.csv", index=False)
    pd.DataFrame(writers_out).to_csv(out / "writer_outcomes.csv", index=False)
    pd.DataFrame(errors).to_csv(out / "id_errors_cross_hand.csv", index=False)
    pd.DataFrame(ties).to_csv(out / "ranking_ties.csv", index=False)
    none = summary[summary.scenario == "none"].set_index(KEY).top1
    known = summary[summary.scenario == "oracle"].set_index(KEY).top1
    acc = summary[(summary.hand_model == "hc183") & (summary.method == "filter")].set_index(KEY).hand_acc
    pd.DataFrame(dict(base=none, oracle=known, threshold=none / known, hand_acc=acc)).reset_index().to_csv(out / "breakeven_analytic.csv", index=False)
    ha_rows = []
    for hm, evidence in hp.items():
        for n in rp.NS:
            un = U[(U.n == n) & U.valid].copy()
            ell = evidence.reindex(un.unit_id).to_numpy()
            row = dict(hand_model=hm, n=n)
            for part, mask in [("shared", un.para <= 6), ("own", un.para >= 7)]:
                row[part] = rp.mean_by_writer((ell[mask] > 0) == un.label.to_numpy()[mask], un.writer_id.to_numpy()[mask], un.unit_id.to_numpy()[mask]).mean()
            meanell = pd.Series(ell, index=un.writer_id).groupby(level=0).mean()
            row["writer_all_units"] = np.mean((meanell > 0).to_numpy() == np.array([w.startswith("L") for w in meanell.index]))
            ha_rows.append(row)
    pd.DataFrame(ha_rows).to_csv(out / "handedness_accuracy.csv", index=False)
    sources = [src / "features/units.csv", src / "features/unit_features.npz", src / "handedness_lowo.csv.gz",
               Path(__file__), Path(rp.__file__)]
    provenance = dict(source=str(src.resolve()), output=str(out.resolve()), python=platform.python_version(),
                      seed=SEED, n_perm=n_perm, n_boot=n_boot, quick=a.quick, settings=120,
                      standardization="selected whole reference paragraphs only, fitted separately per configuration",
                      temperature="ID-only NLL, equal query-writer means, scored writer excluded",
                      estimand="configuration mean per unit, then unit mean per writer, then equal mean over 42 writers",
                      uncertainty="conditional on fixed fitted evidence, observed references and writer cohort; no model refitting",
                      designated_tests="four retrospective comparisons; original paragraph feasibility cells previously inspected",
                      ties="fixed lexical gallery-writer order", hand_model_training="unchanged nested held-writer-out SVM and grouped Platt calibration",
                      sha256={str(p): digest(p) for p in sources}, seconds=round(time.time()-start, 2))
    (out / "protocol_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(test[test.primary].to_string(index=False), flush=True)
    print(f"Done in {time.time()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
