#!/usr/bin/env python3
"""Does handedness help writer identification?  Closed-set identification of the 42 BanglaHand writers with and
without handedness evidence.

Input: unit_features.npz and units.csv of wid_features.py (whole paragraphs, n = 1, and n horizontal strips).

1. Handedness evidence, leave one writer out. For every writer, a linear SVM (C = 0.1, the PRL primary) is trained on
   the shared-prompt units (paragraphs 1-6) of the same size n of the other 41 writers and scores all units of the
   held-out writer. Scores are Platt-calibrated to a log-odds of being left-handed with inner 5-fold cross-validation
   over the 41 training writers. Hand models: hc183 (primary), hc183_adj (ink density and photo intensity regressed
   out inside each training set), stroke (writing-only).
2. Identification. Features are standardised (all shared paragraphs, label free) and L2-normalised; a writer template
   is the mean of their reference paragraphs (whole shared paragraphs); similarity is the cosine.
   Cells: query shared (unit of paragraph p, references = the other 5 prompts for every writer) or own text
   (paragraphs 7-8, references = every subset of k = 1, 2, 3 or 6 shared prompts), for n = 1, 2, 4, 8.
3. Handedness scenarios: oracle (both hands known), query (gallery hands known, query hand predicted; primary),
   both (gallery hands predicted from the writer's references with the leave-one-writer-out model).
4. Fusion: none; filter (rank writers of the predicted hand first); bayes (log softmax(s/T) + log P(hand of w | x));
   loglin (T and weight lambda); rerank5 (bayes among the top 5). T and lambda are fitted by leave-one-writer-out over
   query writers (minimum negative log-likelihood of the true writer), never on the scored writer.
5. Statistics: writer-level paired differences of top-1 (sign-flip permutation, writer bootstrap stratified by hand),
   exact McNemar where a cell has one reference configuration; Holm over the four primary tests.
6. Break-even: analytic threshold (filter helps iff handedness accuracy > top-1 / oracle top-1) and simulation with
   synthetic calibrated handedness evidence of accuracy 0.50-1.00.
7. Prevalence: galleries of the 21 right-handed and 1-3 left-handed writers.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import expit, log_softmax, logsumexp
from scipy.stats import binomtest, norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ID_REPS = ["stroke", "hc183", "resnet18", "efficientnet_b0", "vit_b_16", "stroke_cnn"]
HAND_MODELS = {"hc183": "hc183", "hc183_adj": "hc183", "stroke": "stroke"}
NS = (1, 2, 4, 8)
OWN_K = (1, 2, 3, 6)
T_GRID = np.logspace(-3, 0, 31)
LAM_GRID = np.array([0, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5])
PRIMARY = [("own", 1, 1, "resnet18"), ("own", 1, 1, "vit_b_16"), ("own", 4, 6, "resnet18"), ("own", 4, 6, "vit_b_16")]
SEED = 2031


# ---------------------------------------------------------------- 1. handedness evidence (leave one writer out)
def residualise(Xtr, Ctr, X, C):
    A = np.column_stack([np.ones(len(Ctr)), Ctr]); B = np.linalg.lstsq(A, Xtr, rcond=None)[0]
    return X - np.column_stack([np.ones(len(C)), C]) @ B


def fit_score(Xtr, ytr, Ctr, Xte, Cte, adjust):
    if adjust:
        Xtr, Xte = residualise(Xtr, Ctr, Xtr, Ctr), residualise(Xtr, Ctr, Xte, Cte)
    m = make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.1)).fit(Xtr, ytr)
    return m.decision_function(Xte)


def lowo_writer(X, y, C, g, train_mask, test_idx, adjust, seed):
    """Calibrated left-handed log-odds of the held-out writer's units."""
    tr = np.flatnonzero(train_mask)
    raw = fit_score(X[tr], y[tr], C[tr], X[test_idx], C[test_idx], adjust)
    oof = np.empty(len(tr))
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(tr, y[tr], g[tr]):
        oof[b] = fit_score(X[tr[a]], y[tr[a]], C[tr[a]], X[tr[b]], C[tr[b]], adjust)
    platt = LogisticRegression(C=1e4).fit(oof[:, None], y[tr])
    return test_idx, raw, platt.coef_[0, 0] * raw + platt.intercept_[0]


def handedness_scores(U, F, jobs):
    out = []
    C = np.column_stack([U.ink_density.to_numpy(), F["global_intensity"]])
    for hm, rep in HAND_MODELS.items():
        X = F[rep]
        for n in NS:
            un = (U.n == n) & U.valid
            res = Parallel(n_jobs=jobs)(
                delayed(lowo_writer)(X, U.label.to_numpy(), C, U.writer_id.to_numpy(),
                                     (un & (U.para <= 6) & (U.writer_id != w)).to_numpy(),
                                     np.flatnonzero((un & (U.writer_id == w)).to_numpy()), hm == "hc183_adj", SEED)
                for w in sorted(U.writer_id.unique()))
            for idx, raw, logit in res:
                out.append(pd.DataFrame(dict(unit_id=U.unit_id.to_numpy()[idx], hand_model=hm, raw=raw, logit=logit)))
            print(f"  handedness {hm} n={n} done", flush=True)
    H = pd.concat(out, ignore_index=True)
    return H


# ---------------------------------------------------------------- 2. identification cells
def normalised(F, U, rep):
    X = F[rep].astype(float)
    ref = (U.n == 1).to_numpy() & (U.para <= 6).to_numpy()
    Z = StandardScaler().fit(X[ref]).transform(X)
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)


def cells():
    for n in NS:
        yield dict(query="shared", n=n, k=5)
        for k in OWN_K:
            yield dict(query="own", n=n, k=k)


def cell_queries(U, writers, cell, Z, full_idx):
    """Yields (query unit indices, similarity matrix) for every reference configuration of the cell."""
    q_all = U.index[(U.n == cell["n"]) & U.valid & ((U.para <= 6) if cell["query"] == "shared" else (U.para >= 7))]
    if cell["query"] == "shared":
        configs = [(p, tuple(x for x in range(1, 7) if x != p)) for p in range(1, 7)]
    else:
        configs = [(None, s) for s in itertools.combinations(range(1, 7), cell["k"])]
    for c, (p, refs) in enumerate(configs):
        q = q_all[U.para[q_all] == p] if p is not None else q_all
        T = np.stack([Z[[full_idx[(w, r)] for r in refs]].mean(0) for w in writers])
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        yield c, refs, q.to_numpy(), Z[q.to_numpy()] @ T.T


def ranks(score, t):
    return 1 + (score > score[np.arange(len(t)), t][:, None]).sum(1)


def fit_by_writer(nll, qw):
    """nll: (Q, G) per-query loss on a parameter grid; returns for every query the grid index fitted on other writers."""
    tot = nll.sum(0); best = np.empty(len(qw), int)
    for w in np.unique(qw):
        m = qw == w
        best[m] = np.argmin(tot - nll[m].sum(0))
    return best


def match_logprob(ell_q, gal, ghand):
    """log P(query hand = hand of gallery writer | evidence). gal: None (gallery known) or gallery log-odds (Q, W)."""
    lq = np.stack([np.log(expit(-ell_q)), np.log(expit(ell_q))], 1)            # (Q, 2): right, left
    if gal is None:
        return lq[:, ghand]
    lg = np.stack([np.log(expit(-gal)), np.log(expit(gal))], 2)                # (Q, W, 2)
    return logsumexp(lq[:, None, :] + lg, axis=2)


def evaluate_cell(S, t, qw, qhand, ghand, M, pred_q, pred_g):
    """Ranks of the true writer for every fusion rule. M: (Q, W) log match probability; pred_q, pred_g: predicted hands."""
    out = {"none": ranks(S, t)}
    ll = np.stack([log_softmax(S / T, 1)[np.arange(len(t)), t] for T in T_GRID], 1)
    iT = fit_by_writer(-ll, qw); Tq = T_GRID[iT][:, None]
    post = log_softmax(S / Tq + M, 1)
    out["bayes"] = ranks(post, t)
    nll = np.stack([-log_softmax(S / T + lam * M, 1)[np.arange(len(t)), t] for T in T_GRID for lam in LAM_GRID], 1)
    ig = fit_by_writer(nll, qw)
    Tl, lam = T_GRID[ig // len(LAM_GRID)][:, None], LAM_GRID[ig % len(LAM_GRID)][:, None]
    out["loglin"] = ranks(S / Tl + lam * M, t)
    same = (pred_g == pred_q[:, None]) if pred_g.ndim == 2 else (ghand[None, :] == pred_q[:, None])
    out["filter"] = ranks(S + 10.0 * same, t)                                 # writers of the predicted hand first
    top5 = np.argsort(-S, 1)[:, :5]
    rr = np.full_like(S, -np.inf); np.put_along_axis(rr, top5, np.take_along_axis(post, top5, 1), 1)
    rr = np.where(np.isfinite(rr), rr + 100, S)                                # top 5 reordered by bayes, rest by S
    out["rerank5"] = ranks(rr, t)
    out["_T"] = Tq[:, 0]; out["_lam"] = lam[:, 0]
    return out


def run_identification(U, F, H, writers):
    widx = {w: i for i, w in enumerate(writers)}
    ghand = np.array([1 if w.startswith("L") else 0 for w in writers])
    full_idx = {(w, p): i for i, (w, p, n) in enumerate(zip(U.writer_id, U.para, U.n)) if n == 1}
    Hp = {hm: H[H.hand_model == hm].set_index("unit_id").logit for hm in HAND_MODELS}
    rows, err_rows = [], []
    for rep in ID_REPS:
        Z = normalised(F, U, rep)
        for cell in cells():
            parts = list(cell_queries(U, writers, cell, Z, full_idx))
            q = np.concatenate([p[2] for p in parts]); S = np.concatenate([p[3] for p in parts])
            conf = np.concatenate([np.full(len(p[2]), p[0]) for p in parts])
            refsets = [p[1] for p in parts]
            t = np.array([widx[w] for w in U.writer_id.to_numpy()[q]]); qw = U.writer_id.to_numpy()[q]
            qhand = U.label.to_numpy()[q]
            base = ranks(S, t)
            wrong = base > 1
            pred_w = S.argmax(1)
            err_rows.append(dict(rep=rep, **cell, n_queries=len(t), errors=int(wrong.sum()),
                                 cross_hand_errors=int((ghand[pred_w[wrong]] != qhand[wrong]).sum())))
            # oracle: true hands on both sides
            orc = ranks(S + 10.0 * (ghand[None, :] == qhand[:, None]), t)
            rows.append(_rows(rep, cell, "oracle", "-", "filter", orc, q, qw, conf, U))
            rows.append(_rows(rep, cell, "none", "-", "none", base, q, qw, conf, U))
            for hm in HAND_MODELS:
                evid = {"unit": Hp[hm].reindex(U.unit_id.to_numpy()[q]).to_numpy()}
                if cell["query"] == "own":                                     # all own-text units of the writer
                    own = U[(U.n == cell["n"]) & U.valid & (U.para >= 7)]
                    wmean = pd.Series(Hp[hm].reindex(own.unit_id).to_numpy(), index=own.writer_id).groupby(level=0).mean()
                    evid["writer"] = wmean.reindex(qw).to_numpy()
                for ev, ell in evid.items():
                    hand_tag = hm if ev == "unit" else hm + "_writerlevel"
                    # gallery hand predicted from the references of each configuration (whole shared paragraphs)
                    gl = np.empty_like(S)
                    for c, refs in enumerate(refsets):
                        lg = np.array([np.mean([Hp[hm].get(f"{w}_para{r:02d}_n1_s0", np.nan) for r in refs]) for w in writers])
                        gl[conf == c] = lg
                    for scen, gal in [("query", None), ("both", gl)]:
                        M = match_logprob(ell, gal, ghand)
                        pq = (ell > 0).astype(int)
                        pg = (gl > 0).astype(int) if scen == "both" else ghand
                        res = evaluate_cell(S, t, qw, qhand, ghand, M, pq, pg)
                        for meth in ["bayes", "loglin", "filter", "rerank5"]:
                            r = _rows(rep, cell, scen, hand_tag, meth, res[meth], q, qw, conf, U)
                            r["hand_correct"] = (pq == qhand).astype(float)
                            r["T"] = res["_T"]; r["lam"] = res["_lam"]
                            rows.append(r)
            print(f"  id {rep} {cell} queries={len(t)} top1={np.mean(base == 1):.3f}", flush=True)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(err_rows)


def _rows(rep, cell, scen, hand, meth, rk, q, qw, conf, U):
    return pd.DataFrame(dict(rep=rep, query=cell["query"], n=cell["n"], k=cell["k"], scenario=scen, hand_model=hand,
                             method=meth, unit_id=U.unit_id.to_numpy()[q], writer_id=qw, config=conf, rank=rk))


# ---------------------------------------------------------------- 3. summaries and statistics
def summarise(R, W):
    g = R.groupby(["rep", "query", "n", "k", "scenario", "hand_model", "method"])
    s = g.agg(top1=("rank", lambda r: np.mean(r == 1)), top5=("rank", lambda r: np.mean(r <= 5)),
              mrr=("rank", lambda r: np.mean(1 / r)), mean_pen=("rank", lambda r: np.mean(r) / W),
              pen95=("rank", lambda r: np.quantile(r, 0.95, method="higher") / W),
              hand_acc=("hand_correct", "mean"), n_rows=("rank", "size")).reset_index()
    return s


def writer_diffs(R, base_key, key):
    a = R[base_key].groupby("writer_id")["rank"].apply(lambda r: np.mean(r == 1))
    b = R[key].groupby("writer_id")["rank"].apply(lambda r: np.mean(r == 1))
    return (b - a).dropna()


def paired_tests(R, n_perm, n_boot):
    rng = np.random.default_rng(SEED)
    out = []
    keys = R[["rep", "query", "n", "k"]].drop_duplicates().itertuples(index=False)
    for rep, qy, n, k in keys:
        cellm = (R.rep == rep) & (R["query"] == qy) & (R.n == n) & (R.k == k)
        C = R[cellm]
        base = C[C.method == "none"].sort_values(["unit_id", "config"])
        for (scen, hm, meth), D in C[C.method != "none"].groupby(["scenario", "hand_model", "method"]):
            D = D.sort_values(["unit_id", "config"])
            d = (D.groupby("writer_id")["rank"].apply(lambda r: np.mean(r == 1))
                 - base.groupby("writer_id")["rank"].apply(lambda r: np.mean(r == 1)))
            obs = d.mean()
            flips = rng.choice([-1, 1], size=(n_perm, len(d)))
            p_perm = (1 + np.sum(np.abs(flips @ d.to_numpy() / len(d)) >= abs(obs) - 1e-12)) / (n_perm + 1)
            hand = np.array([w.startswith("L") for w in d.index])
            dv = d.to_numpy()
            boots = np.concatenate([dv[np.flatnonzero(hand == h)][rng.integers(0, (hand == h).sum(), (n_boot, (hand == h).sum()))]
                                    for h in (0, 1)], axis=1).mean(1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            p_mc = np.nan
            if D.config.nunique() == 1:
                x, y = base["rank"].to_numpy() == 1, D["rank"].to_numpy() == 1
                b01, b10 = int((~x & y).sum()), int((x & ~y).sum())
                p_mc = binomtest(b01, b01 + b10).pvalue if b01 + b10 else 1.0
            out.append(dict(rep=rep, query=qy, n=n, k=k, scenario=scen, hand_model=hm, method=meth,
                            diff_top1=obs, ci_lo=lo, ci_hi=hi, p_signflip=p_perm, p_mcnemar=p_mc))
    T = pd.DataFrame(out)
    T["primary"] = False
    for qy, n, k, rep in PRIMARY:
        T.loc[(T["query"] == qy) & (T.n == n) & (T.k == k) & (T.rep == rep) & (T.scenario == "query")
              & (T.hand_model == "hc183") & (T.method == "bayes"), "primary"] = True
    P = T[T.primary].sort_values("p_signflip")
    m = len(P); adj = np.maximum.accumulate([min(1, (m - i) * p) for i, p in enumerate(P.p_signflip)])
    T["p_holm"] = np.nan; T.loc[P.index, "p_holm"] = adj
    return T


# ---------------------------------------------------------------- 4. break-even simulation
def simulate(U, F, writers, cells_sim, reps, accs, n_sim):
    rng = np.random.default_rng(SEED + 1)
    widx = {w: i for i, w in enumerate(writers)}
    ghand = np.array([1 if w.startswith("L") else 0 for w in writers])
    full_idx = {(w, p): i for i, (w, p, n) in enumerate(zip(U.writer_id, U.para, U.n)) if n == 1}
    out = []
    for rep in reps:
        Z = normalised(F, U, rep)
        for cell in cells_sim:
            parts = list(cell_queries(U, writers, cell, Z, full_idx))
            q = np.concatenate([p[2] for p in parts]); S = np.concatenate([p[3] for p in parts])
            t = np.array([widx[w] for w in U.writer_id.to_numpy()[q]]); qw = U.writer_id.to_numpy()[q]
            qhand = U.label.to_numpy()[q]
            ll = np.stack([log_softmax(S / T, 1)[np.arange(len(t)), t] for T in T_GRID], 1)
            Tq = T_GRID[fit_by_writer(-ll, qw)][:, None]
            base = np.mean(ranks(S, t) == 1)
            orc = np.mean(ranks(S + 10.0 * (ghand[None] == qhand[:, None]), t) == 1)
            for a in accs:
                d = 2 * norm.ppf(min(a, 0.999))
                for s in range(n_sim):
                    z = (2 * qhand - 1) * d / 2 + rng.standard_normal(len(t))
                    ell = d * z
                    M = match_logprob(ell, None, ghand)
                    pq = (ell > 0).astype(int)
                    out.append(dict(rep=rep, **cell, acc=a, sim=s, realised_acc=np.mean(pq == qhand), base=base, oracle=orc,
                                    bayes=np.mean(ranks(log_softmax(S / Tq + M, 1), t) == 1),
                                    filter=np.mean(ranks(S + 10.0 * (ghand[None] == pq[:, None]), t) == 1)))
            print(f"  simulation {rep} {cell} done", flush=True)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 5. prevalence
def prevalence(U, F, H, writers, cells_prev, reps, n_left, n_draws):
    rng = np.random.default_rng(SEED + 2)
    widx = {w: i for i, w in enumerate(writers)}
    ghand = np.array([1 if w.startswith("L") else 0 for w in writers])
    full_idx = {(w, p): i for i, (w, p, n) in enumerate(zip(U.writer_id, U.para, U.n)) if n == 1}
    Hp = H[H.hand_model == "hc183"].set_index("unit_id").logit
    L, R = np.flatnonzero(ghand == 1), np.flatnonzero(ghand == 0)
    out = []
    for rep in reps:
        Z = normalised(F, U, rep)
        for cell in cells_prev:
            parts = list(cell_queries(U, writers, cell, Z, full_idx))
            q = np.concatenate([p[2] for p in parts]); S = np.concatenate([p[3] for p in parts])
            t = np.array([widx[w] for w in U.writer_id.to_numpy()[q]]); qw = U.writer_id.to_numpy()[q]
            ell = Hp.reindex(U.unit_id.to_numpy()[q]).to_numpy()
            ll = np.stack([log_softmax(S / T, 1)[np.arange(len(t)), t] for T in T_GRID], 1)
            Tq = T_GRID[fit_by_writer(-ll, qw)][:, None]
            for m in n_left:
                prior = np.log(m / len(R))
                for d in range(n_draws):
                    G = np.concatenate([R, rng.choice(L, m, replace=False)])
                    keep = np.isin(t, G)
                    pos = np.full(len(writers), -1); pos[G] = np.arange(len(G))
                    Sg, tg = S[keep][:, G], pos[t[keep]]
                    gh, e, qh = ghand[G], ell[keep], ghand[t[keep]]
                    M = match_logprob(e, None, gh)
                    res = {"none": ranks(Sg, tg), "bayes": ranks(log_softmax(Sg / Tq[keep] + M, 1), tg),
                           "filter": ranks(Sg + 10.0 * (gh[None] == (e > 0)[:, None]), tg),
                           "filter_prior": ranks(Sg + 10.0 * (gh[None] == (e + prior > 0)[:, None]), tg)}
                    for meth, rk in res.items():
                        for h in (0, 1):
                            sel = qh == h
                            out.append(dict(rep=rep, **cell, n_left=m, draw=d, method=meth, query_hand="left" if h else "right",
                                            gallery=len(G), top1=np.mean(rk[sel] == 1), mean_pen=np.mean(rk[sel]) / len(G),
                                            n_queries=int(sel.sum())))
            print(f"  prevalence {rep} {cell} done", flush=True)
    P = pd.DataFrame(out)
    return P.groupby(["rep", "query", "n", "k", "n_left", "method", "query_hand"]).agg(
        top1=("top1", "mean"), mean_pen=("mean_pen", "mean"), gallery=("gallery", "first"),
        n_queries=("n_queries", "mean"), draws=("draw", "size")).reset_index()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="folder with unit_features.npz and units.csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--quick", action="store_true", help="few permutations, bootstraps and simulations (test run)")
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()
    fin, out = Path(a.features), Path(a.output); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    U = pd.read_csv(fin / "units.csv")
    z = np.load(fin / "unit_features.npz", allow_pickle=True)
    assert (z["unit_id"] == U.unit_id.to_numpy()).all()
    F = {k: z[k] for k in z.files if k != "unit_id"}
    writers = sorted(U.writer_id.unique())
    n_perm, n_boot, n_sim, n_draw = (199, 199, 5, 20) if a.quick else (9999, 2000, 200, 1000)
    print(f"{len(U)} units, {len(writers)} writers, {int((~U.valid).sum())} invalid units", flush=True)

    hpath = out / "handedness_lowo.csv.gz"
    if hpath.exists():
        H = pd.read_csv(hpath)
    else:
        H = handedness_scores(U, F, a.jobs); H.to_csv(hpath, index=False)
    Hm = H.merge(U[["unit_id", "writer_id", "label", "para", "n"]], on="unit_id")
    hand_acc = (Hm.assign(correct=(Hm.logit > 0) == (Hm.label == 1), part=np.where(Hm.para <= 6, "shared", "own"))
                .groupby(["hand_model", "n", "part"]).correct.mean().unstack().reset_index())
    wl = (Hm.groupby(["hand_model", "n", "writer_id", "label"]).logit.mean().reset_index()
          .assign(correct=lambda d: (d.logit > 0) == (d.label == 1)).groupby(["hand_model", "n"]).correct.mean())
    hand_acc = hand_acc.merge(wl.rename("writer_all_units").reset_index(), on=["hand_model", "n"])
    hand_acc.to_csv(out / "handedness_accuracy.csv", index=False)
    print(hand_acc.round(3).to_string(index=False), flush=True)

    R, E = run_identification(U, F, H, writers)
    R.to_csv(out / "id_ranks.csv.gz", index=False)
    E.to_csv(out / "id_errors_cross_hand.csv", index=False)
    S = summarise(R, len(writers))
    S.to_csv(out / "id_summary.csv", index=False)
    T = paired_tests(R[(R.method == "none") | (R.hand_model.isin(["hc183", "-"]))], n_perm, n_boot)
    T.to_csv(out / "id_paired_tests.csv", index=False)

    # break-even: analytic threshold per cell (hard filter) and simulation
    base = S[S.method == "none"].set_index(["rep", "query", "n", "k"]).top1
    orc = S[S.scenario == "oracle"].set_index(["rep", "query", "n", "k"]).top1
    act = S[(S.scenario == "query") & (S.hand_model == "hc183") & (S.method == "bayes")].set_index(["rep", "query", "n", "k"]).hand_acc
    pd.DataFrame(dict(base=base, oracle=orc, threshold=base / orc, hand_acc=act)).reset_index().to_csv(
        out / "breakeven_analytic.csv", index=False)
    sim_cells = [dict(query="own", n=1, k=1), dict(query="own", n=4, k=6), dict(query="own", n=1, k=6),
                 dict(query="shared", n=4, k=5)]
    accs = np.round(np.arange(0.5, 1.0001, 0.05), 2)
    simulate(U, F, writers, sim_cells, ID_REPS, accs, n_sim).to_csv(out / "breakeven_simulation.csv.gz", index=False)
    prevalence(U, F, H, writers, [dict(query="own", n=1, k=1), dict(query="own", n=4, k=6)], ID_REPS,
               (1, 2, 3), n_draw).to_csv(out / "prevalence.csv", index=False)
    (out / "wid_provenance.json").write_text(json.dumps(dict(
        quick=a.quick, n_perm=n_perm, n_boot=n_boot, n_sim=n_sim, n_draws=n_draw, seed=SEED, primary=PRIMARY,
        id_reps=ID_REPS, hand_models=list(HAND_MODELS), T_grid=list(T_GRID), lam_grid=list(LAM_GRID),
        seconds=round(time.time() - t0)), indent=2))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
