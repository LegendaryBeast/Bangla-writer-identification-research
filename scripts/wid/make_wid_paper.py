#!/usr/bin/env python3
"""Figures, tables and macros of the writer identification manuscript (manuscripts/wid_handedness/).

Reads the outputs of run_wid_handedness.py, run_both_control.py and run_wid_extras.py and writes
  figures/wid_macros.tex     every number quoted in the text
  figures/table_main.tex, figures/table_tests.tex
  figures/fig0_method.pdf, fig1_design.pdf, fig2_utility.pdf, fig3_leakage.pdf (and .png)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import wid_features as wf  # noqa: E402

NAME = {"stroke": "STROKE", "hc183": "HC-183", "resnet18": "ResNet-18", "efficientnet_b0": "EffNet-B0",
        "vit_b_16": "ViT-B/16", "stroke_cnn": "Stroke-CNN"}
RCOL = {"hc183": "#111827", "stroke": "#c2410c", "resnet18": "#9ca3af", "efficientnet_b0": "#60a5fa",
        "vit_b_16": "#a78bfa", "stroke_cnn": "#f59e0b"}
REPS = list(NAME)
KEY = ["rep", "query", "n", "k"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=250); plt.close(fig)


def pct(v, d=1):
    return f"{100 * v:.{d}f}"


def signed(v, d=1):
    s = f"{100 * v:+.{d}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def panel(ax, letter):
    ax.text(-0.13, 1.04, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def load(run):
    d = dict(S=pd.read_csv(run / "id_summary.csv"), T=pd.read_csv(run / "id_paired_tests.csv"),
             E=pd.read_csv(run / "id_errors_cross_hand.csv"), HA=pd.read_csv(run / "handedness_accuracy.csv"),
             BA=pd.read_csv(run / "breakeven_analytic.csv"), SIM=pd.read_csv(run / "breakeven_simulation.csv.gz"),
             PREV=pd.read_csv(run / "prevalence.csv"), BC=pd.read_csv(run / "both_control.csv"),
             BCS=pd.read_csv(run / "both_control_summary.csv"), LA=pd.read_csv(run / "leakage_accuracy.csv"),
             LI=pd.read_csv(run / "leakage_identification.csv"), G=pd.read_csv(run / "confidence_gating.csv"),
             LAW=pd.read_csv(run / "filter_law.csv"), U=pd.read_csv(run / "features/units.csv"),
             X=json.loads((run / "extras_provenance.json").read_text()))
    return d


def top1(S, scen, hm, meth):
    return S[(S.scenario == scen) & (S.hand_model == hm) & (S.method == meth)].set_index(KEY).top1


def sim_gain(SIM, cell):
    g = SIM[(SIM["query"] == cell[0]) & (SIM.n == cell[1]) & (SIM.k == cell[2])].groupby(["rep", "acc"])[
        ["base", "bayes", "filter"]].mean()
    return g.assign(gain_bayes=g.bayes - g.base, gain_filter=g["filter"] - g.base).reset_index()


def macros(d):
    S, T, U = d["S"], d["T"], d["U"]
    m = {}
    m["nWriters"] = U.writer_id.nunique(); m["nLeft"] = U[U.label == 1].writer_id.nunique()
    m["nRight"] = U[U.label == 0].writer_id.nunique(); m["nImages"] = int((U.n == 1).sum())
    m["nUnits"] = len(U); m["nInvalid"] = int((~U.valid).sum())
    m["nSettings"] = len(S[S.method == "none"])
    ha = d["HA"].set_index(["hand_model", "n"])
    m["HandShared"] = pct(ha.loc[("hc183", 1), "shared"]); m["HandOwn"] = pct(ha.loc[("hc183", 1), "own"])
    m["HandWriter"] = pct(ha.loc[("hc183", 1), "writer_all_units"])
    strips = ha.loc[[("hc183", n) for n in (2, 4, 8)], ["own", "shared"]].to_numpy()
    m["HandStripMin"], m["HandStripMax"] = pct(strips.min()), pct(strips.max())
    m["HandAdjOwn"] = pct(ha.loc[("hc183_adj", 1), "own"]); m["HandStrokeOwn"] = pct(ha.loc[("stroke", 1), "own"])
    none = top1(S, "none", "-", "none")
    m["StrokeEightOwn"] = pct(none.loc[("stroke", "own", 8, 1)]); m["ScnnFour"] = pct(none.loc[("stroke_cnn", "own", 4, 6)])
    E = d["E"].groupby("rep")[["cross_hand_errors", "errors"]].sum()
    share = E.cross_hand_errors / E.errors
    m["CrossMin"], m["CrossMax"] = pct(share.min(), 0), pct(share.max(), 0)
    m["CrossChance"] = pct((m["nWriters"] // 2) / (m["nWriters"] - 1))

    def tsel(scen, meth, hm="hc183"):
        return T[(T.scenario == scen) & (T.method == meth) & (T.hand_model == (hm if scen != "oracle" else "-"))]
    o = tsel("oracle", "filter")
    m["OracleGain"] = pct(o.diff_top1.mean()); m["OracleMax"] = pct(o.diff_top1.max())
    m["OracleSig"] = int(((o.p_signflip < .05) & (o.diff_top1 > 0)).sum())
    b = tsel("query", "bayes")
    m["BayesGain"] = signed(b.diff_top1.mean())
    m["BayesSigUp"] = int(((b.p_signflip < .05) & (b.diff_top1 > 0)).sum())
    m["BayesSigDown"] = int(((b.p_signflip < .05) & (b.diff_top1 < 0)).sum())
    m["LoglinGain"] = signed(tsel("query", "loglin").diff_top1.mean())
    f = tsel("query", "filter")
    m["FilterGain"] = signed(f.diff_top1.mean()); m["FilterLoss"] = pct(-f.diff_top1.mean())
    m["FilterSigDown"] = int(((f.p_signflip < .05) & (f.diff_top1 < 0)).sum())
    pr = T[T.primary]
    m["PrimDiffMin"], m["PrimDiffMax"] = signed(pr.diff_top1.min()), signed(pr.diff_top1.max())
    m["PrimPmin"] = f"{pr.p_signflip.min():.2f}"
    law = d["X"]["filter_law"]
    m["LawR"] = f"{law['r']:.3f}"; m["LawMae"] = pct(law["mae"]); m["LawBias"] = signed(law["bias"]); m["LawN"] = law["n_points"]
    ba = d["BA"][d["BA"]["query"] == "own"].threshold
    m["ThreshMed"], m["ThreshMin"], m["ThreshMax"] = pct(ba.median(), 0), pct(ba.min(), 0), pct(ba.max(), 0)
    sg = sim_gain(d["SIM"], ("own", 1, 1))
    first = sg[sg.gain_bayes >= 0.005].groupby("rep").acc.min()
    m["SimAccMin"], m["SimAccMax"] = pct(first.min(), 0), pct(first.max(), 0)
    mean_gain = sg.groupby("acc").gain_bayes.mean()
    m["SimAccOne"] = pct(mean_gain[mean_gain >= 0.01].index.min(), 0)
    allsim = pd.concat([sim_gain(d["SIM"], c).assign(cell=str(c)) for c in
                        d["SIM"][["query", "n", "k"]].drop_duplicates().itertuples(index=False, name=None)])
    hi = allsim[np.isclose(allsim.acc, 0.95)].gain_bayes
    m["SimGainHiMin"], m["SimGainHiMax"] = pct(hi.min()), pct(hi.max())
    at = allsim[np.isclose(allsim.acc, 0.65)].gain_bayes
    m["SimGainObsMax"] = pct(at.max())
    lw = d["LAW"][d["LAW"].hand_model == "hc183"].dropna(subset=["bayes_sim_same_acc"])
    m["BayesGap"] = pct((lw.bayes_sim_same_acc - lw.bayes).mean())
    g = d["G"][np.isclose(d["G"].coverage, 0.3)]
    m["GateAcc"] = pct(g.acc_gated.mean(), 0)
    gg = g.filter_gated - g.base
    m["GateMin"], m["GateMax"] = signed(gg.min()), signed(gg.max())
    P = d["PREV"]
    pv = P[(P.rep == "vit_b_16") & (P["query"] == "own") & (P.n == 1) & (P.k == 1)]
    m["PrevLeftNone"] = pct(pv[(pv.method == "none") & (pv.query_hand == "left") & (pv.n_left == 1)].top1.iloc[0])
    fp = pv[(pv.method == "filter_prior") & (pv.query_hand == "left")].top1
    m["PrevLeftPriorMin"], m["PrevLeftPriorMax"] = pct(fp.min()), pct(fp.max())
    w = P.pivot_table(index=KEY + ["n_left", "query_hand"], columns="method", values="top1")
    dl = (w.bayes - w["none"]).xs("left", level="query_hand"); dr = (w.bayes - w["none"]).xs("right", level="query_hand")
    m["PrevBayesLeftMean"], m["PrevBayesLeftMax"] = pct(dl.mean()), pct(dl.max())
    m["PrevBayesRightLoss"] = pct(-dr.min())
    LA = d["LA"][(d["LA"].n == 1)]

    def la(label, regime, part):
        return LA[(LA.label == label) & (LA.regime == regime) & (LA.part == part)].acc
    m["LeakHandWI"] = pct(la("hand", "writer-independent", "own").iloc[0])
    m["LeakHandEn"] = pct(la("hand", "enrolment-trained", "own").iloc[0])
    m["LeakHandEnShared"] = pct(la("hand", "enrolment-trained", "shared").iloc[0])
    m["LeakPseudoWI"] = pct(la("pseudo", "writer-independent", "own").mean())
    pe = la("pseudo", "enrolment-trained", "own")
    m["LeakPseudoEnMin"], m["LeakPseudoEnMax"] = pct(pe.min()), pct(pe.max())
    m["LeakPseudoEnShared"] = pct(la("pseudo", "enrolment-trained", "shared").mean())
    m["LeakPseudoEnMean"] = pct(pe.mean()); m["LeakPseudoAbove"] = pct(pe.mean() - 0.5)
    m["LeakHandGain"] = pct(la("hand", "enrolment-trained", "own").iloc[0] - la("hand", "writer-independent", "own").iloc[0])
    LI = d["LI"]
    m["LeakGainHandEn"] = signed(LI[(LI.label == "hand") & (LI.regime == "enrolment-trained")].gain_bayes.mean())
    m["LeakGainPseudoEn"] = signed(LI[(LI.label == "pseudo") & (LI.regime == "enrolment-trained")].gain_bayes.mean())
    bcs = d["BCS"].set_index("control")
    m["BothGain"] = pct(bcs.hand_mean_gain.iloc[0], 2)
    m["BothPseudo"] = signed(bcs.loc["pseudo", "control_mean_gain"], 2)
    m["BothPseudoScaled"] = signed(bcs.loc["pseudo_scaled", "control_mean_gain"], 2)
    m["BothPseudoScaledP"] = f"{bcs.loc['pseudo_scaled', 'p_ge_hand']:.3f}"
    m["BothRandom"] = signed(bcs.loc["random", "control_mean_gain"], 2)
    m["BothRandomP"] = f"{bcs.loc['random', 'p_ge_hand']:.3f}"
    bt = tsel("both", "bayes")
    m["BothSig"] = int(((bt.p_signflip < .05) & (bt.diff_top1 > 0)).sum())
    return m


def tables(d, out):
    S, T = d["S"], d["T"]
    none, orc = top1(S, "none", "-", "none"), top1(S, "oracle", "-", "filter")
    bay, fil = top1(S, "query", "hc183", "bayes"), top1(S, "query", "hc183", "filter")
    cells = [("own", 1, 1), ("own", 4, 6), ("shared", 4, 5)]
    rows = []
    for rep in REPS:
        v = [pct(s.loc[(rep, *c)]) for c in cells for s in (none, orc, bay, fil)]
        rows.append(f"{NAME[rep]} & " + " & ".join(v) + r" \\")
    (out / "table_main.tex").write_text("% generated by scripts/wid/make_wid_paper.py\n" + "\n".join(rows) + "\n\\bottomrule\n")
    lab = {(1, 1): "Own paragraph, $k=1$", (4, 6): "Own strip ($n=4$), $k=6$"}
    rows = []
    for r in T[T.primary].sort_values(["n", "rep"]).itertuples():
        k = (r.rep, r.query, r.n, r.k)
        rows.append(f"{lab[(r.n, r.k)]} & {NAME[r.rep]} & {pct(none.loc[k])} & {pct(bay.loc[k])} & "
                    f"{signed(r.diff_top1)} [{signed(r.ci_lo)}, {signed(r.ci_hi)}] & {r.p_signflip:.2f} & {r.p_holm:.2f} \\\\")
    ha = d["HA"].set_index(["hand_model", "n"])
    hand_rows = []
    for hm, lab in [("hc183", "HC-183"), ("hc183_adj", "HC-183 adj."), ("stroke", "STROKE")]:
        cells = [f"{pct(ha.loc[(hm, n), 'shared'])} / {pct(ha.loc[(hm, n), 'own'])}" for n in (1, 2, 4, 8)]
        hand_rows.append(f"{lab} & " + " & ".join(cells) + f" & {pct(ha.loc[(hm, 1), 'writer_all_units'])} \\\\")
    (out / "table_hand.tex").write_text("% generated by scripts/wid/make_wid_paper.py\n" + "\n".join(hand_rows) + "\n\\bottomrule\n")
    (out / "table_tests.tex").write_text("% generated by scripts/wid/make_wid_paper.py\n" + "\n".join(rows) + "\n\\bottomrule\n")


def example_query(run, query_unit="L_p16_para08_n1_s0", rep="vit_b_16"):
    """Scores of one real own-text query (six references) before and after Bayes fusion, for the method figure."""
    import run_wid_handedness as wh
    from scipy.special import expit, log_softmax
    U = pd.read_csv(run / "features/units.csv")
    z = np.load(run / "features/unit_features.npz", allow_pickle=True)
    F = {k: z[k] for k in z.files if k != "unit_id"}
    writers = sorted(U.writer_id.unique()); gh = np.array([w[0] == "L" for w in writers]).astype(int)
    full = {(w, p): i for i, (w, p, n) in enumerate(zip(U.writer_id, U.para, U.n)) if n == 1}
    H = pd.read_csv(run / "handedness_lowo.csv.gz"); Hp = H[H.hand_model == "hc183"].set_index("unit_id").logit
    _, _, q, S = next(wh.cell_queries(U, writers, dict(query="own", n=1, k=6), wh.normalised(F, U, rep), full))
    ids = U.unit_id.to_numpy()[q]; t = np.array([writers.index(w) for w in U.writer_id.to_numpy()[q]])
    ell = Hp.reindex(ids).to_numpy()
    ll = np.stack([log_softmax(S / T, 1)[np.arange(len(t)), t] for T in wh.T_GRID], 1)
    T = wh.T_GRID[np.argmax(ll.sum(0))]
    i = int(np.flatnonzero(ids == query_unit)[0])
    before = np.exp(log_softmax(S[i] / T)); after = np.exp(log_softmax(S[i] / T + wh.match_logprob(ell[i:i + 1], None, gh)[0]))
    return dict(before=before, after=after, hand=gh, true=t[i], p_left=float(expit(ell[i])))


def fig_method(out, dataset, run):
    """Procedure as a visual pipeline built on one real query: query and gallery, two encoders, scores and handedness
    evidence, fusion, final ranking. Drawn in one equal-aspect coordinate system so that shapes stay true."""
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
    ex = example_query(run)
    LC, RC, PC, IDC, INK, MUTE = "#c2410c", "#1d4ed8", "#6d28d9", "#1e3a8a", "#374151", "#6b7280"
    fig = plt.figure(figsize=(7.16, 2.506))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 35.0); ax.set_aspect("equal"); ax.axis("off")

    def text(x, y, t, **kw):
        kw.setdefault("ha", "center"); kw.setdefault("va", "center"); kw.setdefault("fontsize", 7); kw.setdefault("color", INK)
        ax.text(x, y, t, **kw)

    def arrow(pts, col=INK, lw=1.0):
        for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
            ax.plot([x0, x1], [y0, y1], color=col, lw=lw, solid_capstyle="butt")
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=8, lw=lw, color=col, shrinkA=0, shrinkB=0))

    def page(path, x, y, w, frame="#9ca3af", strip=None):
        g = wf.load_gray(path); ink = wf.stroke_ink_uncropped(g); r0, r1 = wf.text_rows(ink); H = ink.shape[0]
        a = np.asarray(g.resize((700, round(700 * g.height / g.width))), float)
        lo, hi = max(int(r0 / H * a.shape[0]) - 8, 0), int(r1 / H * a.shape[0]) + 8
        a = a[lo:hi]; bg = np.percentile(a, 90); a = np.clip((a - (bg - 110)) / 110, 0, 1)
        h = w * a.shape[0] / a.shape[1]
        ax.imshow(a, cmap="gray", vmin=0, vmax=1, extent=(x, x + w, y, y + h), interpolation="antialiased", zorder=2)
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=frame, lw=0.7, zorder=3))
        if strip is not None:
            b = wf.strip_bounds(ink, 4)[strip]
            f0, f1 = [(v * H - r0 + 8 * H / a.shape[0]) / (r1 - r0 + 16 * H / a.shape[0]) for v in b]
            ax.add_patch(Rectangle((x + 0.3, y + h * (1 - f1)), w - 0.6, h * (f1 - f0), fill=False, ec=PC, lw=1.1, ls=(0, (3, 2)), zorder=4))
        return h

    def bars(vals, x, y, w, h, star=None):
        order = np.argsort(-vals)[:7]; v = vals[order] / vals[order].max(); bw = w / 7
        for i, (wr, val) in enumerate(zip(order, v)):
            ax.add_patch(Rectangle((x + i * bw + 0.15 * bw, y), 0.7 * bw, max(val, 0.012) * h, fc=LC if ex["hand"][wr] else RC,
                                   ec="k" if wr == ex["true"] else "none", lw=1.0, zorder=3))
            if wr == ex["true"] and star:
                ax.plot(x + i * bw + 0.5 * bw, y + val * h + 1.3, marker="*", ms=7, color="#f59e0b", mec="k", mew=0.4, zorder=4)
        ax.plot([x, x + w], [y, y], color=INK, lw=0.7)

    def note(x, y, t, **kw):
        kw.setdefault("fontsize", 5.4); kw.setdefault("color", MUTE)
        text(x, y, t, **kw)

    def block(x, y, w, h, col, label):
        ax.add_patch(Polygon([(x, y), (x + w, y + 0.18 * h), (x + w, y + 0.82 * h), (x, y + h)], closed=True, fc=col, ec="none", zorder=2))
        text(x + w / 2, y + h / 2, label, color="white", fontsize=6.3, fontweight="bold", linespacing=1.1)

    # gallery (top left) and query (bottom left)
    for j, (cls, wr, par) in enumerate([("Right-Handed", "P3", 2), ("Left-Handed", "P9", 2), ("Left-Handed", "P16", 1)]):
        page(dataset / f"{cls}/{wr}/paragraph_{par:02d}.jpg", 1.0 + 1.1 * j, 20.6 + 1.1 * j, 13.0)
    text(8.7, 33.6, "Gallery", fontweight="bold"); note(8.7, 32.1, "42 writers $\\times$ $k$ pages")
    hq = page(dataset / "Left-Handed/P16/paragraph_08.jpg", 1.0, 3.2, 18.0, frame=PC, strip=1)
    text(10.0, 3.2 + hq + 1.4, "Query", fontweight="bold", color=PC); note(10.0, 1.8, "paragraph or \u00bd, \u00bc, \u215b strip")
    # encoders
    block(26, 18.5, 8, 9, IDC, "Writer\nID")
    block(26, 2.5, 8, 9, LC, "Hand")
    arrow([(16.6, 25.4), (25.8, 25.4)], col=MUTE); note(21.2, 26.5, "templates", fontsize=5.1)
    note(30.0, 17.2, "6 features, cosine"); note(30.0, 1.2, "SVM, writer unseen")
    arrow([(19.2, 8.0), (22.5, 8.0), (22.5, 21.0), (25.8, 21.0)])
    arrow([(22.5, 8.0), (22.5, 7.0), (25.8, 7.0)])
    # identity scores and handedness evidence
    bars(ex["before"], 37.5, 19.0, 16, 8.5)
    text(45.5, 33.6, "Writer scores", fontweight="bold"); note(45.5, 32.1, "7 best of 42 writers")
    pl = ex["p_left"]
    ax.add_patch(Rectangle((37.5, 5.2), 16 * pl, 3.6, fc=LC, ec="none")); ax.add_patch(Rectangle((37.5 + 16 * pl, 5.2), 16 * (1 - pl), 3.6, fc=RC, ec="none"))
    text(37.5 + 8 * pl, 7.0, f"L {pl:.2f}", color="white", fontsize=6.3, fontweight="bold")
    text(37.5 + 16 * pl + 8 * (1 - pl), 7.0, f"R {1 - pl:.2f}", color="white", fontsize=6.3, fontweight="bold")
    text(45.5, 12.6, "P(left)", fontweight="bold"); note(45.5, 11.1, "handedness evidence")
    arrow([(34.2, 23.0), (37.0, 23.0)]); arrow([(34.2, 7.0), (37.0, 7.0)])
    # fusion
    cx, cy = 62.0, 15.0
    ax.add_patch(Circle((cx, cy), 3.6, fc="#ede9fe", ec=PC, lw=1.2, zorder=3))
    text(cx, cy, "+", color=PC, fontsize=15, fontweight="bold")
    text(cx, cy - 6.0, "Fusion", fontweight="bold", color=PC); note(cx, cy - 7.5, "Bayes or filter")
    arrow([(54.0, 23.0), (57.5, 23.0), (59.6, 17.9)]); arrow([(54.0, 7.0), (57.5, 7.0), (59.6, 12.1)])
    # final ranking
    bars(ex["after"], 72.0, 9.0, 22, 14.0, star=True)
    text(83.0, 33.6, "Ranking", fontweight="bold"); note(83.0, 32.1, "after fusion")
    arrow([(65.8, 15.0), (71.2, 15.0)], col=PC)
    text(72.0 + 22 / 14, 7.6, "1", fontsize=6, color=MUTE); text(72.0 + 22 / 14 * 3, 7.6, "2", fontsize=6, color=MUTE)
    text(72.0 + 22 / 14 * 5, 7.6, "rank", fontsize=5.4, color=MUTE, ha="left")
    for kx, fc, ec, lab in [(69.5, LC, "none", "left-handed"), (79.0, RC, "none", "right-handed"), (89.5, "white", "k", "true writer")]:
        ax.add_patch(Rectangle((kx, 2.3), 1.4, 1.4, fc=fc, ec=ec, lw=0.9)); note(kx + 2.0, 3.0, lab, ha="left", color=INK)
    save(fig, out, "fig0_method")


def fig_design(d, out, dataset):
    S, E = d["S"], d["E"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.35, 1.1, 0.9]))
    ax = axes[0]
    ink = wf.stroke_ink_uncropped(wf.load_gray(dataset / "Left-Handed/P1/paragraph_01.jpg"))
    ax.imshow(~ink, cmap="gray", interpolation="antialiased"); H = ink.shape[0]
    for n, col, lw in [(8, "#2563eb", 0.6), (4, "#dc2626", 1.2)]:
        b = wf.strip_bounds(ink, n)
        for y in [f0 for f0, _ in b] + [b[-1][1]]:
            ax.axhline(y * H, color=col, lw=lw)
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    panel(ax, "a")
    ax = axes[1]
    none, orc, bay = top1(S, "none", "-", "none"), top1(S, "oracle", "-", "filter"), top1(S, "query", "hc183", "bayes")
    ns = [1, 2, 4, 8]
    for rep in ["stroke", "vit_b_16", "resnet18", "hc183"]:
        ax.plot(range(4), [none.loc[(rep, "own", n, 6)] for n in ns], color=RCOL[rep], lw=1, marker="o", ms=2.5, label=NAME[rep])
        ax.plot(range(4), [orc.loc[(rep, "own", n, 6)] for n in ns], color=RCOL[rep], lw=0.9, ls="--")
        ax.plot(range(4), [bay.loc[(rep, "own", n, 6)] for n in ns], color=RCOL[rep], lw=0.9, ls=":")
    ax.set_xticks(range(4)); ax.set_xticklabels(["paragraph", "1/2", "1/4", "1/8"])
    ax.set_xlabel("query (own text)"); ax.set_ylabel("top-1 (42 writers)"); ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower left", fontsize=5.6)
    panel(ax, "b")
    ax = axes[2]
    e = E.groupby("rep")[["cross_hand_errors", "errors"]].sum()
    share = (e.cross_hand_errors / e.errors).reindex(REPS)
    ax.bar(range(len(REPS)), share, color=[RCOL[r] for r in REPS], width=0.7)
    n = d["U"].writer_id.nunique()
    ax.axhline((n // 2) / (n - 1), color="k", lw=0.7, ls=":")
    ax.set_xticks(range(len(REPS))); ax.set_xticklabels([NAME[r] for r in REPS], rotation=60, ha="right", fontsize=5.8)
    ax.set_ylabel("errors to other hand"); ax.set_ylim(0, 0.7)
    panel(ax, "c")
    save(fig, out, "fig1_design")


def fig_query(d, out):
    """How identification and the value of handedness change with query size and number of references."""
    S, E = d["S"], d["E"]
    none, orc, bay = top1(S, "none", "-", "none"), top1(S, "oracle", "-", "filter"), top1(S, "query", "hc183", "bayes")
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.0), constrained_layout=True, gridspec_kw=dict(width_ratios=[1.1, 1.1, 1.0]))
    reps = ["stroke", "vit_b_16", "resnet18", "hc183"]
    for ax, letter, xs, key, xl, xt in [
            (axes[0], "a", [1, 2, 4, 8], lambda rep, x: (rep, "own", x, 6), "query size (own text, $k$ = 6)", ["para.", "1/2", "1/4", "1/8"]),
            (axes[1], "b", [1, 2, 3, 6], lambda rep, x: (rep, "own", 1, x), "reference pages $k$ (own paragraph)", ["1", "2", "3", "6"])]:
        for rep in reps:
            ax.plot(range(4), [100 * none.loc[key(rep, x)] for x in xs], color=RCOL[rep], lw=1.1, marker="o", ms=2.6)
            ax.plot(range(4), [100 * orc.loc[key(rep, x)] for x in xs], color=RCOL[rep], lw=0.9, ls="--")
            ax.plot(range(4), [100 * bay.loc[key(rep, x)] for x in xs], color=RCOL[rep], lw=0.9, ls=":")
        ax.set_xticks(range(4)); ax.set_xticklabels(xt); ax.set_xlabel(xl); ax.set_ylim(0, 102); ax.set_xlim(-0.2, 3.2)
        panel(ax, letter)
    axes[0].set_ylabel("top-1 (%), 42 writers")
    for rep in reps:
        axes[0].plot([], [], color=RCOL[rep], lw=1.1, marker="o", ms=2.6, label=NAME[rep])
    axes[0].legend(frameon=False, loc="lower left", fontsize=5.6, handlelength=1.6)
    for ls, lab in [("-", "no handedness"), ("--", "known hand"), (":", "predicted (Bayes)")]:
        axes[1].plot([], [], color="#374151", ls=ls, lw=0.9, label=lab)
    axes[1].legend(frameon=False, loc="lower right", fontsize=5.6, handlelength=2.2)
    ax = axes[2]
    e = E.groupby("rep")[["cross_hand_errors", "errors"]].sum()
    share = (100 * e.cross_hand_errors / e.errors).reindex(REPS)
    ax.bar(range(len(REPS)), share, color=[RCOL[r] for r in REPS], width=0.68)
    n = d["U"].writer_id.nunique(); chance = 100 * (n // 2) / (n - 1)
    ax.axhline(chance, color="k", lw=0.7, ls=":"); ax.text(1.5, chance + 1.2, f"chance ({chance:.1f}%)", fontsize=5.6, ha="center", va="bottom")
    ax.set_xticks(range(len(REPS))); ax.set_xticklabels([NAME[r] for r in REPS], rotation=35, ha="right", fontsize=5.8)
    ax.set_ylabel("errors to other hand (%)"); ax.set_ylim(0, 70)
    panel(ax, "c")
    save(fig, out, "fig_query")


def fig_utility(d, out):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15), constrained_layout=True)
    ax = axes[0]
    L = d["LAW"]
    for hm, col, lab in [("hc183", "#111827", "HC-183"), ("hc183_writerlevel", "#6b7280", "HC-183 writer"),
                         ("hc183_adj", "#059669", "HC-183 adj."), ("stroke", "#c2410c", "STROKE")]:
        x = L[L.hand_model == hm]
        ax.scatter(x.predicted, x["filter"], s=4, color=col, alpha=0.6, lw=0, label=lab)
    ax.plot([0, 1], [0, 1], color="k", lw=0.6)
    ax.set_xlabel("accuracy $\\times$ top-1 known hand"); ax.set_ylabel("top-1 with hand filter")
    ax.text(0.03, 0.9, f"r = {d['X']['filter_law']['r']:.3f}", transform=ax.transAxes, fontsize=6.5)
    ax.legend(frameon=False, loc="lower right", fontsize=5.4, markerscale=2); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    panel(ax, "a")
    ax = axes[1]
    sg = sim_gain(d["SIM"], ("own", 1, 1))
    for col_, c, lab in [("gain_bayes", "#7c3aed", "Bayes fusion"), ("gain_filter", "#d97706", "filter")]:
        g = sg.groupby("acc")[col_]
        ax.plot(g.mean().index, 100 * g.mean(), color=c, lw=1, label=lab)
        ax.fill_between(g.mean().index, 100 * g.min(), 100 * g.max(), color=c, alpha=0.15, lw=0)
    S, LI = d["S"], d["LI"]
    real = S[(S.scenario == "query") & (S.hand_model == "hc183") & (S["query"] == "own") & (S.n == 1) & (S.k == 1)]
    none = top1(S, "none", "-", "none")
    for meth, c in [("bayes", "#7c3aed"), ("filter", "#d97706")]:
        r = real[real.method == meth]
        ax.scatter(r.hand_acc, 100 * (r.top1.to_numpy() - none.reindex(r.set_index(KEY).index).to_numpy()),
                   color=c, s=9, marker="o", edgecolor="white", lw=0.4, zorder=3)
        en = LI[(LI.label == "hand") & (LI.regime == "enrolment-trained") & (LI["query"] == "own") & (LI.n == 1) & (LI.k == 1)]
        ax.scatter(en.acc, 100 * (en[meth] - en.base), color=c, s=11, marker="^", edgecolor="white", lw=0.4, zorder=3)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.text(0.505, 3.0, "Bayes fusion", color="#7c3aed", fontsize=6, ha="left", va="bottom")
    ax.text(0.57, -33, "filter", color="#d97706", fontsize=6, ha="left", va="top")
    ax.scatter([0.80], [-38.5], color="#6b7280", s=9, marker="o"); ax.text(0.815, -38.5, "writer-independent", fontsize=5.4, va="center")
    ax.scatter([0.80], [-44.0], color="#6b7280", s=11, marker="^"); ax.text(0.815, -44.0, "enrolment-trained", fontsize=5.4, va="center")
    ax.set_xlabel("handedness accuracy"); ax.set_ylabel("change in top-1 (points)"); ax.set_ylim(-47, 13); ax.set_xlim(0.48, 1.01)
    panel(ax, "b")
    ax = axes[2]
    P = d["PREV"]
    pv = P[(P.rep == "vit_b_16") & (P["query"] == "own") & (P.n == 1) & (P.k == 1) & (P.query_hand == "left")]
    for meth, c, lab in [("none", "#111827", "none"), ("bayes", "#7c3aed", "Bayes"), ("filter", "#d97706", "filter"),
                         ("filter_prior", "#dc2626", "filter with prior")]:
        x = pv[pv.method == meth].sort_values("n_left")
        ax.plot(x.n_left, 100 * x.top1, color=c, marker="o", ms=2.5, lw=1, label=lab)
        yv = 100 * x.top1.iloc[-1] + {"none": 4.5, "bayes": -5.0, "filter": 0, "filter_prior": 0}[meth]
        ax.text(3.09, yv, lab, color=c, fontsize=5.8, va="center", ha="left")
    ax.set_xticks([1, 2, 3]); ax.set_xlim(0.85, 3.85); ax.set_xlabel("left-handed writers (+21 right-handed)")
    ax.set_ylabel("top-1, left-handed queries (%)"); ax.set_ylim(0, 100)
    panel(ax, "c")
    save(fig, out, "fig2_utility")


def fig_leakage(d, out):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.0), constrained_layout=True)
    LA = d["LA"][(d["LA"].n == 1) & (d["LA"].part == "own")]
    LI = d["LI"]
    regimes = ["writer-independent", "enrolment-trained"]
    for ax, letter, vals, yl in [
            (axes[0], "a", lambda lab, reg: LA[(LA.label == lab) & (LA.regime == reg)].acc.to_numpy() * 100, "label accuracy, own text (%)"),
            (axes[1], "b", lambda lab, reg: LI[(LI.label == lab) & (LI.regime == reg)].groupby("split").gain_bayes.mean().to_numpy() * 100,
             "change in top-1 (points)")]:
        for i, reg in enumerate(regimes):
            p = vals("pseudo", reg)
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(p)), p, s=6, color="#9ca3af", lw=0)
            ax.scatter(i + 0.28, vals("hand", reg), s=22, color="#dc2626", marker="D", lw=0)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["writer-\nindependent", "enrolment-\ntrained"]); ax.set_xlim(-0.5, 1.6)
        ax.set_ylabel(yl)
        if letter == "a":
            ax.axhline(50, color="k", lw=0.6, ls=":"); ax.set_ylim(10, 100)
        else:
            ax.axhline(0, color="k", lw=0.6, ls=":")
        panel(ax, letter)
    axes[0].scatter([], [], s=6, color="#9ca3af", label="random labels"); axes[0].scatter([], [], s=22, color="#dc2626", marker="D", label="handedness")
    axes[0].legend(frameon=False, loc="upper left", fontsize=5.6)
    ax = axes[2]
    BC = d["BC"]
    agg = BC.groupby(["control", "instance"]).gain.mean().reset_index()
    for i, (kind, lab) in enumerate([("pseudo", "random\nlabels"), ("pseudo_scaled", "random labels\nscaled"),
                                     ("random", "random\ndirections")]):
        v = agg[agg.control == kind].gain.to_numpy() * 100
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=4, color="#9ca3af", lw=0)
    hand = agg[agg.control == "hand"].gain.iloc[0] * 100
    ax.axhline(hand, color="#dc2626", lw=1); ax.text(-0.4, hand + 0.03, "handedness", color="#dc2626", fontsize=5.8, va="bottom", ha="left")
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(top=hand + 0.25)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xticks(range(3)); ax.set_xticklabels(["random\nlabels", "random labels\nscaled", "random\ndirections"], fontsize=5.8)
    ax.set_ylabel("change in top-1 (points)"); ax.set_title("both hands predicted", fontsize=6.8)
    panel(ax, "c")
    save(fig, out, "fig3_leakage")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "results/wid_handedness_2026-10-29"))
    ap.add_argument("--out", default=str(ROOT / "manuscripts/wid_handedness/figures"))
    ap.add_argument("--dataset", default=str(ROOT / "dataset"))
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    d = load(run)
    m = macros(d)
    lines = [f"% generated by scripts/wid/make_wid_paper.py from {run.relative_to(ROOT)}; do not edit by hand"]
    lines += [f"\\newcommand{{\\v{k}}}{{{v}}}" for k, v in m.items()]
    (out / "wid_macros.tex").write_text("\n".join(lines) + "\n")
    tables(d, out)
    fig_method(out, Path(a.dataset), run); fig_design(d, out, Path(a.dataset)); fig_query(d, out); fig_utility(d, out); fig_leakage(d, out)
    for k, v in m.items():
        print(f"{k:20s} {v}")


if __name__ == "__main__":
    main()
