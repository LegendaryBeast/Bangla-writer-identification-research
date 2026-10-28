#!/usr/bin/env python3
"""Figures, tables and macros of the writer identification manuscript (manuscripts/wid_handedness/).

Reads the outputs of run_wid_handedness.py, run_both_control.py and run_wid_extras.py and writes
  figures/wid_macros.tex     every number quoted in the text
  figures/table_main.tex, figures/table_tests.tex
  figures/fig1_design.pdf, fig2_utility.pdf, fig3_leakage.pdf (and .png)
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
    (out / "table_tests.tex").write_text("% generated by scripts/wid/make_wid_paper.py\n" + "\n".join(rows) + "\n\\bottomrule\n")


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
    ax.scatter([], [], color="#6b7280", s=9, marker="o", label="writer-independent")
    ax.scatter([], [], color="#6b7280", s=11, marker="^", label="enrolment-trained")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("handedness accuracy"); ax.set_ylabel("change in top-1 (points)"); ax.set_ylim(-40, 12)
    ax.legend(frameon=False, loc="lower right", fontsize=5.4)
    panel(ax, "b")
    ax = axes[2]
    P = d["PREV"]
    pv = P[(P.rep == "vit_b_16") & (P["query"] == "own") & (P.n == 1) & (P.k == 1) & (P.query_hand == "left")]
    for meth, c, lab in [("none", "#111827", "none"), ("bayes", "#7c3aed", "Bayes"), ("filter", "#d97706", "filter"),
                         ("filter_prior", "#dc2626", "filter with prior")]:
        x = pv[pv.method == meth].sort_values("n_left")
        ax.plot(x.n_left, 100 * x.top1, color=c, marker="o", ms=2.5, lw=1, label=lab)
    ax.set_xticks([1, 2, 3]); ax.set_xlabel("left-handed writers in gallery (+21 right)")
    ax.set_ylabel("top-1 of left-handed queries (%)"); ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=5.6, loc="center right")
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
            ax.axhline(50, color="k", lw=0.6, ls=":"); ax.set_ylim(30, 100)
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
    ax.axhline(hand, color="#dc2626", lw=1); ax.text(2.45, hand, "handedness", color="#dc2626", fontsize=5.8, va="bottom", ha="right")
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
    fig_design(d, out, Path(a.dataset)); fig_utility(d, out); fig_leakage(d, out)
    for k, v in m.items():
        print(f"{k:20s} {v}")


if __name__ == "__main__":
    main()
