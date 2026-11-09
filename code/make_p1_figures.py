#!/usr/bin/env python3
"""Revision figures for the IET Biometrics manuscript (generated from saved outputs).

protocol.pdf   Figure 1: evaluation design and the sequence of questions it answers.
precision.pdf  Figure 3: fusion effects against the oracle opportunity, 90%/95% intervals.
mechanism.pdf  Figure 5: filtering identity, oracle vs random partition, within-hand errors.
"""
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'wid-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/iet_biometrics/assets'
P1 = ROOT / 'results/p1_revision_2026-11-09'
RUN = ROOT / 'results/wid_revision_2026-11-08'
NAMES = {'stroke': 'STROKE', 'hc183': 'HC-183', 'resnet18': 'ResNet-18',
         'efficientnet_b0': 'EffNet-B0', 'vit_b_16': 'ViT-B/16', 'stroke_cnn': 'Stroke-CNN'}
COLORS = {'stroke': '#D55E00', 'hc183': '#0072B2', 'resnet18': '#009E73',
          'efficientnet_b0': '#E69F00', 'vit_b_16': '#CC79A7', 'stroke_cnn': '#555555'}
METHOD = {'identity': '#4B5563', 'oracle': '#009E73', 'random': '#9CA3AF', 'filter': '#D55E00',
          'fixed': '#0072B2', 'nested': '#CC79A7'}
INK, MUTED, GRID = '#1F2937', '#6B7280', '#E5E7EB'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 8,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.linewidth': .7, 'axes.edgecolor': '#4B5563',
                     'xtick.color': '#374151', 'ytick.color': '#374151', 'savefig.facecolor': 'white'})


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.png', dpi=220)
    plt.close(fig)


def panel(ax, letter, title=None, x=0.0):
    ax.annotate(f'({letter})', xy=(x, 1.03), xycoords='axes fraction', fontweight='bold', fontsize=9,
                va='bottom', ha='left')
    if title:
        ax.annotate(title, xy=(x, 1.03), xycoords='axes fraction', xytext=(17, 0), textcoords='offset points',
                    fontsize=8.3, va='bottom', ha='left', color=INK)


# --------------------------------------------------------------------------- Figure 1
def box(ax, x, y, w, h, text, fc, ec, size=7.4, weight='normal', color=INK, ha='center', r=.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=.8))
    tx = x + w / 2 if ha == 'center' else x + .012
    ax.text(tx, y + h / 2, text, ha=ha, va='center', fontsize=size, fontweight=weight, color=color, linespacing=1.25)


def arrow(ax, p0, p1, color='#6B7280', rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=8, lw=.9, color=color,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0))


def _line(ax, pts, color='#6B7280', lw=.9, arrow_end=False):
    xs, ys = zip(*pts)
    if arrow_end:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, solid_capstyle='butt')
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>', mutation_scale=7.5, lw=lw, color=color,
                                     shrinkA=0, shrinkB=0))
    else:
        ax.plot(xs, ys, color=color, lw=lw, solid_capstyle='butt')


def _node(ax, x, y, w, h, title, sub):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=.008', fc='#F7F7F8', ec='#4B5563', lw=.75))
    ax.text(x + w / 2, y + h * .7, title, ha='center', va='center', fontsize=7.4, fontweight='bold', color=INK)
    ax.text(x + w / 2, y + h * .32, sub, ha='center', va='center', fontsize=6.6, color='#4B5563', linespacing=1.2)


def protocol_figure():
    fig = plt.figure(figsize=(7.15, 3.15))
    ax = fig.add_axes([0, .39, 1, .61]); ax.set_xlim(0, 1); ax.set_ylim(-.07, 1); ax.axis('off')
    bx = fig.add_axes([0, 0, 1, .36]); bx.set_xlim(0, 1); bx.set_ylim(-.1, 1); bx.axis('off')
    for a, letter, title in ((ax, 'a', 'Paired design: fixed gallery and identity scores, six ranking rules'),
                             (bx, 'b', 'Questions answered in sequence')):
        a.text(.006, .985, f'({letter})', fontweight='bold', fontsize=9, va='top', color=INK)
        a.text(.04, .985, title, fontsize=8.3, va='top', color=INK)
    # pipeline
    _node(ax, .008, .315, .108, .27, 'Query unit $q$', 'paragraph\nor strip')
    _node(ax, .168, .555, .2, .27, 'Identity matcher', 'frozen features; gallery-only\nnormalisation')
    _node(ax, .168, .075, .2, .27, 'Handedness model', 'HC-183 SVM + Platt;\nquery writer held out')
    _line(ax, [(.116, .45), (.142, .45)])
    _line(ax, [(.142, .21), (.142, .69)])
    _line(ax, [(.142, .69), (.168, .69)], arrow_end=True)
    _line(ax, [(.142, .21), (.168, .21)], arrow_end=True)
    _line(ax, [(.368, .69), (.452, .69)]); _line(ax, [(.368, .21), (.452, .21)])
    ax.text(.41, .72, '$s_w$', ha='center', va='bottom', fontsize=8.2, color=INK)
    ax.text(.41, .24, '$p_q$', ha='center', va='bottom', fontsize=8.2, color=INK)
    _line(ax, [(.452, .21), (.452, .69)])
    _line(ax, [(.452, .45), (.482, .45)], arrow_end=True)
    # rule table
    x0, x1, xn, xd = .488, .995, .515, .655
    ax.text(xn, .80, 'Ranking rule', fontsize=7.2, fontweight='bold', color=INK, va='center')
    ax.text(xd, .80, 'Information supplied to the ranking', fontsize=7.2, fontweight='bold', color=INK, va='center')
    ax.plot([x0, x1], [.85, .85], color=INK, lw=.8); ax.plot([x0, x1], [.755, .755], color=INK, lw=.5)
    rules = [('identity', 'Identity only', 'identity scores $s_w$ (baseline)'),
             ('oracle', 'Known hand', 'true query and gallery hands (oracle)'),
             ('random', 'Random partition', 'known arbitrary 21/21 split (benchmark)'),
             ('filter', 'Hard filter', 'predicted hand ranked first'),
             ('fixed', 'Fixed fusion', '$s_w/T+\\log p_q(h_w)$'),
             ('nested', 'Nested fusion', '$s_w/T+\\lambda\\log p_q(h_w)$, $\\lambda$ from development writers')]
    y0, dy = .695, .115
    for i, (key, name, desc) in enumerate(rules):
        yc = y0 - i * dy
        ax.plot([x0 + .006], [yc], marker='s', ms=4.6, color=METHOD[key], mec='none')
        ax.text(xn, yc, name, fontsize=7.2, va='center', color=INK)
        ax.text(xd, yc, desc, fontsize=7.0, va='center', color='#374151')
        if i < len(rules) - 1:
            ax.plot([x0, x1], [yc - dy / 2] * 2, color='#E5E7EB', lw=.5)
    ax.plot([x0, x1], [y0 - 5 * dy - dy / 2] * 2, color=INK, lw=.8)
    ax.text((x0 + x1) / 2, -.055, 'Endpoint: paired top-1 change versus identity only, equal writer weights',
            ha='center', fontsize=6.8, color=MUTED, style='italic')
    # timeline
    steps = [('Opportunity', 'known hand\n$-$ identity'), ('Specificity', 'known hand $-$\nrandom split'),
             ('Achieved utility', 'fusion $-$ identity,\nwith precision'), ('Mechanism', 'joint correctness,\nmargins, rescues'),
             ('Writer cues', 'random labels: held-\nout vs enrolled'), ('Transfer', 'new writers, script\nand sessions')]
    xs = [.085 + i * .166 for i in range(6)]
    yl = .50
    bx.plot([xs[0], xs[-1]], [yl, yl], color='#9CA3AF', lw=.9, zorder=1)
    for i, (x, (head, body)) in enumerate(zip(xs, steps)):
        filled = i < 3
        bx.scatter([x], [yl], s=170, color=INK if filled else 'white', edgecolors=INK, linewidths=1.0, zorder=3)
        bx.text(x, yl, str(i + 1), ha='center', va='center', fontsize=7.8, fontweight='bold',
                color='white' if filled else INK, zorder=4)
        bx.text(x, yl - .21, head, ha='center', va='center', fontsize=7.3, fontweight='bold', color=INK)
        bx.text(x, yl - .46, body, ha='center', va='center', fontsize=6.7, color='#4B5563', linespacing=1.25)
    for (a, b, lab) in ((xs[0], xs[2], 'estimate utility'), (xs[3], xs[5], 'explain it and test its scope')):
        yb = .74
        bx.plot([a - .03, a - .03, b + .03, b + .03], [yb - .04, yb, yb, yb - .04], color='#9CA3AF', lw=.7)
        bx.text((a + b) / 2, yb + .03, lab, ha='center', va='bottom', fontsize=6.8, color=MUTED, style='italic')
    save(fig, 'protocol')


# --------------------------------------------------------------------------- Figure 3
def boot_ci95(row):
    return row.ci95_lo, row.ci95_hi


def precision_figure():
    P = pd.read_csv(P1 / 'precision_equivalence.csv')
    groups = {
        'BanglaHand': [('all 120 settings', 'All 120 settings'),
                       ('resnet18 own n=1 k=1', 'ResNet-18 · paragraph, k=1'),
                       ('vit_b_16 own n=1 k=1', 'ViT-B/16 · paragraph, k=1'),
                       ('resnet18 own n=4 k=6', 'ResNet-18 · quarter, k=6'),
                       ('vit_b_16 own n=4 k=6', 'ViT-B/16 · quarter, k=6')],
        'CSAFE': [('resnet18 S2_LND_r01', 'ResNet-18 · same'),
                  ('vit_b_16 S2_LND_r01', 'ViT-B/16 · same'),
                  ('resnet18 S2_WOZ_r01', 'ResNet-18 · different'),
                  ('vit_b_16 S2_WOZ_r01', 'ViT-B/16 · different')]}
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), gridspec_kw={'width_ratios': [1, 1]})
    for ax, (dataset, rows), letter in zip(axes, groups.items(), 'ab'):
        D = P[P.dataset == dataset]
        n = len(rows)
        for band, alpha in ((2, .55), (1, 1.0)):
            ax.axvspan(-band, band, color='#E0F2FE' if band == 1 else '#F0F9FF', alpha=alpha, zorder=0, lw=0)
        ax.axvline(0, color='#6B7280', lw=.8, zorder=1)
        for i, (key, label) in enumerate(rows):
            y = i
            if i % 2 == 1:
                ax.axhspan(y - .5, y + .5, color='#F9FAFB', zorder=0, lw=0)
            sub = D[D.setting == key].set_index('contrast')
            og = sub.oracle_gain_pts.iloc[0]
            ax.plot([og], [y], marker='D', ms=4.6, color=METHOD['oracle'], mec='white', mew=.5, zorder=4)
            for contrast, off, col in (('fixed_minus_none', -.16, METHOD['fixed']), ('nested_minus_none', .16, METHOD['nested'])):
                r = sub.loc[contrast]
                ax.plot([r.ci95_lo, r.ci95_hi], [y + off] * 2, color=col, lw=.9, solid_capstyle='butt', zorder=2)
                ax.plot([r.ci90_lo, r.ci90_hi], [y + off] * 2, color=col, lw=3.0, solid_capstyle='butt', alpha=.55, zorder=3)
                ax.plot([r.estimate_pts], [y + off], 'o', ms=3.8, color=col, mec='white', mew=.5, zorder=5)
            nested = sub.loc['nested_minus_none']
            txt = 'none' if nested.degenerate else f'±{nested.smallest_equivalence_margin_pts:.1f}'
            ax.text(1.01, y, txt, transform=ax.get_yaxis_transform(), va='center', fontsize=7, color=METHOD['nested'])
        ax.set_yticks(range(n), [lab for _, lab in rows])
        ax.set_ylim(n - .5, -.5)
        ax.tick_params(axis='y', length=0)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', color=GRID, lw=.5, zorder=0)
        ax.set_xlabel('Top-1 change from identity only (points)')
        lim = (-4.5, 10) if dataset == 'BanglaHand' else (-11, 9)
        ax.set_xlim(*lim)
        title = 'BanglaHand, 42 writers' if dataset == 'BanglaHand' else 'CSAFE, 96 writers, session-2 prompt'
        panel(ax, letter, title, x=-.6 if dataset == 'BanglaHand' else -.5)
    handles = [Line2D([], [], color=METHOD['fixed'], marker='o', lw=3, alpha=.8, label='Fixed fusion'),
               Line2D([], [], color=METHOD['nested'], marker='o', lw=3, alpha=.8, label='Nested fusion'),
               Line2D([], [], color=METHOD['oracle'], marker='D', lw=0, label='Known-hand gain'),
               Patch(fc='#E0F2FE', ec='none', label='±1 point'), Patch(fc='#F0F9FF', ec='#BAE6FD', lw=.4, label='±2 points')]
    fig.legend(handles=handles, ncol=5, frameon=False, loc='lower center', bbox_to_anchor=(.55, -.015), fontsize=7.4,
               handlelength=1.8, columnspacing=1.4)
    fig.subplots_adjust(left=.215, right=.95, top=.88, bottom=.27, wspace=.72)
    save(fig, 'precision')


# --------------------------------------------------------------------------- Figure 5
def mechanism_figure():
    I = pd.read_csv(RUN / 'checks_filter_identity.csv')
    O = pd.read_csv(RUN / 'checks_oracle_partitions.csv')
    F = json.loads((RUN / 'checks_findings.json').read_text())
    J = pd.read_csv(P1 / 'mechanism_bangla_j1_by_rep.csv').set_index('rep')
    C = pd.read_csv(P1 / 'mechanism_csafe_j1_pooled.csv').iloc[0]
    S = json.loads((P1 / 'p1_precision_mechanism_summary.json').read_text())
    reps = list(NAMES)
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.6), gridspec_kw={'width_ratios': [1, 1.05, 1.25]})
    a, b, c = axes
    a.scatter(100 * I.product_approximation, 100 * I.filter_top1, s=6, c=[COLORS[r] for r in I.rep], alpha=.55,
              edgecolors='none', rasterized=True)
    a.plot([0, 100], [0, 100], color='#9CA3AF', lw=.8, ls=(0, (3, 2)))
    a.set(xlim=(0, 100), ylim=(0, 100), xlabel='Independence product $aA_{\\mathrm{known}}$ (%)',
          ylabel='Observed filter accuracy (%)')
    a.text(.05, .95, f'456 combinations\n$r$ = {F["product_r"]:.3f}', transform=a.transAxes, va='top', fontsize=7.2, color=INK)
    a.grid(color=GRID, lw=.5)
    panel(a, 'a', 'Filtering identity')
    oo = O.groupby('rep')[['base', 'known_hand', 'random_partition_exact_mean']].mean().reindex(reps)
    y = np.arange(len(reps))
    b.barh(y - .19, 100 * (oo.known_hand - oo.base), height=.36, color=METHOD['oracle'], label='Known hand')
    b.barh(y + .19, 100 * (oo.random_partition_exact_mean - oo.base), height=.36, color=METHOD['random'], label='Random partition')
    b.set_yticks(y, [NAMES[r] for r in reps]); b.invert_yaxis(); b.tick_params(axis='y', length=0)
    b.set_xlabel('Oracle gain over identity (points)')
    b.grid(axis='x', color=GRID, lw=.5); b.set_axisbelow(True)
    b.set_xlim(0, 17); b.set_xticks([0, 5, 10, 15])
    k0, r0 = 100 * (oo.known_hand - oo.base).iloc[0], 100 * (oo.random_partition_exact_mean - oo.base).iloc[0]
    b.text(k0 + .35, -.19, 'known hand', va='center', fontsize=6.9, color=METHOD['oracle'], fontweight='bold')
    b.text(r0 + .35, .19, 'random partition', va='center', fontsize=6.9, color='#6B7280', fontweight='bold')
    panel(b, 'b', 'Oracle versus random split')
    order = [r for r in reps]
    yy = np.arange(len(order))
    ref = 100 * 21 / 41
    c.axvspan(0, ref, color='#FEF2F2', zorder=0, lw=0)
    c.axvline(ref, color='#6B7280', lw=.8, ls=(0, (3, 2)))
    for i, r in enumerate(order):
        row = J.loc[r]
        c.plot([100 * row.ci95_lo, 100 * row.ci95_hi], [i, i], color=COLORS[r], lw=1.4)
        c.plot([100 * row.j1_opposite_hand_share], [i], 'o', ms=4.6, color=COLORS[r], mec='white', mew=.5)
    i = len(order) + .35
    c.plot([100 * C.ci95_lo, 100 * C.ci95_hi], [i, i], color='#4338CA', lw=1.4)
    c.plot([100 * C.j1_opposite_hand_share], [i], 's', ms=4.4, color='#4338CA', mec='white', mew=.5)
    c.set_yticks(list(yy) + [i], [NAMES[r] for r in order] + ['CSAFE (neural)'])
    c.set_ylim(i + .7, -.7); c.tick_params(axis='y', length=0)
    c.set_xlim(20, 80); c.set_xlabel('Sole outranking impostor has\nopposite hand (%, 95% CI)')
    c.text(ref + 1.2, i + .62, 'random 51.2%', fontsize=6.8, color=MUTED, va='bottom')
    c.text(21.5, -.62, 'same-hand\nconfusion', fontsize=6.6, color='#B91C1C', va='top', linespacing=1.1)
    c.grid(axis='x', color=GRID, lw=.5)
    panel(c, 'c', 'Errors cluster within hand')
    fig.subplots_adjust(left=.075, right=.99, bottom=.25, top=.88, wspace=.62)
    save(fig, 'mechanism')


# --------------------------------------------------------------------------- Figure 6 (restyled)
def label_controls_figure():
    L = pd.read_csv(ROOT / 'results/wid_handedness_2026-10-29/leakage_accuracy.csv')
    fig = plt.figure(figsize=(7.15, 2.55))
    sc = fig.add_axes([.0, .02, .54, .86]); sc.set_xlim(0, 1); sc.set_ylim(0, 1); sc.axis('off')
    ax = fig.add_axes([.63, .17, .35, .71])
    sc.text(.02, .97, 'Each writer keeps one fixed random label; only the split changes', fontsize=7.4, va='top', color=MUTED)
    rows = [(.56, 'Writer held out', 'Fit: writers B, C', 'Query: writer A', '#0072B2'),
            (.13, 'Writer overlap', 'Fit: A (shared texts), B, C', 'Query: A (own texts)', '#D55E00')]
    for y, title, train, test, col in rows:
        sc.text(.02, y + .27, title, fontsize=7.8, fontweight='bold', color=col)
        box(sc, .02, y, .40, .2, train, '#F8FAFC', col, size=7.0)
        box(sc, .49, y, .14, .2, 'Model', '#FFFFFF', '#9CA3AF', size=7.0)
        box(sc, .70, y, .28, .2, test, '#F8FAFC', col, size=7.0)
        arrow(sc, (.42, y + .1), (.49, y + .1)); arrow(sc, (.63, y + .1), (.70, y + .1))
    rng = np.random.default_rng(42)
    for j, (regime, col) in enumerate([('writer-independent', '#0072B2'), ('enrolment-trained', '#D55E00')]):
        a = L[(L.n == 1) & (L.part == 'own') & (L.regime == regime)]
        r = 100 * a[a.label == 'pseudo'].acc.to_numpy()
        h = 100 * a[a.label == 'hand'].acc.iloc[0]
        ax.scatter(j + rng.uniform(-.13, .13, len(r)), r, s=13, color='#9CA3AF', alpha=.85, edgecolors='none',
                   label='Random labels' if j == 0 else None, zorder=3)
        ax.scatter([j], [h], s=40, marker='D', color=col, edgecolors='white', linewidths=.6,
                   label=None, zorder=4)
        ax.text(j + .2, h, f'{h:.1f}%', va='center', fontsize=7.2, color=col, fontweight='bold')
    ax.axhline(50, color='#9CA3AF', ls=(0, (3, 2)), lw=.8)
    ax.set_ylim(0, 100); ax.set_xlim(-.5, 1.6)
    ax.set_xticks([0, 1], ['Held out', 'Enrolled']); ax.set_ylabel('Own-text accuracy (%)')
    ax.grid(axis='y', color=GRID, lw=.5); ax.set_axisbelow(True)
    handles = [Line2D([], [], marker='o', lw=0, color='#9CA3AF', label='20 random labelings'),
               Line2D([], [], marker='D', lw=0, color='#4B5563', label='Handedness')]
    ax.legend(handles=handles, frameon=False, loc='lower right', fontsize=7)
    panel(sc, 'a', 'Split designs', x=.02)
    panel(ax, 'b', 'Writer-associated information', x=-.18)
    save(fig, 'label_controls')


# --------------------------------------------------------------------------- Figure 8 (restyled)
def csafe_figure():
    CS = P1 / 'csafe_corrected'
    S = pd.read_csv(CS / 'id_summary.csv')
    S = S[(S.cohort == 'full_balanced') & (S.hand_model == 'hc183')]
    D = pd.read_csv(P1 / 'csafe_primary_discordant_counts.csv')
    cases = [('S2_LND_r01', 'resnet18'), ('S2_LND_r01', 'vit_b_16'), ('S2_WOZ_r01', 'resnet18'), ('S2_WOZ_r01', 'vit_b_16')]
    labels = [f"{NAMES[r]} · {'same' if c == 'S2_LND_r01' else 'different'}" for c, r in cases]
    methods = [('identity_only', 'Identity only', 'identity', 'o'), ('oracle_known_hand', 'Known hand', 'oracle', 'D'),
               ('predicted_hard_filter', 'Hard filter', 'filter', 'v'), ('fixed_fusion_lambda1', 'Fixed fusion', 'fixed', 'o'),
               ('nested_fusion', 'Nested fusion', 'nested', 'o')]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.15, 2.45), gridspec_kw={'width_ratios': [1.35, 1]})
    y = np.arange(len(cases))
    for i, (c, r) in enumerate(cases):
        row = S[(S.condition == c) & (S.rep == r)].set_index('method').top1 * 100
        a.plot([row.predicted_hard_filter, row.oracle_known_hand], [i, i], color='#E5E7EB', lw=4, solid_capstyle='round', zorder=1)
        for j, (m, name, key, mk) in enumerate(methods):
            off = {'fixed_fusion_lambda1': -.16, 'nested_fusion': .16}.get(m, 0)
            a.plot([row[m]], [i + off], marker=mk, ms=5 if mk != 'D' else 4.6, lw=0, color=METHOD[key], mec='white', mew=.5,
                   zorder=3, label=name if i == 0 else None)
    a.set_yticks(y, labels); a.set_ylim(len(cases) - .5, -.5); a.tick_params(axis='y', length=0)
    a.set_xlim(25, 75); a.set_xlabel('Top-1 identification (%)'); a.grid(axis='x', color=GRID, lw=.5)
    a.spines['left'].set_visible(False)
    a.legend(frameon=False, ncol=5, loc='lower left', bbox_to_anchor=(-.42, -.47), fontsize=7, handletextpad=.3, columnspacing=1.0)
    panel(a, 'a', 'Accuracy under each ranking rule', x=-.42)
    for i, (c, r) in enumerate(cases):
        for m, off, key in (('fixed_fusion_lambda1', -.17, 'fixed'), ('nested_fusion', .17, 'nested')):
            d = D[(D.condition == c) & (D.rep == r) & (D.method == m)].iloc[0]
            b.barh(i + off, d.rescued, height=.3, color=METHOD[key], alpha=.95)
            b.barh(i + off, -d.damaged, height=.3, color=METHOD[key], alpha=.45, hatch='////', edgecolor='white', lw=0)
    b.axvline(0, color='#6B7280', lw=.8)
    b.set_yticks(y, [''] * len(cases)); b.set_ylim(len(cases) - .5, -.5); b.tick_params(axis='y', length=0)
    b.set_xlim(-6, 6); b.set_xticks([-6, -4, -2, 0, 2, 4, 6], ['6', '4', '2', '0', '2', '4', '6'])
    tot_r, tot_d = int(D.rescued.sum()), int(D.damaged.sum())
    b.set_xlabel(f'Writers damaged  |  rescued\n(totals {tot_d} | {tot_r})')
    b.grid(axis='x', color=GRID, lw=.5); b.set_axisbelow(True)
    b.spines['left'].set_visible(False)
    panel(b, 'b', 'Decisions changed by fusion', x=0)
    fig.subplots_adjust(left=.19, right=.99, top=.86, bottom=.33, wspace=.12)
    save(fig, 'csafe_identification_utility')


# --------------------------------------------------------------------------- Figure 9 (extensions)
def extensions_figure():
    P2 = ROOT / 'results/p2_extensions_2026-11-09'
    M = pd.read_csv(P2 / 'mechanism.csv'); M = M[M.condition != 'S3_WOZ_r01']
    T = pd.read_csv(P2 / 'paired_tests.csv'); T = T[T.primary]
    I = pd.read_csv(P2 / 'identification_summary.csv')
    J = pd.read_csv(P1 / 'mechanism_bangla_j1_by_rep.csv').set_index('rep')
    C = pd.read_csv(P1 / 'mechanism_csafe_j1_pooled.csv').iloc[0]
    G = pd.read_csv(P2 / 'mechanism_pooled_j1.csv').set_index('attribute')
    fig = plt.figure(figsize=(7.15, 2.75))
    a = fig.add_axes([.155, .2, .175, .66]); b = fig.add_axes([.49, .2, .155, .66]); c = fig.add_axes([.815, .2, .175, .66])
    reps = [('resnet18', 'ResNet-18'), ('vit_b_16', 'ViT-B/16'), ('sift_vlad', 'SIFT-VLAD')]
    rows = [(att, r, f'{lab} · {"gender" if att == "gender" else "hand"}') for att in ('handedness', 'gender') for r, lab in reps]
    y = np.arange(len(rows))
    for i, (att, r, lab) in enumerate(rows):
        m = M[(M.attribute == att) & (M.rep == r)][['known_gain', 'random_gain']].mean()
        base = I[(I.attribute == att) & (I.rep == r) & (I.condition != 'S3_WOZ_r01') & (I.method == 'identity_only')].top1.mean()
        a.barh(i - .19, m.known_gain, height=.36, color=METHOD['oracle'])
        a.barh(i + .19, m.random_gain, height=.36, color=METHOD['random'])
    a.set_yticks(y, [lab for *_, lab in rows]); a.invert_yaxis(); a.tick_params(axis='y', length=0)
    a.set_xlim(0, 9.8); a.set_xticks([0, 2, 4, 6, 8]); a.set_xlabel('Oracle gain, session 2 (points)'); a.grid(axis='x', color=GRID, lw=.5); a.set_axisbelow(True)
    a.axhline(2.5, color='#D1D5DB', lw=.7)
    a.legend(handles=[Patch(fc=METHOD['oracle'], label='known attribute'), Patch(fc=METHOD['random'], label='random partition')],
             frameon=False, fontsize=6.6, loc='lower right', handlelength=1, borderaxespad=.2)
    panel(a, 'a', 'Oracle opportunity', x=-.85)
    pts = [('BH · ResNet-18 · hand', 100 * J.loc['resnet18', 'j1_opposite_hand_share'], 100 * J.loc['resnet18', 'ci95_lo'], 100 * J.loc['resnet18', 'ci95_hi'], 51.2, '#009E73'),
           ('BH · ViT-B/16 · hand', 100 * J.loc['vit_b_16', 'j1_opposite_hand_share'], 100 * J.loc['vit_b_16', 'ci95_lo'], 100 * J.loc['vit_b_16', 'ci95_hi'], 51.2, '#CC79A7'),
           ('CSAFE · neural · hand', 100 * C.j1_opposite_hand_share, 100 * C.ci95_lo, 100 * C.ci95_hi, 50.5, '#4338CA'),
           ('CSAFE · gender', 100 * G.loc['gender', 'share'], 100 * G.loc['gender', 'ci95_lo'], 100 * G.loc['gender', 'ci95_hi'],
            100 * G.loc['gender', 'reference'], '#B45309')]
    for i, (lab, v, lo, hi, ref, col) in enumerate(pts):
        b.plot([ref, ref], [i - .32, i + .32], color='#6B7280', lw=.9, ls=(0, (2, 1.5)))
        b.plot([lo, hi], [i, i], color=col, lw=1.4)
        b.plot([v], [i], 'o', ms=4.6, color=col, mec='white', mew=.5)
    b.set_yticks(range(len(pts)), [p[0] for p in pts]); b.set_ylim(len(pts) - .5, -.6); b.tick_params(axis='y', length=0)
    b.set_xlim(10, 65); b.set_xlabel('Sole outranking impostor in\nopposite group (%)'); b.grid(axis='x', color=GRID, lw=.5)
    panel(b, 'b', 'Within-group errors', x=-1.1)
    order = [('handedness', 'sift_vlad'), ('gender', 'resnet18'), ('gender', 'vit_b_16'), ('gender', 'sift_vlad')]
    labels, yy = [], []
    k = 0
    for att, r in order:
        for cond in ('S2_LND_r01', 'S2_WOZ_r01'):
            for meth, off, col in (('fixed', -.15, METHOD['fixed']), ('nested', .15, METHOD['nested'])):
                t = T[(T.attribute == att) & (T.rep == r) & (T.condition == cond) & (T.method == meth)].iloc[0]
                c.plot([t.ci95_lo, t.ci95_hi], [k + off] * 2, color=col, lw=.9)
                c.plot([t.ci90_lo, t.ci90_hi], [k + off] * 2, color=col, lw=3, alpha=.55, solid_capstyle='butt')
                c.plot([t.delta_pts], [k + off], 'o', ms=3.6, color=col, mec='white', mew=.4)
            labels.append(f"{dict(reps)[r]} · {'gender' if att == 'gender' else 'hand'} · {'same' if 'LND' in cond else 'diff.'}")
            k += 1
    c.axvline(0, color='#6B7280', lw=.8)
    c.set_yticks(range(k), labels, fontsize=6.6); c.set_ylim(k - .5, -.5); c.tick_params(axis='y', length=0)
    c.set_xlim(-11, 7); c.set_xlabel('Change (points)'); c.grid(axis='x', color=GRID, lw=.5)
    c.legend(handles=[Line2D([], [], color=METHOD['fixed'], marker='o', lw=2.5, label='fixed'),
                      Line2D([], [], color=METHOD['nested'], marker='o', lw=2.5, label='nested')],
             frameon=False, fontsize=6.6, loc='upper left', handlelength=1.4, borderaxespad=.2)
    panel(c, 'c', 'Primary fusion contrasts', x=-.95)
    save(fig, 'extensions')


def main():
    protocol_figure()
    precision_figure()
    mechanism_figure()
    label_controls_figure()
    csafe_figure()
    extensions_figure()
    print('wrote', [p.name for p in OUT.glob('*') if p.stem in ('protocol', 'precision', 'mechanism')])


if __name__ == '__main__':
    main()
