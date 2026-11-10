#!/usr/bin/env python3
"""Build a result report and scientific figure from audited external outputs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/external_replication_2026-11-08"
LABELS={"resnet18":"ResNet-18","vit_b_16":"ViT-B/16","hc183":"HC-183","stroke":"STROKE"}
ORDER=list(LABELS)


def main():
    validation=json.loads((OUT/"validation.json").read_text())
    equivalent=json.loads((OUT/"feature_equivalence.json").read_text())
    assert validation["passed"] and equivalent["passed"]
    S=pd.read_csv(OUT/"id_summary.csv")
    T=pd.read_csv(OUT/"paired_tests.csv")
    D=pd.read_csv(OUT/"diagnostics.csv")
    P=pd.read_csv(OUT/"selected_parameters.csv")
    primary=S[(S.cohort=="full_balanced")&(S.direction=="T1_to_T2")]
    tests=T[T.primary]
    table=primary.pivot(index="rep",columns="method",values="top1").reindex(ORDER)
    columns=["identity_only","oracle_known_hand","predicted_hard_filter","fixed_fusion_lambda1","nested_fusion"]
    table[columns].mul(100).to_csv(OUT/"primary_accuracy_percent.csv",float_format="%.6f")
    lines=["# Independent-dataset experiment: MPHD v2", "",
        "Completed 8 November 2026. The initially requested KHATT replication could not proceed through the inspected public routes because the original host failed DNS and a verified writer-handedness mapping was not found in the OCR mirrors. Its official LDC distribution requires access/licensing. Following the user's instruction to check alternatives, the experiment used the publicly accessible MPHD v2 release.", "",
        "Dataset: [MPHD v2](https://doi.org/10.5281/zenodo.23557640), with [original documentation](https://github.com/jampour/MPHD). This is a separate Persian cohort. The release contains 45 left-handed and 455 right-handed writers. All 45 left-handers and 45 seeded right-handers were selected; the individual JSON annotations agree with their CSV handedness labels. All 180 paragraph images passed the frozen feature-validity rules; no writer was excluded for poor performance.", "",
        "The protocol was saved before external prediction or identification outcomes. It is a prospective specification for this extension of an already completed study, not a publicly registered preregistration. Model architecture, feature extraction, SVM C, calibration rules, temperature grid and fusion-weight grid were retained from the original pipeline. No external-outcome-driven feature or weight-grid search was performed.", "",
        "## Primary evaluation", "",
        "Fixed Text1 supplies one gallery reference per writer; variable Text2 supplies one query. Gallery and query are distinct paragraph images from the same original scanned form. The full gallery has 90 writers, balanced 45/45. Identity features are standardized on gallery references only. Each query writer is excluded from SVM training and calibration; nested fusion regenerates development predictions inside that outer writer exclusion. Known gallery handedness remains permitted metadata.", "",
        "Top-1 identification accuracy (%):", "",
        "| Representation | Identity only | Oracle hand | Hard filter | Fixed fusion | Nested fusion |",
        "|---|---:|---:|---:|---:|---:|"]
    for rep,row in table.iterrows():
        lines.append("| "+LABELS[rep]+" | "+" | ".join(f"{100*row[c]:.2f}" for c in columns)+" |")
    lines += ["", "ResNet-18 and ViT-B/16 are the designated primary representations; HC-183 and STROKE are descriptive comparisons. The HC-183 query-handedness predictor has 50.00% balanced accuracy, NLL 0.7140 and Brier score 0.2597 in this direction. Therefore the experiment examines utility under weak external attribute prediction, not the value of a strongly informative predictor.", "",
        "Prespecified fixed/nested differences from identity only (percentage points):", "",
        "| Representation | Fusion | Difference | Conditional 95% CI | Holm-adjusted p |",
        "|---|---|---:|---:|---:|"]
    for row in tests.itertuples():
        method="Fixed" if row.method=="fixed_fusion_lambda1" else "Nested"
        lines.append(f"| {LABELS[row.rep]} | {method} | {100*row.delta_top1:+.2f} | [{100*row.ci_lo:.2f}, {100*row.ci_hi:.2f}] | {row.p_holm_primary:.4f} |")
    lines += ["", "Intervals use 10,000 hand-stratified writer-bootstrap draws, conditional on the enrolled gallery and fitted predictions. Sign-flip tests use 9,999 draws; Holm correction covers the four designated comparisons. A [0,0] conditional interval means the two fitted rules make identical correctness decisions on these observed queries; it does not establish population equivalence or remove model-fitting uncertainty.", "",
        "Oracle gains are +10.00 points for ResNet-18 and +7.78 for ViT-B/16. Their expected gains under an equally sized known random writer partition are +9.59 and +7.54 points, respectively. These are descriptive comparisons, not evidence that the two information sources are equivalent.", "",
        "Fixed ResNet-18 fusion rescues three of 90 queries and damages none; ViT fusion rescues one and damages one. Nested fusion makes the same top-1 correctness decisions as identity alone for both primary representations. The NLL-selected weight is zero in 81/90 ResNet-18 fits and 73/90 ViT fits; positive selected weights do not necessarily change top-1 identity decisions.", "",
        "## Robustness and interpretation", "",
        "The same experiment was refitted in 20 seeded balanced 21/21 galleries, and in the reverse text direction. These overlapping galleries are dependent sensitivity evaluations, not 20 independent dataset replications. The full run has 168 representation/direction/gallery settings and 37,200 saved writer-method outcomes.", "",
        "Across the 20 fixed-to-variable 42-writer galleries:", "",
        "| Representation | Oracle gain range | Fixed-fusion change range | Nested-fusion change range | Hard-filter change range |",
        "|---|---:|---:|---:|---:|"]
    sensitivity=S[(S.cohort!="full_balanced")&(S.direction=="T1_to_T2")].pivot(index=["cohort","rep"],columns="method",values="top1")
    for rep in ORDER:
        z=sensitivity.xs(rep,level="rep")
        ranges=[]
        for method in ("oracle_known_hand","fixed_fusion_lambda1","nested_fusion","predicted_hard_filter"):
            change=100*(z[method]-z.identity_only)
            ranges.append(f"[{change.min():+.2f}, {change.max():+.2f}]")
        lines.append("| "+LABELS[rep]+" | "+" | ".join(ranges)+" |")
    lines += ["", "Ranges are in percentage points and are descriptive, not confidence intervals. Reverse-direction results and every individual setting are retained in id_summary.csv and paired_tests.csv.", "",
        "The result is consistent with the original paper's distinction between oracle gains and operational predicted-attribute utility. It supports the usefulness of the evaluation framework across datasets; it does not establish a universal null effect of handedness. The external predictor is weak, the dataset contains same-form paragraph pairs, and temperature/weight choices optimize development NLL rather than top-1 accuracy. Scanned Persian text also does not independently reproduce Bangla photographic acquisition or separate capture sessions.", "",
        "The unusually strong descriptive STROKE identity result (96.67%) leaves little headroom for fusion in that representation and should not be generalized to independent-session performance.", "",
        "## Validation", "",
        f"Audited {validation['outer_writer_fits_audited']:,} outer writer fits, {validation['attribute_models_audited']:,} final/development attribute models and {validation['inner_calibration_folds_audited']:,} calibration folds. No scored writer appears in its attribute training/calibration, and development validation writers are also excluded. Input, code, image and checkpoint hashes were verified. All reported accuracies match saved writer outcomes. The exact hard-filter identity and fusion margin reconstruction pass. Whole-paragraph adapter features exactly match the original extractor on one image from each handedness group.", "",
        "Dataset files and model weights remain in the ignored dataset/ directory. The manuscript and submission package have not been changed, and nothing has been uploaded or submitted.", "",
        "![External replication figure](external_replication.png)", "",
        "Figure: (a) Full 90-writer fixed-to-variable top-1 performance. (b) Fixed/nested top-1 changes with conditional writer-bootstrap 95% intervals. (c) Changes in 20 overlapping balanced 42-writer galleries; points are descriptive dependent subsamples. All samples are paragraph pairs from a common source form."
    ]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n")
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
        "svg.fonttype":"none","savefig.bbox":"tight"})
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),gridspec_kw={"width_ratios":[1.25,1,1.1]})
    colors={"identity_only":"#555555","oracle_known_hand":"#009E73","predicted_hard_filter":"#D55E00",
        "fixed_fusion_lambda1":"#0072B2","nested_fusion":"#CC79A7"}
    names={"identity_only":"Identity","oracle_known_hand":"Oracle","predicted_hard_filter":"Hard filter",
        "fixed_fusion_lambda1":"Fixed","nested_fusion":"Nested"}
    y=np.arange(4)
    for j,m in enumerate(columns):
        axes[0].plot(100*table[m],y+(j-2)*.12,"o",ms=4,color=colors[m],label=names[m])
    axes[0].set(yticks=y,yticklabels=[LABELS[r] for r in ORDER],xlabel="Top-1 identification (%)",xlim=(0,104))
    axes[0].invert_yaxis()
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,ncol=5,loc="upper center",bbox_to_anchor=(.5,1.02),
        frameon=False,columnspacing=1.5,handletextpad=.3)
    for method,offset in [("fixed_fusion_lambda1",-.1),("nested_fusion",.1)]:
        rows=T[(T.cohort=="full_balanced")&(T.direction=="T1_to_T2")&(T.method==method)].set_index("rep").reindex(ORDER)
        values=100*rows.delta_top1.to_numpy()
        errors=np.stack([values-100*rows.ci_lo.to_numpy(),100*rows.ci_hi.to_numpy()-values])
        axes[1].errorbar(values,y+offset,xerr=np.maximum(errors,0),fmt="o",ms=4,capsize=2,color=colors[method])
    axes[1].axvline(0,color="#888",lw=.8,ls="--")
    axes[1].set(yticks=y,yticklabels=[],xlabel="Change from identity (points)")
    axes[1].invert_yaxis()
    for method,offset in [("fixed_fusion_lambda1",-.12),("nested_fusion",.12)]:
        for i,rep in enumerate(ORDER):
            z=sensitivity.xs(rep,level="rep")
            change=100*(z[method]-z.identity_only)
            jitter=np.linspace(-.055,.055,len(change))
            axes[2].scatter(change,np.full(len(change),i+offset)+jitter,s=10,alpha=.65,color=colors[method],edgecolors="none")
    axes[2].axvline(0,color="#888",lw=.8,ls="--")
    axes[2].set(yticks=y,yticklabels=[],xlabel="Change in 42-writer galleries (points)")
    axes[2].invert_yaxis()
    for ax,label,title in zip(axes,"abc",["90-writer gallery","Conditional uncertainty","Gallery sensitivity"]):
        ax.set_title(f"({label}) {title}",loc="left",fontsize=10,pad=7)
        ax.grid(axis="x",alpha=.16)
        ax.set_ylim(3.45,-.45)
    fig.subplots_adjust(wspace=.18,top=.79,bottom=.18)
    for ext in ("png","svg"):
        fig.savefig(OUT/f"external_replication.{ext}",dpi=220)
    plt.close(fig)
    print(OUT/"RESULTS.md")


if __name__=="__main__":
    main()
