# Additional revisions — fusion, robustness and contribution statement

Version 1.2.0, 8 November 2026. These additions implement the three further priorities requested after the first complete revision. The earlier source and reproducibility ZIPs are retained; the current archives use the `-v1.2.0` suffix.

## 1. Properly cross-validated weight comparison

- Outer evaluation leaves one query writer out. Five development writer folds are regenerated inside the other 41 writers. Each development SVM, standardizer and inner five-fold calibrator excludes both the outer writer and the development validation fold.
- There are 840 development models; the final outer query prediction uses the original, valid model trained on the other 41 writers. Development models train on 32/33 writers, a documented finite-sample CV difference.
- Temperature is held at the same ID-only value as fixed fusion. The known 42-writer gallery and its hand metadata remain permitted candidate information. Outer query outcomes do not select temperature or weight.
- A preset nonnegative grid includes zero (no attribute) and one (fixed fusion); the objective is equal-writer development NLL. Exact ties select the smaller weight.
- Four designated settings by two learned-rule comparators form one eight-comparison Holm family. This is an exploratory post-review analysis.

| Learned-weight contrast, all settings | Mean, points | Conditional 95% writer interval |
|---|---:|---:|
| Compared with no fusion | -0.08 | [-1.13, +0.81] |
| Compared with fixed fusion | +0.49 | [-0.15, +1.28] |

All eight adjusted p-values are 1. The median selected weight is 0.5; zero is selected in 1.3% of writer/setting fits. A smaller observed loss than unit weighting is not interpreted as a demonstrated positive gain over baseline.

## 2. Uncertainty and robustness

- Temperature diagnostics retain all 31-point development loss curves for all 5,040 outer-writer/setting fits, selected values, boundary frequencies, second-best gaps and near-optimal candidate counts.
- Preset half and double temperatures give mean fixed-fusion effects from −1.81 to +0.04 points; conditional intervals include zero. For ranking, changing temperature also changes relative attribute weight, so it is not claimed as an independent evidence source.
- Three additional calibration partitions plus the original partition give own-paragraph hand accuracies of 64.3–66.7%. Only 2.4% of these queries change predicted hand at least once. Fixed-fusion mean effects remain −0.57 to −0.50 points. Full outer SVM scores are invariant; these are correlated calibration-partition replicates, not four independent studies.
- Omitting each writer only from evaluation averaging, while retaining the gallery/models and giving each hand half the weight, yields learned-fusion means from −0.24 to +0.22 points. Oracle gains and large hard-filter losses persist. No writer is removed from the primary analysis.
- Writer-specific influences, class-specific effects, seed-specific proper scoring rules and conditional writer intervals are supplied. They do not replace independent-cohort, cross-session, new-gallery or full-refitting uncertainty.

## 3. Contribution statement

The introduction and abstract now organize the contribution around four distinctions: oracle partition gains, utility achieved by predicted attributes, dependence of attribute/identification errors, and writer-associated information or leakage risks when seen/unseen-writer targets are conflated. The filtering identity is explicitly an analytical diagnostic within that evaluation framework. It is not presented as a new universal law. The conclusion includes both fixed and nested learned-weight results and retains the closed-gallery, one-dataset scope.

## Numerical and document verification

- The independent full 42-candidate softmax reproduces the reduced two-hand weight-selection objective.
- Every development and inner-calibration fold manifest satisfies its writer exclusions. Changing outer query evidence or targets does not change selected weights.
- The nested run reproduces all 5,040 writer-level outcomes of each baseline comparator to floating-point precision.
- Saved writer outcomes reconstruct every reported mean, and the eight-comparison Holm adjustment is checked directly.
- Cached calibration reruns were exercised; an object-array loading issue was corrected before packaging.
- Main manuscript: seven pages; abstract: 179 words. Supplement: fourteen pages. New tables, the full 120-setting comparison and the four-panel diagnostics figure are generated from results.
- The numerical checks, LaTeX/reference/bounding-box checks, visual PDF review and archive integrity checks are recorded in the supplied reports.

## Reproduction

```bash
python code/run_nested_fusion.py --jobs 3
python code/run_robustness_diagnostics.py --jobs 3 --nested results/wid_fusion_robustness_2026-11-08/nested_writer_outcomes.csv
python code/make_fusion_robustness_paper.py
python code/validate_fusion_robustness.py
```

Existing fixed-rule analyses are reproduced using the commands in the root README. Then rebuild both PDFs and run `python code/validate_revision.py --pdf`, followed by `python code/build_revision_package.py`. No external upload, submission or DOI registration is performed.
