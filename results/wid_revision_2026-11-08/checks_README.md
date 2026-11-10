# Additional checks after review

These analyses address the independent review dated 2026-11-08. They are exploratory additions, rather than a prospectively registered confirmation. The original `results/wid_handedness_2026-10-29/` files are read-only inputs.

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python code/run_revision_checks.py --both-controls --jobs 6
OPENBLAS_NUM_THREADS=1 python code/summarize_revision_checks.py
```

The first command caches the 20 pseudo-label and 100 random-direction descriptor controls. Its saved cache can be reused on a subsequent run with matching parameters and handedness evidence. Identification scores in every analysis use the revised protocol: fit the scaler only to the gallery paragraphs selected for the reference configuration. The held-out-writer handedness predictions are reused from the original correctly grouped SVM analysis.

Each setting averages configurations within a query unit and then units within a writer, giving each writer equal weight. Overall effects give each of the 120 settings equal weight. For the available data, every valid unit within a setting has the same number of reference configurations, so this also equals averaging query/configuration rows within each writer.

The 10,000 stratified writer bootstrap draws and 9,999 sign flips condition on the fitted evidence, the enrolled gallery, and its score matrices. Predictors and gallery membership are not refitted or resampled. These intervals therefore do not represent uncertainty of the entire learning procedure on a new gallery. Intervals for the product approximation resample writers rather than treating the 456 correlated setting/model combinations as independent observations.

| File | Contents |
|---|---|
| `checks_filter_identity.csv` | Exact event identity and independence approximation, conditional identification success, covariance residuals and per-combination intervals, Frechet bounds, and the independence-based break-even ratio. All 456 combinations use known gallery labels. |
| `checks_filter_identity_writer.csv` | Per-writer event probabilities underlying those calculations. |
| `checks_dependence_uncertainty.csv` | Writer-bootstrap intervals for pooled approximation bias, mean absolute error and descriptive correlation, overall and by handedness model. |
| `checks_calibration.csv` | Held-out-writer accuracy, Brier score and log loss, with writer-bootstrap intervals. |
| `checks_reliability_bins.csv` | Five fixed probability bins, writer-weighted mean probability and observed frequency, with cluster intervals. Sparse bins have substantial uncertainty. |
| `checks_calibration_writer.csv` | Per-writer proper-score and accuracy observations. |
| `checks_identification_writer.csv` | Per-writer outcomes for the baseline, known hand, expected oracle random partition, primary predicted-query fusion and filtering, and exploratory predicted-both fusion. |
| `checks_effect_intervals.csv` | Paired writer intervals for equal-setting average effects and the four original primary contrasts. Holm correction applies only to those four contrasts. |
| `checks_oracle_partitions.csv` | Known-hand oracle versus 2,000 uniformly drawn balanced binary writer partitions, with the query's true partition membership supplied. Includes the analytic expectation from each query's baseline rank. |
| `checks_oracle_partition_draws.npz`, `checks_random_partitions.npz` | Reproducible partition assignments and top-1 outcomes. Draw quantiles express variation among partitions; they are not sampling confidence intervals. |
| `checks_both_direct_paired.csv` | Direct handedness-descriptor minus mean-control contrasts with paired writer intervals, plus the finite-draw empirical exceedance fraction. |
| `checks_both_own_effect.csv`, `checks_both_direct_paired_own.csv`, `checks_both_own_provenance.json` | Reportable exploratory descriptor-agreement summaries restricted to the 96 own-text settings, with writer intervals/sign flips and direct controls. |
| `checks_both_control_writer.csv.gz` | Per-setting, per-writer outcomes for every realized descriptor control. |
| `checks_both_evidence.npz`, `checks_both_evidence_provenance.json` | Cached pseudo/random descriptor evidence, seeds and pseudo-label assignments. |
| `checks_cross_hand_writer.csv`, `checks_cross_hand_writer_weighted.csv`, `checks_cross_hand_summary.csv` | Error counts, equal-writer/equal-setting rates, and cross-hand shares with intervals. Proximity to 21/41 is not a demonstration of independence. |
| `checks_temperature_writer.csv` | Identity-only temperatures fitted excluding the scored writer's query loss. |
| `checks_findings.json`, `checks_findings.md` | Compact numerical findings. |
| `checks_provenance.json`, `checks_followup_provenance.json`, `checks_log.txt` | Parameters, versions, input hashes and run records. |
| `checks_validation.json` | Agreement with the separate main analysis, exact-identity and bound checks, exhaustive small-gallery tie check, Monte Carlo accuracy and control-row counts. |

The random-partition oracle supplies the same amount of perfectly known binary side information as the handedness oracle. It is an information baseline, not a deployable way to discover a questioned writer's membership.

Only the own-text summaries should be reported as the exploratory predicted-both result. Shared queried pages can enter training of an impostor gallery fold's descriptor, so the all-120-setting predicted-both summaries are retained as historical diagnostic outputs. Own-text paragraphs 7--8 never enter the shared-paragraph auxiliary training. Recreate the restricted tables without refitting models or rewriting the other results with `OPENBLAS_NUM_THREADS=1 python code/summarize_revision_checks.py --own-only`.

Even in the own-text predicted-both analysis, gallery writer-cross-fitted descriptor predictors can contain the queried writer's enrollment samples. Scaled and random controls also use cohort-level score spread, as in the historical experiment. This is an enrolment-conditioned descriptor-agreement analysis. A gain does not establish physiological causation or handedness-specific information. The pseudo labels are deliberately balanced within true hand, so their empirical exceedance fraction describes the designed control family; it is not an exchangeable permutation-test p-value for handedness itself. The direct writer intervals condition on the realized control family.
