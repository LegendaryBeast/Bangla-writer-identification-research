# Nested fusion and within-cohort robustness, 8 November 2026

This directory supplements the fixed-rule results in ../wid_revision_2026-11-08/.

- `nested_development_logits.npz` and `nested_fold_manifest.json`: fresh probabilities and exclusion manifests for nested development; outer-writer rows remain missing by design.
- `nested_selected_parameters.csv` and `nested_weight_validation_grid.csv.gz`: selected weights/temperatures and every candidate development NLL.
- `nested_writer_outcomes.csv`, `nested_summary.csv`, `nested_paired_tests.csv`, `nested_effect_intervals.csv`: held-out fixed/learned/baseline outcomes and conditional paired uncertainty.
- `temperature_validation_grid.csv.gz`, `temperature_selected_writer.csv`, `temperature_setting_summary.csv`: full ID-only temperature diagnostics.
- `calibration_seed_scores.npz`, fold manifests and `calibration_seed_*.csv`: four correlated calibration partitions, their predictions and proper scoring rules.
- `calibration_prediction_variability_*.csv`: per-unit changes and writer-balanced summaries.
- `temperature_multiplier_*.csv`: fixed 0.5/1/2 multiplier sensitivity; no best multiplier is selected.
- `evaluation_writer_omission*.csv`, `class_specific_identification.csv`: evaluation-only writer influence and class-specific outcomes; all models and the gallery stay fixed and no writer is excluded from primary results.
- `*_provenance.json`, `*_validation.json`, `additional_validation_report.json`: input/source hashes, parameters, scope and integrity checks.
- `manuscript_values.json`: generated numerical text inputs.

These are retrospective diagnostics on the observed 42-writer cohort. Writer-bootstrap intervals condition on fitted predictions and the gallery, not full model refitting or new datasets. The eight designated learned-rule comparisons share one Holm family. Calibration seeds are not independent datasets or independent participants. The exact commands are in the root README and accompanying supplement.
