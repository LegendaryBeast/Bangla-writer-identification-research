# Results of the reported run (2026-10-29)

42 writers of BanglaHand; 5,040 units (336 paragraphs, n = 1, and their 2, 4 and 8 horizontal strips), of which 39 strips
with too little writing are marked invalid. Representations: STROKE, HC-183, ResNet-18, EfficientNet-B0, ViT-B/16,
Stroke-CNN. Settings: four query sizes x five reference conditions x six representations = 120.

| File | Written by | Content |
|---|---|---|
| `features/units.csv` | `wid_features.py` | One row per unit: writer, label (left = 1), paragraph, n, strip, row bounds, validity, stroke width, ink density, tiles |
| `features/unit_features.npz` | `wid_features.py` | Feature matrices of all units, aligned with `units.csv` (`unit_id`) |
| `features/unit_feature_provenance.json` | `wid_features.py` | Dimensions, device, library versions, run time |
| `handedness_lowo.csv.gz` | `run_wid_handedness.py` | Leave-one-writer-out SVM score and calibrated left-handed log-odds of every unit (HC-183, adjusted HC-183, STROKE) |
| `handedness_accuracy.csv` | `run_wid_handedness.py` | Handedness accuracy by model, unit size and text type; writer-level accuracy |
| `id_ranks.csv.gz` | `run_wid_handedness.py` | Rank of the true writer for every query, reference set, scenario, handedness model and fusion rule |
| `id_summary.csv` | `run_wid_handedness.py` | Top-1, top-5, MRR and penetration per setting, scenario, handedness model and rule |
| `id_paired_tests.csv` | `run_wid_handedness.py` | Writer-level differences to no fusion: sign-flip p, bootstrap interval, McNemar; primary tests with Holm p |
| `id_errors_cross_hand.csv` | `run_wid_handedness.py` | Identification errors and errors assigned to a writer of the other hand |
| `breakeven_analytic.csv` | `run_wid_handedness.py` | Top-1 without and with known hand, break-even accuracy and achieved accuracy per setting |
| `breakeven_simulation.csv.gz` | `run_wid_handedness.py` | Top-1 with synthetic calibrated evidence of accuracy 0.50-1.00 (200 draws) |
| `prevalence.csv` | `run_wid_handedness.py` | Galleries of 21 right- and 1-3 left-handed writers (1,000 draws): top-1 and penetration by query hand and rule |
| `both_control.csv`, `both_control_summary.csv` | `run_both_control.py` | Scenario with both hands predicted: handedness against 20 random labels (calibrated and scaled) and 100 random directions |
| `leakage_accuracy.csv` | `run_wid_extras.py` | Unit accuracy of handedness and of 20 random labels, writer-independent and enrolment-trained |
| `leakage_identification.csv` | `run_wid_extras.py` | Identification gain of the same evidence per setting |
| `confidence_gating.csv` | `run_wid_extras.py` | Hand filter applied to the most confident share of queries |
| `filter_law.csv` | `run_wid_extras.py` | Observed filter top-1 against accuracy x top-1 with known hand, for every setting and handedness model |
| `*_provenance.json`, `log_*.txt` | all | Parameters, seeds, run times and console logs |
