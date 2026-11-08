# Self-contained implementation and interpretation

The complete typeset specification is `paper/wid_supplement.pdf`. Source implementations are included; no companion manuscript is required to reconstruct descriptors or scores.

## Units and fixed validity rules

Each of 336 photographs gives a full paragraph and 2, 4 and 8 strips: 5,040 candidate units. Shared paragraphs are numbered 1–6 and own texts 7–8. Validity requires at least 200 ink pixels in the high-resolution mask, at least 50 in the 768-pixel mask, computable contour/skeleton descriptors and finite features. Exactly 39 strips fail (one quarter strip and 38 eighth strips); no paragraph or writer is removed. `features/units.csv` preserves all candidate IDs, validity, strip boundaries and covariates. Every retained setting includes all 42 writers.

Strips use the 1,536-pixel mask's text rows (ink fraction >0.01, 0.5th/99.5th percentiles). Cuts move to low-ink rows near equal-height boundaries using a Gaussian-smoothed row profile, standard deviation 96/5 pixels. They keep the photograph's full width. Implementation: `code/wid_features.py`.

## Descriptors and preprocessing

- **HC-183**: resize to width 768; mask pixels 17 levels darker than a Gaussian local illumination estimate (sigma 17), with 2x2 opening. A 4x4 grid contributes nine-bin magnitude-weighted unsigned gradient histograms and ink densities (160 values); 18 row/column projection statistics and five connected-component area quantiles complete 183 dimensions. `code/hc183.py`.
- **STROKE**: width 1,536, local-contrast mask with brightness gates, 2x2 opening, components below 12 pixels removed, text-block crop. Contours with ink on the left are resampled every two pixels; legs use twice mean stroke width (minimum four pixels). Direction, hinge, turning, skeleton endpoint/orientation/slant histograms and topology rates give 208 dimensions. No grayscale values enter the descriptor after mask construction. `code/stroke_features.py`, `code/wid_features.py`.
- **Frozen grayscale networks**: resize width 1,568, retain 224x224 tiles with at least 1% ink, repeat gray into RGB, use ImageNet mean (0.485,0.456,0.406) and standard deviation (0.229,0.224,0.225), remove final classification layers and average tile embeddings. Dimensions: ResNet-18 512, EfficientNet-B0 1,280, ViT-B/16 768. The networks remain in evaluation mode without fine-tuning.
- **Stroke-CNN**: the frozen ResNet-18 checkpoint on binary ink tiles after stroke-width normalization. The exact tile preparation is in `wid_features.py`; checkpoint references are in `CHECKPOINTS.md`.

The original feature extraction used Python 3.13.9, torch 2.14.0, torchvision 0.29.0 on MPS. Exact frozen matrices are supplied. Re-extraction with another accelerator/library can change floating-point details.

## Identification and attribute fitting

Fit each identity scaler only to the selected whole-paragraph gallery references; transform and L2-normalize all references/queries, form normalized mean writer templates, then use cosine scores. A shared query excludes its prompt from every gallery writer; own queries use every combination of 1, 2, 3 or 6 shared reference paragraphs. The 120 settings and all reference configurations are explicit manifests.

HC-183 handedness uses a linear SVM (C=0.1) trained on the other 41 writers' valid shared units of the same size. Five-fold writer-group out-of-fold margins supply logistic calibration (C=10,000). The query writer never enters those stages. This is a cross-validation estimate, not an inference algorithm that selects a model using an unknown query identity. A deployed unseen-writer pipeline would freeze one model/calibrator and development-selected fusion parameters before future queries.

ID-only temperature minimizes equal-writer negative log loss over the other query writers using 31 values from 0.001 to 1. Fixed fusion adds log query-hand probability to scaled identity scores, with known gallery hands. Nested fusion selects weight from {0,.1,.2,.3,.5,.7,1,1.5,2,3,5} using development probabilities regenerated inside each outer exclusion. The 840 development-fold records and their inner calibration exclusions are supplied. Hard filtering places the predicted-hand group first.

## Uncertainty and diagnostic scope

Outcomes average configurations within valid query units, then units within writer; settings and writers have equal aggregate weight. Paired intervals use 10,000 hand-stratified writer bootstrap draws; tests use 9,999 sign flips. Shared resampling indices retain cross-setting dependence. Four designated fixed-rule comparisons have Holm correction; the nested-vs-none/fixed contrasts form a separate eight-test Holm family. These retrospective comparisons are not preregistered confirmation or equivalence tests.

Temperature loss profiles and fixed multipliers, four calibration partitions, evaluation-writer omission, class-specific effects, synthetic evidence and reduced-gallery prevalence analyses are included. They retain or reuse this cohort and do not constitute new-dataset or cross-session validation. The complete inference scope and all additional intervals are in the supplement.

With binary groups, filtered success equals joint correct handedness and correct known-hand identification; the product of marginal accuracies requires independence. The margin diagnostic reconstructs every fixed/nested decision and separates rescued from damaged matches.

Random writer-label prediction under enrolment training diagnoses available writer-associated information and a changed learning problem under overlap. It does not prove a leakage mechanism in the writer-disjoint primary pipeline. The schematic in supplementary Figure S2 makes the split distinction explicit. The observed 52.4–76.2% random-label range is not interpreted as a causal identity contribution or a calibrated chance test.
