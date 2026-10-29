# Paper 3 plan: When does handedness help writer identification?

Agreed scope (2026-10-29): honest "when does it help" framing; saved features plus new crop features (no network training); BanglaHand only (42 writers); Pattern Recognition Letters-style short paper.

Working title: *When does handedness help writer identification? Upper bounds, break-even accuracy and fusion on BanglaHand*

## 1. Motivation and claim to test

The DiB article states that handedness "can narrow the search space of automated writer identification and verification systems". The PRL article shows handedness of unseen writers is predictable (HC-183, 77.4% writer BA). Neither paper tests whether that prediction helps writer identification. This paper does, and reports when it does not.

## 2. What the feasibility check already showed (2026-10-29, saved features, not final numbers)

- Ceiling: shared paragraph → 5 shared references gives STROKE 100%, ViT/ResNet-18 about 96% top-1. Gains are only possible in harder regimes.
- Harder regimes leave room: own text → 1 shared reference gives STROKE 88.1, ViT 66.7, ResNet-18 57.1, Stroke-CNN 53.6, HC-183 38.1.
- Oracle handedness (both sides known) adds 0–12 points; most ID errors are between writers of the same hand.
- Predicted handedness (HC-183 SVM, writer-independent, about 69–76% per paragraph): a hard gallery filter costs 20–30 points; soft fusion gains at most about 1 point, even with the weight tuned on the test data.
- Hard-filter break-even: filtering helps only if handedness accuracy a > Acc_base / Acc_oracle (e.g. 0.57/0.63 ≈ 0.90 for ResNet-18).

These cells have been seen. Pre-specification below applies to the analyses not yet run (crops, calibrated fusion, simulation, prevalence); the paper will say the feasibility check was done first.

## 3. Research questions

- RQ1 (upper bound): How much does known handedness improve closed-set writer identification, by representation and difficulty?
- RQ2 (realistic): Does handedness predicted for the query (writer-independent model) improve identification, with gallery handedness known or predicted?
- RQ3 (break-even): At what handedness accuracy does each fusion rule start to help, analytically and in simulation?
- RQ4 (search space): How much does handedness reduce the part of the gallery an examiner must inspect (penetration rate), and how does that depend on left-handed prevalence?

## 4. Data and difficulty regimes

BanglaHand, 42 writers × 8 paragraphs (paragraphs 1–6 shared prompts, 7–8 own texts).

Identification regimes (query → gallery), each averaged over all choices of reference paragraphs:

| Regime | Query | Gallery per writer |
|---|---|---|
| R1 | shared paragraph | other 5 shared (ceiling reference) |
| R2 | own text (7, 8) | 6 shared |
| R3 | own text | k shared, k = 1, 2, 3 (all subsets averaged) |
| R4 | crop of own text | 1 or 6 shared full paragraphs |

R4 crops: horizontal strips of the text block covering 1/2, 1/4 and 1/8 of its lines (ink-row projection, as in the STROKE text-block crop), features re-extracted on each crop. Starting point: `scripts/run_crop_evaluation.py`.

Identity representations: STROKE, HC-183, frozen ResNet-18, EfficientNet-B0, ViT-B/16, Stroke-CNN (from `results/final42_2026-10-19/features/features.npz`; crops re-extracted). Matching: standardise, L2-normalise, cosine similarity to the mean reference vector of each writer (as in `writer_id_top1`).

## 5. Handedness evidence

- Model: HC-183 linear SVM, C = 0.1 (the PRL primary), trained **leave-one-writer-out**: the model scoring a query never saw that writer. Platt calibration fitted by inner cross-validation on the training writers only.
- Query evidence: per query unit (paragraph or crop); sensitivity: writer-level evidence averaged over both own texts.
- Sensitivity models: adjusted HC-183 (ink density and photo intensity regressed out, as in PRL), and STROKE handedness (writing-only).
- Scenarios:
  - S-oracle: query and gallery hand known (upper bound; forensic case where the suspect's hand is known).
  - S-query: gallery hand known (enrolment metadata), query hand predicted. **Primary.**
  - S-both: both predicted; gallery hand predicted from the writer's reference paragraphs with leave-one-writer-out models.

## 6. Fusion rules

1. Hard filter: rank only gallery writers of the predicted hand.
2. Calibrated Bayesian fusion (primary): P(w | x) ∝ softmax(s_w / T) · P(h_w | x). Temperature T fitted on shared → shared matching among the other writers only; no free weight, no tuning on queries.
3. Weighted log-linear fusion: log-softmax + λ·log P(h_w | x), λ chosen by inner cross-validation over writers (shown as secondary, to check that a tuned weight does not change the conclusion).
4. Re-ranking: handedness used only to reorder the top-k (k = 5) candidates.

## 7. Endpoints and statistics

- Primary endpoint: top-1 accuracy in R3 (k = 1) and R4 (1/4 crop), ResNet-18 and ViT-B/16 identity, S-query, Bayesian fusion vs. no handedness.
- Secondary: top-5, CMC curves, mean reciprocal rank, penetration rate at 95% hit rate.
- Inference: paired McNemar test over queries; 95% writer-cluster bootstrap intervals (2,000 resamples) for the difference; Holm correction over the primary cells only. All else descriptive.
- Rule carried over from PRL: every number in the manuscript comes from a macros file written by a script; nothing typed by hand.

## 8. Break-even analysis (core contribution)

- Analytic: hard filter accuracy ≈ a·Acc_oracle + (1 − a)·0, so it helps iff a > Acc_base / Acc_oracle. Report this threshold for every representation × regime.
- Simulation: replace HC-183 scores with synthetic calibrated scores of controlled accuracy a = 0.50, 0.55, …, 1.00 (Gaussian class-conditional scores with matching separation), keeping the real identity scores. Plot top-1 vs. a for each fusion rule; mark the real HC-183 accuracy and the break-even point.
- Explain why the oracle gain is small: report the share of baseline errors that cross hands.

## 9. Prevalence (RQ4)

Balanced data maximises the handedness information (1 bit). Real populations are far from balanced (about 1–2% left-handed students observed at SUST, about 10% worldwide). Resample galleries with 21 right-handed and 1–3 left-handed writers (about 5–12% left), 1,000 draws, and report penetration rate and top-1 for left and right queries separately. Expected: little gain for right-handed queries, a large gallery reduction for correctly predicted left-handed ones, and the cost of false "left" predictions.

## 10. Figures and tables (letter budget)

- Fig. 1: protocol (regimes, scenarios, fusion rules).
- Fig. 2: top-1 vs. difficulty (R1 → R4) for baseline, oracle, predicted.
- Fig. 3: break-even curves (top-1 vs. handedness accuracy) with the real accuracy marked.
- Fig. 4: prevalence: penetration rate for left and right queries.
- Table 1: top-1 / top-5 per representation × regime × scenario (Bayesian fusion).
- Table 2: fusion rule comparison and error share that crosses hands.

## 11. Code and outputs

- `scripts/wid/extract_crop_features.py`: strip crops and features for all representations.
- `scripts/wid/run_wid_handedness.py`: regimes, leave-one-writer-out handedness, fusion, statistics.
- `scripts/wid/run_breakeven_simulation.py`, `scripts/wid/run_prevalence.py`.
- `scripts/wid/make_wid_figures.py`, `scripts/wid/make_wid_macros.py`.
- Results in `results/wid_handedness_<date>/` with a provenance JSON, as for `final42_2026-10-19`.

## 12. Timeline (about 2 weeks)

| Days | Work |
|---|---|
| 1–2 | Freeze this protocol; leave-one-writer-out calibrated handedness scores; regimes R1–R3 |
| 3–5 | Strip crops and feature extraction (frozen CNN/ViT on CPU/MPS, STROKE, HC-183); R4 |
| 6–7 | Fusion rules, statistics, break-even simulation |
| 8 | Prevalence resampling |
| 9–10 | Figures, macros, tables |
| 11–14 | Manuscript draft, internal review |

## 13. Risks and how they are handled

- Predicted handedness never helps: expected for this classifier; the break-even curves and the oracle / prevalence results are the contribution, and the paper states the required accuracy.
- 42 writers, 84 own-text queries: differences of 1–3 queries are not significant; report intervals, keep the primary cells few, state the limit.
- Handedness and identity evidence are not independent (both from the same images; HC-183 is also an identity feature): include identity = STROKE / ViT with handedness = HC-183, and discuss the independence assumption of the Bayesian product.
- Stroke-darkness cue: repeat with the adjusted handedness model so the result does not rest on an acquisition-sensitive cue.
- Feasibility cells already seen: disclose; keep primary cells to analyses not yet run.
