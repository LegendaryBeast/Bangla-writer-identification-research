# Handedness for writer identification: first full run (2026-10-29)

Run: `scripts/wid/wid_features.py` (43 min, MPS) then `scripts/wid/run_wid_handedness.py` (9 min).
42 writers, 5,040 units (336 paragraphs plus 2, 4 and 8 strips each); 39 strips (0.8%) dropped for too little writing.
Plan: `manuscripts/wid_handedness/PLAN.md`.

## Handedness evidence (leave one writer out, `handedness_accuracy.csv`)

| model | unit accuracy, shared / own (n = 1) | writer level (mean over all units) |
|---|---|---|
| HC-183 | 68.7 / 65.5% | 73.8% |
| HC-183 adjusted | 56.7 / 61.9% | 61.9% |
| STROKE | 53.2 / 48.8% | 54.8% |

Strips give 62–67% (HC-183). STROKE is at chance under leave-one-writer-out.

## Main results (`id_summary.csv`, `id_paired_tests.csv`)

- **Primary tests (pre-specified): no effect.** Bayes fusion with predicted query hand changed top-1 by −1.5 to −0.3 points (ResNet-18 / ViT, own text with 1 reference, and 1/4 strips with 6 references); all p ≥ 0.37, Holm p = 1.
- **Known hand (oracle): +6.0 points on average**, significant in 106 of 120 cells (up to +15 for ViT on 1/8 strips).
- **Predicted query hand:**
  - Bayes fusion: −0.6 points on average; 0 of 120 cells improved.
  - Gallery filter: −14.4 points on average; worse in 89 of 120 cells.
  - Tuned weight (loglin): −0.2 points, because the weight is mostly fitted to about 0.
- **Identification errors are not concentrated in the same hand.** Cross-hand errors make up 44–53% of errors, close to the 51% expected at random. The identity features carry almost no handedness information, which is why knowing the hand helps.
- **Break-even (`breakeven_analytic.csv`, `breakeven_simulation.csv.gz`):**
  - A gallery filter needs handedness accuracy above 0.56–1.00 (median about 0.9); the real accuracy is 0.62–0.69.
  - Simulated independent, calibrated evidence: Bayes fusion gains at least +0.5 points from accuracy 0.60–0.90; at 0.65 the gain is ≤ +1 point; at 0.95 it is +1.5 to +7 points.
  - The real evidence did worse than the simulation at the same accuracy (−0.6 vs about +0.5). It is presumably less well calibrated, or not independent of the identity evidence.
- **Prevalence (`prevalence.csv`, 21 right-handed + 1–3 left-handed writers):**
  - Bayes fusion helps the left-handed minority slightly; see the last section.
  - The prior-corrected filter fails left-handed queries: top-1 2–15% vs 77–79% without it.
  - A model of about 66% accuracy cannot overcome a 5–12% prior.

## Predicted hand on both sides: +0.95 points (29 of 120 cells significant, none worse)

Control (`run_both_control.py`, `both_control_summary.csv`):

| Control | Mean gain |
|---|---|
| 20 random hand-balanced labels, calibrated | +0.03 |
| The same labels, scaled to the spread of the real log-odds | +0.21 (p = 0.048) |
| 100 random HC-183 directions | +0.19 (p = 0.010) |

All controls are below the handedness gain. The gain is therefore specific to the handedness direction, not a generic identity cue from any writer-consistent projection. It was found after the primary analyses (post hoc).

## Leakage control, confidence gating, filter law (`run_wid_extras.py`)

- **Enrolment-trained handedness** (cross-validated over paragraphs, own texts scored by the model fitted on all shared paragraphs): 82.1% on own texts vs 65.5% writer-independent.
- **Random hand-balanced labels under the same training:** 52.4–76.2% (mean 62.6%).
- **Random labels for unseen writers:** 33% (below chance), the known bias of leave-one-out on uninformative labels.
- **Identification gain with enrolment-trained evidence:** +1.1 points for handedness vs +0.1 for random labels.
- **Confidence gating** (top 30% most confident predictions): 75% accurate, top-1 change −8.7 to +0.4 points.
- **Filter law over 456 points** (settings × hand models): r = 0.996, mean absolute error 1.5 points, bias −0.9 points.
- **Prevalence nuance:** Bayes fusion raises top-1 of left-handed queries by +2.7 points on average (up to +14 for weak features) and lowers right-handed queries by at most 1.3 points.

## Other observations

- **Stroke-CNN collapses on strips:** 10% top-1 at n = 4 and 4% at n = 8 (chance is 2.4%). The binary tiles of short strips are too few or too sparse.
- **STROKE stays the strongest identity feature** (own text, 1/8 strip, 1 reference: 68.7%), so its ceiling leaves little room in most cells.

## Deviations from the plan

- Strips are equal parts of the text block, with each cut moved to the emptiest row within half a line pitch; this replaces line groups, because projection-based line detection was unreliable on slanted pages. 1/2, 1/4 and 1/8 strips hold about 3.5, 1.75 and 0.9 lines.
- Primary inference is a writer-level sign-flip permutation (9,999) instead of McNemar, because most cells average over reference subsets. McNemar is reported where a cell has one configuration.

## Manuscript

`manuscripts/wid_handedness/wid_manuscript.tex` (5 pages, compiled). Every number comes from `figures/wid_macros.tex`, written by `scripts/wid/make_wid_paper.py`.
