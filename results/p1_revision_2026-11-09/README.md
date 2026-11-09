# P1 revision analyses (2026-11-09)

These analyses respond to the P1 items of the simulated review in
`paper/notes/SIMULATED_REVIEW_2026-11-09.md`. None of them changes the
manuscript, the submission package or any archived results directory.
**All of them were run after the outcomes had been seen, so they must be
reported as post hoc.**

## Reproduction

```sh
python code/run_p1_precision_mechanism.py      # items 1, 2, 4 (saved outputs only, ~5 s)
python code/run_p1_csafe_adjusted_fix.py       # item 3 (refits CSAFE attribute models, ~35 s)
python code/check_p1_bangla_adjusted.py        # item 3, BanglaHand check (~5 s)
```

Built-in checks, all passing:
- The decomposition totals reproduce the archived oracle-minus-random differences (max error 4e-13 pts BanglaHand, 1e-14 pts CSAFE).
- The CSAFE rerun reproduces the archived unadjusted HC-183 ranks, decisions, margins and log-odds exactly.
- The original BanglaHand adjusted logits reproduce the archive to 2e-16.

Bootstrap: equal-hand-weight writer bootstrap, B = 10,000, seed 20261109. All intervals are conditional on the fitted models and gallery, as in the manuscript.

---

## 1. Precision of the primary fusion contrasts (`precision_equivalence.csv`)

No equivalence margin was declared in advance. The table therefore gives, for
each contrast, the **smallest margin Δ\* for which TOST (α = .05, 90% CI)
establishes equivalence**, the minimum detectable effect at 80% power, and the
largest gain the 90% interval leaves open as a share of the oracle gain.

| Contrast | Estimate | 90% CI | Δ\* | MDE80 | Max gain / oracle gain |
|---|---|---|---|---|---|
| BanglaHand, all 120 settings, nested − none | −0.08 | [−0.96, +0.67] | **0.96** | 1.39 | 11% |
| BanglaHand, all 120 settings, fixed − none | −0.57 | [−2.00, +0.70] | 2.00 | 2.31 | 12% |
| BanglaHand designated, nested − none (4 settings) | −0.89 to +1.19 | — | 1.79–2.58 | 2.32–3.46 | 14–45% |
| BanglaHand designated, fixed − none (4 settings) | −1.49 to +0.20 | — | 2.38–3.57 | 3.56–4.45 | 14–41% |
| CSAFE primary (8 contrasts) | −5.21 to +1.04 | — | 3.12–9.38 | 2.9–7.7 | — |

Interpretation:
- **BanglaHand, aggregate:** nested fusion is equivalent to identity-only within ±1 point, and the data exclude it recovering more than about 11% of the 6.1-point oracle gain. This is a real, positive statement, stronger than "no significant gain".
- **BanglaHand, designated settings:** equivalence holds only at about ±2–3.6 points.
- **CSAFE:** it cannot establish equivalence at any margin below 3 points. Across the eight primary contrasts only 29 writer decisions change (10 rescued, 19 damaged; `csafe_primary_discordant_counts.csv`, counts overlap across contrasts). One writer equals 1.04 points. ResNet-18/S2-WOZ nested fusion changes no decision at all (a degenerate interval, not estimated equivalence).

**Suggested rewording** (abstract l.21, §5 l.544–547, §7 l.653–655): replace "reproduce the distinction" with wording such as: *"CSAFE shows the same direction (no primary contrast favours fusion; 10 rescued versus 19 damaged decisions) but its 96-writer gallery cannot establish equivalence within ±3 points."* Add one sentence to §4.1: *"In a post hoc precision analysis, the 90% interval for nested fusion across all settings (−0.96 to +0.67 points) excludes gains above 11% of the oracle opportunity."*

## 2. Why known handedness is no better than a random partition (`mechanism_*.csv`)

For a query outranked by j impostors, a known partition rescues it only if all j impostors have the opposite hand. A random balanced partition does so with probability C(21, j)/C(41, j). The −0.78-point BanglaHand gap (known − random, mean over 120 settings) decomposes exactly as follows:

| j | Gap contribution (pts) |
|---|---|
| 1 | −0.65 |
| 2 | −0.23 |
| 3 | −0.06 |
| ≥ 4 | +0.16 |

When exactly one impostor outranks the true writer (about 10% of queries per representation), that impostor has the **opposite** hand in:

| Representation | Opposite-hand share [95% CI] | Random reference |
|---|---|---|
| ResNet-18 | **34.2%** [26.0, 43.3] | 51.2% |
| ViT-B/16 | **36.6%** [27.7, 46.4] | 51.2% |
| HC-183 | 41.3% [32.5, 50.0] | 51.2% |
| EffNet-B0 | 47.3% [39.7, 54.9] | 51.2% |
| STROKE | 53.2% [37.0, 68.4] | 51.2% |
| Stroke-CNN | 62.8% [51.0, 73.3] | 51.2% |

The neural matchers' errors cluster within hand. A correct hand label therefore removes fewer real competitors than an arbitrary split does. This is the redundancy the paper discusses but does not demonstrate: the identity features already carry handedness-correlated information.

CSAFE: the pooled neural j = 1 opposite-hand share is 40.0% [27.0, 54.1] versus 50.5%, from only 40 decisions. ViT-B/16 shows the gap in all three conditions (−3.0 to −4.5 pts); ResNet-18 and HC-183 are mixed. The direction is consistent, but CSAFE alone is not conclusive.

Caveat to state: in BanglaHand, same-hand clustering could partly reflect acquisition differences between the two groups (the companion study's photometric cues), not only writing style. CSAFE, where hand is not tied to acquisition, gives weak directional support for style.

**Suggested placement:** add this table, or a compact version, to §4.3 after the random-partition sentence (l.418–423), and add one abstract sentence: *"Identity errors concentrate within handedness groups, so correct handedness removes fewer competitors than a random partition."*

## 3. CSAFE adjusted-HC-183 failure: a numerical bug, now fixed (`csafe_adjusted_fix/`)

**Diagnosis.** The reported "negative Platt slopes in 24/96 outer fits" were in fact slopes of about −1e-17, i.e. calibration collapsed to a constant. There were two rounding paths in the nuisance regression `run_wid_handedness.residualise`, both specific to CSAFE scans:
1. The grayscale q50/q75/q95 covariates are exactly 255 on every page, and q25 is 255 on 98% of pages. The design matrix is rank deficient.
2. HC-183 column 178 (a connected-component area quantile) is 4 on all training pages except two (`w0562_s01_pWOZ_r02`, `w0632_s01_pWOZ_r03`). After residualising, its training SD is 4.8e-16 instead of exactly 0, and StandardScaler divided by it.

Out-of-fold SVM scores for those two pages reached about 2.4e15, with opposite labels. That flattened the Platt slope to 0, so the calibrated logit became the training prior. Under leave-one-writer-out training that prior always points to the *opposite* of the held-out writer's hand, which produced the 34% below-chance accuracy even though the raw AUC was 0.73. The L-BFGS convergence warnings that motivated the v1.1 calibration amendment (Supporting S13.3) were symptoms of these extreme inputs.

**Fix.** `residualise_stable` standardises the covariates, drops zero-variance covariates, and keeps exactly constant feature columns at exactly zero training residual. In exact arithmetic this is the same projection, so the estimator is unchanged.

| CSAFE adjusted HC-183 | Accuracy, original → corrected | Calibrated AUC, original → corrected |
|---|---|---|
| S2 LND | 34.4% → **68.8%** | 0.50 → 0.71 |
| S2 WOZ | 34.4% → **61.5%** | 0.50 → 0.68 |
| S3 WOZ | 35.4% → **64.6%** | 0.53 → 0.70 |

After the fix all 96 outer Platt slopes are positive (minimum 0.157). With the corrected model, fixed/nested fusion still shows no gain for ResNet-18/ViT-B/16 (changes −5.2 to +2.1 pts), and hard filtering costs 18–24 pts instead of 35–42 pts. The sensitivity model now **agrees** with the primary model.

**Unaffected:**
- The 8 primary CSAFE contrasts, which use unadjusted HC-183 and reproduce exactly. The rerun's Holm p-values differ slightly from the archive only because it runs fewer tests, so the sign-flip draws occur in a different order. Keep the archived p-values.
- BanglaHand (`bangla_adjusted_numerical_check.json`): its covariates are non-degenerate, logits change by at most 4e-4, no predicted hand flips, and accuracy stays at 61.9%.

**Manuscript changes needed:**
- Replace l.528–533 ("adjusted HC-183 attains 34.38% … sensitivity failure") with the corrected values.
- Update Supporting S13.3/S13.4 and Tables S17/S18.
- Record a post hoc v1.2 implementation amendment, made after the performance tables were seen, that affects only the secondary adjusted model.

Suggested text: *"A post hoc audit found that the original nuisance regression was numerically unstable on CSAFE scans, whose upper grayscale quantiles are constant. A numerically stable implementation of the same projection gives 68.8%/61.5% accuracy and no fusion gain, consistent with the primary model."*

## 4. Prevalence: kept balanced, split by query hand (`csafe_by_query_hand.csv`)

At the author's direction, no further CSAFE images were downloaded; the balanced 48/48 design is retained. The existing balanced run is split by query hand instead (48 writers per hand; 1 writer = 2.08 pts):

| S2 query | Matcher | Known gain R / L | Fixed change R / L | Nested change R / L |
|---|---|---|---|---|
| LND | ResNet-18 | +12.5 / +2.1 | +4.2 / −2.1 | +4.2 / −2.1 |
| LND | ViT-B/16 | +4.2 / +2.1 | −2.1 / −2.1 | −2.1 / −2.1 |
| WOZ | ResNet-18 | +8.3 / +4.2 | −2.1 / −8.3 | 0.0 / 0.0 |
| WOZ | ViT-B/16 | +8.3 / +4.2 | −2.1 / 0.0 | −2.1 / 0.0 |

Neither query hand shows a consistent fusion gain. Each cell rests on a handful of writers, so these are descriptive only.

**Suggested response-letter justification for the balanced design:**
- Balance prevents majority-class dominance of the attribute predictor and gives the oracle and random-partition benchmarks a symmetric reference (Eq. 5).
- With a skewed prior, the oracle benefit concentrates in the minority hand and depends on the chosen prevalence, which is not fixed across applications.
- Prevalence effects are characterised separately in Supporting S7 (synthetic) and can be read from the class-specific results.

A within-sample imbalanced-gallery variant (all 48 right-handed writers plus a small random subset of left-handed writers) could be added without any download, if a referee asks.
