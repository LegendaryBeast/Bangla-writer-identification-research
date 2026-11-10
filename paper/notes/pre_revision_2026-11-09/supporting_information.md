---
title: "Supporting information"
author:
  - "Tanzim Hasan Prappo"
  - "M. Shahidur Rahman"
date: ""
bibliography: refs.bib
csl: ieee.csl
link-citations: true
reference-section-title: References
---

Supporting information for **Handedness for Bangla writer identification: Oracle gains and predicted-attribute utility**.

This document accompanies the IET Biometrics manuscript. The authoritative numerical outputs are in `results/wid_revision_2026-11-08/`. Identification uses selected-reference normalization and equal writer weights. The primary BanglaHand analyses are retrospective and were not preregistered. Section S13 documents the separately specified CSAFE extension and its numerical implementation amendment.

# S1 Data units, exclusion and weighting {#data-units-exclusion-and-weighting}

BanglaHand provides 336 photographs from 42 writers, 21 in each handedness group, with eight paragraphs per writer. Paragraphs 1--6 use shared prompts; 7--8 are self-chosen texts. The released metadata links every photograph to a pseudonymous writer and condition. All analyses use the released photographs.

Each photograph supplies a full-paragraph unit ($n=1$) and $n=2,4,8$ horizontal strips, giving 5,040 candidate units. At 1,536-pixel image width, rows with ink fraction exceeding 0.01 define a text-row region through its 0.5th and 99.5th percentiles. For $n>1$, nominal equally spaced cuts are moved to a minimum in the smoothed row-ink profile. The smoothing standard deviation is 96/5 pixels; the search half-width is the smaller of 48 pixels and the text-block height divided by $3n$, with a minimum of three pixels. Row fractions transfer these boundaries to the other image scales. Whole-paragraph units use the complete image. Strips preserve the photograph's full width before representation-specific cropping.

A unit is invalid if the cropped high-resolution ink mask has fewer than 200 ink pixels, the 768-pixel mask has fewer than 50 ink pixels, contour/skeleton features cannot be computed, or their values are nonfinite. All representations share this validity mask. Exactly 39 strips are invalid; no paragraph is excluded. Table S1 gives their distribution. Images flagged for bottom-edge truncation in the dataset are retained; this analysis does not claim immunity to truncation.

    $n$   Hand Text       Candidate   Excluded   Retained
  ----- ------ -------- ----------- ---------- ----------
      1      R shared           126          0        126
      1      R own               42          0         42
      1      L shared           126          0        126
      1      L own               42          0         42
      2      R shared           252          0        252
      2      R own               84          0         84
      2      L shared           252          0        252
      2      L own               84          0         84
      4      R shared           504          0        504
      4      R own              168          0        168
      4      L shared           504          0        504
      4      L own              168          1        167
      8      R shared          1008          6       1002
      8      R own              336         16        320
      8      L shared          1008          2       1006
      8      L own              336         14        322

  : **Table S1.** Candidate and excluded units by strip count, text condition and handedness. Excluded units are omitted from all representations.

For a setting, the outcome is averaged across reference configurations for each valid query unit, then across units within its writer, then equally across the 42 writers. Each writer contributes equally even if strip exclusions leave different numbers of units. Across-setting summaries average the 120 setting-specific accuracies equally; they are not a pooled per-strip accuracy. Rows of the saved rank file contain the corresponding normalized analysis weight.

# S2 Self-contained feature specifications {#self-contained-feature-specifications}

All extraction is implemented in `code/wid_features.py`, `hc183.py` and `stroke_features.py`. Feature matrices are supplied so that the identification and control analyses can be reproduced without rerunning neural inference.

**HC-183.** A photograph is converted to grayscale and resized to width 768 with aspect ratio preserved using Lanczos interpolation. For grayscale intensity $I$ on the 0--255 scale, a Gaussian background estimate $B$ with standard deviation 17 pixels defines ink by $(B-I>17)\land(B>120)\land(I<205)$. A four-pixel border is cleared and a $2\times2$ binary opening is applied. Gradients of the mask smoothed with standard deviation 0.7 give nine-bin unsigned orientation histograms and ink density in each of $4\times4$ cells (160 values). Row and column ink projections each contribute seven quantiles (0, .1, .25, .5, .75, .9, 1), mean and standard deviation (18 values). Five area quantiles (.1, .25, .5, .75, .9) of connected components with at least three pixels complete the 183 features.

**STROKE.** The corresponding ink rule is evaluated at width 1,536, with background smoothing standard deviation 34, an eight-pixel cleared border and removal of connected objects smaller than 12 pixels. The text block is cropped using row and column ink fractions (thresholds 0.01 and 0.005), their 0.5th/99.5th percentiles and ten pixels of padding. Stroke width is twice the mean distance-transform value on the skeleton. A contour sampling length $L=\max(4,\operatorname{round}(2\,\mathrm{width}))$ controls feature scale. The 208 components comprise a 16-bin contour-direction histogram, a $12\times12$ hinge histogram, 12 turning-angle bins, 12 endpoint-direction bins, 12 skeleton-orientation bins, nine slant bins, and endpoint, junction and hole counts normalized by skeleton length. Static contours do not reveal pen movement order. The mask discards grayscale information, but this alone does not prove that every acquisition effect is removed.

**Frozen grayscale embeddings.** Grayscale images are resized to width 1,568 using Lanczos interpolation; the tile-selection mask is the nearest-neighbor resize of the 768-pixel HC-183 mask. Nonoverlapping $224\times224$ tiles with at least 1% ink are retained; the bottom edge is padded with median grayscale intensity. If no tile meets the threshold, all tiles are used. Grayscale is repeated into three channels and normalized by ImageNet means $(.485,.456,.406)$ and standard deviations $(.229,.224,.225)$. The checkpoints are torchvision `ResNet18_Weights.IMAGENET1K_V1`, `EfficientNet_B0_Weights.IMAGENET1K_V1` and `ViT_B_16_Weights.IMAGENET1K_V1`; final classification layers are removed. Mean tile embeddings have dimensions 512, 1,280 and 768. Models remain in evaluation mode without fine-tuning.

**Stroke-CNN.** The frozen ResNet-18 above instead receives black-on-white binary tiles from the cropped high-resolution mask. Bilinear resizing targets a stroke width of three pixels, with image dimensions at least 224 pixels; interpolation can produce antialiased edges. White padding completes the tile grid, and tiles require at least 2% ink; if none qualify, all are retained. Input normalization and mean aggregation follow the grayscale embeddings. This representation has 512 components. Its name does not indicate a separately trained handwriting network.

**Adjusted HC-183.** Seven grayscale statistics (mean, standard deviation and .05/.25/.5/.75/.95 quantiles), together with mask ink density, form nuisance covariates. A linear regression with intercept removes their fitted contribution from HC-183. Regression coefficients are fitted within each outer or inner SVM training split, then applied to its training and held-out samples. The adjusted representation is a sensitivity check, not a guarantee of causal removal of acquisition confounding.

# S3 Information available at every fitting stage {#information-available-at-every-fitting-stage}

The identification gallery is closed-set and includes the true query writer. This is separate from holding that writer out of *attribute-model* training. Table S2 distinguishes these information sources.

**Evaluation versus deployment.** The 42 outer fits implement cross-validation, with the known writer identifier used to assign samples to a held-out fold. They do not constitute an inference algorithm that can select a model by an unknown query identity. A deployable unseen-writer pipeline would fit one attribute model and calibrator on development writers, select temperature and fusion weight using separate development predictions, freeze them, and apply the same fitted rule to every future query. Gallery templates and permitted gallery metadata would be formed at enrolment. This study estimates held-out-writer performance with the available cohort and does not validate such a frozen pipeline on a new cohort. Refitting with enrolled-writer pages changes the information regime and requires a separate enrolled-writer evaluation. The exploratory both-sided analysis in S6 uses that broader regime and is not a substitute for external validation.

  Component                               Permitted fitting data
  --------------------------------------- -------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Image descriptors                       Fixed extraction rules and frozen ImageNet checkpoints; no writer-specific network training.
  ID standardization                      Only whole paragraphs selected as gallery references in the current configuration, from all enrolled writers. No queried paragraph or unused reference paragraph.
  ID templates                            Selected standardized, normalized references of the relevant gallery writer.
  Outer attribute model for writer $w$    Valid shared units of the same size from the other 41 writers. Neither shared nor self-chosen units of $w$.
  Inner attribute preprocessing and SVM   Training writers in each of five stratified group folds inside the outer 41-writer set. Nuisance regression, if used, is refitted here.
  Platt calibrator                        Out-of-fold SVM scores and labels of the outer 41 training writers. The query writer is absent.
  ID temperature for queries of $w$       ID-score negative log loss on query writers other than $w$ in the same setting, equally weighted by writer. No attribute probabilities enter temperature fitting.
  Primary fusion                          Fixed unit weight on log attribute probability; no fitted handedness weight.

  : **Table S2.** Fitting hierarchy for the revised primary analysis. The query writer remains enrolled for identity matching, but contributes no samples to its query-handedness model.

The SVM uses scikit-learn `SVC(kernel="linear", C=0.1)` after training-set standardization. Inner calibration uses five shuffled stratified group folds (seed 2031). A logistic regression calibrator with $C=10^4$ is fitted to the out-of-fold raw scores. Thus calibration is estimated without the outer writer, but it is not guaranteed to be accurate for a new population. The 41-writer folds are nearly, rather than exactly, balanced by hand. For strips, retained unit counts can differ; the original attribute fitting uses unit-level training observations. Evaluation gives writers equal weight.

For a full-paragraph query, gallery templates use other whole paragraphs. For strips, the gallery still uses whole paragraphs. Cosine ties are resolved by fixed lexicographic gallery-writer order. The revised run found no exact ties at the maximum baseline score. The hard-filter implementation adds ten to matching-hand cosine scores, which is sufficient because cosine lies in $[-1,1]$; incompatible writers remain lower in the complete ranking rather than being physically deleted. For top-1, this is equivalent to exclusion in the present nonempty binary groups.

Temperature $T$ uses the grid $10^{\operatorname{linspace}(-3,0,31)}$. Minimizing mean ID negative log loss excludes the scored writer's query outcomes and uses equal mean loss per other writer. It may use references of the scored writer as impostor-gallery entries, as permitted by the declared closed-set protocol. The nested learned-weight comparison in Section S8 regenerates development attribute scores inside a second outer exclusion. Primary identification outcomes use the stated selected-reference normalization. Attribute-prediction controls with writer overlap are reported separately in Section S5.

# S4 Filtering identity and known random partitions {#filtering-identity-and-known-random-partitions}

Let a random query be sampled according to the setting's writer-balanced weights. Define $C$ as correct query-handedness prediction and $K$ as correct top-1 with the true hand supplied. Because an incorrect binary decision puts the true writer below every member of a nonempty opposite-hand group,

$$\{\text{filtered success}\}=C\cap K.$$

Consequently, $A_{\mathrm{filter}}=aP(K\mid C)$ exactly, while $aA_{\mathrm{known}}$ requires zero covariance between the two correctness indicators. Cross-hand *baseline identification* errors concern a different relationship and cannot establish this assumption.

The evaluated prediction models are HC-183, adjusted HC-183 and STROKE across all 120 settings, plus writer-level HC-183 on the 96 self-chosen-text settings. This gives $3\times120+96=456$ combinations. Writer-level HC-183 averages log-odds over both self-chosen paragraphs or their strips before predicting. It assumes those documents are already linked to the same questioned writer and uses more writing than a single query; it is a multi-document sensitivity analysis, not a deployable single-query result. The combination count is an analysis grid, not the number of independent experimental replications.

Known random-partition controls uniformly assign 21 writers to each of two groups, supplying the true query membership as well as gallery memberships. The same 2,000 partitions are applied across settings. For a query with $j$ baseline impostors outranking the true writer, every one must lie in the 21-member opposite group for filtered success. Its exact expected success under a uniform partition is

$$P(\text{success}\mid j)=
 \begin{cases}
 \binom{21}{j}/\binom{41}{j},&0\leq j\leq21,\\
 0,&j>21.
 \end{cases}\qquad\text{(S1)}$$

This also accounts for deterministic tie precedence if required. The exact expectation is compared with Monte Carlo partitions; the partition quantiles describe variation among partitions, not a confidence interval for a new population. A writer-bootstrap interval for handedness minus the exact partition expectation instead conditions on the fitted scores and observed gallery.

  Contrast                                     Mean      95% interval   Sign-flip $p$
  ----------------------------------------- ------- ----------------- ---------------
  Known hand vs none                           +6.1    \[+5.1, +7.1\]          0.0001
  Predicted-query fusion vs none               -0.6    \[-2.3, +0.9\]          0.5402
  Predicted-query filter vs none              -14.3   \[-19.3, -9.5\]          0.0001
  Expected random partition vs none            +6.9    \[+6.4, +7.4\]          0.0001
  Known hand vs expected random partition      -0.8    \[-1.8, +0.2\]          0.1795

  : **Table S3.** Across-setting mean changes in top-1, percentage points. Intervals resample writers with common indices across settings, conditional on the observed gallery and fitted evidence.

  Prediction model             Bias \[95% CI\]     MAE \[95% CI\]   Pearson $r$ \[95% CI\]
  ---------------------- --------------------- ------------------ ------------------------
  All combinations         -0.9 \[-2.0, +0.1\]   1.5 \[1.3, 2.5\]   0.996 \[0.988, 0.997\]
  HC-183                   -0.6 \[-1.8, +0.5\]   1.3 \[1.0, 2.2\]   0.996 \[0.989, 0.998\]
  HC-183 adjusted          -0.9 \[-2.1, +0.3\]   1.4 \[1.1, 2.4\]   0.995 \[0.984, 0.997\]
  HC-183, linked texts     -0.6 \[-2.1, +0.8\]   1.2 \[1.1, 2.6\]   0.998 \[0.989, 0.998\]
  STROKE                   -1.7 \[-3.1, -0.1\]   1.9 \[1.4, 3.3\]   0.991 \[0.974, 0.994\]

  : **Table S4.** Independence-product diagnostics, with conditional writer-bootstrap intervals. Bias and mean absolute error (MAE) are percentage points. Correlation pools the stated combinations; these are not independent replications.

Cross-hand baseline errors concern the hand of the top-ranked incorrect writer, rather than the correctness-event independence in the filtering identity. Table S5 reports the ratio of writer-balanced cross-hand error to all error, averaged across settings before taking the ratio. Some representations show greater same-hand confusability than the uniform-impostor reference. Proximity to that reference does not establish independence.

  Representation     Cross-hand error share \[95% CI\], %
  ---------------- --------------------------------------
  STROKE                              55.4 \[43.6, 66.0\]
  HC-183                              42.3 \[36.5, 48.8\]
  ResNet-18                           43.8 \[38.9, 48.4\]
  EffNet-B0                           47.4 \[41.8, 53.4\]
  ViT-B/16                            44.7 \[38.4, 51.0\]
  Stroke-CNN                          52.4 \[48.0, 57.1\]

  : **Table S5.** Share of baseline errors assigned to the opposite hand, with conditional 95% writer-bootstrap intervals. A uniformly selected wrong writer would have opposite hand with probability $21/41=51.2\%$.

# S5 Calibration and writer-associated label controls {#calibration-and-writer-associated-label-controls}

Attribute calibration is evaluated from the outer held-writer-out predictions, separately by unit size and shared versus self-chosen text. Brier score is the writer-balanced mean of $(p-y)^2$; negative log loss is the mean of $\log(1+e^\ell)-y\ell$, evaluated stably. The uninformative balanced prediction has Brier score 0.25 and log loss $\log2$. Reliability plots use five fixed equal-width probability bins. Bin-level writer-bootstrap intervals account for writer clustering; sparse-bin uncertainty should not be interpreted as proof of good or poor calibration.

  Model               Accuracy \[95% CI\], %   Brier score   Negative log loss
  ----------------- ------------------------ ------------- -------------------
  HC-183                 65.5 \[53.6, 77.4\]         0.204               0.602
  HC-183 adjusted        61.9 \[50.0, 73.8\]         0.244               0.684
  STROKE                 48.8 \[35.7, 61.9\]         0.286               0.775

  : **Table S6.** Writer-held-out prediction and proper scoring rules on self-chosen paragraph queries. Intervals are conditional writer bootstraps.

![**Figure S1.** Held-writer-out HC-183 calibration on paragraph queries. Five probability bins; vertical bars are conditional 95% writer-bootstrap intervals. Bin counts and the underlying probabilities are supplied with the results.](assets/calibration.png){width="6.65in"}

The enrolment-trained label control fits the same predictor on all shared units and evaluates self-chosen texts. Shared-text outputs in the writer-overlap control use five-fold splits of paragraphs with writer overlap and an in-sample fit of the score calibrator; they are not used here to claim unbiased calibration. The clean comparison reported in the main text concerns self-chosen paragraphs excluded from training.

For 20 random 21/21 labelings, group 1 includes alternately 10 or 11 left-handed writers, and 11 or 10 right-handed writers. This makes random labels nearly unrelated to true hand while retaining equal group sizes. Figure S2 shows the random-label distribution as a diagnostic of writer-associated information. Fixed balanced labels can induce below-50% leave-one-writer-out performance because the held-out label and training-set class composition are dependent. No nominal-chance test is used to interpret this structured null. Predictability of arbitrary labels establishes available writer-associated structure, not what fraction of true-handedness accuracy has that source.

![**Figure S2.** Writer-overlap diagnostic. (a) Schematic split designs: each writer has one fixed random label; training can exclude that writer or contain their shared paragraphs before testing their own text. (b) Own-paragraph prediction of genuine handedness and 20 random writer labelings. The 50% line is descriptive, not a test against the structured held-out null. Gray points denote labelings. This control does not establish leakage in the primary writer-held-out experiment.](assets/label_controls.png){width="6.65in"}

# S6 Both-predicted descriptor agreement {#both-predicted-descriptor-agreement}

For exploratory both-sided prediction, we report only the 96 self-chosen-text settings. Shared-text queries would also enter the fitting data of impostor gallery-writer models, so their both-sided results are excluded from this reported comparison. Query log-odds are produced as above. A gallery writer's reference log-odds are averaged over its selected full paragraphs, using the attribute model that excluded that gallery writer. If $p_q$ and $p_w$ are the resulting probabilities, the agreement term is

$$M(q,w)=\log\{p_qp_w+(1-p_q)(1-p_w)\},\qquad
 f_w=s_w/T+M(q,w).\qquad\text{(S2)}$$

This is a descriptor-agreement score. It is not established as an identity posterior or evidence attributable specifically to physiological handedness. In particular, the model scoring an impostor gallery writer can have been trained on shared samples of the query writer. The procedure is therefore an enrolled-cohort, writer-cross-fitted diagnostic, not external prediction by a single attribute model trained without the entire queried cohort.

Controls use 20 random-label predictors with the same writer-level cross-fitting, those same scores centered and scaled per unit size to match the real score standard deviation, and 100 random HC-183 directions. Spread matching uses all valid cohort scores, including queried units; it is explicitly a transductive diagnostic matching procedure rather than deployable calibration. It neither matches every aspect of model complexity nor establishes a causal mechanism.

The revised both-predicted score changes top-1 by +1.08 points on average (95% conditional interval \[+0.50, +1.72\]). The controls use selected-reference ID normalization and save writer-level outcomes. Table S7 reports both the empirical control-instance comparison and a direct paired writer comparison of the handedness score with the mean control. These are different uncertainty statements: the empirical exceedance fraction describes only the supplied control generation, while the bootstrap conditions on that ensemble and fitted gallery. The true handedness labeling is outside the support of the nearly hand-balanced random-label controls, so the exceedance fraction is not an exchangeable permutation-test $p$-value. Positive descriptor-consistency gains do not contradict the primary predicted-query result because the available gallery information and score rule differ.

  Control                   Instances   Control gain   Hand minus control \[95% CI\]   $e_{\mathrm{ctrl}}$
  ----------------------- ----------- -------------- ------------------------------- ---------------------
  Random labels                    20           +0.1             +1.0 \[+0.4, +1.6\]                 0.048
  Random labels, scaled            20           +0.4             +0.7 \[+0.1, +1.3\]                 0.048
  Random directions               100           +0.2             +0.9 \[+0.3, +1.5\]                 0.010

  : **Table S7.** Exploratory both-sided fusion versus controls. All gains and differences are percentage points. Paired intervals compare handedness with the mean control; $e_{\mathrm{ctrl}}$ is the share of generated control instances at least as good as handedness, with the plus-one correction.

# S7 Synthetic evidence and prevalence sensitivity {#synthetic-evidence-and-prevalence-sensitivity}

For a target attribute accuracy $a\in\{.50,.55,\ldots,1.00\}$ and $a<1$, Gaussian scores obey

$$z\mid h\sim N\big((2h-1)d/2,1\big),\quad
 d=2\Phi^{-1}(a),\quad\ell=dz.$$

These are calibrated log-odds under the synthetic equal class prior. Noise is generated once for each distinct query unit in a draw and reused across all its reference configurations. The evidence is independent of fixed identification scores conditional on hand. At $a=.50$, fusion receives zero log-odds and filter ties are resolved by independent fair random decisions. At $a=1$, results are set to the exact oracle limit, avoiding finite substitutes for infinite separation. Two hundred draws are used per target accuracy and setting. ID temperature is estimated once from the same ID-only calibration procedure, not tuned to simulation outcomes.

Synthetic results demonstrate behavior under a chosen evidence family; they do not identify an accuracy threshold for arbitrary real predictors. Equal attribute accuracy does not fix confidence, calibration or the allocation of mistakes across identification margins. The complete simulation results are supplied and Fig. S3 shows the self-chosen paragraph, one-reference condition.

![**Figure S3.** Simulated independent calibrated evidence for self-chosen paragraphs with one reference. Curves average 200 draws; shaded ranges describe the spread over representations, not a sampling confidence interval. The exact oracle limit is used at 100% accuracy.](assets/simulation.png){width="6.65in"}

Prevalence experiments use 1,000 draws per condition, selecting 1, 2 or 3 of the 21 left-handed writers together with all 21 right-handed writers. Only queries from selected writers are retained, and normalization uses selected reference paragraphs of the selected gallery. ID-only temperatures fitted on the full 42-writer cohort, excluding each scored writer's query outcomes, remain fixed; they are not recalibrated in every reduced gallery. These perturbations reuse the cohort; they do not simulate recruitment from a new population. The original approximately balanced-source attribute log-odds remain unchanged for fusion and the ordinary filter. Only the prior-decision filter adds $\log(m/21)$ to query log-odds, where $m$ is the selected number of left-handed writers. This is a decision-rule sensitivity analysis, not a claim of exact posterior recalibration under label shift. Class-specific results are reported separately; a population-weighted utility would require a specified query prior and error-cost function. No casework recommendation or universal minority benefit is inferred from this sensitivity analysis.

# S8 Properly nested fusion-weight selection {#properly-nested-fusion-weight-selection}

The fixed score rule assigns unit weight to log handedness evidence. To test a more flexible rule, we select a nonnegative weight by development negative log loss while retaining the same gallery, identity scores, outer query predictions and ID-only temperature. For outer writer $u$, the score is

$$f_{qj}=s_{qj}/T_u+\lambda_u\log p_q(h_j),\qquad
 \Lambda=\{0,.1,.2,.3,.5,.7,1,1.5,2,3,5\}.$$

The development objective is

$$\lambda_u=\underset{\lambda\in\Lambda}{\arg\min}\;
 \frac1{41}\sum_{v\ne u}\overline{\mathrm{NLL}}_v
 \big(T_u,\lambda;\ell^{(-u,-F(v))}\big),$$

where the bar denotes the same within-writer unit/configuration averaging used for primary accuracy, and $F(v)$ is the development writer fold containing $v$. Exact objective ties select the smaller weight. The grid and objective were fixed before fitting the learned-weight comparison, without choosing a rule from its test results. This comparison is exploratory and was not preregistered.

For each outer exclusion, the remaining 41 writers are split into five stratified writer folds (seed 2031). The HC-183 SVM generating each development fold's probabilities trains on only the remaining 32 or 33 writers. Its standardization and an additional five-fold writer-group Platt calibration are fitted entirely inside that training subset. Thus the outer writer and every development validation writer are absent from all relevant attribute fits. The final attribute predictor used on outer query $u$ is the existing model trained and calibrated on the other 41 writers. Development models therefore have fewer training writers than that final outer predictor, a normal finite-sample cross-validation distinction.

The temperature is the same ID-only $T_u$ as for fixed fusion; it is not retuned against outer-writer outcomes or attribute probabilities. All 42 gallery templates and known gallery hands remain available as candidate metadata, including writer $u$ as an impostor for development queries. This matches the declared closed-gallery information set. It does not expose the questioned samples or query targets of $u$ to development fitting. The evaluation estimates a writer-held-out calibration procedure conditional on this gallery; it is not a prospective validation of one universal fitted temperature/weight on unseen casework.

The values $\lambda=0$ and $\lambda=1$ recover baseline ranking and fixed fusion respectively. Automated checks verify these comparator outcomes against the separately generated primary results. Development loss is no larger than either comparator's loss by construction; held-out performance need not improve. The objective is log loss rather than top-1 accuracy. An equivalent two-hand log-sum-exp calculation is checked against the full 42-candidate loss, and changing outer-query targets or their development-score placeholders does not change selected weights. Fold manifests document 840 development models and their inner calibration exclusions.

  Metric           Contrast                         Mean \[95% CI\]   Sign-flip $p$
  ---------------- -------------------- --------------------------- ---------------
  Top-1 (points)   Nested minus none         -0.08 \[-1.13, +0.81\]          0.8784
  Top-1 (points)   Nested minus fixed        +0.49 \[-0.15, +1.28\]          0.2205
  NLL (nats)       Nested minus none      -0.010 \[-0.041, +0.024\]          0.5418
  NLL (nats)       Nested minus fixed     -0.022 \[-0.054, +0.006\]          0.1784

  : **Table S8.** Nested learned-weight contrasts over all 120 settings. Positive top-1 differences favor the learned rule; negative NLL differences favor it. All intervals are conditional 95% writer bootstraps, with common resampled writer indices across settings.

The learned-rule comparisons against no fusion and fixed fusion in the four designated settings form one eight-comparison Holm family. These tests, and their uncertainty, are reported in the main article. The paired bootstrap resamples fitted writer outcomes rather than refitting the complete nested procedure; it therefore does not quantify all uncertainty from data-dependent parameter selection. Full writer-level outcomes and validation losses are supplied for all settings.

  Representation       Median \[IQR\]   $\lambda=0$, %   $\lambda=1$, %      Range
  ---------------- ------------------ ---------------- ---------------- ----------
  STROKE             0.7 \[0.3, 1.0\]              1.5             24.3   0.0--5.0
  HC-183             0.3 \[0.2, 0.5\]              2.6              1.9   0.0--1.0
  ResNet-18          0.5 \[0.3, 0.7\]              2.9             13.2   0.0--1.5
  EffNet-B0          0.5 \[0.5, 0.7\]              0.4             20.2   0.0--1.5
  ViT-B/16           0.5 \[0.5, 0.7\]              0.2             16.7   0.0--1.5
  Stroke-CNN         0.5 \[0.5, 1.0\]              0.0             22.5   0.1--1.5

  : **Table S9.** Selected weight distributions across 42 outer writers and 20 settings per representation. Frequencies describe correlated writer/setting fits, not independent experimental repetitions. IQR: interquartile range.

# S9 Uncertainty and within-cohort robustness diagnostics {#uncertainty-and-within-cohort-robustness-diagnostics}

Figure S4 summarizes the within-cohort parameter and prediction diagnostics.

## S9.1 Temperature selection and loss sensitivity {#temperature-selection-and-loss-sensitivity}

For every setting and held-out writer, we retain the writer-balanced development ID loss at all 31 candidate temperatures, the selected value, the gap to the second-best value, and the number of temperatures within 0.01 nats of the minimum. This tolerance is a descriptive measure of a loss plateau, not a confidence level or a threshold for practical importance. Table S10 reports boundary frequencies and observed ranges. Boundary selection alone does not establish that the optimum is misspecified, particularly when several small temperatures give nearly identical loss.

  Representation     Median $T$            Range   Lower, %   Upper, %   Near-optimal
  ---------------- ------------ ---------------- ---------- ---------- --------------
  STROKE                 0.0501   0.0010--0.0794        5.0        0.0              1
  HC-183                 0.0794   0.0398--0.1259        0.0        0.0              1
  ResNet-18              0.0794   0.0398--0.1585        0.0        0.0              1
  EffNet-B0              0.0794   0.0398--0.1995        0.0        0.0              1
  ViT-B/16               0.0794   0.0398--0.1995        0.0        0.0              1
  Stroke-CNN             0.1885   0.0398--1.0000        0.0       15.8              2

  : **Table S10.** ID-only temperature diagnostics. Lower/upper: percentage of fits selecting $10^{-3}$ or 1. Near-optimal: median number of grid candidates within 0.01 development NLL nats of the selected minimum.

We also evaluate fixed, outcome-independent multipliers $c=.5,1,2$ of each selected temperature, without clipping them to the original grid. For ranking, $s/(cT)+M$ is equivalent to $s/T+cM$. This perturbation therefore probes the relative attribute weight, rather than providing an independent source of evidence beyond the learned-weight comparison. Table S11 gives paired writer uncertainty and the associated fused log loss. No multiplier is selected as a replacement primary rule after inspecting these outcomes.

    Multiplier $c$   Top-1 change \[95% CI\]     Fused NLL \[95% CI\]
  ---------------- ------------------------- ------------------------
               0.5    +0.04 \[-0.98, +0.90\]   2.226 \[1.950, 2.536\]
                 1    -0.57 \[-2.37, +0.91\]   1.879 \[1.710, 2.062\]
                 2    -1.81 \[-4.45, +0.35\]   2.108 \[1.981, 2.242\]

  : **Table S11.** Fixed temperature-multiplier sensitivity over all 120 settings. Top-1 changes are relative to no fusion, in percentage points; NLL is the absolute fused log loss in nats. Intervals condition on the fitted gallery and predictions.

## S9.2 Prediction variability from calibration partitions {#prediction-variability-from-calibration-partitions}

The primary attribute pipeline uses grouped calibration seed 2031. We repeat it with seeds 2032, 2033 and 2034, changing only the inner calibration partitions. Every full outer SVM is refitted on the same training writers, and its raw decision scores are verified unchanged within numerical tolerance. The resulting probability differences therefore characterize calibration-partition variability, not neural initialization, a new feature extractor, independent participants or new datasets.

Table S12 reports the self-chosen-paragraph attribute metrics and identification changes for every seed, retaining 2031 as the primary analysis. We save each unit's probability mean, standard deviation, range, and whether its predicted hand changes at least once across the four partitions. These are correlated partition replicates; they are not averaged as four independent studies, and the best seed is not selected. Each seed-specific interval resamples writers conditional on that seed's fitted predictions.

    Seed   Hand accuracy \[95% CI\], %   Brier     NLL   Fixed fusion change \[95% CI\]
  ------ ----------------------------- ------- ------- --------------------------------
    2031           65.5 \[53.6, 77.4\]   0.204   0.602           -0.57 \[-2.37, +0.91\]
    2032           66.7 \[54.8, 78.6\]   0.201   0.594           -0.50 \[-2.13, +0.83\]
    2033           64.3 \[52.4, 76.2\]   0.202   0.599           -0.56 \[-2.33, +0.90\]
    2034           65.5 \[53.6, 77.4\]   0.205   0.606           -0.52 \[-2.20, +0.86\]

  : **Table S12.** Calibration-partition sensitivity. Attribute metrics use self-chosen paragraphs; fusion changes average all 120 settings and are in percentage points. Accuracy intervals and fusion intervals are conditional writer bootstraps.

## S9.3 Evaluation-writer influence and class-specific effects {#evaluation-writer-influence-and-class-specific-effects}

For each writer, we omit that writer only from evaluation averaging, keeping the 42-writer gallery, fitted models, selected parameters and query predictions fixed. The remaining writers are reweighted to give each hand one half of the total weight. These are influence diagnostics, not reduced-gallery identification experiments, full leave-one-writer-out model refits or independent validation. All 42 writers remain in the primary results; neither prediction errors nor influence values constitute an exclusion criterion.

Table S13 gives the range across all 42 omissions and identifies the largest change. Omission ranges are descriptive ranges, not 95% confidence intervals. In particular, a near-zero fusion mean can change sign under an omission without demonstrating either a benefit or a harmful effect. The full writer list, four designated-setting sensitivities and class-specific estimates are included in the numerical outputs. Table S14 reports the latter averaged across settings.

  Method            All writers       Omission range   Largest change   Writer
  --------------- ------------- -------------------- ---------------- --------
  Known                   +6.11     \[+5.91, +6.26\]             0.19     L_p5
  Fixed fusion            -0.57     \[-0.76, +0.01\]             0.58    R_p21
  Nested fusion           -0.08     \[-0.24, +0.22\]             0.30    R_p21
  Hard filter            -14.29   \[-14.77, -13.10\]             1.19    R_p21

  : **Table S13.** Fixed-gallery evaluation-writer influence on mean top-1 gain, in percentage points. Each hand retains total weight 0.5 after omission; no models are refitted.

  Method              Left-handed \[95% CI\]    Right-handed \[95% CI\]
  --------------- -------------------------- --------------------------
  Known               +7.17 \[+5.56, +8.91\]     +5.05 \[+4.00, +6.14\]
  Fixed fusion        -0.08 \[-2.32, +1.67\]     -1.05 \[-4.04, +1.08\]
  Nested fusion       +0.14 \[-1.21, +1.24\]     -0.31 \[-1.97, +0.97\]
  Hard filter       -14.21 \[-21.31, -7.70\]   -14.37 \[-21.82, -7.83\]

  : **Table S14.** Class-specific top-1 changes from no fusion, percentage points, averaged over all settings with conditional writer-bootstrap intervals. These are observed-cohort subgroup summaries.

![**Figure S4.** Within-cohort diagnostics. (a) Selected temperature distributions across outer-writer/setting fits; boxes show quartiles, whiskers extend to 1.5 IQR and points show values outside them. (b) Learned-weight frequencies across the 5,040 fits. (c) Fixed-fusion mean differences and conditional writer-bootstrap intervals for each calibration partition. (d) Changes in mean gain after each evaluation-writer omission. None of these panels represents independent-dataset validation.](assets/fusion_robustness.png){width="6.65in"}

# S10 Decision margins and opposing fusion effects {#decision-margins-and-opposing-fusion-effects}

This retrospective diagnostic reuses the original calibrated HC-183 query log-odds and the already selected temperatures and nested weights. No parameter is refitted and all 42 writers and 120 settings remain. Let $w^*$ be the true writer, $K_q$ the event that it leads all same-hand candidates, and define

$$m_q=\frac{s_{qw^*}-\max_{j:h_j\ne h_{w^*}}s_{qj}}{T_q},\qquad
 r_q=\log\frac{p_q(h_{w^*})}{p_q(1-h_{w^*})}.$$

Since the attribute term is constant within each hand group, fused top-1 success, away from ties, is exactly $K_q\cap\{m_q+\lambda_qr_q>0\}$. The computation also implements the fixed candidate-order rule for ties. This is an algebraic decomposition of the tested rule, not an additional prediction model. The true hand and identity are used for retrospective analysis only. It follows that fusion cannot repair a wrong within-hand ordering; it can only change the competition between the two groups.

A rescued decision is incorrect before fusion and correct afterwards; a damaged decision is correct before fusion and incorrect afterwards. Their difference equals the net top-1 change for every query/reference decision. The same equal-unit, equal-writer and equal-setting averaging as the main analysis gives fixed-fusion rescue and damage rates of 2.02% and 2.59%, respectively; nested weighting gives 1.33% and 1.42%. Thus a small mean change combines opposing changes rather than indicating that the attribute never affects ranking. The plot illustrates how signed attribute evidence interacts with the identity margin.

Intervals use 10,000 hand-stratified writer bootstraps (seed 20361109), with the same writer indices for all contrasts and settings. They are descriptive marginal intervals conditional on the gallery, predictions and selected parameters. The individual illustrated decisions share queries and writers and are not independent observations. Machine-readable writer outcomes, intervals, the illustrative decisions and input hashes are supplied as `margin_writer_outcomes.csv`, `margin_effect_intervals.csv`, `margin_example.csv` and `margin_validation.json`. Every reconstructed decision is compared with direct candidate-score ranking; writer-level fused accuracies are independently matched to the existing fixed and nested outputs.

![**Figure S5.** Decision changes and margin diagnostic. (a) Rescued and damaged decisions under fixed and nested fusion, averaging all 120 settings with equal writer weights; bars are marginal conditional 95% writer-bootstrap intervals. (b) HC-183, self-chosen full paragraphs and one reference, fixed fusion: only decisions for which the true writer leads its hand group are plotted. The vertical line is the baseline boundary $m=0$; the dashed line is the fusion boundary $m+r=0$. Points above the dashed line succeed after fusion. Each point is a query/reference choice; repeated choices are correlated. The true-hand axes are retrospective diagnostics, not available prediction inputs.](assets/margin_diagnostic.png){width="6.65in"}

# S11 Complete revised identification grid {#complete-revised-identification-grid}

Table S15 reports all 120 settings for the primary HC-183 query predictor, including fixed and nested learned-weight fusion. The machine-readable results additionally contain top-5 accuracy, reciprocal rank, alternative handedness models, writer-level outcomes, temperatures and paired intervals. "Shared" uses the other five prompt pages as references. "Own" uses self-chosen query texts. Query size $n=1$ is a whole paragraph; larger $n$ is the number of strips per page. Thresholds are $A_{\mathrm{none}}/A_{\mathrm{known}}$ under the independence approximation, not empirically established universal requirements.

  Representation   Text       $n$   $k$    None   Known   Fixed   Nested   Filter   Threshold
  ---------------- -------- ----- ----- ------- ------- ------- -------- -------- -----------
  STROKE           shared       1     5   100.0   100.0   100.0    100.0     68.7       100.0
  STROKE           shared       2     5    99.2    99.4    99.2     99.2     65.3        99.8
  STROKE           shared       4     5    96.1    96.9    95.7     95.7     63.7        99.2
  STROKE           shared       8     5    86.6    88.7    86.7     86.7     56.5        97.6
  STROKE           own          1     1    92.3    95.4    93.3     92.5     62.3        96.7
  STROKE           own          2     1    87.7    91.0    87.0     87.7     59.3        96.4
  STROKE           own          4     1    82.8    86.9    83.4     83.6     57.0        95.3
  STROKE           own          8     1    68.4    72.8    67.9     68.5     45.1        94.0
  STROKE           own          1     2    94.9    97.3    95.5     94.9     63.7        97.6
  STROKE           own          2     2    91.6    93.6    90.8     91.3     61.2        97.9
  STROKE           own          4     2    86.5    90.0    86.8     87.1     59.3        96.2
  STROKE           own          8     2    71.4    75.3    70.9     71.8     46.9        94.8
  STROKE           own          1     3    95.2    97.4    95.5     95.2     63.9        97.7
  STROKE           own          2     3    92.6    94.4    92.0     92.3     62.1        98.1
  STROKE           own          4     3    87.9    91.0    88.3     88.5     60.2        96.6
  STROKE           own          8     3    72.4    76.1    71.8     72.6     47.4        95.2
  STROKE           own          1     6    95.2    98.8    96.4     95.2     65.5        96.4
  STROKE           own          2     6    92.9    94.6    91.7     92.3     61.9        98.1
  STROKE           own          4     6    89.0    91.1    88.7     89.0     60.0        97.7
  STROKE           own          8     6    72.6    76.2    72.2     73.3     47.8        95.4
  HC-183           shared       1     5    82.9    86.1    80.6     82.9     57.9        96.3
  HC-183           shared       2     5    67.1    74.0    65.1     67.3     48.6        90.6
  HC-183           shared       4     5    42.3    51.4    41.3     41.8     34.0        82.2
  HC-183           shared       8     5    25.4    35.2    24.6     24.7     21.9        72.2
  HC-183           own          1     1    39.7    48.0    37.9     38.5     28.0        82.6
  HC-183           own          2     1    35.2    43.3    34.1     34.1     29.0        81.4
  HC-183           own          4     1    23.3    31.3    22.6     23.0     19.0        74.6
  HC-183           own          8     1    14.2    21.7    14.1     14.1     11.8        65.6
  HC-183           own          1     2    47.4    53.4    43.1     45.3     30.8        88.7
  HC-183           own          2     2    43.4    51.2    41.2     42.5     34.5        84.7
  HC-183           own          4     2    27.3    35.3    26.5     27.1     22.4        77.4
  HC-183           own          8     2    17.0    25.8    16.5     16.9     14.3        65.9
  HC-183           own          1     3    51.4    57.4    48.0     50.5     33.7        89.4
  HC-183           own          2     3    47.0    54.5    44.8     46.7     37.1        86.3
  HC-183           own          4     3    29.7    38.4    29.1     29.7     24.0        77.5
  HC-183           own          8     3    18.7    28.1    17.6     18.6     15.5        66.6
  HC-183           own          1     6    57.1    61.9    54.8     54.8     36.9        92.3
  HC-183           own          2     6    54.8    60.1    50.6     53.6     40.5        91.1
  HC-183           own          4     6    33.0    42.9    32.1     32.7     25.6        77.1
  HC-183           own          8     6    21.3    30.8    19.3     20.7     16.8        69.0
  ResNet-18        shared       1     5    96.8    96.8    96.0     96.4     66.7       100.0
  ResNet-18        shared       2     5    80.4    85.3    79.0     79.4     55.6        94.2
  ResNet-18        shared       4     5    63.8    70.2    62.9     63.5     46.1        90.8
  ResNet-18        shared       8     5    17.4    24.8    18.0     17.8     17.0        70.4
  ResNet-18        own          1     1    62.9    67.9    62.1     62.7     42.3        92.7
  ResNet-18        own          2     1    43.8    50.8    44.5     44.4     34.7        86.3
  ResNet-18        own          4     1    35.0    42.6    34.6     35.0     29.6        82.2
  ResNet-18        own          8     1    10.4    18.0    11.1     10.8     11.9        57.5
  ResNet-18        own          1     2    73.1    77.3    72.5     72.7     48.8        94.6
  ResNet-18        own          2     2    52.6    58.8    51.7     52.5     40.0        89.4
  ResNet-18        own          4     2    40.4    46.8    39.7     40.2     32.2        86.4
  ResNet-18        own          8     2    11.2    17.4    11.7     11.3     12.1        64.0
  ResNet-18        own          1     3    77.7    82.1    77.2     77.9     52.6        94.7
  ResNet-18        own          2     3    55.9    61.6    55.1     55.8     42.2        90.8
  ResNet-18        own          4     3    43.1    48.7    42.2     42.8     33.7        88.4
  ResNet-18        own          8     3    11.7    17.9    12.2     11.7     12.5        65.2
  ResNet-18        own          1     6    83.3    85.7    81.0     83.3     54.8        97.2
  ResNet-18        own          2     6    60.1    65.5    58.9     58.9     45.2        91.8
  ResNet-18        own          4     6    47.4    51.7    46.0     46.6     35.8        91.9
  ResNet-18        own          8     6    12.3    18.8    12.7     12.0     13.1        65.4
  EffNet-B0        shared       1     5    96.0    96.8    96.0     96.0     65.9        99.2
  EffNet-B0        shared       2     5    74.8    82.5    74.8     75.2     54.2        90.6
  EffNet-B0        shared       4     5    64.0    70.8    63.7     63.7     45.4        90.3
  EffNet-B0        shared       8     5    16.7    24.4    17.0     17.0     16.2        68.4
  EffNet-B0        own          1     1    51.4    60.3    51.0     51.8     36.7        85.2
  EffNet-B0        own          2     1    40.7    49.9    39.9     40.6     33.5        81.5
  EffNet-B0        own          4     1    33.0    41.9    33.1     33.8     27.8        78.9
  EffNet-B0        own          8     1    12.9    18.3    13.5     13.3     12.5        70.3
  EffNet-B0        own          1     2    62.4    69.4    61.8     63.4     41.7        89.8
  EffNet-B0        own          2     2    49.6    58.4    48.6     49.3     39.2        84.9
  EffNet-B0        own          4     2    39.8    48.5    40.4     40.8     32.1        82.1
  EffNet-B0        own          8     2    14.4    20.0    14.9     14.9     13.7        72.2
  EffNet-B0        own          1     3    67.4    73.2    66.5     67.6     44.4        92.1
  EffNet-B0        own          2     3    54.1    61.8    52.7     53.9     41.7        87.5
  EffNet-B0        own          4     3    43.6    52.1    44.0     45.0     34.7        83.6
  EffNet-B0        own          8     3    15.8    21.6    16.3     16.1     14.6        73.2
  EffNet-B0        own          1     6    70.2    75.0    69.0     70.2     45.2        93.7
  EffNet-B0        own          2     6    57.1    64.3    57.1     58.3     43.5        88.9
  EffNet-B0        own          4     6    49.2    57.6    48.9     50.1     38.2        85.5
  EffNet-B0        own          8     6    17.6    23.1    17.8     17.9     16.1        76.1
  ViT-B/16         shared       1     5    96.4    96.4    96.4     96.4     65.9       100.0
  ViT-B/16         shared       2     5    86.1    89.7    85.1     86.1     58.1        96.0
  ViT-B/16         shared       4     5    65.1    72.4    64.9     65.0     45.9        89.9
  ViT-B/16         shared       8     5    27.7    42.8    28.3     28.0     27.1        64.6
  ViT-B/16         own          1     1    68.1    73.8    68.3     69.2     45.8        92.2
  ViT-B/16         own          2     1    48.7    56.8    48.5     48.8     38.9        85.7
  ViT-B/16         own          4     1    37.5    46.0    37.1     37.4     31.2        81.5
  ViT-B/16         own          8     1    17.4    26.2    18.3     18.1     16.7        66.4
  ViT-B/16         own          1     2    76.4    81.4    76.8     77.2     50.2        93.9
  ViT-B/16         own          2     2    55.4    62.9    54.5     55.1     43.3        88.1
  ViT-B/16         own          4     2    42.0    50.7    42.1     42.3     34.7        82.9
  ViT-B/16         own          8     2    19.3    29.4    19.7     19.3     18.6        65.4
  ViT-B/16         own          1     3    80.0    84.8    79.3     79.6     52.4        94.3
  ViT-B/16         own          2     3    57.6    65.1    57.1     57.9     44.9        88.4
  ViT-B/16         own          4     3    44.9    53.6    44.8     44.9     36.3        83.7
  ViT-B/16         own          8     3    20.7    31.6    20.8     20.4     19.9        65.4
  ViT-B/16         own          1     6    83.3    85.7    82.1     83.3     53.6        97.2
  ViT-B/16         own          2     6    61.3    68.5    61.3     60.7     47.6        89.6
  ViT-B/16         own          4     6    48.3    57.1    48.0     48.0     39.1        84.6
  ViT-B/16         own          8     6    23.8    34.8    23.2     24.0     22.1        68.5
  Stroke-CNN       shared       1     5    91.7    94.4    90.9     91.7     64.7        97.1
  Stroke-CNN       shared       2     5    67.3    75.2    67.1     67.9     48.2        89.4
  Stroke-CNN       shared       4     5    15.9    24.0    16.6     16.5     14.4        66.1
  Stroke-CNN       shared       8     5     4.1     5.7     2.5      2.5      2.5        72.1
  Stroke-CNN       own          1     1    58.7    70.8    59.9     59.7     43.8        82.9
  Stroke-CNN       own          2     1    42.3    52.5    40.9     41.8     33.3        80.5
  Stroke-CNN       own          4     1    10.1    15.5     9.7      9.9      7.9        65.1
  Stroke-CNN       own          8     1     3.6     6.5     3.1      3.1      2.9        55.3
  Stroke-CNN       own          1     2    67.5    77.9    66.8     67.2     48.9        86.6
  Stroke-CNN       own          2     2    49.0    58.0    46.9     48.2     37.3        84.5
  Stroke-CNN       own          4     2    10.1    15.9    10.3     10.4      8.3        63.7
  Stroke-CNN       own          8     2     3.7     6.2     2.3      2.5      2.2        59.7
  Stroke-CNN       own          1     3    70.6    81.1    70.1     71.1     50.1        87.1
  Stroke-CNN       own          2     3    52.5    60.4    50.2     51.2     38.8        87.0
  Stroke-CNN       own          4     3    10.5    15.9    10.1     10.2      8.2        65.9
  Stroke-CNN       own          8     3     3.7     6.0     2.3      2.4      2.2        62.8
  Stroke-CNN       own          1     6    71.4    83.3    72.6     72.6     50.0        85.7
  Stroke-CNN       own          2     6    54.2    61.9    51.8     54.2     39.9        87.5
  Stroke-CNN       own          4     6    10.7    15.5    10.7     11.0      8.3        69.2
  Stroke-CNN       own          8     6     3.9     5.1     2.0      2.0      1.8        76.3

  : **Table S15.** Complete revised identification results, accuracy and approximate threshold in %.

# S12 Reproduction and provenance {#reproduction-and-provenance}

The main manuscript additionally reports representation-level paired utility contrasts. Each writer contributes the equal mean of the 20 settings for a representation. The 10,000 hand-stratified writer resamples use seed 20361108 and the same sampled indices for every contrast. These are exploratory marginal intervals with no simultaneous-coverage claim. The generated numerical contrasts and input hashes are in `representation_utility_intervals.csv` and `representation_utility_provenance.json` in the fusion/robustness results directory.

The versioned archive [release v1.6.0](https://github.com/LegendaryBeast/Bangla-writer-identification-research/releases/tag/v1.6.0), `wid-reproducibility-v1.6.0.zip`, contains analysis code, frozen features, required classification inputs, revised results, self-contained methods and figure-generation scripts. Dataset images are downloaded separately from doi:10.17632/n4sjstdzfy.1. Code is MIT licensed and numerical results are CC BY 4.0. A file-level SHA-256 manifest verifies archive contents.

Starting from the archive root, verification and figure generation use saved outputs without fitting new models. The full mode recomputes analyses from frozen features, reusing hash-verified development caches. The release README lists individual commands and an option to refit those caches:

    python -m pip install -r requirements-frozen.txt
    python code/reproduce.py --verify
    python code/reproduce.py --figures
    python code/reproduce.py --full --jobs 3
    cd paper
    latexmk -pdf -interaction=nonstopmode -halt-on-error wid_supplement.tex

The additional random-label predictor fitting is the slowest part; cached evidence avoids repeating it. The released feature file preserves the original ordering through `unit_id`; every analysis verifies alignment. Source-code and input hashes, seeds, parameter counts and runtime/library versions are saved alongside the outputs. The original extraction used Python 3.13.9, torch 2.14.0 and torchvision 0.29.0 on MPS; the revised classical analyses use the environment recorded in their provenance. Re-extraction on another accelerator can change the last digits of floating-point embeddings. The reported revised results use the supplied feature matrices.

The four designated fixed-fusion settings were chosen after an exploratory feasibility analysis of paragraph-level conditions. Learned-weight comparisons and robustness analyses are exploratory; no analysis was preregistered. Identification normalization and temperature fitting use only the information permitted by the stated closed-gallery protocol. Conditional bootstrap intervals do not encompass model-refitting, new-gallery or new-session uncertainty. The external CSAFE evaluation in Section S13 assesses a separate cohort and capture sessions; its conditional intervals still do not propagate full fitting uncertainty.

# S13 External CSAFE validation {#external-csafe-validation}

## S13.1 Source, cohort and capture conditions {#source-cohort-and-capture-conditions}

The secondary data are the [CSAFE Handwriting Database, version 2](https://doi.org/10.25380/iastate.10062203.v2), credited to Amy Crawford, Anyesha Ray, Alicia Carriquiry, James Kruse and Marc Peterson, under CC BY 4.0. Crawford, Ray and Carriquiry describe the original 90-writer release in *Data in Brief* 28 (2020), 105059, doi:10.1016/j.dib.2019.105059. The version-2 survey instead has 475 unique writer IDs: 48 left-handed, 423 right-handed, three ambidextrous and one missing hand. Every survey ID matches nine original image filenames in each of three session archives (12,825 images in total).

The analysis includes all 48 left-handed writers and 48 right-handed writers sampled without replacement from sorted IDs using NumPy `default_rng(20261108)`. The four ambiguous/missing-hand records are excluded before sampling. No writer is removed according to classification accuracy. The selected IDs and source-image hashes are provided in the archive. Each writer supplies the nine pages in Table S16. Printed headers have been removed by the provider. Distinct sessions and source pages are preserved; no crop of a gallery page is used as a query.

  Session   Prompt/repetition         Pages Role
  --------- ----------------------- ------- --------------------------------
  1         LND, repetition 1            96 Gallery and auxiliary training
  1         LND, repetitions 2--3       192 Auxiliary training only
  1         WOZ, repetitions 1--3       288 Auxiliary training only
  2         LND, repetition 1            96 Primary, same prompt
  2         WOZ, repetition 1            96 Primary, different prompt
  3         WOZ, repetition 1            96 Secondary, different prompt

  : **Table S16.** Source pages selected for each of the 96 CSAFE writers. LND is the London Letter; WOZ is the Wizard of Oz passage.

Where enrollment-relative timing is available, session 2 follows session 1 for 94/96 writers (median 33 days; range 18--252), and session 3 follows session 1 for 95/96 (median 68; range 42--272). Missing timing values are retained under their original session labels. These records do not support claiming that every observed interval is at least 21 days.

## S13.2 Features and nested fitting {#features-and-nested-fitting}

The original $n=1$ extraction is applied to whole handwriting pages, with $k=1$ gallery reference. HC-183, STROKE and the eight nuisance covariates are extracted on all 864 pages. Frozen ResNet-18 and ViT-B/16 embeddings are needed for the 96 gallery and 288 query pages. Auxiliary-only pages have no neural embedding; their intentional missing entries never enter identity scoring. All pages pass the common validity rules. Exact checkpoint names, URLs, hashes, source hashes and accelerator information are recorded. A fixed-sample comparison against the original extraction function, including a prefetched auxiliary page, gives zero difference after the original float32 storage in all compared features.

For the full gallery, an outer attribute fit uses six session-1 pages from each of the other 95 writers. Its calibrator uses their five-fold writer-group out-of-fold SVM scores. Five development folds inside this outer exclusion regenerate attribute evidence, with another five writer-group folds for each development calibrator. Thus both the outer writer and each development validation group are absent from the corresponding attribute fit. All 96 known gallery hands and selected references remain available as candidate metadata. Temperature and fusion weight are selected separately for each query condition using other query writers. The grids and the $C=0.1$ linear SVM are unchanged. The same fitting rules are refitted within each balanced 42-writer gallery.

The primary HC-183 model and the original STROKE and adjusted HC-183 sensitivity models are fixed before CSAFE outcome inspection. Eight primary comparisons concern fixed/nested versus identity-only ranking for ResNet-18 and ViT-B/16 in both session-2 conditions. Holm adjustment covers all eight. Session 3, alternate predictors, handcrafted identity representations and overlapping galleries are secondary descriptive analyses. Local specification of these CSAFE contrasts is not a globally preregistered external-dataset sampling plan.

## S13.3 Numerically equivalent logistic calibration {#numerically-equivalent-logistic-calibration}

Initial fitting produced L-BFGS convergence warnings for the adjusted sensitivity model. Before inspecting CSAFE performance tables, an implementation amendment centered and scaled the training out-of-fold SVM scores while retaining the original logistic objective. Let $z_i=(s_i-\bar s)/\sigma_s$ and let $C_{\mathrm{cal}}=10^4$. The fitted objective is

$$L(a,b)=\frac1N\sum_i\bigl[\log(1+e^{a z_i+b})-y_i(a z_i+b)\bigr]+\frac{a^2}{2C_{\mathrm{cal}}N\sigma_s^2}.\qquad\text{(S3)}$$

The raw-score slope is $a/\sigma_s$; the intercept is unpenalized. The transformed penalty therefore equals the original raw-slope L2 penalty. A zero score standard deviation uses scale one. L-BFGS-B uses analytic gradients, a maximum of 2,000 iterations and a gradient certificate of $10^{-7}$. All primary and sensitivity fits are regenerated with this implementation; no feature, writer, SVM penalty, endpoint or fusion grid is selected from outcomes. The largest observed gradient infinity norm is $9.51\times10^{-9}$, with every optimizer reporting success. An independent numerical check matches the original logistic estimator's probabilities within $2.27\times10^{-11}$ on a well-conditioned reference case and verifies the raw/transformed objective equality. Numerical optimizer convergence is distinct from predictive calibration quality.

## S13.4 Attribute and identification results {#attribute-and-identification-results}

Table S17 reports all nine full-gallery handedness conditions. The balanced cohort makes ordinary accuracy equal to balanced accuracy. Its intervals resample writers within hand and retain the fitted model. Raw and calibrated AUCs pool different writer-held-out score sets. Adjusted HC-183 has negative Platt slopes in 24/96 outer fits, compared with none for the unadjusted models. This diagnostic does not prove a specific acquisition or leakage mechanism.

  Query    Model           Acc. (%) 95% interval           NLL   Brier   Raw AUC   Cal. AUC
  -------- ------------- ---------- ------------------ ------- ------- --------- ----------
  S2/LND   HC-183             67.71 \[58.33, 77.08\]     0.636   0.221     0.737      0.711
  S2/LND   STROKE             67.71 \[58.33, 77.08\]     0.638   0.213     0.770      0.741
  S2/LND   HC-183 adj.        34.38 \[25.00, 43.75\]     0.686   0.245     0.727      0.501
  S2/WOZ   HC-183             58.33 \[48.96, 67.71\]     0.682   0.239     0.651      0.641
  S2/WOZ   STROKE             60.42 \[51.04, 69.79\]     0.695   0.226     0.725      0.687
  S2/WOZ   HC-183 adj.        34.38 \[25.00, 43.75\]     0.675   0.241     0.692      0.495
  S3/WOZ   HC-183             65.62 \[56.25, 75.00\]     0.645   0.225     0.725      0.702
  S3/WOZ   STROKE             58.33 \[47.92, 67.71\]     0.660   0.224     0.724      0.702
  S3/WOZ   HC-183 adj.        35.42 \[26.04, 44.79\]     0.677   0.240     0.725      0.530

  : **Table S17.** CSAFE handedness prediction on the full 48/48 cohort. Intervals are conditional 95% writer-bootstrap intervals; NLL and Brier scores are losses. AUC columns concern raw and calibrated scores.

Table S18 gives the eight primary paired contrasts. The percentile-bootstrap intervals are conditional, not simultaneous and not inversions of the discrete paired sign-flip tests. An interval can exclude zero while the Holm-adjusted test does not reject; a $[0,0]$ interval only describes identical correctness outcomes of the fitted rules. No primary adjusted test rejects at 0.05. Table S19 retains every full-gallery comparator, including the secondary session-3 and alternative-predictor results.

  Prompt      ID features   Fusion     Change 95% interval          Holm $p$
  ----------- ------------- -------- -------- ------------------- ----------
  Same        ResNet-18     Fixed       +1.04 \[-4.17, 6.25\]         1.0000
  Same        ResNet-18     Nested      +1.04 \[-4.17, 6.25\]         1.0000
  Same        ViT-B/16      Fixed       -2.08 \[-6.25, 2.08\]         1.0000
  Same        ViT-B/16      Nested      -2.08 \[-6.25, 2.08\]         1.0000
  Different   ResNet-18     Fixed       -5.21 \[-10.42, -1.04\]       0.4888
  Different   ResNet-18     Nested      +0.00 \[0.00, 0.00\]          1.0000
  Different   ViT-B/16      Fixed       -1.04 \[-3.12, 0.00\]         1.0000
  Different   ViT-B/16      Nested      -1.04 \[-3.12, 0.00\]         1.0000

  : **Table S18.** Primary CSAFE fixed/nested changes from identity only, in percentage points. Prompt relation is to the session-1 LND gallery; all queries are session 2.

  Query    Hand model    ID features        ID   Known   Filter   Fixed   Nested
  -------- ------------- ------------- ------- ------- -------- ------- --------
  S2/LND   HC-183        ResNet-18       61.46   68.75    43.75   62.50    62.50
  S2/LND   HC-183        ViT-B/16        64.58   67.71    39.58   62.50    62.50
  S2/LND   HC-183        HC-183          44.79   48.96    32.29   40.62    39.58
  S2/LND   HC-183        STROKE          70.83   73.96    48.96   71.88    71.88
  S2/LND   STROKE        ResNet-18       61.46   68.75    52.08   62.50    62.50
  S2/LND   STROKE        ViT-B/16        64.58   67.71    48.96   63.54    63.54
  S2/LND   STROKE        HC-183          44.79   48.96    33.33   41.67    43.75
  S2/LND   STROKE        STROKE          70.83   73.96    52.08   71.88    71.88
  S2/LND   HC-183 adj.   ResNet-18       61.46   68.75    22.92   62.50    61.46
  S2/LND   HC-183 adj.   ViT-B/16        64.58   67.71    22.92   62.50    63.54
  S2/LND   HC-183 adj.   HC-183          44.79   48.96    13.54   42.71    44.79
  S2/LND   HC-183 adj.   STROKE          70.83   73.96    26.04   70.83    70.83
  S2/WOZ   HC-183        ResNet-18       54.17   60.42    31.25   48.96    54.17
  S2/WOZ   HC-183        ViT-B/16        55.21   61.46    34.38   54.17    54.17
  S2/WOZ   HC-183        HC-183          22.92   33.33    19.79   22.92    23.96
  S2/WOZ   HC-183        STROKE          67.71   69.79    39.58   65.62    67.71
  S2/WOZ   STROKE        ResNet-18       54.17   60.42    37.50   52.08    54.17
  S2/WOZ   STROKE        ViT-B/16        55.21   61.46    36.46   55.21    54.17
  S2/WOZ   STROKE        HC-183          22.92   33.33    18.75   22.92    21.88
  S2/WOZ   STROKE        STROKE          67.71   69.79    46.88   63.54    65.62
  S2/WOZ   HC-183 adj.   ResNet-18       54.17   60.42    17.71   51.04    52.08
  S2/WOZ   HC-183 adj.   ViT-B/16        55.21   61.46    19.79   54.17    55.21
  S2/WOZ   HC-183 adj.   HC-183          22.92   33.33    13.54   22.92    20.83
  S2/WOZ   HC-183 adj.   STROKE          67.71   69.79    23.96   66.67    65.62
  S3/WOZ   HC-183        ResNet-18       46.88   56.25    35.42   50.00    50.00
  S3/WOZ   HC-183        ViT-B/16        58.33   60.42    36.46   55.21    55.21
  S3/WOZ   HC-183        HC-183          23.96   34.38    23.96   25.00    23.96
  S3/WOZ   HC-183        STROKE          53.12   59.38    40.62   54.17    54.17
  S3/WOZ   STROKE        ResNet-18       46.88   56.25    32.29   48.96    47.92
  S3/WOZ   STROKE        ViT-B/16        58.33   60.42    34.38   58.33    58.33
  S3/WOZ   STROKE        HC-183          23.96   34.38    17.71   21.88    21.88
  S3/WOZ   STROKE        STROKE          53.12   59.38    37.50   52.08    52.08
  S3/WOZ   HC-183 adj.   ResNet-18       46.88   56.25    21.88   47.92    47.92
  S3/WOZ   HC-183 adj.   ViT-B/16        58.33   60.42    22.92   56.25    53.12
  S3/WOZ   HC-183 adj.   HC-183          23.96   34.38    11.46   22.92    22.92
  S3/WOZ   HC-183 adj.   STROKE          53.12   59.38    23.96   54.17    54.17

  : **Table S19.** Complete 96-writer CSAFE top-1 comparisons (%). ID is identity only; Known supplies the true query hand; Filter uses the predicted hand. Alternate attribute models and session 3 are secondary.

## S13.5 Gallery sensitivity and reproducibility {#gallery-sensitivity-and-reproducibility}

Twenty seeded balanced 21/21 galleries are drawn from the selected 96-writer pool and refit independently for the primary HC-183 predictor. Their overlapping memberships make them dependent sensitivity samples. Figure S6 shows their changes alongside the attribute checks. They characterize gallery composition within this pool, not a population-level distribution or 20 independent dataset replications. The complete 196-setting summary, all 50,880 writer-method outcomes and selected parameters accompany the external reproducibility archive.

![**Figure S6.** CSAFE secondary checks. (a) All predefined attribute models and query conditions, with conditional writer-bootstrap 95% intervals. (b) Fixed/nested identification changes across 20 overlapping balanced 42-writer galleries. Points are dependent descriptive subsamples.](assets/csafe_attribute_robustness.png){width="6.65in"}

The archive `csafe-replication-2026-11-08.zip` supplies the frozen features, original and amended protocol specifications, source/member hashes, checkpoint references, fitting manifests, numerical outputs and figure/table scripts. Raw CSAFE images and neural checkpoint binaries are obtained separately. The downloaded survey MD5 and every selected image's source ZIP size/CRC and SHA-256 are checked; the complete multi-gigabyte source ZIP MD5s are not claimed to have been verified after partial retrieval. Validation audits 1,128 outer exclusions, 6,768 final/development attribute models and 33,840 calibration folds, verifies all image roles and reported accuracies, and checks the filtering identity and exact fusion-margin reconstruction. A fresh run from the extracted package, with supplied fitting caches removed, reproduces all nine core numerical and manifest outputs exactly. This is computational reproducibility, not an additional empirical replication.

From the archive root, saved-feature reproduction uses:

    python code/test_csafe_calibration.py
    python code/run_csafe_replication.py --jobs 4
    python code/validate_csafe_replication.py --saved-only
    python code/summarize_csafe_replication.py

Move the supplied `attribute_cache/` aside to force fresh fitting. For source-image reproduction, run `acquire_csafe_replication.py`, then `extract_csafe_features.py`; full validation omits `–saved-only`. Saved-only validation covers numerical outputs; the original full audit additionally verifies images and checkpoints.
