---
title: "Handedness for Bangla writer identification: Oracle gains and predicted-attribute utility"
author:
  - "Tanzim Hasan Prappo"
  - "M. Shahidur Rahman"
date: ""
bibliography: refs.bib
csl: ieee.csl
link-citations: true
reference-section-title: References
---

Department of Computer Science and Engineering, Shahjalal University of Science and Technology, Sylhet 3114, Bangladesh

Corresponding author: Tanzim Hasan Prappo (2021331006@student.sust.edu). Coauthor email: rahmanms@sust.edu.

## Abstract {#abstract .unnumbered}

Attribute-classification accuracy can misrepresent the identification value of a soft biometric. We present a controlled framework separating oracle gains, prediction errors, evidence dependence and writer-associated information, and apply it to BanglaHand: 42 writers, six representations and 120 identification settings. Normalization uses selected gallery references only; query-handedness models exclude the queried writer from training and calibration. Known handedness improves top-1 accuracy by 6.1 percentage points, compared with 6.9 points for an expected known random partition. With query handedness predicted at 65.5% accuracy on self-chosen paragraphs, calibrated fixed-weight fusion changes identification accuracy by -0.6 points, properly nested learned weighting by -0.08, and hard filtering by -14.3. Neither fusion rule establishes a gain in the designated comparisons. Filtering and margin diagnostics explain how joint correctness and confidence affect ranking. Random writer labels expose writer-associated information in enrolled-writer attribute evaluation. An external CSAFE evaluation uses 96 writers and separate gallery/query capture sessions. Handedness accuracy is 67.7% for same-prompt and 58.3% for cross-prompt session-2 queries; none of eight primary fusion comparisons establishes a Holm-adjusted gain. Session-3 and gallery-subsample analyses provide secondary checks. The framework provides reusable controls for evaluating auxiliary attributes in biometric candidate ranking, helping researchers distinguish oracle opportunity from operational benefit and make fusion decisions using measured identification utility.

**Keywords:** Writer identification; Soft biometrics; Handedness; Bangla handwriting; Score fusion; Evaluation protocol

# 1 Introduction

A soft biometric describes a person without uniquely identifying them. Gender, age and handedness can supplement identity matching by changing candidate scores or restricting a gallery [@jain2004soft; @dantcheva2016soft]. Handwriting studies have predicted such attributes from static images [@liwicki2011gender; @siddiqi2015gender; @bouadjenek2016soft; @almaadeed2016handedness; @morera2018handedness; @rahmanian2021handedness]. The downstream question is different: does the predicted attribute improve identification of the writer?

Two issues complicate this question. First, an incorrect hard attribute decision can exclude the true writer. Its utility depends on identification performance conditional on attribute correctness, not just the marginal accuracy of each system. Soft fusion avoids automatic exclusion, but depends on probability calibration and on whether the attribute scores add information to the identity scores. Both can come from the same image and contain redundant evidence. Second, handwriting carries writer-associated information [@bulacu2007text; @schomaker2007codebooks]. An attribute predictor evaluated on previously seen writers can exploit that information. Such evaluation may match an enrolment-conditioned application, but does not estimate attribute generalization to unseen writers [@kaufman2012leakage].

The contribution of this study is a controlled framework for evaluating whether predicted soft-biometric information provides downstream identification utility. Separating oracle gains, prediction errors, evidence dependence and writer-associated information shows why attribute-classification accuracy alone can be misleading. Known random partitions benchmark oracle opportunity; calibrated fixed and nested learned-weight fusion measure achieved utility; and joint-correctness, margin and random-label diagnostics distinguish sources of apparent gains. The elementary filtering identity and standard fusion rules are diagnostic tools and comparators.

Building on the companion study [@prappo2026handedness], which examines handedness classification and photometric cues, we reuse BanglaHand [@banglahand2026data], its preprocessing, descriptors and frozen network representations. The present study evaluates whether predicted handedness provides downstream writer-identification utility. Its methodological contribution is a controlled evaluation that separates oracle gains from operational gains through random-partition controls, nested fusion-weight selection, and analyses of error dependence and writer-associated information.

The BanglaHand analysis is complemented by an external CSAFE evaluation of the same utility question on independent writers, separate source pages and capture sessions [@crawford2020database; @crawford2026csafe].

The paper is organized as follows. Section 2 reviews related work; Section 3 defines the protocol; Section 4 reports the results; and Sections 5--7 discuss their implications, scope and conclusions.

# 2 Related work

## 2.1 Soft biometrics and downstream utility

Soft-biometric fusion predates handwriting-based handedness prediction. Jain et al. combine demographic information with a primary biometric, while Dantcheva et al. distinguish attribute estimation, database indexing and recognition applications [@jain2004soft; @dantcheva2016soft]. These foundations motivate auxiliary evidence, but its usefulness depends on the available metadata, prediction uncertainty and primary matcher. A correct attribute can narrow a candidate set; a predicted attribute can also demote the true candidate. Consequently, attribute recognition and identification improvement require different endpoints. The present work examines this distinction for handwriting through paired comparisons with the same gallery, identity scores and query samples.

## 2.2 Handedness and demographic prediction

Early handwriting studies investigated gender and other demographic traits using online dynamics, offline shape and texture, and combinations of descriptors [@liwicki2011gender; @siddiqi2015gender; @bouadjenek2016soft; @hassaine2013icdar]. Al-Maadeed et al. study a 121-writer QUWI subset with Arabic and English documents, combining handcrafted features, fuzzy conceptual reduction and a nearest-neighbour classifier [@almaadeed2016handedness]. Their principal evaluation concerns handedness classification and feature reduction. Multiple scripts and text conditions broaden that evaluation, but do not by themselves measure the change in an identity ranking caused by predicted handedness.

Morera et al. use CNN word predictions and line-level voting on IAM and KHATT, explicitly separating training and test individuals [@morera2018handedness]. They report handedness overall accuracy of 90.70% on IAM and 70.91% on KHATT. Their respective left-handed recalls are 63.40% and 38.64%, illustrating why overall accuracy must be interpreted alongside class composition and class-specific performance. Rahmanian and Shayegan compare DenseNet201, InceptionV3 and Xception on the same two databases and report 99.14% KHATT handedness accuracy [@rahmanian2021handedness]. This is an attribute-classification result; the accessible publisher record does not establish all details needed for a matched writer-disjoint comparison. We therefore make no inference about writer overlap in that study.

The companion study evaluates writer-independent Bangla handedness classification and photometric cues [@prappo2026handedness]. Its 77.4% balanced accuracy aggregates SVM margins over six shared paragraphs under repeated five-fold evaluation. Here, a calibrated single-query probability modifies a closed-gallery identity ranking under leave-one-writer-out fitting. Different text aggregation, fitting and endpoints prevent a direct numerical comparison. Table 1 separates these prediction tasks from writer-recognition tasks and the utility endpoint considered here.

## 2.3 Writer recognition and evaluation controls

Text-independent writer recognition models individual writing style through contour directions, hinges, allographs and contour-fragment codebooks [@bulacu2007text; @schomaker2007codebooks]. Learned representations can target identity directly: FragNet combines feature-pyramid and fragment pathways for word- and page-level writer identification across four benchmarks [@he2020fragnet]. In Bangla, Paul et al. combine feature extraction and genetic feature selection for writer verification [@paul2024bangla]; BanglaWriting and the Boise State collection support related handwriting-recognition research [@mridha2021banglawriting; @majid2018bangla]. Verification, recognition of written content and closed-set identity ranking address different decisions. Our frozen representations provide controlled comparators, not a claim to outperform handwriting-specialized recognition networks.

Evaluation design also determines what an apparent improvement means. Writer separation addresses generalization to new individuals; it does not alone establish that an attribute supplies information beyond the identity matcher. Leakage analysis requires specifying the information available at each fitting stage [@kaufman2012leakage], and fusion-weight selection must be separated from evaluation to avoid selection bias [@cawley2010selection]. We bring these concerns together by measuring oracle opportunity, predicted-attribute utility, error dependence and writer-associated predictability, with an external evaluation on separate CSAFE pages and capture sessions. This combination defines the contribution rather than a new attribute classifier or an accuracy ranking across incompatible protocols.

  Study                                                                               Data and representation                                                       Evaluated endpoint                                                       Relation to the present question
  ----------------------------------------------------------------------------------- ----------------------------------------------------------------------------- ------------------------------------------------------------------------ -----------------------------------------------------------------------------
  Jain et al.; Dantcheva et al. [@jain2004soft; @dantcheva2016soft]                   Primary biometrics with auxiliary personal attributes                         Recognition, indexing and soft-biometric fusion                          Establish the broader motivation for auxiliary information.
  Al-Maadeed et al. [@almaadeed2016handedness]                                        QUWI, 121 writers; Arabic/English; handcrafted features and fuzzy reduction   Handedness classification and feature reduction                          Attribute predictability; downstream ranking change is a separate endpoint.
  Morera et al. [@morera2018handedness]                                               IAM/KHATT; CNN words, line voting; separate train/test writers                Handedness overall accuracy: 90.70%/70.91%; left recall: 63.40%/38.64%   Shows the importance of class composition and prediction unit.
  Rahmanian and Shayegan [@rahmanian2021handedness]                                   IAM/KHATT; three deep CNN architectures                                       Gender and handedness; 99.14% KHATT handedness accuracy                  Strong attribute result; detailed split remains unverified here.
  Bulacu and Schomaker; Schomaker et al. [@bulacu2007text; @schomaker2007codebooks]   Textural/allographic descriptors and contour-fragment codebooks               Writer identification and verification                                   Establish identity-matching representations, rather than attribute utility.
  He and Schomaker [@he2020fragnet]                                                   Four benchmarks; feature-pyramid and fragment pathways                        Word- and page-level writer identification                               Learns identity representations; not a handedness-fusion comparison.
  Paul et al. [@paul2024bangla]                                                       Offline Bangla, 100 writers; handcrafted/CNN features and genetic selection   Writer verification; reported mean accuracy 94.54%                       Verifies a claimed writer; distinct from closed-set top-1 ranking.
  Companion preprint [@prappo2026handedness]                                          BanglaHand, 42 balanced writers; shared-prompt aggregation                    Writer-independent handedness and photometric sensitivity                Shared cohort and representations; a different task and endpoint.

  : **Table 1.** Representative related work, organized by the decision evaluated. Reported results retain their original endpoints and are not a common performance ranking.

# 3 Materials and methods

## 3.1 Queries, references and representations

BanglaHand contains 336 paragraph photographs from 42 writers (21 left-handed), with common materials and one controlled photographic session [@banglahand2026data]. Each writer supplied six shared prompts and two self-chosen texts. A query is a whole paragraph or one of 2, 4 or 8 horizontal strips, approximately 3.5, 1.75 or 0.9 lines. Strip boundaries follow low-ink rows near equally spaced cuts. Of 5040 candidate units, 39 fail the fixed feature-validity rules; exclusion counts and weighting are reported in the supplement. At width 1,536 pixels, rows with ink fraction above 0.01 define a text region through its 0.5th and 99.5th percentiles. Nominal strip cuts move to minima of the row-ink profile smoothed with standard deviation 96/5 pixels. The search half-width is the smaller of 48 pixels and the text-block height divided by $3n$, with a minimum of three pixels. Row fractions transfer the boundaries across image scales. A unit is invalid if its high-resolution cropped mask contains fewer than 200 ink pixels, its 768-pixel mask contains fewer than 50, contour or skeleton extraction fails, or features are nonfinite. All representations use the same validity mask; the 39 excluded units are strips, with no paragraph or writer removed. Photographs flagged for bottom-edge truncation remain included. The exclusion breakdown is in Supporting Table S1.

Six representations are compared: STROKE, HC-183, frozen ImageNet ResNet-18, EfficientNet-B0 and ViT-B/16 embeddings [@he2016resnet; @tan2019efficientnet; @dosovitskiy2021vit], and Stroke-CNN. No network is trained on these writers. The following specifications define the descriptors and checkpoints used throughout the evaluation.

## 3.2 Preprocessing and descriptor construction

All extraction is implemented in `code/wid_features.py`, `hc183.py` and `stroke_features.py`. Feature matrices are supplied so that the identification and control analyses can be reproduced without rerunning neural inference.

**HC-183.** A photograph is converted to grayscale and resized to width 768 with aspect ratio preserved using Lanczos interpolation. For grayscale intensity $I$ on the 0--255 scale, a Gaussian background estimate $B$ with standard deviation 17 pixels defines ink by $(B-I>17)\land(B>120)\land(I<205)$. A four-pixel border is cleared and a $2\times2$ binary opening is applied. Gradients of the mask smoothed with standard deviation 0.7 give nine-bin unsigned orientation histograms and ink density in each of $4\times4$ cells (160 values). Row and column ink projections each contribute seven quantiles (0, .1, .25, .5, .75, .9, 1), mean and standard deviation (18 values). Five area quantiles (.1, .25, .5, .75, .9) of connected components with at least three pixels complete the 183 features.

**STROKE.** The corresponding ink rule is evaluated at width 1,536, with background smoothing standard deviation 34, an eight-pixel cleared border and removal of connected objects smaller than 12 pixels. The text block is cropped using row and column ink fractions (thresholds 0.01 and 0.005), their 0.5th/99.5th percentiles and ten pixels of padding. Stroke width is twice the mean distance-transform value on the skeleton. A contour sampling length $L=\max(4,\operatorname{round}(2\,\mathrm{width}))$ controls feature scale. The 208 components comprise a 16-bin contour-direction histogram, a $12\times12$ hinge histogram, 12 turning-angle bins, 12 endpoint-direction bins, 12 skeleton-orientation bins, nine slant bins, and endpoint, junction and hole counts normalized by skeleton length. Static contours do not reveal pen movement order. The mask discards grayscale information, but this alone does not prove that every acquisition effect is removed.

**Frozen grayscale embeddings.** Grayscale images are resized to width 1,568 using Lanczos interpolation; the tile-selection mask is the nearest-neighbor resize of the 768-pixel HC-183 mask. Nonoverlapping $224\times224$ tiles with at least 1% ink are retained; the bottom edge is padded with median grayscale intensity. If no tile meets the threshold, all tiles are used. Grayscale is repeated into three channels and normalized by ImageNet means $(.485,.456,.406)$ and standard deviations $(.229,.224,.225)$. The checkpoints are torchvision `ResNet18_Weights.IMAGENET1K_V1`, `EfficientNet_B0_Weights.IMAGENET1K_V1` and `ViT_B_16_Weights.IMAGENET1K_V1`; final classification layers are removed. Mean tile embeddings have dimensions 512, 1,280 and 768. Models remain in evaluation mode without fine-tuning.

**Stroke-CNN.** The frozen ResNet-18 above instead receives black-on-white binary tiles from the cropped high-resolution mask. Bilinear resizing targets a stroke width of three pixels, with image dimensions at least 224 pixels; interpolation can produce antialiased edges. White padding completes the tile grid, and tiles require at least 2% ink; if none qualify, all are retained. Input normalization and mean aggregation follow the grayscale embeddings. This representation has 512 components. Its name does not indicate a separately trained handwriting network.

**Adjusted HC-183.** Seven grayscale statistics (mean, standard deviation and .05/.25/.5/.75/.95 quantiles), together with mask ink density, form nuisance covariates. A linear regression with intercept removes their fitted contribution from HC-183. Regression coefficients are fitted within each outer or inner SVM training split, then applied to its training and held-out samples. The adjusted representation is a sensitivity check, not a guarantee of causal removal of acquisition confounding.

## 3.3 Gallery construction and identity matching

Identification is closed-set over all writers. A shared-prompt query is compared with the other five prompts of every writer. Self-chosen-text queries are compared with every subset of $k=1,2,3$ or 6 shared reference paragraphs. For each reference configuration, feature means and standard deviations are fitted *only to its selected whole-paragraph gallery references*. The resulting transform is applied to references and queries. Each transformed vector is $L_2$-normalized; a gallery template is the normalized mean of that writer's reference vectors. Cosine similarity $s_w$ supplies the baseline ranking. Query pages and unused reference pages do not fit this transform. Four query sizes, five reference conditions and six representations give 120 settings.

Table 2 separates the closed identification gallery from the writer-held-out attribute-training data. Holding out a writer for attribute prediction does not remove that writer from the identity gallery.

  Component                               Permitted fitting data
  --------------------------------------- -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Image descriptors                       Fixed extraction rules and frozen ImageNet checkpoints; no writer-specific network training.
  ID standardization                      Only whole paragraphs selected as gallery references in the current configuration, from all enrolled writers. No queried paragraph or unused reference paragraph.
  ID templates                            Selected standardized, normalized references of the relevant gallery writer.
  Outer attribute model for writer $w$    Valid shared units of the same size from the other 41 writers. Neither shared nor self-chosen units of $w$.
  Inner attribute preprocessing and SVM   Training writers in each of five stratified group folds inside the outer 41-writer set. Nuisance regression, if used, is refitted here.
  Platt calibrator                        Out-of-fold SVM scores and labels of the outer 41 training writers. The query writer is absent.
  ID temperature for queries of $w$       ID-score negative log loss on query writers other than $w$ in the same setting, equally weighted by writer. No attribute probabilities enter temperature fitting.
  Fixed fusion                            Unit weight on log attribute probability; no fitted handedness weight.
  Nested weight                           Development folds inside the outer 41-writer set; probabilities regenerate with both the outer writer and development fold excluded from attribute fitting and calibration.

  : **Table 2.** Information available at each stage of primary evaluation. The query writer remains enrolled for identity matching, but contributes no samples to its query-handedness model.

## 3.4 Handedness prediction and score fusion

For each query writer and unit size, a linear SVM ($C=0.1$) is trained on valid shared units of the other 41 writers using HC-183. Standardization is fitted within training data. Five-fold stratified group cross-validation over those training writers supplies out-of-fold scores for logistic calibration [@platt1999probabilistic]. The outer writer's samples and labels are excluded from both stages. Let $p_q=\sigma(\ell_q)$ be the resulting left-hand probability. Adjusted HC-183 and STROKE predictors provide sensitivity checks.

These are cross-validated performance estimates. Writer identity assigns evaluation folds; it is not an input available to a deployed predictor. Deployment would freeze one attribute model and its calibration, temperature and fusion weight using separate development writers, then score every future query without choosing a model by its unknown identity. Training on the enrolled writers instead defines a different target. We do not estimate performance of one universal deployed predictor here.

In the *known* scenario, both query and gallery hands are supplied. This oracle restricts the top-ranked group to the query's true hand. In the primary *predicted-query* scenario, gallery hands are known, and $p_q$ is predicted. Probabilistic score fusion ranks writers by

$$f_w=s_w/T+\log p_q(h_w),\qquad\text{(1)}$$

where $p_q(1)=p_q$ and $p_q(0)=1-p_q$. For each setting, $T$ is selected from 31 logarithmically spaced values between $10^{-3}$ and 1 by minimizing identification negative log loss on the other query writers, with equal writer weights. Temperature fitting uses identity scores and writer targets, not attribute scores or the held-out query writer's outcomes. Gallery references remain available because this is closed-set identification.

Equation (1) is a score-combination rule. It has a Bayesian interpretation only under additional assumptions about the dependence of its evidence and the calibration prior. We do not assume these hold for two representations of the same handwriting. Hard filtering instead ranks all writers with the predicted hand ahead of the other hand, preserving identity-score order within each group. In this balanced gallery, an incorrect hand prediction makes top-1 success impossible.

## 3.5 Nested fusion-weight comparison

We additionally rank candidates by $s_w/T+\lambda\log p_q(h_w)$, selecting a nonnegative weight for each held-out query writer. The other 41 writers supply five development folds, regenerating attribute probabilities inside the outer exclusion to avoid selection bias [@cawley2010selection]. Each development attribute model and its inner five-fold writer-group calibrator exclude both the outer writer and the entire development fold. Equal-writer development negative log loss selects $\lambda\in\{0,.1,.2,.3,.5,.7,1,1.5,2,3,5\}$ at the same ID-only temperature as fixed fusion. Final query predictions use the outer model trained on the other 41 writers. The grid includes no attribute evidence ($\lambda=0$) and the fixed rule ($\lambda=1$); exact ties select the smaller weight. Known gallery metadata remain available in development and evaluation. The supplementary methods specify the complete nested fitting hierarchy.

## 3.6 Filtering identity and controls

Let $C$ denote correct attribute prediction and $K$ successful identification with the true query hand supplied. With both groups nonempty, filtered success is $C\cap K$. Thus

$$A_{\mathrm{filter}}=aP(K\mid C)=aA_{\mathrm{known}}+\operatorname{Cov}(\mathbf{1}_C,\mathbf{1}_K),\qquad\text{(2)}$$

where $a=P(C)$. Only if the two correctness events are independent does this reduce to

$$\begin{aligned}
 A_{\mathrm{filter}}&\approx aA_{\mathrm{known}},\\
 a_{\mathrm{break}}&\approx A_{\mathrm{none}}/A_{\mathrm{known}}.
 \end{aligned}\qquad\text{(3)}$$

We measure the conditional accuracies and approximation residuals rather than treating the product as universal. The two marginal accuracies alone permit the bounds

$$\begin{aligned}
 \max(0,a+A_{\mathrm{known}}-1)&\leq A_{\mathrm{filter}},\\
 A_{\mathrm{filter}}&\leq\min(a,A_{\mathrm{known}}).
 \end{aligned}\qquad\text{(4)}$$

Known balanced random writer partitions test how much oracle benefit comes from supplying an arbitrary correct partition. With 21 writers in each group, a query whose true writer is outranked by $j$ impostors survives an oracle filter only when all $j$ are assigned to the opposite group. Uniform balanced partitions therefore give the exact expectation

$$P(\mathrm{success}\mid j)=\begin{cases}\binom{21}{j}/\binom{41}{j},&0\leq j\leq21,\\0,&j>21.\end{cases}\qquad\text{(5)}$$

The same deterministic tie precedence is used for baseline ranks and partition success. The writer-weighted expectation provides an exact benchmark; 2,000 common Monte Carlo partitions characterize partition-to-partition variation across settings. Partition quantiles are not independent-cohort confidence intervals. Separately, 20 nearly hand-balanced random 21/21 writer labelings test whether an attribute model can predict arbitrary writer-associated labels. For each labeling, the positive group contains 10 or 11 of the 21 left-handed writers. Writer-held-out prediction is compared with enrolment-trained prediction, where other paragraphs of the same writer are available. The latter is an explicitly different generalization target; it is not a causal decomposition of genuine handedness accuracy.

## 3.7 Endpoints and inferential scope

Top-1 accuracy is averaged over valid query units and reference configurations within each writer, then equally over writers. Across-setting means give each of the 120 settings equal weight. Paired differences use 9,999 writer-level sign flips and 95% intervals from 10,000 hand-stratified writer bootstraps. The same sampled writer indices are used across settings for aggregate intervals, preserving their dependence. These intervals condition on the observed gallery and fitted predictions; they do not include new-gallery sampling or full model-refitting uncertainty.

Four settings were designated after an exploratory feasibility analysis of paragraph-level conditions: ResNet-18 and ViT-B/16 on self-chosen paragraphs with one reference and quarter strips with six references. These analyses were not preregistered. Holm correction covers the four fixed-fusion comparisons. Learned-weight comparisons against no fusion and fixed fusion form eight exploratory contrasts (four settings, two comparators), with Holm correction across all eight. Other robustness analyses are descriptive. No equivalence margin was specified, so nonsignificance is not interpreted as absence of a useful effect.

## 3.8 External validation across capture sessions

The CSAFE Handwriting Database provides an independent English-language evaluation [@crawford2020database; @crawford2026csafe]. Its version-2 release contains 475 writers; this differs from the 90-writer release described in the original article. The released survey identifies 48 left-handed, 423 right-handed, three ambidextrous and one unreported-hand writer. We include all 48 left-handed writers and 48 right-handed writers sampled with seed 20261108. Each selected writer contributes six session-1 pages (London Letter, LND, and Wizard of Oz, WOZ, each repeated three times) for auxiliary-model training. One LND page is the sole identity-gallery reference. Queries are separate session-2 pages using LND or WOZ, plus a session-3 WOZ page for secondary evaluation. Thus the primary conditions distinguish matching prompts from cross-prompt identification across sessions.

The original whole-unit extraction rules, SVM $C=0.1$, checkpoints and fusion grids are retained. We evaluate ResNet-18, ViT-B/16, HC-183 and STROKE identity features, with HC-183 as the primary handedness predictor and the two original sensitivity predictors. All 864 selected pages pass the validity rules. Each outer query writer's six training pages are excluded from its attribute model and calibrator. Nested development probabilities additionally exclude their validation writers; temperature and weight targets exclude the outer query writer. Known gallery references and hands remain permitted metadata. Numerical score scaling preserves the original logistic-calibration objective; Supporting Section S13 documents the convergence check and its implementation amendment.

Eight primary contrasts compare fixed and nested fusion with identity only for ResNet-18 and ViT-B/16 in both session-2 conditions, with Holm correction across all eight. Session 3, alternate attribute models and 20 refitted balanced 21/21 galleries are secondary checks. Confidence intervals condition on the fitted models and enrolled gallery. The CSAFE design was specified before inspecting its performance outcomes; it is an extension of the existing study, not a globally preregistered dataset-sampling plan. This is within-dataset refitting on an external cohort, not transfer of a frozen Bangla-trained predictor.

# 4 Results

## 4.1 Known and predicted query handedness

The primary predictor attains 68.7% accuracy on shared paragraphs and 65.5% on self-chosen paragraphs; averaging paragraph log-odds within a writer gives 73.8% over all eight paragraphs. The latter uses more writing and is not the accuracy available for a single query. Sensitivity predictors and strip accuracies appear in the supplement. For self-chosen paragraphs, the primary predictor has Brier score 0.204 and negative log loss 0.602; the reliability plots and intervals do not isolate miscalibration as the cause of fusion failure.

Known handedness improves top-1 accuracy by 6.1 percentage points on average (95% conditional interval \[+5.1, +7.1\]). Probabilistic score fusion changes it by -0.6 points \[-2.3, +0.9\], and hard filtering by -14.3 points \[-19.3, -9.5\]. Table 3 shows two representative reference conditions. Differences between representations are substantial, so an across-setting mean should not be interpreted as the expected accuracy of an arbitrary deployed system. Figure 1 shows the effects of query size and reference count.

  ---------------- ------------------------------ ------- -------- -------- ---------------------------------- ------- -------- --------
                     Self-chosen paragraph, $k=1$                             Self-chosen quarter strip, $k=6$                  
  Representation                             None   Known   Fusion   Filter                               None   Known   Fusion   Filter
  STROKE                                     92.3    95.4     93.3     62.3                               89.0    91.1     88.7     60.0
  HC-183                                     39.7    48.0     37.9     28.0                               33.0    42.9     32.1     25.6
  ResNet-18                                  62.9    67.9     62.1     42.3                               47.4    51.7     46.0     35.8
  EffNet-B0                                  51.4    60.3     51.0     36.7                               49.2    57.6     48.9     38.2
  ViT-B/16                                   68.1    73.8     68.3     45.8                               48.3    57.1     48.0     39.1
  Stroke-CNN                                 58.7    70.8     59.9     43.8                               10.7    15.5     10.7      8.3
  ---------------- ------------------------------ ------- -------- -------- ---------------------------------- ------- -------- --------

  : **Table 3.** Top-1 identification accuracy (%), equally weighted by writer. Known: query and gallery hands supplied. Fusion and Filter: query hand predicted using writer-held-out HC-183; gallery hands supplied. For $k=1$, all six reference choices are averaged within writer.

![**Figure 1.** Identification accuracy for self-chosen texts. (a) Query size with six whole-paragraph references per writer. (b) Reference count for paragraph queries. Solid: no handedness; dashed: known query and gallery hands; dotted: probabilistic fusion with predicted query handedness. (c) Writer-level paired differences for the four designated comparisons, with 95% conditional bootstrap intervals. Representations omitted from (a,b) remain in the full numerical results.](assets/performance.png){width="6.65in"}

Neither fixed nor nested learned weighting establishes a gain in the designated comparisons (Table 4). Across all settings, nested weighting changes top-1 by -0.08 points \[-1.13, +0.81\], and differs from fixed weighting by +0.49 points \[-0.15, +1.28\]. The median selected weight is 0.5; 1.3% of outer-writer/setting fits choose zero. This comparison permits downweighting redundant evidence without selecting a weight on the scored writer. It does not exclude small improvements or establish that other fusion families cannot help.

  ------------------- ---------------- ------------------------ ------------------------ ----------------------------- --------------------- -------
                                                                                                                        Holm $p$, Nested vs  
  Self-chosen query   Representation           Fixed \[95% CI\]        Nested \[95% CI\]   Nested $-$ Fixed \[95% CI\]         None           Fixed
  Paragraph, $k=1$    ResNet-18          -0.79 \[-3.77, +1.79\]   -0.20 \[-2.18, +1.59\]        +0.60 \[-0.99, +2.38\]         1.000          1.000
  Paragraph, $k=1$    ViT-B/16           +0.20 \[-2.58, +2.78\]   +1.19 \[-0.40, +2.78\]        +0.99 \[-0.60, +2.78\]         1.000          1.000
  Quarter, $k=6$      ResNet-18          -1.49 \[-3.87, +0.89\]   -0.89 \[-2.68, +0.89\]        +0.60 \[-1.49, +2.98\]         1.000          1.000
  Quarter, $k=6$      ViT-B/16           -0.30 \[-3.57, +2.68\]   -0.30 \[-2.98, +1.79\]        +0.00 \[-1.49, +1.19\]         1.000          1.000
  ------------------- ---------------- ------------------------ ------------------------ ----------------------------- --------------------- -------

  : **Table 4.** Four designated settings: fixed and nested learned-weight fusion. Entries are paired top-1 differences in percentage points with conditional 95% writer-bootstrap intervals. Fixed and Nested refer to differences from no fusion. Holm columns adjust the eight new learned-rule comparisons (four settings, two comparators). All comparisons are exploratory.

## 4.2 Representation-specific utility

Figure 2 compares writer-level means over the 20 settings per representation. Nested-fusion changes range from -0.64 to +0.42 points, and its correction relative to fixed fusion from +0.18 to +1.08 points. The overall mean conceals differences in oracle opportunity and prediction costs. These marginal exploratory intervals condition on the fitted gallery and predictions; they do not justify selecting a winning representation after inspection.

![**Figure 2.** Representation-specific utility, averaging 20 settings and all 42 writers equally. (a) Top-1 changes from identification without handedness. (b) Nested minus fixed fusion on a separate scale. Bars are marginal conditional 95% writer-bootstrap intervals with shared resampling across contrasts, not simultaneous confidence bands.](assets/representation_utility.png){width="6.65in"}

## 4.3 Conditional filtering utility

Across 456 combinations of setting and prediction model, the independence product tracks filtering accuracy ($r=0.996$), with mean absolute error 1.5 points and signed bias -0.9 points. These combinations reuse writers and are not independent replications. The residual analyses in Fig. 3 and supplementary conditional accuracies show why the exact identity in Eq. (2) is preferable when predictions are available. For HC-183 identification on self-chosen paragraphs with one reference, the product predicts 31.4% versus 28.0% observed. Known-hand identification succeeds on 42.7% of queries with correct hand predictions and 58.0% with incorrect predictions. Agreement over a broad performance range can therefore conceal decision-relevant residuals.

For self-chosen queries, the approximate break-even accuracy has median 86% and range 55--98%. This is a property of the tested identifiers and reference conditions under Eq. (3), not a general handedness requirement. A stronger baseline leaves less room for an imperfect filter to help. A known balanced random partition improves accuracy by 6.9 points on average. Known handedness differs from that expectation by -0.8 points (95% conditional interval \[-1.8, +0.2\]). Thus the oracle benefit is consistent with useful partition information, without a distinctive advantage over the random-partition benchmark.

![**Figure 3.** Filtering diagnostics. (a) Observed filtering accuracy versus the independence product for all evaluated combinations; the line indicates equality. (b) Signed approximation residuals by representation, exposing discrepancies hidden by pooled correlation. (c) Known-handedness gain versus the expected gain of a known balanced random writer partition. These are descriptive comparisons on the observed cohort, not independent dataset replications.](assets/diagnostics.png){width="6.65in"}

## 4.4 Attribute calibration and prediction information

Table 5 reports writer-held-out accuracy and proper scoring rules for the primary and sensitivity predictors on self-chosen full paragraphs. Accuracy, Brier score and log loss measure different aspects of the predictions; none alone specifies how those scores interact with identity margins. Reliability-bin plots and their uncertainty are retained in Supporting Figure S1. The adjusted HC-183 predictor removes fitted grayscale and ink-density covariates inside each training split; it is a sensitivity analysis rather than a causal isolation of acquisition effects.

  Model               Accuracy \[95% CI\], %   Brier score   Negative log loss
  ----------------- ------------------------ ------------- -------------------
  HC-183                 65.5 \[53.6, 77.4\]         0.204               0.602
  HC-183 adjusted        61.9 \[50.0, 73.8\]         0.244               0.684
  STROKE                 48.8 \[35.7, 61.9\]         0.286               0.775

  : **Table 5.** Writer-held-out prediction and proper scoring rules on self-chosen paragraph queries. Intervals are conditional writer bootstraps.

## 4.5 Writer-associated information and exploratory extensions

With enrolment training, paragraph handedness accuracy on self-chosen texts is 82.1%, compared with 65.5% for held-out writers. Under enrolment training, the random-label accuracies range from 52.4% to 76.2%. This demonstrates that writer-associated information can support arbitrary labels. It diagnoses a change in the learning problem under writer overlap; it does not demonstrate leakage in the primary writer-held-out pipeline, quantify an identity contribution to genuine handedness accuracy, or rule out acquisition signals. Fixed balanced labels and leave-one-writer-out training also create a structured null; the full random-label distributions are reported without treating a nominal 50% reference as an exact null distribution.

Figure 4 makes the two writer-information conditions explicit. The random label stays fixed for a writer, while the training/test relation changes. The distribution under enrolment training is therefore a control for writer-associated predictability, not proof of a leakage mechanism in the primary writer-held-out pipeline.

![**Figure 4.** Writer-overlap diagnostic. (a) Schematic split designs: each writer has one fixed random label; training can exclude that writer or contain their shared paragraphs before testing their own text. (b) Own-paragraph prediction of genuine handedness and 20 random writer labelings. The 50% line is descriptive, not a test against the structured held-out null. Gray points denote labelings. This control does not establish leakage in the primary writer-held-out experiment.](assets/label_controls.png){width="6.65in"}

Exploratory both-sided descriptor agreement improves top-1 by +1.08 points \[+0.50, +1.72\] over the 96 self-chosen-text settings (Supplement S6). It replaces known gallery-hand labels with predicted reference descriptors and changes the score rule. An impostor reference model can include the query writer's enrolment pages, so this enrolled-cohort diagnostic does not establish unseen-cohort handedness utility or contradict the primary protocol. Shared-text queries are excluded because those query pages enter impostor-model fitting. Synthetic and reduced-gallery sensitivities remain conditional on their stated models and this cohort.

## 4.6 Within-cohort robustness

Across four grouped calibration partitions, handedness accuracy on self-chosen paragraphs ranges from 64.3% to 66.7%; 2.4% of these queries change predicted hand at least once. Mean gains from fixed fusion range from -0.57 to -0.50 points. Temperature selection reaches the lower grid endpoint in 0.8% and the upper endpoint in 2.6% of fits. Halving or doubling the selected temperatures gives mean fusion changes from -1.81 to +0.04 points. Omitting each writer from class-balanced evaluation averaging yields mean changes under nested fusion ranging from -0.24 to +0.22 points, while hard filtering remains substantially harmful. The supplement reports loss profiles, paired intervals and every writer's influence. These diagnostics retain the observed gallery and do not constitute independent-dataset validation.

Table 6 gives the calibration-partition results rather than treating one random fold assignment as definitive. Figure 5 places temperature selection, learned weights and evaluation-writer influence beside those prediction changes. The diagnostics vary specific components of the fitted pipeline and retain the enrolled gallery; their scope differs from refitting on a new cohort or evaluating a new capture session.

    Seed   Hand accuracy \[95% CI\], %   Brier     NLL   Fixed fusion change \[95% CI\]
  ------ ----------------------------- ------- ------- --------------------------------
    2031           65.5 \[53.6, 77.4\]   0.204   0.602           -0.57 \[-2.37, +0.91\]
    2032           66.7 \[54.8, 78.6\]   0.201   0.594           -0.50 \[-2.13, +0.83\]
    2033           64.3 \[52.4, 76.2\]   0.202   0.599           -0.56 \[-2.33, +0.90\]
    2034           65.5 \[53.6, 77.4\]   0.205   0.606           -0.52 \[-2.20, +0.86\]

  : **Table 6.** Calibration-partition sensitivity. Attribute metrics use self-chosen paragraphs; fusion changes average all 120 settings and are in percentage points. Accuracy intervals and fusion intervals are conditional writer bootstraps.

![**Figure 5.** Within-cohort diagnostics. (a) Selected temperature distributions across outer-writer/setting fits; boxes show quartiles, whiskers extend to 1.5 IQR and points show values outside them. (b) Learned-weight frequencies across the 5,040 fits. (c) Fixed-fusion mean differences and conditional writer-bootstrap intervals for each calibration partition. (d) Changes in mean gain after each evaluation-writer omission. None of these panels represents independent-dataset validation.](assets/fusion_robustness.png){width="6.65in"}

## 4.7 Decision margins and opposing fusion effects

Fusion preserves candidate ranking within each handedness group. Define the true-writer margin against the best opposite-hand candidate and the signed attribute log-odds as

$$\begin{aligned}m_q&=\frac{s_{w^*}-\max_{h_w\ne h_{w^*}}s_w}{T},\\r_q&=\log\frac{p_q(h_{w^*})}{p_q(1-h_{w^*})}.\end{aligned}\qquad\text{(6)}$$

Away from score ties, identification succeeds exactly when the true writer leads its own hand group and $m_q+\lambda r_q>0$. Thus fusion cannot repair a wrong within-group ordering. The true identity and handedness enter only this retrospective diagnostic; they are not available inputs to the operational query predictor.

A rescued decision is incorrect before fusion and correct afterwards; a damaged decision changes in the opposite direction. Their difference equals the net top-1 change. Across the 120 settings, fixed fusion rescues 2.02% of decisions and damages 2.59%; nested weighting rescues 1.33% and damages 1.42%. Figure 6 shows these opposing effects with conditional writer-bootstrap intervals and illustrates the exact margin boundary for HC-183 paragraph queries with one reference.

Direct candidate-score ranking and the analytical decision reconstruction agree on all 664,680 evaluated query/reference decisions, with no exact boundary ties. Writer-level fused accuracies independently match the fixed and nested outputs. This deterministic consistency check explains the fitted score rule; it does not add independent experimental observations.

![**Figure 6.** Decision changes and margin diagnostic. (a) Rescued and damaged decisions under fixed and nested fusion, averaging all 120 settings with equal writer weights; bars are marginal conditional 95% writer-bootstrap intervals. (b) HC-183, self-chosen full paragraphs and one reference, fixed fusion: only decisions for which the true writer leads its hand group are plotted. The vertical line is the baseline boundary $m=0$; the dashed line is the fusion boundary $m+r=0$. Points above the dashed line succeed after fusion. Each point is a query/reference choice; repeated choices are correlated. The true-hand axes are retrospective diagnostics, not available prediction inputs.](assets/margin_diagnostic.png){width="6.65in"}

## 4.8 External validation on CSAFE

The primary handedness predictor reaches 67.71% accuracy on session-2 LND queries (conditional 95% interval 58.33--77.08%), 58.33% on session-2 WOZ (48.96--67.71%), and 65.62% on session-3 WOZ (56.25--75.00%). Balanced and ordinary accuracy coincide in this 48/48 cohort. These values separate attribute predictability from the identity-ranking changes in Table 7 and Fig. 7.

With matching prompts across sessions, known handedness increases ResNet-18 identification from 61.46% to 68.75%, whereas fixed and nested fusion both achieve 62.50%. ViT-B/16 changes from 64.58% to 62.50% under either fusion rule. With different prompts, ResNet-18 changes from 54.17% to 48.96% under fixed fusion and remains at 54.17% under nested fusion; ViT-B/16 changes from 55.21% to 54.17%. None of the eight primary contrasts is significant after Holm correction (all adjusted $p\geq0.4888$). This does not establish equivalence. Individual percentile-bootstrap intervals are conditional, are not simultaneous, and are not inversions of the discrete sign-flip tests.

Hard filtering reduces primary neural identification by 17.71--25.00 percentage points across the two session-2 conditions. The STROKE handedness sensitivity model attains 67.71% and 60.42%, while adjusted HC-183 attains 34.38% in both; these differences are retained in the supplement. Negative fitted Platt slopes occur in 24 of 96 adjusted-model outer fits, so its pooled raw and calibrated score sets differ substantially. This sensitivity failure does not identify a causal acquisition mechanism. Session-3 comparisons and the 20 overlapping gallery subsamples further characterize variability without adding independent dataset replications.

  Prompt      Identity features     Identity only   Known hand   Hard filter   Fixed   Nested
  ----------- ------------------- --------------- ------------ ------------- ------- --------
  Same        ResNet-18                     61.46        68.75         43.75   62.50    62.50
  Same        ViT-B/16                      64.58        67.71         39.58   62.50    62.50
  Different   ResNet-18                     54.17        60.42         31.25   48.96    54.17
  Different   ViT-B/16                      55.21        61.46         34.38   54.17    54.17

  : **Table 7.** CSAFE session-2 top-1 identification accuracy (%) with the primary HC-183 handedness model and one session-1 reference per writer. The gallery contains 48 writers of each hand. "Same" and "Different" refer to the query prompt relative to the LND reference. All eight fixed/nested primary contrasts have Holm-adjusted $p\geq0.4888$.

![**Figure 7.** External CSAFE validation using separate gallery/query sessions and the primary HC-183 handedness model. (a) Top-1 identification in the 96-writer gallery. (b) Fixed and nested changes from identity only with conditional 95% writer-bootstrap intervals. Both session-2 conditions belong to the eight-comparison primary family. Intervals retain fitted models and the gallery; they are not multiplicity-adjusted.](assets/csafe_identification_utility.png){width="6.65in"}

# 5 Discussion

The methodological lesson is that oracle gains and attribute accuracy do not establish downstream identification utility. Correct metadata reveal an opportunity to improve ranking; an attribute predictor must still supply useful evidence at the decisions where it matters. In BanglaHand, neither calibrated fixed fusion nor properly nested learned weighting establishes a gain in the designated comparisons. The CSAFE validation extends this distinction to a separate population, English writing and distinct source pages and sessions: even the 67.71% same-prompt handedness predictor yields no reliably demonstrated primary fusion gain. The cross-prompt and session-3 differences show why this result must remain tied to its declared query conditions and fitted rules. The random-partition benchmark further shows that the oracle advantage is consistent with correct binary partition information, without a distinctive handedness advantage. Hard filtering turns an incorrect hand prediction into an unrecoverable top-1 error. Soft fusion avoids that veto, but calibration alone cannot ensure that overlapping evidence will improve the identity ranking.

Table 8 makes the methodological comparison explicit. Prior attribute studies establish whether handwriting supports demographic prediction, while identity-focused studies develop the matcher itself. Our contribution is to intervene on the auxiliary information while holding the identity comparison fixed. The known-hand and random-partition controls quantify an opportunity that a predictor may fail to realize; nested weighting then tests whether a flexible, calibrated use of that predictor changes the finding. This is a complementary evaluation question, not evidence that our representations are superior to the recognition systems in Table 1.

  Study                                                                  H    WD       W        O    R    F    D   Data scope             S
  --------------------------------------------------------------------- ---- ---- ------------ ---- ---- ---- ---- --------------------- ----
  Al-Maadeed et al. [@almaadeed2016handedness]                           Y    U        --       NR   NR   NR   NR  QUWI subset            NR
  Morera et al. [@morera2018handedness]                                  Y    Y        --       NR   NR   NR   NR  IAM and KHATT          NR
  Rahmanian and Shayegan [@rahmanian2021handedness]                      Y    U        --       U    U    U    U   IAM and KHATT          U
  Bulacu/Schomaker studies [@bulacu2007text; @schomaker2007codebooks]    --   --      ID/V      --   --   --   --  Identity benchmarks    U
  He and Schomaker [@he2020fragnet]                                      --   --       ID       --   --   --   --  Four benchmarks        U
  Paul et al. [@paul2024bangla]                                          --   --       V        --   --   --   --  Bangla, 100 writers    NR
  Companion preprint [@prappo2026handedness]                             Y    Y        --       --   --   --   --  BanglaHand             --
  Present study                                                          Y    Y    $\Delta$ID   Y    Y    Y    Y   BanglaHand; CSAFE      Y

  : **Table 8.** Methodological coverage of representative cited approaches. H: handedness prediction; WD: writer-disjoint attribute evaluation; W: writer endpoint (ID = identification, V = verification, $\Delta$ID = paired identification change); O: known-versus-predicted handedness comparison; R: known random-partition benchmark; F: calibrated fusion with nested weight selection; D: joint-error/margin and random-writer-label diagnostics; S: separate gallery/query capture-session evaluation. Y = evaluated; -- = outside the reported task; NR = not reported as the specified evaluation in the inspected source; U = detailed protocol unverified from the accessible record. Counts describe study datasets, not independent statistical replications.

The interaction between confidence and identity margins determines which decisions fusion changes. The decision-level analysis in the Results distinguishes rescued identifications from damaged ones and verifies the exact ranking boundary.

Predictable handedness [@prappo2026handedness] is compatible with these outcomes: attribute errors can erase oracle gains, and attribute cues may duplicate identity evidence. Our fusion results do not imply that handedness is unidentifiable.

Nested weighting reduces the mean loss relative to fixed fusion. Since the weight minimizes development log loss, this result concerns that explicit selection objective rather than an exhaustive search for top-1-optimal fusion. The study does not identify a universal attribute-accuracy threshold for soft fusion. The filtering identity serves as a diagnostic of the joint correctness distribution; its product approximation requires independence. Even equal marginal accuracies need not imply equal utility. Likewise, high attribute accuracy on enrolled writers may be appropriate for a declared closed-set application, while overstating performance on unseen writers. The random-label control diagnoses this distinction without assigning a causal fraction to identity.

The practical implication is a testable evaluation sequence. An auxiliary-attribute system should first establish an oracle opportunity, then compare predicted fusion with an identity-only baseline on the same queries, selecting its calibration and fusion parameters without the evaluated writer's outcomes. Decision-level rescues and damage show where the score change helps, while writer-overlap controls distinguish an attribute-generalization question from an enrolment-conditioned one. Separate capture sessions then test whether the conclusion survives a change in source pages and acquisition. Our controls make these questions answerable even when the final fusion gain is inconclusive. Their broader value is an auditable decision about whether to deploy an auxiliary score, rather than treating attribute accuracy as a proxy for identification benefit.

# 6 Scope and limitations

BanglaHand provides a deliberately balanced cohort of 21 left- and 21 right-handed writers under controlled acquisition [@banglahand2026data]. Recruiting the left-handed group required substantial effort; this balance enables comparisons without majority-class dominance. The estimates describe this 42-writer closed gallery from one region, device and photographic session. The companion study shares this cohort and is not independent validation. CSAFE supplies an independent population and separate-session scans, extending the evaluation beyond the original photographic acquisition; it does not establish transfer to all scripts, devices or deployment conditions. Each writer contributes two self-chosen paragraphs, and strips vary text quantity without adding independent writers.

The uncertainty intervals condition on the enrolled gallery and fitted predictions; they do not propagate full model-refitting or population-sampling variability. BanglaHand calibration partitions vary one fitting stage and writer omission changes evaluation averaging while retaining its gallery. CSAFE tests new sessions and refits models in overlapping gallery subsamples, but those dependent subsamples do not provide independent-cohort confidence intervals. Its whole-page, one-reference conditions do not reproduce every BanglaHand text-size/reference setting, and same-prompt matching must be distinguished from cross-prompt identification.

The computational comparison uses handcrafted descriptors and frozen ImageNet representations, allowing reproducible evaluation without task-specific network training. Handwriting-specialized training, fine-tuning, stronger identifiers and alternative fusion objectives remain outside this comparison; their different error patterns or margins could change fusion utility. Handedness is the sole real auxiliary attribute tested. BDSHWA records handedness for Bengali--English online handwriting but includes one left-handed participant [@bdshwa2026], so it cannot supply a balanced offline replication.

The primary predicted-query protocol assumes reliable gallery handedness. Knowing a suspect's handedness does not establish the hand of the questioned writer. The known scenario assumes both are reliably available. Closed-set top-1 ranking does not evaluate open-set rejection, evidential likelihood ratios, disguised writing, or non-dominant-hand writing. Application to forensic candidate prioritization would require validation under those operational conditions and explicit treatment of uncertain attributes and error costs. Predicted handedness alone cannot justify exclusion.

# 7 Conclusions

The controlled comparisons show why oracle gains and attribute accuracy cannot substitute for measured identification utility. On BanglaHand, correct query and gallery handedness improves writer ranking, whereas fixed and properly nested learned-weight fusion of the tested query predictor produce no reliably demonstrated gain over identification alone, and hard filtering causes substantial losses. Independent CSAFE writers and capture sessions reproduce the distinction between oracle opportunity and achieved predicted-attribute utility in the tested comparisons. The analytical filtering diagnostic connects success to joint correctness, while the oracle, predicted-attribute and random-label controls distinguish different sources of apparent utility. Random-label prediction shows that enrolled-writer attribute accuracy can contain substantial writer-associated information. Further datasets, fitting objectives and deployment conditions can assess the remaining limits of transfer; the present results do not establish a universal absence of handedness utility. Future fusion methods should be evaluated on downstream utility, calibration and error dependence, with the query/gallery information available at deployment stated explicitly.

# Data and code availability {#data-and-code-availability .unnumbered}

The BanglaHand code, frozen features, setting manifests, numerical results and self-contained supplementary methods are archived as [reproducibility ZIP v1.6.0](https://github.com/LegendaryBeast/Bangla-writer-identification-research/releases/download/v1.6.0/wid-reproducibility-v1.6.0.zip) at <https://github.com/LegendaryBeast/Bangla-writer-identification-research/releases/tag/v1.6.0>. The release supplies exact checkpoint references and table/figure reproduction commands. Code is MIT licensed; results and frozen features are CC BY 4.0. Photographs are excluded; obtain BanglaHand separately from Mendeley Data (doi:10.17632/n4sjstdzfy.1) [@banglahand2026data].

The separate CSAFE extension, `csafe-replication-2026-11-08.zip`, is supplied with the submission and contains its complete code, frozen features, protocols, fitting manifests and reproduction instructions. It is distinct from public release v1.6.0. CSAFE version 2 is obtained separately at doi:10.25380/iastate.10062203.v2 under CC BY 4.0 [@crawford2026csafe]; original images and checkpoint binaries are excluded from the analysis archive.

# Author contributions {#author-contributions .unnumbered}

Tanzim Hasan Prappo: Conceptualization, Methodology, Software, Validation, Writing -- original draft, Writing -- review & editing. M. Shahidur Rahman: Supervision, Writing -- review & editing.

# Ethics statement {#ethics-statement .unnumbered}

For BanglaHand, informed consent was obtained voluntarily from all participants before data collection. Handwriting samples use pseudonymous writer identifiers. Formal institutional ethical approval was not required under the policies of Shahjalal University of Science and Technology because data collection involved non-invasive, voluntary handwriting tasks with no personal or health identifiers recorded. Writers received an honorarium paid by the authors from personal funds.

The external evaluation reuses publicly released, pseudonymous CSAFE images and handedness labels under CC BY 4.0 [@crawford2020database; @crawford2026csafe]. No additional participants were recruited or contacted for that secondary analysis.

# Funding {#funding .unnumbered}

This research received no specific grant.

# Conflicts of interest {#conflicts-of-interest .unnumbered}

The authors declare no conflict of interest regarding the publication of this article.

# Declaration of generative AI use {#declaration-of-generative-ai-use .unnumbered}

An AI assistant was used to audit analysis code, and assist with drafting. The authors are responsible for verifying the analyses, references and final text, and for all publication claims.

# Supplementary material {#supplementary-material .unnumbered}

Supporting information contains self-contained feature and preprocessing specifications, the fitting hierarchy, filtering and random-partition derivations, calibration and random-label controls, nested fusion selection, uncertainty and sensitivity analyses, decision margins, the complete 120-setting BanglaHand grid, the external CSAFE protocol and full results, and reproduction instructions.
