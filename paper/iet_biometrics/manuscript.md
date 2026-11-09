---
title: "Does predicted handedness improve writer identification? Oracle gains versus achieved utility in Bangla and English handwriting"
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

Soft-biometric attributes are usually judged by how accurately they can be predicted, yet identification depends on whether a predicted attribute improves the rank of the true writer. We measure this downstream utility for handwriting handedness with a paired protocol that fixes the gallery and identity scores and varies only the auxiliary information: known hand, a known random partition, hard filtering, and calibrated fixed or nested learned-weight fusion. Handedness models exclude the queried writer from training and calibration, and identity normalization uses gallery references only. On BanglaHand (42 writers, six representations, 120 settings), known handedness raises top-1 accuracy by 6.1 percentage points, whereas an arbitrary known 21/21 partition raises it by 6.9. Identity errors cluster within hand: when one impostor outranks the true writer, it has the opposite hand in 34--37% of cases for neural matchers, against 51% for a random impostor. With query handedness predicted at 65.5% accuracy, hard filtering lowers top-1 accuracy by 14.3 points and nested fusion changes it by $-0.08$ points; the 90% interval ($-0.96$ to $+0.67$) excludes recovering more than 11% of the oracle gain. On 96 CSAFE writers, with gallery and queries from separate capture sessions, none of eight primary fusion contrasts shows a gain, although this cohort resolves effects only beyond about $\pm3$ points. Attribute accuracy, and even a correct attribute, therefore does not establish identification utility. The protocol separates opportunity, specificity, achieved utility, mechanism and writer-associated information, and applies to other auxiliary attributes.

**Keywords:** Writer identification; Soft biometrics; Handedness; Score fusion; Evaluation protocol; Bangla handwriting

# 1 Introduction

A soft biometric describes a person without uniquely identifying them. Gender, age and handedness can supplement identity matching by changing candidate scores or restricting a gallery [@jain2004soft; @dantcheva2016soft]. Handwriting studies have predicted such attributes from static images [@liwicki2011gender; @siddiqi2015gender; @bouadjenek2016soft; @almaadeed2016handedness; @morera2018handedness; @rahmanian2021handedness]. For identification, however, the relevant question is different: does the predicted attribute move the true writer up the candidate list?

Three issues separate attribute accuracy from identification utility. First, an incorrect hard attribute decision can exclude the true writer, so filtering depends on how attribute errors coincide with identification errors, not on either accuracy alone. Soft fusion avoids automatic exclusion but depends on calibration and on whether attribute evidence adds information to identity evidence extracted from the same image. Second, a correct attribute can help simply because any correct partition shrinks the candidate set; an oracle gain is therefore not evidence that the attribute itself is informative. Third, handwriting carries writer-specific style [@bulacu2007text; @schomaker2007codebooks], so an attribute predictor evaluated on previously seen writers can exploit identity rather than the attribute, and does not estimate generalization to unseen writers [@kaufman2012leakage].

We address these issues for handedness with a paired evaluation in which the gallery, identity scores and queries are fixed and only the auxiliary information changes (Fig. 1). The contributions are:

-   **A reusable evaluation protocol** that separates the opportunity offered by a correct attribute, its specificity against a known random partition with an exact expectation, the utility achieved by calibrated fixed and properly nested fusion, the decision-level mechanism, and writer-associated information.

-   **Evidence that the oracle gain is not handedness-specific.** Known handedness yields 0.8 points less than a random partition, because the impostors that outrank the true writer tend to share its hand.

-   **A precise null result for predicted handedness.** Across 120 BanglaHand settings, nested fusion stays within $\pm1$ point of identity-only ranking, and hard filtering is consistently harmful.

-   **A cross-session English check** on 96 CSAFE writers [@crawford2020database; @crawford2026csafe] that reaches the same conclusion, with its precision limits stated.

The study reuses BanglaHand [@banglahand2026data] and the preprocessing and representations of a companion study on handedness classification [@prappo2026handedness]; it asks a different question with a different endpoint. Section 2 reviews related work, Section 3 defines the protocol, Section 4 reports the results, and Sections 5--7 discuss implications, limitations and conclusions.

![**Figure 1.** Evaluation protocol. (a) For every query, the identity scores $s_w$ and the gallery are fixed; only the auxiliary information supplied to the ranking rule changes. The handedness model never sees the queried writer during training or calibration. (b) The six questions the protocol answers in sequence, each with its control: filled steps 1--3 estimate utility, and open steps 4--6 explain it and test its scope.](assets/protocol.png){width="6.65in"}

# 2 Related work

## 2.1 Soft biometrics and score fusion

Score-level fusion is the standard way to combine biometric evidence [@kittler1998combining; @ross2003fusion]. Jain et al. combine demographic information with a primary biometric, and Dantcheva et al. distinguish attribute estimation, database indexing and recognition applications [@jain2004soft; @dantcheva2016soft]. In face recognition, demographic attributes and facial marks improve matching and retrieval [@park2010soft]. These results motivate auxiliary evidence, but its value depends on the available metadata, prediction uncertainty and the primary matcher: a correct attribute narrows a candidate set, whereas a predicted attribute can also demote the true candidate. Attribute recognition and identification improvement therefore require different endpoints.

## 2.2 Handedness and demographic prediction

Handwriting studies have predicted gender and other demographic traits from online dynamics, offline shape and texture, and their combinations [@liwicki2011gender; @siddiqi2015gender; @bouadjenek2016soft; @hassaine2013icdar]. Al-Maadeed et al. classify handedness on a 121-writer QUWI subset with Arabic and English documents using handcrafted features and fuzzy conceptual reduction [@almaadeed2016handedness]. Morera et al. use CNN word predictions and line-level voting on IAM and KHATT with separate training and test individuals [@morera2018handedness]; their overall handedness accuracies of 90.70% and 70.91% coincide with left-handed recalls of only 63.40% and 38.64%, showing why accuracy must be read alongside class composition. Rahmanian and Shayegan report 99.14% KHATT handedness accuracy with deep CNNs [@rahmanian2021handedness]; the accessible record does not establish whether writers were disjoint across splits. The companion study reports 77.4% balanced accuracy for writer-independent Bangla handedness by aggregating SVM margins over six paragraphs [@prappo2026handedness]. These studies establish predictability. None measures the change in an identity ranking that a predicted attribute causes (Table 1).

## 2.3 Writer recognition and evaluation controls

Text-independent writer identification models individual style through contour directions, hinges, allographs and contour-fragment codebooks [@bulacu2007text; @schomaker2007codebooks], and through learned local descriptors and deep fragment networks [@christlein2017unsupervised; @he2020fragnet]. For Bengali script, text-independent identification [@chanda2010bengali] and verification with genetic feature selection [@paul2024bangla] have been reported, and BanglaWriting and the Boise State collection support related recognition research [@mridha2021banglawriting; @majid2018bangla]. Our frozen representations are controlled comparators, not a claim to outperform handwriting-specialized networks.

Evaluation design determines what an apparent improvement means. Writer separation addresses generalization to new individuals but does not show that an attribute adds information beyond the identity matcher. Leakage analysis requires the information available at each fitting stage to be specified [@kaufman2012leakage], fusion-weight selection must be separated from evaluation to avoid selection bias [@cawley2010selection], and a nonsignificant difference supports absence of a meaningful effect only through an equivalence analysis [@lakens2017equivalence]. We combine these controls in one protocol.

  Study                                                                                             Data and representation                                                       Evaluated endpoint                                             Relation to the present question
  ------------------------------------------------------------------------------------------------- ----------------------------------------------------------------------------- -------------------------------------------------------------- --------------------------------------------------------------------------------------------------------
  Jain et al.; Dantcheva et al.; Park and Jain [@jain2004soft; @dantcheva2016soft; @park2010soft]   Primary biometrics with auxiliary personal attributes                         Recognition, indexing, retrieval and soft-biometric fusion     Motivate auxiliary information; utility of a *predicted* handwriting attribute is a separate question.
  Al-Maadeed et al. [@almaadeed2016handedness]                                                      QUWI, 121 writers; Arabic/English; handcrafted features and fuzzy reduction   Handedness classification and feature reduction                Attribute predictability; downstream ranking change not measured.
  Morera et al. [@morera2018handedness]                                                             IAM/KHATT; CNN words, line voting; separate train/test writers                Handedness accuracy 90.70%/70.91%; left recall 63.40%/38.64%   Shows the importance of class composition and prediction unit.
  Rahmanian and Shayegan [@rahmanian2021handedness]                                                 IAM/KHATT; three deep CNN architectures                                       Gender and handedness; 99.14% KHATT handedness accuracy        Strong attribute result; writer split not verifiable from the record.
  Bulacu and Schomaker; Schomaker et al. [@bulacu2007text; @schomaker2007codebooks]                 Textural/allographic descriptors and contour-fragment codebooks               Writer identification and verification                         Identity-matching representations rather than attribute utility.
  Christlein et al.; He and Schomaker [@christlein2017unsupervised; @he2020fragnet]                 Learned local descriptors; feature-pyramid and fragment pathways              Writer identification and retrieval                            Learned identity representations; no handedness fusion.
  Chanda et al.; Paul et al. [@chanda2010bengali; @paul2024bangla]                                  Offline Bengali/Bangla handwriting                                            Writer identification; verification (94.54% mean accuracy)     Bangla-script identity matching; no auxiliary attribute.
  Companion preprint [@prappo2026handedness]                                                        BanglaHand, 42 balanced writers; shared-prompt aggregation                    Writer-independent handedness and photometric sensitivity      Same cohort and representations; a different task and endpoint.

  : **Table 1.** Representative related work, organized by the decision evaluated. Reported results retain their original endpoints and are not a common performance ranking.

# 3 Materials and methods

## 3.1 Data and query units

BanglaHand contains 336 paragraph photographs from 42 writers (21 left-handed), with common materials and one controlled photographic session [@banglahand2026data]. Each writer supplied six shared prompts and two self-chosen texts. A query is a whole paragraph or one of 2, 4 or 8 horizontal strips (about 3.5, 1.75 or 0.9 text lines), with boundaries moved to low-ink rows near equally spaced cuts. Of 5,040 candidate units, 39 strips fail fixed feature-validity rules; no paragraph or writer is removed. Strip construction, validity rules and the exclusion breakdown are given in Supporting Section S1.

## 3.2 Representations

Six identity representations are compared (Table 2): two handcrafted descriptors and four frozen ImageNet networks [@he2016resnet; @tan2019efficientnet; @dosovitskiy2021vit]. No network is trained or fine-tuned on these writers. HC-183 is also the primary handedness predictor; an adjusted HC-183, from which a linear regression on seven grayscale statistics and ink density is removed within every training split, and STROKE serve as sensitivity predictors. Exact extraction rules, thresholds and checkpoint identifiers are given in Supporting Section S2, and frozen feature matrices are released so that every analysis can be reproduced without neural inference.

  Representation   Input                                                 Dimension Content
  ---------------- --------------------------------------------------- ----------- -------------------------------------------------------------------------------------------------------------------------------------------
  STROKE           Ink mask at 1,536 px width, cropped text block              208 Contour-direction, hinge, turning-angle, endpoint-direction, skeleton-orientation and slant histograms; endpoint, junction and hole rates
  HC-183           Ink mask at 768 px width                                    183 Orientation histograms and ink density in $4\times4$ cells; row and column projection statistics; connected-component area quantiles
  ResNet-18        Grayscale tiles with at least 1% ink                        512 Frozen ImageNet ResNet-18 embedding
  EffNet-B0        Grayscale tiles with at least 1% ink                      1,280 Frozen ImageNet EfficientNet-B0 embedding
  ViT-B/16         Grayscale tiles with at least 1% ink                        768 Frozen ImageNet ViT-B/16 embedding
  Stroke-CNN       Binary stroke tiles rescaled to 3 px stroke width           512 Frozen ResNet-18 applied to binarized strokes; not a separately trained handwriting network

  : **Table 2.** Representations. All are fixed before evaluation; the neural networks use torchvision ImageNet checkpoints in evaluation mode. Tiles are $224\times224$ pixels and embeddings are averaged over tiles.

## 3.3 Gallery construction and identity matching

Identification is closed-set over all 42 writers. A shared-prompt query is compared with the other five prompts of every writer. Self-chosen-text queries are compared with every subset of $k=1,2,3$ or 6 shared reference paragraphs. For each reference configuration, feature means and standard deviations are fitted *only to the selected whole-paragraph gallery references* and applied to references and queries. Vectors are $L_2$-normalized, a gallery template is the normalized mean of a writer's references, and cosine similarity $s_w$ gives the baseline ranking. Four query sizes, five reference conditions and six representations give 120 settings. Table 3 lists the data permitted at every fitting stage; holding a writer out of attribute training does not remove that writer from the identity gallery.

  Component                               Permitted fitting data
  --------------------------------------- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Image descriptors                       Fixed extraction rules and frozen ImageNet checkpoints; no writer-specific training.
  ID standardization                      Only whole paragraphs selected as gallery references in the current configuration. No queried paragraph or unused reference paragraph.
  ID templates                            Selected standardized, normalized references of the relevant gallery writer.
  Outer attribute model for writer $w$    Valid shared units of the same size from the other 41 writers. Neither shared nor self-chosen units of $w$.
  Inner attribute preprocessing and SVM   Training writers in each of five stratified group folds inside the outer 41-writer set; nuisance regression, if used, is refitted here.
  Platt calibrator                        Out-of-fold SVM scores and labels of the outer 41 training writers.
  ID temperature for queries of $w$       Identity-score negative log loss on query writers other than $w$, equally weighted by writer; no attribute probabilities.
  Fixed fusion                            Unit weight on log attribute probability; no fitted handedness weight.
  Nested weight                           Development folds inside the outer 41-writer set; probabilities regenerated with both the outer writer and the development fold excluded from attribute fitting and calibration.

  : **Table 3.** Information available at each fitting stage. The query writer remains enrolled for identity matching but contributes no samples to its own handedness model.

## 3.4 Handedness prediction and ranking rules

For each query writer and unit size, a linear SVM ($C=0.1$) is trained on valid shared units of the other 41 writers, with standardization fitted on the training data. Five-fold stratified group cross-validation over those writers supplies out-of-fold scores for logistic calibration [@platt1999probabilistic], giving the left-hand probability $p_q=\sigma(\ell_q)$. Writer identity only assigns evaluation folds; a deployed system would instead freeze one attribute model, calibrator, temperature and weight on development writers.

The *known* scenario supplies true query and gallery hands and restricts the top-ranked group to the query's hand; it measures the opportunity offered by a correct attribute. In the primary *predicted-query* scenario, gallery hands are known and $p_q$ is predicted. Hard filtering ranks all writers of the predicted hand first, preserving identity order within each group; in a balanced gallery, an incorrect prediction then makes top-1 success impossible. Probabilistic score fusion ranks writers by

$$f_w=s_w/T+\log p_q(h_w),\qquad\text{(1)}$$

where $p_q(1)=p_q$ and $p_q(0)=1-p_q$. For each setting, $T$ is selected from 31 logarithmically spaced values between $10^{-3}$ and 1 by minimizing identification negative log loss on the other query writers, using identity scores only. Equation (1) is a score-combination rule; it has a Bayesian reading only if the two evidence sources are conditionally independent, which need not hold for two representations of the same handwriting.

## 3.5 Nested fusion weight

To allow redundant evidence to be downweighted, we also rank by $s_w/T+\lambda\log p_q(h_w)$ with a nonnegative weight selected for each held-out query writer. The other 41 writers supply five development folds; each development attribute model and its inner calibrator exclude both the outer writer and the entire development fold, avoiding selection bias [@cawley2010selection]. Equal-writer development negative log loss selects $\lambda\in\{0,.1,.2,.3,.5,.7,1,1.5,2,3,5\}$ at the same identity-only temperature. The grid includes no attribute evidence ($\lambda=0$) and fixed fusion ($\lambda=1$); ties select the smaller weight. Supporting Section S8 gives the complete nesting hierarchy.

## 3.6 Filtering identity, random partitions and error clustering

Let $C$ denote a correct hand prediction and $K$ top-1 success with the true query hand supplied. With both hand groups nonempty, filtered success is $C\cap K$, so

$$A_{\mathrm{filter}}=aP(K\mid C)=aA_{\mathrm{known}}+\operatorname{Cov}(\mathbf{1}_C,\mathbf{1}_K),\qquad\text{(2)}$$

where $a=P(C)$. Only if the two correctness events are independent does this reduce to

$$\begin{aligned}
 A_{\mathrm{filter}}&\approx aA_{\mathrm{known}},\\
 a_{\mathrm{break}}&\approx A_{\mathrm{none}}/A_{\mathrm{known}}.
 \end{aligned}\qquad\text{(3)}$$

We therefore report conditional accuracies and approximation residuals. The two marginal accuracies alone permit the bounds

$$\begin{aligned}
 \max(0,a+A_{\mathrm{known}}-1)&\leq A_{\mathrm{filter}},\\
 A_{\mathrm{filter}}&\leq\min(a,A_{\mathrm{known}}).
 \end{aligned}\qquad\text{(4)}$$

A known balanced random partition tests whether the oracle gain is specific to handedness. With 21 writers per group, a query whose true writer is outranked by $j$ impostors survives an oracle filter only if all $j$ fall in the opposite group, so a uniform balanced partition succeeds with exact probability

$$P(\mathrm{success}\mid j)=\begin{cases}\binom{21}{j}/\binom{41}{j},&0\leq j\leq21,\\0,&j>21.\end{cases}\qquad\text{(5)}$$

Writing $\pi_j$ for the writer-weighted share of queries with $j$ outranking impostors and $\rho_j$ for the observed rate at which the known hand rescues them, the gap between known handedness and the random partition decomposes exactly as

$$\Delta_{\mathrm{known}-\mathrm{random}}=\sum_{j\geq1}\pi_j\Big[\rho_j-\binom{21}{j}\big/\binom{41}{j}\Big].\qquad\text{(6)}$$

For $j=1$, $\rho_1$ is simply the share of sole outranking impostors that have the opposite hand, against $21/41=51.2\%$ for a random impostor. Values below 51.2% mean that identity errors cluster within hand, so a correct hand label removes fewer real competitors than an arbitrary split. Separately, 20 nearly hand-balanced random 21/21 writer labelings test whether an attribute model can learn arbitrary writer-associated labels when the test writer's other paragraphs are, or are not, in the training data.

## 3.7 Endpoints, inference and precision

Top-1 accuracy is averaged over valid query units and reference configurations within each writer, then equally over writers; across-setting means weight the 120 settings equally. Paired differences use 9,999 writer-level sign flips and 10,000 hand-stratified writer bootstraps, with common resampled writers across settings for aggregate intervals. Intervals condition on the observed gallery and fitted predictions and do not include model-refitting or population-sampling uncertainty.

Four settings were designated after an exploratory feasibility analysis of paragraph-level conditions: ResNet-18 and ViT-B/16 on self-chosen paragraphs with one reference and on quarter strips with six references. Holm correction covers the four fixed-fusion comparisons and, separately, the eight learned-weight comparisons against no fusion and fixed fusion. None of these analyses was preregistered.

Because nonsignificance alone cannot support absence of a useful effect, we add a post hoc precision analysis. For each primary contrast we report the 90% bootstrap interval, which corresponds to two one-sided tests at $\alpha=.05$ [@lakens2017equivalence]; the smallest margin $\Delta^*$ for which equivalence holds; the minimum detectable effect at 80% power, $2.80\times$ the bootstrap standard error; and the largest gain left open by the interval as a share of the same setting's oracle gain. No margin was fixed before the data were seen, so margins are reported as a continuum rather than as a pass/fail test.

## 3.8 Cross-session evaluation on CSAFE

The CSAFE Handwriting Database provides an independent English-language cohort [@crawford2020database; @crawford2026csafe]. Its version-2 release contains 475 writers: 48 left-handed, 423 right-handed, three ambidextrous and one with unreported hand. We include all 48 left-handed writers and 48 right-handed writers sampled with seed 20261108, retaining a balanced design. Each writer contributes six session-1 pages (London Letter, LND, and Wizard of Oz, WOZ, each written three times) for attribute training; one LND page is the sole gallery reference. Primary queries are separate session-2 LND (same prompt) and WOZ (different prompt) pages; a session-3 WOZ page is secondary. The original extraction rules, SVM, checkpoints and fusion grids are retained, with ResNet-18, ViT-B/16, HC-183 and STROKE as identity features and HC-183 as the primary handedness predictor. All 864 pages pass the validity rules, and each query writer's six training pages are excluded from its attribute model and calibrator.

Eight primary contrasts compare fixed and nested fusion with identity-only ranking for ResNet-18 and ViT-B/16 in both session-2 conditions, with Holm correction across all eight. Session 3, alternative attribute models and 20 refitted balanced 21/21 galleries are secondary. The design was specified before CSAFE outcomes were inspected. Models are refitted within CSAFE rather than transferred from BanglaHand. A post hoc audit found that the nuisance regression of the adjusted sensitivity model was numerically unstable on CSAFE scans, whose upper grayscale quantiles are constant at white. A numerically stable implementation of the same projection replaces it (Supporting Section S16); the primary model is unaffected.

# 4 Results

## 4.1 Oracle opportunity and achieved utility

The primary handedness predictor attains 68.7% accuracy on shared paragraphs and 65.5% on self-chosen paragraphs (Brier score 0.204, log loss 0.602). Averaging log-odds over all eight paragraphs of a writer gives 73.8%, but uses more writing than a single query.

Known handedness improves top-1 accuracy by 6.1 points on average (95% conditional interval $+5.1$ to $+7.1$). With predicted query handedness, fixed fusion changes accuracy by $-0.6$ points ($-2.3$ to $+0.9$), nested fusion by $-0.08$ points ($-1.13$ to $+0.81$), and hard filtering by $-14.3$ points ($-19.3$ to $-9.5$). Table 4 shows two representative conditions and Fig. 2 the effects of query size and reference count. Nested weighting improves on fixed fusion by $+0.49$ points ($-0.15$ to $+1.28$); its median selected weight is 0.5, and 1.3% of fits choose zero.

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

  : **Table 4.** Top-1 identification accuracy (%), equally weighted by writer. Known: true query and gallery hands. Fusion and Filter: query hand predicted by writer-held-out HC-183; gallery hands known. For $k=1$, all six reference choices are averaged within writer.

![**Figure 2.** Identification accuracy for self-chosen texts. (a) Query size with six whole-paragraph references per writer. (b) Reference count for paragraph queries. Solid: no handedness; dashed: known query and gallery hands; dotted: fusion with predicted query handedness. (c) Writer-level paired differences for the four designated comparisons, with 95% conditional bootstrap intervals. Representations omitted from (a,b) appear in the complete grid (Supporting Table S15).](assets/performance.png){width="6.65in"}

## 4.2 Precision of the fusion results

Figure 3 places each fusion effect beside the oracle gain it could at best recover. Across all 120 settings, the 90% interval for nested fusion is $-0.96$ to $+0.67$ points: nested fusion is equivalent to identity-only ranking within $\pm0.96$ points and recovers at most 11% of the 6.1-point oracle gain. Fixed fusion is equivalent within $\pm2.0$ points. In the four designated settings, the smallest supported margins are 1.8--2.6 points for nested and 2.4--3.6 points for fixed fusion; all four Holm-adjusted fixed-fusion and all eight learned-weight comparisons have $p=1.000$ (Supporting Tables S21 and S22). The minimum detectable effect at 80% power is 1.4 points for nested fusion across settings. The design therefore excludes a practically useful average gain, but individual settings retain uncertainty of about $\pm2$ points.

![**Figure 3.** Fusion effects against the oracle opportunity. Points are paired top-1 changes from identity-only ranking; thick bars are 90% and thin bars 95% conditional writer-bootstrap intervals. Diamonds give the known-hand gain in the same setting. Shading marks $\pm1$ and $\pm2$ points. Values at right are the smallest equivalence margin $\Delta^*$ supported for nested fusion (two one-sided tests, $\alpha=.05$; margins are post hoc). (a) BanglaHand, all settings and the four designated settings. (b) CSAFE primary contrasts; for ResNet-18 with a different prompt, nested fusion changes no decision.](assets/precision.png){width="6.65in"}

## 4.3 Representation-specific utility

Figure 4 averages the 20 settings of each representation. Nested-fusion changes range from $-0.64$ to $+0.42$ points, and its improvement over fixed fusion from $+0.18$ to $+1.08$ points. The oracle opportunity differs widely, from 2.6 points for STROKE to 7.6 points for HC-183, but no representation converts it into a fusion gain. These marginal exploratory intervals do not justify selecting a representation after inspection.

![**Figure 4.** Representation-specific utility, averaging 20 settings and all 42 writers equally. (a) Top-1 changes from identification without handedness. (b) Nested minus fixed fusion on a separate scale. Bars are marginal conditional 95% writer-bootstrap intervals with shared resampling across contrasts.](assets/representation_utility.png){width="6.65in"}

## 4.4 Why the oracle gain is not handedness-specific

Across 456 combinations of setting and prediction model, the independence product tracks filtering accuracy closely ($r=0.996$; mean absolute error 1.5 points), but residuals matter for individual decisions (Fig. 5a). For HC-183 on self-chosen paragraphs with one reference, the product predicts 31.4% against 28.0% observed, because known-hand identification succeeds on 42.7% of queries with correct hand predictions but 58.0% of those with incorrect predictions. The approximate break-even attribute accuracy for filtering has median 86% (range 55--98%) for self-chosen queries, far above the 65.5% achieved.

A known balanced random partition improves accuracy by 6.9 points, so known handedness falls 0.8 points short of it ($-1.8$ to $+0.2$; Fig. 5b). Equation (6) attributes $-0.65$ of the $-0.78$-point gap to queries with a single outranking impostor and $-0.23$ to queries with two. When one impostor outranks the true writer, it has the opposite hand in only 34.2% of cases for ResNet-18 (95% interval 26.0--43.3%), 36.6% for ViT-B/16 (27.7--46.4%) and 41.3% for HC-183 (32.5--50.0%), against 51.2% for a random impostor (Fig. 5c; Supporting Table S23). The binary-stroke Stroke-CNN is the exception (62.8%). Identity errors of the stronger matchers therefore concentrate within hand. A correct hand label removes fewer real competitors than an arbitrary split, and handedness-related information is already partly present in the identity scores.

![**Figure 5.** Why the oracle gain is not specific to handedness. (a) Observed filtering accuracy against the independence product $aA_{\mathrm{known}}$ for 456 setting/prediction-model combinations; colours follow the representations in (c). (b) Mean oracle gain from known handedness and from the exact expectation of a known random 21/21 partition (Eq. 5). (c) For queries with exactly one impostor ranked above the true writer, the share of those impostors with the opposite hand, with 95% writer-bootstrap intervals; the dashed line is the random-impostor rate. The CSAFE point pools ResNet-18 and ViT-B/16 in both session-2 conditions (reference 50.5%). Points are descriptive summaries of reused writers, not independent replications.](assets/mechanism.png){width="6.65in"}

## 4.5 Attribute calibration

Table 5 reports writer-held-out accuracy and proper scoring rules for the three predictors on both datasets. Accuracy, Brier score and log loss measure different aspects of the predictions; none alone determines how predictions interact with identity margins. Reliability plots (Supporting Figure S1) do not isolate miscalibration as the cause of the absent fusion gain.

  ----------------- --------------------- ------- ------- --------------------- ---------------------
                               BanglaHand                     CSAFE accuracy, % 
  Model                       Accuracy, %   Brier     NLL           Same prompt      Different prompt
  HC-183              65.5 \[53.6, 77.4\]   0.204   0.602   67.7 \[58.3, 77.1\]   58.3 \[49.0, 67.7\]
  HC-183 adjusted     61.9 \[50.0, 73.8\]   0.244   0.684   68.8 \[59.4, 78.1\]   61.5 \[51.0, 70.8\]
  STROKE              48.8 \[35.7, 61.9\]   0.286   0.775   67.7 \[58.3, 77.1\]   60.4 \[51.0, 69.8\]
  ----------------- --------------------- ------- ------- --------------------- ---------------------

  : **Table 5.** Writer-held-out handedness prediction. BanglaHand: self-chosen paragraph queries. CSAFE: session-2 page queries, models trained on session-1 pages of the other 95 writers. Intervals are conditional 95% writer bootstraps; Brier score and negative log loss (NLL) are losses. Adjusted HC-183 on CSAFE uses the numerically corrected implementation.

## 4.6 Writer-associated information

When the test writer's shared paragraphs are included in training, handedness accuracy on that writer's self-chosen texts rises from 65.5% to 82.1%. Under the same enrolment training, 20 random writer labelings are predicted with 52.4--76.2% accuracy (Fig. 6). Writer-associated information can therefore support arbitrary labels, and enrolled-writer accuracy overstates attribute generalization. The control does not show leakage in the primary writer-held-out pipeline, and it does not assign a fraction of genuine handedness accuracy to identity. An exploratory both-sided descriptor-agreement score improves top-1 by $+1.08$ points ($+0.50$ to $+1.72$) over the 96 self-chosen-text settings, but it uses predicted gallery descriptors from models that can include the query writer's enrolment pages; it is an enrolled-cohort diagnostic, not unseen-writer utility (Supporting Section S6).

![**Figure 6.** Writer-overlap diagnostic. (a) Split designs: each writer keeps one fixed random label; training either excludes that writer or contains their shared paragraphs before their own texts are tested. (b) Own-text accuracy for genuine handedness (diamonds) and for 20 random writer labelings (gray points). The dashed 50% line is descriptive; held-out training creates a structured null rather than an exact 50% reference.](assets/label_controls.png){width="6.65in"}

## 4.7 Within-cohort robustness

Across four calibration-fold partitions, self-chosen handedness accuracy ranges from 64.3% to 66.7% and mean fixed-fusion changes from $-0.57$ to $-0.50$ points. Temperature selection reaches a grid endpoint in 3.4% of fits, and halving or doubling the selected temperatures gives mean fusion changes from $-1.81$ to $+0.04$ points. Omitting each evaluation writer in turn changes mean nested-fusion effects by $-0.24$ to $+0.22$ points, while hard filtering remains substantially harmful. Supporting Section S9 (Table S12 and Figure S4) reports loss profiles, weight distributions and every writer's influence.

## 4.8 Decision margins: rescued and damaged identifications

Fusion preserves candidate ranking within each hand group. Define the true-writer margin against the best opposite-hand candidate and the signed attribute log-odds as

$$\begin{aligned}m_q&=\frac{s_{w^*}-\max_{h_w\ne h_{w^*}}s_w}{T},\\r_q&=\log\frac{p_q(h_{w^*})}{p_q(1-h_{w^*})}.\end{aligned}\qquad\text{(7)}$$

Away from ties, identification succeeds exactly when the true writer leads its own hand group and $m_q+\lambda r_q>0$, so fusion cannot repair a wrong within-group ordering. A decision is *rescued* if it is wrong before fusion and correct after, and *damaged* in the opposite case; their difference is the net top-1 change. Across the 120 settings, fixed fusion rescues 2.02% of decisions and damages 2.59%, and nested weighting rescues 1.33% and damages 1.42% (Fig. 7). Predicted handedness thus changes few decisions and its benefits and costs nearly cancel. Direct candidate ranking and this analytical reconstruction agree on all 664,680 query/reference decisions.

![**Figure 7.** Decision changes and margin diagnostic. (a) Rescued and damaged decisions under fixed and nested fusion, averaging all 120 settings with equal writer weights; bars are conditional 95% writer-bootstrap intervals. (b) HC-183, self-chosen full paragraphs and one reference, fixed fusion, for decisions in which the true writer leads its hand group. The vertical line is the baseline boundary $m=0$ and the dashed line the fusion boundary $m+r=0$; points above it succeed after fusion. True-hand axes are retrospective diagnostics, not prediction inputs.](assets/margin_diagnostic.png){width="6.65in"}

## 4.9 Cross-session evaluation on CSAFE

The primary handedness predictor reaches 67.7% accuracy on session-2 LND queries, 58.3% on session-2 WOZ and 65.6% on session-3 WOZ (Table 5). With matching prompts across sessions, known handedness raises ResNet-18 identification from 61.46% to 68.75%, whereas fixed and nested fusion both reach 62.50%; ViT-B/16 falls from 64.58% to 62.50% under either rule (Table 6, Fig. 8). With a different prompt, fixed fusion lowers ResNet-18 from 54.17% to 48.96%, while nested fusion leaves it unchanged. None of the eight primary contrasts shows a significant gain after Holm correction (all adjusted $p\geq0.4888$). Across the eight contrasts, fusion rescues 10 decisions and damages 19. Because one writer is worth 1.04 points, the smallest equivalence margins are 3.1--5.2 points for nested fusion (Fig. 3b). CSAFE therefore agrees in direction with BanglaHand but cannot by itself exclude effects of 2--3 points. Hard filtering reduces neural identification by 17.7--25.0 points.

The STROKE and corrected adjusted HC-183 sensitivity predictors reach 60.4--68.8% accuracy and likewise produce no fusion gain (Supporting Tables S17 and S19). Splitting the balanced gallery by query hand shows no consistent gain for left- or right-handed queries (Supporting Table S20). For the neural matchers, the sole outranking impostor has the opposite hand in 40.0% of cases (27.0--54.1%, 40 decisions) against 50.5% for a random impostor, the same direction as in BanglaHand (Fig. 5c).

  Prompt      Identity features     Identity only   Known hand   Hard filter   Fixed   Nested
  ----------- ------------------- --------------- ------------ ------------- ------- --------
  Same        ResNet-18                     61.46        68.75         43.75   62.50    62.50
  Same        ViT-B/16                      64.58        67.71         39.58   62.50    62.50
  Different   ResNet-18                     54.17        60.42         31.25   48.96    54.17
  Different   ViT-B/16                      55.21        61.46         34.38   54.17    54.17

  : **Table 6.** CSAFE session-2 top-1 identification accuracy (%) with the primary HC-183 handedness model and one session-1 reference per writer. The gallery contains 48 writers of each hand. "Same" and "Different" refer to the query prompt relative to the LND reference. All eight fixed/nested primary contrasts have Holm-adjusted $p\geq0.4888$.

![**Figure 8.** Cross-session CSAFE evaluation with the primary HC-183 handedness model in the 96-writer gallery. (a) Top-1 identification under each ranking rule for the four primary cases (same or different prompt relative to the session-1 gallery page); the gray bar spans hard filtering to the known-hand oracle. (b) Writer decisions rescued (solid, right) and damaged (hatched, left) by fixed (blue) and nested (pink) fusion. Intervals for these contrasts are shown in Fig. 3b.](assets/csafe_identification_utility.png){width="6.65in"}

# 5 Discussion

The central lesson is that neither attribute accuracy nor an oracle gain establishes identification utility. In BanglaHand, a correct hand label raises top-1 accuracy by 6.1 points, yet a correct arbitrary partition raises it by more. The oracle measures what any binary partition offers a closed gallery, not what handedness offers. Once query handedness must be predicted, calibrated fixed fusion and properly nested learned weighting both remain within about one point of identity-only ranking on average, and hard filtering turns every wrong hand prediction into an unrecoverable top-1 error. The cross-session CSAFE cohort, with different writers, script, pages and acquisition, shows the same pattern with lower precision.

Two mechanisms explain the absent gain. First, the matchers already carry handedness-related information: their errors cluster within hand, so the impostors that the attribute could remove are disproportionately the ones it cannot. This redundancy is also why nested weighting selects a median weight of 0.5 rather than 1. Second, a 65% predictor offers little evidence per query (log loss 0.602 against 0.693 for chance), so it moves a decision only when the identity margin is already small. Rescued and damaged decisions are then nearly equal in number. Filtering would need roughly 86% attribute accuracy to break even for these matchers. Predictable handedness [@prappo2026handedness] is compatible with both observations: an attribute can be learnable from handwriting and still add nothing to a matcher that has learned the same style.

These results have a direct practical reading. An auxiliary-attribute system should be judged on the sequence in Fig. 1b: establish the oracle opportunity, test whether it exceeds a random partition, compare predicted-attribute fusion with identity-only ranking on the same queries with parameters fitted without the evaluated writer, report the precision of that comparison, examine rescued and damaged decisions, and separate unseen-writer generalization from enrolment-conditioned accuracy with random-label controls. Each step answers a question that attribute accuracy cannot. The protocol applies unchanged to gender, age or any other binary or categorical attribute, with Eq. (5) generalized to the attribute's group sizes.

Prior attribute studies establish whether handwriting supports demographic prediction, and identity-focused studies develop the matcher itself (Table 1). The present evaluation is complementary: it holds the identity comparison fixed and intervenes only on the auxiliary information. It does not show that our representations are superior to handwriting-specialized recognition systems, and stronger matchers with larger margins could change the size of both the opportunity and the redundancy.

# 6 Scope and limitations

BanglaHand is a deliberately balanced 42-writer cohort from one region, device and photographic session; the companion study shares this cohort and is not independent validation. Balanced galleries give the oracle and random-partition benchmarks a symmetric reference and keep the attribute predictor from being dominated by the majority class. Under natural left-hand prevalence, the oracle benefit would concentrate in the minority hand and depend on the chosen prevalence; Supporting Section S7 examines reduced-prevalence galleries. CSAFE adds an independent population, English text and separate-session scans, but its 96-writer gallery resolves effects only beyond about $\pm3$ points, and its one-reference whole-page conditions do not reproduce every BanglaHand setting.

Intervals condition on the enrolled gallery and fitted predictions and do not propagate model-refitting or population-sampling variability. The designated settings, the equivalence analysis and the CSAFE sensitivity-model correction are post hoc and are reported as such; the CSAFE primary analysis was specified before its outcomes were seen. The comparison uses handcrafted descriptors and frozen ImageNet networks, and handedness is the only real auxiliary attribute tested; handwriting-specialized training, other fusion objectives and other attributes remain to be evaluated. BDSHWA records handedness for Bengali--English online handwriting but includes one left-handed participant [@bdshwa2026], so it cannot supply a balanced offline replication.

The predicted-query protocol assumes reliable gallery handedness. Knowing a suspect's handedness does not establish the hand of a questioned writer, and closed-set top-1 ranking does not evaluate open-set rejection, likelihood ratios, disguised writing or non-dominant-hand writing. Forensic candidate prioritization would require validation under those conditions with explicit error costs. Predicted handedness alone cannot justify excluding a candidate.

# 7 Conclusions

Correct handedness improves Bangla writer identification, but no more than an arbitrary correct partition of the gallery, because identity errors already cluster within hand. Predicted handedness at realistic accuracy recovers at most a small fraction of that opportunity: nested fusion is within one point of identity-only ranking across 120 settings, and hard filtering is harmful. A cross-session English cohort agrees in direction. Attribute accuracy and oracle gains are therefore poor proxies for identification utility. Auxiliary attributes in biometric candidate ranking should be evaluated with paired opportunity, specificity, utility, precision, mechanism and writer-information controls, and with the query and gallery information available at deployment stated explicitly.

# Data and code availability {#data-and-code-availability .unnumbered}

The BanglaHand code, frozen features, setting manifests, numerical results and self-contained supplementary methods are archived as [reproducibility ZIP v1.6.0](https://github.com/LegendaryBeast/Bangla-writer-identification-research/releases/download/v1.6.0/wid-reproducibility-v1.6.0.zip) at <https://github.com/LegendaryBeast/Bangla-writer-identification-research/releases/tag/v1.6.0>. The release supplies exact checkpoint references and table/figure reproduction commands. Code is MIT licensed; results and frozen features are CC BY 4.0. Photographs are excluded; obtain BanglaHand separately from Mendeley Data (doi:10.17632/n4sjstdzfy.1) [@banglahand2026data].

Two further archives are supplied with the submission: the CSAFE extension (`csafe-replication-2026-11-08.zip`) and the revision analyses (`wid-revision-analyses-2026-11-09.zip`), which contain the precision, error-clustering and corrected CSAFE sensitivity-model scripts and outputs. CSAFE version 2 is obtained separately at doi:10.25380/iastate.10062203.v2 under CC BY 4.0 [@crawford2026csafe]; original images and checkpoint binaries are excluded from both archives.

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

Supporting information contains feature and preprocessing specifications, the fitting hierarchy, filtering and random-partition derivations, calibration and random-label controls, nested fusion selection, robustness and decision-margin analyses, the complete 120-setting BanglaHand grid, the CSAFE protocol and full results, the post hoc precision and error-clustering analyses, the numerical correction of the CSAFE sensitivity model, and reproduction instructions.
