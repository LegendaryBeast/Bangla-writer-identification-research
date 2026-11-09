# Simulated peer review — IET Biometrics submission

**Manuscript:** *Handedness for Bangla writer identification: Oracle gains and predicted-attribute utility* (Prappo & Rahman), `submission/iet_biometrics/Manuscript.pdf`, 13 pp., build 2026-11-09
**Supporting information read:** S1, S5, S7, S13 (CSAFE protocol, amendment, Tables S5, S13, S17, S18)
**Mode:** full multi-perspective review (5 reviewers + editorial synthesis)
**Note:** This is a simulated review. Scores are evidence-based judgements, not a prediction of what real referees will say.

Line numbers below refer to the manuscript's margin numbering.

---

## Phase 0 — Field analysis and reviewer configuration

| Item | Assessment |
|---|---|
| Primary field | Biometrics: offline writer identification |
| Secondary fields | Soft biometrics, score-level fusion, evaluation methodology / ML leakage |
| Paradigm | Quantitative, retrospective computational experiment with paired within-writer contrasts |
| Main result type | A **null / inconclusive utility result** plus an evaluation framework |
| Target venue | IET Biometrics (Wiley OA), research article |
| Maturity | High: submission-ready build, reproducibility archive, extensive supplement |

| Reviewer | Configured identity | Focus |
|---|---|---|
| EIC | Associate Editor, IET Biometrics; soft-biometric fusion and evaluation standards | Fit, novelty, framing, presentation |
| R1 Methodology | Biometric evaluation statistician (paired designs, calibration, ISO/IEC 19795-style reporting) | Inference, power, calibration, selection |
| R2 Domain | Offline writer identification researcher (Indic/Bangla scripts, writer retrieval) | Literature, baselines, domain interpretation |
| R3 Perspective | Forensic statistics / questioned-document examination | Operational realism, endpoints, ethics |
| DA | Devil's Advocate | Core claim, logic chain, "so what?" |

---

## Phase 1 — Independent reports

### EIC report

**Summary.** The paper asks whether *predicted* handedness improves closed-set writer identification. It separates oracle opportunity (known hands, +6.1 pts) from achieved utility (fixed fusion −0.6, nested −0.08, hard filter −14.3). It adds a known random-partition benchmark (+6.9 pts) and filtering, margin and random-label diagnostics, and runs an independent cross-session check on 96 CSAFE writers.

**Strengths**
1. The question fits the journal well. Separating "the attribute can be predicted" from "the predicted attribute improves identification" is under-examined in soft biometrics.
2. The study is unusually honest about what it can and cannot claim: no equivalence claim (l.325–326), conditional intervals (l.314–316), and a clear deployment caveat (l.237–244).
3. Leakage control is explicit and auditable (Table 2), and the reproducibility release is strong.
4. The CSAFE extension adds a second population, a second script and separate capture sessions.

**Weaknesses**
1. **The framing is out of balance.** The title leads with "Bangla", but half of the evidence is English (CSAFE). The subtitle "Oracle gains and predicted-attribute utility" does not tell a reader what was found. *Fix:* use a title that states the finding and the scope, for example "Does predicted handedness improve writer identification? Oracle gains versus achieved utility on Bangla and English handwriting".
2. **The contribution claim is larger than the parts that make it up.** The paper says it contributes "a controlled framework" (l.21, l.65). Yet it calls Eq. (2) an "elementary filtering identity" (l.29–31), and nested selection (Cawley & Talbot) is standard practice. The new parts are the random-partition benchmark (Eq. 5) and the way the controls are combined. *Fix:* state exactly what is new in one sentence, and present the framework as a numbered evaluation protocol (a box or checklist) that readers can reuse.
3. **Readability.** About a third of the sentences in Sections 4–6 are scope disclaimers (for example l.405, l.476–477, l.505–507, l.534–536). Taken one by one each is correct. Taken together they bury the findings and read as defensive. *Fix:* keep one consolidated "Inferential scope" paragraph (3.7 already exists) and remove the repeated per-result disclaimers.
4. **Too many display items.** There are 7 figures and 8 tables in 13 pages. Figure 5, Table 6 and Table 8 are diagnostic or positioning material and belong in the supplement.
5. **Table 8 grades prior work against columns defined by this study's own design**, so every other study shows NR or U. Referees often read this as self-serving. *Fix:* drop it or merge it into Table 1 as a short "endpoint evaluated" column.
6. **Dependence on the authors' own unreviewed work.** The cohort, the representations and the motivation come from the authors' own dataset [13] and a companion SSRN preprint that has not been peer reviewed [12].

**Recommendation:** Major revision.

---

### Reviewer 1 — Methodology

**Strengths:** paired writer-level design; sign-flip tests with stratified bootstrap; separate nested weight selection; temperature fitted on identity scores only; reproducible decision reconstruction (664,680 decisions, l.502–505). I checked internal consistency and found no errors: 6.1 − 6.9 = −0.8; rescue − damage equals the net change (2.02 − 2.59 = −0.57; 1.33 − 1.42 ≈ −0.08); 42 × 8 × 15 = 5040 units; 96 × 9 = 864 CSAFE pages; 48 + 423 + 3 + 1 = 475.

**Major issues**

1. **No smallest effect of interest and no power or precision analysis.** The headline contrasts have 95% intervals of about ±3 points (Table 4). On CSAFE, one writer equals 1.04 points: the ResNet-18 same-prompt "gain" of 61.46 → 62.50 is exactly one writer (Table 7). The design cannot tell "no utility" apart from a 1–2 point utility, and 1–2 points may be practically relevant. *Fix:* declare a margin (for example ±2 pts top-1) with a justification and run an equivalence test (TOST) or report the minimum detectable effect at 80% power for each primary contrast. If the margin cannot be met, say directly that the study is inconclusive at that resolution.

2. **The CSAFE adjusted-HC-183 sensitivity model is broken, not merely weak.** Table S17 shows raw AUC 0.727 but calibrated AUC 0.501 and accuracy 34.38% [25.00, 43.75], which is below chance with an interval that excludes 50%. 24/96 Platt fits have negative slopes (l.530–531). Calibration is inverting a scorer that works. The manuscript calls this a "sensitivity failure" (l.532–533). It is a pipeline artefact, so the adjusted model's fusion rows in Table S18 cannot be interpreted. *Fix:* constrain the Platt slope to ≥ 0 (or fall back to the uncalibrated sign), or use isotonic or temperature-only calibration, and rerun. Then explain why the inner out-of-fold scores flip sign for these writers (possibly nuisance regression fitted on very small inner folds).

3. **The designated settings were chosen after exploration** (l.317–320, "after an exploratory feasibility analysis"). The disclosure is honest, but Holm correction over four contrasts chosen after looking at the data does not control the family-wise error that matters. *Fix:* either treat all 120 settings × 2 rules as the primary family under BH-FDR, or report what the feasibility analysis looked at and confirm it did not use any fusion outcomes.

4. **The single headline number depends on how settings are weighted.** The −0.6 average weights 120 heterogeneous settings equally, including 1/8-strip and Stroke-CNN quarter-strip conditions where baseline accuracy is 10.7% (Table 3). *Fix:* report per-representation and per-query-size estimates as primary results, and say why equal weighting across settings is the estimand of interest.

5. **There is no upper bound on what soft fusion could achieve.** λ is selected by development log loss, not top-1 (l.575–578). Readers cannot tell whether the null reflects the selection objective or the information content. *Fix:* report an *oracle-λ* ceiling, i.e. the λ that maximises top-1 on the evaluation writers themselves, labelled as optimistic. If even the oracle ceiling is below about 1 pt, the null is much stronger. If it is around 3 pts, the bottleneck is the selection objective.

6. **Unexplained Stroke-CNN collapse.** Accuracy falls from 58.7% (paragraph) to 10.7% (quarter strip, k = 6), which is far steeper than any other representation (Table 3). This could be a tiling or rescaling artefact (binary tiles rescaled to a 3-px stroke on short strips). *Fix:* give the tile counts per unit size and confirm this is not a bug.

**Minor**
- Table 4's caption says "All comparisons are exploratory", but §3.7 makes the four fixed-fusion contrasts designated with Holm correction, and their Holm p-values are not shown. Make the two consistent and add the fixed-fusion Holm column.
- Train/test text mismatch: the predictor is trained on shared prompts and tested on self-chosen text (65.5% vs 68.7% on shared). State this as a design choice in §3.4.
- Bring the minimum-margin and tie-handling rule into the main text in one sentence; it currently appears only through l.294.

**Rigor score: 74/100.** Leakage handling is exemplary, but the inferential resolution is too coarse for the strength of the conclusions.

---

### Reviewer 2 — Domain (writer identification)

**Strengths:** text-independent setup with shared-prompt galleries and self-chosen queries; same-prompt vs cross-prompt split on CSAFE; good use of classical contour/hinge features (STROKE, 92.3% top-1).

**Major issues**

1. **The literature base is thin for this venue.** There are 26 references. Missing are (a) modern writer identification and retrieval, e.g. Christlein and colleagues' unsupervised feature learning / VLAD-based writer retrieval pipelines, and recent deep writer-ID work beyond FragNet; (b) Bangla/Indic writer identification specifically, e.g. Adak & Chaudhuri's Bangla writer identification work and the Halder et al. survey of offline writer identification; (c) soft-biometric fusion outside handwriting, e.g. Park & Jain (face matching and retrieval with soft biometrics), which reports the same "small attribute, small gain" pattern. *Please verify each suggested citation yourself before adding it.*

2. **The paper's most interesting domain finding is in the supplement.** Table S5 shows that baseline identity errors fall on the opposite hand *less* often than chance (HC-183 42.3% [36.5, 48.8], ResNet-18 43.8% [38.9, 48.4], vs 51.2% expected). In other words, identity matchers already confuse writers *within* the same hand. This is the mechanism behind "known handedness ≈ random partition" (−0.8 pts, l.419–423): a hand filter removes fewer real competitors than a random split does. The main text says only that there is no "distinctive advantage". *Fix:* move Table S5 into §4.3 and state the mechanism. It turns a null into a positive, publishable finding: **identity features already encode handedness-correlated style, so the attribute adds redundant evidence.**

3. **Matcher strength limits generality.** All matchers are frozen ImageNet features or handcrafted descriptors. The limitation is acknowledged (l.626–631), but the answer to the utility question depends directly on matcher margins. *Fix:* add one handwriting-specific, training-light matcher (e.g. SIFT/RootSIFT + VLAD in the style of the writer-retrieval literature) on CSAFE at least.

4. **Why HC-183 is the primary predictor is not explained.** The companion reports 77.4% balanced accuracy from multi-paragraph SVM margins, and here 8-paragraph aggregation reaches 73.8% (l.369). *Fix:* explain the choice, and test utility at the *best achievable* accuracy (e.g. a multi-paragraph query condition). A null at 74–77% is far more informative than a null at 65%.

**Minor**
- l.129–141 (strip-cut search half-widths, smoothing σ = 96/5, etc.) is implementation detail; move it to the supplement.
- "Stroke-CNN" suggests a trained network; rename it (e.g. "Binary-ResNet-18"), since l.202–203 has to disclaim the name.
- Paul et al. [18] is verification; Table 1 handles this well.

**Domain contribution score: 62/100.**

---

### Reviewer 3 — Perspective (forensic / operational)

**Strengths:** the forensic caveats in §6 are correct (suspect handedness ≠ questioned-writer handedness; no LR; no disguised writing). The conclusion that predicted handedness alone cannot justify exclusion (l.644–645) is the right practical message.

**Major issues**

1. **Balanced galleries are unrealistic and change the answer.** BanglaHand is 21/21 and the CSAFE sample is 48/48, but natural left-hand prevalence is about 10% (CSAFE itself: 48/471). With a 10% prior, a correct "left" call removes about 90% of the gallery, while a correct "right" call removes about 10%. Utility is strongly asymmetric, and the balanced design averages that away. S7 explores prevalence only synthetically. *Fix:* rerun the CSAFE primary contrasts with the **natural-prevalence gallery (all 471 single-hand writers)**, reporting left- and right-hand queries separately. This uses data you already have, and it is the analysis a forensic reader will ask for first.

2. **Top-1 is the wrong single endpoint for candidate prioritisation.** Forensic and investigative use is shortlisting. A hard filter that costs 14 pts at top-1 may cost much less at rank-10 or rank-20, and soft fusion may help at rank-k without helping top-1. *Fix:* add CMC curves or rank-5/10 and mean reciprocal rank for the primary contrasts.

3. **The claim of a "reusable framework for auxiliary attributes" rests on one real attribute.** The abstract and §5 claim generality (l.22–23, l.587–600), but only handedness is tested. *Fix:* either narrow the claim, or apply the protocol to a second attribute. Check whether the CSAFE survey's other demographic fields (e.g. gender) can be used.

4. **Forensic context is missing.** Questioned-document practice treats handedness determination from static writing as unreliable. Citing that literature (e.g. standard handwriting-identification texts) would support the paper's own result and make it relevant to forensic readers.

**Minor**
- Ethics (l.690–698): handwriting images are biometric data. IET will expect the institutional policy that grants the waiver to be named or cited, not just asserted.
- l.604 ("Recruiting the left-handed group required substantial effort") does not support any claim; remove it.

**Practical significance score: 60/100.**

---

### Devil's Advocate report

**Strongest counter-argument (≈250 words).**
The headline result was predictable before any fusion experiment was run, so it is not informative. A predictor with 65.5% accuracy and log loss 0.602 (chance is 0.693) supplies about 0.09 nats per query. The paper's own break-even analysis puts the median accuracy needed for filtering to pay off at 86% (l.414–415). With 42 writers and intervals of about ±3 points, a null was close to certain whether or not handedness has real utility. CSAFE does not fix this: 96 writers, one query per condition, so one writer moves the result by 1.04 points, and a 58% cross-prompt predictor that is barely above chance. Calling this a replication (l.653–655, "reproduce the distinction") treats two underpowered nulls as agreeing evidence, but non-significance cannot be reproduced in that sense. On top of the null, the "framework" is an elementary probability identity, standard nested CV and a combinatorial expectation. What is left is a well-audited negative result on a weak predictor, presented as a methodological contribution. The paper's most defensible new finding, that identity errors are concentrated within the same hand and that this explains why known handedness gives no advantage over a random partition, is not stated in the main text at all.

**Issue list**

| Severity | Dimension | Location | Issue |
|---|---|---|---|
| **CRITICAL** | Logic / inference | Abstract l.21; §7 l.653–655; §5 l.544–547 | Non-significant results in two underpowered samples are presented as *reproducing* a finding. Without a declared effect margin, "no reliably demonstrated gain" cannot support the conclusion's comparative claims. Needs an equivalence/MDE analysis (R1-1) or rewording to "inconclusive at ±X pts". |
| MAJOR | Contribution | l.21–31, Table 8 | Novelty is overstated; the reusable product (an accuracy→utility curve plus a protocol checklist) is not delivered in the main text. S7's synthetic curve, *at what calibrated accuracy does soft fusion begin to help for a given matcher?*, is the reusable result and should be a main figure. |
| MAJOR | Confirmation bias | l.528–533 | A sensitivity model that falls below chance is described as a "sensitivity failure" and set aside rather than diagnosed. Anomalies that weaken the pipeline should be explained, not just labelled. |
| MAJOR | Alternative path | l.369, §3.4 | The strongest available predictor (multi-paragraph, 73.8–77.4%) is not the one tested for utility. The paper tests the hypothesis where it is least likely to succeed. |
| MAJOR | Overgeneralisation | l.22–23, l.587–600 | Generality to "auxiliary attributes in biometric candidate ranking" rests on one attribute. |
| MINOR | Presentation | throughout | The amount of hedging reads as defensive and makes it harder to tell what the authors believe. |

**Ignored alternative explanations**
- Known ≈ random partition because of *same-hand confusability* (Table S5), not because handedness carries no information. This is stated nowhere in the main text.
- The null may come from the NLL selection objective rather than from information content (oracle-λ is untested).
- Balanced galleries may hide asymmetric utility for left-hand queries.

**Missing stakeholder perspectives:** forensic examiners (rank-k shortlisting, natural prevalence); system designers deciding whether to collect handedness metadata at enrolment (a cost/benefit framing).

**Observations (non-defects):** leakage accounting in Table 2 is a model for the field. The numbers are internally consistent. The CSAFE protocol was fixed before outcomes were seen, and the calibration amendment was made before performance tables were inspected (S13.3). The distinction between writer-held-out and enrolment-trained prediction (Fig. 4) is careful and correct.

---

## Phase 2 — Editorial synthesis

### Consensus (raised by 3 or more reviewers, from different angles)
1. **Results are too imprecise for the conclusions as worded.** EIC-W2, R1-1, DA-CRITICAL.
2. **The best positive finding is buried.** R2-2 (same-hand confusability), DA (alternative explanation), EIC-W2 (contribution unclear).
3. **The generality and novelty claims are too large.** EIC-W2, R3-3, DA-MAJOR.

### Disagreements, arbitrated
- *Is the null itself publishable?* The DA says it was predictable. R1 and the EIC say a properly controlled null has value. **Arbitration:** it is publishable *if* the paper (a) states its resolution (MDE/equivalence), (b) explains the mechanism (Table S5), and (c) gives readers the accuracy→utility relationship. Without these it is a negative result with no explanation.
- *Should more matchers be added?* R2 asks for a handwriting-specific matcher; the EIC wants fewer display items. **Arbitration:** add one matcher on CSAFE only and put its results in the supplement; report the conclusion in one main-text sentence.

### DA CRITICAL issue (flagged)
The "reproduce" wording and the conclusions' comparative claims must be supported by an equivalence or MDE analysis, or reworded. Under the review rules, the decision cannot be Accept while this stands.

### Scores

| Dimension | Score (0–100) | Basis |
|---|---|---|
| Originality / novelty | 58 | Random-partition benchmark is new; other parts are standard (EIC-W2, DA) |
| Methodological rigor | 74 | Excellent leakage control; weak inferential resolution, broken sensitivity calibration (R1) |
| Evidence supports claims | 68 | Null claims are worded carefully, but "reproduce" and generality claims go further than the data (DA, R3-3) |
| Literature & positioning | 55 | 26 refs; missing modern and Bangla writer ID and soft-biometric fusion work (R2-1) |
| Clarity & presentation | 52 | Clean figures; heavy hedging, 15 display items, numeric abstract (EIC-W3/W4) |
| Significance / impact | 60 | Useful lesson for soft-biometric evaluation; limited by balanced galleries and top-1 only (R3) |
| Reproducibility & transparency | 92 | Public code, frozen features, manifests, hashes, protocol amendments documented |
| **Overall (weighted)** | **≈ 65 / 100** | |

**Rating:** solid, careful, and below the bar for acceptance as it stands. The weakness is less how the work was done than how its findings are framed and how precise they are.

### Decision: **MAJOR REVISION**

The study is rigorous and transparent and fits IET Biometrics. It currently presents an underpowered null as a replicated conclusion, overstates a "framework" built from mostly standard parts, and keeps its most explanatory result in the supplement. All of the required analyses can be done with data and code you already have.

### Revision roadmap (prioritised)

**P1 — required for acceptance**
1. Declare a smallest effect of interest; add TOST or MDE for the 4 BanglaHand + 8 CSAFE primary contrasts. Reword l.21, l.544–547 and l.653–655 to match. *(R1-1, DA-CRITICAL)*
2. Move Table S5 into §4.3 and state the same-hand-confusability mechanism for known ≈ random partition. Make it a headline finding in the abstract. *(R2-2, DA)*
3. Fix the CSAFE adjusted-HC-183 calibration (slope ≥ 0 or an alternative calibrator), rerun, and explain the sign flips. *(R1-2, DA)*
4. Rerun CSAFE primary contrasts with the natural-prevalence gallery (471 writers), split by query hand. *(R3-1)*
5. Narrow the title, abstract and contribution claims; replace "framework" language with a numbered evaluation protocol box. *(EIC-W1/W2, R3-3, DA)*

**P2 — strongly recommended**
6. Add an oracle-λ ceiling for soft fusion. *(R1-5)*
7. Test utility at the best achievable predictor accuracy (multi-paragraph / companion model). *(R2-4, DA)*
8. Promote the S7 synthetic accuracy→utility curve to a main figure. *(DA)*
9. Add rank-k / CMC / MRR for the primary contrasts. *(R3-2)*
10. Expand the literature (modern writer ID, Bangla writer ID, soft-biometric fusion, forensic handedness), aiming for 40+ verified references. *(R2-1, R3-4)*

**P3 — presentation**
11. Move Fig. 5, Table 6, Table 8 and strip-segmentation details to the supplement; consolidate the disclaimers into §3.7. *(EIC-W3/W4/W5, R2-minor)*
12. Explain the Stroke-CNN quarter-strip collapse; rename "Stroke-CNN". *(R1-6, R2-minor)*
13. Reconcile the Table 4 caption with §3.7 and add the fixed-fusion Holm p-values. *(R1-minor)*
14. Cite the ethics-waiver policy; remove l.604. *(R3-minor)*

**Expected effect:** P1 + P2 would likely move this to minor revision or accept territory (overall ≈ 75–80). The two largest gains come from P1-2 and P1-4, because they turn an unexplained null into an explained mechanism plus an operational finding.
