# Independent domain-expert review

**Manuscript:** *Soft biometrics for writer identification: A utility law from Bangla handedness*  
**Reviewed file:** `paper/wid_manuscript.pdf`, seven pages  
**Review date:** 8 November 2026  
**Recommendation:** **Major revision**

This assessment concerns the submitted PDF, supplemented by targeted checks of primary literature and dataset records. The analysis code and raw experiments were not audited or rerun. Page references below refer to the PDF. Mathematical checks use the manuscript's rounded reported values.

## Overall assessment

The manuscript asks a useful question that attribute-classification studies often leave unanswered: does predicting handedness actually improve writer identification? The distinction between known query/gallery attributes, predicted query attributes, and predictions on both sides is valuable. On this collection, the reported evidence supports a clear practical finding: hard gallery filtering with the tested writer-independent handedness predictor substantially reduces identification accuracy. The random-label experiment also supplies a useful diagnostic for writer-associated information in attribute prediction.

The present framing exceeds those findings. Equation (1) is an explicitly conditional independence model, subsequently promoted into a broadly applicable “utility law.” The primary tests do not exclude small positive fusion effects, and the manuscript's synthetic and both-predicted experiments themselves show that accuracy alone does not determine utility. Claims about current offline handwriting, soft biometrics in general, and forensic casework need substantial qualification. Several missing protocol details and a placeholder code DOI also prevent a fully independent evaluation.

I see a potentially publishable empirical and evaluation-protocol contribution after these issues are addressed. The strongest version would explain **when the tested handedness evidence helps or harms this identification pipeline**, with its assumptions and uncertainty explicit.

## Strengths

- The work measures downstream identification utility rather than treating attribute accuracy as sufficient evidence of usefulness.
- Holding the queried writer out of attribute training addresses an important generalization issue. Writer-grouped inner calibration is also a positive design choice.
- Multiple representations, query lengths, and reference conditions probe substantial variation within the available collection.
- The known-attribute benchmark makes the gap between potentially useful information and imperfect predictions interpretable.
- Writer-level paired differences, writer-bootstrap intervals, four declared primary contrasts, and Holm correction are stronger reporting choices than treating every strip or setting as independent.
- The manuscript discloses its small, single-region, single-device collection and distinguishes exploratory analyses from its primary comparisons.

## Major comments

### 1. Present Equation (1) as an independence approximation and validate its residuals

**Locations:** p. 2, §4 and Eq. (1); p. 3, §5.3; abstract; pp. 4–5, Discussion and Conclusion.

Let \(C\) denote correct query-handedness prediction, and \(K\) denote correct identification when the true hand is supplied. Under the stated binary filtering rule, both hand groups are nonempty, so filtered top-1 success is exactly the event \(C\cap K\). Consequently,

\[
A_{\mathrm{filter}}=P(C\cap K)
=aP(K\mid C)
=aA_{\mathrm{known}}+
\operatorname{Cov}(\mathbf{1}_C,\mathbf{1}_K).
\]

The paper explicitly assumes independence when deriving its product. The derivation is therefore not an algebraic error. The problem is the later suggestion that two marginal accuracies generally determine filtering utility. Shared image quality, writer difficulty, text amount, and overlapping feature representations can associate the two correctness events.

The high pooled correlation across 456 combinations is encouraging descriptive agreement, but these are correlated evaluations on the same writers. It does not establish that the residual is negligible for deployment decisions. For example, Table 2 reports 65.5% handedness accuracy on own-text paragraphs. For HC-183 identification in Table 3, the product predicts \(0.655\times49.2\%=32.2\%\), whereas filtering achieves 28.0%: a **−4.2-point residual**. For ViT-B/16, the corresponding residual is approximately −3.1 points. These differences are consequential relative to the small fusion gains discussed elsewhere.

**Required revision:** State the exact conditional identity first, label the product an independence approximation, and condition the break-even rule on that assumption. Report residuals and \(P(K\mid C)\) versus \(P(K\mid\neg C)\), stratified by representation and query condition, with writer-level uncertainty. Add a residual plot alongside the pooled correlation. The cross-hand fraction of identification errors does not test the correctness-event independence required here.

The 87% median is also specific to the tested collection and mix of settings. Using Table 3's own-paragraph results gives the following approximate thresholds under the independence model:

| Identification representation | Baseline top-1 | Known-hand top-1 | Approximate filtering threshold |
|---|---:|---:|---:|
| STROKE | 92.7% | 95.4% | 97.2% |
| HC-183 | 42.1% | 49.2% | 85.6% |
| ResNet-18 | 61.7% | 68.7% | 89.8% |
| ViT-B/16 | 68.7% | 73.8% | 93.1% |

These differences should be prominent whenever the manuscript states an accuracy requirement.

### 2. Distinguish “no detected improvement” from evidence that useful improvement is absent

**Locations:** abstract; p. 3, §5.2; p. 4, Table 4; p. 5, Conclusion.

The four primary tests show no statistically demonstrated gain. Their confidence intervals nevertheless permit positive effects of 0.9–2.7 percentage points, depending on the setting. Failure to reject zero does not establish that a practically useful gain is absent.

The conclusion that a soft biometric helps only when it is known or predicted far more accurately than current offline handwriting allows is especially difficult to reconcile with the manuscript's own results. Synthetic evidence at the achieved attribute accuracy can improve identification, and the exploratory both-predicted scenario reports a +0.95-point average gain. These results indicate that calibration, error dependence, and the operational scenario matter alongside accuracy.

**Required revision:** Use language such as “no reliable improvement was detected in the primary predicted-query experiments.” Restrict conclusions to the tested predictors, representations, population, and acquisition regime. If claiming that gains are negligible, define a practically meaningful margin and assess whether the confidence intervals exclude it; do not select a margin retrospectively to obtain a desired conclusion. Report uncertainty for the average across settings rather than relying on the number of individually significant settings.

### 3. Establish what the probabilistic fusion and synthetic experiment actually identify

**Locations:** p. 2, §4; pp. 3–4, §5.3; p. 4, §5.5; Figs. 3 and 4.

The ranking score multiplies an identity-score softmax by a handedness posterior derived from the same handwriting. This is not automatically a Bayesian posterior update: the evidence sources can be dependent, particularly when both components use HC-183. A prior term may cancel in the balanced setting, but the distinction between a posterior and a likelihood ratio must be clear when discussing prevalence.

The Gaussian simulation specifies one family of calibrated evidence and one relationship to identification difficulty. Its approximate 60–80% or 70% thresholds do not establish a general threshold for Bayesian fusion. Different confidence distributions and different alignments of errors with identification margins can yield different utility at the same attribute accuracy. The observed gap between real and synthetic evidence already illustrates this limitation.

The exact fusion formula for uncertain gallery handedness in the both-predicted scenario is also missing. Further, separate tests of handedness and random controls against zero do not test the difference between handedness and those controls. A +0.95-point gain versus smaller positive control gains is suggestive, but needs a direct paired comparison before attributing the difference specifically to handedness.

**Required revision:** Derive the assumptions behind the fusion rule or describe it as probabilistic score fusion. Specify the exact both-predicted formula, calibration/training priors, and treatment of prevalence. Report held-out-writer calibration with log loss or Brier score and reliability plots, plus fitted fusion-weight and temperature summaries. State the synthetic class-conditional score distribution and whether parameters are refitted at every accuracy. Compare both-predicted handedness directly with the controls using paired differences and intervals. Treat its interpretation as exploratory.

### 4. Make the entire validation hierarchy explicit

**Locations:** p. 2, §§3–4 and Algorithm 1.

The PDF establishes that each query writer is excluded from its own attribute model and that inner Platt calibration is grouped by writer. Those are strengths. It does not fully specify the training population for every other learned transformation and fusion component.

“Standardised without labels” does not say which samples fit the standardization. It is also unclear how nuisance regression and fusion tuning are nested. A potential issue arises if globally assembled leave-one-writer-out predictions are reused to fit fusion: the attribute model producing development predictions for writer \(v\) may have included the eventual test writer \(w\). Excluding \(w\)'s queries from the final fusion fit does not necessarily exclude \(w\)'s information from the generation of those development predictions. This is a risk to clarify, not demonstrated contamination from the PDF alone.

**Required revision:** Add a nesting diagram and a table stating the permitted writers, paragraphs, labels, and gallery metadata for standardization, nuisance regression, feature/model selection, SVM fitting, Platt calibration, temperature fitting, and fusion-weight fitting. Regenerate development evidence within the outer exclusion when claiming strict unseen-writer generalization. Alternatively, explicitly define any transductive or enrolment-conditioned protocol and its intended interpretation.

Also document the feasibility analysis that preceded the protocol: what was inspected, which settings were covered, and what decisions followed. A dated protocol or version history would clarify how much of the four primary comparisons was specified before their outcomes were available. Explain whether bootstrap intervals condition on the fitted predictions/gallery or include model refitting.

### 5. Narrow what the random-label control proves

**Locations:** abstract; p. 4, §5.4; p. 5, Discussion and Conclusion.

Predicting arbitrary writer-level labels under enrolment training is a persuasive demonstration that writer-associated information can drive an attribute classifier. It does not decompose 82.1% real-handedness accuracy into identity information and genuine handedness information. Therefore, “most” of the enrolment improvement reflecting the writer and “such accuracy measures writer recognition” are stronger statements than the control establishes. Single-session acquisition also leaves stable writer characteristics and persistent acquisition signals difficult to separate.

Writer overlap is inappropriate as an estimate of attribute generalization to unseen writers. It can nevertheless be legitimate in an explicitly enrolment-conditioned identification system. The issue is whether the evaluation matches the advertised target, rather than writer overlap being inherently invalid.

**Required revision:** State that overlapping-writer accuracy **can substantially reflect writer-associated information** and does not estimate unseen-writer attribute accuracy. Explain the constrained random-label generation and provide its full distribution and uncertainty. Because fixed balanced labels and leave-one-out training can produce a non-50% null, interpret the reported 33.3% result against a matching randomization procedure. Clarify “balanced for hand,” including how odd-sized hand groups are handled.

### 6. Separate the information value of a known partition from handedness-specific and forensic claims

**Locations:** p. 2, definition of the known scenario; p. 3, §5.1; pp. 4–5, Discussion; abstract.

A correct balanced binary label removes roughly half the impostor gallery even if the label is an arbitrary writer partition. The +6-point oracle benefit is useful, but does not establish a distinctive physiological advantage of handedness. The existing predicted-random-label control answers a different question.

Likewise, the 44–53% cross-hand error fraction compared with 51.2% is a descriptive observation. Without clustered uncertainty or an appropriate test, it does not establish independence of errors from handedness.

The operational interpretation also needs care. The known scenario supplies the hand of both the questioned writer and the gallery writers. Knowing a suspect's hand supplies gallery information only; it does not determine the questioned writer's hand. The closed-set evaluation assumes the true writer is enrolled and measures ranking within 42 writers from one acquisition regime. It does not establish forensic attribution reliability or performance with an unenrolled writer, cross-session documents, or disguised/nondominant-hand writing.

**Required revision:** Compare known handedness against repeated known balanced-random partitions on the same rankings, with writer-level uncertainty. Matching the random-partition benefit would still support the practical filtering result; it would clarify what information the oracle contributes. Present known-hand results as an oracle or externally informed candidate-ranking benchmark. Replace unconditional casework advice with a statement conditional on reliable query and gallery attributes. Broader operational claims need broader validation; a carefully bounded paper need not solve every forensic deployment problem.

### 7. Strengthen and correct the literature positioning

**Locations:** pp. 1–2, Related work; Tables 1 and 5; p. 5, limitations.

The credible novelty is the particular joint evaluation of Bangla writer identification, known/predicted handedness, filtering thresholds, and writer-associated prediction controls. The present comparison does not establish a general first result simply by being the first row to tick all selected columns.

Table 5 explicitly judges absence of prior methods from titles and abstracts. Those sources cannot establish that a full article omitted an analysis. The most relevant papers should be checked in full, including their writer splits and any attribute-assisted identification experiments. The primary [Al-Maadeed et al. handedness article](https://link.springer.com/article/10.1186/s13640-015-0097-y) is a relevant starting point. This review did not establish a prior handwriting paper containing the same complete experimental package; the concern is insufficient support for exclusivity, rather than a demonstrated duplicate contribution.

One factual claim needs narrowing: “No other Bangla collection with handedness labels exists” is too absolute. The [BDSHWA primary dataset record](https://data.mendeley.com/datasets/99t9jhvksv/1), published 28 April 2026, describes Bengali–English online handwriting from 29 participants and reports 28 right-handed and one left-handed participant. This does **not** make it suitable for balanced replication of the present offline experiment. It does undermine the unqualified absence statement.

**Required revision:** Verify coverage from full texts and use “not reported” or “unclear” when warranted. Narrow the availability statement to the absence of an identified suitable independent **offline Bangla benchmark with sufficient left-handed representation**. Discuss the distinction rather than implying that this small online collection would be an adequate replication dataset.

### 8. Supply a self-contained, accessible reproduction package

**Locations:** p. 2, feature and sensitivity-analysis descriptions; p. 5, availability statement; p. 7, reference [14].

The public [BanglaHand record](https://data.mendeley.com/datasets/n4sjstdzfy/1) supports the stated 42-writer, balanced photographic collection. However, the analysis-code reference is still `doi:10.5281/zenodo.XXXXXXX`, and important methods are delegated to another submitted manuscript. The current submission therefore cannot be independently reproduced from its cited materials.

**Required revision:** Replace the placeholder with an accessible archive or reviewer supplement. Include feature definitions, frozen-model checkpoints, tile sampling/aggregation, image and mask preprocessing, normalization scope, strip exclusion rules, complete split manifests, tuning procedures, random seeds, simulation code, and per-writer results. Explain how the 456 combinations are constructed from the base settings and alternative handedness models. Provide a stable preprint of [14] or reproduce its essential methods in a supplement.

## Additional comments

1. Distinguish percentage points from relative percentages consistently, especially when summarizing gains and thresholds.
2. Explain how writer-level accuracy is constructed from unit predictions, and distinguish it from unit-level query accuracy when discussing practical prediction quality.
3. Report excluded strip counts by writer, hand, prompt condition, and size; state how writers with different surviving unit counts are weighted.
4. Distinguish top-1 identification, verification, and character recognition results in literature tables. Their reported percentages are not directly comparable performance benchmarks.
5. Explain why the selected primary comparisons are representative, particularly given STROKE's much stronger identification performance. This is a reporting request, not a requirement to add post hoc primary tests.
6. Give uncertainty and aggregation definitions for prevalence experiments, and separate minority-query benefit from population-weighted performance.
7. Table 5 would be more informative if it prioritized evaluation protocols and actual downstream fusion results over numerous unverified negative cells.
8. A title emphasizing an empirical evaluation or a conditional model would better match the evidence unless the independence and generalization issues are resolved.

## Revision priorities

1. **Correct the theoretical and empirical scope:** state the conditional filtering identity, qualify thresholds, and replace universal negative conclusions with the actual primary findings and their uncertainty.
2. **Resolve fusion and validation ambiguities:** specify both-predicted fusion, calibration, evidence dependence, and the full nesting hierarchy.
3. **Add targeted controls using the existing collection:** known random partitions, conditional-error/residual analyses, and direct paired comparisons against both-predicted controls.
4. **Complete reproducibility and literature verification:** release the missing materials, document advance specification, correct the dataset-availability statement, and narrow casework implications.

The present manuscript needs substantive revision rather than copyediting. Its central empirical warning remains valuable, and the required changes can largely be addressed through clearer claims, transparent protocol documentation, and focused analysis of the existing data. External or cross-session validation becomes essential if the authors retain broad forensic or cross-population recommendations.
