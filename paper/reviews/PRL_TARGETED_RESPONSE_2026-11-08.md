# Response to the supplied PRL review — version 1.4.0

The supplied review evaluated an earlier title and a 14-page supplement. Version 1.3.0 had already added the representation-specific figure and updated the author list, title and companion citation. This revision addresses the remaining comments against the current source. No additional dataset, participant or photograph was introduced, and all 42 writers remain in every primary setting. The requested two-author list and exact AI disclosure are preserved.

## 1. Limited population generalization

**Addressed by restricting interpretation; external validation remains an acknowledged limitation.** The main abstract, inference section, robustness section, discussion and conclusion identify the evidence as conditional, closed-gallery and single-cohort. The 120 settings reuse writers and do not stand in for new populations. Calibration variants, writer influence, margins and simulations remain within-cohort diagnostics. No independent-cohort or cross-session claim has been added. The review's suggestion to add another dataset is not pursued, following the user's explicit constraint. These changes improve interpretation but do not resolve population generalization empirically.

## 2. Evaluative contribution and novelty

**Addressed in the introduction.** The paper now leads with a controlled empirical framework for downstream soft-biometric utility. The joint design separates correct binary partition information, attainable predictor/fusion gains, correctness dependence, and writer-associated label information. The elementary filtering identity and standard fusion rules are explicitly diagnostic tools and comparators. No new theorem, identification architecture or universal utility law is claimed. Redundant explanations were removed to keep the seven-page limit.

## 3. Cross-validation versus deployment

**Addressed in main Section 3.2 and Supplement S3.** Known writer identifiers assign evaluation folds; unknown identity is not an inference input. A real unseen-writer pipeline would train one attribute model, calibrate and select temperature/weight from separate development writers, freeze the pipeline, and apply the same fitted rule to every future query. The reported outer fits estimate held-out-writer performance and are not a single universally deployed predictor or a rule that selects a model by the query's true identity. Using enrolled-writer training pages instead changes the generalization target. The same distinction now also contextualizes the both-sided diagnostic.

## 4. Strength of identity comparators

**Addressed by making the boundary explicit.** The discussion states that the neural representations are frozen ImageNet comparisons and that utility for handwriting-specialized, fine-tuned or stronger systems is not established. Different identity margins and errors could change fusion utility. The paper does not treat these experiments as state-of-the-art recognition benchmarking. Developing a new architecture is not necessary to interpret the controlled comparisons and would not itself supply external population validation.

## 5. Positive exploratory both-sided result

**Addressed visibly in main Section 4.5, with Supplement S6 retained.** The main text now gives the +1.08-point top-1 gain and conditional interval [+0.50,+1.72] over the 96 self-chosen-text settings. It explains that both-sided descriptor agreement replaces known gallery-hand labels with predicted reference descriptors and uses a different score rule. An impostor reference model can contain the query writer's shared enrolment pages. Shared-text query results are excluded because actual questioned pages would enter impostor fitting. This is an enrolled-cohort descriptor-consistency diagnostic, not proof of a specifically physiological handedness benefit or unseen-cohort transfer. Its positive effect therefore does not contradict the primary protocol's result. Existing random-label/random-direction controls and their limits remain in S6.

## 6. Practical lesson: accuracy, dependence, confidence and margins

**Addressed with an exact decision reconstruction and an additional supplementary figure.** Because binary attribute evidence is constant within a hand group, fusion preserves within-hand rank. For the true writer to win, it must first lead its group; its scaled margin against the strongest opposite-hand candidate plus weighted signed attribute log-odds must then exceed zero, subject to the declared tie rule. The true writer/hand are used only for retrospective diagnostics. This makes explicit how confidence and identity margins mediate downstream utility, alongside the filtering joint-correctness identity.

`code/run_margin_diagnostic.py` reused the existing features, calibrated scores, temperatures and nested weights. It reconstructed all 664,680 fixed/nested query/reference decisions across the 120 settings exactly, with no exact boundary ties. It independently reproduced every writer-level fixed/nested top-1 value. Rescued-minus-damaged equals the accuracy change for every decision and every writer/setting.

Across settings, fixed fusion rescues 2.02% of decisions but damages 2.59%; nested fusion rescues 1.33% but damages 1.42%. These are equal-writer, equal-setting averages, not pooled counts across differently sized settings. Supplement S10 provides the derivation, conditional intervals, rescue/damage bars and an illustrative margin plot. The diagnostic does not tune a new rule or claim independent validation. Input hashes, reconstruction checks, intervals and writer-level outcomes are supplied in the reproducibility package.

## 7. Figure readability

**Addressed for both numbering versions.** The review's original Figure 2 is now Figure 3 (filtering diagnostics). Its representation labels are horizontal, residual boxplots and oracle bars share aligned representation rows, and a common legend replaces crowding inside the small panels. The current Figure 2 (representation-specific utility) also has larger markers, shared rows, and removes duplicated labels to give the paired-comparison panel more space. Vector artwork is retained. Final pages and separate figures were rendered for visual inspection.

## 8. Formatting and submission materials

**Manuscript materials prepared; author attestations remain author-only.** Verified PRL's live guide in a browser on 8 November 2026:

https://www.sciencedirect.com/journal/pattern-recognition-letters/publish/guide-for-authors

The mandatory submission section confirms double-column format, a seven-page maximum including references, institutional emails in the first-page footnote and submission system, separate highlights (3–5 items, at most 85 characters each), and the journal authorship form uploaded as Cover Letter. The linked official template additionally specifies a 200-word abstract. The main manuscript remains seven pages, its abstract is 179 words, institutional emails appear for both remaining authors, and the five separate highlights satisfy the stated length limit.

Downloaded the unmodified official authorship/formatting document:

https://legacyfileshare.elsevier.com/promis_misc/patrec-authorship-and-formatting.pdf

It is supplied in `submission/revised/PRL-authorship-and-formatting-official.pdf`, with a companion `AUTHORSHIP_COMPLETION.md` containing known manuscript details and fields the authors must verify. No signature, final approval, publication-history assertion or editor authorization has been fabricated. The existing conference/blind drafts were not changed or submitted. Any current-manuscript preprint/conference status must be determined by the authors; the cited SSRN companion paper is not automatically a preprint of this manuscript.

## 9. Reproducibility and final checks

The code and complete version 1.4.0 reproducibility archive accompany this revision. Main and supplementary sources, vector figures, scientific results, frozen feature inputs, input hashes, reconstruction checks and figure code are included. The main integrity validator now verifies the new margin inputs and independently checks the decompositions, means and agreement with the earlier writer outcomes, alongside the prior scientific invariants. Numerical/document checks and rendered-page inspection are separate from evidence of external generalization. No acceptance prediction or zero-error certification is made.
