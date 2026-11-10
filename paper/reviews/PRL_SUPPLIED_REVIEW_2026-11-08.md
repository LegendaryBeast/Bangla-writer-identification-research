# PRL Manuscript Review

Journal: Pattern Recognition Letters

Paper: When does handedness help writer identification? A controlled evaluation on Bangla handwriting

I reviewed your 7-page manuscript alongside its 14-page supplementary document.

Your central finding is scientifically meaningful: knowing a writer's handedness can improve identification, but predicting handedness does not necessarily provide useful additional evidence. Your experiments distinguish this carefully.

The strongest quantitative results are:

| Experiment                          | Change in top-1 accuracy |
| ----------------------------------- | ------------------------ |
| Known handedness                    | +6.1 percentage points   |
| Known random binary partition       | +6.9 points              |
| Predicted-handedness fusion         | −0.6 points              |
| Nested learned-weight fusion        | −0.08 points             |
| Predicted-handedness hard filtering | −14.3 points             |

The comparison with random partitions is particularly important: it shows that the oracle improvement is not demonstrably specific to handedness. The supplement also provides substantially more methodological detail than is usually possible in a short paper.

Overall PRL submission rating

# 7.4/10

Borderline competitive

Recommendation: Submit after targeted revisions. The strongest risk is insufficient novelty or external validation, not obvious methodological unsoundness.

## 1. Detailed ratings

| Criterion                      | Score  |
| ------------------------------ | ------ |
| Originality and novelty        | 6.8/10 |
| Technical soundness            | 8.5/10 |
| Experimental methodology       | 8.2/10 |
| Statistical analysis           | 8.5/10 |
| Strength of empirical evidence | 6.5/10 |
| Writing and organization       | 8.5/10 |
| Reproducibility documentation  | 9.0/10 |
| PRL scope and relevance        | 8.0/10 |

The reproducibility score concerns the documentation supplied. I have not executed the code or independently reproduced the results, and the mentioned reproducibility ZIP was not among the two uploaded files.

## 2. Why the paper is promising

The strongest aspect is the careful evaluation protocol. Your paper separates oracle metadata, predicted metadata, error dependence, and writer-associated information instead of reporting only handedness-classification accuracy.

I especially value the following:

- Properly nested fusion-weight selection: Section 3.3 and Supplement S8 address selection bias rather than tuning on the evaluated writer.
- Random-partition and random-label controls: These provide stronger interpretations than a simple baseline comparison.
- Statistical discipline: Writer-level bootstrap intervals, paired tests, Holm correction, and explicit caveats about correlated settings.
- Reproducibility: The supplement contains feature specifications, fitting hierarchies, complete results, robustness diagnostics, and commands for reproduction.

PRL explicitly includes biometrics, document processing, machine learning, and empirical pattern-recognition studies within its scope.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://shop.elsevier.com\&sz=32)

Elsevier Shop



## 3. Biggest likely reviewer criticisms

High

Limited dataset generalization

The experiment uses only 42 writers from one collection and one photographic session. The 120 settings create diverse comparisons, but not independent populations. This restricts how broadly the negative fusion result can be interpreted.

High

Contribution may be seen as primarily evaluative

The filtering identity is mathematically elementary, and you do not introduce a new identification architecture. The genuine contribution is the controlled evaluation framework and its empirical findings. That is defensible, but some reviewers may expect greater methodological novelty.

Medium

Deployment interpretation of writer-held-out models

Your cross-validation protocol excludes the true query writer from attribute training, which is appropriate for estimating unseen-writer generalization. However, deployment cannot select a different model based on an unknown query identity. Clarify that this is a cross-validated performance estimate, not literally one deployable universal predictor.

Medium

Strength of the identification benchmarks

STROKE provides a strong comparison, but the neural embeddings are frozen ImageNet features rather than handwriting-specialized identification systems. Reviewers could question whether the conclusion extends to stronger modern systems.

Medium

Exploratory positive result needs clearer context

Supplement S6 reports a +1.08-point gain for both-sided descriptor agreement, with a conditional interval of [+0.50, +1.72]. That result is interesting but uses a different information protocol and has limitations. Explain its significance and those limitations more visibly in the main paper.

## 4. Five changes I would prioritize

1. Strengthen the novelty statement in the introduction. Clearly identify the contribution as a controlled evaluation of downstream soft-biometric utility, not a new fusion algorithm or new mathematical theorem.
2. Add independent-cohort validation if feasible. This would be the single most valuable improvement. If unavailable, preserve the carefully restricted claims you currently make.
3. Clarify the operational meaning of writer-held-out attribute prediction. Separate cross-validation estimation from how a predictor would actually be trained and deployed.
4. Make the practical lesson more prominent. Attribute accuracy alone is insufficient; the joint correctness distribution, calibration, and identification margins determine utility.
5. Improve Figure 2 readability and the discussion of S6. The dense axis labels and exploratory positive result could be communicated more clearly without greatly expanding the paper.

## 5. Formatting readiness

Your seven-page main manuscript appears to fit the page limit stated in Elsevier's published PRL formatting instructions. Those instructions also specify an abstract of at most 200 words, required research highlights, and an authorship confirmation document. Verify the current submission checklist before uploading.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://legacyfileshare.elsevier.com\&sz=32)

legacyfileshare.elsevier.com



## 6. Final editorial judgment

Scope match

Good

Methodological credibility

Strong

Novelty threshold

Borderline

Experimental generalization

Limited

Submission readiness

Minor-to-targeted revision

My decision as a simulated PRL reviewer would be: Borderline / Weak Accept on methodological quality, but with substantial concern about novelty and the single-cohort evidence.

An editor or reviewer who values carefully controlled empirical studies could see this as a worthwhile short paper. A reviewer who requires a new recognition method or stronger external validation could reject it.

I would submit this to PRL after strengthening the framing and clarifying the evaluation-to-deployment distinction. I would not recommend spending months developing an entirely new architecture unless reviewers specifically require it. Your strongest contribution is the rigor of the evidence, and the paper should lead with that.