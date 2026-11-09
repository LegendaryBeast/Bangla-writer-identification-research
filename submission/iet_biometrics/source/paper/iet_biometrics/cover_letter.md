---
title: Cover letter
date: 9 November 2026
---

Editors, IET Biometrics

Dear Editors,

Please consider “Does predicted handedness improve writer identification? Oracle gains versus achieved utility in Bangla and English handwriting” as a Research Article in IET Biometrics.

Soft-biometric attributes are usually judged by how accurately they can be predicted. The study asks the question that matters for identification: does a predicted attribute move the true writer up the candidate list? We answer it for handwriting handedness with a paired protocol that fixes the gallery and identity scores and varies only the auxiliary information. The protocol separates the opportunity offered by a correct attribute, its specificity against a known random partition, the utility achieved by calibrated fixed and properly nested fusion, the decision-level mechanism, and writer-associated information. It is relevant to IET Biometrics readers working on soft biometrics, information fusion and biometric performance evaluation.

The results are as follows:

- Across 42 BanglaHand writers, six representations and 120 settings, known handedness improves top-1 accuracy by 6.1 percentage points, but an arbitrary known 21/21 partition improves it by 6.9.
- The reason is that identity errors cluster within hand. When one impostor outranks the true writer, it has the opposite hand in only 34–37% of cases for the neural matchers, against 51% for a random impostor.
- With query handedness predicted at 65.5% accuracy, hard filtering loses 14.3 points.
- Nested fusion stays within ±1 point of identity-only ranking across settings. Its 90% interval excludes recovering more than 11% of the oracle gain.

A cross-session check on 96 CSAFE writers, balanced by handedness, uses gallery and query pages from different capture sessions. None of eight primary fusion comparisons shows a gain. We state that this cohort resolves effects only beyond about ±3 points.

The companion preprint, “Writer-independent handedness identification from offline Bangla handwriting under controlled acquisition” (doi:10.2139/ssrn.7662278), evaluates attribute classification and photometric cues on the same BanglaHand cohort with overlapping representations. The present work adds the downstream identity-ranking endpoints, the oracle and random-partition controls, nested fusion, the precision and error-clustering analyses, and the independent CSAFE cohort. The relationship is cited and explained in the manuscript.

The supporting information gives the full protocols, results and reproducibility details. Post hoc analyses are labelled as such, including a numerical correction to one secondary CSAFE sensitivity model that does not affect any primary result.

The BanglaHand code, frozen features and numerical results are in the repository's versioned v1.6.0 release, linked in the manuscript, and the data are at doi:10.17632/n4sjstdzfy.1. Two archives accompany this submission: one with the CSAFE code, features and results, and one with the revision analyses. The CSAFE source data are at doi:10.25380/iastate.10062203.v2.

This manuscript is not under consideration by another journal. Thank you for considering the work.

Sincerely,

Tanzim Hasan Prappo  
Corresponding author  
Department of Computer Science and Engineering, Shahjalal University of Science and Technology, Sylhet 3114, Bangladesh  
2021331006@student.sust.edu
