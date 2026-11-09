# Version 1.3.0 manuscript revision (8 November 2026)

The main manuscript now has the title *Handedness for Bangla writer identification: Oracle gains and predicted-attribute utility*. The title makes the measured downstream contribution explicit and states the dataset/script scope.

## Companion-paper comparison

Verified the SSRN landing record in a browser and inspected the complete seven-page PDF, not only the abstract. Primary source: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7662278; DOI 10.2139/ssrn.7662278. Posted 23 October 2026; preprint, not peer reviewed. The bibliography retains all five authors of that separate work.

The companion's Table 1 reports HC-183 writer balanced accuracy of 77.4% in condition A and 75.0% in shared-to-own transfer (condition C). Its methods (Section 3.2) average raw paragraph SVM margins over a test writer's selected paragraphs, using 20 repetitions of stratified five-fold writer-disjoint cross-validation. Condition A aggregates six shared paragraphs; condition C aggregates two own paragraphs. Standardization is training-fold only and SVM C=0.1. These claims concern attribute prediction and acquisition/photometric cues.

The present 65.5% result is calibrated single-own-paragraph prediction, with leave-one-writer-out SVM fitting and writer-group out-of-fold calibration inside the other 41 writers. Identity ranking then uses that probability on the same questioned unit as the identity score. The writer-level 73.8% figure in this article instead averages calibrated log-odds over all eight paragraphs. Neither is the companion's exact endpoint: folds, training-set sizes, aggregation, calibration and text conditions differ. Numerical equality is therefore not expected, and there is no matched-protocol discrepancy established by these values.

No direct contradiction was found. Above-chance handedness prediction does not imply a downstream identity gain: incorrect predictions, ranking margins, calibration and dependence can remove the oracle advantage. The companion does not demonstrate improved writer identification under this article's tested rules. Its photometric findings are neither independently replicated nor causally resolved here. Both articles reuse the same 42-writer cohort and overlapping descriptors; the new paper explicitly cites that overlap and differentiates downstream utility controls from the earlier attribute study. This is an extension, not an independent dataset replication.

The complete companion PDF was inspected locally; its SHA-256 is 6b4206816373ae0892962de465c8bc9b94dcff182f05b51df253c61fb3fd7a2f. It is not redistributed in the reproducibility archive.

## Added analysis and figure

`code/make_representation_utility.py` reads revised writer-level fixed-rule outcomes and properly nested learned-rule outcomes. It retains every writer and all 20 settings per representation. Within each representation it averages settings within writer and then gives writers equal weight. It generates 30 paired contrasts (five per representation): known, fixed, nested and filter versus no handedness, plus nested versus fixed.

Uncertainty uses 10,000 hand-stratified writer bootstraps, seed 20361108, with shared resampling indices across every contrast. These are conditional marginal exploratory intervals, not simultaneous confidence bands, pipeline-refitting intervals or evidence from new writers/sessions/galleries. The six representations are not selected using this analysis. No difficult writer was excluded and no classification model was retrained for the added figure.

The new main-text figure displays oracle and predicted utility by representation, with a separate scale for nested-minus-fixed differences. Nested-versus-baseline mean changes range from -0.64 to +0.42 percentage points; nested-minus-fixed changes range from +0.18 to +1.08 points. Every corresponding nested-versus-baseline interval crosses zero. The aggregate conclusions remain unchanged.

Machine-readable results are `representation_utility_intervals.csv`; fitting scope, input hashes and reconstruction checks are in `representation_utility_provenance.json`, both under the fusion/robustness results directory. Figure assets are vector PDF plus PNG and numeric macros. The main integrity validator independently reconstructs all 30 estimates from underlying outcomes and verifies source hashes.

## Author list and disclosure

At the user's explicit request, the revised article and supplement retain Tanzim Hasan Prappo and M. Shahidur Rahman. Removed the second, third and fourth authors from the current byline, PDF metadata, current software citation author list and CRediT statement. Existing contribution roles for the retained authors were preserved; no removed contribution was reassigned. Author lists of cited works and the dataset remain accurate and unchanged. Historical drafts and earlier release archives remain preserved.

The AI disclosure now uses the exact requested wording: “An AI assistant was used to audit analysis code, and assist with drafting. The authors are responsible for verifying the analyses, references and final text, and for all publication claims.” The detailed analysis/revision provenance remains available. This editing operation is not a statement that the authors have already completed their final verification or approval.

## Delivery and validation

The main manuscript uses standard template typography and remains within seven pages, with three main figures and two tables. The new figure is additional substantive analysis rather than decorative content. Rebuilt the supplement, removed an unnecessary forced page break after the complete results table, and verified the final layout. Final validation and file hashes are recorded in the machine-readable reports and version 1.3.0 archive manifest. All final pages were rendered for visual inspection. No external submission, publication or author communication occurred.
