# Literature and layout revision

Revised the current IET Biometrics manuscript on 9 November 2026 in response to the author's request. The title and methodological contribution remain unchanged. This revision runs no new experiments and adds no new datasets.

## Content

- Rebuilt Related Work around soft-biometric utility, handedness prediction, and writer-recognition/evaluation protocols.
- Added Table 1, summarizing representative literature by data, representation, endpoint and relevance. Original study accuracies are attributed to their original tasks rather than ranked against the present study.
- Added a compact organization paragraph at the end of the Introduction.
- Added Discussion Table 8, comparing methodological coverage across attribute prediction, writer separation, identity endpoints, oracle/predicted comparisons, random-partition controls, nested fusion, error diagnostics, data scope and separate capture sessions.
- Expanded the Discussion to explain how the controls support a practical evaluation sequence for auxiliary biometric information.
- Added the verified FragNet reference. The main paper now has 26 references, eight tables, seven figures and six numbered equations.

The comparison is representative rather than an exhaustive survey. “NR” means the specified evaluation was not reported in the inspected source; “U” means the detailed protocol was not verified from the accessible record. Neither is evidence of a defective protocol or identity leakage. Handedness accuracy, writer verification and closed-set identification utility are different endpoints.

## Primary-source checks

Sources were inspected on 9 November 2026:

- Al-Maadeed et al., [publisher full text](https://link.springer.com/article/10.1186/s13640-015-0097-y): 121-writer QUWI subset, Arabic/English, handcrafted features, fuzzy reduction and nearest-neighbour classification. Writer-disjoint details were not established in this audit.
- Morera et al., [publisher full text](https://onlinelibrary.wiley.com/doi/10.1155/2018/3891624): explicitly separate training/test individuals; CNN word predictions and line voting. IAM/KHATT handedness overall accuracies are 90.70%/70.91%; respective left-handed recalls are 63.40%/38.64%.
- Rahmanian and Shayegan, [publisher record](https://link.springer.com/article/10.1007/s11042-020-10170-7): DenseNet201, InceptionV3 and Xception; reported KHATT handedness accuracy 99.14%. The accessible record does not establish the detailed split; no inference about writer overlap is made.
- Dantcheva et al., [author-hosted survey](https://www.cse.msu.edu/~rossarun/pubs/DantchevaEliaRossSoftBiometrics_TIFS2016.pdf): distinctions between auxiliary attributes, indexing and biometric recognition.
- He and Schomaker, [institutional record](https://research.rug.nl/en/publications/fragnet-writer-identification-using-deep-fragment-networks/): FragNet, IEEE TIFS 15 (2020), 3013–3022, doi:10.1109/TIFS.2020.2981236; feature-pyramid/fragment pathways and four benchmarks. Capture-session details are marked unverified.
- Paul et al., [publisher full text](https://onlinelibrary.wiley.com/doi/full/10.4218/etrij.2023-0188): Bangla writer verification, 100 writers, feature selection and reported mean accuracy 94.54%.
- Cawley and Talbot, [JMLR primary article](https://jmlr.org/papers/v11/cawley10a.html): model-selection bias and separation of selection from evaluation.
- [Official IET Biometrics author instructions](https://ietresearch.onlinelibrary.wiley.com/page/journal/ietbm/homepage/author-guidelines): initial reference-style flexibility, numbered citations, editable submission and table presentation.

## Formatting

The large empty Discussion column was caused by the template's older cuted output routines interacting with the current LaTeX kernel. No strip environment is needed in this manuscript. The preamble now preserves and restores the current kernel's ordinary two-column output routines without editing USG.cls.

Heading skips were reduced, body leading adjusted from 11.5 to 11.2 points while retaining the 9-point font, and wide-float placement corrected. The calibration table is allowed to defer to avoid overfilling a stack of floats. A final bibliography break before reference 14 balances the last two columns. The balance package is not used. The seven figure assets are unchanged.

## Verification

- Main typeset PDF: 13 pages; all pages rendered and visually inspected.
- Editable main Word document: 17 pages; all pages rendered and visually inspected, with eight editable tables, seven images and native Word mathematics.
- All six pre-existing scientific table blocks are identical to the pre-revision source, apart from float placement; all seven pre-existing figure blocks are unchanged apart from float placement.
- Abstract, metadata, supporting-information source and PDF are unchanged by this revision. Original PRL files match the retained conversion-input hashes.
- Main LaTeX log has no overfull boxes, unresolved references/citations, missing-character warnings or compilation errors.
- Word verification checks 1,452 decimal table values across the main manuscript and supporting information.
- An isolated build from the submission ZIP is recorded separately in Independent_build_validation.json; archive CRC and SHA256SUMS checks are performed by the finalizer.

These are consistency and presentation checks, not a guarantee of journal acceptance or a substitute for author approval.
