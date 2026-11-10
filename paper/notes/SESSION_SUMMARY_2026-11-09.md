# Session summary: 9 November 2026

The work covered the IET Biometrics manuscript *Does predicted handedness improve writer identification? Oracle gains versus achieved utility in Bangla and English handwriting* (formerly *Handedness for Bangla writer identification: Oracle gains and predicted-attribute utility*), on branch `wid-handedness`.

Commits made in this session:
- `fd55f38`: manuscript revision, P1 analyses, submission package.
- `015d757`: SIFT-VLAD matcher, gender attribute, expanded literature.

Nothing was pushed.

---

## 1. Simulated peer review

- **Five-reviewer review.** Simulated panel: editor, methodology, domain, forensic perspective and devil's advocate.
- **Report:** [`SIMULATED_REVIEW_2026-11-09.md`](SIMULATED_REVIEW_2026-11-09.md).
- **Initial rating:** about 65/100, Major Revision.
- **Main problems found:**
  - the null results were presented as "reproduced" without any analysis of their precision;
  - the error-clustering mechanism was buried in the supplement;
  - the CSAFE adjusted-HC-183 sensitivity model was below chance (34%);
  - the balanced galleries were not justified;
  - the novelty was overstated and the literature thin.

## 2. P1 revision analyses (`results/p1_revision_2026-11-09/`)

| Analysis | Script | Main result |
|---|---|---|
| Precision and equivalence (TOST via 90% CI, smallest margin Δ*, MDE) | `code/run_p1_precision_mechanism.py` | BanglaHand nested fusion is within ±0.96 points of identity-only across 120 settings and recovers at most 11% of the 6.1-point oracle gain. CSAFE resolves effects only beyond about ±3 points (10 rescued and 19 damaged decisions). |
| Random-partition gap decomposition | same | Known hand falls 0.78 points short of a random partition. The case of a single outranking impostor contributes −0.65 of that. Sole outranking impostors have the opposite hand in 34% (ResNet-18) and 37% (ViT-B/16) of cases, against 51% at random. |
| CSAFE adjusted-model bug | `code/run_p1_csafe_adjusted_fix.py`, `code/build_p1_csafe_corrected.py` | Saturated grayscale covariates and a near-constant HC-183 column caused rounding blow-ups: out-of-fold scores of about 1e15 flattened the calibration slope to zero. A numerically stable implementation of the same projection raises accuracy from 34% to 68.8 / 61.5 / 64.6%. Primary results are unchanged and BanglaHand is unaffected (`code/check_p1_bangla_adjusted.py`). |
| Query-hand split | `code/run_p1_precision_mechanism.py` | No consistent gain for either hand. On your instruction, no extra CSAFE data was downloaded and the balanced 48/48 design was kept. |

The corrected CSAFE tables are in `csafe_corrected/`. Only the adjusted-model rows were replaced, in place, so every archived primary value and interval is unchanged.

## 3. Manuscript revision (`paper/iet_biometrics/`)

- **Front matter:**
  - new question-style title;
  - the abstract leads with the findings;
  - the introduction lists the contributions explicitly.
- **Methods:**
  - descriptor details are condensed into a representations table;
  - the decomposition equation (Eq. 6) is added;
  - a post hoc precision paragraph is added;
  - the CSAFE correction is disclosed.
- **Results:**
  - new subsections on precision and on why the oracle gain is not handedness-specific;
  - the calibration table now covers both datasets;
  - the CSAFE text reports direction plus precision instead of "reproduce".
- **Moved to the supplement:** the former Table 4 (designated settings), Table 6 (calibration seeds), Table 8 (coverage) and Figure 5 (robustness).
- **Discussion, limitations and conclusions:** rewritten, with the scattered caveats consolidated.
- **Supplement:**
  - corrected CSAFE tables and figure;
  - new sections S14 (precision), S15 (decomposition), S16 (numerical correction), S17 (coverage table) and S18 (extensions).
- **Cover letter:** rewritten for the new title and findings.
- **Backup:** the pre-revision sources are in `paper/notes/pre_revision_2026-11-09/`.

## 4. Figures (`code/make_p1_figures.py`)

- **New:**
  - Figure 1, the evaluation protocol, restyled on your request in a restrained journal style;
  - Figure 3, a precision forest plot;
  - Figure 5, the mechanism figure;
  - Figure 9, the extensions figure.
- **Restyled:**
  - Figure 6, the label controls;
  - Figure 8, CSAFE accuracy and rescued/damaged decisions.
- **Shared style:** the same Okabe–Ito method colours throughout, direct labels and thin rules.

## 5. Extensions (`results/p2_extensions_2026-11-09/`)

The protocol was written to `protocol.json` before any outcome was computed.

- **SIFT-VLAD matcher** (`code/extract_csafe_sift_vlad.py`):
  - pipeline: RootSIFT, PCA to 64, a 100-word VLAD, PCA whitening to 128;
  - the codebook comes from unlabeled enrolment pages only;
  - it reaches 92.7% / 95.8% / 90.6% top-1;
  - known handedness adds nothing in session 2, and fusion changes at most one decision.
- **Gender as a second attribute** (`code/run_p2_csafe_extensions.py`):
  - the predictor is weak: balanced accuracy 55–58%;
  - known gender is no better than a random partition with the same group sizes;
  - errors cluster within gender: 29.5% opposite-gender, against 46.2% at random;
  - none of the 16 Holm-corrected primary contrasts shows a gain.
- **Regression check:** the refitted handedness model reproduces the archived logits exactly.

## 6. Literature

- **Count:** expanded from 26 to 52 references; each new one was checked against a bibliographic record.
- **Soft biometrics:** Jain 2004 (integrating soft traits), Dantcheva 2011, Reid 2014, Nixon 2015 and Park & Jain 2010.
- **Fusion:** Kittler 1998, Ross & Jain 2003, Jain 2005 (normalization) and Nandakumar 2008.
- **Writer identification:** Srihari 2002, Lowe 2004, Arandjelović 2012, Jégou 2010, Christlein 2017 (two papers), Fiel 2015, Louloudis 2013, Chanda 2010 and Adak 2019.
- **Gender from handwriting:** Djeddi 2015 and Illouz 2018.
- **Statistics and calibration:** Lakens 2017, Holm 1979, Varma & Simon 2006, Gneiting & Raftery 2007 and Guo 2017.
- **Table 1:** extended accordingly.

## 7. Submission package (`submission/iet_biometrics/`)

- **Contents:** rebuilt PDFs, Word files, cover letter, figures, CSVs, sources, README, validation records and checksums.
- **New archive:** `reproducibility/wid-revision-analyses-2026-11-09.zip`.
- **Previous package:** preserved in `previous/2026-11-09-pre-revision/`.
- **Build scripts:** `code/finalize_p1_submission.py` assembles the package; `code/verify_iet_build.py` rebuilds both PDFs from the zipped sources and matches every page.

**Final state:**
- main PDF: 14 pages, 9 figures, 6 tables, 52 references, 282-word abstract;
- supplement: 23 pages;
- all 158 checksums verify.

**Repository housekeeping:** `.gitignore` now excludes rebuildable submission zips, the pre-revision snapshot and revision model caches.

## 7b. Final strengthening round (`results/p3_sweep_2026-11-09/`, `code/run_p3_sweep_rank_counts.py`)

- **Accuracy sweep (new Figure 4, Table S29).** Simulated attribute evidence that is calibrated and independent of the matcher (a best case for fusion) was fused at accuracies from 50% to 100%.
  - BanglaHand needs a median 69% accuracy (range 54–91%) for +1 point, and 85% (80–91%) to recover half the oracle gain.
  - At the achieved 65–67%, even this evidence gains only −0.6 to +3.0 points.
  - The real predictor falls below the simulated curve in 22 of 24 settings, by 1.2 points on average.
  - CSAFE needs 67–95% for +1 point, and SIFT-VLAD gains nothing at any accuracy.
- **Rank-based endpoints (Table S30).**
  - BanglaHand fixed fusion changes top-5 accuracy by +0.03 points (−1.45 to +1.34), and nested fusion changes MRR by +0.001.
  - On CSAFE, 5 of 80 fusion contrasts have unadjusted intervals that exclude zero (4 negative, 1 positive), consistent with chance.
- **Sample sizes (Table S31, Figure 5c).** BanglaHand clustering estimates rest on 437–1,398 query units per matcher, from all 42 writers. Each CSAFE estimate rests on 4–22 decisions.
- **Softer claims.** Contributions 1–3, the abstract, the claim about stronger matchers and the conclusion now say "in these cohorts", "consistent with" and "bounded null".
- **Moved.** The representation-utility figure is now Supporting Figure S8.
- **Not done.** A second Bangla cohort for handedness is not available. BanglaWriting (260 writers, gender only, one page each) was judged not worth the download.
- **Final state.** The main PDF is 14 pages, with 9 figures, 6 tables, 52 references and a 284-word abstract. The supplement is 26 pages with 31 tables. All 164 checksums verify and the independent rebuild passes.

## 8. Still to do before upload

- [ ] Open the DOCX files in Word and check the layout. They passed structural and value checks but could not be rendered here.
- [ ] Get coauthor approval for the revised title.
- [ ] Add a citable institutional policy for the ethics waiver, if available.
- [ ] Enter the ORCID in the submission portal.
- [ ] Push the branch when ready.
