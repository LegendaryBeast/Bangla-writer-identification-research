# Handedness for Bangla writer identification

Reproducibility materials for *Handedness for Bangla writer identification: Oracle gains and predicted-attribute utility*.

The revised analysis evaluates 42 BanglaHand writers in 120 closed-set settings. Identification normalization is fitted only to the selected gallery reference paragraphs. The queried writer is excluded from query-handedness training and calibration. Top-1 outcomes give writers equal weight. Correct query and gallery handedness improves ranking; the tested predicted-query fusion yields no reliably demonstrated improvement, while hard filtering causes substantial losses. The filtering product is an independence approximation, with an exact conditional identity and empirical residual checks provided.

The revision is retrospective. Version 1.6.0 provides the public scientific release at https://github.com/LegendaryBeast/Bangla-writer-identification-research/releases/tag/v1.6.0, with a directly linked ZIP, complete manifests, checkpoint references and saved-output reproduction commands. Raw photographs, private editorial materials and the unpublished main manuscript are excluded from that public release. The manuscript adds explicit reuse/new-work distinctions and a cautious random-label split diagram. Version 1.5.0 makes the final methodological framing explicit in the abstract, introduction and discussion, retains the separate supplement, and adds a graphical abstract, cover letter, editable unsigned authorship form and audited upload bundle. No additional experiment or dataset was introduced. The final materials and upload mapping are in `submission/prl_final/README.md`. Version 1.4.0 clarified evaluation versus deployment, makes the evaluative contribution and benchmark limits explicit, promotes the exploratory both-sided result and its information restrictions, improves figure readability, and adds a verified margin/rescue/damage diagnostic using the same dataset. Version 1.3.0 added a main-text representation-level utility figure with paired conditional intervals and explicitly compares the scope with the companion SSRN preprint (doi:10.2139/ssrn.7662278), which uses the same cohort. The title, authorship and disclosure have been updated. Version 1.2.0 added properly nested fusion-weight selection, temperature loss profiles and multiplier sensitivity, calibration-partition variability, and fixed-gallery writer-influence diagnostics. It also includes the known random-partition benchmark, probability calibration, direct control comparisons, corrected simulations and conditional writer-bootstrap uncertainty. It does not establish a universal accuracy threshold or validate forensic attribution.

## Files

| Path | Content |
|---|---|
| `paper/wid_manuscript.pdf` and `.tex` | Revised main manuscript |
| `paper/wid_supplement.pdf` and `.tex` | Self-contained methods, controls and complete numerical grid |
| `paper/figures_revision/` | Generated revised tables, numeric macros and vector figures |
| `results/wid_revision_2026-11-08/` | Authoritative fixed-rule results and provenance |
| `results/wid_fusion_robustness_2026-11-08/` | Nested learned-weight comparison and within-cohort diagnostics |
| `results/wid_handedness_2026-10-29/` | Preserved historical outputs and feature inputs |
| `code/revised_protocol.py` | Reference-only scoring and writer-balanced estimation |
| `code/run_revised_identification.py` | Revised 120-setting identification analysis |
| `code/run_revision_checks.py` | Filtering identity, known partitions, calibration, paired controls |
| `code/summarize_revision_checks.py` | Conditional uncertainty for approximation diagnostics |
| `code/run_revision_simulations.py` | Corrected Gaussian endpoints and reduced-gallery sensitivity |
| `code/run_nested_fusion.py` | Outer writer exclusions, fresh nested development probabilities, weight selection |
| `code/run_robustness_diagnostics.py` | Calibration-partition, temperature and evaluation-writer diagnostics |
| `code/make_revision_paper.py` | Fixed-rule numeric macros, tables and figures |
| `code/make_fusion_robustness_paper.py` | Additional learned-weight and robustness assets |
| `code/make_representation_utility.py` | All-writer representation-level contrasts, paired intervals and main-text figure |
| `code/run_margin_diagnostic.py` | Exact fusion decision reconstruction and rescued/damaged match diagnostics |
| `code/validate_fusion_robustness.py` | Nested-exclusion and additional numerical checks |
| `code/validate_revision.py` | Data/output integrity and scientific invariants |
| `code/build_revision_package.py` | Reproducibility and LaTeX-source archives with SHA-256 manifests |
| `code/prepare_prl_submission.py` | Data-derived graphical abstract, cover letter and unsigned editable authorship form |
| `code/finalize_prl_submission.py` | Explicit numerical cross-checks, upload mapping and verified final submission bundle |
| `paper/notes/REVISION_RECORD_2026-11-08.md` | Initial review response and validation record |
| `paper/notes/FUSION_ROBUSTNESS_REVISION_2026-11-08.md` | Version 1.2.0 additions and results |
| `paper/notes/COMPANION_AND_UTILITY_REVISION_2026-11-08.md` | Version 1.3.0 comparison, authorship/disclosure changes and figure |
| `paper/reviews/PRL_TARGETED_RESPONSE_2026-11-08.md` | Version 1.4.0 point-by-point response to the supplied review |

The older scripts and figure directory reproduce the historical analysis, not the revised article. The separate IEEE blind draft and existing conference submission are outside this revision.

## Dataset and frozen features

BanglaHand is available under CC BY 4.0 from [Mendeley Data](https://doi.org/10.17632/n4sjstdzfy.1). Download images separately into `dataset/`, preserving `metadata/images.csv` and the handedness image folders. Images are not redistributed in the code archive.

The supplied `results/wid_handedness_2026-10-29/features/unit_features.npz` and `units.csv` allow reproduction without neural feature extraction. The matching held-writer-out attribute predictions are in `handedness_lowo.csv.gz`. The archive includes these inputs and their hashes. To regenerate them from images, use the historical `wid_features.py` and `run_wid_handedness.py`; the latter's ID outputs remain historical. Exact extraction settings and checkpoint names appear in the supplement. The feature and prediction inputs used in the revised results are supplied, rather than relying on an inaccessible companion manuscript.

## Reproduce the revised results

Run from the repository root with Python 3.13.9:

```bash
python -m pip install -r requirements-frozen.txt
python code/run_revised_identification.py
python code/run_revision_checks.py --both-controls --jobs 6
python code/summarize_revision_checks.py
python code/summarize_revision_checks.py --own-only
python code/run_revision_simulations.py
python code/run_nested_fusion.py --jobs 3
python code/run_robustness_diagnostics.py --jobs 3 --nested results/wid_fusion_robustness_2026-11-08/nested_writer_outcomes.csv
python code/make_revision_paper.py
python code/make_fusion_robustness_paper.py
python code/make_representation_utility.py
python code/run_margin_diagnostic.py
python code/validate_fusion_robustness.py
python code/validate_revision.py
```

Fitted development probabilities and calibration-partition scores are cached. The nested weight comparison regenerates development probabilities inside every outer writer exclusion; it never selects the weight from globally reused leave-one-writer-out probabilities. It retains all 42 gallery candidates and known gallery hands as permitted metadata. Temperature is held at the original ID-only value; the weight minimizes writer-balanced development NLL. The primary query predictions and all comparators use identical held-out inputs.

Calibration seeds are correlated partition replicates. Evaluation-writer omission changes averaging only, retains half the weight for each hand, and leaves the gallery/models fixed. Neither is independent-dataset validation, and no difficult writer is excluded. The additional analysis uses the same 10,000-draw conditional writer bootstrap, and Holm correction covers the eight learned-rule contrasts (four settings, two comparators).

Random-label model fitting is the slowest original analysis; its evidence is also cached. Full runs use 9,999 sign flips, 10,000 bootstrap draws, 2,000 known partitions, 20 random-label predictors, 100 random directions, 200 Gaussian draws and 1,000 prevalence draws. Provenance files record seeds, input hashes and fitting scopes. Intervals condition on the observed gallery and fitted predictions; they do not resample the entire fitting procedure or a new population.

For feature re-extraction, install `requirements.txt` (including torch/torchvision), download the dataset, then run:

```bash
python code/wid_features.py --dataset dataset --output results/wid_handedness_2026-10-29/features
python code/run_wid_handedness.py --features results/wid_handedness_2026-10-29/features --output results/wid_handedness_2026-10-29
```

Use a separate results folder if preserving the supplied historical inputs. Other accelerator/library versions can change the last digits of embeddings. Reproducing the revised tables uses the supplied frozen features.

Build the two PDFs with a TeX distribution providing `elsarticle`:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error wid_manuscript.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error wid_supplement.tex
```

Archive construction runs from the repository root:

```bash
python code/build_revision_package.py
```

Each archive contains a manifest of hashes. No external publication or DOI registration is performed by these scripts.

## Licences

Code: MIT (`LICENSE`). Numerical results and frozen feature matrices: CC BY 4.0 (`RESULTS_LICENSE.md`). Dataset photographs: the separately deposited dataset's CC BY 4.0 terms. The original code/results archive is preserved; its former DOI placeholder is not a published archive identifier.

## Final PRL submission materials

After rebuilding and validating the PDFs, run `python code/prepare_prl_submission.py`, `python code/finalize_prl_submission.py`, and `python code/build_revision_package.py`. The preparation script reads manuscript-specific history from `submission/prl_final/source/Publication_status.json`; it never copies the previous manuscript's signature or records unconfirmed author approval. The finalizer checks current PDF hashes and numerical consistency before packaging. It does not upload anything. Graphical-abstract PDF, TIFF, PNG and editable SVG alternatives are supplied; the vector PDF is the preferred separate upload. Only author verification, final coauthor approval and signing/date entry remain outside the technical preparation.
