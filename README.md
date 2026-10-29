# Soft biometrics for writer identification: a utility law from Bangla handedness

Code and results of the paper *Soft Biometrics for Writer Identification: A Utility Law from Bangla Handedness*.

The study identifies the 42 writers of the BanglaHand dataset from Bangla paragraphs and horizontal strips of them, with six
representations in 120 settings, and fuses handedness evidence from a model that never saw the queried writer. Known
handedness raises top-1 accuracy by 6.0 points; handedness predicted from the handwriting (65.5% accurate) does not help.
A utility law, top-1 with filtering = handedness accuracy x top-1 with known handedness, predicts the filter results
(r = 0.996), and a random-label control shows that attribute accuracies obtained on enrolled writers mainly measure
writer recognition.

## Contents

| Path | Content |
|---|---|
| `code/` | All code used in the paper (see below) |
| `results/wid_handedness_2026-10-29/` | All outputs of the reported run; see its `README.md` |
| `dataset/` | BanglaHand images and metadata (input; download separately, see below) |
| `paper/` | Manuscript sources (Elsevier and IEEE versions), bibliography, figures and tables |
| `submission/` | Anonymised conference submission (PDF and LaTeX source) |
| `zenodo/` | Upload-ready archive of code and results, with its metadata |

## Data

The study uses the BanglaHand dataset (336 paragraph photographs, 42 writers, 21 left-handed), available at Mendeley Data,
doi:[10.17632/n4sjstdzfy.1](https://doi.org/10.17632/n4sjstdzfy.1), under a CC BY 4.0 licence. Download it and place it so
that `dataset/metadata/images.csv` and the `Left-Handed/` and `Right-Handed/` folders sit in `dataset/`. The dataset is
not redistributed here.

## Code

| Script | Purpose |
|---|---|
| `code/hc183.py` | HC-183 ink mask and gradient descriptor |
| `code/stroke_features.py` | Writing-only STROKE features and binary ink tiles |
| `code/wid_features.py` | Features of paragraphs and of 2, 4 and 8 horizontal strips (six representations) |
| `code/run_wid_handedness.py` | Leave-one-writer-out handedness evidence, identification in 120 settings, fusion, statistics, break-even simulation, prevalence |
| `code/run_both_control.py` | Random-label and random-direction controls for the scenario with both hands predicted |
| `code/run_wid_extras.py` | Leakage control (enrolment-trained vs writer-independent, random labels), confidence gating, filter law |
| `code/make_wid_paper.py` | All figures, tables and the macros file with every number quoted in the paper |

## Reproduction

```bash
python -m pip install -r requirements.txt
python code/wid_features.py --dataset dataset --output results/wid_handedness_2026-10-29/features
python code/run_wid_handedness.py --features results/wid_handedness_2026-10-29/features --output results/wid_handedness_2026-10-29
python code/run_both_control.py --features results/wid_handedness_2026-10-29/features --run results/wid_handedness_2026-10-29
python code/run_wid_extras.py --features results/wid_handedness_2026-10-29/features --run results/wid_handedness_2026-10-29
python code/make_wid_paper.py
```

Run times of the reported run on an Apple M-series laptop: features 43 min (MPS), main analysis 9 min, controls 23 min,
extras 12 min. `--quick` runs every analysis with few permutations and draws as a functional test. All seeds are fixed;
the classical analyses reproduce exactly for the same library versions, while CNN and ViT embeddings computed on other
hardware can differ in the last digits.

The figures and tables in `paper/figures/` are written by `code/make_wid_paper.py`; every number in the manuscripts is a
macro from `paper/figures/wid_macros.tex`.

## Licence

Code: MIT (`LICENSE`). Results: CC BY 4.0. BanglaHand dataset: CC BY 4.0 (Mendeley Data).
