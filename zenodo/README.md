# Zenodo upload: code and results

File to upload: `wid-handedness-code-results-v1.0.0.zip` (122 MB, 43 files, `SHA256SUMS.txt` inside).
It holds `README.md`, `LICENSE`, `CITATION.cff`, `requirements.txt`, `code/` and `results/wid_handedness_2026-10-29/`
(including the 60 MB unit features and 65 MB per-query ranks). The BanglaHand images are not included; they are cited
by their own DOI.

## Double-blind review

The record names the authors. While the paper is under double-blind review:
1. Create the upload as a **draft** and use **Reserve DOI**; do not publish yet.
2. Keep "will be released upon acceptance" in the submitted paper.
3. After acceptance, publish the record and put the DOI in the camera-ready paper (`paper/wid_manuscript.tex`
   already has the sentence with `doi:10.5281/zenodo.XXXXXXX`; replace the X's).

## Metadata for the Zenodo form

- **Resource type:** Software
- **Title:** Soft Biometrics for Writer Identification: A Utility Law from Bangla Handedness (code and results)
- **Version:** 1.0.0
- **Publication date:** the day you publish
- **Creators:** Prappo, Tanzim Hasan; Hassan, Sayeed; Ali, Sayem; Dhar, Sukhendra Nath; Rahman, M. Shahidur
  (affiliation for all: Department of Computer Science and Engineering, Shahjalal University of Science and Technology)
- **Licence:** MIT License (the README states that the results are CC BY 4.0)
- **Keywords:** writer identification; soft biometrics; handedness; Bangla handwriting; score fusion; data leakage
- **Related works:**
  - *Requires / Is supplemented by* — Dataset — `10.17632/n4sjstdzfy.1` (BanglaHand, Mendeley Data)
  - *Is supplement to* — the paper's DOI, once published
- **Description:**

> Code and results for the paper "Soft Biometrics for Writer Identification: A Utility Law from Bangla Handedness".
> The study identifies the 42 writers of the BanglaHand dataset (doi:10.17632/n4sjstdzfy.1) from Bangla paragraphs and
> strips of them with six representations in 120 settings, and fuses handedness evidence from a model that never saw the
> queried writer. Known handedness raises top-1 accuracy by 6.0 points; handedness predicted from the handwriting does
> not help. A utility law (top-1 with filtering = handedness accuracy × top-1 with known handedness) predicts the filter
> results (r = 0.996), and a random-label control shows that attribute accuracies obtained on enrolled writers mainly
> measure writer recognition. The archive contains all analysis code (Python), every result file of the reported run,
> and instructions to reproduce all figures, tables and numbers of the paper.

## Rebuilding the archive

From the repository root, after any change to `code/` or `results/`:

```bash
NAME=wid-handedness-code-results-v1.0.0; STAGE=$(mktemp -d)/$NAME; mkdir -p $STAGE
cp -R README.md LICENSE CITATION.cff requirements.txt code results $STAGE/
find $STAGE -name __pycache__ -prune -exec rm -rf {} +
(cd $STAGE && find . -type f | sort | xargs shasum -a 256 > SHA256SUMS.txt)
(cd $(dirname $STAGE) && zip -qr -X "$OLDPWD/zenodo/$NAME.zip" $NAME)
```
