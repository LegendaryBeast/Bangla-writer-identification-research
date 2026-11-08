# Handedness for Bangla writer identification

Research code and reproducibility materials for **Handedness for Bangla writer identification: Oracle gains and predicted-attribute utility**.

**Version 1.6.0** | [Release and complete ZIP](https://github.com/LegendaryBeast/Bangla-writer-identification-research/releases/tag/v1.6.0) | [Direct archive](https://github.com/LegendaryBeast/Bangla-writer-identification-research/releases/download/v1.6.0/wid-reproducibility-v1.6.0.zip) | [Dataset](DATASET.md)

The controlled study separates the benefit of correct metadata from the utility achieved using predicted handedness. Across 42 writers, six representations and 120 settings, known handedness changes top-1 accuracy by +6.1 percentage points; an expected known random partition gives +6.9. Fixed fusion changes it by -0.6, properly nested learned weighting by -0.08, and hard filtering by -14.3. These are descriptive across-setting averages on one observed cohort, not independent-population validation or a universal attribute-accuracy threshold.

## Quick start from supplied numerical inputs

```bash
python -m pip install -r requirements-frozen.txt
python code/reproduce.py --verify
python code/reproduce.py --figures
```

Verification and figure generation require no photographs, network access, model downloads or model refitting. The frozen feature matrices and exact calibrated scores are included. All numerical tables and the three main figures plus five supplementary figures have generation commands in [ARTIFACT_MAP.md](docs/ARTIFACT_MAP.md).

```bash
# Recompute the revised analysis from frozen features, using verified caches:
python code/reproduce.py --full --jobs 3
# Regenerate development probabilities and calibration partitions instead:
python code/reproduce.py --full --jobs 3 --refit-development
```

See [REPRODUCE.md](docs/REPRODUCE.md) for individual commands, optional feature extraction and runtime scope. Python 3.13.9 and pinned package versions produced the supplied results. Full fitting is substantially slower than the verification/figure modes.

## Contents

| Path | Role |
|---|---|
| `code/` | Complete scientific fitting, scoring, controls, simulation, plotting and numerical-verification code |
| `manifests/settings.csv` | All 120 settings |
| `manifests/reference_configurations.csv` | All 1,152 setting/reference-configuration rows |
| `manifests/*labels*` | Exact seeded random-label assignments |
| `manifests/checkpoints.json` | Exact pretrained weight identifiers and official URLs |
| `results/wid_handedness_2026-10-29/` | Required frozen features, valid-unit manifest, calibrated attribute inputs and enrolment-label control |
| `results/wid_revision_2026-11-08/` | Revised fixed-rule numerical results and controls |
| `results/wid_fusion_robustness_2026-11-08/` | Nested folds/weights, uncertainty, sensitivity and margin diagnostics |
| `paper/wid_supplement.pdf` and `.tex` | Self-contained supplementary methods and results |
| `paper/figures_revision/` | Every generated table, numeric macro and vector figure |
| `SHA256SUMS.txt` | File-level archive integrity manifest |

Raw photographs and the original dataset ZIP are **not included**. Obtain them through [DATASET.md](DATASET.md). Frozen numerical features and pseudonymous analysis manifests are derived inputs included for reproduction. The main manuscript and journal-administration files are distributed separately through the submission process.

## Interpretation and related work

The companion handedness-classification preprint ([doi:10.2139/ssrn.7662278](https://ssrn.com/abstract=7662278)) uses the same cohort, preprocessing and overlapping representations. This study measures downstream identity-ranking utility under explicit oracle/predicted information, nested weighting and dependence controls. It is an extension using shared data, not independent replication. See [METHODS.md](docs/METHODS.md).

The random-label control shows how writer overlap changes the prediction target. It does not establish a particular leakage mechanism in the primary writer-held-out experiment or a causal fraction of handedness accuracy. Both-sided agreement is a separate enrolled-cohort diagnostic; only the 96 own-text settings support the reported +1.08-point gain.

Code: MIT. Numerical results and frozen features: CC BY 4.0. Dataset photographs: the separately deposited dataset's CC BY 4.0 terms. Use `CITATION.cff` and cite the dataset independently. Scientific scope, input hashes and portable provenance are described in [PROVENANCE.md](docs/PROVENANCE.md).
