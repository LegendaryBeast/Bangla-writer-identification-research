#!/usr/bin/env python3
"""Stage the scientific reproducibility release from an explicit file allowlist."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
VERSION='1.6.0'
REPO='https://github.com/LegendaryBeast/Bangla-writer-identification-research'
SCIENCE=['hc183.py','stroke_features.py','wid_features.py','run_wid_handedness.py',
         'run_wid_extras.py','run_both_control.py','revised_protocol.py','run_revised_identification.py',
         'run_revision_checks.py','summarize_revision_checks.py','run_revision_simulations.py',
         'run_nested_fusion.py','run_robustness_diagnostics.py','make_revision_paper.py',
         'make_fusion_robustness_paper.py','make_representation_utility.py','run_margin_diagnostic.py',
         'validate_fusion_robustness.py','verify_results.py','reproduce.py','prepare_cached_inputs.py']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def portable(value):
    if isinstance(value,dict):return {portable(k):portable(v) for k,v in value.items()}
    if isinstance(value,list):return [portable(x) for x in value]
    if isinstance(value,str):return value.replace(str(ROOT)+'/', '').replace(str(ROOT),'.')
    return value


def stage(target):
    def cp(relative):
        source=ROOT/relative;dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if source.suffix=='.json':
            data=portable(json.loads(source.read_text()))
            dest.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
        else:shutil.copy2(source,dest)
    for name in SCIENCE:cp(Path('code')/name)
    for name in ['LICENSE','RESULTS_LICENSE.md','CITATION.cff','requirements.txt','requirements-frozen.txt']:cp(Path(name))
    c=target/'CITATION.cff';c.write_text(c.read_text()+f'url: "{REPO}"\n')
    requirements=target/'requirements-frozen.txt'
    requirements.write_text(requirements.read_text()+'pymupdf==1.27.2.3\n')
    for name in ['wid_supplement.tex','wid_supplement.pdf','references.bib','highlights.txt']:cp(Path('paper')/name)
    for p in (ROOT/'paper/figures_revision').iterdir():
        if p.is_file() and p.suffix in {'.tex','.pdf','.png'}:cp(p.relative_to(ROOT))
    historical=Path('results/wid_handedness_2026-10-29')
    for name in ['handedness_lowo.csv.gz','leakage_accuracy.csv','extras_provenance.json','wid_provenance.json']:
        cp(historical/name)
    for name in ['units.csv','unit_features.npz','unit_feature_provenance.json']:cp(historical/'features'/name)
    for folder in ['wid_revision_2026-11-08','wid_fusion_robustness_2026-11-08']:
        for p in (ROOT/'results'/folder).iterdir():
            if p.is_file() and (p.suffix in {'.csv','.npz','.json'} or p.name.endswith('.csv.gz')):
                if p.name=='validation_report.json':continue
                cp(p.relative_to(ROOT))
    # Supplemental descriptor-control explanation; no classifier behavior changes.
    p=target/'code/run_both_control.py';s=p.read_text()
    s=s.replace('If the handedness gain lies within the distribution of these controls, it reflects identity, not handedness.',
        'Similar control performance shows that descriptor agreement can use writer-associated structure.\n'
        'It does not causally attribute the handedness gain or prove leakage in a writer-disjoint primary experiment.')
    p.write_text(s)
    docs(target)
    manifests(target)
    audit_stage(target)


def manifests(target):
    path=target/'manifests';path.mkdir(exist_ok=True)
    S=pd.read_csv(target/'results/wid_revision_2026-11-08/id_summary.csv')
    settings=S[S.scenario=='none'][['rep','query','n','k','n_writers']].sort_values(['rep','query','n','k'])
    assert len(settings)==120 and (settings.n_writers==42).all()
    settings.to_csv(path/'settings.csv',index=False)
    configs=[]
    for r in settings.itertuples(index=False):
        choices=[(q,tuple(i for i in range(1,7) if i!=q)) for q in range(1,7)] if r.query=='shared' else [(None,x) for x in itertools.combinations(range(1,7),r.k)]
        for j,(q,refs) in enumerate(choices):
            assert len(refs)==r.k and (q is None or q not in refs)
            configs.append(dict(rep=r.rep,query=r.query,n=r.n,k=r.k,configuration_id=j,
                                shared_query_paragraph=q,reference_paragraphs='|'.join(map(str,refs)),
                                gallery_writers=42,scaler_fit='selected whole-paragraph references only'))
    assert len(configs)==1152
    pd.DataFrame(configs).to_csv(path/'reference_configurations.csv',index=False)
    import run_wid_extras as extras
    U=pd.read_csv(target/'results/wid_handedness_2026-10-29/features/units.csv')
    writers=sorted(U.writer_id.unique());rows=[]
    for i,labels in enumerate(extras.splits(writers,20,np.random.default_rng(extras.SEED))):
        rows.extend(dict(labeling=i,writer_id=w,pseudo_label=labels[w],generator_seed=extras.SEED) for w in writers)
    pd.DataFrame(rows).to_csv(path/'enrolment_control_random_labels.csv',index=False)
    both=json.loads((target/'results/wid_revision_2026-11-08/checks_both_evidence_provenance.json').read_text())
    (path/'both_descriptor_control_labels.json').write_text(json.dumps(dict(seed=both['seed'],positive_writer_groups=both['pseudo_groups']),indent=2)+'\n')
    checkpoints=[
        dict(representation='ResNet-18; Stroke-CNN',dimension=512,weight_enum='ResNet18_Weights.IMAGENET1K_V1',
             url='https://download.pytorch.org/models/resnet18-f37072fd.pth',sha256_prefix='f37072fd'),
        dict(representation='EfficientNet-B0',dimension=1280,weight_enum='EfficientNet_B0_Weights.IMAGENET1K_V1',
             url='https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth',sha256_prefix='7f5810bc'),
        dict(representation='ViT-B/16',dimension=768,weight_enum='ViT_B_16_Weights.IMAGENET1K_V1',
             url='https://download.pytorch.org/models/vit_b_16-c867db91.pth',sha256_prefix='c867db91')]
    (path/'checkpoints.json').write_text(json.dumps(dict(torch='2.14.0',torchvision='0.29.0',weights=checkpoints),indent=2)+'\n')


def docs(target):
    d=target/'docs';d.mkdir(exist_ok=True)
    files={
    'README.md':f'''# Handedness for Bangla writer identification

Research code and reproducibility materials for **Handedness for Bangla writer identification: Oracle gains and predicted-attribute utility**.

**Version {VERSION}** | [Release and complete ZIP]({REPO}/releases/tag/v{VERSION}) | [Direct archive]({REPO}/releases/download/v{VERSION}/wid-reproducibility-v{VERSION}.zip) | [Dataset](DATASET.md)

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
''',
    'DATASET.md':'''# Dataset (download separately)

**BanglaHand: A Dataset of Handwritten Paragraphs for Writer-Independent Handedness Recognition**, Mendeley Data, version 1, published 25 October 2026.

- Persistent DOI: https://doi.org/10.17632/n4sjstdzfy.1
- Dataset page and download: https://data.mendeley.com/datasets/n4sjstdzfy/1
- License: CC BY 4.0, as specified by the dataset record.
- Collection: 336 paragraph photographs from 42 writers (21 left-handed and 21 right-handed), six shared prompts and two own texts each.

No photographs, dataset archive, EXIF bundle or participant-identifying files are uploaded in this repository or release ZIP. The supplied numerical feature matrices, pseudonymous unit/fold manifests and model scores are derived analysis inputs. They allow all reported numerical tables and figures to be reproduced without downloading the photographs.

For optional extraction, download the original collection and place its contents under `dataset/`, preserving the supplied metadata and image-folder structure. That directory is ignored by Git. The original collection's checksums should be verified using its own manifest. See `docs/REPRODUCE.md` for commands.

Dataset contributors: Tanzim Hasan Prappo, Sayeed Hassan, Sayem Ali, Sukhendra Nath Dhar and M. Shahidur Rahman. Dataset authorship is distinct from the author list of the present article.
''',
    'docs/METHODS.md':'''# Self-contained implementation and interpretation

The complete typeset specification is `paper/wid_supplement.pdf`. Source implementations are included; no companion manuscript is required to reconstruct descriptors or scores.

## Units and fixed validity rules

Each of 336 photographs gives a full paragraph and 2, 4 and 8 strips: 5,040 candidate units. Shared paragraphs are numbered 1–6 and own texts 7–8. Validity requires at least 200 ink pixels in the high-resolution mask, at least 50 in the 768-pixel mask, computable contour/skeleton descriptors and finite features. Exactly 39 strips fail (one quarter strip and 38 eighth strips); no paragraph or writer is removed. `features/units.csv` preserves all candidate IDs, validity, strip boundaries and covariates. Every retained setting includes all 42 writers.

Strips use the 1,536-pixel mask's text rows (ink fraction >0.01, 0.5th/99.5th percentiles). Cuts move to low-ink rows near equal-height boundaries using a Gaussian-smoothed row profile, standard deviation 96/5 pixels. They keep the photograph's full width. Implementation: `code/wid_features.py`.

## Descriptors and preprocessing

- **HC-183**: resize to width 768; mask pixels 17 levels darker than a Gaussian local illumination estimate (sigma 17), with 2x2 opening. A 4x4 grid contributes nine-bin magnitude-weighted unsigned gradient histograms and ink densities (160 values); 18 row/column projection statistics and five connected-component area quantiles complete 183 dimensions. `code/hc183.py`.
- **STROKE**: width 1,536, local-contrast mask with brightness gates, 2x2 opening, components below 12 pixels removed, text-block crop. Contours with ink on the left are resampled every two pixels; legs use twice mean stroke width (minimum four pixels). Direction, hinge, turning, skeleton endpoint/orientation/slant histograms and topology rates give 208 dimensions. No grayscale values enter the descriptor after mask construction. `code/stroke_features.py`, `code/wid_features.py`.
- **Frozen grayscale networks**: resize width 1,568, retain 224x224 tiles with at least 1% ink, repeat gray into RGB, use ImageNet mean (0.485,0.456,0.406) and standard deviation (0.229,0.224,0.225), remove final classification layers and average tile embeddings. Dimensions: ResNet-18 512, EfficientNet-B0 1,280, ViT-B/16 768. The networks remain in evaluation mode without fine-tuning.
- **Stroke-CNN**: the frozen ResNet-18 checkpoint on binary ink tiles after stroke-width normalization. The exact tile preparation is in `wid_features.py`; checkpoint references are in `CHECKPOINTS.md`.

The original feature extraction used Python 3.13.9, torch 2.14.0, torchvision 0.29.0 on MPS. Exact frozen matrices are supplied. Re-extraction with another accelerator/library can change floating-point details.

## Identification and attribute fitting

Fit each identity scaler only to the selected whole-paragraph gallery references; transform and L2-normalize all references/queries, form normalized mean writer templates, then use cosine scores. A shared query excludes its prompt from every gallery writer; own queries use every combination of 1, 2, 3 or 6 shared reference paragraphs. The 120 settings and all reference configurations are explicit manifests.

HC-183 handedness uses a linear SVM (C=0.1) trained on the other 41 writers' valid shared units of the same size. Five-fold writer-group out-of-fold margins supply logistic calibration (C=10,000). The query writer never enters those stages. This is a cross-validation estimate, not an inference algorithm that selects a model using an unknown query identity. A deployed unseen-writer pipeline would freeze one model/calibrator and development-selected fusion parameters before future queries.

ID-only temperature minimizes equal-writer negative log loss over the other query writers using 31 values from 0.001 to 1. Fixed fusion adds log query-hand probability to scaled identity scores, with known gallery hands. Nested fusion selects weight from {0,.1,.2,.3,.5,.7,1,1.5,2,3,5} using development probabilities regenerated inside each outer exclusion. The 840 development-fold records and their inner calibration exclusions are supplied. Hard filtering places the predicted-hand group first.

## Uncertainty and diagnostic scope

Outcomes average configurations within valid query units, then units within writer; settings and writers have equal aggregate weight. Paired intervals use 10,000 hand-stratified writer bootstrap draws; tests use 9,999 sign flips. Shared resampling indices retain cross-setting dependence. Four designated fixed-rule comparisons have Holm correction; the nested-vs-none/fixed contrasts form a separate eight-test Holm family. These retrospective comparisons are not preregistered confirmation or equivalence tests.

Temperature loss profiles and fixed multipliers, four calibration partitions, evaluation-writer omission, class-specific effects, synthetic evidence and reduced-gallery prevalence analyses are included. They retain or reuse this cohort and do not constitute new-dataset or cross-session validation. The complete inference scope and all additional intervals are in the supplement.

With binary groups, filtered success equals joint correct handedness and correct known-hand identification; the product of marginal accuracies requires independence. The margin diagnostic reconstructs every fixed/nested decision and separates rescued from damaged matches.

Random writer-label prediction under enrolment training diagnoses available writer-associated information and a changed learning problem under overlap. It does not prove a leakage mechanism in the writer-disjoint primary pipeline. The schematic in supplementary Figure S2 makes the split distinction explicit. The observed 52.4–76.2% random-label range is not interpreted as a causal identity contribution or a calibrated chance test.
''',
    'docs/CHECKPOINTS.md':'''# Exact model checkpoint references

No network is trained or fine-tuned on the study writers. Pretrained weights are not redistributed. Numerical reproduction uses the supplied frozen features and does not download weights.

| Representation | torchvision 0.29.0 enum | Official checkpoint | Filename SHA-256 prefix |
|---|---|---|---|
| ResNet-18 and binary Stroke-CNN | `ResNet18_Weights.IMAGENET1K_V1` | https://download.pytorch.org/models/resnet18-f37072fd.pth | `f37072fd` |
| EfficientNet-B0 | `EfficientNet_B0_Weights.IMAGENET1K_V1` | https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth | `7f5810bc` |
| ViT-B/16 | `ViT_B_16_Weights.IMAGENET1K_V1` | https://download.pytorch.org/models/vit_b_16-c867db91.pth | `c867db91` |

These URLs and enum names were verified against the installed torchvision 0.29.0 package. The suffixes above are the official filename hash prefixes, not claims of full SHA-256 digests. The originally generated feature matrices are independently covered by the release's complete SHA-256 manifest. `manifests/checkpoints.json` is machine readable.

Feature extraction uses torch 2.14.0 and the custom image preparation in `wid_features.py`, not an assumed default enum transform. Official architecture documentation: https://docs.pytorch.org/vision/stable/models.html
''',
    'docs/REPRODUCE.md':'''# Reproduce from the released numerical inputs

Use Python 3.13.9 and run commands from the repository/archive root. The release ZIP and Git tag include the same scientific files. Verify `SHA256SUMS.txt` before regenerating outputs; regenerated PDF timestamps may differ.

```bash
python -m pip install -r requirements-frozen.txt
python code/reproduce.py --verify
python code/reproduce.py --figures
```

`--verify` independently checks feature/unit alignment, all writers/settings, exact filtering identities, paired effect reconstructions, nested fold exclusions, selected weights, controls, sensitivity summaries and margin decompositions. It fits no model. `--figures` regenerates every table, numeric macro and figure and then runs both scientific verifiers. The primary and nested manuscript grids remain separate from historical classification inputs.

## Full revised analysis from frozen features

```bash
python code/prepare_cached_inputs.py
python code/run_revised_identification.py
python code/run_revision_checks.py --both-controls --jobs 3
python code/summarize_revision_checks.py
python code/summarize_revision_checks.py --own-only
python code/run_revision_simulations.py
python code/run_nested_fusion.py --jobs 3
python code/run_robustness_diagnostics.py --jobs 3 --nested results/wid_fusion_robustness_2026-11-08/nested_writer_outcomes.csv
python code/reproduce.py --figures
```

The equivalent entry point is `python code/reproduce.py --full --jobs 3`. It verifies original input hashes before rebinding only the installed file paths in two cache metadata files. To regenerate all development/calibration fits, use `--refit-development`; that mode retains the old metadata as `.bak` and forces new fitting. Full analyses are substantially slower than saved-output verification/plotting. Seeds and environment versions are recorded beside the results; different linear algebra libraries can affect final floating-point digits.

## Optional feature extraction and initial attribute refitting

Photographs must first be obtained from `DATASET.md` and placed in `dataset/` with the released metadata structure. These commands require the larger dependency set and official pretrained model downloads:

```bash
python -m pip install -r requirements.txt
python code/wid_features.py --dataset dataset --output results/wid_handedness_2026-10-29/features
python code/run_wid_handedness.py --features results/wid_handedness_2026-10-29/features --output results/wid_handedness_2026-10-29
python code/run_wid_extras.py --features results/wid_handedness_2026-10-29/features --run results/wid_handedness_2026-10-29 --jobs 3
python code/reproduce.py --full --jobs 3 --refit-development
```

The original `run_wid_handedness.py` and `run_wid_extras.py` retain their historical identification/simulation routines because they also generate the reused attribute-classification inputs. Their historical identification outputs do **not** supply the revised article. Always run the revised pipeline after those commands. Use a separate copy of the archive to preserve supplied frozen inputs.

## Typeset supplementary methods

With TeX Live/latexmk installed:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error wid_supplement.tex
```

The main article is supplied separately through journal submission. Its three figures, two tables and numeric source inputs are all regenerated and included here, as mapped in `ARTIFACT_MAP.md`.
''',
    'docs/ARTIFACT_MAP.md':'''# Every table and figure: source and command

All outputs below are written to `paper/figures_revision/`. Run `python code/reproduce.py --figures` for the complete set. No model refitting is needed. CSV paths are under `results/`.

| Article item | Asset | Generator | Principal numerical inputs |
|---|---|---|---|
| Main Table 1 | `table_main.tex` | `make_revision_paper.py` | revised `id_summary.csv` |
| Main Table 2 | `table_nested_designated.tex` | `make_fusion_robustness_paper.py` | revised `id_paired_tests.csv`; nested `nested_paired_tests.csv` |
| Main Figure 1 | `performance.pdf` | `make_revision_paper.py` | revised summaries and designated paired tests |
| Main Figure 2 | `representation_utility.pdf` | `make_representation_utility.py` | revised `checks_identification_writer.csv`; `nested_writer_outcomes.csv` |
| Main Figure 3 | `diagnostics.pdf` | `make_revision_paper.py` | `checks_filter_identity.csv`, `checks_oracle_partitions.csv`, `checks_findings.json` |
| Supplement Figure S1 | `calibration.pdf` | `make_revision_paper.py` | `checks_reliability_bins.csv` |
| Supplement Figure S2 | `label_controls.pdf` | `make_revision_paper.py` | original `leakage_accuracy.csv`; split schematic is conceptual |
| Supplement Figure S3 | `simulation.pdf` | `make_revision_paper.py` | revised `breakeven_simulation.csv.gz` |
| Supplement Figure S4 | `fusion_robustness.pdf` | `make_fusion_robustness_paper.py` | temperature profiles, nested weights, seed effects, writer omission |
| Supplement Figure S5 | `margin_diagnostic.pdf` | `run_margin_diagnostic.py` | frozen features/scores; selected T/weights; fixed/nested outcomes |
| Supplement Tables S1–S7 | exclusions, partitions, dependence, cross-hand, calibration, both controls | `make_revision_paper.py` | unit manifest and revised `checks_*.csv` |
| Supplement Tables S8–S14 | nested effects/parameters, temperature, seed, omission, class effects | `make_fusion_robustness_paper.py` | nested and robustness CSVs |
| Supplement Table S15 | `table_all_settings_nested.tex` | `make_fusion_robustness_paper.py` | revised `id_summary.csv`; `nested_summary.csv` |

Additional table inputs (`table_all_settings.tex`, interval/dependence tables) are supplied even where the current supplement uses the expanded nested grid. Scalar text values come from `revision_macros.tex`, `fusion_robustness_macros.tex`, `representation_utility_macros.tex` and `margin_macros.tex`, generated by the same commands. All confidence intervals and per-writer values remain machine readable.
''',
    'docs/PROVENANCE.md':'''# Provenance and release scope

The release preserves frozen numerical features, pseudonymous unit IDs, original calibrated attribute scores, random-label assignments, selected parameters, fold exclusions, writer outcomes, numerical results and scientific seeds. Raw photographs, unpublished main-manuscript files, private review correspondence, administrative forms and local execution logs are outside this scientific release.

Absolute local filesystem prefixes in JSON metadata were converted to repository-relative paths for portability and privacy. Numeric values, input byte hashes and fold memberships were not changed. The optional cache-path binder checks every recorded input byte hash before converting only its path key to the installed location expected by the original loader.

`SHA256SUMS.txt` records the exact files in this release. Input hashes in generation-provenance records describe the original computation. Documentation and packaging may subsequently change; in particular the historical both-control module's explanatory docstring was corrected to avoid causal attribution. This does not alter its fitting/scoring behavior. The archive manifest, supplied outcomes and independent numerical checks permit the scientific inputs and results to be verified separately.

Intervals condition on the observed gallery, fitted predictions and selected parameters. Correlated settings, repeated calibration partitions, writer omission and gallery resampling do not constitute independent population validation. All original 42 writers remain in the primary results. Retrospective revisions are not described as preregistration.
'''}
    for name,text in files.items():(target/name).write_text(text)
    (target/'.gitignore').write_text('dataset/\n__pycache__/\n*.py[cod]\n.venv/\n.DS_Store\n*.aux\n*.log\n*.out\n*.spl\n*.blg\n*.fls\n*.fdb_latexmk\n*.synctex.gz\n*.bak\n_reproduction_logs/\n*.zip\n')


def audit_stage(target):
    paths=[p for p in target.rglob('*') if p.is_file() and '.git' not in p.relative_to(target).parts]
    forbidden=[]
    for p in paths:
        rel=p.relative_to(target)
        if 'dataset' in rel.parts or p.suffix.lower() in {'.jpg','.jpeg','.tif','.tiff'}:forbidden.append(str(rel))
        if p.suffix in {'.json','.md','.py','.txt','.csv','.tex','.cff'}:
            text=p.read_text(errors='replace')
            assert '/Users/' not in text and '/.codex/' not in text,'Unportable local metadata: '+str(rel)
    assert not forbidden,forbidden
    assert not (target/'paper/wid_manuscript.pdf').exists()
    assert (target/'results/wid_handedness_2026-10-29/features/unit_features.npz').is_file()
    (target/'manifests/release_scope.json').write_text(json.dumps(dict(version=VERSION,settings=120,writers=42,
        reference_configuration_rows=1152,dataset_photographs_included=False,frozen_features_included=True,
        primary_manuscript_included=False,scientific_input_hashes_preserved=True),indent=2)+'\n')


def archive(target):
    # A local cache-path test may bind code keys to the isolated checkout.
    # Canonicalize only path strings before the portable archive is hashed.
    def canonical(value):
        if isinstance(value,dict):return {canonical(k):canonical(v) for k,v in value.items()}
        if isinstance(value,list):return [canonical(x) for x in value]
        if isinstance(value,str):return portable(value).replace(str(target)+'/', '')
        return value
    for folder in ['results','manifests']:
        for p in (target/folder).rglob('*.json'):
            p.write_text(json.dumps(canonical(json.loads(p.read_text())),indent=2,allow_nan=False)+'\n')
    files=sorted(p for p in target.rglob('*') if p.is_file() and '.git' not in p.relative_to(target).parts
                 and '__pycache__' not in p.parts and p.name!='SHA256SUMS.txt' and p.suffix not in {'.log','.aux','.out','.spl','.blg','.fls','.fdb_latexmk','.bak'})
    manifest=''.join(f'{sha(p)}  {p.relative_to(target)}\n' for p in files)
    (target/'SHA256SUMS.txt').write_text(manifest)
    dest=ROOT/'submission/public';dest.mkdir(parents=True,exist_ok=True)
    zpath=dest/f'wid-reproducibility-v{VERSION}.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[target/'SHA256SUMS.txt']:
            info=zipfile.ZipInfo(str(p.relative_to(target)),date_time=(2026,11,8,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            z.writestr(info,p.read_bytes())
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        for line in z.read('SHA256SUMS.txt').decode().splitlines():
            expected,name=line.split('  ',1);assert hashlib.sha256(z.read(name)).hexdigest()==expected
    (dest/(zpath.name+'.sha256')).write_text(sha(zpath)+'  '+zpath.name+'\n')
    print(json.dumps(dict(archive=str(zpath),files=len(files)+1,bytes=zpath.stat().st_size,sha256=sha(zpath)),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,type=Path);p.add_argument('--archive-only',action='store_true');a=p.parse_args()
    if not a.archive_only:stage(a.stage)
    else:audit_stage(a.stage)
    archive(a.stage)


if __name__=='__main__':main()
