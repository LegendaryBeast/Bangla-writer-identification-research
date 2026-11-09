# CSAFE cross-session replication

Dataset: CSAFE Handwriting Database v2, DOI https://doi.org/10.25380/iastate.10062203.v2 . Original metadata and image archive directories were verified before performance testing. The current release contains 475 writers (48 left, 423 right, 3 ambidextrous, 1 missing). This study uses all 48 binary left-handed writers and 48 seeded right-handed writers. Raw images and model weights remain under ignored dataset/ paths and are not redistributed.

The original Data in Brief article (https://doi.org/10.1016/j.dib.2019.105059) describes the earlier 90-writer release; report the version-2 counts from its official metadata instead of confusing the two versions. The dataset is supplied under CC BY 4.0.

## Frozen design

protocol_v1.json was written before any CSAFE prediction or identification outcomes. protocol_v1_1.json records the active numerical amendment: initial logistic-solver convergence warnings prompted centering/scaling the training OOF decision scores, with the L2 penalty transformed to preserve the exact original C=10000 objective. All attribute fits were rerun with an analytic gradient certificate before examining performance tables. No feature, C, writer, endpoint or fusion-grid choice changed. This is a prospective specification for an extension of an existing study, not a public preregistration. Six long-text session-1 pages per other writer train the auxiliary classifier. One session-1 LND page per candidate supplies the identity gallery. Queries are distinct physical pages from session 2 (same and different prompt) and session 3 (different prompt). All pages of an outer held-out writer are excluded from auxiliary training/calibration. Writer-separated nested development selects fusion weights; gallery metadata remain available as in the original task.

Eight primary comparisons cover fixed/nested fusion against identity only for ResNet18 and ViT in both session-2 conditions, using the primary HC183 attribute model. Holm correction covers all eight. STROKE and intensity-adjusted HC183 attribute models, handcrafted identification, session 3, and 20 overlapping size-matched galleries are predefined secondary analyses. They are not selected according to which accuracy looks highest. Confidence intervals condition on fitted models and enrolled galleries; overlapping subsamples are not independent replication datasets.

## Reproduction

Install requirements-csafe.txt in a compatible Python environment. Run from the project root:

```sh
python code/acquire_csafe_replication.py
python -c "import torch, torchvision.models as m; torch.hub.set_dir('dataset/model_cache'); [w.get_state_dict(progress=True, check_hash=True) for w in (m.ResNet18_Weights.IMAGENET1K_V1, m.ViT_B_16_Weights.IMAGENET1K_V1)]"
python code/extract_csafe_features.py --threads 4
python code/run_csafe_replication.py --jobs 4
python code/validate_csafe_replication.py
python code/summarize_csafe_replication.py
```

The optional prefetch_csafe_handcrafted.py warms auxiliary-only feature caches using identical original extraction operations; it changes no model or preprocessing rule. Feature provenance records the source code, checkpoint hashes and neural accelerator. Other accelerators may change the last digits of embeddings; supplied frozen features support exact saved-feature reproduction.

The acquisition script retrieves selected original ZIP members with size/CRC verification and records their SHA256 hashes. It verifies the downloaded survey MD5. It does not claim to verify the MD5 of entire multi-gigabyte image ZIPs that were not fully downloaded.

A saved-output audit can run without raw images/checkpoints using `python code/validate_csafe_replication.py --saved-only`; its report records omitted raw-input checks. The full original audit is separate. No manuscript, publication claim or submission archive is changed by these experiment scripts.

Completed: all 196 settings, 50,880 writer-method outcomes, raw-input/fitting audits, objective-equivalence checks and a fresh independent saved-feature rerun passed. All nine core output files matched exactly after fresh fitting. Read RESULTS.md for the findings. The local predefined primary family concerns CSAFE contrasts; this is not a globally preregistered external-dataset sampling program. CSAFE and its analysis were fixed before inspecting its performance outcomes.
