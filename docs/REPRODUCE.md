# Reproduce from the released numerical inputs

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
