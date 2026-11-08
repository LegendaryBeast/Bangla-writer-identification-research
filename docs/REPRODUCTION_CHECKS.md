# Release reproduction checks — 8 November 2026

The package was copied to an isolated checkout and tested from that location, without dataset photographs or pretrained weight downloads.

- `python code/reproduce.py --verify`: passed 121 numerical/input-integrity checks and 5,259 nested-fold/robustness checks.
- `python code/reproduce.py --figures`: regenerated every table, numeric macro and figure, then passed both verifiers again. No classifier was fitted by this mode.
- `python code/prepare_cached_inputs.py`: verified all original cache input byte hashes and successfully rebound the installed code paths. No probability, label, model parameter or input hash was changed.
- 120 unique settings and 1,152 reference-configuration rows were enumerated from the same protocol as the scoring code.
- All three official model-weight URLs returned HTTP 200 to HEAD requests; weights were not downloaded or redistributed.
- Dataset photographs, the original dataset ZIP, private correspondence, administrative forms and the unpublished main manuscript are excluded.

The supplied self-contained supplementary methods were separately compiled and rendered for layout inspection. These are reproducibility/consistency checks on the existing cohort, not new population validation.
