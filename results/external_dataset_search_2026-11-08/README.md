# Alternative dataset verification — 8 November 2026

The screenshot shows `benhachem/KHATT`, a public OCR pilot mirror. Its loader exposes image/text, and its Groundtruth-Unicode.xlsx contains filenames and transcriptions, not handedness. No verified writer-to-handedness mapping was found there. Downloading this mirror alone will not enable the proposed handedness-assisted identification experiment.

The official CSAFE Handwriting Database v2 is a more actionable candidate: https://doi.org/10.25380/iastate.10062203.v2 . The original article describes a 90-writer release, but the current official metadata contains 475 distinct writer IDs: 48 left-handed, 423 right-handed, 3 ambidextrous, and 1 missing handedness. These counts were read directly from the original surveydata_v2.csv. The current release documents 27 samples per writer across three acquisition sessions. A 48/48 balanced evaluation may therefore be possible subject to image matching and quality checks; ambidextrous/missing labels must not be silently reassigned.

Firemaker is a second documented option: https://doi.org/10.5281/zenodo.1194612 . Its official description specifies 250 writers, four writing conditions on separate pages, and an original Firemaker-writer-info.dat whose third column is handedness. The actual left/right counts have not yet been audited. Only normal copied and normal self-generated pages should enter the primary natural-writing protocol; uppercase and disguised samples would be separately designated robustness conditions.

No performance outcomes have been generated for either candidate. Choosing an accessible dataset with authenticated labels does not guarantee above-chance handedness prediction. Dataset eligibility should be based on metadata and protocol compatibility rather than on whether a fitted classifier achieves a preferred accuracy.

The removed MPHD experiment results have not been restored or added to the manuscript.
