# Provenance and release scope

The release preserves frozen numerical features, pseudonymous unit IDs, original calibrated attribute scores, random-label assignments, selected parameters, fold exclusions, writer outcomes, numerical results and scientific seeds. Raw photographs, unpublished main-manuscript files, private review correspondence, administrative forms and local execution logs are outside this scientific release.

Absolute local filesystem prefixes in JSON metadata were converted to repository-relative paths for portability and privacy. Numeric values, input byte hashes and fold memberships were not changed. The optional cache-path binder checks every recorded input byte hash before converting only its path key to the installed location expected by the original loader.

`SHA256SUMS.txt` records the exact files in this release. Input hashes in generation-provenance records describe the original computation. Documentation and packaging may subsequently change; in particular the historical both-control module's explanatory docstring was corrected to avoid causal attribution. This does not alter its fitting/scoring behavior. The archive manifest, supplied outcomes and independent numerical checks permit the scientific inputs and results to be verified separately.

Intervals condition on the observed gallery, fitted predictions and selected parameters. Correlated settings, repeated calibration partitions, writer omission and gallery resampling do not constitute independent population validation. All original 42 writers remain in the primary results. Retrospective revisions are not described as preregistration.
