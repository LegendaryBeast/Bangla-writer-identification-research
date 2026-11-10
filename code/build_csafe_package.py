#!/usr/bin/env python3
"""Package audited CSAFE code, features and results without raw images/weights."""
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/csafe_replication_2026-11-08'
DEST=ROOT/'submission/csafe_replication'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    validation=json.loads((OUT/'validation.json').read_text())
    assert validation['passed'] and validation['raw_images_checkpoints_verified']
    assert json.loads((OUT/'feature_equivalence.json').read_text())['passed']
    assert json.loads((OUT/'calibration_objective_validation.json').read_text())['passed']
    scripts=['acquire_csafe_replication.py','extract_csafe_features.py','prefetch_csafe_handcrafted.py',
        'run_csafe_replication.py','csafe_calibration.py','test_csafe_calibration.py',
        'verify_csafe_feature_equivalence.py','validate_csafe_replication.py','summarize_csafe_replication.py',
        'build_csafe_package.py','wid_features.py','hc183.py','stroke_features.py','run_wid_handedness.py',
        'revised_protocol.py','run_nested_fusion.py']
    files=[p for p in OUT.rglob('*') if p.is_file() and not p.name.endswith('.partial.csv')]
    files += [ROOT/'code'/name for name in scripts]+[ROOT/'requirements-csafe.txt',ROOT/'LICENSE']
    assert not any(p.suffix.lower() in ('.jpg','.jpeg','.tif','.tiff') or 'dataset' in p.relative_to(ROOT).parts for p in files)
    sums=''.join(f'{sha(p)}  {p.relative_to(ROOT)}\n' for p in sorted(files))
    DEST.mkdir(exist_ok=True,parents=True)
    archive=DEST/'csafe-replication-2026-11-08.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(ROOT)))
        z.writestr('SHA256SUMS.txt',sums)
        z.writestr('README.md','''# CSAFE cross-session replication

Read results/csafe_replication_2026-11-08/RESULTS.md and README.md. Install requirements-csafe.txt in a compatible Python environment. The original design is protocol_v1.json; protocol_v1_1.json records the objective-preserving numerical calibration amendment before performance inspection.

Saved-feature reproduction (no raw data or GPU required):

```
python code/test_csafe_calibration.py
python code/validate_csafe_replication.py --saved-only
python code/run_csafe_replication.py --jobs 4
python code/validate_csafe_replication.py --saved-only
python code/summarize_csafe_replication.py
```

To force fresh auxiliary fitting, move results/csafe_replication_2026-11-08/attribute_cache/ aside before the analysis command. Source-image reproduction and exact checkpoint links are in the experiment README. Raw images and checkpoint binaries are downloaded separately; none is distributed here. The feature matrix, fold manifests and numerical outputs are included. The original full audit and the saved-only audit have different scopes and separate reports. The archive contains analysis materials; manuscript sources are supplied separately. No raw images or checkpoint binaries are redistributed.
''')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for line in z.read('SHA256SUMS.txt').decode().splitlines():
            digest,name=line.split('  ',1);assert hashlib.sha256(z.read(name)).hexdigest()==digest
        assert not any(name.startswith('dataset/') for name in z.namelist())
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(files)),indent=2))


if __name__=='__main__':main()
