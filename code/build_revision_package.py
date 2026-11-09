#!/usr/bin/env python3
"""Build self-contained review archives; never upload or publish anything."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'submission/revised'


def digest(data):return hashlib.sha256(data).hexdigest()


def build(name,paths,readme):
    entries={str(p.relative_to(ROOT)):p.read_bytes() for p in sorted(set(paths))}
    entries['ARCHIVE_README.md']=readme.encode()
    manifest=''.join(f'{digest(data)}  {name}\n' for name,data in sorted(entries.items()))
    entries['SHA256SUMS.txt']=manifest.encode()
    target=DEST/name
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for key,data in sorted(entries.items()):
            info=zipfile.ZipInfo(key,date_time=(2026,11,8,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            z.writestr(info,data)
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:raise AssertionError('Archive CRC failure')
        for line in z.read('SHA256SUMS.txt').decode().splitlines():
            expected,key=line.split('  ',1)
            if digest(z.read(key))!=expected:raise AssertionError('Archive hash mismatch: '+key)
    return {'file':str(target.relative_to(ROOT)),'files':len(entries),'bytes':target.stat().st_size,'sha256':digest(target.read_bytes())}


def main():
    report=json.loads((ROOT/'results/wid_revision_2026-11-08/validation_report.json').read_text())
    if not report['passed'] or len(report['pdfs'])!=2:
        raise RuntimeError('Run validate_revision.py --pdf successfully before building archives')
    for entry in report['pdfs']:
        current=ROOT/entry['file']
        if digest(current.read_bytes())!=entry['sha256']:
            raise RuntimeError('PDF changed after validation: '+entry['file'])
    extra=json.loads((ROOT/'results/wid_fusion_robustness_2026-11-08/additional_validation_report.json').read_text())
    if not extra['passed']:raise RuntimeError('Additional fusion/robustness checks must pass')
    DEST.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/'paper/wid_manuscript.tex',ROOT/'paper/wid_supplement.tex',ROOT/'paper/references.bib',ROOT/'paper/highlights.txt']
    sources += [ROOT/'paper/reviews/PRL_TARGETED_RESPONSE_2026-11-08.md',
                ROOT/'submission/revised/PRL-authorship-and-formatting-official.pdf',
                ROOT/'submission/revised/AUTHORSHIP_COMPLETION.md']
    sources+=list((ROOT/'paper/figures_revision').glob('*.tex'))+list((ROOT/'paper/figures_revision').glob('*.pdf'))
    final=ROOT/'submission/prl_final'
    sources += [p for folder in ['upload','assets','source'] for p in (final/folder).glob('*') if p.is_file()]
    sources += [p for p in [final/'README.md',final/'Technical_audit.md',final/'Technical_audit.json',
                            final/'Numerical_claim_crosscheck.csv',final/'Table_cell_crosscheck.csv',
                            final/'Preparation_validation.json'] if p.exists()]
    pdfs=[ROOT/'paper/wid_manuscript.pdf',ROOT/'paper/wid_supplement.pdf']
    text='''# Revised manuscript source\n\nMain manuscript: paper/wid_manuscript.tex. Supplement: paper/wid_supplement.tex.\nResearch highlights: paper/highlights.txt. Use a TeX distribution with elsarticle.\nThe point-by-point response is in paper/reviews/PRL_TARGETED_RESPONSE_2026-11-08.md.\nThe manuscript-specific authorship form, graphical abstract and cover letter are in submission/prl_final/upload/. The original blank official form is retained in submission/revised/.\nThe form is unsigned; final author attestations and submission-system entries remain author tasks.\nRun latexmk -pdf -interaction=nonstopmode -halt-on-error for each source from paper/.\nThe figures are vector PDFs and all numerical text/table inputs are included.\nSHA256SUMS.txt verifies every enclosed file except the manifest itself.\nNo external publication, submission or DOI registration has occurred.\n'''
    archives=[build('wid-submission-source-2026-11-08-v1.5.0.zip',sources+pdfs,text)]
    files=sources+pdfs+[ROOT/p for p in ['README.md','LICENSE','RESULTS_LICENSE.md','CITATION.cff','requirements.txt','requirements-frozen.txt']]
    files+=list((ROOT/'code').glob('*.py'))
    for folder in [ROOT/'results/wid_handedness_2026-10-29',ROOT/'results/wid_revision_2026-11-08',ROOT/'results/wid_fusion_robustness_2026-11-08']:
        files.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix not in {'.log'} and '__pycache__' not in p.parts)
    files+=list((ROOT/'paper/notes').glob('*.md'))
    files+=list((ROOT/'paper/reviews').glob('*.md'))
    files+=list((ROOT/'paper/figures_revision').glob('*.png'))
    text='''# Revised reproducibility materials\n\nSee README.md for commands, information scopes, versions and licences.\nThe revised main and supplementary PDFs and all required LaTeX assets are in paper/.\nresults/wid_revision_2026-11-08/ supplies the fixed-rule identification evidence.\nresults/wid_fusion_robustness_2026-11-08/ supplies nested learned weights and within-cohort diagnostics.\nresults/wid_handedness_2026-10-29/ retains historical outputs and the frozen feature/attribute inputs.\nHistorical identification summaries do not supply the revised manuscript's values.\nUnchanged enrolment/random-label attribute classification controls are explicitly reused.\nDataset photographs are NOT enclosed; obtain them separately using doi:10.17632/n4sjstdzfy.1.\nThe frozen feature file and predictions allow analysis reproduction without downloading images or model weights.\nSHA256SUMS.txt verifies every enclosed file except the manifest itself.\nConditional intervals omit new-gallery and model-refitting uncertainty.\nNo external publication, submission or DOI registration has occurred.\n'''
    archives.append(build('wid-reproducibility-2026-11-08-v1.5.0.zip',files,text))
    (DEST/'archive_manifest-v1.5.0.json').write_text(json.dumps(archives,indent=2)+'\n')
    print(json.dumps(archives,indent=2))


if __name__=='__main__':main()
