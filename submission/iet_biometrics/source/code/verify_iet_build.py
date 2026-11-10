#!/usr/bin/env python3
"""Rebuild isolated IET sources and compare every page with delivered PDFs."""
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'submission/iet_biometrics'
if not DEST.exists() and (ROOT.parent / 'Manuscript.pdf').exists():
    DEST = ROOT.parent


def normalized_pages(path):
    with fitz.open(path) as doc:
        return [re.sub(r'\s+', ' ', p.get_text()).strip() for p in doc]


def main():
    report = {}
    with tempfile.TemporaryDirectory(prefix='iet-source-build-') as directory:
        archive = DEST / 'wid-iet-biometrics-submission-2026-11-09.zip'
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.namelist():
                if member.startswith('source/paper/iet_biometrics/'):
                    assert '..' not in Path(member).parts
                    zipped.extract(member, directory)
        source = Path(directory) / 'source/paper/iet_biometrics'
        for stem, filename in [('manuscript', 'Manuscript.pdf'),
                               ('supporting_information', 'Supporting_information.pdf')]:
            result = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode',
                                     '-halt-on-error', stem + '.tex'], cwd=source,
                                    capture_output=True, text=True, errors='replace')
            assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
            log = (source / (stem + '.log')).read_text(errors='replace')
            assert not re.search(r'Overfull \\[hv]box|undefined references|Citation .* undefined|'
                                 r'Missing character|LaTeX Error|Emergency stop', log, re.I)
            expected = normalized_pages(DEST / filename)
            actual = normalized_pages(source / (stem + '.pdf'))
            assert actual == expected, (stem, 'isolated-build page text differs')
            report[stem] = dict(pages=len(actual), isolated_build_passed=True,
                                all_page_text_matches_deliverable=True,
                                delivered_pdf_sha256=hashlib.sha256((DEST / filename).read_bytes()).hexdigest(),
                                normalized_page_text_sha256=[
                                    hashlib.sha256(s.encode()).hexdigest() for s in actual])
    (DEST / 'Independent_build_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: {a: b for a, b in v.items() if a != 'normalized_page_text_sha256'}
                      for k, v in report.items()}, indent=2))


if __name__ == '__main__':
    main()
