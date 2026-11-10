#!/usr/bin/env python3
"""Assemble the revised IET Biometrics submission package (9 November 2026 revision).

The previous package is snapshotted into submission/iet_biometrics/previous/ before any
file is replaced. The script verifies the LaTeX logs, cross-references and citation
order, copies PDFs/DOCX/figures/tables, builds the revision-analysis archive, refreshes
the source tree, writes validation records and checksums, and rebuilds the package zip.
"""
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'paper/iet_biometrics'
DEST = ROOT / 'submission/iet_biometrics'
PREV = DEST / 'previous'
STAMP = '2026-11-09-pre-revision'
ZIPNAME = 'wid-iet-biometrics-submission-2026-11-09.zip'
REVZIP = 'wid-revision-analyses-2026-11-09.zip'
PANDOC = shutil.which('pandoc') or '/opt/homebrew/bin/pandoc'
GUIDE = 'https://ietresearch.onlinelibrary.wiley.com/page/journal/ietbm/homepage/author-guidelines'
BUILD_JUNK = {'.aux', '.log', '.fdb_latexmk', '.fls', '.out', '.pag', '.blg', '.synctex.gz'}
REVISION_CODE = ['run_p3_sweep_rank_counts.py', 'extract_csafe_sift_vlad.py', 'run_p2_csafe_extensions.py', 'run_p1_precision_mechanism.py', 'run_p1_csafe_adjusted_fix.py', 'check_p1_bangla_adjusted.py',
                 'build_p1_csafe_corrected.py', 'make_p1_figures.py', 'make_p1_si_tables.py', 'finalize_p1_submission.py',
                 'run_csafe_replication.py', 'csafe_calibration.py', 'run_wid_handedness.py', 'revised_protocol.py',
                 'run_nested_fusion.py', 'summarize_csafe_replication.py']
SOURCE_CODE = ['run_p3_sweep_rank_counts.py', 'extract_csafe_sift_vlad.py', 'run_p2_csafe_extensions.py', 'build_iet_docx.py', 'convert_iet_biometrics.py', 'verify_iet_build.py', 'make_p1_figures.py',
               'make_p1_si_tables.py', 'finalize_p1_submission.py', 'make_revision_paper.py',
               'make_representation_utility.py', 'summarize_csafe_replication.py']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    target = PREV / STAMP
    if target.exists():
        return target
    target.mkdir(parents=True)
    for item in DEST.iterdir():
        if item.name == 'previous':
            continue
        if item.is_dir():
            shutil.copytree(item, target / item.name)
        else:
            shutil.copy2(item, target / item.name)
    return target


def check_latex():
    tex = (SOURCE / 'manuscript.tex').read_text()
    aux = (SOURCE / 'manuscript.aux').read_text()
    cites = [k.strip() for g in re.findall(r'\\cite\w*\{([^}]+)\}', tex) for k in g.split(',')]
    ordered = list(dict.fromkeys(cites))
    bib = re.findall(r'\\bibcite\{([^}]+)\}', aux)
    assert ordered == bib, 'citation order differs from bibliography order'
    labels = dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}', aux))
    refs = re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', tex)
    assert all(r in labels for r in refs), [r for r in refs if r not in labels]
    for stem in ('manuscript', 'supporting_information'):
        log = (SOURCE / f'{stem}.log').read_text(errors='replace')
        assert not re.search(r'Overfull \\[hv]box|undefined references|Citation .* undefined|Missing character|'
                             r'LaTeX Error|Emergency stop', log, re.I), stem
    figs = re.findall(r'\\begin\{figure\*\}.*?\\end\{figure\*\}', tex, re.S)
    names = [re.search(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', b).group(1) for b in figs]
    tables = len(re.findall(r'\\begin\{table\*\}', tex))
    eqs = len(re.findall(r'\\begin\{(?:equation|align)\}', tex))
    meta = json.loads((SOURCE / 'metadata.json').read_text())
    words = re.findall(r"[\w]+(?:[-'][\w]+)*", re.sub(r'\\[A-Za-z]+|[{}$]', ' ', meta['abstract']))
    assert len(words) <= 300
    for phrase in ['An AI assistant was used to audit analysis code, and assist with drafting.',
                   'Formal institutional ethical approval was not required under the policies of Shahjalal University']:
        assert phrase in tex
    return dict(citations=len(ordered), figures=names, tables=tables, equations=eqs, abstract_words=len(words))


def check_pdf(path):
    with fitz.open(path) as doc:
        text = '\n'.join(p.get_text() for p in doc)
        assert all(bad not in text for bad in ['??', 'PLEASE INSERT', '10.1002/0000', 'AIChE Journal'])
        for i, page in enumerate(doc):
            assert '\ufffd' not in page.get_text(), (path, i + 1)
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    for span in line['spans']:
                        x0, y0, x1, y1 = span['bbox']
                        assert x0 >= -1 and y0 >= -1 and x1 <= page.rect.width + 1 and y1 <= page.rect.height + 1
        return len(doc)


def cover_letter():
    out = DEST / 'Cover_letter.pdf'
    subprocess.run([PANDOC, 'cover_letter.md', '-o', str(out), '--pdf-engine=pdflatex', '-V', 'geometry:margin=25mm',
                    '-V', 'fontsize=11pt', '-V', 'colorlinks=false'],
                   cwd=SOURCE, check=True)
    return out


def revision_archive():
    out = DEST / 'reproducibility' / REVZIP
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in REVISION_CODE:
            z.write(ROOT / 'code' / name, f'code/{name}')
        for folder in ('p1_revision_2026-11-09', 'p2_extensions_2026-11-09', 'p3_sweep_2026-11-09'):
            res = ROOT / 'results' / folder
            for p in sorted(res.rglob('*')):
                if p.is_file() and not p.name.startswith('.') and not (p.suffix == '.npz' and 'attribute_cache' in p.parts):
                    z.write(p, f'results/{folder}/{p.relative_to(res)}')
        z.writestr('README_ARCHIVE.md', '''# Revision analyses, 9 November 2026

Post hoc analyses added in the 9 November 2026 revision of the IET Biometrics manuscript.
See results/p1_revision_2026-11-09/README.md for findings and interpretation.

Place this archive's code/ and results/ folders in the repository root, next to the
archived results/wid_revision_2026-11-08, results/wid_fusion_robustness_2026-11-08,
results/wid_handedness_2026-10-29 (release v1.6.0) and results/csafe_replication_2026-11-08
(csafe-replication-2026-11-08.zip) folders, then run:

    python code/run_p1_precision_mechanism.py   # precision, error clustering, query-hand split
    python code/run_p1_csafe_adjusted_fix.py    # stable nuisance regression, CSAFE full gallery
    python code/check_p1_bangla_adjusted.py     # BanglaHand is unaffected
    python code/build_p1_csafe_corrected.py     # corrected CSAFE tables and figures
    python code/extract_csafe_sift_vlad.py      # SIFT-VLAD features (needs CSAFE images)
    python code/run_p2_csafe_extensions.py      # SIFT-VLAD matcher and gender attribute
    python code/run_p3_sweep_rank_counts.py     # accuracy sweep, rank endpoints, sample sizes
    python code/make_p1_figures.py              # revised manuscript figures
    python code/make_p1_si_tables.py            # revised supporting tables

Attribute-model cache arrays (*.npz under attribute_cache/) are omitted; the scripts refit them.
All analyses are post hoc and are reported as such in the manuscript.
''')
    (DEST / 'reproducibility' / (REVZIP + '.sha256')).write_text(f'{sha(out)}  {REVZIP}\n')
    return out


def refresh_source():
    src = DEST / 'source'
    paper = src / 'paper/iet_biometrics'
    if paper.exists():
        shutil.rmtree(paper)
    def ignore(directory, names):
        skip = set()
        for n in names:
            p = Path(directory) / n
            if p.suffix in BUILD_JUNK or n.endswith('.synctex.gz'):
                skip.add(n)
            elif Path(directory) == SOURCE and p.suffix == '.pdf':
                skip.add(n)
        return skip
    shutil.copytree(SOURCE, paper, ignore=ignore)
    code = src / 'code'
    code.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_CODE:
        shutil.copy2(ROOT / 'code' / name, code / name)


def main():
    previous = snapshot()
    info = check_latex()
    rows = []
    for src, target in [(SOURCE / 'manuscript.pdf', DEST / 'Manuscript.pdf'),
                        (SOURCE / 'supporting_information.pdf', DEST / 'Supporting_information.pdf')]:
        pages = check_pdf(src)
        shutil.copy2(src, target)
        rows.append(dict(file=str(target.relative_to(ROOT)), pages=pages, sha256=sha(target)))
    cl = cover_letter()
    rows.append(dict(file=str(cl.relative_to(ROOT)), pages=check_pdf(cl), sha256=sha(cl)))
    figures = DEST / 'figures'
    if figures.exists():
        shutil.rmtree(figures)
    figures.mkdir()
    for n, name in enumerate(info['figures'], 1):
        shutil.copy2(SOURCE / name, figures / f'Figure_{n}.pdf')
    for name in ('Complete_120_setting_grid.csv', 'CSAFE_complete_196_settings.csv'):
        shutil.copy2(SOURCE / 'tables' / name, DEST / name)
    revision_archive()
    refresh_source()
    # Stale validation records from the previous package are kept only in the snapshot.
    (DEST / 'Layout_revision_validation.json').unlink(missing_ok=True)
    ind = DEST / 'Independent_build_validation.json'
    if ind.exists():
        rec = json.loads(ind.read_text())
        current = {'manuscript': sha(DEST / 'Manuscript.pdf'), 'supporting_information': sha(DEST / 'Supporting_information.pdf')}
        if any(rec.get(k, {}).get('delivered_pdf_sha256') != v for k, v in current.items()):
            ind.unlink()  # stale: rerun code/verify_iet_build.py, then this script
    for old in DEST.glob('wid-iet-biometrics-submission-*.zip*'):
        old.unlink()
    word = json.loads(Path('/private/tmp/wid-iet-word-qa/structural_checks.json').read_text())
    (DEST / 'Word_validation.json').write_text(json.dumps(dict(
        passed=True, method='Pandoc conversion with native Word mathematics; every decimal table value checked against '
        'the Markdown source; structural counts verified.', all_pages_visually_reviewed=False,
        visual_review_note='No Word/LibreOffice renderer was available in this environment; open the DOCX files and '
        'check pagination, figure sizing and table widths before upload.',
        documents=[dict(**d, sha256=sha(ROOT / d['file'])) for d in word],
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [SOURCE / 'manuscript.md', SOURCE / 'supporting_information.md',
                                                               SOURCE / 'cover_letter.md', SOURCE / 'refs.bib', SOURCE / 'ieee.csl']},
        review_date='2026-11-09'), indent=2) + '\n')
    report = dict(passed=True, date='9 November 2026', target='IET Biometrics', pdfs=rows,
                  main_figures=len(info['figures']), main_tables=info['tables'], main_equations=info['equations'],
                  main_references=info['citations'], abstract_words=info['abstract_words'],
                  supplementary_tables=31, supplementary_figures=8,
                  revision='Title, abstract, introduction, results and discussion revised; post hoc precision and '
                           'error-clustering analyses; numerically corrected CSAFE adjusted sensitivity model; '
                           '26 new verified references; SIFT-VLAD matcher and gender attribute on CSAFE (Figure 9, S18); Figures 1, 3, 5 new and Figures 6, 8 restyled; former Table 4, '
                           'Table 6, Table 8 and Figure 5 moved to the supporting information.',
                  previous_package_snapshot=str(previous.relative_to(ROOT)),
                  pdf_visual_review='Every page of the main PDF rendered and reviewed after the final build.',
                  guide=GUIDE, submission_made=False, coauthor_approval_certified=False,
                  orcid='To be provided by submitting author in the journal portal')
    (DEST / 'Conversion_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    write_readme(rows, info)
    files = sorted(p for p in DEST.rglob('*') if p.is_file() and not p.name.startswith('.') and 'previous' not in p.relative_to(DEST).parts
                   and p.name != 'SHA256SUMS.txt')
    (DEST / 'SHA256SUMS.txt').write_text(''.join(f'{sha(p)}  {p.relative_to(DEST)}\n' for p in files))
    archive = DEST / ZIPNAME
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in files + [DEST / 'SHA256SUMS.txt']:
            z.write(p, str(p.relative_to(DEST)))
    (DEST / (ZIPNAME + '.sha256')).write_text(f'{sha(archive)}  {ZIPNAME}\n')
    print(json.dumps(report, indent=2))


def write_readme(rows, info):
    pages = {Path(r['file']).name: r['pages'] for r in rows}
    (DEST / 'README.md').write_text(f'''# IET Biometrics submission package

Revised on 9 November 2026. Main PDF: **{pages['Manuscript.pdf']} pages**, with **{len(info['figures'])} figures, {info['tables']} tables and {info['citations']} references**. Supporting PDF: **{pages['Supporting_information.pdf']} pages**, separate. The single-column editable Word versions have different pagination.

Title: *Does predicted handedness improve writer identification? Oracle gains versus achieved utility in Bangla and English handwriting*.

## Files to use

- `Manuscript.docx` is the editable main submission. `Manuscript.pdf` is its Wiley-template typeset counterpart.
- `Supporting_information.pdf` / `.docx` contain the complete methods, controls, grids, the post hoc revision analyses (Sections S14–S16), the moved comparison table (S17) and the SIFT-VLAD and gender extensions (S18) and the accuracy sweep, rank endpoints and sample sizes (S19).
- `Cover_letter.pdf` / `.docx` are addressed to IET Biometrics.
- `figures/` contains the {len(info['figures'])} vector main figures. `Complete_120_setting_grid.csv` contains the BanglaHand grid. `CSAFE_complete_196_settings.csv` contains 980 method rows across 196 external settings, with the corrected adjusted-model rows.
- `reproducibility/csafe-replication-2026-11-08.zip` contains the original CSAFE code, frozen features, protocols, manifests and results. `reproducibility/{REVZIP}` contains the revision analyses (precision, error clustering, corrected CSAFE sensitivity model, SIFT-VLAD matcher, gender attribute, figure and table generators).
- `source/` contains matched Markdown and LaTeX sources, references, figure assets and build helpers.
- `previous/{STAMP}/` is an untouched copy of the package as it was before this revision.

## What changed in this revision

- The title and abstract now state the question and the main finding.
- The contributions are listed explicitly, and a new protocol figure (Figure 1) is added.
- A post hoc precision analysis is added: Figure 3 and Supporting Tables S21–S22. Nested fusion is within ±0.96 points of identity-only ranking across settings.
- An error-clustering decomposition explains why known handedness is no better than a random partition: Figure 5 and Supporting Table S23.
- A numerical instability in the CSAFE adjusted sensitivity model is corrected: Section S16 and Table S24. Its below-chance accuracy (34%) becomes 61–69%. Primary results are unchanged.
- A handwriting-specific SIFT-VLAD matcher (91–96% top-1) and gender as a second attribute are evaluated on CSAFE under a protocol fixed before their outcomes (Figure 9, Section S18). The opportunity disappears with the stronger matcher, errors cluster within gender too, and none of 16 primary fusion contrasts shows a gain.
- An accuracy sweep with synthetic independent evidence (Figure 4, Table S29) shows that fusion would need roughly 70–90% attribute accuracy to help; rank-based endpoints (top-5, MRR; Table S30) and the sample sizes behind every clustering estimate (Table S31) are reported, and the contribution claims are worded more cautiously. The representation-utility figure moves to Figure S8.
- The related work is expanded to 52 references (26 new, each checked against a bibliographic record). Figures 6 and 8 are restyled. The former Table 4, Table 6, Table 8 and Figure 5 move to the supporting information.

## Verified journal requirements

The [official author instructions]({GUIDE}) were checked on 9 November 2026:

- Editable Word is accepted. LaTeX submissions also need a review PDF and supporting sources.
- Author identities remain visible for single-anonymized review.
- The Abstract has {info['abstract_words']} words, below 300.
- Numbered citations follow first appearance; this is checked automatically.
- Line numbers are present in the main files.

The submitting author must supply an ORCID in the portal and obtain final coauthor approval. This package does not certify those approvals or make a journal submission.

## Before upload

- The DOCX files passed structural checks: every table value matched the source, and native equations and images are present. They were **not** visually rendered here; open them in Word and check pagination, figure size and table width.
- The revised title differs from the earlier approved title. Confirm it with the coauthor.

## Build sources

From `source/paper/iet_biometrics/`, run:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supporting_information.tex
```

For Word, run `python code/convert_iet_biometrics.py markdown` and then `python code/build_iet_docx.py` from the repository root. Figures are regenerated by `code/make_p1_figures.py` and the revision tables by `code/make_p1_si_tables.py`.

See `Conversion_validation.json`, `Word_validation.json`, `Independent_build_validation.json` and `SHA256SUMS.txt` for verification.
''')


if __name__ == '__main__':
    main()
