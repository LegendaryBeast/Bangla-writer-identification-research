#!/usr/bin/env python3
"""Verify and package the IET manuscript and completed CSAFE validation."""
import csv
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import fitz

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'paper/iet_biometrics'
DEST=ROOT/'submission/iet_biometrics'
GUIDE='https://ietresearch.onlinelibrary.wiley.com/page/journal/ietbm/homepage/author-guidelines'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    inputs=json.loads((SOURCE/'conversion_inputs.json').read_text())
    for path,digest in inputs.items():
        assert sha(ROOT/path)==digest,('PRL input changed',path)
    word=json.loads((DEST/'Word_validation.json').read_text())
    assert word['passed'] and word['all_pages_visually_reviewed']
    for row in word['documents']:
        assert sha(ROOT/row['file'])==row['sha256']
    for name,digest in word['source_sha256'].items():
        assert sha(ROOT/name)==digest,('Word source changed after verification',name)
    maintex=(SOURCE/'manuscript.tex').read_text()
    suptex=(SOURCE/'supporting_information.tex').read_text()
    mainaux=(SOURCE/'manuscript.aux').read_text()
    cites=[k for group in re.findall(r'\\cite\w*\{([^}]+)\}',maintex) for k in group.split(',')]
    ordered=list(dict.fromkeys(cites))
    bibkeys=re.findall(r'\\bibcite\{([^}]+)\}',mainaux)
    assert ordered==bibkeys and len(ordered)==26
    labels=dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}',mainaux))
    assert all(k in labels for k in re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',maintex))
    assert len(re.findall(r'\\begin\{figure\*\}',maintex))==7
    assert len(re.findall(r'\\begin\{table\*\}',maintex))==8
    assert [labels[k] for k in ['fig:performance','fig:representation','fig:diagnostics','fig:mainlabels','fig:mainrobustness','fig:mainmargins']]==list('123456')
    assert [labels[k] for k in ['tab:mainnesting','tab:main','tab:tests','tab:maincalibration','tab:mainseeds']]==list('23456')
    assert labels['tab:related']=='1' and labels['tab:positioning']=='8'
    assert labels['fig:csafeutility']=='7' and labels['tab:csafemain']=='7'
    assert 'The dataset was released separately' not in maintex
    assert 'empirical study' not in maintex.lower()
    assert 'Handedness for Bangla writer identification: Oracle gains and predicted-attribute utility' in maintex
    assert re.findall(r'\\author\[1\]\{([^}]+)\}',maintex)==['Tanzim Hasan Prappo','M. Shahidur Rahman']
    disclosure=('An AI assistant was used to audit analysis code, and assist with drafting. The authors '
                'are responsible for verifying the analyses, references and final text, and for all publication claims.')
    assert disclosure in maintex
    assert 'Formal institutional ethical approval was not required under the policies of Shahjalal University of Science and Technology' in maintex
    assert not re.search(r'original study.s focus|followed review|post-review|historical, incompletely',maintex)
    meta=json.loads((SOURCE/'metadata.json').read_text())
    abstract_plain=re.sub(r'\\[A-Za-z]+|[{}]',' ',meta['abstract'])
    words=re.findall(r"[\w]+(?:[-'][\w]+)*",abstract_plain)
    assert len(words)<=300
    bbl=(SOURCE/'manuscript.bbl').read_text()
    for doi in ['10.17632/n4sjstdzfy.1','10.2139/ssrn.7662278','10.17632/99t9jhvksv.1']:
        assert doi in bbl
    margin=json.loads((ROOT/'results/wid_fusion_robustness_2026-11-08/margin_validation.json').read_text())
    assert margin['passed'] and margin['decisions_reconstructed']==664680 and margin['exact_boundary_ties']==0
    pdfrows=[]
    for src,target in [(SOURCE/'manuscript.pdf',DEST/'Manuscript.pdf'),
                       (SOURCE/'supporting_information.pdf',DEST/'Supporting_information.pdf'),
                       (DEST/'Cover_letter.pdf',DEST/'Cover_letter.pdf')]:
        doc=fitz.open(src)
        text='\n'.join(page.get_text() for page in doc)
        assert all(bad not in text for bad in ['??','Added at production','PLEASE INSERT','10.1002/0000','AIChE Journal'])
        for i,page in enumerate(doc):
            assert '\ufffd' not in page.get_text(),(src,i+1)
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line['spans']:
                        x0,y0,x1,y1=span['bbox']
                        assert x0>=-1 and y0>=-1 and x1<=page.rect.width+1 and y1<=page.rect.height+1,(src,i+1,span)
        if src!=target:shutil.copy2(src,target)
        pdfrows.append(dict(file=str(target.relative_to(ROOT)),pages=len(doc),sha256=sha(target)))
    assert pdfrows[0]['pages']>7 and pdfrows[1]['pages']>=14
    for stem in ['manuscript','supporting_information']:
        log=(SOURCE/(stem+'.log')).read_text(errors='replace')
        assert not re.search(r'Overfull \\[hv]box|undefined references|Citation .* undefined|Missing character|LaTeX Error|Emergency stop',log,re.I),stem
    # Accompany the wide complete grid with an editable 120-row CSV.
    rows=[]
    for line in (ROOT/'paper/figures_revision/table_all_settings_nested.tex').read_text().splitlines():
        if '&' in line and not line.lstrip().startswith('%'):
            row=[x.strip() for x in line.removesuffix('\\\\').strip().split('&')]
            assert len(row)==10
            rows.append(row)
    assert len(rows)==120
    tablecsv=SOURCE/'tables/Complete_120_setting_grid.csv'
    with tablecsv.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['representation','text','n','k','none_pct','known_pct','fixed_pct','nested_pct','filter_pct','independence_threshold_pct']);w.writerows(rows)
    shutil.copy2(tablecsv,DEST/tablecsv.name)
    figures=DEST/'figures';figures.mkdir(exist_ok=True)
    figureblocks=re.findall(r'\\begin\{figure\*\}.*?\\end\{figure\*\}',maintex,re.S)
    names=[re.search(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',block).group(1) for block in figureblocks]
    assert len(names)==7
    for n,name in enumerate(names,1):
        asset=SOURCE/name
        original=(ROOT/'paper/figures_revision'/asset.name if n<=6 else ROOT/'results/csafe_replication_2026-11-08'/asset.name)
        assert sha(asset)==sha(original)
        shutil.copy2(asset,figures/f'Figure_{n}.pdf')
    # The former graphical abstract described BanglaHand only; omit it here.
    shutil.rmtree(DEST/'optional',ignore_errors=True)
    external=ROOT/'results/csafe_replication_2026-11-08'
    for name in ['validation.json','feature_equivalence.json','calibration_objective_validation.json','independent_recompute_validation.json']:
        assert json.loads((external/name).read_text())['passed'],name
    externalcsv=SOURCE/'tables/CSAFE_complete_196_settings.csv'
    assert len(list(csv.DictReader(externalcsv.open())))==980
    shutil.copy2(externalcsv,DEST/externalcsv.name)
    repro=DEST/'reproducibility';repro.mkdir(exist_ok=True)
    for name in ['csafe-replication-2026-11-08.zip','csafe-replication-2026-11-08.zip.sha256']:
        shutil.copy2(ROOT/'submission/csafe_replication'/name,repro/name)
    report=dict(passed=True,date='9 November 2026',target='IET Biometrics',pdfs=pdfrows,
        main_figures=7,main_tables=8,main_equations=6,main_references=26,abstract_words=len(words),
        supplementary_tables=19,supplementary_figures=6,full_grid_rows=120,external_settings=196,external_method_rows=980,
        word_table_decimal_values_checked=word['table_decimal_values_checked'],
        original_prl_files_unchanged=True,current_figures_1_2_preserved=True,
        new_experiments=True,new_datasets=True,external_validation="CSAFE v2; 96 writers; separate capture sessions",original_title_and_methodological_framing_retained=True,
        literature_layout_revision_date='2026-11-09',new_experiments_in_literature_layout_revision=False,
        ethics_source='Author-supplied Data in Brief manuscript, page 6: institutional approval not required under SUST policy.',
        guide=GUIDE,template='Author-supplied WileyDesign.zip, USG class',
        submission_made=False,coauthor_approval_certified=False,orcid='To be provided by submitting author in the journal portal',
        pdf_visual_review='Every page rendered and reviewed. Standard kernel column output restored; heading spacing and float placement corrected without altering figure artwork.')
    (DEST/'Conversion_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (SOURCE/'CONVERSION_NOTES.md').write_text('''# IET Biometrics conversion

The main manuscript retains the approved title and controlled-framework contribution. Completed CSAFE validation is integrated as an external evaluation, without reframing the paper as an empirical study. The external extension introduces an independent cohort, source pages and capture sessions; auxiliary models are refitted within that cohort rather than transferred from BanglaHand. The original PRL files remain unchanged. The recent Abstract and limitations rewrite is retained; the removed dataset-release sentence is not reintroduced. Figures 1–7 keep their current artwork; Figure 1 retains its single-line legend. The 9 November revision adds a sourced related-work synthesis, a compact organization sentence, a literature summary table and a methodological comparison table. The template loads an older cuted output routine despite using no strip environments; the main preamble preserves the current LaTeX kernel routines for ordinary two-column output. Heading spacing, line leading and float placement are adjusted to remove avoidable gaps.

| Material | Location in the expanded main paper | Supporting detail retained |
|---|---|---|
| Literature synthesis and contribution comparison | Main Tables 1 and 8 | Primary-source audit in the revision notes |
| Strip construction, validity rules and descriptors | Materials and methods, Sections 3.1–3.2 | S1–S2 |
| Fitting hierarchy | Main Table 2 | S3, Table S2 |
| Exact random-partition expectation | Main Equation (5) | S4, Equation (S1) |
| Calibration and sensitivity predictors | Main Table 5 | S5, Table S6 and Figure S1 |
| Writer-overlap label controls | Main Figure 4 | S5, Figure S2 |
| Calibration-partition variability | Main Table 6 | S9, Table S12 |
| Temperature, learned weights and writer influence | Main Figure 5 | S9, Figure S4 |
| Decision margins and rescue/damage | Main Equation (6) and Figure 6 | S10, Figure S5 |
| External CSAFE protocol and primary findings | Sections 3.8 and 4.8, Table 7 and Figure 7 | S13, Tables S16–S19 and Figure S6 |

The CSAFE analysis uses 96 writers and separate capture sessions, with eight primary Holm-adjusted comparisons. Secondary conditions, all alternate-model results and numerical calibration diagnostics are retained. The separate reproducibility archive contains complete results, including 20 overlapping balanced gallery subsamples. They are descriptive dependent subsamples, not independent datasets.

Supporting information stays self-contained, preserving its S numbering. The complete 120-setting grid and reproduction details stay there; a CSV copy of the grid accompanies the submission.

The guide's proposed capture-ratio confidence intervals, post hoc equivalence testing, selective filtering, new attribute predictors and new deployment splits were not performed. Unrun proposals are not presented as results. The oracle benchmark is not interpreted as a causal decomposition or a universal claim that handedness lacks identity information.

The ethics declaration follows the author-supplied Data in Brief manuscript (page 6). It reports that institutional approval was not required under SUST policy for the voluntary, non-invasive collection; no committee approval or exemption reference has been invented. Consent, pseudonymous writer IDs and the authors-funded honorarium are retained. The user-requested AI statement remains verbatim.

The supplied USG class is preserved. The uppercase LETTERSP.STY support filename is normalized to lettersp.sty. Preamble patches remove the sample Allergy logo and production badge, suppress unassigned publication metadata, and ensure readable table typography and consistent review-page footers. The main preamble disables the class shipout geometry reset and relaxes wide-float placement; no forced float barrier interrupts the Discussion. The final reference columns are balanced with an explicit break before reference 14. ietnumeric.bst is an unsrtnat-derived submission style retaining citation order, using author initials and adding DOI output for dataset/preprint records. Word uses the provided numeric IEEE CSL style; both are permitted by IET's initial-submission reference policy.
''')
    readme=f'''# IET Biometrics submission package

Prepared on 9 November 2026. Main PDF: **{pdfrows[0]['pages']} pages**, with **7 figures, 8 tables and 26 references**. Supporting PDF: **{pdfrows[1]['pages']} pages**, separate. The single-column editable Word versions have different pagination.

## Files to use

- `Manuscript.docx` is the editable main submission. `Manuscript.pdf` is its Wiley-template typeset counterpart.
- `Supporting_information.pdf` / `.docx` contain the complete methods, controls and grid.
- `Cover_letter.pdf` / `.docx` are addressed to IET Biometrics.
- `figures/` contains seven vector main figures; `Complete_120_setting_grid.csv` contains the BanglaHand grid and `CSAFE_complete_196_settings.csv` contains 980 method rows across 196 external settings.
- `reproducibility/csafe-replication-2026-11-08.zip` contains the external code, frozen features, protocols, manifests and results. The primary-study graphical abstract is omitted because it does not summarize the external extension.
- `source/` contains matched Markdown and LaTeX exports, references, figure assets and build helpers. `source/paper/iet_biometrics/CONVERSION_NOTES.md` maps promoted material.

## Verified journal requirements

Checked the [official author instructions]({GUIDE}) on 9 November 2026: editable Word is accepted; LaTeX submissions also need a review PDF and supporting sources. Author identities remain visible for single-anonymized review. The Abstract is below 300 words. Numbered citations follow first appearance. Figures and editable tables are included, and supporting information is described before the references. Line numbers are present in the main files.

The submitting author must supply an ORCID in the portal and obtain final coauthor approval. This package does not certify those approvals or make a journal submission. The previously supplied PRL authorship form is not an IET form and is excluded.

## Build sources

From `source/paper/iet_biometrics/`, run:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supporting_information.tex
```

The supplied template depends on a normal TeX installation including STIX2, booktabs, tabularx, xurl, lineno and placeins. For Word, use Pandoc with `--citeproc`, `refs.bib` and `ieee.csl`; the helper in `source/code/build_iet_docx.py` applies the verified review styling. The Markdown contains explicit equation and caption numbers and needs no pandoc-crossref filter. LaTeX and Markdown are matched conversion exports; synchronize them after subsequent scientific edits.

The BanglaHand analyses, features and results are available in the [versioned reproducibility release](https://github.com/LegendaryBeast/Bangla-writer-identification-research/releases/tag/v1.6.0). Current plot-styling scripts accompany these submission sources. Raw dataset images and checkpoint binaries are excluded. Obtain BanglaHand data from doi:10.17632/n4sjstdzfy.1 and CSAFE v2 from doi:10.25380/iastate.10062203.v2. The supplied CSAFE archive is separate from public release v1.6.0; it includes the original protocol and the objective-preserving calibration implementation amendment.

See `Conversion_validation.json`, `Layout_revision_validation.json`, `Word_validation.json`, `Independent_build_validation.json` and `SHA256SUMS.txt` for verification. Both LaTeX documents were rebuilt from an extracted copy of this package, and every page's text matched the delivered PDF. These checks establish conversion consistency, not independent-dataset validation or journal acceptance.
'''
    (DEST/'README.md').write_text(readme)
    shutil.rmtree(DEST/'source',ignore_errors=True)
    source_dest=DEST/'source/paper/iet_biometrics';source_dest.mkdir(parents=True,exist_ok=True)
    for p in SOURCE.iterdir():
        if p.is_file() and p.suffix in ['.tex','.md','.bib','.bst','.cls','.sty','.csl','.json','.bbl']:
            shutil.copy2(p,source_dest/p.name)
    for directory in ['Fonts','tables']:
        shutil.copytree(SOURCE/directory,source_dest/directory,dirs_exist_ok=True)
    (source_dest/'assets').mkdir(exist_ok=True)
    used={name for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',maintex+suptex) if name.startswith('assets/')}
    for asset in used:
        for suffix in ['.pdf','.png']:
            p=SOURCE/asset
            p=p.with_suffix(suffix)
            shutil.copy2(p,source_dest/'assets'/p.name)
    (source_dest/'images').mkdir(exist_ok=True)
    for name in ['Wiley_logo.eps','Wiley_logo-eps-converted-to.pdf']:
        shutil.copy2(SOURCE/'images'/name,source_dest/'images'/name)
    code_dest=DEST/'source/code';code_dest.mkdir(parents=True,exist_ok=True)
    for name in ['build_iet_docx.py','make_revision_paper.py','make_representation_utility.py','integrate_csafe_iet.py','summarize_csafe_replication.py','verify_iet_build.py','revise_iet_literature_layout.py']:
        shutil.copy2(ROOT/'code'/name,code_dest/name)
    # Author conversion helpers are included for traceability; direct LaTeX and
    # Markdown builds above do not require the original PRL manuscript files.
    for name in ['convert_iet_biometrics.py','finalize_iet_conversion.py']:
        shutil.copy2(ROOT/'code'/name,code_dest/name)
    notes_dest=DEST/'source/paper/notes';notes_dest.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'paper/notes/LITERATURE_LAYOUT_REVISION_2026-11-09.md',notes_dest/'LITERATURE_LAYOUT_REVISION_2026-11-09.md')
    archive=DEST/'wid-iet-biometrics-submission-2026-11-09.zip'
    files=[p for p in DEST.rglob('*') if p.is_file() and p!=archive and p!=archive.with_suffix('.zip.sha256') and p.name!='SHA256SUMS.txt' and not (p.parent==DEST and (p.suffix=='.zip' or p.name.endswith('.zip.sha256')))]
    assert not any(p.suffix.lower() in ['.jpg','.jpeg','.tif','.tiff'] for p in files)
    sums=''.join(f'{sha(p)}  {p.relative_to(DEST)}\n' for p in sorted(files))
    (DEST/'SHA256SUMS.txt').write_text(sums)
    archive=DEST/'wid-iet-biometrics-submission-2026-11-09.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[DEST/'SHA256SUMS.txt']:z.write(p,str(p.relative_to(DEST)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for line in z.read('SHA256SUMS.txt').decode().splitlines():
            digest,name=line.split('  ',1);assert hashlib.sha256(z.read(name)).hexdigest()==digest
    archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print(json.dumps(report,indent=2))
    print('Archive:',archive,'bytes:',archive.stat().st_size)


if __name__=='__main__':main()
