#!/usr/bin/env python3
"""Audit and package final PRL materials without retraining or submitting."""
import hashlib
import json
import re
import shutil
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import fitz
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'submission/prl_final'
BASE=ROOT/'results/wid_revision_2026-11-08'
EXTRA=ROOT/'results/wid_fusion_robustness_2026-11-08'


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    report=json.loads((BASE/'validation_report.json').read_text())
    assert report['passed'] and len(report['pdfs'])==2
    for r in report['pdfs']:assert digest(ROOT/r['file'])==r['sha256'],'Outdated document validation'
    additional=json.loads((EXTRA/'additional_validation_report.json').read_text())
    assert additional['passed']
    additional_checks=additional['numerical_and_scope_checks']
    prep=json.loads((DEST/'Preparation_validation.json').read_text())
    assert prep['passed']
    for name,expected in prep['source_sha256'].items():assert digest(ROOT/name)==expected,'Outdated submission material input: '+name
    for name,expected in prep['files'].items():assert digest(ROOT/name)==expected,'Submission file changed after preparation: '+name
    macros={}
    for p in (ROOT/'paper/figures_revision').glob('*macros.tex'):
        macros.update(dict(re.findall(r'\\newcommand\{\\(r\w+)\}\{([^{}]*)\}',p.read_text())))
    claims=[]
    def check(macro,value,source,location):
        displayed=macros[macro]
        decimals=len(displayed.partition('.')[2])
        expected=float(f'{value:.{decimals}f}')
        assert float(displayed)==expected,(macro,displayed,value)
        claims.append(dict(macro=macro,displayed=displayed,unrounded=value,source=source,location=location))
    E=pd.read_csv(BASE/'checks_effect_intervals.csv')
    for prefix,contrast in [('Known','known_minus_none'),('Fusion','query_bayes_minus_none'),
                             ('Filter','query_filter_minus_none'),('OracleVsRandom','known_hand_minus_random_partition_expectation')]:
        r=E[(E.scope=='all_120_settings')&(E.contrast==contrast)].iloc[0]
        for suffix,col in [('Gain','estimate'),('Lo','ci_lo'),('Hi','ci_hi')]:
            macro='r'+prefix+suffix if prefix!='OracleVsRandom' or suffix!='Gain' else 'rOracleVsRandom'
            check(macro,100*r[col],'checks_effect_intervals.csv','Main results / Supplement S4')
    r=E[(E.scope=='all_120_settings')&(E.contrast=='known_random_expected_minus_none')].iloc[0]
    check('rRandomKnownGain',100*r.estimate,'checks_effect_intervals.csv','Abstract / Main results / Supplement S4')
    N=pd.read_csv(EXTRA/'nested_effect_intervals.csv')
    for prefix,contrast in [('Nested','nested_minus_none'),('NestedVsFixed','nested_minus_fixed')]:
        r=N[(N.scope=='all_120_settings')&(N.metric=='top1')&(N.contrast==contrast)].iloc[0]
        for suffix,col in [('Gain','estimate'),('Lo','ci_lo'),('Hi','ci_hi')]:
            check('r'+prefix+suffix,100*r[col],'nested_effect_intervals.csv','Abstract / Main results / Supplement S8')
    H=pd.read_csv(BASE/'handedness_accuracy.csv');r=H[(H.hand_model=='hc183')&(H.n==1)].iloc[0]
    for macro,col in [('rHandShared','shared'),('rHandOwn','own'),('rHandWriter','writer_all_units')]:
        check(macro,100*r[col],'handedness_accuracy.csv','Main results / Supplement S5 and full numerical outputs')
    r=pd.read_csv(BASE/'checks_both_own_effect.csv').iloc[0]
    for suffix,col in [('Gain','estimate'),('Lo','ci_lo'),('Hi','ci_hi')]:
        check('rBoth'+suffix,100*r[col],'checks_both_own_effect.csv','Main exploratory result / Supplement S6')
    M=pd.read_csv(EXTRA/'margin_effect_intervals.csv')
    for method,prefix in [('fixed','Fixed'),('nested','Nested')]:
        for metric,suffix in [('rescued','Rescued'),('damaged','Damaged')]:
            r=M[(M.method==method)&(M.metric==metric)].iloc[0]
            check('r'+prefix+suffix,100*r.estimate,'margin_effect_intervals.csv','Main Discussion / Supplement S10')
    values=pd.read_csv(DEST/'assets/Graphical_abstract_values.csv')
    for r in values.itertuples():
        frame=N[(N.scope=='all_120_settings')&(N.metric=='top1')] if r.contrast=='nested_minus_none' else E[E.scope=='all_120_settings']
        ref=frame[frame.contrast==r.contrast].iloc[0]
        assert np.allclose([r.estimate,r.ci_lo,r.ci_hi],[ref.estimate,ref.ci_lo,ref.ci_hi],atol=2e-14,rtol=0)
    pd.DataFrame(claims).to_csv(DEST/'Numerical_claim_crosscheck.csv',index=False)
    # Compare every primary setting in the printed main table and complete supplement.
    cells=pd.read_csv(BASE/'id_summary.csv')
    assert cells[cells.scenario=='none'][['rep','query','n','k']].drop_duplicates().shape[0]==120
    assert (cells.n_writers==42).all()
    names={'STROKE':'stroke','HC-183':'hc183','ResNet-18':'resnet18','EffNet-B0':'efficientnet_b0',
           'ViT-B/16':'vit_b_16','Stroke-CNN':'stroke_cnn'}
    table_checks=[]
    def texrows(name):
        return [[x.strip() for x in line.removesuffix('\\\\').strip().split('&')]
                for line in (ROOT/'paper/figures_revision'/name).read_text().splitlines()
                if '&' in line and not line.lstrip().startswith('%')]
    def cellcheck(table,row,column,text,expected,digits):
        assert float(text)==float(f'{expected:.{digits}f}'),(table,row,column,text,expected)
        table_checks.append(dict(table=table,row=row,column=column,displayed=text,unrounded=expected))
    def accuracy(rep,query,n,k,scenario,hand,method):
        r=cells[(cells.rep==rep)&(cells['query']==query)&(cells.n==n)&(cells.k==k)&
                (cells.scenario==scenario)&(cells.hand_model==hand)&(cells.method==method)]
        assert len(r)==1
        return r.top1.iloc[0]
    methods=[('none','-','none'),('oracle','-','filter'),('query','hc183','bayes'),('query','hc183','filter')]
    mainrows=texrows('table_main.tex');assert len(mainrows)==6
    for row in mainrows:
        rep=names[row[0]]
        expected=[100*accuracy(rep,'own',n,k,*method) for n,k in [(1,1),(4,6)] for method in methods]
        for j,(text,value) in enumerate(zip(row[1:],expected)):
            cellcheck('Main Table 1',rep,j,text,value,1)
    learned=pd.read_csv(EXTRA/'nested_summary.csv').set_index(['rep','query','n','k','method']).top1
    grid=texrows('table_all_settings_nested.tex');assert len(grid)==120
    seen=set()
    for row in grid:
        rep,query,n,k=names[row[0]],row[1],int(row[2]),int(row[3]);key=(rep,query,n,k);assert key not in seen;seen.add(key)
        plain,known,fixed,filtered=[accuracy(*key,*m) for m in methods]
        expected=[plain,known,fixed,learned.loc[key+('nested',)],filtered,plain/known]
        for j,(text,value) in enumerate(zip(row[4:],expected)):
            cellcheck('Supplement Table S15',str(key),j,text,100*value,1)
    fixed_tests=pd.read_csv(BASE/'id_paired_tests.csv')
    nested_tests=pd.read_csv(EXTRA/'nested_paired_tests.csv')
    designated=texrows('table_nested_designated.tex');assert len(designated)==4
    for row in designated:
        rep=names[row[1]];n=1 if row[0].startswith('Paragraph') else 4;k=1 if n==1 else 6
        f=fixed_tests[(fixed_tests.rep==rep)&(fixed_tests['query']=='own')&(fixed_tests.n==n)&(fixed_tests.k==k)&
                      (fixed_tests.scenario=='query')&(fixed_tests.hand_model=='hc183')&(fixed_tests.method=='bayes')].iloc[0]
        d=nested_tests[(nested_tests.rep==rep)&(nested_tests['query']=='own')&(nested_tests.n==n)&(nested_tests.k==k)]
        a=d[d.contrast=='nested_minus_none'].iloc[0];b=d[d.contrast=='nested_minus_fixed'].iloc[0]
        for col,ref in [(2,[f.diff_top1,f.ci_lo,f.ci_hi]),(3,[a.estimate,a.ci_lo,a.ci_hi]),(4,[b.estimate,b.ci_lo,b.ci_hi])]:
            texts=re.findall(r'[+-]?\d+\.\d+',row[col]);assert len(texts)==3
            for j,(text,value) in enumerate(zip(texts,ref)):
                cellcheck('Main Table 2',str((rep,n,k)),f'{col}:{j}',text,100*value,2)
        for col,value in [(5,a.p_holm),(6,b.p_holm)]:cellcheck('Main Table 2',str((rep,n,k)),col,row[col],value,3)
    pd.DataFrame(table_checks).to_csv(DEST/'Table_cell_crosscheck.csv',index=False)
    bib=(ROOT/'paper/wid_manuscript.bbl').read_text()
    entries=set(re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}',bib))
    maintex=(ROOT/'paper/wid_manuscript.tex').read_text()
    cites={k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}',maintex) for k in group.split(',')}
    assert entries==cites,(entries-cites,cites-entries)
    assert len(cites)==23
    for stem,target,expected in [('wid_manuscript','Manuscript.pdf',7),('wid_supplement','Supplementary_material.pdf',14)]:
        p=ROOT/'paper'/f'{stem}.pdf';d=fitz.open(p);assert len(d)==expected
        shutil.copy2(p,DEST/'upload'/target)
    figures=re.findall(r'\\begin\{figure\*\}(.*?)\\end\{figure\*\}',maintex,re.S)
    assert len(figures)==3
    captions=[]
    for i,block in enumerate(figures,1):
        asset=re.search(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}',block).group(1)
        shutil.copy2(ROOT/'paper'/asset,DEST/'upload'/f'Figure_{i}.pdf')
        start=block.index('\\caption{')+len('\\caption{');depth=1;end=start
        while depth:
            if block[end]=='{' and (end==0 or block[end-1]!='\\'):depth+=1
            if block[end]=='}' and (end==0 or block[end-1]!='\\'):depth-=1
            end+=1
        text=block[start:end-1].replace('\\%','%').replace('~',' ')
        captions.append(f'Figure {i}. '+text)
    (DEST/'upload/Figure_captions.txt').write_text('\n\n'.join(captions)+'\n')
    fields=PdfReader(DEST/'upload/Authorship_confirmation_form.pdf').get_fields()
    assert fields['signature'].get('/V','')==fields['signature_date'].get('/V','')==''
    assert fields['manuscript_preprints'].get('/V','')=='None for this manuscript.'
    assert fields['conference_history'].get('/V','')=='None.'
    status=json.loads((DEST/'source/Publication_status.json').read_text())
    audit=dict(passed=True,version='1.6.0',main_pages=7,supplement_pages=14,abstract_words=report['abstract_words'],
        explicit_numeric_claim_checks=len(claims),graphical_abstract_values_checked=5,
        printed_table_cells_checked=len(table_checks),
        scientific_and_document_checks=report['checks'],additional_scientific_checks=additional_checks,
        bibliography_entries=len(entries),extra_experiments_run=False,extra_datasets_added=False,
        author_history='Confirmed by user: no history for this manuscript; distinct companion disclosed.',
        coauthor_approval=status.get('coauthor_approval'),signature='blank; to be completed by author',
        data_link='https://data.mendeley.com/datasets/n4sjstdzfy/1',companion_link='https://ssrn.com/abstract=7662278',
        guide='https://www.sciencedirect.com/journal/pattern-recognition-letters/publish/guide-for-authors',
        guide_checked='8 November 2026 via browser; graphical abstract required, >=1328 pixels wide; separate file.',
        scope='Numerical consistency, metadata, package integrity and document readiness; author approvals are not certified.')
    (DEST/'Technical_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    audit_md=f'''# Final technical audit — 8 November 2026

Version 1.6.0. Main manuscript: **7 pages**. Supplement: **14 pages**, separate. Abstract: **{report['abstract_words']} words**.

The final revision changes framing, figure presentation and submission materials only. No additional experiment, participant or dataset was introduced. All 42 writers remain in the 120 primary settings.

## Numerical consistency

- {len(claims)} explicit checks compare displayed main-text effects, intervals, handedness accuracies, both-sided gains and rescue/damage rates against the numerical outputs that supply the supplement.
- All five graphical-abstract estimates and interval endpoints match those same source outputs.
- {len(table_checks)} printed cells in both main tables and the full 120-setting supplementary grid match the numerical source values at the displayed precision.
- The existing integrity audit independently reconstructs writer-level outcomes, comparisons, filtering identities and representation summaries; the additional nested/robustness audit passes {additional_checks:,} checks.
- Main and supplementary tables are generated from the same revised summaries; the complete supplementary grid includes fixed and nested fusion for every setting. Shared macros prevent separate text copies from drifting.
- The exploratory +1.08-point result uses only the 96 own-text settings. It is not replaced with the invalid historical all-setting both-sided comparison.

Detailed mappings: `Numerical_claim_crosscheck.csv` and `Table_cell_crosscheck.csv`. Machine-readable report: `Technical_audit.json`. These are technical checks, not a zero-error certification or independent-cohort validation.

## Documents and references

All 23 compiled bibliography entries are cited, and all in-text references resolve. The dataset record was rechecked at https://data.mendeley.com/datasets/n4sjstdzfy/1 (doi:10.17632/n4sjstdzfy.1; version 1, CC BY 4.0). The companion SSRN record, DOI 10.2139/ssrn.7662278, was verified in the browser during this revision session and its complete PDF was inspected in the earlier revision. Its overlapping cohort and different endpoint are disclosed in the article and cover letter.

Both retained authors and institutional emails match the main paper, PDF metadata and submission materials. The exact user-requested AI disclosure, CRediT roles, consent, funding, competing-interests and data/code declarations are retained. Rendered pages and artwork were visually inspected separately from numeric validation.

## Current PRL requirements

Checked the live guide on 8 November 2026: https://www.sciencedirect.com/journal/pattern-recognition-letters/publish/guide-for-authors

The mandatory submission instructions specify double columns and at most seven pages including references. Both institutional emails appear on the first manuscript page. Five separate highlights satisfy the 85-character limit. The current guide requires a separate graphical abstract, despite the older template calling it optional. Vector PDF plus TIFF/PNG/SVG alternatives are supplied, with a raster width exceeding 1,328 pixels. The authorship form uses the official wording and the layout of the supplied previous form.

## Author actions still required

The user confirmed no preprint or submission history for this exact manuscript; the previously supplied preprint is distinct companion work. Those manuscript-specific fields are completed. The previous paper's signature and date were not reused. The corresponding author must obtain final coauthor approval, verify the declarations, and sign/date the form before upload. No journal submission has been made.
'''
    (DEST/'Technical_audit.md').write_text(audit_md)
    readme='''# PRL final submission materials — version 1.6.0

## Upload files

| File in `upload/` | Intended item |
|---|---|
| `Manuscript.pdf` | Main manuscript, seven pages |
| `Supplementary_material.pdf` | Separate supplementary methods/results, fourteen pages |
| `Graphical_abstract.pdf` | Separate graphical abstract (preferred vector file) |
| `Highlights.txt` | Separate research highlights |
| `Cover_letter.pdf` | Editor-facing cover letter |
| `Authorship_confirmation_form.pdf` | Authorship confirmation; PRL guide specifies Cover Letter item type |
| `Figure_1.pdf`, `Figure_2.pdf`, `Figure_3.pdf` | Separate vector files for the three main figures |
| `Figure_captions.txt` | Figure captions, also present in the manuscript |

The authorship form is interactive and unsigned. Its manuscript-specific history fields are completed from the user's confirmation. Verify all declarations, obtain M. Shahidur Rahman's final approval, and sign/date the form before submission. The previous paper's signature was not copied. This package does not certify approval or submit anything.

`assets/` contains graphical-abstract PNG/TIFF previews, editable SVG, numeric values and a caption. Upload one graphical-abstract version, preferably the vector PDF. `source/` contains editable cover-letter text and the recorded history status. The scientific reproducibility ZIP is publicly available at https://github.com/LegendaryBeast/Bangla-writer-identification-research/releases/tag/v1.6.0. The confidential main-manuscript source remains in `paper/wid_manuscript.tex` in the local workspace; it is not included in the public release.

See `Technical_audit.md`, `Technical_audit.json` and `Numerical_claim_crosscheck.csv` for the audit. Every archived file is covered by SHA256SUMS.txt.

The current guide was checked on 8 November 2026: https://www.sciencedirect.com/journal/pattern-recognition-letters/publish/guide-for-authors
'''
    (DEST/'README.md').write_text(readme)
    files=[p for p in DEST.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name!='SHA256SUMS.txt']
    hashes=''.join(f'{digest(p)}  {p.relative_to(DEST)}\n' for p in sorted(files))
    (DEST/'SHA256SUMS.txt').write_text(hashes)
    archive=DEST/'wid-prl-submission-2026-11-08-v1.6.0.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files+[DEST/'SHA256SUMS.txt']:z.write(p,str(p.relative_to(DEST)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for line in z.read('SHA256SUMS.txt').decode().splitlines():
            h,name=line.split('  ',1);assert hashlib.sha256(z.read(name)).hexdigest()==h
    print(json.dumps(audit,indent=2));print('Submission archive:',archive,'bytes:',archive.stat().st_size)


if __name__=='__main__':main()
