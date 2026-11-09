#!/usr/bin/env python3
"""Build editable IET review documents from the conversion Markdown sources.

Run with the Codex bundled Python runtime. Scientific prose, equations and
tables are passed through Pandoc; OOXML edits here apply review formatting.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import zipfile
from decimal import Decimal
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'paper/iet_biometrics'
DEST = ROOT / 'submission/iet_biometrics'
PANDOC = shutil.which('pandoc') or '/opt/homebrew/bin/pandoc'
QA = Path('/private/tmp/wid-iet-word-qa')


def source_cell_text(node):
    if isinstance(node,list):
        return ''.join(source_cell_text(x) for x in node)
    if isinstance(node,dict):
        kind=node.get('t')
        if kind=='Str':
            return node['c']
        if kind in ['Math','Code']:
            return node['c'][1]
        if kind in ['Space','SoftBreak','LineBreak']:
            return ' '
        return source_cell_text(node.get('c',[]))
    return ''


def source_tables(node):
    if isinstance(node,list):
        out=[]
        for x in node:
            out.extend(source_tables(x))
        return out
    if isinstance(node,dict):
        if node.get('t')=='Table':
            return [node]
        return source_tables(node.get('c',[]))
    return []


def decimals(text):
    text=text.replace('\u2212','-').replace('\u2013','-')
    return [Decimal(x) for x in re.findall(r'[+-]?\d+\.\d+',text)]


def check_table_values(doc, source):
    ast_path=SOURCE/source.replace('.md','.json')
    if not ast_path.exists():
        return 0
    ast=json.loads(ast_path.read_text())
    tables=source_tables(ast['blocks'])
    assert len(tables)==len(doc.tables),('Table count changed',len(tables),len(doc.tables))
    checked=0
    for i,(node,table) in enumerate(zip(tables,doc.tables),1):
        c=node['c'];rows=list(c[3][1])
        for body in c[4]:
            rows.extend(body[2]);rows.extend(body[3])
        rows.extend(c[5][1])
        source_text=' '.join(source_cell_text(cell[4]) for row in rows for cell in row[1])
        seen=set();actual=[]
        for row in table.rows:
            for cell in row.cells:
                if cell._tc in seen:
                    continue
                seen.add(cell._tc)
                actual.append(''.join(cell._tc.xpath('.//w:t/text()|.//m:t/text()')))
        expected=decimals(source_text);observed=decimals(' '.join(actual))
        assert expected==observed,('Decimal table content differs',i,expected,observed)
        checked+=len(expected)
    return checked


def font(style, size, *, bold=None, italic=None):
    style.font.name = 'Times New Roman'
    style.font.size = Pt(size)
    style.font.color.rgb = RGBColor(0, 0, 0)
    if bold is not None:
        style.font.bold = bold
    if italic is not None:
        style.font.italic = italic
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn('w:rFonts'))
    if fonts is None:
        fonts = OxmlElement('w:rFonts')
        rpr.insert(0, fonts)
    for attr in ('ascii', 'hAnsi', 'eastAsia', 'cs'):
        fonts.set(qn('w:' + attr), 'Times New Roman')
    for attr in ('asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme'):
        fonts.attrib.pop(qn('w:' + attr), None)
    color = rpr.find(qn('w:color'))
    if color is not None:
        for attr in ('themeColor', 'themeShade', 'themeTint'):
            color.attrib.pop(qn('w:' + attr), None)


def reference(path: Path):
    doc = Document()
    for name in ['First Paragraph','Compact','Author','Date','Table Caption','Image Caption','Bibliography','Source Code']:
        if name not in doc.styles:
            created = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            created.base_style = doc.styles['Normal']
    for section in doc.sections:
        section.page_width = Cm(21)
        section.page_height = Cm(29.7)
        section.top_margin = section.bottom_margin = Cm(2)
        section.left_margin = section.right_margin = Cm(2)
        section.header_distance = section.footer_distance = Cm(.8)
    for name in ['Normal', 'Body Text', 'First Paragraph', 'Compact']:
        style = doc.styles[name]
        font(style, 11)
        pf = style.paragraph_format
        pf.line_spacing = 1.12
        pf.space_before = Pt(0)
        pf.space_after = Pt(4)
        pf.widow_control = True
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for i, size in [(1, 13), (2, 11.5), (3, 11), (4, 11)]:
        style = doc.styles[f'Heading {i}']
        font(style, size, bold=True)
        pf = style.paragraph_format
        pf.space_before = Pt(11 if i == 1 else 8)
        pf.space_after = Pt(4)
        pf.keep_with_next = True
        pf.keep_together = True
        pf.page_break_before = False
        pf.line_spacing = 1
    font(doc.styles['Title'], 16, bold=True)
    for name in ['Title','Subtitle']:
        ppr=doc.styles[name].element.pPr
        if ppr is not None:
            for node in list(ppr):
                if node.tag==qn('w:pBdr'):
                    ppr.remove(node)
    doc.styles['Title'].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.styles['Title'].paragraph_format.space_after = Pt(9)
    doc.styles['Title'].paragraph_format.keep_with_next = True
    for name in ['Subtitle', 'Author', 'Date']:
        if name in doc.styles:
            font(doc.styles[name], 11)
            doc.styles[name].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            doc.styles[name].paragraph_format.space_after = Pt(4)
    for name in ['Caption', 'Table Caption', 'Image Caption']:
        if name in doc.styles:
            font(doc.styles[name], 9.5)
            doc.styles[name].paragraph_format.line_spacing = 1
            doc.styles[name].paragraph_format.space_after = Pt(5)
            doc.styles[name].paragraph_format.space_before = Pt(4)
    code=doc.styles['Source Code']
    font(code,9)
    code.font.name='Consolas'
    cf=code.element.get_or_add_rPr().find(qn('w:rFonts'))
    for attr in ('ascii','hAnsi','eastAsia','cs'):
        cf.set(qn('w:'+attr),'Consolas')
    code.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.LEFT
    code.paragraph_format.line_spacing=1.0
    code.paragraph_format.space_before=Pt(4)
    code.paragraph_format.space_after=Pt(6)
    if 'Hyperlink' in doc.styles:
        doc.styles['Hyperlink'].font.color.rgb = RGBColor(0, 0, 0)
        doc.styles['Hyperlink'].font.underline = False
    doc.save(path)


def border(parent, name, val='nil', *, size=5, color='555555'):
    tag = qn('w:' + name)
    node = parent.find(tag)
    if node is None:
        node = OxmlElement('w:' + name)
        parent.append(node)
    node.set(qn('w:val'), val)
    if val != 'nil':
        node.set(qn('w:sz'), str(size))
        node.set(qn('w:color'), color)


def table_format(table, available):
    ncols = len(table.columns)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    props = table._tbl.tblPr
    borders = props.find(qn('w:tblBorders'))
    if borders is None:
        borders = OxmlElement('w:tblBorders')
        props.append(borders)
    for edge in ['left', 'right', 'insideV', 'insideH']:
        border(borders, edge)
    for edge in ['top', 'bottom']:
        border(borders, edge, 'single', size=6, color='333333')
    lengths = [3] * ncols
    for row in table.rows:
        for j, cell in enumerate(row.cells):
            span = cell._tc.tcPr.find(qn('w:gridSpan'))
            if span is None or span.get(qn('w:val')) == '1':
                text = cell.text.replace('\n', ' ')
                lengths[j] = max(lengths[j], min(32, len(text)))
    weights = [math.sqrt(n+2) for n in lengths]
    if ncols == 2 and max((len(row.cells[1].text) for row in table.rows), default=0) > 85:
        weights = [.25, .75]
    widths = [int(available*w/sum(weights)) for w in weights]
    for j, col in enumerate(table.columns):
        col.width = widths[j]
    for j, grid in enumerate(table._tbl.tblGrid):
        grid.set(qn('w:w'), str(round(widths[j] / 635)))
    seen = set()
    for ri, row in enumerate(table.rows):
        trpr = row._tr.get_or_add_trPr()
        if trpr.find(qn('w:cantSplit')) is None:
            trpr.append(OxmlElement('w:cantSplit'))
        if ri == 0 and trpr.find(qn('w:tblHeader')) is None:
            trpr.append(OxmlElement('w:tblHeader'))
        for ci, cell in enumerate(row.cells):
            if cell._tc in seen:
                continue
            seen.add(cell._tc)
            tcpr = cell._tc.get_or_add_tcPr()
            span_node = tcpr.find(qn('w:gridSpan'))
            span = int(span_node.get(qn('w:val'))) if span_node is not None else 1
            cell.width = sum(widths[ci:ci+span])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            margin = tcpr.find(qn('w:tcMar'))
            if margin is None:
                margin = OxmlElement('w:tcMar')
                tcpr.append(margin)
            for edge, val in [('top', 45), ('bottom', 45), ('left', 55), ('right', 55)]:
                node = margin.find(qn('w:' + edge))
                if node is None:
                    node = OxmlElement('w:' + edge)
                    margin.append(node)
                node.set(qn('w:w'), str(val))
                node.set(qn('w:type'), 'dxa')
            cb = tcpr.find(qn('w:tcBorders'))
            if cb is not None:
                tcpr.remove(cb)
            if ri == 0:
                cb = OxmlElement('w:tcBorders')
                border(cb, 'bottom', 'single', size=4)
                tcpr.append(cb)
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(0)
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1
                p.paragraph_format.keep_with_next = False
                if len(table.rows) <= 10 and ri < len(table.rows)-1:
                    p.paragraph_format.keep_with_next = True
                elif len(table.rows) > 10 and ri < 2:
                    # Keep the header with at least two data rows.
                    p.paragraph_format.keep_with_next = True
                p.paragraph_format.keep_together = True
                narrative = len(cell.text) > 45 or len(cell.text.split()) > 7
                p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if ci == 0 or narrative
                               else WD_ALIGN_PARAGRAPH.CENTER)
                for run in p.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(8.5 if ncols >= 9 else 9)
                    run.font.color.rgb = RGBColor(0, 0, 0)
                    if ri == 0:
                        run.font.bold = True


def patch(path: Path, *, source, main=False, cover=False):
    doc = Document(path)
    for section in doc.sections:
        section.page_width, section.page_height = Cm(21), Cm(29.7)
        section.top_margin = section.bottom_margin = Cm(2)
        section.left_margin = section.right_margin = Cm(2)
        section.header_distance = section.footer_distance = Cm(.8)
        if main:
            numbering = OxmlElement('w:lnNumType')
            numbering.set(qn('w:countBy'), '5')
            numbering.set(qn('w:start'), '1')
            numbering.set(qn('w:distance'), '180')
            numbering.set(qn('w:restart'), 'continuous')
            section._sectPr.append(numbering)
        footer = section.footer.paragraphs[0]
        footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = footer.add_run()
        run.font.size = Pt(9)
        begin = OxmlElement('w:fldChar');begin.set(qn('w:fldCharType'), 'begin')
        instruction = OxmlElement('w:instrText');instruction.text = ' PAGE '
        end = OxmlElement('w:fldChar');end.set(qn('w:fldCharType'), 'end')
        run._r.extend([begin, instruction, end])
    available = doc.sections[0].page_width-doc.sections[0].left_margin-doc.sections[0].right_margin
    for shape in doc.inline_shapes:
        if shape.width > available:
            ratio = available / shape.width
            shape.width = int(shape.width*ratio)
            shape.height = int(shape.height*ratio)
    paragraphs = doc.paragraphs
    front = main
    for i,p in enumerate(paragraphs):
        text = p.text.strip()
        p.paragraph_format.widow_control = True
        if front and text == 'Abstract':
            front = False
        if front:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if p.style.name == 'Title' and cover:
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        if p.style.name in ['Title','Subtitle']:
            ppr=p._p.get_or_add_pPr()
            for node in list(ppr):
                if node.tag==qn('w:pBdr'):
                    ppr.remove(node)
        if p.style.name.startswith('Heading'):
            for run in p.runs:
                run.font.color.rgb = RGBColor(0,0,0)
        if p.style.name in ['Bibliography','References'] or p._p.get('custom-style') == 'Bibliography':
            p.paragraph_format.left_indent = Inches(.22)
            p.paragraph_format.first_line_indent = Inches(-.22)
            p.paragraph_format.keep_together = True
            p.paragraph_format.line_spacing = 1
            p.paragraph_format.space_after = Pt(2)
            for run in p.runs:
                run.font.size = Pt(10)
        if p._p.xpath('.//w:drawing'):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.paragraph_format.keep_together = True
            p.paragraph_format.space_before = Pt(5)
        if p.style.name in ['Image Caption','Caption'] and not text.startswith('Table'):
            p.paragraph_format.keep_together = True
            p.paragraph_format.keep_with_next = False
        if p.style.name == 'Table Caption' or text.startswith(('Table 1.', 'Table 2.', 'Table S')):
            p.paragraph_format.keep_together = True
            p.paragraph_format.keep_with_next = True
        if cover:
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.line_spacing = 1.06
            p.paragraph_format.space_after = Pt(7)
    for table in doc.tables:
        table_format(table, available)
    doc.core_properties.author = 'Tanzim Hasan Prappo; M. Shahidur Rahman'
    doc.core_properties.last_modified_by = ''
    doc.core_properties.comments = ''
    doc.save(path)
    checked=check_table_values(doc,source)
    with zipfile.ZipFile(path) as z:
        xml = z.read('word/document.xml').decode()
        assert not re.search(r'TODO|TBD|PLACEHOLDER|@@|\?\?', xml)
        math_count = xml.count('<m:oMath>') + xml.count('<m:oMathPara>')
        stats = {'file': str(path.relative_to(ROOT)), 'paragraphs': len(doc.paragraphs),
                 'tables': len(doc.tables), 'images': len(doc.inline_shapes),
                 'native_math_nodes': math_count, 'line_numbering': main}
        stats['table_decimal_values_checked']=checked
        return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['main','supplement','cover'])
    args = ap.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    QA.mkdir(parents=True,exist_ok=True)
    ref = QA/'reference.docx'
    reference(ref)
    specs=[('main','manuscript.md','Manuscript.docx'),
           ('supplement','supporting_information.md','Supporting_information.docx'),
           ('cover','cover_letter.md','Cover_letter.docx')]
    stats=[]
    for key,source,target in specs:
        if args.only and key != args.only:
            continue
        cmd=[PANDOC,source,'--from','markdown+tex_math_dollars+pipe_tables+grid_tables',
             '--to','docx','--reference-doc',str(ref),'--resource-path',str(SOURCE),
             '--output',str(DEST/target)]
        if key != 'cover':
            cmd += ['--citeproc','--bibliography','refs.bib','--csl','ieee.csl']
        subprocess.run(cmd,cwd=SOURCE,check=True)
        stats.append(patch(DEST/target,source=source,main=key=='main',cover=key=='cover'))
    (QA/'structural_checks.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(json.dumps(stats,indent=2))


if __name__ == '__main__':
    main()
