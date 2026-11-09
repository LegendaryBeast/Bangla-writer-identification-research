#!/usr/bin/env python3
"""Prepare PRL submission artwork, a cover letter and an unsigned editable form.

No model fitting, external upload, signature copying or approval attestation.
The graphical abstract reads the same numerical outputs as the manuscript.
"""
import hashlib
import json
import shutil
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd

from make_revision_paper import ROOT
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.font_manager import findfont, FontProperties
from PIL import Image
import fitz
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether

DEST=ROOT/'submission/prl_final'
UPLOAD=DEST/'upload'
ASSETS=DEST/'assets'
SOURCE=DEST/'source'
TITLE='Handedness for Bangla writer identification: Oracle gains and predicted-attribute utility'
AUTHORS='Tanzim Hasan Prappo and M. Shahidur Rahman'
EMAIL='2021331006@student.sust.edu'
COMPANION='10.2139/ssrn.7662278'
DATASET='10.17632/n4sjstdzfy.1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def effect_data():
    base=ROOT/'results/wid_revision_2026-11-08/checks_effect_intervals.csv'
    nested=ROOT/'results/wid_fusion_robustness_2026-11-08/nested_effect_intervals.csv'
    E=pd.read_csv(base);N=pd.read_csv(nested)
    rows=[]
    for label,contrast,color in [('Known query hand','known_minus_none','#19846D'),
                                  ('Known random group','known_random_expected_minus_none','#6D817D'),
                                  ('Fixed fusion','query_bayes_minus_none','#1874B0'),
                                  ('Nested fusion','nested_minus_none','#9C5C91'),
                                  ('Hard filter','query_filter_minus_none','#C85824')]:
        d=N[(N.scope=='all_120_settings')&(N.metric=='top1')&(N.contrast==contrast)] if contrast=='nested_minus_none' else E[(E.scope=='all_120_settings')&(E.contrast==contrast)]
        assert len(d)==1
        r=d.iloc[0]
        rows.append(dict(label=label,contrast=contrast,estimate=r.estimate,ci_lo=r.ci_lo,ci_hi=r.ci_hi,color=color))
    return pd.DataFrame(rows),[base,nested]


def graphical_abstract(E):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'pdf.fonttype':42,
                         'svg.fonttype':'none','axes.labelsize':10.5})
    fig=plt.figure(figsize=(6.9,2.76),facecolor='white')
    canvas=fig.add_axes([0,0,1,1]);canvas.set_axis_off()
    canvas.text(.025,.95,'Oracle gains do not guarantee predicted-attribute utility',
                ha='left',va='top',fontsize=12.2,fontweight='bold',color='#173449')
    canvas.text(.025,.835,'BanglaHand: 42 writers | 6 representations | 120 settings',
                ha='left',va='top',fontsize=10.2,color='#394E5D')
    canvas.add_patch(FancyBboxPatch((.025,.17),.325,.56,boxstyle='round,pad=0.012,rounding_size=0.02',
                                    facecolor='#F3F7FA',edgecolor='#D4DFE7',lw=.8))
    canvas.add_patch(Rectangle((.055,.568),.045,.112,facecolor='white',edgecolor='#637C8E',lw=.8))
    for y in [.65,.626,.602]:canvas.plot([.063,.093],[y,y],color='#637C8E',lw=.7)
    canvas.text(.12,.622,'Questioned writing',va='center',fontsize=10.1,color='#173449')
    def box(x,y,w,h,text):
        canvas.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=0.01',
                                        facecolor='white',edgecolor='#9DB4C4',lw=.8))
        canvas.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9.7,color='#173449',linespacing=1.1)
    box(.047,.365,.128,.14,'Identity\nscores')
    box(.209,.365,.12,.14,'Predicted\nhand')
    box(.102,.205,.17,.105,'Writer ranking')
    for a,b in [((.179,.57),(.111,.515)),((.205,.57),(.269,.515)),
                ((.111,.355),(.16,.32)),((.269,.355),(.215,.32))]:
        canvas.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=.9,color='#637C8E'))
    canvas.text(.188,.125,'Query writer held out;\ngallery hands supplied',ha='center',va='center',
                fontsize=9.2,color='#394E5D',linespacing=1.15)
    ax=fig.add_axes([.628,.255,.345,.49])
    ax.axvline(0,color='#657B89',ls=':',lw=.9,zorder=0)
    for i,r in E.iterrows():
        ax.errorbar(100*r.estimate,i,xerr=[[100*(r.estimate-r.ci_lo)],[100*(r.ci_hi-r.estimate)]],
                    fmt='o',color=r.color,markersize=4,capsize=2,lw=1.3)
        digits=2 if r.contrast=='nested_minus_none' else 1
        ax.text(100*r.ci_hi+1.0,i,f'{100*r.estimate:+.{digits}f}',va='center',ha='left',fontsize=10.2,color=r.color,fontweight='bold')
    ax.axhline(1.5,color='#D4DFE7',lw=.8)
    ax.set_yticks(np.arange(5),E.label,fontsize=9.8)
    ax.set_ylim(4.55,-.55);ax.set_xlim(-22,15)
    ax.set_xticks([-20,-10,0,10],['-20','-10','0','+10'],fontsize=9.5)
    ax.set_xlabel('Top-1 change (percentage points)',fontsize=10)
    ax.tick_params(axis='y',length=0,pad=5)
    ax.spines[['top','left','right']].set_visible(False)
    ax.spines['bottom'].set_color('#637C8E')
    canvas.text(.025,.025,'Mean change vs identity only | conditional 95% writer intervals | one cohort',
                fontsize=9.2,color='#394E5D',va='bottom')
    canvas.set_xlim(0,1);canvas.set_ylim(0,1)
    meta={'Title':'Graphical abstract: '+TITLE,'Author':AUTHORS,
          'Subject':'Oracle controls and predicted-query utility; all 120 settings; conditional writer intervals.'}
    fig.savefig(UPLOAD/'Graphical_abstract.pdf',metadata=meta)
    fig.savefig(ASSETS/'Graphical_abstract.svg')
    fig.savefig(ASSETS/'Graphical_abstract.png',dpi=400)
    plt.close(fig)
    with Image.open(ASSETS/'Graphical_abstract.png') as im:
        im.convert('RGB').save(ASSETS/'Graphical_abstract.tif',compression='tiff_lzw',dpi=(400,400))
    E.to_csv(ASSETS/'Graphical_abstract_values.csv',index=False)


def cover_letter():
    paras=[
        'Dear Editors,',
        'Please consider the enclosed manuscript as a research report in Pattern Recognition Letters.',
        'The contribution is a controlled framework for testing whether predicted soft-biometric information improves identification. Attribute-classification accuracy and gains obtained with correct metadata do not establish the value of predicted attributes for identification. Our comparisons separate oracle opportunity, prediction errors, dependence between evidence sources, and writer-associated information. The resulting methodological finding is relevant to biometrics, document analysis and other recognition pipelines that combine a primary matcher with a predicted auxiliary attribute.',
        'On BanglaHand, using 42 writers, six representations and 120 identification settings, known handedness improves top-1 accuracy by 6.1 percentage points; a known random binary partition provides a comparable 6.9-point benchmark. In contrast, calibrated fixed fusion changes accuracy by -0.6 points, properly nested learned weighting by -0.08 points, and hard filtering by -14.3 points. Neither fusion rule establishes a gain in the designated comparisons. Joint-correctness and decision-margin diagnostics explain why these results concern achieved identification utility rather than the predictability of handedness alone.',
        f'The distinct companion preprint, "Writer-independent handedness identification from offline Bangla handwriting under controlled acquisition" (doi:{COMPANION}), studies unseen-writer handedness prediction and photometric cues. The present manuscript reuses the same collection and overlapping representations; this relationship is explicitly cited and disclosed. Its question, identification endpoints, oracle and random-partition controls, nested fusion comparison, and error-dependence diagnostics are distinct from the companion study. We do not present the shared cohort as independent validation.',
        'The manuscript is intended as a concise methodological study, with claims restricted to the tested collection and rules. It contains seven pages, with the 14-page supplement supplied separately. The supplementary reproducibility archive provides code, frozen features, numerical results and input hashes; the dataset is publicly available at doi:10.17632/n4sjstdzfy.1. Separate highlights and a graphical abstract accompany the manuscript.',
        'Thank you for considering this manuscript.',
    ]
    md='# Cover letter\n\n8 November 2026\n\nEditors, Pattern Recognition Letters\n\n'
    md+='\n\n'.join(paras)+f'\n\nSincerely,\n\nTanzim Hasan Prappo\nCorresponding author\nDepartment of Computer Science and Engineering\nShahjalal University of Science and Technology\nSylhet 3114, Bangladesh\n{EMAIL}\n'
    (SOURCE/'Cover_letter.md').write_text(md)
    for name,weight in [('WIDSans','normal'),('WIDSansBold','bold')]:
        pdfmetrics.registerFont(TTFont(name,findfont(FontProperties(family='DejaVu Sans',weight=weight))))
    body=ParagraphStyle('body',fontName='WIDSans',fontSize=10.2,leading=14,spaceAfter=9,textColor=colors.HexColor('#142B3A'))
    small=ParagraphStyle('small',parent=body,fontSize=9.2,leading=12,spaceAfter=0)
    title=ParagraphStyle('title',parent=body,fontName='WIDSansBold',fontSize=11,leading=14,spaceAfter=10)
    doc=SimpleDocTemplate(str(UPLOAD/'Cover_letter.pdf'),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=43,bottomMargin=40,
                          title='Cover letter: '+TITLE,author=AUTHORS)
    story=[Paragraph('8 November 2026',small),Spacer(1,10),
           Paragraph('Editors<br/>Pattern Recognition Letters',body),
           Paragraph(escape(TITLE),title)]
    story += [Paragraph(escape(p),body) for p in paras]
    story.append(KeepTogether([Paragraph('Sincerely,',body),
        Paragraph('Tanzim Hasan Prappo<br/>Corresponding author<br/>Department of Computer Science and Engineering<br/>'
                  'Shahjalal University of Science and Technology<br/>Sylhet 3114, Bangladesh<br/>'+EMAIL,small)]))
    doc.build(story)


def author_form(status):
    template=ROOT/'submission/revised/PRL-authorship-and-formatting-official.pdf'
    src=fitz.open(template);doc=fitz.open();doc.insert_pdf(src,from_page=0,to_page=0)
    p=doc[0]
    p.insert_text((42.52,64),'Manuscript',fontname='hebo',fontsize=8.5,color=(.2,.2,.2))
    p.insert_textbox(fitz.Rect(42.52,73,527,103),TITLE,fontname='tiro',fontsize=10.2,lineheight=1.2)
    p.insert_text((42.52,111),AUTHORS,fontname='tiro',fontsize=9)
    p.insert_text((42.52,125),'Related companion (distinct manuscript): doi:'+COMPANION,fontname='tiro',fontsize=8.6)
    def field(name,rect,value='',size=10,multiline=False):
        w=fitz.Widget();w.field_name=name;w.field_type=fitz.PDF_WIDGET_TYPE_TEXT
        w.rect=fitz.Rect(rect);w.field_value=value or '';w.text_font='TiRo';w.text_fontsize=size
        w.text_color=(0,0,0);w.border_width=0
        if multiline:w.field_flags=fitz.PDF_TX_FIELD_IS_MULTILINE
        p.add_widget(w)
    field('corresponding_author',(153,221.8,347,234.5),'Tanzim Hasan Prappo')
    field('signature',(83,406,278,420))
    field('signature_date',(302,406,414,420))
    field('manuscript_preprints',(42.52,477,528,532),status.get('manuscript_preprints'),multiline=True)
    field('conference_history',(42.52,574,528,613),status.get('conference_history'),multiline=True)
    field('republication_justification',(42.52,646,528,714),status.get('republication_justification'),multiline=True)
    p.insert_text((42.52,747),'Signature and date are to be completed after final author approval.',fontname='tiro',fontsize=8.5,color=(.3,.3,.3))
    doc.set_metadata(dict(title='Authorship confirmation: '+TITLE,author=AUTHORS,
                          subject='Unsigned manuscript-specific PRL form; no previous signature or approval reused.'))
    doc.save(UPLOAD/'Authorship_confirmation_form.pdf',garbage=4,deflate=True)
    doc.close()
    reader=PdfReader(UPLOAD/'Authorship_confirmation_form.pdf')
    fields=reader.get_fields()
    expected={'corresponding_author','signature','signature_date','manuscript_preprints','conference_history','republication_justification'}
    assert set(fields)==expected
    assert fields['corresponding_author'].get('/V')=='Tanzim Hasan Prappo'
    for name in ['signature','signature_date']:assert not fields[name].get('/V')
    for a in reader.pages[0].get('/Annots',[]):
        widget=a.get_object();assert widget.get('/Subtype')=='/Widget'
        assert widget.get('/T') in fields and widget.get('/V','')==fields[widget['/T']].get('/V','')
        assert widget.get('/AP',{}).get('/N') is not None
    for name in ['manuscript_preprints','conference_history','republication_justification']:
        assert fields[name].get('/V','')==(status.get(name) or '')
    return template


def main():
    for p in [UPLOAD,ASSETS,SOURCE]:p.mkdir(parents=True,exist_ok=True)
    status_path=SOURCE/'Publication_status.json'
    if not status_path.exists():
        status_path.write_text(json.dumps(dict(manuscript_preprints=None,conference_history=None,
            republication_justification=None,coauthor_approval='not recorded',
            note='Do not infer current-manuscript history or approval from the previous paper form.'),indent=2)+'\n')
    status=json.loads(status_path.read_text())
    E,inputs=effect_data();graphical_abstract(E);cover_letter();template=author_form(status)
    shutil.copy2(ROOT/'paper/highlights.txt',UPLOAD/'Highlights.txt')
    (ASSETS/'Graphical_abstract_caption.txt').write_text(
        'Oracle gains versus predicted-attribute utility in Bangla writer identification. '
        'Points give equally weighted mean top-1 changes over 120 settings and 42 writers; '
        'bars are conditional 95% hand-stratified writer-bootstrap intervals. '
        'Known query handedness and the expected known balanced random partition are oracle controls. '
        'Fixed fusion, nested learned fusion and hard filtering use predicted query handedness and known gallery hands. '
        'The study uses one controlled cohort; intervals do not include new-gallery or full model-refitting uncertainty.\n')
    files=[UPLOAD/'Graphical_abstract.pdf',UPLOAD/'Cover_letter.pdf',UPLOAD/'Authorship_confirmation_form.pdf']
    for f in files:
        d=fitz.open(f);assert len(d)==1,(f,len(d))
        for p in d:
            for span in [s for b in p.get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans']]:
                x0,y0,x1,y1=span['bbox'];assert min(x0,y0)>=-1 and x1<=p.rect.width+1 and y1<=p.rect.height+1,(f,span)
    with Image.open(ASSETS/'Graphical_abstract.png') as im:
        assert im.width>=1328 and im.width>im.height
        pixels=list(im.size)
    report=dict(passed=True,graphical_abstract_pixels=pixels,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs+[template,status_path,Path(__file__)]},
        files={str(p.relative_to(ROOT)):sha(p) for p in files},
        form_fields=sorted(PdfReader(files[2]).get_fields()),signature_reused=False,
        approval_recorded=status.get('coauthor_approval')=='confirmed',
        scope='Submission material generation and technical checks only; no new experiment or external submission.')
    (DEST/'Preparation_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
