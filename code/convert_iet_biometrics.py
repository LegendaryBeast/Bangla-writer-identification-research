#!/usr/bin/env python3
"""Convert the approved PRL sources into a separate IET Biometrics package.

This is a presentation conversion: scientific results and figure files are not
recomputed. Markdown retains editable citation keys and native LaTeX mathematics.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/iet_biometrics'
DEST = ROOT / 'submission/iet_biometrics'
PANDOC = '/opt/homebrew/bin/pandoc'
TITLE = 'Does predicted handedness improve writer identification? Oracle gains versus achieved utility in Bangla and English handwriting'
AUTHORS = ['Tanzim Hasan Prappo', 'M. Shahidur Rahman']
AFFILIATION = ('Department of Computer Science and Engineering, Shahjalal University of Science '
               'and Technology, Sylhet 3114, Bangladesh')
AI = ('An AI assistant was used to audit analysis code, and assist with drafting. The authors '
      'are responsible for verifying the analyses, references and final text, and for all publication claims.')
ETHICS = ("Informed consent was obtained voluntarily from all participants before data collection. "
          "Handwriting samples use pseudonymous writer identifiers. Formal institutional ethical "
          "approval was not required under the policies of Shahjalal University of Science and "
          "Technology because data collection involved non-invasive, voluntary handwriting tasks "
          "with no personal or health identifiers recorded. Writers received an honorarium paid "
          "by the authors from personal funds.")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def expand(text):
    macros = {}
    for p in (ROOT/'paper/figures_revision').glob('*macros.tex'):
        macros.update(re.findall(r'\\newcommand\{\\(r\w+)\}\{([^{}]*)\}', p.read_text()))
    for k in sorted(macros, key=len, reverse=True):
        text = re.sub(r'\\'+re.escape(k)+r'\b(?:\{\})?', lambda _: macros[k], text)
    def include(m):
        if m.group(1).endswith('_macros.tex'):
            return ''
        path = ROOT/'paper'/m.group(1)
        if not path.suffix:
            path = path.with_suffix('.tex')
        if not path.exists():
            path = OUT/m.group(1)
        return expand(path.read_text())
    text = re.sub(r'\\input\{([^}]+)\}', include, text)
    text = re.sub(r'(?m)^\s*%.*\n?', '', text)
    return text


def latex_body(text):
    return text.split(r'\section{Introduction}', 1)[1].split(r'\bibliographystyle', 1)[0]


def promote_supporting_material(body):
    """Bring completed methods and diagnostics into the longer main article."""
    sup=expand((ROOT/'paper/wid_supplement.tex').read_text()).replace('figures_revision/','assets/')
    def environment(kind,label,newlabel):
        blocks=re.findall(r'\\begin\{'+kind+r'\}.*?\\end\{'+kind+r'\}',sup,re.S)
        found=[b for b in blocks if r'\label{'+label+'}' in b]
        assert len(found)==1,(kind,label)
        block=found[0].replace('{'+label+'}','{'+newlabel+'}')
        block=block.replace(r'\begin{'+kind+'}',r'\begin{'+kind+'*}')
        block=block.replace(r'\end{'+kind+'}',r'\end{'+kind+'*}')
        return block
    # Full, already implemented extraction specifications replace delegation to SI.
    features=sup.split(r'\section{Self-contained feature specifications}',1)[1].split(r'\section{Information available at every fitting stage}',1)[0].strip()
    original=('The six representations are STROKE (208 contour and skeleton features), HC-183 (183 ink-mask features), frozen ImageNet ResNet-18, EfficientNet-B0 and ViT-B/16 embeddings \\cite{he2016resnet,tan2019efficientnet,dosovitskiy2021vit}, and Stroke-CNN (the same frozen ResNet-18 applied to binary ink tiles). No network is trained on these writers. The supplement specifies mask thresholds, scales, tiles, descriptor components and checkpoints without requiring a companion manuscript.')
    replacement=('Six representations are compared: STROKE, HC-183, frozen ImageNet ResNet-18, EfficientNet-B0 and ViT-B/16 embeddings \\cite{he2016resnet,tan2019efficientnet,dosovitskiy2021vit}, and Stroke-CNN. No network is trained on these writers. The following specifications define the descriptors and checkpoints used throughout the evaluation.\n\n'
                 +r'\subsection{Preprocessing and descriptor construction}'+'\n'+features+'\n\n'
                 +r'\subsection{Gallery construction and identity matching}')
    assert original in body;body=body.replace(original,replacement)
    anchor='exclusion counts and weighting are reported in the supplement.'
    units=(' At width 1,536 pixels, rows with ink fraction above 0.01 define a text region through its 0.5th and 99.5th percentiles. Nominal strip cuts move to minima of the row-ink profile smoothed with standard deviation 96/5 pixels. The search half-width is the smaller of 48 pixels and the text-block height divided by $3n$, with a minimum of three pixels. Row fractions transfer the boundaries across image scales. A unit is invalid if its high-resolution cropped mask contains fewer than 200 ink pixels, its 768-pixel mask contains fewer than 50, contour or skeleton extraction fails, or features are nonfinite. All representations use the same validity mask; the 39 excluded units are strips, with no paragraph or writer removed. Photographs flagged for bottom-edge truncation remain included. The exclusion breakdown is in Supporting Table S1.')
    assert anchor in body;body=body.replace(anchor,anchor+units)
    nesting=environment('table','tab:nesting','tab:mainnesting')
    nesting=nesting.replace('Fitting hierarchy for the revised primary analysis.','Information available at each stage of primary evaluation.')
    nesting=nesting.replace(r'Primary fusion & Fixed unit weight on log attribute probability; no fitted handedness weight.\\\bottomrule',
                            r'Fixed fusion & Unit weight on log attribute probability; no fitted handedness weight.\\'+ '\n'+
                            r'Nested weight & Development folds inside the outer 41-writer set; probabilities regenerate with both the outer writer and development fold excluded from attribute fitting and calibration.\\\bottomrule')
    anchor=r'\subsection{Handedness prediction and score fusion}'
    body=body.replace(anchor,('Table~\\ref{tab:mainnesting} separates the closed identification gallery from the writer-held-out attribute-training data. Holding out a writer for attribute prediction does not remove that writer from the identity gallery.\n\n'+nesting+'\n\n'+anchor))
    oracle=(r'\begin{equation}'+'\n'+r'P(\mathrm{success}\mid j)=\begin{cases}\binom{21}{j}/\binom{41}{j},&0\leq j\leq21,\\0,&j>21.\end{cases}'+'\n'+r'\label{eq:randompartition}'+'\n'+r'\end{equation}')
    anchor='Known balanced random writer partitions test how much oracle benefit comes from supplying an arbitrary correct partition.'
    replacement=(anchor+' With 21 writers in each group, a query whose true writer is outranked by $j$ impostors survives an oracle filter only when all $j$ are assigned to the opposite group. Uniform balanced partitions therefore give the exact expectation\n'+oracle+'\nThe same deterministic tie precedence is used for baseline ranks and partition success. The writer-weighted expectation provides an exact benchmark; 2,000 common Monte Carlo partitions characterize partition-to-partition variation across settings. Partition quantiles are not independent-cohort confidence intervals.')
    assert anchor in body;body=body.replace(anchor,replacement)
    # Move robustness behind the filtering/label diagnostics to keep first figure
    # citations in their original order. Figures 1--3 retain their approved files.
    a=body.index(r'\subsection{Within-cohort robustness}')
    b=body.index(r'\subsection{Conditional filtering utility}',a)
    robustness=body[a:b].strip();body=body[:a]+body[b:]
    # Add the existing calibration table before the label-control subsection.
    calibration=environment('table','tab:calibration','tab:maincalibration')
    calibration_text=(r'\subsection{Attribute calibration and prediction information}'+'\n'
        'Table~\\ref{tab:maincalibration} reports writer-held-out accuracy and proper scoring rules for the primary and sensitivity predictors on self-chosen full paragraphs. Accuracy, Brier score and log loss measure different aspects of the predictions; none alone specifies how those scores interact with identity margins. Reliability-bin plots and their uncertainty are retained in Supporting Figure S1. The adjusted HC-183 predictor removes fitted grayscale and ink-density covariates inside each training split; it is a sensitivity analysis rather than a causal isolation of acquisition effects.\n\n'+calibration+'\n\n')
    anchor=r'\subsection{Writer-associated information and exploratory extensions}'
    assert anchor in body;body=body.replace(anchor,calibration_text+anchor)
    overlap=environment('figure','fig:labels','fig:mainlabels')
    anchor='Exploratory both-sided descriptor agreement improves top-1 by'
    body=body.replace(anchor,('Figure~\\ref{fig:mainlabels} makes the two writer-information conditions explicit. The random label stays fixed for a writer, while the training/test relation changes. The distribution under enrolment training is therefore a control for writer-associated predictability, not proof of a leakage mechanism in the primary writer-held-out pipeline.\n\n'+overlap+'\n\n'+anchor))
    seed=environment('table','tab:seeddiag','tab:mainseeds')
    robfig=environment('figure','fig:robustness','fig:mainrobustness')
    robustness+='\n\nTable~\\ref{tab:mainseeds} gives the calibration-partition results rather than treating one random fold assignment as definitive. Figure~\\ref{fig:mainrobustness} places temperature selection, learned weights and evaluation-writer influence beside those prediction changes. The diagnostics vary specific components of the fitted pipeline and retain the enrolled gallery; their scope differs from refitting on a new cohort or evaluating a new capture session.\n\n'+seed+'\n\n'+robfig+'\n\n'
    # Promote the verified decision-level explanation and its existing figure.
    paragraph=re.search(r'The practical criterion also involves identity margins\..*?(?=\n\n)',body,re.S).group(0)
    body=body.replace(paragraph,'The interaction between confidence and identity margins determines which decisions fusion changes. The decision-level analysis in the Results distinguishes rescued identifications from damaged ones and verifies the exact ranking boundary.')
    marginfig=environment('figure','fig:margins','fig:mainmargins')
    margins=(r'\subsection{Decision margins and opposing fusion effects}'+'\n'
        'Fusion preserves candidate ranking within each handedness group. Define the true-writer margin against the best opposite-hand candidate and the signed attribute log-odds as\n'
        +r'\begin{equation}'+'\n'+r'\begin{aligned}m_q&=\frac{s_{w^*}-\max_{h_w\ne h_{w^*}}s_w}{T},\\r_q&=\log\frac{p_q(h_{w^*})}{p_q(1-h_{w^*})}.\end{aligned}'+'\n'+r'\label{eq:decisionmargin}'+'\n'+r'\end{equation}'+'\n'
        'Away from score ties, identification succeeds exactly when the true writer leads its own hand group and $m_q+\\lambda r_q>0$. Thus fusion cannot repair a wrong within-group ordering. The true identity and handedness enter only this retrospective diagnostic; they are not available inputs to the operational query predictor.\n\n'
        'A rescued decision is incorrect before fusion and correct afterwards; a damaged decision changes in the opposite direction. Their difference equals the net top-1 change. Across the 120 settings, fixed fusion rescues 2.02\\% of decisions and damages 2.59\\%; nested weighting rescues 1.33\\% and damages 1.42\\%. Figure~\\ref{fig:mainmargins} shows these opposing effects with conditional writer-bootstrap intervals and illustrates the exact margin boundary for HC-183 paragraph queries with one reference.\n\n'
        'Direct candidate-score ranking and the analytical decision reconstruction agree on all 664,680 evaluated query/reference decisions, with no exact boundary ties. Writer-level fused accuracies independently match the fixed and nested outputs. This deterministic consistency check explains the fitted score rule; it does not add independent experimental observations.\n\n'+marginfig+'\n\n')
    anchor=r'\section{Discussion}'
    assert anchor in body;body=body.replace(anchor,robustness+margins+anchor)
    return body


def prepare():
    OUT.mkdir(exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    source = (ROOT/'paper/wid_manuscript.tex').read_text()
    abstract = expand(re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', source, re.S).group(1).strip())
    keywords = ('Writer identification | Soft biometrics | Handedness | Bangla handwriting | '
                'Score fusion | Evaluation protocol')
    body = r'\section{Introduction}'+latex_body(source)
    body = body.replace(r'\section{Data and evaluation protocol}', r'\section{Materials and methods}')
    body = body.replace(r'\subsection{Scope and limitations}', r'\section{Scope and limitations}')
    body = body.replace(r'\section{Conclusion}', r'\section{Conclusions}')
    body = body.replace('The supplement distinguishes this nested comparison from the historical, incompletely nested weight fit.',
                        'The supplementary methods specify the complete nested fitting hierarchy.')
    old = ("Four designated comparisons retain the original study's focus: ResNet-18 and ViT-B/16 on self-chosen paragraphs with one reference and quarter strips with six references. Holm correction covers these four comparisons. They are retrospective evaluation contrasts, not preregistered confirmation: a feasibility analysis had already inspected paragraph-level conditions, and the present normalization correction followed review. The new learned-weight comparisons against no fusion and fixed fusion form eight exploratory contrasts (four settings, two comparators), with Holm correction across all eight.")
    new = ("Four settings were designated after an exploratory feasibility analysis of paragraph-level conditions: ResNet-18 and ViT-B/16 on self-chosen paragraphs with one reference and quarter strips with six references. These analyses were not preregistered. Holm correction covers the four fixed-fusion comparisons. Learned-weight comparisons against no fusion and fixed fusion form eight exploratory contrasts (four settings, two comparators), with Holm correction across all eight.")
    assert old in body
    body = body.replace(old, new)
    # Separate declarations in the order used by the journal; retain exact AI wording.
    data_start = body.index(r'\section*{Data and code availability}')
    data = body[data_start:].split(r'\par\vspace', 1)[0]
    credit = re.search(r'\\section\*\{CRediT authorship contribution statement\}\s*(.*?)\n\\par', body, re.S).group(1).strip()
    declarations = data + '\n\n' + r'\section*{Author contributions}'+'\n'+credit+'\n\n'
    declarations += r'\section*{Ethics statement}'+'\n'+ETHICS+'\n\n'
    declarations += r'\section*{Funding}'+'\nThis research received no specific grant.\n\n'
    declarations += r'\section*{Conflicts of interest}'+'\nThe authors declare no conflict of interest regarding the publication of this article.\n\n'
    declarations += r'\section*{Declaration of generative AI use}'+'\n'+AI+'\n\n'
    declarations += (r'\section*{Supplementary material}'+'\n'
                     'Supporting information contains self-contained feature and preprocessing specifications, '
                     'the fitting hierarchy, filtering and random-partition derivations, calibration and '
                     'random-label controls, nested fusion selection, uncertainty and sensitivity analyses, '
                     'decision margins, the complete 120-setting grid, and reproduction instructions.\n')
    body = expand(body[:data_start] + declarations)
    body = body.replace('figures_revision/', 'assets/')
    body = promote_supporting_material(body)
    for environment in ('figure*', 'table*'):
        for placement in ('ht', 't'):
            body = body.replace('\\begin{' + environment + '}[' + placement + ']', '\\begin{' + environment + '}[!t]')
    (OUT/'manuscript_body.tex').write_text(body)
    (OUT/'abstract.tex').write_text(abstract+'\n')
    # The class calls a bibliography style itself. Suppress only that automatic
    # choice so the submission can use citation-order numeric references.
    header = r'''\let\IETbibliographystyle\bibliographystyle
\def\bibliographystyle#1{}
\documentclass[ASNA,twocolumn]{USG}
\let\bibliographystyle\IETbibliographystyle
\bibliographystyle{ietnumeric}
\usepackage{amsmath,booktabs,tabularx,xurl,etoolbox,placeins}
\AtBeginEnvironment{tabularx}{\fontsize{8}{10}\selectfont}
\usepackage[switch]{lineno}
\floatpagestyle{plain}
\rotfloatpagestyle{plain}
\graphicspath{{assets/}{images/}}
\articletype{RESEARCH ARTICLE}
\journal{IET Biometrics}
\copyyear{}
\volume{}
\startpage{1}
\articledoi{}
\historydates{}
\doiheadtext{}
\received{}
\revised{}
\accepted{}
\makeatletter
\patchcmd{\@maketitle}{\includegraphics[width=56mm]{allergy.eps}}{}{}{\PackageError{IET}{Cannot remove sample journal logo}{}}
\patchcmd{\@@articletype}{\includegraphics[width=0.70in]{images/openaccess}}{}{}{\PackageError{IET}{Cannot remove production badge}{}}
\makeatother
\authormark{Prappo and Rahman}
\titlemark{Oracle gains and predicted-attribute utility}
\hypersetup{hidelinks,pdftitle={TITLE},pdfauthor={Tanzim Hasan Prappo; M. Shahidur Rahman}}
\raggedbottom
\setcounter{dbltopnumber}{3}
\setcounter{topnumber}{3}
\setcounter{totalnumber}{6}
\renewcommand{\dbltopfraction}{.9}
\renewcommand{\topfraction}{.9}
\renewcommand{\textfraction}{.08}
\renewcommand{\floatpagefraction}{.7}
\renewcommand{\dblfloatpagefraction}{.7}
\setlength{\dbltextfloatsep}{12pt plus 2pt minus 2pt}
\setlength{\dblfloatsep}{10pt plus 2pt minus 2pt}

\RemoveFromHook{shipout/after}[USG]
\begin{document}
\title{TITLE}
\author[1]{Tanzim Hasan Prappo}
\author[1]{M. Shahidur Rahman}
\address[1]{\orgdiv{Department of Computer Science and Engineering, }\orgname{Shahjalal University of Science and Technology, }\orgaddress{\state{Sylhet 3114, }\country{Bangladesh}}}
\corres{Tanzim Hasan Prappo (\email{2021331006@student.sust.edu}), corresponding author. Coauthor email: \email{rahmanms@sust.edu}.}
\keywords{KEYWORDS}
\abstract[ABSTRACT]{ABSTRACT}
\maketitle
\pagestyle{plain}
\thispagestyle{plain}
\linenumbers
'''.replace('TITLE', TITLE).replace('KEYWORDS', keywords).replace('ABSTRACT}', abstract+'}')
    # Avoid replacing the literal [ABSTRACT] caption when injecting content.
    (OUT/'manuscript.tex').write_text(header + body + '\n\\bibliography{refs}\n\\end{document}\n')
    # Keep citation-order numeric references and ensure dataset/preprint DOIs
    # are printed too: upstream unsrtnat omits DOI fields for @misc entries.
    bst=Path(subprocess.check_output(['kpsewhich','unsrtnat.bst'],text=True).strip()).read_text()
    misc=re.search(r'FUNCTION \{misc\}.*?(?=FUNCTION)',bst,re.S).group(0)
    fixed=misc.replace('  format.url output','  format.doi output\n  format.url output')
    assert fixed!=misc
    bst=bst.replace(misc,fixed)
    (OUT/'ietnumeric.bst').write_text('% IET submission variant: unsrtnat with DOI output for misc records.\n'+bst)
    sup = (ROOT/'paper/wid_supplement.tex').read_text()
    sup = sup.replace('figures_revision/', 'assets/')
    sup = sup.replace('Supplementary methods and results', 'Supporting information')
    sup = sup.replace('This document accompanies the revised manuscript. The authoritative numerical outputs are in \\path{results/wid_revision_2026-11-08/}; the original run is retained separately for provenance. Primary identification results in the revised article use selected-reference normalization and equal writer weights. The revisions and additional controls are retrospective.',
                      'This document accompanies the IET Biometrics manuscript. The authoritative numerical outputs are in \\path{results/wid_revision_2026-11-08/}. Identification uses selected-reference normalization and equal writer weights. Analyses are retrospective and were not preregistered.')
    sup = sup.replace('No new participants or images were collected for this revision.', 'All analyses use the released photographs.')
    old = 'The historical learned-weight outputs are omitted because their development attribute scores were not regenerated inside a second outer exclusion. The new nested comparison in Section~\\ref{sec:supnested} corrects that limitation. Historical identification outputs are retained for traceability and do not supply revised identification results. Unchanged attribute-prediction controls are reused as explicitly noted in Section~S5.'
    new = 'The nested learned-weight comparison in Section~\\ref{sec:supnested} regenerates development attribute scores inside a second outer exclusion. Primary identification outcomes use the stated selected-reference normalization. Attribute-prediction controls with writer overlap are reported separately in Section~S5.'
    assert old in sup
    sup = sup.replace(old, new)
    sup = sup.replace('Shared-text outputs in the historical control use', 'Shared-text outputs in the writer-overlap control use')
    sup = sup.replace('The grid and objective are fixed before this additional run, without choosing a rule from its test results. This is a post-review exploratory comparison, not preregistration.',
                      'The grid and objective were fixed before fitting the learned-weight comparison, without choosing a rule from its test results. This comparison is exploratory and was not preregistered.')
    start = sup.index('The original feasibility notes inspected paragraph-level settings')
    end = sup.index(r'\end{document}', start)
    sup = sup[:start] + ('The four designated fixed-fusion settings were chosen after an exploratory feasibility analysis of paragraph-level conditions. Learned-weight comparisons and robustness analyses are exploratory; no analysis was preregistered. Identification normalization and temperature fitting use only the information permitted by the stated closed-gallery protocol. Conditional bootstrap intervals do not encompass model-refitting, new-gallery or new-session uncertainty. Independent-cohort and cross-session replication can assess transfer beyond these acquisition conditions.\n') + sup[end:]
    sup = expand(sup)
    (OUT/'supporting_information.tex').write_text(sup)
    cover = f'''---
title: Cover letter
date: 8 November 2026
---

Editors, IET Biometrics

Dear Editors,

Please consider “{TITLE}” as a Research Article in IET Biometrics.

The study examines when a predicted soft biometric provides useful evidence for closed-set writer identification. It addresses soft biometrics and information fusion, handwriting as a biometric modality, and system design and performance evaluation. Its contribution is a controlled comparison separating oracle opportunity, operational benefit, error dependence and writer-associated information.

Across 42 writers, six representations and 120 settings, known handedness improves top-1 accuracy by 6.1 percentage points, compared with 6.9 points for an expected known random binary partition. With query handedness predicted at 65.5% accuracy on self-chosen paragraphs, calibrated fixed fusion changes identification by -0.6 points, nested learned weighting by -0.08 points, and hard filtering by -14.3 points. Neither fusion rule establishes a gain in the designated comparisons. Filtering and decision-margin diagnostics explain why attribute accuracy and correct metadata alone do not establish identification utility.

The distinct companion preprint, “Writer-independent handedness identification from offline Bangla handwriting under controlled acquisition” (doi:10.2139/ssrn.7662278), evaluates handedness classification and photometric cues. Both studies use the same cohort and overlapping representations. The present work adds downstream identity-ranking endpoints, oracle and random-partition controls, nested fusion-weight selection, and joint-error and writer-overlap diagnostics. This relationship is cited and explained in the manuscript; the shared cohort is not presented as independent validation.

The article includes detailed feature specifications, the fitting hierarchy, calibration and writer-overlap controls, robustness diagnostics and verified decision-margin analysis. Supporting information provides the complete 120-setting grid and reproduction details. Code, frozen features, setting manifests and numerical results are available at https://github.com/LegendaryBeast/Bangla-writer-identification-research/releases/tag/v1.6.0. The photographs are accessible through doi:10.17632/n4sjstdzfy.1.

This manuscript is not under consideration by another journal. Thank you for considering the work.

Sincerely,

Tanzim Hasan Prappo  
Corresponding author  
{AFFILIATION}  
2021331006@student.sust.edu
'''
    (OUT/'cover_letter.md').write_text(cover)
    (OUT/'metadata.json').write_text(json.dumps(dict(title=TITLE,authors=AUTHORS,affiliation=AFFILIATION,
        abstract=abstract,keywords=keywords.split(' | '),ethics_source='Author-supplied Data in Brief manuscript, page 6',
        ethics_statement=ETHICS,new_experiments=False,new_datasets=False),indent=2)+'\n')
    print('Prepared separate IET sources; PRL source and figure files untouched.')


def labels_from_aux(path):
    return dict(re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}', Path(path).read_text()))


def walk(x, fn):
    if isinstance(x, list):
        return [walk(i, fn) for i in x]
    if isinstance(x, dict):
        x={k:walk(v,fn) for k,v in x.items()}
        return fn(x)
    return x


def markdown():
    meta=json.loads((OUT/'metadata.json').read_text())
    for stem in ['manuscript','supporting_information']:
        prefix='' if stem=='manuscript' else 'S'
        tex=(OUT/(stem+'.tex')).read_text()
        if stem=='manuscript':
            body=(OUT/'manuscript_body.tex').read_text()
        else:
            body=tex.split(r'\vspace{-12pt}',1)[1].split(r'\end{document}',1)[0]
        labels=labels_from_aux(OUT/(stem+'.aux'))
        equation_numbers=[]
        def numbered_equation(m):
            env,math=m.group(1),m.group(2)
            label=re.search(r'\\label\{([^}]+)\}',math)
            number=labels[label.group(1)] if label else prefix+str(len(equation_numbers)+1)
            equation_numbers.append(number)
            math=re.sub(r'\\label\{[^}]+\}|\\nonumber','',math).strip()
            if env=='align' and number=='2' and not prefix:
                math=r'A_{\mathrm{filter}}=aP(K\mid C)=aA_{\mathrm{known}}+\operatorname{Cov}(\mathbf{1}_C,\mathbf{1}_K),'
            elif env=='align':
                math=r'\begin{aligned}'+math+r'\end{aligned}'
            return '\n\n$$'+math+r'\qquad\text{('+number+')}$$\n\n'
        body=re.sub(r'\\begin\{(equation|align)\}(.*?)\\end\{\1\}',numbered_equation,body,flags=re.S)
        body=re.sub(r'\\eqref\{([^}]+)\}',lambda m:'('+labels[m.group(1)]+')',body)
        body=re.sub(r'\\ref\{([^}]+)\}',lambda m:labels[m.group(1)],body)
        body=re.sub(r'\\label\{[^}]+\}','',body)
        body=re.sub(r'\\path\{([^}]+)\}',lambda m:r'\texttt{'+m.group(1).replace('_',r'\_')+'}',body)
        body=re.sub(r'\\rm\s+([A-Za-z]+)',lambda m:r'\mathrm{'+m.group(1)+'}',body)
        body=body.replace(r'\begin{table*}',r'\begin{table}').replace(r'\end{table*}',r'\end{table}')
        body=re.sub(r'\\cmidrule(?:\[[^]]*\])?(?:\([^)]*\))?\{[^}]*\}','',body)
        body=re.sub(r'\\endfirsthead.*?\\endhead','',body,flags=re.S)
        body=re.sub(r'\\paragraph\{([^}]+)\}',lambda m:r'\textbf{'+m.group(1)+'}',body)
        # Keep mathematical grouping but remove the layout-only wrappers.
        body=body.replace(r'\FloatBarrier','').replace(r'\begin{samepage}','').replace(r'\end{samepage}','')
        body=body.replace('\\allowbreak ','').replace('\\allowbreak','')  # layout-only line-break hints
        ast=json.loads(subprocess.run([PANDOC,'-f','latex','-t','json'],input=body,text=True,
                                     capture_output=True,check=True).stdout)
        headers=[0,0,0,0,0,0]
        counts={'Figure':0,'Table':0,'Equation':len(equation_numbers)}
        def convert(x):
            typ=x.get('t')
            if typ=='Header':
                level,attrs,inlines=x['c']
                if 'unnumbered' not in attrs[1]:
                    headers[level-1]+=1
                    for j in range(level,len(headers)):headers[j]=0
                    number='.'.join(str(n) for n in headers[:level])
                    if prefix:number=prefix+number
                    x['c'][2]=[{'t':'Str','c':number},{'t':'Space'}]+inlines
            elif typ in ['Figure','Table']:
                counts[typ]+=1
                caption=x['c'][1]
                # Pandoc captions carry [short caption, list of blocks].
                blocks=caption[1]
                if blocks and blocks[0]['t'] in ['Plain','Para']:
                    blocks[0]['c']=[{'t':'Strong','c':[{'t':'Str','c':f'{typ} {prefix}{counts[typ]}.'}]},{'t':'Space'}]+blocks[0]['c']
            elif typ=='Math':
                x['c'][1]=re.sub(r'\\label\{[^}]+\}','',x['c'][1])
            elif typ=='Image':
                target=x['c'][2][0]
                x['c'][2][0]=re.sub(r'\.pdf$','.png',target)
                x['c'][0][2]=[['width','6.65in']]
            return x
        ast=walk(ast,convert)
        blocks=[]
        for block in ast['blocks']:
            if block['t']!='Para':
                blocks.append(block)
                continue
            inline=[]
            for item in block['c']:
                if item.get('t')=='Math' and item['c'][0]['t']=='DisplayMath':
                    if inline:
                        blocks.append({'t':'Para','c':inline})
                        inline=[]
                    blocks.append({'t':'Para','c':[item]})
                else:
                    inline.append(item)
            if inline:
                blocks.append({'t':'Para','c':inline})
        ast['blocks']=blocks
        if stem=='manuscript':
            # Keep the authors/title editable and include contact details.
            front=(AFFILIATION+'\n\nCorresponding author: Tanzim Hasan Prappo '
                   '(2021331006@student.sust.edu). Coauthor email: rahmanms@sust.edu.\n\n'
                   '## Abstract {.unnumbered}\n\n'+meta['abstract'].replace(r'\%','%')+'\n\n'
                   '**Keywords:** '+ '; '.join(meta['keywords'])+'\n\n')
            ast_front=json.loads(subprocess.run([PANDOC,'-f','markdown','-t','json'],input=front,
                                  text=True,capture_output=True,check=True).stdout)
            ast['blocks']=ast_front['blocks']+ast['blocks']
        else:
            ast_front=json.loads(subprocess.run([PANDOC,'-f','markdown','-t','json'],input=f'Supporting information for **{TITLE}**.\n',text=True,capture_output=True,check=True).stdout)
            ast['blocks']=ast_front['blocks']+ast['blocks']
        md=subprocess.run([PANDOC,'-f','json','-t','markdown+tex_math_dollars+pipe_tables+grid_tables','--wrap=none'],
                          input=json.dumps(ast),text=True,capture_output=True,check=True).stdout
        yaml='---\ntitle: '+json.dumps(TITLE if stem=='manuscript' else 'Supporting information')+'\nauthor:\n'
        yaml+=''.join('  - '+json.dumps(a)+'\n' for a in AUTHORS)
        yaml+='date: ""\nbibliography: refs.bib\ncsl: ieee.csl\nlink-citations: true\nreference-section-title: References\n---\n\n'
        (OUT/(stem+'.md')).write_text(yaml+md)
        (OUT/(stem+'.json')).write_text(json.dumps(ast,ensure_ascii=False,indent=2)+'\n')
        (OUT/(stem+'_conversion.json')).write_text(json.dumps(dict(counts=counts,equation_numbers=equation_numbers,labels=labels),indent=2)+'\n')
        print(stem,'converted:',counts,'cross-references:',len(labels))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','markdown'])
    args=parser.parse_args()
    globals()[args.stage]()
