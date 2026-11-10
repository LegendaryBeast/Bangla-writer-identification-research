#!/usr/bin/env python3
"""Strengthen the IET literature synthesis and repair standard two-column output."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper/iet_biometrics'

RELATED = r'''\section{Related work}
\label{sec:related}
\subsection{Soft biometrics and downstream utility}
Soft-biometric fusion predates handwriting-based handedness prediction. Jain et al.\ combine demographic information with a primary biometric, while Dantcheva et al.\ distinguish attribute estimation, database indexing and recognition applications \cite{jain2004soft,dantcheva2016soft}. These foundations motivate auxiliary evidence, but its usefulness depends on the available metadata, prediction uncertainty and primary matcher. A correct attribute can narrow a candidate set; a predicted attribute can also demote the true candidate. Consequently, attribute recognition and identification improvement require different endpoints. The present work examines this distinction for handwriting through paired comparisons with the same gallery, identity scores and query samples.

\subsection{Handedness and demographic prediction}
Early handwriting studies investigated gender and other demographic traits using online dynamics, offline shape and texture, and combinations of descriptors \cite{liwicki2011gender,siddiqi2015gender,bouadjenek2016soft,hassaine2013icdar}. Al-Maadeed et al.\ study a 121-writer QUWI subset with Arabic and English documents, combining handcrafted features, fuzzy conceptual reduction and a nearest-neighbour classifier \cite{almaadeed2016handedness}. Their principal evaluation concerns handedness classification and feature reduction. Multiple scripts and text conditions broaden that evaluation, but do not by themselves measure the change in an identity ranking caused by predicted handedness.

Morera et al.\ use CNN word predictions and line-level voting on IAM and KHATT, explicitly separating training and test individuals \cite{morera2018handedness}. They report handedness overall accuracy of 90.70\% on IAM and 70.91\% on KHATT. Their respective left-handed recalls are 63.40\% and 38.64\%, illustrating why overall accuracy must be interpreted alongside class composition and class-specific performance. Rahmanian and Shayegan compare DenseNet201, InceptionV3 and Xception on the same two databases and report 99.14\% KHATT handedness accuracy \cite{rahmanian2021handedness}. This is an attribute-classification result; the accessible publisher record does not establish all details needed for a matched writer-disjoint comparison. We therefore make no inference about writer overlap in that study.

The companion study evaluates writer-independent Bangla handedness classification and photometric cues \cite{prappo2026handedness}. Its 77.4\% balanced accuracy aggregates SVM margins over six shared paragraphs under repeated five-fold evaluation. Here, a calibrated single-query probability modifies a closed-gallery identity ranking under leave-one-writer-out fitting. Different text aggregation, fitting and endpoints prevent a direct numerical comparison. Table~\ref{tab:related} separates these prediction tasks from writer-recognition tasks and the utility endpoint considered here.

\subsection{Writer recognition and evaluation controls}
Text-independent writer recognition models individual writing style through contour directions, hinges, allographs and contour-fragment codebooks \cite{bulacu2007text,schomaker2007codebooks}. Learned representations can target identity directly: FragNet combines feature-pyramid and fragment pathways for word- and page-level writer identification across four benchmarks \cite{he2020fragnet}. In Bangla, Paul et al.\ combine feature extraction and genetic feature selection for writer verification \cite{paul2024bangla}; BanglaWriting and the Boise State collection support related handwriting-recognition research \cite{mridha2021banglawriting,majid2018bangla}. Verification, recognition of written content and closed-set identity ranking address different decisions. Our frozen representations provide controlled comparators, not a claim to outperform handwriting-specialized recognition networks.

Evaluation design also determines what an apparent improvement means. Writer separation addresses generalization to new individuals; it does not alone establish that an attribute supplies information beyond the identity matcher. Leakage analysis requires specifying the information available at each fitting stage \cite{kaufman2012leakage}, and fusion-weight selection must be separated from evaluation to avoid selection bias \cite{cawley2010selection}. We bring these concerns together by measuring oracle opportunity, predicted-attribute utility, error dependence and writer-associated predictability, with an external evaluation on separate CSAFE pages and capture sessions. This combination defines the contribution rather than a new attribute classifier or an accuracy ranking across incompatible protocols.

\begin{table*}[!t]
\caption{Representative related work, organized by the decision evaluated. Reported results retain their original endpoints and are not a common performance ranking.}\label{tab:related}
\centering
\begin{tabularx}{\textwidth}{@{}p{.18\textwidth}p{.23\textwidth}p{.24\textwidth}X@{}}\toprule
Study & Data and representation & Evaluated endpoint & Relation to the present question\\\midrule
Jain et al.; Dantcheva et al.\ \cite{jain2004soft,dantcheva2016soft} & Primary biometrics with auxiliary personal attributes & Recognition, indexing and soft-biometric fusion & Establish the broader motivation for auxiliary information.\\
Al-Maadeed et al.\ \cite{almaadeed2016handedness} & QUWI, 121 writers; Arabic/English; handcrafted features and fuzzy reduction & Handedness classification and feature reduction & Attribute predictability; downstream ranking change is a separate endpoint.\\
Morera et al.\ \cite{morera2018handedness} & IAM/KHATT; CNN words, line voting; separate train/test writers & Handedness overall accuracy: 90.70\%/70.91\%; left recall: 63.40\%/38.64\% & Shows the importance of class composition and prediction unit.\\
Rahmanian and Shayegan \cite{rahmanian2021handedness} & IAM/KHATT; three deep CNN architectures & Gender and handedness; 99.14\% KHATT handedness accuracy & Strong attribute result; detailed split remains unverified here.\\
Bulacu and Schomaker; Schomaker et al.\ \cite{bulacu2007text,schomaker2007codebooks} & Textural/allographic descriptors and contour-fragment codebooks & Writer identification and verification & Establish identity-matching representations, rather than attribute utility.\\
He and Schomaker \cite{he2020fragnet} & Four benchmarks; feature-pyramid and fragment pathways & Word- and page-level writer identification & Learns identity representations; not a handedness-fusion comparison.\\
Paul et al.\ \cite{paul2024bangla} & Offline Bangla, 100 writers; handcrafted/CNN features and genetic selection & Writer verification; reported mean accuracy 94.54\% & Verifies a claimed writer; distinct from closed-set top-1 ranking.\\
Companion preprint \cite{prappo2026handedness} & BanglaHand, 42 balanced writers; shared-prompt aggregation & Writer-independent handedness and photometric sensitivity & Shared cohort and representations; a different task and endpoint.\\\bottomrule
\end{tabularx}
\end{table*}

'''

COMPARISON = r'''
Table~\ref{tab:positioning} makes the methodological comparison explicit. Prior attribute studies establish whether handwriting supports demographic prediction, while identity-focused studies develop the matcher itself. Our contribution is to intervene on the auxiliary information while holding the identity comparison fixed. The known-hand and random-partition controls quantify an opportunity that a predictor may fail to realize; nested weighting then tests whether a flexible, calibrated use of that predictor changes the finding. This is a complementary evaluation question, not evidence that our representations are superior to the recognition systems in Table~\ref{tab:related}.

\begin{table*}[!t]
\caption{Methodological coverage of representative cited approaches. H: handedness prediction; WD: writer-disjoint attribute evaluation; W: writer endpoint (ID = identification, V = verification, $\Delta$ID = paired identification change); O: known-versus-predicted handedness comparison; R: known random-partition benchmark; F: calibrated fusion with nested weight selection; D: joint-error/margin and random-writer-label diagnostics; S: separate gallery/query capture-session evaluation. Y = evaluated; -- = outside the reported task; NR = not reported as the specified evaluation in the inspected source; U = detailed protocol unverified from the accessible record. Counts describe study datasets, not independent statistical replications.}\label{tab:positioning}
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\textwidth}{@{}p{.22\textwidth}cccccccXc@{}}\toprule
Study & H & WD & W & O & R & F & D & Data scope & S\\\midrule
Al-Maadeed et al.\ \cite{almaadeed2016handedness} & Y & U & -- & NR & NR & NR & NR & QUWI subset & NR\\
Morera et al.\ \cite{morera2018handedness} & Y & Y & -- & NR & NR & NR & NR & IAM and KHATT & NR\\
Rahmanian and Shayegan \cite{rahmanian2021handedness} & Y & U & -- & U & U & U & U & IAM and KHATT & U\\
Bulacu/Schomaker studies \cite{bulacu2007text,schomaker2007codebooks} & -- & -- & ID/V & -- & -- & -- & -- & Identity benchmarks & U\\
He and Schomaker \cite{he2020fragnet} & -- & -- & ID & -- & -- & -- & -- & Four benchmarks & U\\
Paul et al.\ \cite{paul2024bangla} & -- & -- & V & -- & -- & -- & -- & Bangla, 100 writers & NR\\
Companion preprint \cite{prappo2026handedness} & Y & Y & -- & -- & -- & -- & -- & BanglaHand & --\\
Present study & Y & Y & $\Delta$ID & Y & Y & Y & Y & BanglaHand; CSAFE & Y\\\bottomrule
\end{tabularx}
\end{table*}

The practical implication is a testable evaluation sequence. An auxiliary-attribute system should first establish an oracle opportunity, then compare predicted fusion with an identity-only baseline on the same queries, selecting its calibration and fusion parameters without the evaluated writer's outcomes. Decision-level rescues and damage show where the score change helps, while writer-overlap controls distinguish an attribute-generalization question from an enrolment-conditioned one. Separate capture sessions then test whether the conclusion survives a change in source pages and acquisition. Our controls make these questions answerable even when the final fusion gain is inconclusive. Their broader value is an auditable decision about whether to deploy an auxiliary score, rather than treating attribute accuracy as a proxy for identification benefit.
'''


def main():
    body = (PAPER / 'manuscript_body.tex').read_text()
    begin = body.index(r'\section{Related work}')
    end = body.index(r'\section{Materials and methods}', begin)
    body = body[:begin] + RELATED + body[end:]
    if 'The paper is organized as follows.' not in body:
        body = body.replace(r'\section{Related work}',
            'The paper is organized as follows. Section 2 reviews related work; Section 3 defines the protocol; Section 4 reports the results; and Sections 5--7 discuss their implications, scope and conclusions.\n\n'+r'\section{Related work}', 1)
    if r'\label{tab:positioning}' not in body:
        body = body.replace(r'\section{Scope and limitations}', COMPARISON+'\n'+r'\section{Scope and limitations}', 1)
    start = body.index('Table~\\ref{tab:positioning} makes the methodological comparison explicit.')
    end = body.index(r'\end{table*}', start) + len(r'\end{table*}')
    comparison = body[start:end]
    body = body[:start] + body[end:]
    anchor = 'calibration alone cannot ensure that overlapping evidence will improve the identity ranking.'
    body = body.replace(anchor, anchor+'\n\n'+comparison, 1)
    (PAPER / 'manuscript_body.tex').write_text(body)
    source = (PAPER / 'manuscript.tex').read_text()
    header = source.split(r'\section{Introduction}', 1)[0]
    if 'IETkerneloutputdblcol' not in header:
        # USG loads cuted 2021 although this manuscript uses no strip environments.
        # Preserve the current kernel routines for standard two-column floats.
        pre = r'''\makeatletter
\let\IETkerneloutputdblcol\@outputdblcol
\let\IETkernelspecialoutput\@specialoutput
\newtoks\IETkerneloutput
\IETkerneloutput=\output
\makeatother
'''
        post = r'''\makeatletter
\let\@outputdblcol\IETkerneloutputdblcol
\let\@specialoutput\IETkernelspecialoutput
\output=\IETkerneloutput
\makeatother
'''
        header = pre + header.replace(r'\let\bibliographystyle\IETbibliographystyle', post+r'\let\bibliographystyle\IETbibliographystyle', 1)
        # Retain the template's font hierarchy with less excessive heading gaps.
        patches = r'''\patchcmd{\normalsize}{11.5\@p@t}{11.2\@p@t}{}{}
\patchcmd{\section}{-27\@p@t}{-14\@p@t}{}{}
\patchcmd{\subsection}{-23\@p@t}{-12\@p@t}{}{}
\patchcmd{\subsubsection}{-20\@p@t}{-10\@p@t}{}{}
'''
        header = header.replace(r'\patchcmd{\@maketitle}', patches+r'\patchcmd{\@maketitle}', 1)
    columns = r'''\let\IETOriginalBibitem\bibitem
\renewcommand{\bibitem}[2][]{\ifstrequal{#2}{crawford2020database}{\newpage}{}\IETOriginalBibitem[#1]{#2}}
'''
    source = header+body+'\n'+columns+'\\bibliography{refs}\n\\end{document}\n'
    source = source.replace('[tp]', '[!t]')
    # Defer the fourth float in a tall stack instead of overfilling that page.
    pattern = r'\\begin\{table\*\}\[!t\](?:(?!\\end\{table\*\}).)*?\\label\{tab:maincalibration\}'
    source, count = re.subn(pattern, lambda m: m.group().replace('[!t]', '[tp]', 1), source, flags=re.S)
    assert count == 1
    (PAPER / 'manuscript.tex').write_text(source)
    style = PAPER / 'ietnumeric.bst'
    style.write_text(style.read_text().replace('{ff~}{vv~}{ll}{, jj}', '{f.~}{vv~}{ll}{, jj}'))
    refs = (PAPER / 'refs.bib').read_text()
    if '@article{he2020fragnet,' not in refs:
        refs += r'''
@article{he2020fragnet,
 author={He, Sheng and Schomaker, Lambert},
 title={{FragNet}: Writer Identification using Deep Fragment Networks},
 journal={IEEE Trans. Inf. Forensics Secur.},
 volume={15}, pages={3013--3022}, year={2020},
 doi={10.1109/TIFS.2020.2981236}
}
'''
    (PAPER / 'refs.bib').write_text(refs)
    print('Revised literature synthesis, two comparison tables, organization and output-routine layout.')


if __name__ == '__main__':
    main()
