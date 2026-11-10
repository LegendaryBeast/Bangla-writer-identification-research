# Literature and citation audit — 8 November 2026

Scope: the 22 unique citation keys used by `paper/wid_manuscript.tex` at the start of this audit. This audit checked primary publisher, author, conference and dataset records. A verified bibliographic record is not equivalent to a verified experimental protocol. No manuscript or bibliography was edited in this task.

## Changes supported by the sources

1. **Do not suggest that Morera et al. (2018) evaluated the same writers in training and testing.** Its accessible full text explicitly states that training and test individuals are separated (§2.2). It evaluates demographic classification on IAM and KHATT, using words extracted from lines and voting to classify a line. Its IAM handedness overall accuracy is 90.70%, with right/left recall 93.02%/63.40%; the balanced recall calculated from those two values is 78.21%. KHATT overall accuracy is 70.91%, with recalls 74.24%/38.64% (balanced recall 56.44%, calculated). These overall accuracies are not directly comparable to accuracy on BanglaHand's balanced cohort. Metadata in the bibliography matches the publisher. [Publisher full text](https://onlinelibrary.wiley.com/doi/10.1155/2018/3891624).

2. **Rahmanian and Shayegan (2021): report the result accurately, but mark writer separation unverified.** The primary publisher abstract confirms IAM/English and KHATT/Arabic, DenseNet201/InceptionV3/Xception, 84% IAM gender accuracy and 99.14% KHATT handedness accuracy. The full experimental section is subscription-only in the accessible record; this audit could not verify its split, class balance, aggregation or absence/presence of writer overlap. Do not attribute leakage to this work without inspecting the full protocol. All existing bibliographic fields match the publisher. [Publisher record](https://link.springer.com/article/10.1007/s11042-020-10170-7).

3. **Al-Maadeed et al. (2016): prefer the reported experiment/table to an inconsistent conclusion.** The accessible full text describes 121 writers, Arabic/English and fuzzy feature reduction with k-NN. §4.5.2 reports precision 71.99%, recall 71.87%, F-measure 71.93%, and accuracy 71.86 ± 7.74%. §4.5 states a 95%/5% training/validation split but does not establish clearly separated writers. Its conclusion confusingly describes 83.43% as detection performance although the main text identifies that number as a maximum reduction rate. Keep an approximate 72% F-measure if a number is needed, rather than asserting 83.43% recognition accuracy. [Publisher full text](https://link.springer.com/article/10.1186/s13640-015-0097-y).

4. **Known-attribute fusion is established prior work.** Jain, Dass and Nandakumar (2004) describe combining fingerprint recognition with gender, ethnicity and height, reporting roughly five percentage points improvement in a 263-user dataset. The new contribution should be the evaluation of handedness utility under the specified Bangla protocol, not the general idea of soft-biometric fusion. The bibliography metadata is consistent with the author copy and publisher record. [Author manuscript](https://biometrics.cse.msu.edu/Publications/SoftBiometrics/JainDassNandakumar_SoftBiometrics_ICBA2004.pdf), [publisher chapter](https://link.springer.com/chapter/10.1007/978-3-540-25948-0_99). The author PDF was search-index accessible but direct opening timed out; this audit does not claim a complete fresh reading of its methods.

5. **Correct the absolute Bangla data-availability claim.** BDSHWA was published on 28 April 2026; its primary record describes Bengali–English online handwriting from 29 participants, reporting 28 right-handed and one left-handed participant. This is a different acquisition modality and lacks sufficient left-handed representation for the proposed balanced independent replication. The statement that no other Bangla collection has any handedness labels is nevertheless too broad. [Dataset record](https://data.mendeley.com/datasets/99t9jhvksv/1).

6. **Resolve the Platt year/version discrepancy.** The existing entry gives a chapter in the MIT Press book as 1999. The publisher's primary book record dates *Advances in Large-Margin Classifiers* to 29 September 2000. A 1999 preprint also circulated. If citing the published book chapter, use 2000 and the publisher's hyphenated book title; retain the existing citation key if useful. If intentionally citing the 1999 preprint, identify it as that version. The publisher book page verifies the editors, publisher and date; it did not expose a chapter-level table of contents, so chapter-title variants should not be silently reconstructed from secondary references. [MIT Press book record](https://mitpress.mit.edu/9780262194488/advances-in-large-margin-classifiers/).

7. **Remove the negative coverage matrix based on titles/abstracts.** The accessible sources establish several tasks positively, but absence of a task from an abstract is insufficient to mark it as absent from an entire paper. A compact positive comparison of objective, data, evaluation unit and verified writer split is defensible. “Not verified in this audit” is preferable to an unsupported cross. No source examined here establishes that the precise Bangla experiment was previously published, but this is not a systematic novelty proof.

## Citation-by-citation metadata and claim checks

“Matches” below means the supplied title, authors, venue, year, volume/pages or article number, and identifier are consistent with the primary record to the extent it exposes them. It does not certify every experimental detail or the local implementation.

| Citation key | Primary record and verified metadata | Status / permissible use |
|---|---|---|
| `jain2004soft` | Jain, Dass, Nandakumar; ICBA 2004, LNCS 3072, pp. 731–738; DOI 10.1007/978-3-540-25948-0_99. [Publisher](https://link.springer.com/chapter/10.1007/978-3-540-25948-0_99), [author copy](https://biometrics.cse.msu.edu/Publications/SoftBiometrics/JainDassNandakumar_SoftBiometrics_ICBA2004.pdf). | Matches. General auxiliary-trait recognition, not writer identification. |
| `dantcheva2016soft` | Dantcheva, Elia, Ross; *IEEE TIFS* 11(3), 441–467 (2016); DOI 10.1109/TIFS.2015.2480381. [Author institutional copy](https://www.eurecom.edu/publication/4587/download/cm-publi-4587.pdf), [IEEE](https://ieeexplore.ieee.org/document/7273870). | Matches. Supports overview of extraction, indexing and fusion; not new experimental evidence for this manuscript's threshold. |
| `liwicki2011gender` | Liwicki, Schlapbach, Bunke; *Pattern Analysis and Applications* 14, 87–92 (2011); DOI 10.1007/s10044-010-0178-6. [Publisher](https://link.springer.com/article/10.1007/s10044-010-0178-6). | Matches. Gender classification combining online/offline information; final combined test accuracy 67.57%. Online-first 2010 does not invalidate final issue year 2011. |
| `siddiqi2015gender` | Siddiqi, Djeddi, Raza, Souici-Meslati; *Pattern Analysis and Applications* 18, 887–899 (2015); DOI 10.1007/s10044-014-0371-0. [Publisher](https://link.springer.com/article/10.1007/s10044-014-0371-0). | Matches. Scanned handwriting gender classification. Do not infer its full split or absence of auxiliary experiments from the abstract. |
| `hassaine2013icdar` | Hassaïne, Al-Maadeed, Aljaam, Jaoua; ICDAR 2013, pp.1417–1421; DOI 10.1109/ICDAR.2013.286. [IEEE record](https://ieeexplore.ieee.org/document/6628847), [conference organiser](https://iapr.org/archives/icdar2013/program/competitions.html), [IAPR TC11 dataset](https://tc11.cvc.uab.es/datasets/GenderIdentifify2013_1). | Core metadata matches. The author-provided paper abstract reports 194 teams. This is a gender competition, not evidence of attribute-assisted writer ID. |
| `bouadjenek2016soft` | Bouadjenek, Nemmour, Chibani; *Applied Soft Computing* 46, 980–990 (2016); DOI 10.1016/j.asoc.2015.10.021. [Publisher record](https://www.sciencedirect.com/science/article/abs/pii/S1568494615006584), [authors' institutional publications](https://lisic.usthb.dz/index.php/144-2/). | Matches. Positive description: gender/age/handedness prediction using SVM predictors and fuzzy MIN–MAX aggregation on Arabic/English data. Do not confuse predictor combination with attribute-assisted identification. |
| `almaadeed2016handedness` | Al-Maadeed, Ferjani, Elloumi, Jaoua; *EURASIP Journal on Image and Video Processing* 2016, article 1; DOI 10.1186/s13640-015-0097-y. [Publisher](https://link.springer.com/article/10.1186/s13640-015-0097-y). | Matches; use the protocol/result qualifications above. Article number 1 is not a one-page paper. |
| `morera2018handedness` | Morera, Sánchez, Vélez, Moreno; *Complexity* 2018, article 3891624, 14 pp.; DOI 10.1155/2018/3891624. [Publisher full text](https://onlinelibrary.wiley.com/doi/10.1155/2018/3891624). | Matches. Writer-disjoint train/test is expressly stated. |
| `rahmanian2021handedness` | Rahmanian, Shayegan; *Multimedia Tools and Applications* 80, 35341–35364 (2021); DOI 10.1007/s11042-020-10170-7. [Publisher](https://link.springer.com/article/10.1007/s11042-020-10170-7). | Matches. 99.14% is specifically KHATT handedness accuracy. Full protocol unverified. |
| `banglahand2026data` | Prappo, Hassan, Ali, Dhar, Rahman; *BanglaHand: A Dataset of Handwritten Paragraphs for Writer-Independent Handedness Recognition*, Mendeley Data V1, 25 October 2026; DOI 10.17632/n4sjstdzfy.1. [Record](https://data.mendeley.com/datasets/n4sjstdzfy/1). | Real accessible primary record. Prefer full deposited title over the shortened `[dataset] BanglaHand`. It confirms42 writers,21/21 hands, 336 photographs. |
| `bulacu2007text` | Bulacu, Schomaker; *IEEE TPAMI*29(4), 701–717 (2007); DOI 10.1109/TPAMI.2007.1009. [Author institutional paper](https://www.ai.rug.nl/~mbulacu/tpami2007-bulacu-schomaker.pdf). | Matches. Supports contour directional/textural and allographic distributions for writer recognition. |
| `schomaker2007codebooks` | Schomaker, Franke, Bulacu; *Pattern Recognition Letters*28(6), 719–727 (2007); DOI 10.1016/j.patrec.2006.08.005. [Author institutional paper](https://www.ai.rug.nl/~mbulacu/prl2007-schomaker-franke-bulacu.pdf). | Matches. Supports codebooks of fragmented connected-component contours. |
| `kaufman2012leakage` | Kaufman, Rosset, Perlich, Stitelman; *ACM TKDD*6(4), article 15, pp. 1–21 (2012); DOI 10.1145/2382577.2382579. [ACM](https://dl.acm.org/doi/10.1145/2382577.2382579). | Matches journal version. Do not replace with the three-author 2011 conference version without changing metadata. Leakage depends on what is legitimately available for the target deployment. |
| `prappo2026handedness` | Author manuscript described locally as submitted to PRL2026. [BanglaHand record](https://data.mendeley.com/datasets/n4sjstdzfy/1) lists a companion methodology submission with a different title. | **Not independently verified as a public manuscript or published article.** Either provide a stable accessible preprint with the exact current title or describe as unpublished/submitted. Do not make its uninspectable methods essential to reproducibility. |
| `mridha2021banglawriting` | Mridha, Ohi, Ali, Emon, Kabir; *Data in Brief*34, 106633 (2021); DOI 10.1016/j.dib.2020.106633. [Full article in PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744928/), [author preprint](https://arxiv.org/abs/2011.07499). | Matches.260 writers; one page per writer. Published online2020, issue 2021. Supports dataset availability, not a directly comparable writer-ID accuracy benchmark. |
| `majid2018bangla` | Nishatul Majid and Elisa H. Barney Smith; ICFHR 2018, 380–385; DOI 10.1109/ICFHR-2018.2018.00073. [Author institutional publication](https://scholarworks.boisestate.edu/electrical_facpubs/405/). | Matches.100 writers and95.4% isolated-character recognition are supported by this paper record; that percentage is not writer identification. A separate poster reports96.8% under another dataset comparison, so keep source/result tied together. |
| `paul2024bangla` | Paul, Dutta, Sarkar, Roy, Das; *ETRI Journal*46(4), 648–659 (2024); DOI 10.4218/etrij.2023-0188. [Publisher full text](https://onlinelibrary.wiley.com/doi/10.4218/etrij.2023-0188). | Matches.94.54% is the reported 100-writer text-dependent verification result. Do not compare it numerically with closed-set text-independent top-1 identification. |
| `he2016resnet` | He, Zhang, Ren, Sun; CVPR 2016, 770–778; DOI 10.1109/CVPR.2016.90. [CVF accepted paper](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html). | Matches. Architecture citation does not identify the exact frozen weights or preprocessing; those remain implementation specifications. |
| `tan2019efficientnet` | Tan, Le; ICML 2019/PMLR 97, 6105–6114. [Official proceedings](https://proceedings.mlr.press/v97/tan19a.html). | Matches. Add the official URL if desired. The current entry does not need an invented DOI. |
| `dosovitskiy2021vit` | Dosovitskiy et al.; ICLR2021. [Author camera-ready record](https://arxiv.org/abs/2010.11929), [OpenReview](https://openreview.net/forum?id=YicbFdNTTy). | Author list/title match. arXiv record identifies the 2021 ICLR camera-ready version; initial preprint 2020. OpenReview presents browser verification in this environment. |
| `platt1999probabilistic` | John C.Platt; chapter in Smola,Bartlett,Schölkopf,Schuurmans (eds.), *Advances in Large-Margin Classifiers*, MIT Press. [Publisher book](https://mitpress.mit.edu/9780262194488/advances-in-large-margin-classifiers/). | Change year to 2000 when citing published book, or explicitly cite the 1999 preprint; see version caveat above. |
| `papadatou2020handedness` | Papadatou-Pastou,Ntolka,Schmitz,Martin,Munafò,Ocklenburg,Paracchini; *Psychological Bulletin*146(6), 481–524 (2020); DOI 10.1037/bul0000229. [Author institutional record](https://research-repository.st-andrews.ac.uk/handle/10023/19889). | Matches. Supports cultural moderation and an overall 10.6% prevalence estimate; does not establish a 1–2% prevalence for Bangladesh or this recruitment population. |

## Compact related-work wording for the revision

The following is proposed author prose, not a quotation. Citation keys refer to the existing bibliography.

> Auxiliary traits have long been fused with primary biometric scores, including fingerprint recognition with demographic information (`jain2004soft,dantcheva2016soft`). Handwriting studies have investigated gender and handedness prediction using handcrafted descriptors and neural networks (`liwicki2011gender,siddiqi2015gender,bouadjenek2016soft,almaadeed2016handedness,morera2018handedness,rahmanian2021handedness`). Morera et al. explicitly separate training and test writers; differences in class balance, evaluation unit and split prevent direct comparison of published accuracies. Here, we measure the downstream change in closed-set Bangla writer identification when handedness is known or predicted, and evaluate a conditional approximation for hard filtering alongside controls for writer information. These results characterise the tested cohort and systems rather than a universal accuracy requirement for soft-biometric fusion.

Recommended limitation wording:

> We did not identify a suitable independent offline Bangla dataset with sufficient left-handed representation for balanced replication. BDSHWA includes Bengali–English online handwriting but reports only one left-handed participant among 29 (`bdshwa2026data`), so its modality and class composition differ substantially from this experiment.

Do not keep “first to address all columns,” “none of these studies,” “errors were unrelated to hand,” or “far more accurately than offline handwriting currently allows” as literature-backed conclusions. The independent empirical analyses must establish the narrower claims made in the revised paper.

## Suggested added dataset entry

```bibtex
@misc{bdshwa2026data,
  author = {Ahmed, Md Roton and Rahim, Md Abdur and Zaman, Md. Rifat and
            Mian, Riaz-ul-Haque and Ahmed, Md. Toukir and Miah, Abu Saleh Musa},
  title = {{BDSHWA}: {Bengali--English} Online Handwriting Dataset for Forensic Biometric Analysis},
  howpublished = {Mendeley Data, V1},
  year = {2026},
  doi = {10.17632/99t9jhvksv.1},
  url = {https://data.mendeley.com/datasets/99t9jhvksv/1}
}
```

The spelling and contributor order above follow the deposited primary record. This audit checks the record's metadata, not all files or the validity of its claimed model performance.

## Pattern Recognition Letters format check

The live [ScienceDirect Guide for Authors](https://www.sciencedirect.com/journal/pattern-recognition-letters/publish/guide-for-authors) returned 403, including via the Elsevier redirect. Consequently this audit cannot certify every current 2026 submission requirement.

The accessible official [Elsevier authorship and formatting PDF](https://legacyfileshare.elsevier.com/promis_misc/patrec-authorship-and-formatting.pdf), bearing © 2023, states on its fourth PDF page:

- Seven total manuscript pages, with one additional page after revision; narrative, figures, tables and references are included.
- An abstract no longer than 200 words.
- Use `elsarticle.cls` and additionally `prletter.sty`.

Keep the revised abstract at or below 200 words and the initial submission at or below seven pages. These choices satisfy the stricter accessible official guidance. Do not use a third-party 250-word listing to relax the official template's 200-word limit. Treat graphical-abstract/highlights requirements and current upload form requirements as unverified until the live submission guide can be inspected.

## Remaining verification limits

- The complete Rahmanian2021, Liwicki2011 and Siddiqi2015 experimental protocols were not accessible from their publisher previews; no negative coverage or leakage findings are assigned to them.
- The submitted companion paper has no verified stable public manuscript location in the supplied citation. Its metadata and reported 77.4% result require author-provided traceability.
- This was a targeted audit of cited sources, not an exhaustive systematic search establishing a global priority claim.
- Numerical comparisons across these studies involve different tasks, scripts, class priors and sampling units; a common percentage scale does not make them comparable benchmarks.
