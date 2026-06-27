# Advancements in Multilingual Biomedical Natural Language Processing: Exploring Large Language Models for Named Entity Recognition and Linking

Replication code for the cross-lingual (English, Dutch, Italian) evaluation of a
zero-shot GPT-4o + embedding-based entity-linking pipeline on the SemEval-2014 Task 7
(ShARe/CLEF eHealth 2014 Task 2) disorder corpus.

> Mazzucato S, et al. *Advancements in Multilingual Biomedical Natural Language
> Processing: Exploring Large Language Models for Named Entity Recognition and Linking.*
> 2026. Manuscript under review.

This repository contains the evaluation code and baseline scripts needed to replicate
the methodology of the paper. The clinical corpus and GPT-4o predictions are not
included (see *Data access* sections below).

---

## What this repository contains

| Path | Description |
|------|-------------|
| `evaluation/m2_matcher.py` | M2 content-aligned span matcher (core NER evaluation logic) |
| `evaluation/ner_eval.py`   | NER evaluation (GPT-4o output vs gold, with bootstrap CI) |
| `evaluation/bel_eval.py`   | Biomedical entity linking: oracle + realistic, accuracy@k + MRR |
| `baselines/medcat_baseline.py`    | MedCAT baseline (overlap + strict F1, concept accuracy) |
| `baselines/quickumls_baseline.py` | QuickUMLS P/R curve (Jaccard threshold sweep) |
| `baselines/gliner_baseline.py`    | GLiNER zero-shot NER baseline (overlap F1) |
| `figures/figures_v2.py`           | Reproduce Figure 2 (self-contained, no data needed) |
| `prompts/gpt4o_annotation_prompt.txt` | Full zero-shot annotation prompt used with GPT-4o (Section S1 of the Supplementary Material) |
| `data/predictions/`               | GPT-4o zero-shot predictions — available on request from the authors (clinical text is derived from MIMIC-II and cannot be redistributed publicly; requires PhysioNet DUA) |
| `data/README.md`                  | Data-access instructions and expected file formats |
| `requirements.txt`                | Pinned dependencies |

---

## Setup

```bash
# 1. Create and activate a Python environment
conda create -n llm-ner-replication python=3.10
conda activate llm-ner-replication

# 2. Install dependencies
pip install -r requirements.txt

# 3. For MedCAT, download the base spaCy model
python -m spacy download en_core_web_md
```

A GPU is recommended for the BEL embedding step (`bel_eval.py`) but not required.

---

## Reproducing Figure 2 (no data needed)

The figure uses the locked numbers from the paper and can be reproduced without the
corpus:

```bash
cd figures
MPLBACKEND=Agg python figures_v2.py
# Outputs: results.svg and results.png
```

---

## NER evaluation

Evaluate GPT-4o bracket-annotated output against gold annotations using the M2
content-aligned matcher:

```bash
python evaluation/ner_eval.py \
    --annot data/SC_test_loc_english_annot.csv \
    --raw   data/SC_test_eng.csv \
    --gpt   data/predictions/SC_test_loc_english_zero_shot.csv \
    --lang  english \
    --out   results_ner_english.json
```

The M2 matcher (see the Methods, *Evaluation*, in the paper) handles the coordinate
drift that arises when GPT-4o reformats the input text. It:
1. parses annotated output into line-column spans;
2. aligns gold and predicted line sequences with `difflib`;
3. matches spans on aligned lines using character-level overlap;
4. applies a `shares_token` guard to prevent cross-line false positives.

Two gold settings are reported, each under both the overlap and the strict
(exact-boundary) criterion:
- **full**: all gold spans (CUI-bearing + CUI-less);
- **clean_cb**: CUI-bearing gold only; predictions overlapping a CUI-less gold span are
  counted as *unlinkable* (correct span, no UMLS mapping) rather than as false positives.

---

## MedCAT baseline

```bash
python baselines/medcat_baseline.py \
    --model_pack data/medcat_model/umls_sm_pt2ch_533bab5115c6c2d6.zip \
    --docs data/en_docs.json \
    --gold data/en_gold.json \
    --out  medcat_results.json
```

The model pack (`umls_sm_pt2ch_533bab5115c6c2d6.zip`) is a UMLS "small" (`umls_sm`) pack
from the MedCAT model repository; downloading it **requires a valid UMLS licence**. It
was run with `medcat==1.16.12` (v1 API; not 2.x).

---

## QuickUMLS baseline

```bash
# Build the QuickUMLS database first (one-time; requires a UMLS licence)
python -m quickumls.install <path_to_umls_meta_dir> data/quickumls_db

# Run the P/R curve sweep
python baselines/quickumls_baseline.py \
    --db   data/quickumls_db \
    --docs data/en_docs.json \
    --gold data/en_gold.json \
    --out  quickumls_pr_curve.csv \
    --verify
```

Paper operating points (English, 131 test documents, overlap matching):
- thr=0.70: P=0.428, R=0.677, F1=0.524
- thr=0.80: P=0.634, R=0.649, F1=0.641

---

## GLiNER baseline

```bash
python baselines/gliner_baseline.py \
    --docs  data/en_docs.json \
    --gold  data/en_gold.json \
    --model urchade/gliner_medium-v2.1 \
    --threshold 0.5 \
    --out   gliner_results.json
```

The script runs GLiNER zero-shot using the 12 UMLS disorder semantic-type labels
as entity types. Paper results (English, 131 docs, overlap matching):
P=0.44, R=0.11, F1=0.19. Dutch F1=0.13, Italian F1=0.11.

---

## BEL evaluation

```bash
python evaluation/bel_eval.py \
    --queries      data/bel/SC_test_loc_english_common_entities.csv \
    --oracle_train data/bel/SC_train_loc_english_common_entities.csv \
    --umls         data/umls_english.csv \
    --model        cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR \
    --cache_dir    embedding_cache \
    --out          bel_results_english.json
```

**Important:** the paper uses `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR`
(multilingual, XLM-R based). The original SapBERT
(`SapBERT-from-PubMedBERT-fulltext`) has a case-sensitivity issue: it maps Title-case
clinical terms to `[UNK]`, producing near-identical embeddings for single-word entities.
All entity texts are lowercased before encoding to guard against this in any cased model.

---

## GPT-4o annotation

The full zero-shot annotation prompt is in [`prompts/gpt4o_annotation_prompt.txt`](prompts/gpt4o_annotation_prompt.txt)
(reproduced verbatim from Supplementary Material, Section S1). It instructs GPT-4o to
wrap disorder mentions in `[[...]]` brackets while preserving the original text unchanged.

The GPT-4o predictions used in the paper are available on request from the authors
(`data/predictions/`). They are not included in this public repository because the
annotation files contain clinical text derived from MIMIC-II, which cannot be
redistributed (PhysioNet data use agreement). Researchers who hold a valid PhysioNet
credential may contact the corresponding author.

**Re-running GPT-4o will not reproduce identical outputs** because of model updates;
the evaluation scripts work with any output that follows the `[[entity]]` convention.

### Translation pipeline

The Dutch and Italian corpora were produced by translating the English ShARe/CLEF
corpus with GPT-4 Turbo, using span-preserving markers `[[span][CUI]]` to track entity
boundaries through translation. The translation code is not included in this repository.
The translated corpus and predictions are available on request together with the
English predictions (same PhysioNet credential requirement applies).

---

## Citation

See [CITATION.cff](CITATION.cff) for the full citation metadata.
GitHub renders this as a **"Cite this repository"** button on the repo page.

If you prefer BibTeX, use the *Export BibTeX* option from that button,
or cite the paper directly once a DOI is assigned.
