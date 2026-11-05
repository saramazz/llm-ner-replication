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
| `baselines/gliner_baseline.py`    | GLiNER zero-shot NER baseline (sliding window; overlap and strict F1) |
| `figures/figures_v2.py`           | Reproduce Figure 3 (self-contained, no data needed) |
| `prompts/gpt4o_annotation_prompt.txt` | Full few-shot annotation prompt used with GPT-4o, verbatim (Section S1 of the Supplementary Material) |
| `prompts/gliner_labels_*.txt`     | GLiNER entity labels (the 11 prompt categories; the generic labels of the ablation) |
| `Extr_Val_R/R/`                   | Corpus standardisation, annotation embedding (`[[span][CUI]]`), translation and annotation projection (R) |
| `src/01_annotation.py`            | The script that sent the prompt and the notes to GPT-4o |
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
    --annot   data/SC_test_loc_english_annot.csv \
    --raw     data/SC_test_eng.csv \
    --doc_ids data/doc_ids_131.txt \
    --labels  prompts/gliner_labels_gpt4o.txt \
    --out     results_gliner_english.json
```

GLiNER (`urchade/gliner_multi-v2.1`) truncates its input at 384 words, so each note is
processed in sliding windows of 384 words overlapping by 64. Two configurations are reported
(131 docs, overlap matching against the full gold):

| Labels | `--threshold` | English | Dutch | Italian |
|---|---|---|---|---|
| 11 prompt categories (`prompts/gliner_labels_gpt4o.txt`) | 0.5 (library default) | 0.22 | 0.11 | 0.15 |
| generic labels (`prompts/gliner_labels_generic.txt`) | 0.1 (selected on the training split) | 0.69 | 0.67 | 0.69 |

The label set and threshold were selected by overlap F1 on the 299 training notes, without
using the test notes; the full sweep is in Additional file 1, Section S6. `--window 0` (whole
note) gives the windowing ablation.
`--doc_ids` lists the 131 evaluated notes, one id per line (not distributed: the ids fall
under the data use agreement; see `data/README.md`).

---

## BEL evaluation

### Building the BEL input files

The linking queries are the gold mentions that GPT-4o recognised: gold rows matched
one-to-one by a predicted span under the M2 overlap matcher of `ner_eval.py`, queried by
their gold surface form. The restricted ("oracle") candidates are the recognised gold of
train and test. One CUI rule applies to every language: a valid CUI is linkable; empty,
`null`, `CUI-less` and `C0000000` become `C0000000` (not linkable); malformed values are
dropped from queries and candidates, and the script reports how many.

```bash
python evaluation/build_bel_queries.py \
    --test  data/SC_test_loc_english_annot.csv  data/SC_test_eng.csv  data/SC_test_loc_english_zero_shot.csv \
    --train data/SC_train_loc_english_annot.csv data/SC_train_eng.csv data/SC_train_loc_english_zero_shot.csv \
    --doc_ids data/doc_ids_131.txt \
    --out_queries data/bel/SC_test_loc_english_common_entities.csv \
    --out_oracle  data/bel/SC_oracle_loc_english_entities.csv
```

On the 131 test notes this gives 3,680 (English), 4,320 (Dutch) and 4,549 (Italian) queries,
the query sets of the paper. Do not use `src/06_bel_extract_common.py` from the original
pipeline for this: it pairs CUIs and spans with `zip()` over two different regexes, which
misaligns them after any marker without a `C\d+` CUI (all Dutch CUI-less mentions).

### Running the evaluation

```bash
python evaluation/bel_eval.py \
    --queries      data/bel/SC_test_loc_english_common_entities.csv \
    --oracle_train data/bel/SC_oracle_loc_english_entities.csv \
    --umls         data/umls_english.csv \
    --model        cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR \
    --cache_dir    embedding_cache \
    --out          bel_results_english.json
```

**Important:** the paper uses `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR`
(multilingual, XLM-R based). The original SapBERT
(`SapBERT-from-PubMedBERT-fulltext`) has a case-sensitivity issue: it maps Title-case
clinical terms to `[UNK]`, producing near-identical embeddings for single-word entities.
With the default `--encoding lower_cls`, entity texts are lowercased and encoded with CLS
pooling (the setting used for multilingual SapBERT in the paper). The paper's other two
encoders were run with sentence-transformers mean pooling on the original text; reproduce
them with `--encoding st`:

```bash
python evaluation/bel_eval.py ... --model sentence-transformers/all-MiniLM-L6-v2 --encoding st
python evaluation/bel_eval.py ... --model dmis-lab/biobert-v1.1 --encoding st
```

Embeddings are cached per model and encoding under `--cache_dir`.

---

## GPT-4o annotation

The annotation prompt is in [`prompts/gpt4o_annotation_prompt.txt`](prompts/gpt4o_annotation_prompt.txt),
verbatim as sent (Supplementary Material, Section S1). It states the annotation guidelines
with worked input-output examples (a few-shot prompt) and instructs GPT-4o to wrap disorder
mentions in `[[...]]` brackets while preserving the original text unchanged. The calling
code is [`src/01_annotation.py`](src/01_annotation.py) (model `gpt-4o`, temperature 0.5,
max_tokens 2000); it reads the prompt from `saved_results/annotation_guidelines_SC.txt`, so
copy the prompt file there.

The GPT-4o predictions used in the paper are available on request from the authors
(`data/predictions/`). They are not included in this public repository because the
annotation files contain clinical text derived from MIMIC-II, which cannot be
redistributed (PhysioNet data use agreement). Researchers who hold a valid PhysioNet
credential may contact the corresponding author.

**Re-running GPT-4o will not reproduce identical outputs** because of model updates;
the evaluation scripts work with any output that follows the `[[entity]]` convention.

### Translation pipeline

Run from the repository root, with the PhysioNet release placed under
`Extr_Val_R/Corpora_raw/CLEF2014/` (all `Extr_Val_R/Corpora_*` folders are git-ignored):

| Step | Script (`Extr_Val_R/R/Process_ShareClef/`) | What it does |
|---|---|---|
| 1 | `1_SC_{train,test}_standardize.R` | reads notes and annotations into tables |
| 3 | `3_SC_{train,test}_MakeInTextAnnotations.R` | wraps each gold mention in `[[span][CUI]]` by script (CUI-less: `[C0000000]`) |
| 4 | `4_SC_test_train_translate_GPT.R` | translates each note through the OpenAI API (Italian: `gpt-4o-mini`) |
| 5 | `5_SC_{train,test}_ProcessTranslationToCorpus.R` | extracts the markers with a regular expression and writes the translated corpus and gold |

Shared functions are in `Extr_Val_R/R/Tool_functions.R`. The Dutch corpus is the GPT-4 Turbo
(`gpt-4-0125-preview`) translation of Seinen et al. (JAMIA 2024), produced with the same
protocol. As in that work, the translated ShARe/CLEF corpora cannot be made publicly
available because of the PhysioNet data use agreement.

> **Data use.** Steps 4 and the GPT-4o annotation send note text to the OpenAI API.
> PhysioNet's data use agreement does not allow sending these data to third-party services
> unless a compliant route is in place (see
> <https://physionet.org/news/post/llm-responsible-use/>). The scripts document what was run;
> do not run them on PhysioNet data without such a route. API keys are read from the
> environment or a git-ignored `.env` (see `.env.example`); never commit a key.

---

## Citation

See [CITATION.cff](CITATION.cff) for the full citation metadata.
GitHub renders this as a **"Cite this repository"** button on the repo page.

If you prefer BibTeX, use the *Export BibTeX* option from that button,
or cite the paper directly once a DOI is assigned.
