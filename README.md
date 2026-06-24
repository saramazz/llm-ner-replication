# Replication Code: Zero-shot Clinical NER with GPT-4o

> Mazzucato et al. (2026). *GPT-4o for zero-shot clinical disorder recognition:
> a cross-lingual evaluation on SemEval-2014 Task 7 / ShARe-CLEF 2014.*
> JAMIA Open (JAMIA-2026-0242).

This repository contains the evaluation code and baseline scripts needed to
replicate the methodology of the paper. The clinical corpus itself is not
included (access requires a PhysioNet credential — see [`data/README.md`](data/README.md)).

---

## What this repository contains

| Path | Description |
|------|-------------|
| `evaluation/m2_matcher.py` | M2 content-aligned span matcher (core NER evaluation logic) |
| `evaluation/ner_eval.py`   | NER evaluation script (GPT-4o output vs gold, with bootstrap CI) |
| `evaluation/bel_eval.py`   | Biomedical Entity Linking: oracle + realistic, accuracy@k + MRR |
| `baselines/medcat_baseline.py`    | MedCAT v1 baseline (overlap + strict F1) |
| `baselines/quickumls_baseline.py` | QuickUMLS P/R curve (Jaccard threshold sweep) |
| `figures/figures_v2.py`           | Reproduce Figure 2 (self-contained, no data needed) |
| `data/README.md`                  | Data access instructions and expected file formats |

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

GPU is recommended for the BEL embedding step (bel_eval.py) but not required.

---

## Reproducing Figure 2 (no data needed)

The figure contains locked numbers from the paper and can be reproduced without
the corpus:

```bash
cd figures
MPLBACKEND=Agg python figures_v2.py
# Outputs: results.svg and results.png
```

---

## NER evaluation

Evaluate GPT-4o bracket-annotated output against gold annotations using the
M2 content-aligned matcher:

```bash
python evaluation/ner_eval.py \
    --annot data/SC_test_loc_english_annot.csv \
    --raw   data/SC_test_eng.csv \
    --gpt   data/SC_test_loc_english_zero_shot.csv \
    --lang  english \
    --out   results_ner_english.json
```

The M2 matcher (described in Appendix B of the paper) handles the coordinate
drift that arises when GPT-4o reformats the input text. It:
1. Parses annotated output into line-column spans.
2. Aligns gold and predicted line sequences with `difflib`.
3. Matches spans on aligned lines using character-level overlap.
4. Applies a `shares_token` guard to prevent cross-line false positives.

Three evaluation modes are reported:
- **full**: all gold spans (CUI-bearing + CUI-less)
- **clean_cb**: CUI-bearing gold only; predictions on CUI-less spans are
  classified as *unlinkable* (correct span, no UMLS mapping) rather than FP

---

## MedCAT baseline

```bash
python baselines/medcat_baseline.py \
    --model_pack data/medcat_model/umls_sm_pt2ch_533bab5115c6c2d6.zip \
    --docs data/en_docs.json \
    --gold data/en_gold.json \
    --out  medcat_results.json
```

The model pack (`umls_sm_pt2ch_533bab5115c6c2d6.zip`) was obtained from the MedCAT
model zoo. It requires `medcat==1.10.x` (the paper results were produced with `medcat==1.10.3`; not compatible with 2.x).

---

## QuickUMLS baseline

```bash
# Build the QuickUMLS database first (one-time)
python -m quickumls.install <path_to_umls_meta_dir> data/quickumls_db

# Run the P/R curve sweep
python baselines/quickumls_baseline.py \
    --db   data/quickumls_db \
    --docs data/en_docs.json \
    --gold data/en_gold.json \
    --out  quickumls_pr_curve.csv \
    --verify
```

Paper operating points (English, 131 test docs):
- thr=0.70: P=0.428, R=0.677, F1=0.524
- thr=0.80: P=0.634, R=0.649, F1=0.641

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

**Important**: the paper uses `cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR`
(multilingual XLM-R based). The original SapBERT (`SapBERT-from-PubMedBERT-fulltext`)
has a case-sensitivity bug: it maps Title-case clinical terms to `[UNK]`, producing
identical embeddings for all single-word entities. All entity texts are lowercased
before encoding to defend against similar issues in any cased model.

---

## GPT-4o annotation

The zero-shot GPT-4o predictions used in the paper were generated with the
`gpt-4o` model. The exact API snapshot cannot be confirmed from the code (no
date-pinned model string was logged at annotation time). The prompt instructs
the model to wrap disorder mentions with `[[...]]` brackets. **Re-running GPT-4o
will not reproduce identical outputs** due to model updates; the evaluation
scripts work with any output following the `[[entity]]` annotation convention.

The annotation prompt is included in the paper (Appendix A).

---

## Citation

```bibtex
@article{mazzucato2026gpt4o,
  title   = {GPT-4o for zero-shot clinical disorder recognition:
             a cross-lingual evaluation on SemEval-2014 Task 7 / ShARe-CLEF 2014},
  author  = {Mazzucato, Sara and ...},
  journal = {JAMIA Open},
  year    = {2026},
  note    = {JAMIA-2026-0242}
}
```
