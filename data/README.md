# Data Access and Format

## Corpus: SemEval-2014 Task 7 / ShARe-CLEF eHealth 2014

The experiments use the **ShARe/CLEF eHealth Evaluation Lab 2014** corpus
(SemEval-2014 Task 7, Subtask A), which consists of de-identified clinical notes
annotated with UMLS disorder mentions (the ShARe corpus, built on MIMIC-II v2.5 notes).

**The data cannot be redistributed.** Access requires PhysioNet credentialed access
and the Data Use Agreement of the ShARe/CLEF eHealth 2014 Task 2 project.

### How to obtain the data

1. Create a PhysioNet account at https://physionet.org
2. Become a credentialed user and complete the required human-subjects research training
3. Request access to https://physionet.org/content/shareclefehealth2014task2/ and sign
   its Data Use Agreement

**Corpus statistics:**
- Train: 199 documents + 100 development documents (used as training)
- Test:  133 documents
- Total: 432 clinical notes (discharge summaries, ECG/ECHO/radiology reports)
- Languages: English (original), Dutch and Italian (projected translations)

---

## Expected file formats

### NER evaluation (`evaluation/ner_eval.py`)

**Gold annotation file** (`--annot`):
```
id,from,to,value,CUI
doc_001,10,18,diabetes,C0011849
doc_001,42,55,hypertension,C0020538
doc_001,80,95,chest pain,CUI-less
```
- `id`:    document identifier (must match raw and GPT files)
- `from`:  character start offset in the raw document text (0-indexed)
- `to`:    character end offset (exclusive)
- `value`: entity surface form
- `CUI`:   UMLS concept ID; encoding of non-linkable spans:
  - English: `CUI-less` or `Cui-less`
  - Dutch:   empty cell (NaN)
  - Italian: `C0000000`

**Raw document file** (`--raw`):
```
id,text
doc_001,"Admission Date: ... Patient is a 67-year-old male with diabetes..."
```

**GPT-4o output file** (`--gpt`):
```
id,annotation
doc_001,"Admission Date: ... Patient is a 67-year-old male with [[diabetes]]..."
```
The annotation field contains the document text with entity mentions wrapped in
`[[...]]` markers. This is the direct output format of the zero-shot GPT-4o prompt.

### BEL evaluation (`evaluation/bel_eval.py`)

**Common entities file** (`--queries`, `--oracle_train`):
```
id,entity,CUI_cl
doc_001,diabetes,C0011849
doc_001,hypertension,C0020538
doc_001,chest pain,C0000000
```
These files contain the entity mentions that appear in both the gold annotation
and the GPT-4o output (the "common entities" used for entity linking evaluation).

**UMLS concept names** (`--umls`):
```
C0011849,diabetes mellitus
C0011849,DM
C0020538,hypertension
```
Two-column CSV (CUI, entity_text). Obtain from UMLS MRCONSO.RRF filtered to:
- Target language (ENG / DUT / ITA)
- Disorder semantic types:
  T020, T190, T049, T019, T047, T050, T037, T048, T191, T046, T184, T033

### Baseline scripts

**MedCAT** (`baselines/medcat_baseline.py`):
```json
// --docs: en_docs.json
[{"doc_id": "doc_001", "text": "Patient presents with..."}]

// --gold: en_gold.json
{"doc_001": [{"start": 18, "end": 25, "cui": "C0011849"}]}
```

**QuickUMLS** (`baselines/quickumls_baseline.py`): same format as MedCAT.

**GLiNER** (`baselines/gliner_baseline.py`): same format as MedCAT.

---

## UMLS resources

Beyond the corpus, you need a UMLS release (2024AB used in the paper):
- Download from https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html
- A UMLS license is free but requires registration

For QuickUMLS, build the database with:
```bash
python -m quickumls.install <umls_dir>/META <output_db_dir>
```

For the BEL realistic evaluation, extract language-specific concept names:
```bash
grep -P '\tENG\t' MRCONSO.RRF | awk -F'|' '{print $1","$9}' > umls_english.csv
grep -P '\tDUT\t' MRCONSO.RRF | awk -F'|' '{print $1","$9}' > umls_dutch.csv
grep -P '\tITA\t' MRCONSO.RRF | awk -F'|' '{print $1","$9}' > umls_italian.csv
```
