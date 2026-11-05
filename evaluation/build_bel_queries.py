"""
build_bel_queries.py — build the BEL input files that bel_eval.py expects.

BEL query set (as in the paper): the gold mentions recognised by GPT-4o, i.e. the gold
rows matched one-to-one by a predicted span under M2 overlap matching (all gold, file
order, the same greedy matcher as ner_eval.py). The query text is the gold surface form.

Gold label (CUI_cl), one rule for every language:
  - a valid CUI (C followed by 7 digits, not C0000000)          -> that CUI
  - empty / NaN / 'null' / 'CUI-less' (any case) / 'C0000000'   -> 'C0000000' (not linkable)
  - anything else (malformed)                                    -> row dropped and counted
Malformed rows are left out of both the queries and the restricted candidate set.

Outputs
-------
--out_queries: test mentions to link (restricted to --doc_ids when given)
--out_oracle:  restricted ("oracle") candidates = recognised gold of train + test, over
               every document that has a prediction, train first. Pass it to bel_eval.py
               as --oracle_train; bel_eval.py appends the queries and keeps the first row
               per surface form, which leaves this candidate set unchanged.

Usage
-----
    python evaluation/build_bel_queries.py \
        --test   data/SC_test_loc_english_annot.csv  data/SC_test_eng.csv  data/SC_test_loc_english_zero_shot.csv \
        --train  data/SC_train_loc_english_annot.csv data/SC_train_eng.csv data/SC_train_loc_english_zero_shot.csv \
        --doc_ids data/doc_ids_131.txt \
        --out_queries data/bel/SC_test_loc_english_common_entities.csv \
        --out_oracle  data/bel/SC_oracle_loc_english_entities.csv

Input formats are those of ner_eval.py (annot: id, from, to, [value], CUI; raw: id, text;
predictions: id, annotation with [[entity]] markers).

Do not use the extractor in the original pipeline (src/06_bel_extract_common.py): it pairs
CUIs and spans with zip() over two different regexes, so every marker without a C\\d+ CUI
shifts the CUIs of the following mentions in the same note.
"""

from __future__ import annotations
import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from m2_matcher import RE_BRACKET, abs_to_lc, align_lines, extract_lc, shares_token

NON_LINKABLE = "C0000000"
RE_CUI = re.compile(r"C\d{7}")
CUI_LESS_MARKERS = {"", "null", "nan", "cui-less", "c0000000"}


def cui_class(raw) -> str:
    """'cui' (linkable), 'cui_less' or 'malformed'. Language-independent."""
    if not isinstance(raw, str):
        return "cui_less"
    s = raw.strip()
    if s.lower() in CUI_LESS_MARKERS:
        return "cui_less"
    return "cui" if RE_CUI.fullmatch(s) else "malformed"


def _offset(x: str) -> int:
    f = float(x)                     # offsets may be stored as "306.0"
    if f != int(f):
        raise ValueError(f"non-integral offset {x!r}")
    return int(f)


def matched_gold(gold_lc: list, pred_lc: list, line_map: dict) -> list[int]:
    """Indices of gold spans matched by match_full's greedy one-to-one overlap rule."""
    used = [False] * len(pred_lc)
    hits = []
    for i, (gl, gs, ge, gt) in enumerate(gold_lc):
        mapped = line_map.get(gl, gl)
        for j, (pl, ps, pe, pt) in enumerate(pred_lc):
            if not used[j] and pl == mapped and max(gs, ps) < min(ge, pe) and shares_token(gt, pt):
                used[j] = True
                hits.append(i)
                break
    return hits


def recognised_gold(annot_path, raw_path, pred_path, doc_ids=None) -> pd.DataFrame:
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    annot, raw_df, pred_df = read(annot_path), read(raw_path), read(pred_path)
    raw_by = dict(zip(raw_df["id"], raw_df["text"]))
    pred_by = dict(zip(pred_df["id"], pred_df["annotation"]))

    by_doc: dict[str, list] = {}
    for r in annot.to_dict("records"):
        s, e = _offset(r["from"]), _offset(r["to"])
        text = r["value"] if "value" in annot.columns else raw_by.get(r["id"], "")[s:e]
        by_doc.setdefault(r["id"], []).append((s, e, text, r["CUI"]))

    docs = sorted(set(raw_by) & set(pred_by))
    if doc_ids is not None:
        missing = sorted(set(doc_ids) - set(docs))
        if missing:
            raise ValueError(f"{len(missing)} --doc_ids have no raw text or prediction")
        docs = sorted(doc_ids)

    rows = []
    for doc in docs:
        raw = raw_by[doc]
        pred_clean, pred_lc = extract_lc(pred_by[doc], RE_BRACKET)
        lmap = align_lines(raw.split("\n"), pred_clean.split("\n"))
        placed = []
        for s, e, text, cui in by_doc.get(doc, []):
            lc = abs_to_lc(raw, s, e)
            if lc is not None:
                placed.append(((*lc, text), text, cui))
        for i in matched_gold([p[0] for p in placed], pred_lc, lmap):
            _, text, cui = placed[i]
            cls = cui_class(cui)
            rows.append({"id": doc, "entity": text,
                         "CUI_cl": cui.strip() if cls == "cui" else NON_LINKABLE,
                         "cui_class": cls})
    return pd.DataFrame(rows, columns=["id", "entity", "CUI_cl", "cui_class"])


def _drop_malformed(df: pd.DataFrame, name: str) -> pd.DataFrame:
    counts = Counter(df["cui_class"])
    print(f"{name}: {len(df)} recognised gold mentions "
          f"(cui {counts['cui']}, cui_less {counts['cui_less']}, malformed {counts['malformed']} dropped)")
    return df[df["cui_class"] != "malformed"]


def main():
    ap = argparse.ArgumentParser(description="Build BEL query and restricted-candidate files.")
    ap.add_argument("--test", nargs=3, required=True, metavar=("ANNOT", "RAW", "PRED"))
    ap.add_argument("--train", nargs=3, required=True, metavar=("ANNOT", "RAW", "PRED"))
    ap.add_argument("--doc_ids", help="test documents to query, one id per line (default: all)")
    ap.add_argument("--out_queries", required=True)
    ap.add_argument("--out_oracle", required=True)
    args = ap.parse_args()

    ids = None
    if args.doc_ids:
        ids = {l.strip() for l in Path(args.doc_ids).read_text().splitlines() if l.strip()}

    queries = _drop_malformed(recognised_gold(*args.test, doc_ids=ids), "queries (test)")
    oracle = _drop_malformed(
        pd.concat([recognised_gold(*args.train), recognised_gold(*args.test)], ignore_index=True),
        "restricted candidates (train + test)")

    for df, path in ((queries, args.out_queries), (oracle, args.out_oracle)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    print(f"Wrote {args.out_queries} ({len(queries)} rows) and {args.out_oracle} "
          f"({len(oracle)} rows, {oracle['entity'].nunique()} distinct surface forms)")


if __name__ == "__main__":
    main()
