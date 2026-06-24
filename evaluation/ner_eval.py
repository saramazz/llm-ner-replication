"""
ner_eval.py — NER evaluation for GPT-4o zero-shot output.

Evaluates GPT-4o bracket-annotated predictions against gold annotations
using the M2 content-aligned matcher (see m2_matcher.py).

Usage
-----
    python evaluation/ner_eval.py \
        --annot  data/SC_test_loc_english_annot.csv \
        --raw    data/SC_test_eng.csv \
        --gpt    data/SC_test_loc_english_zero_shot.csv \
        --lang   english \
        --out    results_ner_english.json

Expected CSV formats
--------------------
--annot (gold annotation file):
    id, from, to, value, CUI
    - id:    document identifier
    - from:  character start offset (absolute, in raw text)
    - to:    character end offset
    - value: entity surface form
    - CUI:   UMLS concept ID; use 'CUI-less' / 'C0000000' for non-linkable spans

--raw (raw document text):
    id, text
    - id:   document identifier (must match annot)
    - text: full document text (no annotation markers)

--gpt (GPT-4o output):
    id, annotation
    - id:         document identifier (must match annot)
    - annotation: annotated text with [[entity]] bracket markers

Languages
---------
  english: CUI-less encoded as 'CUI-less'
  dutch:   CUI-less encoded as NaN (empty cell)
  italian: CUI-less encoded as 'C0000000'
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from m2_matcher import (
    RE_BRACKET,
    abs_to_lc,
    align_lines,
    bootstrap_f1,
    extract_abs,
    extract_lc,
    match_clean_cb,
    match_full,
    match_strict,
    prf1,
)

CUI_LESS_BY_LANG = {
    "english": {"CUI-less", "Cui-less"},
    "dutch":   {None, float("nan")},    # stored as NaN in the CSV
    "italian": {"C0000000"},
}


def is_cui_less(cui_val, lang: str) -> bool:
    """Return True if the CUI value represents a non-linkable (CUI-less) entity."""
    if lang == "dutch":
        return pd.isna(cui_val) or str(cui_val).strip() == ""
    return str(cui_val) in CUI_LESS_BY_LANG.get(lang, set())


def evaluate(
    annot_path: Path,
    raw_path:   Path,
    gpt_path:   Path,
    lang:       str,
) -> dict:
    """
    Run M2 NER evaluation for one language.

    Returns a dict with keys: full, naive_cb, clean_cb (each with P/R/F1/CI95).
    """
    annot_df = pd.read_csv(annot_path, dtype=str)
    raw_df   = pd.read_csv(raw_path,   dtype=str)
    gpt_df   = pd.read_csv(gpt_path,   dtype=str)

    raw_by = {r["id"]: r["text"]       for _, r in raw_df.iterrows()}
    gpt_by = {r["id"]: r["annotation"] for _, r in gpt_df.iterrows()}

    # Build per-doc CUI-bearing (cb) and CUI-less (cl) span lists
    cb_by: dict[str, list] = defaultdict(list)
    cl_by: dict[str, list] = defaultdict(list)

    for _, row in annot_df.iterrows():
        doc = row["id"]
        s, e = int(row["from"]), int(row["to"])
        ent  = str(row.get("value", raw_by.get(doc, "")[s:e]))
        if is_cui_less(row.get("CUI"), lang):
            cl_by[doc].append((s, e, ent))
        else:
            cb_by[doc].append((s, e, ent))

    common = sorted(
        (set(cb_by) | set(cl_by)) & set(gpt_by) & set(raw_by)
    )
    if not common:
        raise ValueError(
            "No document IDs match across annot / raw / gpt files. "
            "Check that the 'id' column is consistent."
        )

    full_docs:  dict[str, dict] = {}
    clean_docs: dict[str, dict] = {}

    for doc in common:
        raw_text = raw_by[doc]
        gpt_text = gpt_by[doc]

        # Parse GPT line-column spans
        gpt_clean, gpt_lc = extract_lc(gpt_text, RE_BRACKET)

        # Align gold lines to predicted lines
        lmap = align_lines(raw_text.split("\n"), gpt_clean.split("\n"))

        # Convert gold absolute offsets to line-column
        def to_lc(spans):
            out = []
            for s, e, t in spans:
                lc = abs_to_lc(raw_text, s, e)
                if lc:
                    out.append((lc[0], lc[1], lc[2], t))
            return out

        gold_cb  = to_lc(cb_by.get(doc, []))
        gold_cl  = to_lc(cl_by.get(doc, []))
        gold_all = gold_cb + gold_cl

        # Full evaluation (all gold)
        tp_ov, fp_ov, fn_ov  = match_full(gold_all, gpt_lc, lmap)
        tp_st, fp_st, fn_st  = match_strict(gold_all, gpt_lc, lmap)
        full_docs[doc] = {
            "tp": tp_ov, "fp": fp_ov, "fn": fn_ov,
            "tp_s": tp_st, "fp_s": fp_st, "fn_s": fn_st,
            "n_gold": len(gold_all),
        }

        # Clean CUI-bearing evaluation
        tp_c, fp_c, fn_c, unlink = match_clean_cb(gold_cb, gold_cl, gpt_lc, lmap)
        tp_cs, fp_cs, fn_cs      = match_strict(gold_cb, gpt_lc, lmap)[:3]
        clean_docs[doc] = {
            "tp": tp_c, "fp": fp_c, "fn": fn_c, "unlinkable": unlink,
            "tp_s": tp_cs, "fp_s": fp_cs, "fn_s": fn_cs,
            "n_gold": len(gold_cb),
        }

    # Aggregate
    def agg(docs):
        return (
            sum(d["tp"]  for d in docs.values()),
            sum(d["fp"]  for d in docs.values()),
            sum(d["fn"]  for d in docs.values()),
            sum(d["tp_s"] for d in docs.values()),
            sum(d["fp_s"] for d in docs.values()),
            sum(d["fn_s"] for d in docs.values()),
            sum(d["n_gold"] for d in docs.values()),
        )

    tp_f, fp_f, fn_f, tp_fs, fp_fs, fn_fs, ng_f = agg(full_docs)
    tp_c, fp_c, fn_c, tp_cs, fp_cs, fn_cs, ng_c = agg(clean_docs)
    unlink_tot = sum(d["unlinkable"] for d in clean_docs.values())

    n_cl = sum(len(cl_by.get(d, [])) for d in common)
    n_cb = sum(len(cb_by.get(d, [])) for d in common)

    ci_full  = bootstrap_f1(full_docs)
    ci_clean = bootstrap_f1(clean_docs)

    P_f,  R_f,  F_f  = prf1(tp_f,  fp_f,  fn_f)
    Ps_f, Rs_f, Fs_f = prf1(tp_fs, fp_fs, fn_fs)
    P_c,  R_c,  F_c  = prf1(tp_c,  fp_c,  fn_c)
    Ps_c, Rs_c, Fs_c = prf1(tp_cs, fp_cs, fn_cs)

    return {
        "n_docs":       len(common),
        "gold_total":   n_cb + n_cl,
        "cui_less":     n_cl,
        "cui_bearing":  n_cb,
        "full": {
            "TP": tp_f, "FP": fp_f, "FN": fn_f,
            "P": P_f, "R": R_f, "F1_overlap": F_f, "F1_strict": Fs_f,
            "CI95_overlap": list(ci_full),
        },
        "clean_cb": {
            "gold_CB": n_cb,
            "TP": tp_c, "FP": fp_c, "FN": fn_c, "unlinkable": unlink_tot,
            "P": P_c, "R": R_c, "F1_overlap": F_c, "F1_strict": Fs_c,
            "CI95_overlap": list(ci_clean),
        },
    }


def main():
    ap = argparse.ArgumentParser(
        description="M2 NER evaluation for GPT-4o bracket-annotated output."
    )
    ap.add_argument("--annot", required=True,
                    help="Gold annotation CSV (columns: id, from, to, value, CUI)")
    ap.add_argument("--raw",   required=True,
                    help="Raw document text CSV (columns: id, text)")
    ap.add_argument("--gpt",   required=True,
                    help="GPT-4o output CSV (columns: id, annotation)")
    ap.add_argument("--lang",  required=True,
                    choices=["english", "dutch", "italian"],
                    help="Language (determines CUI-less encoding)")
    ap.add_argument("--out",   default="results_ner.json",
                    help="Output JSON path")
    args = ap.parse_args()

    results = evaluate(
        Path(args.annot),
        Path(args.raw),
        Path(args.gpt),
        args.lang,
    )

    Path(args.out).write_text(json.dumps(results, indent=2))

    print(f"\nLanguage: {args.lang}   ({results['n_docs']} docs)")
    print(f"Gold total: {results['gold_total']}   "
          f"CUI-bearing: {results['cui_bearing']}   "
          f"CUI-less: {results['cui_less']}")
    for mode in ("full", "clean_cb"):
        r = results[mode]
        ci = r["CI95_overlap"]
        print(f"  {mode:<10}  "
              f"P={r['P']:.4f}  R={r['R']:.4f}  "
              f"F1_ov={r['F1_overlap']:.4f}  CI95=[{ci[0]:.4f},{ci[1]:.4f}]  "
              f"F1_strict={r['F1_strict']:.4f}")
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
