#!/usr/bin/env python3
"""
gliner_baseline.py — GLiNER zero-shot NER baseline (sliding window).

Runs GLiNER (Zaratiana et al., 2024) on the raw test notes in the same zero-shot,
no-fine-tuning setting used for GPT-4o, and scores it with the M2 matcher
(evaluation/m2_matcher.py) against the full gold, overlap and strict.

GLiNER truncates every input to config.max_len words (384 for gliner_multi-v2.1),
and every evaluation note is longer, so a whole note is read only up to that limit.
Each note is therefore split into windows of --window words (default: max_len)
overlapping by --overlap words. Offsets are mapped back to the note, and each
window keeps only the entities that start in the part it owns (every overlap is
split at its midpoint). --window 0 passes whole notes (truncated by GLiNER).

Paper results (131 test docs, overlap matching against full gold; windowed,
the 11 disorder categories of the GPT-4o prompt as labels, threshold 0.5):
  English F1 0.22 / Dutch F1 0.11 / Italian F1 0.15
The whole-note and generic-label settings are in Additional file 1, Section S6.

Setup
-----
    pip install gliner

Usage
-----
    python baselines/gliner_baseline.py \
        --annot  data/SC_test_loc_english_annot.csv \
        --raw    data/SC_test_eng.csv \
        --labels prompts/gliner_labels_gpt4o.txt \
        --out    results_gliner_english.json

    # Whole-note ablation with the generic labels:
    python baselines/gliner_baseline.py ... --window 0 \
        --labels prompts/gliner_labels_generic.txt

Input formats are those of evaluation/ner_eval.py (--annot: id, from, to, CUI[, value];
--raw: id, text). --doc_ids restricts the run to the ids listed one per line (the
paper evaluates 131 of the 133 test notes, see data/README.md).
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from m2_matcher import abs_to_lc, match_full, match_strict, prf1  # noqa: E402


def load_labels(path: str) -> list[str]:
    """One entity label per line; '#' lines are comments."""
    labels = [l.strip() for l in Path(path).read_text().splitlines()
              if l.strip() and not l.startswith("#")]
    if not labels:
        raise SystemExit(f"no labels in {path}")
    return labels


def make_windows(model, text: str, window: int, overlap: int):
    """Windows over GLiNER's own words.

    Returns ([(char_start, char_end, own_start, own_end)], [words per window]).
    """
    words = list(model.data_processor.words_splitter(text))
    if not words:
        return [], []
    if window <= 0:
        return [(0, len(text), 0, len(text))], [len(words)]
    spans, a = [], 0
    while True:
        b = min(a + window, len(words))
        spans.append((a, b))
        if b == len(words):
            break
        a = b - overlap
    out = []
    for k, (a, b) in enumerate(spans):
        own_a = 0 if k == 0 else words[(a + spans[k - 1][1]) // 2][1]
        own_b = len(text) if k == len(spans) - 1 else words[(spans[k + 1][0] + b) // 2][1]
        out.append((words[a][1], words[b - 1][2], own_a, own_b))
    return out, [b - a for a, b in spans]


def predict_doc(model, text: str, labels: list[str], args):
    """Entities with absolute offsets in the note, deduplicated across windows."""
    wins, sizes = make_windows(model, text, args.window, args.overlap)
    if not wins:
        return [], []
    results = model.inference([text[s:e] for s, e, _, _ in wins], labels,
                              threshold=args.threshold, batch_size=args.batch_size)
    best = {}
    for (s, _, own_a, own_b), ents in zip(wins, results):
        for ent in ents:
            start, end = ent["start"] + s, ent["end"] + s
            if own_a <= start < own_b:
                key = (start, end)
                if key not in best or ent["score"] > best[key]["score"]:
                    best[key] = {"start": start, "end": end, "label": ent["label"], "score": ent["score"]}
    return sorted(best.values(), key=lambda x: x["start"]), sizes


def read_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description="GLiNER zero-shot NER baseline (sliding window).")
    ap.add_argument("--annot", required=True, help="Gold annotation CSV (id, from, to, CUI; optional value)")
    ap.add_argument("--raw", required=True, help="Raw notes CSV (id, text)")
    ap.add_argument("--doc_ids", help="File with one document id per line (default: all notes in --raw)")
    ap.add_argument("--labels", default="prompts/gliner_labels_gpt4o.txt",
                    help="File with one entity label per line")
    ap.add_argument("--model", default="urchade/gliner_multi-v2.1",
                    help="GLiNER checkpoint (HuggingFace ID or local path)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--window", type=int, help="Words per window (default: checkpoint max_len; 0 = whole note)")
    ap.add_argument("--overlap", type=int, default=64, help="Words shared by consecutive windows")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", default="results_gliner.json", help="Output JSON path")
    args = ap.parse_args()

    try:
        from gliner import GLiNER
    except ImportError:
        raise SystemExit("GLiNER is not installed. Run: pip install gliner")

    labels = load_labels(args.labels)
    print(f"Loading GLiNER model: {args.model}")
    model = GLiNER.from_pretrained(args.model)
    if args.window is None:
        args.window = model.config.max_len
    if args.window and not 0 <= args.overlap < args.window:
        raise SystemExit("--overlap must be smaller than --window")

    raw_by = {r["id"]: r["text"] for r in read_rows(args.raw)}
    doc_ids = (sorted(l.strip() for l in Path(args.doc_ids).read_text().splitlines() if l.strip())
               if args.doc_ids else sorted(raw_by))
    gold_by = defaultdict(list)
    for r in read_rows(args.annot):
        gold_by[r["id"]].append(r)

    totals = defaultdict(int)
    doc_results, n_chunks, sizes = {}, [], []
    for i, did in enumerate(doc_ids, 1):
        raw = raw_by[did]
        ents, win_sizes = predict_doc(model, raw, labels, args)
        n_chunks.append(len(win_sizes))
        sizes += win_sizes
        pred = [(*abs_to_lc(raw, e["start"], e["end"]), raw[e["start"]:e["end"]]) for e in ents]
        gold = []
        for r in gold_by[did]:
            s, e = int(float(r["from"])), int(float(r["to"]))
            if lc := abs_to_lc(raw, s, e):
                gold.append((*lc, r.get("value", raw[s:e])))  # no 'value' column: surface from the note
        tp, fp, fn = match_full(gold, pred, {})
        tp_s, fp_s, fn_s = match_strict(gold, pred, {})
        doc_results[did] = {"tp": tp, "fp": fp, "fn": fn, "tp_strict": tp_s, "fp_strict": fp_s,
                            "fn_strict": fn_s, "n_pred": len(pred), "n_gold": len(gold),
                            "n_windows": len(win_sizes)}
        for k, v in doc_results[did].items():
            totals[k] += v
        print(f"  {i}/{len(doc_ids)} ...", end="\r", flush=True)
    print()

    P, R, F1 = prf1(totals["tp"], totals["fp"], totals["fn"])
    _, _, F1_s = prf1(totals["tp_strict"], totals["fp_strict"], totals["fn_strict"])
    print(f"Overlap — P={P:.4f}  R={R:.4f}  F1={F1:.4f}   Strict F1={F1_s:.4f}")
    print(f"TP={totals['tp']}  FP={totals['fp']}  FN={totals['fn']}  (gold {totals['n_gold']})")

    result = {
        "config": {"model": args.model, "max_len": model.config.max_len, "threshold": args.threshold,
                   "window": args.window, "overlap": args.overlap, "labels_file": args.labels,
                   "labels": labels},
        "n_docs": len(doc_ids),
        "aggregate": {
            "overlap": {"P": round(P, 4), "R": round(R, 4), "F1": round(F1, 4),
                        "TP": totals["tp"], "FP": totals["fp"], "FN": totals["fn"]},
            "strict": {"F1": round(F1_s, 4), "TP": totals["tp_strict"],
                       "FP": totals["fp_strict"], "FN": totals["fn_strict"]},
        },
        "windows_per_doc": {"mean": round(statistics.mean(n_chunks), 2), "max": max(n_chunks)},
        "words_per_window": {"mean": round(statistics.mean(sizes), 1), "min": min(sizes), "max": max(sizes)},
        "per_document": doc_results,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
