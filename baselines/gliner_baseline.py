#!/usr/bin/env python3
"""
gliner_baseline.py — GLiNER zero-shot NER baseline.

Runs GLiNER (Zaratiana et al., 2023) on the English test corpus in the same
zero-shot, no-fine-tuning setting used for GPT-4o. The model is prompted with
the 12 UMLS DISORDER semantic-type labels as entity types.

Paper results (English, 131 test docs, overlap matching against full gold):
  Precision 0.44 / Recall 0.11 / F1 0.19

Paper results (Dutch, overlap):
  F1 0.13

Paper results (Italian, overlap):
  F1 0.11

The gap versus GPT-4o reflects GLiNER's smaller context window and the lack of
instruction-following capability for clinical annotation guidelines.

Setup
-----
    pip install gliner
    # or: pip install "gliner>=0.1"

Usage
-----
    python baselines/gliner_baseline.py \
        --docs data/en_docs.json \
        --gold data/en_gold.json \
        --out  gliner_results.json

    # Adjust the model checkpoint:
    python baselines/gliner_baseline.py \
        --docs data/en_docs.json \
        --gold data/en_gold.json \
        --model urchade/gliner_medium-v2.1 \
        --out   gliner_results.json

Expected input formats
----------------------
--docs: JSON array of document objects
    [{"doc_id": "d001", "text": "Patient presents with..."}, ...]

--gold: JSON object mapping document IDs to mention lists
    {"d001": [{"start": 18, "end": 27}, ...], ...}
    (start/end are absolute character offsets in the raw document text)

Output format
-------------
JSON file with per-document predictions and aggregate P/R/F1.
"""

import argparse
import json
from pathlib import Path

# The 12 UMLS disorder semantic types used as GLiNER entity-type labels.
# These mirror the categories in the GPT-4o annotation prompt (Section S1).
DISORDER_LABELS = [
    "disease or syndrome",
    "signs and symptoms",
    "pathologic function",
    "injury or poisoning",
    "neoplastic process",
    "congenital abnormality",
    "acquired abnormality",
    "anatomical abnormality",
    "mental or behavioral dysfunction",
    "cell or molecular dysfunction",
    "experimental model of disease",
    "finding",
]


def overlap_match(pred_spans: list[dict], gold_spans: list[dict]) -> tuple[int, int, int]:
    """Return (TP, FP, FN) using the same overlap criterion as the paper."""
    used = [False] * len(gold_spans)
    tp = fp = 0
    for p in pred_spans:
        hit = False
        for j, g in enumerate(gold_spans):
            if not used[j] and p["start"] < g["end"] and g["start"] < p["end"]:
                used[j] = hit = True
                break
        if hit:
            tp += 1
        else:
            fp += 1
    fn = sum(1 for u in used if not u)
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser(
        description="GLiNER zero-shot NER baseline (overlap + strict F1)."
    )
    ap.add_argument("--docs", required=True,
                    help="Documents JSON ([{doc_id, text}, ...])")
    ap.add_argument("--gold", required=True,
                    help="Gold mentions JSON ({doc_id: [{start, end}]})")
    ap.add_argument("--model", default="urchade/gliner_medium-v2.1",
                    help="GLiNER model checkpoint (HuggingFace ID or local path)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="GLiNER prediction confidence threshold (default: 0.5)")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Number of documents per inference batch")
    ap.add_argument("--out", default="gliner_results.json",
                    help="Output JSON path")
    args = ap.parse_args()

    try:
        from gliner import GLiNER
    except ImportError:
        raise SystemExit(
            "GLiNER is not installed. Run: pip install gliner"
        )

    print(f"Loading GLiNER model: {args.model}")
    model = GLiNER.from_pretrained(args.model)

    with open(args.docs) as f:
        docs = json.load(f)
    with open(args.gold) as f:
        gold_raw = json.load(f)

    gold = {
        d: [{"start": m["start"], "end": m["end"]} for m in ms]
        for d, ms in gold_raw.items()
    }

    print(f"Running GLiNER on {len(docs)} documents "
          f"(threshold={args.threshold}, batch_size={args.batch_size})")

    preds = {}
    for i in range(0, len(docs), args.batch_size):
        batch = docs[i: i + args.batch_size]
        texts = [d["text"] for d in batch]
        batch_entities = model.batch_predict_entities(
            texts,
            DISORDER_LABELS,
            threshold=args.threshold,
        )
        for doc, entities in zip(batch, batch_entities):
            preds[doc["doc_id"]] = [
                {"start": e["start"], "end": e["end"], "label": e["label"], "score": e["score"]}
                for e in entities
            ]
        done = min(i + args.batch_size, len(docs))
        print(f"  {done}/{len(docs)} ...", end="\r", flush=True)

    print()

    # Aggregate evaluation (overlap matching)
    total_tp = total_fp = total_fn = 0
    doc_results = {}
    for doc in docs:
        did = doc["doc_id"]
        g = gold.get(did, [])
        p = [{"start": e["start"], "end": e["end"]} for e in preds.get(did, [])]
        tp, fp, fn = overlap_match(p, g)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        doc_results[did] = {"tp": tp, "fp": fp, "fn": fn, "n_pred": len(p), "n_gold": len(g)}

    P = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    R = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0

    print(f"\nOverlap matching — P={P:.4f}  R={R:.4f}  F1={F1:.4f}")
    print(f"TP={total_tp}  FP={total_fp}  FN={total_fn}")

    result = {
        "model": args.model,
        "threshold": args.threshold,
        "labels": DISORDER_LABELS,
        "n_docs": len(docs),
        "aggregate": {
            "overlap": {"P": round(P, 4), "R": round(R, 4), "F1": round(F1, 4),
                        "TP": total_tp, "FP": total_fp, "FN": total_fn},
        },
        "paper_results": {
            "english": {"F1": 0.19, "P": 0.44, "R": 0.11},
            "dutch":   {"F1": 0.13},
            "italian": {"F1": 0.11},
            "note": "urchade/gliner_medium-v2.1, threshold=0.5, overlap matching",
        },
        "per_document": doc_results,
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
