#!/usr/bin/env python3
r"""
medcat_baseline.py — PART A2: MedCAT baseline on the 133 English docs,
scored with the SAME overlap + strict P/R/F1 scheme as PART A1.

WHY medcat 1.16.x (not 2.x): your model pack `umls_sm_wstatus_2021_aa.zip` is a
2021 / MedCAT-v1 pack. The PyPI `medcat` 2.x line uses a new model-pack format and
will NOT load a v1 pack directly. So pin a 1.16.x.

INSTALL (env picor, inside WSL)
-------------------------------
    conda activate picor
    # core (pulls torch, spacy, transformers, gensim, blis, peft, datasets ...):
    pip install "medcat==1.16.12"
    # MedCAT needs a base spaCy English pipeline available:
    python -m spacy download en_core_web_md
    # sanity check the API path:
    python -c "from medcat.cat import CAT; print('medcat import OK')"

Put the pack here (you already have the path in your error):
    data/medcat_model/umls_sm_wstatus_2021_aa.zip

RUN (heavy-ish: CDB load + 133 docs; ~minutes, a couple GB RAM)
---------------------------------------------------------------
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 19 \
      python medcat_baseline.py \
        --model_pack data/medcat_model/umls_sm_wstatus_2021_aa.zip \
        --docs  path/to/en_docs.json \
        --gold  path/to/en_gold.json \
        --out   medcat_a2_results.json

Expected input formats (ADAPT the two loaders below to your repo if different):
    en_docs.json : [{"doc_id": "d1", "text": "..."}, ...]
    en_gold.json : {"d1": [{"start": 10, "end": 18, "cui": "C0011849"}, ...], ...}
"""
import argparse, json, os, sys, time

# Keep BLAS/torch from oversubscribing your 6 cores (mirrors revision_analysis.py).
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")


# ---------- data loaders (ADAPT if your schema differs) ----------
def load_docs(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [(d["doc_id"], d["text"]) for d in data]


def load_gold(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # {doc_id: [{start,end,cui}, ...]}


# ---------- MedCAT inference ----------
def run_medcat(model_pack, docs, target_tuis=None, min_acc=0.0):
    from medcat.cat import CAT
    print(f"[medcat] loading {model_pack} ...", flush=True)
    t0 = time.time()
    cat = CAT.load_model_pack(model_pack)
    print(f"[medcat] loaded in {time.time()-t0:.1f}s", flush=True)

    preds = {}
    for i, (doc_id, text) in enumerate(docs, 1):
        res = cat.get_entities(text)
        ents = []
        for e in res.get("entities", {}).values():
            if e.get("acc", 1.0) < min_acc:
                continue
            # type_ids = TUI codes (e.g. ["T047"]), types = human labels
            if target_tuis and not (set(e.get("type_ids", [])) & target_tuis):
                continue
            ents.append({"start": int(e["start"]), "end": int(e["end"]),
                         "cui": e["cui"]})
        preds[doc_id] = ents
        if i % 25 == 0:
            print(f"[medcat] {i}/{len(docs)} docs", flush=True)
    return preds


# ---------- scoring (overlap + strict, micro-averaged) ----------
def _overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def score(preds, gold, mode="overlap", require_cui=False):
    """
    Greedy micro P/R/F1. mode='overlap' (any char overlap) or 'strict' (exact span).
    require_cui=True also demands the predicted CUI equals the gold CUI (entity
    linking); False scores span detection only (NER-style, like A1).
    """
    tp = fp = n_gold = 0
    for doc_id, p_ents in preds.items():
        g_ents = list(gold.get(doc_id, []))
        n_gold += len(g_ents)
        used = [False] * len(g_ents)
        for p in p_ents:
            hit = False
            for j, g in enumerate(g_ents):
                if used[j]:
                    continue
                span_ok = (p["start"] == g["start"] and p["end"] == g["end"]) \
                    if mode == "strict" else _overlap(p, g)
                if span_ok and (not require_cui or p["cui"] == g["cui"]):
                    used[j] = True
                    hit = True
                    break
            tp += hit
            fp += (not hit)
    fn = n_gold - tp
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return {"P": round(P, 4), "R": round(R, 4), "F1": round(F1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_pack", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", default="medcat_a2_results.json")
    ap.add_argument("--min_acc", type=float, default=0.0,
                    help="drop MedCAT entities below this confidence (0=keep all)")
    ap.add_argument("--require_cui", action="store_true",
                    help="score entity LINKING (CUI must match), not just spans")
    ap.add_argument("--tuis", default="",
                    help="comma-separated TUIs to keep (e.g. T047,T184,...); "
                         "empty = keep all, like QuickUMLS in A1")
    a = ap.parse_args()

    docs = load_docs(a.docs)
    gold = load_gold(a.gold)
    target_tuis = set(t.strip() for t in a.tuis.split(",") if t.strip()) or None

    preds = run_medcat(a.model_pack, docs, target_tuis=target_tuis,
                       min_acc=a.min_acc)

    results = {
        "n_docs": len(docs),
        "n_pred": sum(len(v) for v in preds.values()),
        "n_gold": sum(len(v) for v in gold.values()),
        "overlap": score(preds, gold, "overlap", a.require_cui),
        "strict": score(preds, gold, "strict", a.require_cui),
        "config": {"min_acc": a.min_acc, "require_cui": a.require_cui,
                   "tuis": sorted(target_tuis) if target_tuis else "all"},
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== MedCAT PART A2 (English, {} docs) ===".format(len(docs)))
    for k in ("overlap", "strict"):
        r = results[k]
        print(f"{k:8} P={r['P']:.3f} R={r['R']:.3f} F1={r['F1']:.3f} "
              f"(tp={r['tp']} fp={r['fp']} fn={r['fn']})")
    print(f"\n[ok] wrote {a.out}")
    print("Drop this row next to QuickUMLS / GPT-4o in the A1 table.")


if __name__ == "__main__":
    sys.exit(main())
