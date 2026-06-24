#!/usr/bin/env python3
"""
quickumls_baseline.py — QuickUMLS precision/recall sweep (Jaccard 0.50–0.95).

Sweeps the Jaccard similarity threshold and reports P/R/F1 at each point.
Uses overlapping_criteria='length' and the 12 UMLS DISORDER semantic types,
matching the evaluation protocol in the paper.

Note: QuickUMLS uses a LevelDB backend that cannot be shared between processes.
      Each threshold is evaluated in a fresh subprocess to avoid lock conflicts.

Setup
-----
1. Install QuickUMLS (requires Python 3.8–3.10; see https://github.com/Georgetown-IR-Lab/QuickUMLS):
       pip install quickumls
2. Build a QuickUMLS database from UMLS MRCONSO.RRF:
       python -m quickumls.install <path_to_umls_dir> <output_db_dir>
   Use the full UMLS release; the same database was used for all experiments.

Usage
-----
    python baselines/quickumls_baseline.py \
        --db   data/quickumls_db \
        --docs data/en_docs.json \
        --gold data/en_gold.json \
        --out  quickumls_pr_curve.csv

Expected input formats
----------------------
--docs: JSON array of document objects
    [{"doc_id": "d001", "text": "Patient presents with..."}, ...]

--gold: JSON object mapping document IDs to mention lists
    {"d001": [{"start": 18, "end": 27}, ...], ...}
    (start/end are absolute character offsets in the raw document text)

Verification
------------
At threshold 0.70, the paper reports P=0.428, R=0.677, F1=0.524 on the
English test set (131 docs). The --verify flag checks this automatically.
"""

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

DISORDER_TUIS = {
    "T020", "T190", "T049", "T019", "T047", "T050",
    "T037", "T048", "T191", "T046", "T184", "T033",
}

# Paper operating points for verification
VERIFY_POINTS = {
    0.70: {"P": 0.4279, "F1": 0.5242},
    0.80: {"P": 0.5344, "F1": 0.6413},
}
VERIFY_TOL = 0.003


_SUBPROCESS_TEMPLATE = """
import json
from quickumls import QuickUMLS

matcher = QuickUMLS(
    {db!r},
    threshold={thr},
    similarity_name='jaccard',
    overlapping_criteria='length',
    accepted_semtypes={tuis!r},
    window=5,
)

with open({docs!r}) as f:
    docs = [(d['doc_id'], d['text']) for d in json.load(f)]
with open({gold!r}) as f:
    gold_raw = json.load(f)
gold = {{d: [{{'start': m['start'], 'end': m['end']}} for m in ms]
        for d, ms in gold_raw.items()}}

preds = {{}}
for doc_id, text in docs:
    spans = []
    for grp in matcher.match(text, best_match=True, ignore_syntax=False):
        for m in grp:
            spans.append({{'start': m['start'], 'end': m['end']}})
    preds[doc_id] = spans

tp = fp = n_gold = 0
for doc_id, g in gold.items():
    p = preds.get(doc_id, [])
    n_gold += len(g)
    used = [False] * len(g)
    for pp in p:
        hit = False
        for j, gg in enumerate(g):
            if not used[j] and pp['start'] < gg['end'] and gg['start'] < pp['end']:
                used[j] = hit = True; break
        tp += hit; fp += (not hit)
fn = n_gold - tp
P  = tp / (tp + fp) if (tp + fp) else 0.
R  = tp / (tp + fn) if (tp + fn) else 0.
F1 = 2*P*R/(P+R) if (P+R) else 0.
print(json.dumps({{'tp':tp,'fp':fp,'fn':fn,'n_gold':n_gold,
                   'P':round(P,4),'R':round(R,4),'F1':round(F1,4)}}))
"""


def run_threshold(db: str, docs: str, gold: str, thr: float) -> dict:
    import subprocess
    code = _SUBPROCESS_TEMPLATE.format(
        db=db,
        tuis=sorted(DISORDER_TUIS),
        thr=thr,
        docs=docs,
        gold=gold,
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed at thr={thr}:\n{result.stderr[-1000:]}"
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser(
        description="QuickUMLS P/R sweep across Jaccard thresholds."
    )
    ap.add_argument("--db",   required=True, help="QuickUMLS database directory")
    ap.add_argument("--docs", required=True,
                    help="Documents JSON ([{doc_id, text}, ...])")
    ap.add_argument("--gold", required=True,
                    help="Gold mentions JSON ({doc_id: [{start, end}]})")
    ap.add_argument("--out",  default="quickumls_pr_curve.csv",
                    help="Output CSV path")
    ap.add_argument("--thresholds", default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
                    help="Comma-separated threshold values to evaluate")
    ap.add_argument("--verify", action="store_true",
                    help="Check results against paper operating points at thr=0.70 and 0.80")
    args = ap.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]

    with open(args.docs) as f:
        n_docs = len(json.load(f))
    with open(args.gold) as f:
        n_gold = sum(len(v) for v in json.load(f).values())
    print(f"Docs: {n_docs}   Gold mentions: {n_gold}\n")

    rows = []
    for thr in thresholds:
        print(f"  thr={thr:.2f} ...", end=" ", flush=True)
        r = run_threshold(args.db, args.docs, args.gold, thr)
        rows.append({"threshold": thr, **r})
        print(f"P={r['P']:.4f} R={r['R']:.4f} F1={r['F1']:.4f} "
              f"TP={r['tp']} FP={r['fp']} FN={r['fn']}")

    # Verification
    if args.verify:
        print()
        all_ok = True
        for vp_thr, expected in VERIFY_POINTS.items():
            row = next((r for r in rows if abs(r["threshold"] - vp_thr) < 1e-9), None)
            if row is None:
                print(f"  thr={vp_thr}: not in threshold grid — skipped")
                continue
            ok_p  = abs(row["P"]  - expected["P"])  <= VERIFY_TOL
            ok_f1 = abs(row["F1"] - expected["F1"]) <= VERIFY_TOL
            status = "PASS" if (ok_p and ok_f1) else "FAIL"
            print(f"  Verify thr={vp_thr}: "
                  f"P={row['P']:.4f} (want {expected['P']}±{VERIFY_TOL}), "
                  f"F1={row['F1']:.4f} (want {expected['F1']}±{VERIFY_TOL}) -> {status}")
            if not (ok_p and ok_f1):
                all_ok = False
        if not all_ok:
            print("  WARNING: one or more verification checks failed. "
                  "Check QuickUMLS version and database.")

    # Best F1
    best = max(rows, key=lambda r: r["F1"])
    print(f"\nBest F1={best['F1']:.4f} at thr={best['threshold']:.2f} "
          f"(P={best['P']:.4f}, R={best['R']:.4f})")

    # Write CSV
    fields = ["threshold", "P", "R", "F1", "tp", "fp", "fn", "n_gold"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(f"\nResults written to {args.out}")

    # Optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        R_vals = [r["R"] for r in rows]
        P_vals = [r["P"] for r in rows]
        plt.figure(figsize=(5, 4))
        plt.plot(R_vals, P_vals, "-o", ms=4, label="QuickUMLS (length overlap)")
        for r in rows:
            if abs(r["threshold"] - 0.70) < 1e-9:
                plt.scatter([r["R"]], [r["P"]], s=100, c="red", zorder=5,
                            label=f"thr=0.70 (paper)")
                plt.annotate(
                    f"thr=0.70\nP={r['P']:.3f}\nF1={r['F1']:.3f}",
                    (r["R"], r["P"]),
                    textcoords="offset points", xytext=(8, -18), fontsize=8,
                )
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("QuickUMLS P/R curve (English)")
        plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
        png_out = Path(args.out).with_suffix(".png")
        plt.savefig(png_out, dpi=150)
        print(f"Plot written to {png_out}")
    except Exception as exc:
        print(f"[plot skipped] {exc}")


if __name__ == "__main__":
    main()
