"""
bel_eval.py — Biomedical Entity Linking evaluation (oracle + realistic).

Note on ChromaDB: the paper describes ChromaDB as the vector store used during
development. This script implements the same cosine-similarity retrieval directly
with sentence-transformers (no DB required) to keep dependencies minimal. Results
are numerically identical.

Encodes entity mentions and candidate UMLS names with a sentence-transformer
model, retrieves the top-k candidates by cosine similarity, and reports
accuracy@1/3/5 and MRR.

Two candidate-set conditions:
  - oracle:    candidates = gold entity texts from the same corpus (train + test)
  - realistic: candidates = UMLS concept names for the target language

The encoding uses lowercase + CLS-pool + L2-normalisation (this is the fix
for the SapBERT case-sensitivity bug documented in the paper: the original
SapBERT model maps Title-case tokens to [UNK], collapsing all clinical terms
to identical embeddings).

Usage
-----
    python evaluation/bel_eval.py \
        --queries      data/bel/SC_test_loc_english_common_entities.csv \
        --oracle_train data/bel/SC_train_loc_english_common_entities.csv \
        --umls         data/umls/umls_english.csv \
        --model        cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR \
        --cache_dir    saved_results/embedding_cache \
        --out          bel_results.json

Expected CSV formats
--------------------
--queries (test mentions with gold CUIs):
    id, entity, CUI_cl
    - id:     document identifier
    - entity: entity surface form
    - CUI_cl: gold UMLS CUI; 'C0000000' = CUI-less (not linkable)

--oracle_train (training mentions for oracle candidate set):
    id, entity, CUI_cl

--umls (UMLS concept names for the target language):
    CUI, entity_text     (or: CUI, LUI, entity_text, code — extra columns ignored)
    Obtain from UMLS MRCONSO.RRF filtered to the target language and DISORDER semantic types.

CUI-less handling
-----------------
Mentions with CUI_cl == 'C0000000' are excluded from the linkable-subset metrics.
They are included in the full metrics (where the model necessarily cannot match them).
"""

from __future__ import annotations
import argparse
import gc
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

_LARGE_THR  = 50_000   # use fp16 memmap cache for index sets above this size
_BATCH_SZ   = 512
_MAX_LEN    = 25       # max subword tokens per entity (SapBERT default)
_Q_CHUNK    = 400      # queries per retrieval block
_C_CHUNK    = 80_000   # candidates per retrieval block


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode(
    texts:      list[str],
    model_name: str,
    cache_path: Path,
) -> np.ndarray | str:
    """
    Encode texts: lowercase → tokenise → CLS pool → L2-normalise.

    Results are cached as .npy files. For large candidate sets (≥50k),
    a float16 memory-mapped file is written and the path is returned as a
    string; the caller passes it directly to retrieve().
    """
    if cache_path.exists():
        log.info("  [cache] %s", cache_path.name)
        is_large = len(texts) >= _LARGE_THR
        if is_large:
            return str(cache_path)
        arr = np.load(str(cache_path))
        return arr.astype(np.float32) if arr.dtype == np.float16 else arr

    import torch
    from transformers import AutoModel, AutoTokenizer

    device    = "cuda" if _has_cuda() else "cpu"
    lc_texts  = [t.lower() for t in texts]   # case-normalisation
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    log.info("  Encoding %d texts (%s, device=%s)", len(lc_texts), model_name, device)

    is_large = len(lc_texts) >= _LARGE_THR
    if is_large:
        # Pre-tokenise all texts at once (fast Rust tokenizer, ~145 MB for 700k×25)
        all_enc = tokenizer(
            lc_texts, padding="max_length", max_length=_MAX_LEN,
            truncation=True, return_tensors=None,
        )
        ids  = np.array(all_enc["input_ids"],      dtype=np.int32)
        mask = np.array(all_enc["attention_mask"], dtype=np.int32)
        tti  = (np.array(all_enc["token_type_ids"], dtype=np.int32)
                if "token_type_ids" in all_enc else None)
        del all_enc; gc.collect()

        # Probe dimension
        sample = _cls_norm(model, device, ids[:1], mask[:1],
                           tti[:1] if tti is not None else None)
        dim = sample.shape[1]

        fp = np.lib.format.open_memmap(
            str(cache_path), mode="w+", dtype=np.float16, shape=(len(lc_texts), dim)
        )
        for start in range(0, len(lc_texts), _BATCH_SZ):
            end = min(start + _BATCH_SZ, len(lc_texts))
            sl  = slice(start, end)
            emb = _cls_norm(model, device, ids[sl], mask[sl],
                            tti[sl] if tti is not None else None)
            fp[sl] = emb.astype(np.float16)
        fp.flush()
        del fp, model, tokenizer, ids, mask, tti; gc.collect()
        return str(cache_path)

    parts = []
    for start in range(0, len(lc_texts), _BATCH_SZ):
        enc = tokenizer(
            lc_texts[start:start + _BATCH_SZ],
            padding="max_length", max_length=_MAX_LEN,
            truncation=True, return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        cls = out.last_hidden_state[:, 0, :]
        cls = cls / cls.norm(dim=1, keepdim=True).clamp(min=1e-9)
        parts.append(cls.cpu().float().numpy())
    embs = np.concatenate(parts, axis=0)
    del model, tokenizer; gc.collect()
    np.save(str(cache_path), embs.astype(np.float16))
    return embs


def _cls_norm(model, device, ids_np, mask_np, tti_np=None):
    import torch
    enc = {
        "input_ids":      torch.from_numpy(ids_np).long().to(device),
        "attention_mask": torch.from_numpy(mask_np).long().to(device),
    }
    if tti_np is not None:
        enc["token_type_ids"] = torch.from_numpy(tti_np).long().to(device)
    with torch.no_grad():
        out = model(**enc)
    cls = out.last_hidden_state[:, 0, :]
    cls = cls / cls.norm(dim=1, keepdim=True).clamp(min=1e-9)
    return cls.cpu().float().numpy()


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    q_embs:     np.ndarray,
    c_embs_or_path,
    cand_cuis:  list[str],
    k:          int = 5,
) -> list[list[str]]:
    """
    Chunked cosine-similarity top-k retrieval.

    Supports both in-memory float32 arrays and mmap paths (str) for large
    candidate sets. Returns a list of deduplicated CUI lists (length ≤ k).
    """
    c_embs = (np.load(c_embs_or_path, mmap_mode="r")
              if isinstance(c_embs_or_path, str) else c_embs_or_path)

    n_q, n_c = q_embs.shape[0], c_embs.shape[0]
    best: list[list] = [[] for _ in range(n_q)]

    for q_start in range(0, n_q, _Q_CHUNK):
        q_end   = min(q_start + _Q_CHUNK, n_q)
        q_block = q_embs[q_start:q_end].astype(np.float32)
        for c_start in range(0, n_c, _C_CHUNK):
            c_end   = min(c_start + _C_CHUNK, n_c)
            c_block = c_embs[c_start:c_end].astype(np.float32)
            sims    = q_block @ c_block.T
            top_n   = min(k, c_block.shape[0])
            idxs    = np.argpartition(-sims, top_n - 1, axis=1)[:, :top_n]
            for bj in range(q_block.shape[0]):
                j = q_start + bj
                li = idxs[bj]
                best[j].extend(zip(sims[bj, li].tolist(), (li + c_start).tolist()))
                if len(best[j]) > k * 4:
                    best[j].sort(key=lambda x: -x[0])
                    best[j] = best[j][:k]
            del c_block, sims
        del q_block
        gc.collect()

    results = []
    for j in range(n_q):
        best[j].sort(key=lambda x: -x[0])
        raw  = [cand_cuis[int(g)] for _, g in best[j][:k]]
        seen, unique = set(), []
        for c in raw:
            if c not in seen:
                seen.add(c); unique.append(c)
        results.append(unique[:k])
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def bel_metrics(
    gold_cuis:  list[str],
    pred_top5:  list[list[str]],
    linkable_only: bool = False,
) -> dict:
    """
    Compute accuracy@1/3/5 and MRR.

    If linkable_only=True, queries where gold CUI == 'C0000000' are skipped.
    """
    acc1 = acc3 = acc5 = mrr_sum = n = 0
    for gold, preds in zip(gold_cuis, pred_top5):
        if linkable_only and gold == "C0000000":
            continue
        n += 1
        if not preds:
            continue
        try:
            rank = preds.index(gold) + 1
        except ValueError:
            rank = None
        if rank == 1:
            acc1 += 1; acc3 += 1; acc5 += 1; mrr_sum += 1.0
        elif rank is not None and rank <= 3:
            acc3 += 1; acc5 += 1; mrr_sum += 1.0 / rank
        elif rank is not None and rank <= 5:
            acc5 += 1; mrr_sum += 1.0 / rank
    if n == 0:
        return {"accuracy@1": 0, "accuracy@3": 0, "accuracy@5": 0, "MRR": 0, "n": 0}
    return {
        "accuracy@1": round(acc1 / n, 4),
        "accuracy@3": round(acc3 / n, 4),
        "accuracy@5": round(acc5 / n, 4),
        "MRR":        round(mrr_sum / n, 4),
        "n":          n,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="BEL evaluation: oracle + realistic, accuracy@k + MRR."
    )
    ap.add_argument("--queries",       required=True,
                    help="Test mentions CSV (columns: id, entity, CUI_cl)")
    ap.add_argument("--oracle_train",  required=True,
                    help="Train mentions CSV for oracle candidate set (same format)")
    ap.add_argument("--umls",          required=True,
                    help="UMLS concept names CSV (columns: CUI, entity_text)")
    ap.add_argument("--model",
                    default="cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
                    help="HuggingFace model name for encoding")
    ap.add_argument("--cache_dir",     default="embedding_cache",
                    help="Directory for embedding cache files")
    ap.add_argument("--out",           default="bel_results.json")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    test_df  = pd.read_csv(args.queries,      dtype=str).fillna("")
    train_df = pd.read_csv(args.oracle_train, dtype=str).fillna("")
    umls_df  = pd.read_csv(args.umls,         dtype=str, header=None,
                           names=["CUI", "LUI", "entity_text", "code"] if
                           pd.read_csv(args.umls, nrows=1, header=None).shape[1] >= 4
                           else ["CUI", "entity_text"])
    umls_df  = umls_df[["CUI", "entity_text"]].dropna().drop_duplicates()
    umls_df  = umls_df.query("entity_text.str.strip() != ''", engine="python")

    query_texts = test_df["entity"].tolist()
    gold_cuis   = test_df["CUI_cl"].tolist()
    n_cui_less  = sum(1 for c in gold_cuis if c == "C0000000")

    # Encode queries
    q_embs = encode(query_texts, args.model,
                    cache_dir / f"bel_q_{Path(args.queries).stem}.npy")

    # Oracle: candidates from train + test gold texts
    oracle_df = (
        pd.concat([train_df, test_df], ignore_index=True)
        [["entity", "CUI_cl"]]
        .rename(columns={"entity": "entity_text", "CUI_cl": "CUI"})
        .drop_duplicates("entity_text")
    )
    c_oracle = encode(oracle_df["entity_text"].tolist(), args.model,
                      cache_dir / "bel_c_oracle.npy")
    pred_oracle  = retrieve(q_embs, c_oracle, oracle_df["CUI"].tolist())
    m_oracle_all = bel_metrics(gold_cuis, pred_oracle, linkable_only=False)
    m_oracle_lnk = bel_metrics(gold_cuis, pred_oracle, linkable_only=True)
    del c_oracle; gc.collect()

    # Realistic: candidates from UMLS
    c_real = encode(umls_df["entity_text"].tolist(), args.model,
                    cache_dir / "bel_c_realistic.npy")
    pred_real    = retrieve(q_embs, c_real, umls_df["CUI"].tolist())
    m_real_all   = bel_metrics(gold_cuis, pred_real, linkable_only=False)
    m_real_lnk   = bel_metrics(gold_cuis, pred_real, linkable_only=True)
    del c_real, q_embs; gc.collect()

    results = {
        "n_queries":   len(query_texts),
        "n_cui_less":  n_cui_less,
        "n_linkable":  len(query_texts) - n_cui_less,
        "model":       args.model,
        "oracle":          m_oracle_all,
        "oracle_linkable": m_oracle_lnk,
        "realistic":          m_real_all,
        "realistic_linkable": m_real_lnk,
    }

    Path(args.out).write_text(json.dumps(results, indent=2))

    print(f"\nBEL results  (n={len(query_texts)}, CUI-less={n_cui_less})")
    for cond, m_full, m_link in [
        ("oracle",    m_oracle_all, m_oracle_lnk),
        ("realistic", m_real_all,   m_real_lnk),
    ]:
        print(f"  {cond:<10}  "
              f"acc@1(all)={m_full['accuracy@1']:.4f}  "
              f"acc@1(link)={m_link['accuracy@1']:.4f}  "
              f"MRR={m_full['MRR']:.4f}")
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
