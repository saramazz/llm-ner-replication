"""
m2_matcher.py — M2 content-aligned span matcher.

The M2 matcher handles the coordinate drift that arises when a language model
returns annotated text with minor reformatting. Instead of matching spans by
absolute character offset, it:

  1. Parses the GPT output into (line, col_start, col_end, text) tuples.
  2. Aligns gold and predicted line sequences with difflib (handles inserted/
     deleted blank lines and minor reformatting).
  3. Matches spans on aligned lines using character-level overlap.
  4. Applies a shares_token guard: a match is only accepted if the predicted
     and gold entity texts share at least one alphanumeric token (prevents
     cross-line false positives when two short spans happen to overlap after
     line re-alignment).

Three matching modes:
  - full       : all gold spans (CUI-bearing + CUI-less)
  - naive_cb   : CUI-bearing gold only; predictions overlapping CUI-less → FP
  - clean_cb   : CUI-bearing gold only; predictions overlapping CUI-less →
                 unlinkable (consumed but not counted as FP)

Reference
---------
Mazzucato et al. (2026). Zero-shot clinical NER with GPT-4o: cross-lingual
evaluation on SemEval-2014 Task 7 / ShARe-CLEF 2014. JAMIA Open.
"""

from __future__ import annotations
import difflib
import random
import re

RE_BRACKET = re.compile(r'\[\[(.+?)\]\]')
RE_TOKEN   = re.compile(r"[a-zA-ZÀ-ÖØ-öø-ÿ0-9]+")


# ---------------------------------------------------------------------------
# Text parsing helpers
# ---------------------------------------------------------------------------

def extract_abs(annotated_text: str, regex: re.Pattern = RE_BRACKET):
    """
    Strip annotation markers and return (clean_text, abs_spans).

    abs_spans: list of (abs_start, abs_end, entity_text)
    Absolute character offsets are relative to the clean text.
    """
    parts, spans, pos, orig = [], [], 0, 0
    for m in regex.finditer(annotated_text):
        entity = m.group(1)
        pre    = annotated_text[orig:m.start()]
        parts.append(pre)
        pos  += len(pre)
        orig  = m.end()
        spans.append((pos, pos + len(entity), entity))
        parts.append(entity)
        pos  += len(entity)
    parts.append(annotated_text[orig:])
    return "".join(parts), spans


def extract_lc(annotated_text: str, regex: re.Pattern = RE_BRACKET):
    """
    Strip annotation markers and return (clean_text, lc_spans).

    lc_spans: list of (line_index, col_start, col_end, entity_text)
    Line-column coordinates are relative to the clean text.
    """
    clean_lines, spans = [], []
    for li, line in enumerate(annotated_text.split("\n")):
        col = orig = 0
        parts = []
        for m in regex.finditer(line):
            entity = m.group(1)
            parts.append(line[orig:m.start()])
            col  += m.start() - orig
            orig  = m.end()
            spans.append((li, col, col + len(entity), entity))
            parts.append(entity)
            col  += len(entity)
        parts.append(line[orig:])
        clean_lines.append("".join(parts))
    return "\n".join(clean_lines), spans


def abs_to_lc(raw_text: str, abs_s: int, abs_e: int):
    """
    Convert absolute character offsets to (line, col_start, col_end).
    Returns None if the offset falls outside the text.
    """
    pos = 0
    for li, line in enumerate(raw_text.split("\n")):
        end = pos + len(line)
        if pos <= abs_s <= end:
            return li, abs_s - pos, abs_e - pos
        pos = end + 1
    return None


def shares_token(a: str, b: str) -> bool:
    """True if the two entity strings share at least one alphanumeric token."""
    return bool(
        set(RE_TOKEN.findall(a.lower())) & set(RE_TOKEN.findall(b.lower()))
    )


# ---------------------------------------------------------------------------
# Line alignment
# ---------------------------------------------------------------------------

def align_lines(gold_lines: list[str], pred_lines: list[str]) -> dict[int, int]:
    """
    Build a gold-line → pred-line mapping using difflib SequenceMatcher.

    Both 'equal' and 'replace' blocks are mapped (replace handles minor
    whitespace / punctuation drift). Insertions / deletions are skipped.
    """
    sm = difflib.SequenceMatcher(None, gold_lines, pred_lines, autojunk=False)
    mapping: dict[int, int] = {}
    for tag, gi1, gi2, pi1, pi2 in sm.get_opcodes():
        if tag in ("equal", "replace"):
            for g, p in zip(range(gi1, gi2), range(pi1, pi2)):
                mapping[g] = p
    return mapping


# ---------------------------------------------------------------------------
# Span matching (per-document)
# ---------------------------------------------------------------------------

def match_full(
    gold_lc: list,
    pred_lc: list,
    line_map: dict[int, int],
) -> tuple[int, int, int]:
    """
    Greedy overlap match: all gold spans against all predicted spans.

    Returns (tp, fp, fn).
    """
    used = [False] * len(pred_lc)
    tp = fn = 0
    for gl, gs, ge, gt in gold_lc:
        mapped = line_map.get(gl, gl)
        hit = False
        for j, (pl, ps, pe, pt) in enumerate(pred_lc):
            if (
                not used[j]
                and pl == mapped
                and max(gs, ps) < min(ge, pe)
                and shares_token(gt, pt)
            ):
                used[j] = hit = True
                tp += 1
                break
        if not hit:
            fn += 1
    fp = sum(1 for u in used if not u)
    return tp, fp, fn


def match_strict(
    gold_lc: list,
    pred_lc: list,
    line_map: dict[int, int],
) -> tuple[int, int, int]:
    """
    Strict match: spans must be exactly equal (same line + same col offsets).

    Returns (tp, fp, fn).
    """
    used = [False] * len(pred_lc)
    tp = fn = 0
    for gl, gs, ge, gt in gold_lc:
        mapped = line_map.get(gl, gl)
        hit = False
        for j, (pl, ps, pe, pt) in enumerate(pred_lc):
            if (
                not used[j]
                and pl == mapped
                and gs == ps
                and ge == pe
                and shares_token(gt, pt)
            ):
                used[j] = hit = True
                tp += 1
                break
        if not hit:
            fn += 1
    fp = sum(1 for u in used if not u)
    return tp, fp, fn


def match_clean_cb(
    gold_cb_lc: list,
    gold_cl_lc: list,
    pred_lc:    list,
    line_map:   dict[int, int],
) -> tuple[int, int, int, int]:
    """
    Clean CUI-bearing match.

    Pass 1: match predictions against CUI-bearing gold (overlap + shares_token).
    Pass 2: remaining predictions that overlap a CUI-less gold span are classified
            as 'unlinkable' (correct span detection but no UMLS mapping possible)
            and consumed without penalty.
    FP = predictions not consumed in either pass.

    Returns (tp, fp, fn, n_unlinkable).
    """
    used = [False] * len(pred_lc)

    # Pass 1: CUI-bearing gold
    tp = fn = 0
    for gl, gs, ge, gt in gold_cb_lc:
        mapped = line_map.get(gl, gl)
        hit = False
        for j, (pl, ps, pe, pt) in enumerate(pred_lc):
            if (
                not used[j]
                and pl == mapped
                and max(gs, ps) < min(ge, pe)
                and shares_token(gt, pt)
            ):
                used[j] = hit = True
                tp += 1
                break
        if not hit:
            fn += 1

    # Pass 2: CUI-less spans → unlinkable (no shares_token check needed here)
    unlinkable = 0
    for gl, gs, ge, _ in gold_cl_lc:
        mapped = line_map.get(gl, gl)
        for j, (pl, ps, pe, _) in enumerate(pred_lc):
            if not used[j] and pl == mapped and max(gs, ps) < min(ge, pe):
                used[j] = True
                unlinkable += 1
                break

    fp = sum(1 for u in used if not u)
    return tp, fp, fn, unlinkable


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return (precision, recall, F1), all rounded to 4 decimal places."""
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return round(P, 4), round(R, 4), round(F, 4)


def bootstrap_f1(
    per_doc_stats: dict[str, dict],
    n: int   = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """
    95% bootstrap CI on overlap F1 by per-document resampling.

    per_doc_stats: {doc_id: {"tp": int, "fp": int, "fn": int}}
    Returns (lower_95, upper_95).
    """
    rng  = random.Random(seed)
    docs = list(per_doc_stats)
    f1s  = []
    for _ in range(n):
        sample = rng.choices(docs, k=len(docs))
        tp = sum(per_doc_stats[d]["tp"] for d in sample)
        fp = sum(per_doc_stats[d]["fp"] for d in sample)
        fn = sum(per_doc_stats[d]["fn"] for d in sample)
        _, _, f = prf1(tp, fp, fn)
        f1s.append(f)
    f1s.sort()
    return round(f1s[int(0.025 * n)], 4), round(f1s[int(0.975 * n)], 4)
