"""
Shared utilities for self-bench scripts:
  - LLM client construction
  - Answer extraction helpers
  - Metric computation
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# ── LLM helpers ─────────────────────────────────────────────────────────

def make_llm(key_file: str | None = None):
    """Return a (LLMClient, Config) tuple using the existing DrugClaw stack."""
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from drugclaw.config import Config
    from drugclaw.llm_client import LLMClient

    cfg = Config(key_file=key_file)
    return LLMClient(cfg), cfg


def llm_predict(llm, system_prompt: str, user_prompt: str) -> str:
    """Single-turn LLM call. Returns raw text response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return llm.generate(messages, temperature=0.0)


# ── Answer extraction ───────────────────────────────────────────────────

def extract_answer(text: str, valid_labels: list[str]) -> str | None:
    """
    Extract a classification label from LLM free-text output.

    Strategy:
      1. Try to parse JSON with an "answer" key.
      2. Look for **Answer: <label>** pattern.
      3. Fall back to first occurrence of any valid label.

    Returns None if nothing matched.
    """
    text_clean = text.strip()

    # 1) JSON extraction
    try:
        obj = json.loads(text_clean)
        if isinstance(obj, dict):
            ans = obj.get("answer", obj.get("label", obj.get("prediction")))
            if ans is not None:
                return _match_label(str(ans), valid_labels)
    except json.JSONDecodeError:
        pass

    # 2) Pattern: Answer: <label>
    m = re.search(r"(?:answer|prediction|label)\s*[:=]\s*[\"']?([^\"'\n,]+)", text_clean, re.I)
    if m:
        candidate = m.group(1).strip().rstrip(".")
        matched = _match_label(candidate, valid_labels)
        if matched:
            return matched

    # 3) First valid label found in text
    lower = text_clean.lower()
    for label in valid_labels:
        if label.lower() in lower:
            return label

    return None


def _match_label(candidate: str, valid_labels: list[str]) -> str | None:
    c = candidate.strip().lower()
    for label in valid_labels:
        if c == label.lower():
            return label
    # partial match
    for label in valid_labels:
        if label.lower() in c or c in label.lower():
            return label
    return None


# ── Metrics ─────────────────────────────────────────────────────────────

def compute_metrics(
    golds: list[str], preds: list[str], labels: list[str] | None = None
) -> dict[str, Any]:
    """
    Compute accuracy + per-class precision/recall/F1 + macro-F1.
    Returns a dict ready for JSON serialisation.
    """
    assert len(golds) == len(preds)
    n = len(golds)
    if n == 0:
        return {"total": 0, "accuracy": 0.0}

    correct = sum(g == p for g, p in zip(golds, preds))
    accuracy = round(correct / n, 4)

    all_labels = labels or sorted(set(golds) | set(preds))
    per_class: dict[str, dict] = {}
    for lab in all_labels:
        tp = sum(g == lab and p == lab for g, p in zip(golds, preds))
        fp = sum(g != lab and p == lab for g, p in zip(golds, preds))
        fn = sum(g == lab and p != lab for g, p in zip(golds, preds))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[lab] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": sum(g == lab for g in golds),
        }

    f1_values = [v["f1"] for v in per_class.values()]
    macro_f1 = round(sum(f1_values) / len(f1_values), 4) if f1_values else 0.0

    return {
        "total": n,
        "correct": correct,
        "accuracy": accuracy,
        "f1_macro": macro_f1,
        "per_class": per_class,
    }
