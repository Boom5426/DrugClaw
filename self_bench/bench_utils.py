"""
Shared utilities for self-bench scripts:
  - LLM client construction
  - Answer extraction helpers
  - Metric computation
  - DrugClaw system integration (maskself / RAG mode)
  - Bench logger (per-sample LLM input/output persistence)
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


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


# ── Dataset → Skill name mapping ──────────────────────────────────────

DATASET_SKILL_MAP: dict[str, str] = {
    "ade_corpus": "ADE Corpus",
    "ddi_corpus": "DDI Corpus 2013",
    "drugprot": "DrugProt",
    "phee": "PHEE",
    "dilirank": "DILIrank",
    "n2c2_2018": "n2c2 2018 Track 2",
    "psytar": "PsyTAR",
}


# ── DrugClaw system helpers ────────────────────────────────────────────

def make_system(key_file: str | None = None):
    """Return a DrugClawSystem instance for RAG-mode benchmarking."""
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from drugclaw.config import Config
    from drugclaw.main_system import DrugClawSystem

    cfg = Config(key_file=key_file)
    return DrugClawSystem(cfg, enable_logging=False)


def system_predict(
    system, query: str, dataset: str, maskself: bool
) -> tuple[str, dict]:
    """
    Query the DrugClaw RAG system.  Returns (answer_text, full_result_dict).

    maskself=True  → exclude the dataset's own skill from retrieval
    maskself=False → allow all resources including self
    """
    resource_filter = None
    if maskself:
        self_skill = DATASET_SKILL_MAP.get(dataset)
        if self_skill:
            resource_filter = [
                s.name
                for s in system.skill_registry.get_registered_skills()
                if s.name != self_skill
            ]

    result = system.query(
        query,
        thinking_mode="simple",
        resource_filter=resource_filter,
        verbose=False,
    )
    return result.get("answer", ""), result


# ── Bench logger ───────────────────────────────────────────────────────

class BenchLogger:
    """
    Per-run logger that persists every sample's LLM input/output,
    analogous to QueryLogger but tailored for benchmark runs.

    Directory layout:
        <log_dir>/<dataset>_<mode>_<timestamp>/
            config.json          # run configuration
            results.json         # final metrics (written at end)
            sample_0000.json     # per-sample record
            sample_0001.json
            ...
    """

    def __init__(self, log_dir: str, dataset: str, maskself: bool | None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = (
            "masked" if maskself is True
            else ("rag" if maskself is False else "direct")
        )
        self.run_dir = Path(log_dir) / f"{dataset}_{mode}_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._count = 0

        (self.run_dir / "config.json").write_text(
            json.dumps(
                {"dataset": dataset, "maskself": maskself, "mode": mode, "timestamp": ts},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def log_sample(
        self,
        idx: int,
        input_data: dict,
        raw_response: str,
        extracted: str | None,
        gold: str,
    ):
        record = {
            "sample_id": idx,
            "input": input_data,
            "output": {
                "raw_response": raw_response,
                "extracted_answer": extracted,
                "gold_label": gold,
                "correct": extracted == gold,
            },
        }
        self._count += 1
        (self.run_dir / f"sample_{idx:04d}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def save_results(self, metrics: dict):
        (self.run_dir / "results.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[BenchLogger] {self._count} samples logged → {self.run_dir}")


# ── Generic classification bench runner ────────────────────────────────

def run_classification_bench(
    *,
    dataset_name: str,
    labels: list[str],
    default_label: str,
    system_prompt: str,
    samples: list[dict],
    format_prompt: Callable[[dict], str],
    key_file: str | None = None,
    maskself: bool | None = None,
    log_dir: str | None = None,
) -> dict:
    """
    Run a classification benchmark with optional DrugClaw RAG and logging.

    maskself=None  → direct LLM classification (no RAG)
    maskself=False → DrugClaw RAG with all resources (retrieval test)
    maskself=True  → DrugClaw RAG with self resource excluded (other-resource test)
    """
    if not samples:
        return {"error": "No data loaded", "total": 0}

    use_system = maskself is not None

    if use_system:
        system = make_system(key_file)
    else:
        llm, _ = make_llm(key_file)

    logger = BenchLogger(log_dir, dataset_name, maskself) if log_dir else None

    golds: list[str] = []
    preds: list[str] = []

    for i, sample in enumerate(samples):
        user_prompt = format_prompt(sample)

        if use_system:
            query = f"{system_prompt}\n\n{user_prompt}"
            resp, sys_result = system_predict(system, query, dataset_name, maskself)
            input_data: dict[str, Any] = {"query": query, "maskself": maskself}
            if sys_result.get("retrieved_text"):
                input_data["retrieved_context"] = sys_result["retrieved_text"][:2000]
        else:
            resp = llm_predict(llm, system_prompt, user_prompt)
            input_data = {"system_prompt": system_prompt, "user_prompt": user_prompt}

        pred = extract_answer(resp, labels) or default_label
        golds.append(sample["gold"])
        preds.append(pred)

        if logger:
            logger.log_sample(i, input_data, resp, pred, sample["gold"])

    metrics = compute_metrics(golds, preds, labels)
    if logger:
        logger.save_results(metrics)
    return metrics
