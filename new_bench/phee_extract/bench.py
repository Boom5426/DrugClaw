"""
Pharma Event Extraction — PHEE (pharmacovigilance).

Task  : Given a clinical text sentence, extract all (drug, adverse_event) pairs.
Data  : new_bench_data/phee/test.json  — JSONL, original PHEE annotation format.
        Each line: {id, context, is_mult_event, annotations:[{events:[...]}]}
        Gold tuples come from Adverse_event entries:
          drug  ← Treatment.Drug.text[0][0]
          event ← Effect.text[0][0]

QA    : Q = task background + "Text: {sentence}\nExtract all (drug, adverse_event) pairs."
        A = set of (drug, adverse_event) tuples extracted from annotations

This dataset does NOT have QA pairs in the raw data. QA pairs are constructed
here from the free-text sentence (Q) and the gold annotation tuples (A).

Metric: Micro-F1, Precision, Recall on (drug, event) tuple sets
"""

import json
import re
from pathlib import Path

DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "new_bench_data/phee/test.json"
)

SYSTEM_PROMPT = "You are an expert pharmaceutical AI assistant."

TASK_BACKGROUND = """\
Task: Pharmacovigilance adverse event extraction (PHEE)

PHEE is a pharmacovigilance event extraction dataset built from case reports in \
medical literature. Each sample contains a clinical text sentence that may \
describe one or more adverse drug events. Your task is to extract all \
(drug, adverse_event) pairs present in the text.

For each adverse event, identify:
- Drug:  the drug or treatment that caused it
- Event: the adverse effect or clinical outcome observed

Output one pair per line in this exact format:
  Drug: <drug name>, Event: <adverse event description>

If no adverse events are described, output: NONE\
"""


def _load_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _extract_gold_tuples(record: dict) -> set[tuple[str, str]]:
    """Extract gold (drug, adverse_event) tuples from a PHEE annotation record."""
    tuples: set[tuple[str, str]] = set()
    for ann in record.get("annotations", []):
        for ev in ann.get("events", []):
            if ev.get("event_type", "").strip().lower() != "adverse_event":
                continue

            drug_texts = (
                ev.get("Treatment", {})
                  .get("Drug", {})
                  .get("text", [[]])
            )
            drug = drug_texts[0][0] if drug_texts and drug_texts[0] else ""

            effect_texts = ev.get("Effect", {}).get("text", [[]])
            effect = effect_texts[0][0] if effect_texts and effect_texts[0] else ""

            if drug and effect:
                tuples.add((drug.lower().strip(), effect.lower().strip()))
    return tuples


def load_data(max_samples: int = 0) -> list[dict]:
    """
    Build QA pairs from PHEE annotations.

    Each record becomes:
      query        — the sentence text (the "question" input)
      gold_tuples  — set of (drug, adverse_event) string pairs (the "answer")
    """
    raw = _load_jsonl(DATA_PATH)
    rows = []
    for r in raw:
        gold_tuples = _extract_gold_tuples(r)
        if not gold_tuples:
            continue
        rows.append({
            "sentence":    r.get("context", ""),
            "gold_tuples": gold_tuples,
        })
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def _format_prompt(s: dict) -> str:
    return (
        f"{TASK_BACKGROUND}\n\n"
        f"Text: {s['sentence']}\n\n"
        f"Extract all adverse drug events (drug → adverse effect) from the text above."
    )


def run(
    key_file: str | None = None,
    max_samples: int = 0,
    mode: str = "direct",
    maskself: bool = False,
    log_dir: str | None = None,
) -> dict:
    from new_bench.bench_utils import run_extraction_bench

    return run_extraction_bench(
        dataset_name="phee_extract",
        system_prompt=SYSTEM_PROMPT,
        samples=load_data(max_samples),
        format_prompt=_format_prompt,
        mode=mode,
        maskself=maskself,
        key_file=key_file,
        log_dir=log_dir,
    )
