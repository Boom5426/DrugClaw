"""
PHEE — Pharmacovigilance event type classification benchmark.

Task: Given a sentence about a drug and its effect, classify whether the
      event is an adverse_event or a therapeutic_outcome.
Labels: adverse_event, therapeutic_outcome
Format: JSON array [{drug, effect, event_type, sentence}, ...]
"""

import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "resources_metadata/drug_nlp/PHEE/phee.json"

LABELS = ["adverse_event", "therapeutic_outcome"]

SYSTEM_PROMPT = """\
You are a pharmacovigilance NLP classifier.

Task: Given a sentence describing a drug's effect, classify the event type.

Event types:
- adverse_event: An undesirable or harmful effect caused by the drug.
- therapeutic_outcome: A beneficial or intended therapeutic effect of the drug.

You must respond with ONLY a JSON object: {"answer": "<label>"}
where <label> is one of: adverse_event, therapeutic_outcome
"""

USER_TEMPLATE = """\
Drug: {drug}
Effect: {effect}
Sentence: {sentence}

Is this an adverse event or a therapeutic outcome?
Respond with JSON: {{"answer": "adverse_event"}} or {{"answer": "therapeutic_outcome"}}
"""


def _load_json_or_jsonl(path: Path) -> list[dict]:
    """Load a JSON array or a JSONL (one JSON object per line) file."""
    text = path.read_text(encoding="utf-8").strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        # Fall back to JSONL (one JSON object per line)
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _parse_record(r: dict) -> dict | None:
    """
    Parse a single PHEE record.  Supports two formats:

    1. Simplified: {"drug": ..., "effect": ..., "event_type": ..., "sentence": ...}
    2. Original annotation JSONL: {"id": ..., "context": ..., "annotations": [...]}
       where drug/effect/event_type are nested inside annotations[].events[].
    """
    # Format 1: simplified (has "drug" key directly)
    if "drug" in r:
        return {
            "drug": r["drug"],
            "effect": r["effect"],
            "sentence": r["sentence"],
            "gold": r["event_type"],
        }

    # Format 2: original PHEE annotation
    if "annotations" not in r:
        return None
    for ann in r.get("annotations", []):
        for ev in ann.get("events", []):
            # Extract event_type and normalise to lowercase with underscore
            raw_type = ev.get("event_type", "")
            event_type = raw_type.strip().lower().replace(" ", "_")

            # Extract drug name from Treatment.Drug.text
            treatment = ev.get("Treatment", {})
            drug_info = treatment.get("Drug", {})
            drug_texts = drug_info.get("text", [[]])
            drug = drug_texts[0][0] if drug_texts and drug_texts[0] else "unknown"

            # Extract effect from Effect.text
            effect_info = ev.get("Effect", {})
            effect_texts = effect_info.get("text", [[]])
            effect = effect_texts[0][0] if effect_texts and effect_texts[0] else "unknown"

            sentence = r.get("context", "")
            return {
                "drug": drug,
                "effect": effect,
                "sentence": sentence,
                "gold": event_type,
            }
    return None


def load_data(max_samples: int = 0) -> list[dict]:
    raw = _load_json_or_jsonl(DATA_PATH)
    rows = []
    for r in raw:
        parsed = _parse_record(r)
        if parsed:
            rows.append(parsed)
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def run(
    key_file: str | None = None,
    max_samples: int = 0,
    maskself: bool | None = None,
    log_dir: str | None = None,
) -> dict:
    from self_bench.bench_utils import run_classification_bench

    return run_classification_bench(
        dataset_name="phee",
        labels=LABELS,
        default_label="adverse_event",
        system_prompt=SYSTEM_PROMPT,
        samples=load_data(max_samples),
        format_prompt=lambda s: USER_TEMPLATE.format(
            drug=s["drug"], effect=s["effect"], sentence=s["sentence"],
        ),
        key_file=key_file,
        maskself=maskself,
        log_dir=log_dir,
    )
