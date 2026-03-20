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


def load_data(max_samples: int = 0) -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for r in raw:
        rows.append({
            "drug": r["drug"],
            "effect": r["effect"],
            "sentence": r["sentence"],
            "gold": r["event_type"],
        })
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def run(key_file: str | None = None, max_samples: int = 0) -> dict:
    from self_bench.bench_utils import make_llm, llm_predict, extract_answer, compute_metrics

    llm, _ = make_llm(key_file)
    data = load_data(max_samples)
    if not data:
        return {"error": "No data loaded", "total": 0}

    golds, preds = [], []
    for sample in data:
        prompt = USER_TEMPLATE.format(
            drug=sample["drug"], effect=sample["effect"], sentence=sample["sentence"],
        )
        resp = llm_predict(llm, SYSTEM_PROMPT, prompt)
        pred = extract_answer(resp, LABELS) or "adverse_event"
        golds.append(sample["gold"])
        preds.append(pred)

    return compute_metrics(golds, preds, LABELS)
