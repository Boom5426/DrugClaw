"""
ADE Corpus V2 — Binary classification benchmark.

Task: Given a sentence from a biomedical article, classify whether it
      describes an Adverse Drug Event (ADE) or not.
Labels: ADE, NoADE
Format: CSV (drug, adverse_event, sentence, pmid)
        Positive rows have drug + adverse_event filled; negatives would not.
"""

import csv
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "resources_metadata/drug_nlp/ADE_Corpus/ade_corpus.csv"

LABELS = ["ADE", "NoADE"]

SYSTEM_PROMPT = """\
You are a biomedical NLP classifier.

Task: Determine whether the given sentence describes an Adverse Drug Event (ADE).
An ADE is an undesirable effect caused by a drug at normal doses.

You must respond with ONLY a JSON object in this exact format:
{"answer": "<label>"}

where <label> is one of: ADE, NoADE
"""

USER_TEMPLATE = """\
Sentence: {sentence}

Classify this sentence: does it describe an Adverse Drug Event (ADE)?
Respond with JSON: {{"answer": "ADE"}} or {{"answer": "NoADE"}}
"""


def load_data(max_samples: int = 0) -> list[dict]:
    """Load ADE Corpus CSV. Each row with drug+adverse_event is a positive ADE sample."""
    rows = []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            gold = "ADE" if r.get("adverse_event", "").strip() else "NoADE"
            rows.append({
                "sentence": r["sentence"],
                "gold": gold,
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
        prompt = USER_TEMPLATE.format(sentence=sample["sentence"])
        resp = llm_predict(llm, SYSTEM_PROMPT, prompt)
        pred = extract_answer(resp, LABELS) or "NoADE"
        golds.append(sample["gold"])
        preds.append(pred)

    return compute_metrics(golds, preds, LABELS)
