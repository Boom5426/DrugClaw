"""
DILIrank — Multi-class DILI concern level classification benchmark.

Task: Given a drug name, classify its Drug-Induced Liver Injury (DILI)
      concern level.
Labels: Most-DILI-Concern, Less-DILI-Concern, No-DILI-Concern, Ambiguous-DILI-Concern
Format: CSV (Drug Name, vDILIConcern)
"""

import csv
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "resources_metadata/drug_toxicity/DILIrank/dilirank.csv"

LABELS = [
    "Most-DILI-Concern",
    "Less-DILI-Concern",
    "No-DILI-Concern",
    "Ambiguous-DILI-Concern",
]

SYSTEM_PROMPT = """\
You are a pharmacology and drug safety expert.

Task: Given a drug name, classify its Drug-Induced Liver Injury (DILI) risk
level based on your knowledge.

DILI concern levels:
- Most-DILI-Concern: Drugs with the strongest evidence of causing DILI.
- Less-DILI-Concern: Drugs with some evidence of DILI but less severe.
- No-DILI-Concern: Drugs with no significant evidence of causing DILI.
- Ambiguous-DILI-Concern: Drugs with conflicting or inconclusive evidence.

You must respond with ONLY a JSON object: {"answer": "<label>"}
where <label> is one of: Most-DILI-Concern, Less-DILI-Concern, No-DILI-Concern, Ambiguous-DILI-Concern
"""

USER_TEMPLATE = """\
Drug: {drug}

What is the DILI concern level for this drug?
Respond with JSON: {{"answer": "<concern_level>"}}
"""


def load_data(max_samples: int = 0) -> list[dict]:
    rows = []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "drug": r["Drug Name"],
                "gold": r["vDILIConcern"],
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
        prompt = USER_TEMPLATE.format(drug=sample["drug"])
        resp = llm_predict(llm, SYSTEM_PROMPT, prompt)
        pred = extract_answer(resp, LABELS) or "Ambiguous-DILI-Concern"
        golds.append(sample["gold"])
        preds.append(pred)

    return compute_metrics(golds, preds, LABELS)
