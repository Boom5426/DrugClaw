"""
DDI Corpus 2013 — Multi-class DDI type classification benchmark.

Task: Given a sentence mentioning two drugs, classify the Drug-Drug
      Interaction type.
Labels: advise, effect, mechanism, int
Format: TSV (drug1, drug2, ddi_type, sentence)
"""

import csv
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "resources_metadata/drug_nlp/DDI_Corpus_2013/ddi_corpus.tsv"

LABELS = ["advise", "effect", "mechanism", "int"]

SYSTEM_PROMPT = """\
You are a biomedical NLP classifier specialized in Drug-Drug Interactions.

Task: Given a sentence and two drug names, classify the type of drug-drug
interaction described.

Interaction types:
- advise: A recommendation or advice regarding the concomitant use of two drugs.
- effect: A description of the pharmacodynamic effect of a drug interaction.
- mechanism: A description of the pharmacokinetic mechanism of interaction.
- int: A generic drug interaction without further specification.

You must respond with ONLY a JSON object: {"answer": "<label>"}
where <label> is one of: advise, effect, mechanism, int
"""

USER_TEMPLATE = """\
Drug 1: {drug1}
Drug 2: {drug2}
Sentence: {sentence}

What type of drug-drug interaction is described? Respond with JSON: {{"answer": "<type>"}}
"""


def load_data(max_samples: int = 0) -> list[dict]:
    rows = []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append({
                "drug1": r["drug1"],
                "drug2": r["drug2"],
                "sentence": r["sentence"],
                "gold": r["ddi_type"],
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
            drug1=sample["drug1"], drug2=sample["drug2"], sentence=sample["sentence"],
        )
        resp = llm_predict(llm, SYSTEM_PROMPT, prompt)
        pred = extract_answer(resp, LABELS) or "int"
        golds.append(sample["gold"])
        preds.append(pred)

    return compute_metrics(golds, preds, LABELS)
