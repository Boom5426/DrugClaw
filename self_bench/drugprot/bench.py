"""
DrugProt — Multi-class drug–protein relation classification benchmark.

Task: Given a sentence and a (drug, protein) entity pair, classify the
      relation type.
Labels: INHIBITOR, SUBSTRATE, ACTIVATOR, AGONIST, ANTAGONIST, etc.
Format: TSV (pmid, entity1, entity2, relation_type, sentence)
"""

import csv
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "resources_metadata/drug_nlp/DrugProt/drugprot.tsv"

# Primary labels from BioCreative VII DrugProt shared task
LABELS = [
    "INHIBITOR", "SUBSTRATE", "ACTIVATOR", "AGONIST", "ANTAGONIST",
    "INDIRECT-DOWNREGULATOR", "INDIRECT-UPREGULATOR",
    "DIRECT-REGULATOR", "PRODUCT-OF", "PART-OF",
]

SYSTEM_PROMPT = """\
You are a biomedical NLP classifier specialized in drug–protein relations.

Task: Given a sentence and two entities (a drug and a protein/gene),
classify the relation type between them.

Possible relation types:
- INHIBITOR: The drug inhibits the protein.
- SUBSTRATE: The drug is a substrate of the protein (enzyme).
- ACTIVATOR: The drug activates the protein.
- AGONIST: The drug is an agonist of the protein (receptor).
- ANTAGONIST: The drug is an antagonist of the protein (receptor).
- INDIRECT-DOWNREGULATOR: The drug indirectly downregulates the protein.
- INDIRECT-UPREGULATOR: The drug indirectly upregulates the protein.
- DIRECT-REGULATOR: The drug directly regulates the protein.
- PRODUCT-OF: The drug is a product of the protein (enzyme).
- PART-OF: The drug is part of the protein complex.

You must respond with ONLY a JSON object: {"answer": "<label>"}
"""

USER_TEMPLATE = """\
Entity 1 (drug): {entity1}
Entity 2 (protein/gene): {entity2}
Sentence: {sentence}

Classify the drug–protein relation. Respond with JSON: {{"answer": "<relation_type>"}}
"""


def load_data(max_samples: int = 0) -> list[dict]:
    rows = []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append({
                "entity1": r["entity1"],
                "entity2": r["entity2"],
                "sentence": r["sentence"],
                "gold": r["relation_type"],
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
            entity1=sample["entity1"], entity2=sample["entity2"],
            sentence=sample["sentence"],
        )
        resp = llm_predict(llm, SYSTEM_PROMPT, prompt)
        pred = extract_answer(resp, LABELS) or "INHIBITOR"
        golds.append(sample["gold"])
        preds.append(pred)

    return compute_metrics(golds, preds, LABELS)
