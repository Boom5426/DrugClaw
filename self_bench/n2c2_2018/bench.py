"""
n2c2 2018 Track 2 — Binary ADE classification benchmark.

Task: Given a clinical text sentence, classify whether an Adverse Drug Event
      is present (positive) or absent (negative).
Labels: positive, negative
Format: TSV (drug, adverse_event, label, sentence)
"""

import csv
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "resources_metadata/drug_nlp/n2c2_2018/n2c2_2018.tsv"

LABELS = ["positive", "negative"]

SYSTEM_PROMPT = """\
You are a clinical NLP classifier.

Task: Given a clinical text sentence mentioning a drug, classify whether the
sentence describes an Adverse Drug Event (ADE).

Labels:
- positive: The sentence describes an ADE (the drug caused an adverse effect).
- negative: The sentence does not describe an ADE.

You must respond with ONLY a JSON object: {"answer": "<label>"}
where <label> is one of: positive, negative
"""

USER_TEMPLATE = """\
Drug: {drug}
Sentence: {sentence}

Does this sentence describe an Adverse Drug Event caused by the drug?
Respond with JSON: {{"answer": "positive"}} or {{"answer": "negative"}}
"""


def load_data(max_samples: int = 0) -> list[dict]:
    rows = []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append({
                "drug": r["drug"],
                "sentence": r["sentence"],
                "gold": r["label"],
            })
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
        dataset_name="n2c2_2018",
        labels=LABELS,
        default_label="negative",
        system_prompt=SYSTEM_PROMPT,
        samples=load_data(max_samples),
        format_prompt=lambda s: USER_TEMPLATE.format(
            drug=s["drug"], sentence=s["sentence"],
        ),
        key_file=key_file,
        maskself=maskself,
        log_dir=log_dir,
    )
