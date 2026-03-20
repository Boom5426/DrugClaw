"""
PsyTAR — ADR sentence classification benchmark.

Task: Given a patient review sentence for a psychiatric medication, classify
      whether it describes an Adverse Drug Reaction (ADR).
Labels: ADR, not_ADR
Format: CSV (drug, adverse_event, sentence, label)
"""

import csv
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "resources_metadata/drug_nlp/PsyTAR/psytar.csv"

LABELS = ["ADR", "not_ADR"]

SYSTEM_PROMPT = """\
You are a clinical NLP classifier specialized in patient-reported outcomes.

Task: Given a sentence from a patient review about a psychiatric medication,
classify whether it describes an Adverse Drug Reaction (ADR).

An ADR is any undesirable effect of a drug, such as side effects, withdrawal
symptoms, or negative experiences.

Labels:
- ADR: The sentence describes an adverse drug reaction.
- not_ADR: The sentence does NOT describe an adverse drug reaction (it may
  describe effectiveness, general experience, or other non-ADR content).

You must respond with ONLY a JSON object: {"answer": "<label>"}
where <label> is one of: ADR, not_ADR
"""

USER_TEMPLATE = """\
Drug: {drug}
Sentence: {sentence}

Does this sentence describe an Adverse Drug Reaction (ADR)?
Respond with JSON: {{"answer": "ADR"}} or {{"answer": "not_ADR"}}
"""


def load_data(max_samples: int = 0) -> list[dict]:
    rows = []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # The metadata has "ADR" in the label column for positive samples
            gold = "ADR" if r.get("label", "").strip().upper() == "ADR" else "not_ADR"
            rows.append({
                "drug": r["drug"],
                "sentence": r["sentence"],
                "gold": gold,
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
        dataset_name="psytar",
        labels=LABELS,
        default_label="not_ADR",
        system_prompt=SYSTEM_PROMPT,
        samples=load_data(max_samples),
        format_prompt=lambda s: USER_TEMPLATE.format(
            drug=s["drug"], sentence=s["sentence"],
        ),
        key_file=key_file,
        maskself=maskself,
        log_dir=log_dir,
    )
