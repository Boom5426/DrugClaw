"""
Drug Repurposing — RepoDB.

Task  : Given a disease, rank candidate drugs by evidence for treatment.
Data  : new_bench_data/RepoDB/full.csv
        CSV: drug_name, drugbank_id, ind_name, ind_id, NCT, status, phase, DetailedStatus
        Gold = all "Approved" drugs for each disease (ind_name).
        Only diseases with ≥3 approved drugs are included (~200 diseases).

QA    : This dataset does NOT have QA pairs in the raw data. QA pairs are
        constructed here from the CSV:
          Q = "What drugs could treat {disease}? List them ranked by strength of evidence."
          A = set of all approved drug names for that disease (gold_drugs)

Metric: Recall@10, Recall@50, MRR, NDCG@20
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "new_bench_data/RepoDB/full.csv"
)

MIN_APPROVED_DRUGS = 3
MAX_DISEASES       = 200

SYSTEM_PROMPT = "You are an expert pharmaceutical AI assistant."

TASK_BACKGROUND = """\
Task: Drug repurposing — disease-to-drug retrieval (RepoDB)

RepoDB is a curated database of approved and investigational drug-disease \
associations compiled from clinical trial records and approved drug labels. \
Given a disease name, your task is to recall all drugs known to treat it, \
ranked from most to least evidence-based.

List one drug per line, numbered. Aim to include at least 20 drugs; include \
all approved agents you are aware of.

Format:
1. <drug name>
2. <drug name>
...\
"""


def load_data(max_samples: int = 0, seed: int = 42) -> list[dict]:
    """
    Build QA pairs from RepoDB CSV.

    Groups rows by disease (ind_name), keeps diseases with ≥ MIN_APPROVED_DRUGS
    approved entries, and returns one query dict per disease:
      disease    — disease name (the question key)
      gold_drugs — set of approved drug names (the gold answer)
    """
    disease_drugs: dict[str, set[str]] = defaultdict(set)

    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status", "").strip() == "Approved":
                disease = row["ind_name"].strip()
                drug    = row["drug_name"].strip()
                if disease and drug:
                    disease_drugs[disease].add(drug)

    qualified = [
        {"disease": disease, "gold_drugs": drugs}
        for disease, drugs in disease_drugs.items()
        if len(drugs) >= MIN_APPROVED_DRUGS
    ]

    rng = random.Random(seed)
    rng.shuffle(qualified)
    cap = MAX_DISEASES
    if max_samples > 0:
        cap = min(cap, max_samples)
    return qualified[:cap]


def _format_prompt(q: dict) -> str:
    return (
        f"{TASK_BACKGROUND}\n\n"
        f"Disease: {q['disease']}\n\n"
        f"What drugs could treat this disease? Rank by strength of evidence."
    )


def run(
    key_file: str | None = None,
    max_samples: int = 0,
    mode: str = "direct",
    maskself: bool = False,
    log_dir: str | None = None,
) -> dict:
    from new_bench.bench_utils import run_ranking_bench

    return run_ranking_bench(
        dataset_name="repodb",
        system_prompt=SYSTEM_PROMPT,
        queries=load_data(max_samples),
        format_prompt=_format_prompt,
        k_recall=[10, 50],
        k_ndcg=20,
        mode=mode,
        maskself=maskself,
        key_file=key_file,
        log_dir=log_dir,
    )
