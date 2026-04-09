"""
Drug-Drug Interaction — DDInter.

Task A (binary)  : Is there a clinically significant interaction between Drug A and Drug B?
                   Labels: yes, no  (Major or Moderate = yes, Minor = no)
Task B (severity): What is the severity of interaction? Major / Moderate / Minor
                   Labels: Major, Moderate, Minor  (all 1500 samples)

Data  : new_bench_data/DDInter/ddinter_benchmark.json
        JSON array [{drug_1, drug_2, label (0/1/2), severity, query}, ...]
        label 0 = Minor, 1 = Moderate, 2 = Major; balanced: 500 each.
QA    : Q = "What is the severity of interaction between {drug_1} and {drug_2}?"
        A = "Major" / "Moderate" / "Minor"  (or "yes"/"no" for binary task)
Metrics: Accuracy, macro-F1 (severity); AUROC, F1 (binary)
"""

import json
from pathlib import Path

DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "new_bench_data/DDInter/ddinter_benchmark.json"
)

SEVERITY_LABELS = ["Major", "Moderate", "Minor"]
BINARY_LABELS   = ["yes", "no"]

_BINARY_MAP = {
    "Major":    "yes",
    "Moderate": "yes",
    "Minor":    "no",
}

SYSTEM_PROMPT = "You are an expert pharmaceutical AI assistant."

TASK_BACKGROUND_SEVERITY = """\
Task: Drug-drug interaction severity classification (DDInter)

DDInter is a comprehensive drug-drug interaction database. Interaction severity \
is classified into three levels:
- Major:    potentially life-threatening or may cause permanent damage; \
the combination should generally be avoided.
- Moderate: the interaction may worsen the patient's condition; requires close \
monitoring or dose adjustment.
- Minor:    limited clinical significance; monitoring may be appropriate but \
modification is usually not required.

Given two drugs, classify the severity of their interaction.

Respond with ONLY a JSON object: {"answer": "<severity>"}
where <severity> is one of: Major, Moderate, Minor\
"""

TASK_BACKGROUND_BINARY = """\
Task: Drug-drug interaction detection (DDInter — binary)

DDInter is a comprehensive drug-drug interaction database. A drug pair is \
considered to have a clinically significant (harmful) interaction when the \
severity is Major or Moderate; Minor interactions are not considered harmful \
for the purpose of this task.

Given two drugs, determine whether there is a clinically significant interaction.

Respond with ONLY a JSON object: {"answer": "<label>"}
where <label> is one of: yes, no\
"""


def load_data(max_samples: int = 0) -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for r in raw:
        severity = r["severity"].strip()
        binary   = _BINARY_MAP.get(severity, "no")
        rows.append({
            "drug_1":   r["drug_1"],
            "drug_2":   r["drug_2"],
            "severity": severity,
            "binary":   binary,
            "gold":     severity,
        })

    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def _severity_question(s: dict) -> str:
    return (
        f"{TASK_BACKGROUND_SEVERITY}\n\n"
        f"What is the severity of interaction between {s['drug_1']} and {s['drug_2']}?"
    )


def _binary_question(s: dict) -> str:
    return (
        f"{TASK_BACKGROUND_BINARY}\n\n"
        f"Is there a clinically significant interaction between {s['drug_1']} and {s['drug_2']}?"
    )


def run(
    key_file: str | None = None,
    max_samples: int = 0,
    mode: str = "direct",
    maskself: bool = False,
    log_dir: str | None = None,
    task: str = "severity",
) -> dict:
    """
    task="severity" (default) — 3-class Major/Moderate/Minor classification.
    task="binary"             — binary yes/no harmful interaction + AUROC.
    task="both"               — run both and return combined result dict.
    """
    from new_bench.bench_utils import run_bench, run_binary_bench

    samples = load_data(max_samples)

    if task == "binary":
        binary_samples = [{**s, "gold": s["binary"]} for s in samples]
        return run_binary_bench(
            dataset_name="ddinter_binary",
            pos_label="yes",
            neg_label="no",
            default_label="no",
            system_prompt=SYSTEM_PROMPT,
            samples=binary_samples,
            format_prompt=_binary_question,
            mode=mode,
            maskself=maskself,
            key_file=key_file,
            log_dir=log_dir,
        )

    if task == "both":
        sev_result = run(
            key_file=key_file, max_samples=max_samples, mode=mode,
            maskself=maskself, log_dir=log_dir, task="severity",
        )
        bin_result = run(
            key_file=key_file, max_samples=max_samples, mode=mode,
            maskself=maskself, log_dir=log_dir, task="binary",
        )
        return {"severity": sev_result, "binary": bin_result}

    # Default: severity
    return run_bench(
        dataset_name="ddinter_severity",
        labels=SEVERITY_LABELS,
        default_label="Moderate",
        system_prompt=SYSTEM_PROMPT,
        samples=samples,
        format_prompt=_severity_question,
        mode=mode,
        maskself=maskself,
        key_file=key_file,
        log_dir=log_dir,
    )
