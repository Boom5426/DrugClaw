"""
Drug QA — PubMedQA filtered benchmark.

Task  : Given a biomedical question and background context, answer yes / no / maybe.
Labels: yes, no, maybe
Data  : new_bench_data/PubMedQA/pubmedqa_pqa_labeled_train_drug_samples.json
        JSON array [{pubid, question, context, long_answer, final_decision}, ...]
        context.contexts — list of background paragraph strings
QA    : Q = context paragraphs + question
        A = final_decision ("yes" / "no" / "maybe")
Metric: Accuracy (exact match on yes/no/maybe)
"""

import ast
import json
from pathlib import Path

DATA_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "new_bench_data/PubMedQA/pubmedqa_pqa_labeled_train_drug_samples.json"
)

LABELS = ["yes", "no", "maybe"]

SYSTEM_PROMPT = "You are an expert pharmaceutical AI assistant."

TASK_BACKGROUND = """\
Task: Biomedical question answering (PubMedQA — drug-focused subset)

PubMedQA is a biomedical question answering dataset built from PubMed research \
articles. Each sample provides a research question and background context \
extracted from the article abstract. Based solely on the evidence in the \
provided context, answer the question with one of three labels:
- "yes":   the context clearly supports an affirmative answer.
- "no":    the context clearly does not support an affirmative answer.
- "maybe": the evidence is mixed, insufficient, or uncertain.

Respond with ONLY a JSON object: {"answer": "<label>"}
where <label> is one of: yes, no, maybe\
"""


def _parse_context(raw) -> str:
    if isinstance(raw, dict):
        ctx = raw
    else:
        try:
            ctx = ast.literal_eval(str(raw))
        except Exception:
            return str(raw)[:1000]

    paragraphs = ctx.get("contexts", [])
    if paragraphs:
        return "\n\n".join(str(p) for p in paragraphs[:5])
    return str(raw)[:1000]


def load_data(max_samples: int = 0) -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for r in raw:
        gold = r.get("final_decision", "").strip().lower()
        if gold not in LABELS:
            continue
        rows.append({
            "question": r["question"],
            "context":  _parse_context(r["context"]),
            "gold":     gold,
        })

    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def _format_prompt(s: dict) -> str:
    return (
        f"{TASK_BACKGROUND}\n\n"
        f"Context:\n{s['context']}\n\n"
        f"Question: {s['question']}"
    )


def run(
    key_file: str | None = None,
    max_samples: int = 0,
    mode: str = "direct",
    maskself: bool = False,
    log_dir: str | None = None,
) -> dict:
    from new_bench.bench_utils import run_bench

    return run_bench(
        dataset_name="drug_qa",
        labels=LABELS,
        default_label="maybe",
        system_prompt=SYSTEM_PROMPT,
        samples=load_data(max_samples),
        format_prompt=_format_prompt,
        mode=mode,
        maskself=maskself,
        key_file=key_file,
        log_dir=log_dir,
    )
