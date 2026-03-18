#!/usr/bin/env python3

"""
LiverTox Entity Lookup Skill
link: https://www.ncbi.nlm.nih.gov/books/NBK547852/
Query liver toxicity information for one or more drug entities
from the LiverTox dataset (NBK547852).

Example entities:
- acetaminophen
- amoxicillin
- isoniazid
"""

import json
from pathlib import Path
from lxml import etree

DATA_DIR = Path("/blue/qsong1/wang.qing/AgentLLM/Survey100/resources_metadata/drug_toxicity/LiverTox/livertox_NBK547852")


def load_documents():
    docs = []

    for file in DATA_DIR.glob("*.nxml"):
        tree = etree.parse(str(file))
        sections = tree.xpath("//sec")

        for sec in sections:
            title = " ".join(sec.xpath("./title//text()"))
            text = " ".join(sec.xpath(".//p//text()"))

            if text.strip():
                docs.append({
                    "title": title,
                    "text": text
                })

    return docs


def lookup_entities(entities):
    docs = load_documents()
    results = {}

    for entity in entities:
        entity_lower = entity.lower()
        matches = []

        for d in docs:
            if entity_lower in d["title"].lower() or entity_lower in d["text"].lower():
                matches.append({
                    "section": d["title"],
                    "snippet": d["text"][:400]
                })

        results[entity] = matches[:5]

    return results


if __name__ == "__main__":

    import sys, json as _json
    if len(sys.argv) > 1:

        _cli_entities = sys.argv[1:]
        _results = lookup_entities(_cli_entities)
        for _e, _info in _results.items():
            print(f"=== {_e} ===")
            if isinstance(_info, dict):
                print(_json.dumps(_info, indent=2, ensure_ascii=False, default=str))
            else:
                print(_info)
        sys.exit(0)

    # --- original demo below ---
    # Example entities (LLM can modify this list)
    entities = [
        "acetaminophen",
        "amoxicillin"
    ]

    results = lookup_entities(entities)

    print(json.dumps(results, indent=2))