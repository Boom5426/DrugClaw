"""
n2c2 2018 Track 2 - Adverse Drug Event Extraction from Clinical Notes
Category: Drug-centric | Type: Dataset | Subcategory: Drug NLP/Text Mining
Link: https://huggingface.co/datasets/bigbio/n2c2_2018_track2
Paper: https://academic.oup.com/jamia/article-abstract/27/1/3/5581277

Annotated MIMIC-III discharge summaries for ADE and medication entity extraction.

Entity types : Drug, ADE, Reason, Frequency, Dosage, Route, Duration, Strength, Form
Relation types: Drug-ADE, Drug-Reason, Drug-Frequency, Drug-Dosage,
                Drug-Route, Drug-Duration, Drug-Strength, Drug-Form
Stats         : ~505 notes, train/test split
Access        : HuggingFace (may require dataset agreement acceptance)

Usage:
    from 33_n2c2_2018_Track2 import N2C2Track2
    db = N2C2Track2()                        # loads dataset once
    db.query("aspirin")                      # single entity
    db.query(["aspirin", "warfarin"])         # multiple entities
"""

from __future__ import annotations
import json, re
from typing import Union


class N2C2Track2:
    """Queryable wrapper for the n2c2 2018 Track 2 dataset."""

    HF_PATH = "bigbio/n2c2_2018_track2"
    HF_NAME = "n2c2_2018_track2_bigbio_kb"

    def __init__(self):
        from datasets import load_dataset
        self._ds = load_dataset(
            self.HF_PATH, name=self.HF_NAME, trust_remote_code=True
        )
        # merge all splits into one flat list for searching
        self._records = []
        for split in self._ds:
            self._records.extend(self._ds[split])
        self._index = self._build_index()

    # ---- internal helpers ------------------------------------------------- #
    def _build_index(self) -> dict[str, list[int]]:
        """Map lowercased entity text -> list of record indices."""
        idx: dict[str, list[int]] = {}
        for i, rec in enumerate(self._records):
            for ent in rec.get("entities", []):
                for txt in ent.get("text", []):
                    key = txt.strip().lower()
                    if key:
                        idx.setdefault(key, []).append(i)
        return idx

    @staticmethod
    def _passage_text(rec: dict) -> str:
        parts = []
        for p in rec.get("passages", []):
            parts.extend(p.get("text", []))
        return " ".join(parts)[:500]

    def _related_entities(self, rec: dict, ent_id: str) -> list[dict]:
        """Return entities linked to *ent_id* via relations."""
        ent_map = {e["id"]: e for e in rec.get("entities", [])}
        related = []
        for rel in rec.get("relations", []):
            arg_ids = [a["ref_id"] for a in rel.get("arg_ids", [])]
            if ent_id in arg_ids:
                for aid in arg_ids:
                    if aid != ent_id and aid in ent_map:
                        partner = ent_map[aid]
                        related.append({
                            "relation": rel.get("type", ""),
                            "type": partner.get("type", ""),
                            "text": partner.get("text", []),
                        })
        return related

    def _match_records(self, entity: str) -> list[int]:
        key = entity.strip().lower()
        # exact match first
        if key in self._index:
            return sorted(set(self._index[key]))
        # substring match fallback
        hits = set()
        for indexed_key, idxs in self._index.items():
            if key in indexed_key or indexed_key in key:
                hits.update(idxs)
        return sorted(hits)

    # ---- public API ------------------------------------------------------- #
    def query(
        self,
        entities: Union[str, list[str]],
        max_hits: int = 10,
        pretty: bool = True,
    ) -> list[dict]:
        """
        Query one or more entities; return related information.

        Args:
            entities : entity name or list of names (case-insensitive)
            max_hits : max results per entity (default 10)
            pretty   : if True, also print human-readable output

        Returns:
            List of result dicts, each containing:
              - query, doc_id, passage_snippet,
              - matched_entities [{type, text}],
              - related_entities [{relation, type, text}]
        """
        if isinstance(entities, str):
            entities = [entities]

        results = []
        for ent_name in entities:
            rec_idxs = self._match_records(ent_name)[:max_hits]
            for ri in rec_idxs:
                rec = self._records[ri]
                snippet = self._passage_text(rec)
                matched, related = [], []
                for e in rec.get("entities", []):
                    texts = [t.lower() for t in e.get("text", [])]
                    if any(ent_name.lower() in t or t in ent_name.lower() for t in texts):
                        matched.append({"type": e.get("type"), "text": e.get("text")})
                        related.extend(self._related_entities(rec, e["id"]))
                results.append({
                    "query": ent_name,
                    "doc_id": rec.get("id", ""),
                    "passage_snippet": snippet,
                    "matched_entities": matched,
                    "related_entities": related,
                })

        if pretty:
            self._print(results)
        return results

    @staticmethod
    def _print(results: list[dict]):
        for r in results:
            print(f"\n--- Query: {r['query']} | Doc: {r['doc_id']} ---")
            print(f"Snippet: {r['passage_snippet'][:200]}...")
            for m in r["matched_entities"]:
                print(f"  [MATCH] {m['type']}: {m['text']}")
            for rel in r["related_entities"]:
                print(f"  [REL]   {rel['relation']} -> {rel['type']}: {rel['text']}")

    def info(self) -> dict:
        """Return dataset metadata summary."""
        return {
            "name": "n2c2 2018 Track 2",
            "description": "ADE extraction from MIMIC-III discharge summaries",
            "entity_types": [
                "Drug", "ADE", "Reason", "Frequency",
                "Dosage", "Route", "Duration", "Strength", "Form",
            ],
            "relation_types": [
                "Drug-ADE", "Drug-Reason", "Drug-Frequency",
                "Drug-Dosage", "Drug-Route", "Drug-Duration",
                "Drug-Strength", "Drug-Form",
            ],
            "total_records": len(self._records),
            "unique_entities": len(self._index),
            "source": self.HF_PATH,
        }


# ---- CLI ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys
    db = N2C2Track2()
    print(json.dumps(db.info(), indent=2))
    queries = sys.argv[1:] or ["aspirin"]
    db.query(queries, max_hits=3)