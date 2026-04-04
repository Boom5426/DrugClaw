from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .drug_identifier_sources import (
    CompositeDrugIdentifierSource,
    ResolvedIdentifierRecord,
    normalize_chembl_identifier,
    normalize_inchikey,
    normalize_pubchem_cid,
)


_CHEMBL_PATTERN = re.compile(r"\bchembl[:\s-]?(\d+)\b", re.IGNORECASE)
_PUBCHEM_CID_PATTERNS = [
    re.compile(r"\b(?:pubchem\s+)?cid[:\s]+(\d+)\b", re.IGNORECASE),
    re.compile(r"\bpubchem\s+compound[:\s]+(\d+)\b", re.IGNORECASE),
]
_INCHIKEY_PATTERN = re.compile(r"\b([A-Z]{14}-[A-Z]{10}-[A-Z])\b", re.IGNORECASE)


@dataclass
class StructuredInputResolver:
    source: Any

    @classmethod
    def default(cls, config: Any) -> "StructuredInputResolver":
        timeout = int(getattr(config, "STRUCTURED_IDENTIFIER_TIMEOUT", 5))
        cache_size = int(getattr(config, "STRUCTURED_IDENTIFIER_CACHE_SIZE", 256))
        source = CompositeDrugIdentifierSource.default(
            timeout=timeout,
            cache_size=cache_size,
        )
        return cls(source=source)

    def detect_identifiers(self, query: str) -> list[dict[str, str]]:
        detected_identifiers: list[dict[str, str]] = []
        text = str(query)

        for match in _CHEMBL_PATTERN.finditer(text):
            normalized_value = normalize_chembl_identifier(match.group(0))
            if normalized_value:
                detected_identifiers.append(
                    {
                        "type": "chembl_id",
                        "raw_text": match.group(0),
                        "normalized_value": normalized_value,
                    }
                )

        for pattern in _PUBCHEM_CID_PATTERNS:
            for match in pattern.finditer(text):
                normalized_value = normalize_pubchem_cid(match.group(1))
                if normalized_value:
                    detected_identifiers.append(
                        {
                            "type": "pubchem_cid",
                            "raw_text": match.group(0),
                            "normalized_value": normalized_value,
                        }
                    )

        for match in _INCHIKEY_PATTERN.finditer(text):
            normalized_value = normalize_inchikey(match.group(1))
            if normalized_value:
                detected_identifiers.append(
                    {
                        "type": "inchikey",
                        "raw_text": match.group(1),
                        "normalized_value": normalized_value,
                    }
                )

        unique_identifiers: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for item in detected_identifiers:
            key = (item["type"], item["normalized_value"])
            if key in seen:
                continue
            seen.add(key)
            unique_identifiers.append(item)
        return unique_identifiers

    def resolve_query(self, query: str) -> dict[str, Any]:
        original_query = str(query)
        detected_identifiers = self.detect_identifiers(original_query)
        resolved_records: list[dict[str, str]] = []
        errors: list[dict[str, str]] = []
        canonical_drug_names: list[str] = []
        drug_mentions: list[dict[str, str]] = []
        seen_mentions: set[tuple[str, str, str, str]] = set()

        for identifier in detected_identifiers:
            records = self.source.resolve_identifier(
                identifier["type"],
                identifier["normalized_value"],
            )
            if not records:
                errors.append(
                    {
                        "identifier_type": identifier["type"],
                        "identifier_value": identifier["normalized_value"],
                        "error": "not_found",
                    }
                )
                continue

            for record in records:
                record_dict = (
                    record.to_dict()
                    if isinstance(record, ResolvedIdentifierRecord)
                    else dict(record)
                )
                if record_dict.get("status") == "resolved" and record_dict.get(
                    "canonical_drug_name"
                ):
                    resolved_records.append(record_dict)
                    canonical_name = str(record_dict["canonical_drug_name"]).strip()
                    if canonical_name and canonical_name not in canonical_drug_names:
                        canonical_drug_names.append(canonical_name)
                    mention_key = (
                        identifier["raw_text"],
                        identifier["type"],
                        identifier["normalized_value"],
                        canonical_name,
                    )
                    if canonical_name and mention_key not in seen_mentions:
                        seen_mentions.add(mention_key)
                        drug_mentions.append(
                            {
                                "raw_text": identifier["raw_text"],
                                "mention_type": identifier["type"],
                                "normalized_value": identifier["normalized_value"],
                                "canonical_drug_name": canonical_name,
                                "resolution_stage": "identifier",
                                "source": str(record_dict.get("source", "")).strip(),
                            }
                        )
                else:
                    errors.append(
                        {
                            "identifier_type": record_dict.get(
                                "identifier_type",
                                identifier["type"],
                            ),
                            "identifier_value": record_dict.get(
                                "identifier_value",
                                identifier["normalized_value"],
                            ),
                            "error": record_dict.get("error", "not_found"),
                        }
                    )

        normalized_query = original_query
        if len(detected_identifiers) == 1 and len(canonical_drug_names) == 1:
            normalized_query = normalized_query.replace(
                detected_identifiers[0]["raw_text"],
                canonical_drug_names[0],
                1,
            )
            status = "resolved"
        elif errors and not resolved_records:
            status = "error" if any(error["error"] != "not_found" for error in errors) else "unresolved"
        elif canonical_drug_names:
            status = "ambiguous"
        else:
            status = "unresolved"

        return {
            "original_query": original_query,
            "normalized_query": normalized_query,
            "status": status,
            "detected_identifiers": detected_identifiers,
            "resolved_records": resolved_records,
            "errors": errors,
            "drug_mentions": drug_mentions,
            "rewrite_applied": normalized_query != original_query,
        }
