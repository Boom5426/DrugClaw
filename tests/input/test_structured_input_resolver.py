from __future__ import annotations

import json
from pathlib import Path

from drugclaw.config import Config
from drugclaw.drug_identifier_sources import (
    ChEMBLIdentifierSource,
    PubChemIdentifierSource,
    ResolvedIdentifierRecord,
)
from drugclaw.structured_input_resolver import StructuredInputResolver


def _write_key_file(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "api_key": "",
                "base_url": "https://example.com/v1",
                "model": "test-model",
            }
        ),
        encoding="utf-8",
    )


def _record(
    identifier_type: str,
    identifier_value: str,
    canonical_name: str,
) -> ResolvedIdentifierRecord:
    return ResolvedIdentifierRecord(
        identifier_type=identifier_type,
        identifier_value=identifier_value,
        canonical_name=canonical_name,
        source="stub",
        status="resolved",
    )


class _SourceStub:
    def __init__(self, mapping: dict[tuple[str, str], list[ResolvedIdentifierRecord]]) -> None:
        self.mapping = mapping
        self.calls: list[tuple[str, str]] = []

    def resolve_identifier(
        self,
        identifier_type: str,
        identifier_value: str,
    ) -> list[ResolvedIdentifierRecord]:
        self.calls.append((identifier_type, identifier_value))
        return self.mapping.get((identifier_type, identifier_value), [])


def test_config_exposes_structured_identifier_defaults(tmp_path: Path) -> None:
    key_file = tmp_path / "navigator_api_keys.json"
    _write_key_file(key_file)

    config = Config(key_file=str(key_file))

    assert config.ENABLE_STRUCTURED_IDENTIFIER_RESOLUTION is True
    assert config.STRUCTURED_IDENTIFIER_TIMEOUT == 5
    assert config.STRUCTURED_IDENTIFIER_CACHE_SIZE == 256


def test_chembl_source_resolves_pref_name(monkeypatch) -> None:
    source = ChEMBLIdentifierSource(timeout=5)
    monkeypatch.setattr(source, "_fetch_json", lambda url: {"pref_name": "IMATINIB"})

    result = source.resolve_chembl_id("chembl:941")

    assert result.status == "resolved"
    assert result.identifier_type == "chembl_id"
    assert result.identifier_value == "CHEMBL941"
    assert result.canonical_name == "imatinib"


def test_pubchem_source_resolves_cid_title(monkeypatch) -> None:
    source = PubChemIdentifierSource(timeout=5)
    monkeypatch.setattr(
        source,
        "_fetch_json",
        lambda url: {"PropertyTable": {"Properties": [{"CID": 5291, "Title": "Imatinib"}]}},
    )

    result = source.resolve_pubchem_cid("cid:5291")

    assert result.status == "resolved"
    assert result.identifier_type == "pubchem_cid"
    assert result.identifier_value == "5291"
    assert result.canonical_name == "imatinib"


def test_pubchem_source_resolves_inchikey_title(monkeypatch) -> None:
    source = PubChemIdentifierSource(timeout=5)
    monkeypatch.setattr(
        source,
        "_fetch_json",
        lambda url: {"PropertyTable": {"Properties": [{"CID": 5291, "Title": "Imatinib"}]}},
    )

    result = source.resolve_inchikey("ktufnokkbvmgrw-uhfffaoysa-n")

    assert result.status == "resolved"
    assert result.identifier_type == "inchikey"
    assert result.identifier_value == "KTUFNOKKBVMGRW-UHFFFAOYSA-N"
    assert result.canonical_name == "imatinib"


def test_structured_input_resolver_rewrites_chembl_query() -> None:
    resolver = StructuredInputResolver(
        source=_SourceStub(
            {
                ("chembl_id", "CHEMBL941"): [
                    _record("chembl_id", "CHEMBL941", "imatinib")
                ]
            }
        )
    )

    result = resolver.resolve_query(
        "What are the known drug targets of CHEMBL941?"
    )

    assert result["status"] == "resolved"
    assert result["normalized_query"] == "What are the known drug targets of imatinib?"
    assert result["detected_identifiers"] == [
        {
            "type": "chembl_id",
            "raw_text": "CHEMBL941",
            "normalized_value": "CHEMBL941",
        }
    ]


def test_structured_input_resolver_supports_pubchem_cid_formats() -> None:
    resolver = StructuredInputResolver(
        source=_SourceStub(
            {
                ("pubchem_cid", "5291"): [
                    _record("pubchem_cid", "5291", "imatinib")
                ]
            }
        )
    )

    result = resolver.resolve_query(
        "What are the known drug targets of PubChem CID 5291?"
    )

    assert result["status"] == "resolved"
    assert result["normalized_query"] == "What are the known drug targets of imatinib?"


def test_structured_input_resolver_supports_inchikey_queries() -> None:
    resolver = StructuredInputResolver(
        source=_SourceStub(
            {
                ("inchikey", "KTUFNOKKBVMGRW-UHFFFAOYSA-N"): [
                    _record(
                        "inchikey",
                        "KTUFNOKKBVMGRW-UHFFFAOYSA-N",
                        "imatinib",
                    )
                ]
            }
        )
    )

    result = resolver.resolve_query(
        "What are the known drug targets of KTUFNOKKBVMGRW-UHFFFAOYSA-N?"
    )

    assert result["status"] == "resolved"
    assert result["normalized_query"] == "What are the known drug targets of imatinib?"


def test_structured_input_resolver_keeps_unresolved_query_unchanged() -> None:
    resolver = StructuredInputResolver(source=_SourceStub({}))

    result = resolver.resolve_query(
        "What are the known drug targets of CHEMBL999999999?"
    )

    assert result["status"] == "unresolved"
    assert result["normalized_query"] == "What are the known drug targets of CHEMBL999999999?"


def test_structured_input_resolver_emits_identifier_mentions() -> None:
    resolver = StructuredInputResolver(
        source=_SourceStub(
            {
                ("chembl_id", "CHEMBL941"): [
                    _record("chembl_id", "CHEMBL941", "imatinib")
                ]
            }
        )
    )

    result = resolver.resolve_query(
        "What does CHEMBL941 target?"
    )

    assert result["rewrite_applied"] is True
    assert result["drug_mentions"] == [
        {
            "raw_text": "CHEMBL941",
            "mention_type": "chembl_id",
            "normalized_value": "CHEMBL941",
            "canonical_drug_name": "imatinib",
            "resolution_stage": "identifier",
            "source": "stub",
        }
    ]
